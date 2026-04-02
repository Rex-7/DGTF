import hashlib
import os
import pickle
import numpy as np
import torch
import copy
import torch.optim as optim
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from tqdm import trange
from sklearn.metrics import f1_score


def get_rep(cnn, PI_tensor_layer, linear, batch_PI, device, batch_size):
    cnn = cnn.to(device)
    PI_tensor_layer = PI_tensor_layer.to(device)
    linear = linear.to(device)
    batch_PI = batch_PI.to(device)

    PI_emb = cnn(batch_PI)
    PI_emb = PI_tensor_layer(PI_emb).reshape(batch_size, -1)  # B * hidden_dim^3
    PI_emb = PI_emb.float()
    PI_emb = linear(PI_emb) 
    return PI_emb

class Client_GC():
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args,aug):
        
        self.model = model
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.aug = aug

        self.args = args
        self.device = args.device

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

        self.train_preprocessed_batches = []
        self.test_preprocessed_batches = []
        self.val_preprocessed_batches = []
        self.pm_train = []
        self.lamda = 0
        self.train_samples = 0
        self.track = []

        self.global_consensus = None

        self.neibor_models = []
        
        self.dataLoader = self.preprocess_datasets()
    
    def preprocess_datasets(self):
        print(f"Preprocessing dataset... (This may take some time, but only needs to be done once)")

        graphs_original = self.dataLoader['train_ori']
        aug1_graphs = self.dataLoader['train_aug1']
        aug2_graphs = self.dataLoader['train_aug2']
        PIs_aug1 = self.dataLoader['PI_aug1']
        PIs_aug2 = self.dataLoader['PI_aug2']

        processed_aug1 = []
        processed_aug2 = []

        print("Processing first augmented graph group...")
        for idx in range(len(graphs_original)):
            orig = graphs_original[idx]
            aug1 = aug1_graphs[idx] 

            processed = match_nodes(orig, aug1, 'cpu', self.aug)
            processed_aug1.append(processed)

        print("Processing second augmented graph group...")
        for idx in range(len(graphs_original)):
            orig = graphs_original[idx]
            aug2 = aug2_graphs[idx]

            processed = match_nodes(orig, aug2, 'cpu', self.aug)
            processed_aug2.append(processed)

        return {
            'train_ori': graphs_original,
            'train_aug1': processed_aug1,
            'train_aug2': processed_aug2,
            'PI_aug1': PIs_aug1,
            'PI_aug2': PIs_aug2,
            'train_finetune':self.dataLoader['train_finetune'],
            'PI_train':self.dataLoader['PI_train'],
            'test_finetune':self.dataLoader['test_finetune'],
            'PI_test':self.dataLoader['PI_test']
        }        

    def get_features(self):
        features = []
        labels = []
        dataloader = self.dataLoader
        self.model.eval()
        test_graphs_eval = dataloader['test_finetune']
        test_PIs = dataloader['PI_test']
        idx = np.arange(len(test_graphs_eval))

        with torch.no_grad():
            for i in range(0, len(test_graphs_eval), 64):
                sampled_idx = idx[i:i+64]
                if len(sampled_idx) == 0:
                    continue
                batch_graph = [test_graphs_eval[j] for j in sampled_idx]
                batch_PI = torch.stack([test_PIs[j] for j in sampled_idx])
                batch_labels = torch.LongTensor([graph.y for graph in batch_graph]).to(self.device)

                rep = self.model.base.get_cnn_features( batch_PI)
                rep = rep.view(rep.size(0), -1)

                features.append(rep.cpu().numpy())
                labels.append(batch_labels.cpu().numpy())

        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0), test_graphs_eval
                
    def joint_train(self, local_epoch):
        print(f"{self.name} starting joint training...")
        
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        self.model.to(self.device)
        if hasattr(self.model, 'base'):
            self.model.base.device = self.device
        self._optimizer_state_to_gpu()
        train_stats = train_gc_joint(self, self.model, self.dataLoader, local_epoch, self.args.device, self.optimizer, self.scheduler)
        self.model.to("cpu")
        if hasattr(self.model, 'base'):
            self.model.base.device = "cpu"
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        self.neibor_models = []
        
        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        
        self.neibor_models = []
        
        self._optimizer_state_to_cpu()

        print(f"Client {self.name} joint training completed")

    def finetune(self, local_epoch):
        self.model.to(self.device)
        if hasattr(self.model, 'base'):
            self.model.base.device = self.device
        self._optimizer_state_to_gpu()
        linear_probe(self, self.model, self.dataLoader, local_epoch, self.args.device, self.optimizer, self.scheduler)

        self.model.to("cpu")
        if hasattr(self.model, 'base'):
            self.model.base.device = "cpu"
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        self._optimizer_state_to_cpu()

    def evaluate(self):
        return eval_gc_test(self.model, self.args.device, self)

    

    def _optimizer_state_to_cpu(self):
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

    def _optimizer_state_to_gpu(self):
        if self.optimizer is None:
            return
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])

def linear_probe(client, model, dataloaders, local_epoch, device, optimizer, scheduler):
    train_graphs = dataloaders['train_finetune']
    train_PIs = dataloaders['PI_train']
    graphs_without_labels = remove_labels(train_graphs)

    for param in model.parameters():
        param.requires_grad = False
    
    params_finetune = list(model.head.parameters()) + list(model.base.attend.parameters()) + list(model.base.output_tensor_layer.parameters())
    
    for param in params_finetune:
        param.requires_grad = True
    
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=client.args.finetune_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    num_train_samples = len(train_graphs)
    batch_size = client.args.batch_size
    num_batches_per_epoch = (num_train_samples + batch_size - 1) // batch_size

    best_acc = 0.0
    best_model_state_dict = None

    for epoch in range(local_epoch):
        model.train()
        shuffled_indices = np.random.permutation(num_train_samples)
        pbar = trange(num_batches_per_epoch, unit='batch', desc=f"Finetune Epoch {epoch+1}/{local_epoch}")

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_train_samples)
            batch_indices = shuffled_indices[start_idx:end_idx]
            if len(batch_indices) == 0:
                continue
            graphs_with_labels = [train_graphs[idx] for idx in batch_indices]
            graphs_batch = [graphs_without_labels[idx] for idx in batch_indices]
            PIs_batch = torch.stack([train_PIs[idx] for idx in batch_indices]).to(device)

            labels = torch.LongTensor([graph.y for graph in graphs_with_labels]).to(device)

            _, score = model(graphs_batch, None, PIs_batch, None, mode='finetune')
            loss = criterion(score, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        _, current_acc = eval_gc_test(model, device, client)
        print(f"Epoch {epoch+1}/{local_epoch}, Current Test Accuracy: {current_acc:.4f}, Best Test Accuracy: {best_acc:.4f}")

        if current_acc > best_acc:
            best_acc = current_acc
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"New best accuracy: {best_acc:.4f} at epoch {epoch+1}")

    if best_model_state_dict is not None:
        print(f"Loading best model state dict with accuracy: {best_acc:.4f}")
        model.load_state_dict(best_model_state_dict)
    else:
        print("No improvement in accuracy during finetuning or no finetuning epochs run.")

    for param in model.parameters():
        param.requires_grad = True

def train_gc_joint(client, model, dataloaders, local_epoch, device, optimizer, scheduler):
    model.train()
    
    graphs_without_labels = dataloaders['train_ori']
    processed_aug1 = dataloaders['train_aug1']
    processed_aug2 = dataloaders['train_aug2']
    PIs_aug1 = dataloaders['PI_aug1']
    PIs_aug2 = dataloaders['PI_aug2']
    
    num_train_samples = len(graphs_without_labels)
    batch_size = client.args.batch_size
    num_batches_per_epoch = (num_train_samples + batch_size - 1) // batch_size
    
    alpha = client.args.alpha
    epoch_losses = []
    
    for epoch in range(local_epoch):
        accumulate_loss_cl = 0.0
        accumulate_loss_distill = 0.0
        accumulate_loss_total = 0.0
        processed_samples = 0
        
        shuffled_indices = np.random.permutation(num_train_samples)
        
        for batch_idx in range(num_batches_per_epoch):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_train_samples)
            selected_idx = shuffled_indices[start_idx:end_idx]
            
            current_batch_size = len(selected_idx)
            if current_batch_size == 0:
                continue
            
            try:
                batch_graph1_full = [processed_aug1[idx] for idx in selected_idx]
                batch_graph2_full = [processed_aug2[idx] for idx in selected_idx]
                batch_PIs1_full = torch.stack([PIs_aug1[idx] for idx in selected_idx]).to(device)
                batch_PIs2_full = torch.stack([PIs_aug2[idx] for idx in selected_idx]).to(device)
            except IndexError as e:
                print(f"Index error occurred in batch {batch_idx}: {e}")
                continue
            
            micro_bs = 8
            accum_steps = (current_batch_size + micro_bs - 1) // micro_bs
            optimizer.zero_grad(set_to_none=True)
            micro_loss_cl_sum = 0.0
            micro_loss_distill_sum = 0.0
            micro_loss_total_sum = 0.0
            
            for micro_start in range(0, current_batch_size, micro_bs):
                micro_end = min(micro_start + micro_bs, current_batch_size)
                sel = slice(micro_start, micro_end)
                micro_size = micro_end - micro_start
                
                batch_graph1 = batch_graph1_full[sel]
                batch_graph2 = batch_graph2_full[sel]
                if len(batch_graph1) < 2:
                    continue
                batch_PIs1 = batch_PIs1_full[sel]
                batch_PIs2 = batch_PIs2_full[sel]
                
                max_nodes = max([g.x.size(0) for g in batch_graph1 + batch_graph2])
                batch_graph1 = pad_graphs(batch_graph1, max_nodes)
                batch_graph2 = pad_graphs(batch_graph2, max_nodes)
                
                try:
                    loss_cl = model.base(batch_graph1=batch_graph1, batch_graph2=batch_graph2, 
                                       PIs_aug1=batch_PIs1, PIs_aug2=batch_PIs2, mode='cl')
                    
                    del batch_graph1, batch_graph2
                    
                    loss_distill = None
                    if len(client.neibor_models) > 0:
                        neighbor_batch_size = min(2, len(client.neibor_models))
                        all_neighbor_reps = []
                        
                        for i in range(0, len(client.neibor_models), neighbor_batch_size):
                            end_i = min(i + neighbor_batch_size, len(client.neibor_models))
                            current_neighbors = client.neibor_models[i:end_i]
                            
                            with torch.no_grad():
                                batch_neighbor_reps = []
                                for cnn, PI_tensor_layer in current_neighbors:
                                    R_cli_x1 = get_rep(cnn, PI_tensor_layer, model.linear, 
                                                     batch_PIs1, device, micro_size)
                                    R_cli_x2 = get_rep(cnn, PI_tensor_layer, model.linear, 
                                                     batch_PIs2, device, micro_size)
                                    neighbor_rep = torch.cat([R_cli_x1, R_cli_x2], dim=0)
                                    batch_neighbor_reps.append(neighbor_rep)
                                    del R_cli_x1, R_cli_x2
                                
                                if batch_neighbor_reps:
                                    stacked_reps = torch.stack(batch_neighbor_reps, dim=1)
                                    all_neighbor_reps.append(stacked_reps)
                                    del batch_neighbor_reps
                            
                            torch.cuda.empty_cache()
                        
                        if all_neighbor_reps:
                            neibor_reps = torch.cat(all_neighbor_reps, dim=1)
                            del all_neighbor_reps
                            
                            rep_s1 = get_rep(model.base.cnn, model.base.PI_tensor_layer, model.linear, 
                                           batch_PIs1, device, micro_size)
                            rep_s2 = get_rep(model.base.cnn, model.base.PI_tensor_layer, model.linear, 
                                           batch_PIs2, device, micro_size)
                            rep_s = torch.cat([rep_s1, rep_s2], dim=0)
                            del rep_s1, rep_s2
                            
                            loss_distill = model.distillation(rep_s, neibor_reps, device)
                            
                            del rep_s, neibor_reps
                    
                    if loss_cl is not None and not torch.isnan(loss_cl) and not torch.isinf(loss_cl):
                        total_loss = loss_cl
                        
                        if loss_distill is not None and not torch.isnan(loss_distill) and not torch.isinf(loss_distill):
                            total_loss = total_loss + alpha * loss_distill
                            micro_loss_distill_sum += loss_distill.item() * micro_size
                        
                        if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                            (total_loss / accum_steps).backward()
                            
                            micro_loss_cl_sum += loss_cl.item() * micro_size
                            micro_loss_total_sum += total_loss.item() * micro_size
                        else:
                            print(f"Fatal error: nan/inf total loss at client {client.name}")
                            return {'contrastiveLosses': float('nan')}
                    else:
                        print(f"Fatal error: nan/inf cl loss at client {client.name}")
                        return {'contrastiveLosses': float('nan')}
                    
                    del batch_PIs1, batch_PIs2, loss_cl, total_loss
                    if loss_distill is not None:
                        del loss_distill
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Client {client.name} out of memory in micro-batch, skipping")
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        continue
                    else:
                        raise e
            
            optimizer.step()
            
            processed_samples += current_batch_size
            accumulate_loss_cl += micro_loss_cl_sum
            accumulate_loss_distill += micro_loss_distill_sum
            accumulate_loss_total += micro_loss_total_sum
            
            del batch_graph1_full, batch_graph2_full, batch_PIs1_full, batch_PIs2_full
            torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                import gc
                gc.collect()
        
        scheduler.step()
        
        if processed_samples > 0:
            avg_loss_cl = accumulate_loss_cl / processed_samples
            avg_loss_distill = accumulate_loss_distill / processed_samples if accumulate_loss_distill > 0 else 0.0
            avg_loss_total = accumulate_loss_total / processed_samples
            
            print(f"{client.name} -- Epoch {epoch+1}/{local_epoch} "
                  f"Contrastive Loss: {avg_loss_cl:.6f}, "
                  f"Distillation Loss: {avg_loss_distill:.6f}, "
                  f"Total Loss: {avg_loss_total:.6f}")
            
            epoch_losses.append(avg_loss_total)
        else:
            print(f"{client.name} -- Epoch {epoch+1}/{local_epoch} No samples processed.")
            epoch_losses.append(float('nan'))
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    final_avg_loss = epoch_losses[-1] if epoch_losses else float('nan')
    return {
        'contrastiveLosses': final_avg_loss
    }

def eval_gc_test(model, device, client):
    model.eval()
    test_graphs_eval = client.dataLoader['test_finetune']
    test_PIs = client.dataLoader['PI_test']
   
    all_preds = []
    all_labels = []
    
    test_embeddings = pass_data_iteratively(model, test_graphs_eval, test_PIs)
    test_pred = test_embeddings.max(1, keepdim=True)[1]
    test_labels = torch.LongTensor([graph.y for graph in test_graphs_eval]).to(device)

    test_pred_np = test_pred.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()
    test_pred = test_pred.cpu()
    test_labels = test_labels.cpu()
    test_correct = test_pred.eq(test_labels.view_as(test_pred)).sum().cpu().item()
    acc_test = test_correct / float(len(test_graphs_eval))
    
    f1_macro = f1_score(test_labels_np, test_pred_np, average='macro')

    return acc_test, f1_macro
