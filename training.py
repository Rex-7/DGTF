import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib as mpl
import json
import seaborn as sns
from torch_geometric.data import DataLoader
import copy
from setupGC import *
import time
from sklearn.decomposition import PCA


def save_finetune_params_state(model):
    finetune_params = {}
    for name, param in model.head.named_parameters():
        finetune_params[f'head.{name}'] = param.data.clone()
    for name, param in model.base.attend.named_parameters():
        finetune_params[f'base.attend.{name}'] = param.data.clone()
    for name, param in model.base.output_tensor_layer.named_parameters():
        finetune_params[f'base.output_tensor_layer.{name}'] = param.data.clone()
    return finetune_params

def restore_finetune_params_state(model, saved_state):
    for name, param in model.head.named_parameters():
        if f'head.{name}' in saved_state:
            param.data.copy_(saved_state[f'head.{name}'])
    for name, param in model.base.attend.named_parameters():
        if f'base.attend.{name}' in saved_state:
            param.data.copy_(saved_state[f'base.attend.{name}'])
    for name, param in model.base.output_tensor_layer.named_parameters():
        if f'base.output_tensor_layer.{name}' in saved_state:
            param.data.copy_(saved_state[f'base.output_tensor_layer.{name}'])

def run_DGTF_joint(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, summary_writer=None):
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    round_results = pd.DataFrame()

    log_file_path = None
    log_file_handle = None
    if hasattr(args, 'save_dir') and args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        log_filename = f'{args.alg}_{args.data_group}_local-epoch_{local_epoch}-a{args.alpha}_la{args.lambda1}_round_results_seed{args.seed}.txt'
        log_file_path = os.path.join(args.save_dir, log_filename)
        log_file_handle = open(log_file_path, 'w')
        log_file_handle.write(f"Experiment Results (DGTF_joint) for Algorithm: {args.alg}\n")
        log_file_handle.write("="*80 + "\n\n")
    
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        if c_round == 1:
            print("***************First round: Only contrastive learning, skipping distillation*************")
            for client1 in clients:
                client1.neibor_models = []
        else:
            print("***************Setting neighbor models for distillation*************")
            for client1 in clients:
                client1.neibor_models = []
                for client2 in clients:
                    if client1.name != client2.name:
                        client1.neibor_models.append((client2.model.base.cnn, client2.model.base.PI_tensor_layer))
            
        print("***************Starting joint training*************")
        for client in clients:
            client.joint_train(local_epoch)
            torch.cuda.empty_cache()
            gc.collect()
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        print("***************Joint training completed*********************")
        
        print(f"***************Round {c_round} fine-tuning starts*************")
        round_accs = []
        round_f1_macros = []
        
        for client in clients:
            saved_state = save_finetune_params_state(client.model)
            
            local_finetune_epoch = 40
            client.finetune(local_finetune_epoch)
            
            acc, f1_macro = client.evaluate()
            round_accs.append(acc)
            round_f1_macros.append(f1_macro)
            
            print(f"Round {c_round} - Client {client.name}: acc={acc:.4f}, f1_macro={f1_macro:.4f}")
            if log_file_handle:
                log_file_handle.write(f"  Round {c_round} - Client {client.name}: Acc={acc:.4f}, F1-macro={f1_macro:.4f}\n")
            
            restore_finetune_params_state(client.model, saved_state)
            
            gc.collect()
        
        mean_acc = np.mean(round_accs)
        mean_f1_macro = np.mean(round_f1_macros)
        round_results.loc['mean', f'round_{c_round}_acc'] = mean_acc
        round_results.loc['mean', f'round_{c_round}_f1_macro'] = mean_f1_macro
        
        print(f"Round {c_round} - Average Accuracy: {mean_acc:.4f}, Average F1-macro: {mean_f1_macro:.4f}")
        
        if summary_writer:
            summary_writer.add_scalar(f'Round_Finetune/Mean_Acc_{args.alg}', mean_acc, c_round)
            summary_writer.add_scalar(f'Round_Finetune/Mean_F1_Macro_{args.alg}', mean_f1_macro, c_round)
        if log_file_handle:
            log_file_handle.write(f"Round {c_round} - Average Acc: {mean_acc:.4f}, Average F1-macro: {mean_f1_macro:.4f}\n\n")
        
        print(f"***************Round {c_round} fine-tuning completed*************")
        
        if c_round % 1 == 0:
            losses = []
            for idx in range(len(clients)):
                loss = clients[idx].train_stats['contrastiveLosses']
                losses.append(loss)

            mean_loss = np.mean(losses)
            
            if np.isnan(mean_loss):
                print(f"Fatal error: Invalid average loss (NaN) detected in round {c_round}. Terminating task.")
                if summary_writer:
                    summary_writer.add_scalar(f'Test/Loss/Mean_{args.alg}', mean_loss, c_round)
                return round_results

            print(f"Round {c_round} - loss: {mean_loss}")

            if summary_writer:
                summary_writer.add_scalar(f'Test/Loss/Mean_{args.alg}', mean_loss, c_round)
    
    frame = pd.DataFrame()
    local_finetune_epoch = 100
    print("Federation communication ended, performing final client fine-tuning...")
    all_accs = []
    all_f1_macros = []
    for client in clients:
        client.finetune(local_finetune_epoch)
        acc, f1_macro = client.evaluate()
        frame.loc[client.name, 'test_acc'] = acc
        frame.loc[client.name, 'test_f1_macro'] = f1_macro
        all_accs.append(acc)
        all_f1_macros.append(f1_macro)
        gc.collect()
        print(f"client {client.name} - acc: {acc}, f1_macro: {f1_macro}")
    
    mean_acc = np.mean(all_accs)
    mean_f1_macro = np.mean(all_f1_macros)
    frame.loc['mean', 'test_acc'] = mean_acc
    frame.loc['mean', 'test_f1_macro'] = mean_f1_macro
    print(f"Final average accuracy for all clients: {mean_acc:.4f}, Average F1-macro: {mean_f1_macro:.4f}")

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    print(fs)
    

    if log_file_handle:
        log_file_handle.close()
        print(f"\nRound results saved to: {log_file_path}")

    return frame







