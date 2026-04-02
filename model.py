import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.checkpoint import checkpoint

from mlp import MLP, MLP_output
from cnn import CNN
from tltorch import TRL, TCL

class CCA(nn.Module):
    def __init__(self, vector_size):
        super(CCA, self).__init__()

        self.weights = torch.nn.Parameter(torch.randn(vector_size, vector_size))
        self.weights.requires_grad_(True)

    def forward(self, query_vector, vectors):

        p = F.softmax(query_vector @ self.weights, dim=1)
        q = F.softmax(vectors @ self.weights, dim=1)
        kl_divergences = F.kl_div(p, q, reduction='none').sum(dim=2)
        weights = F.softmax(-kl_divergences, dim=1)

        return weights
    
    
class TASSL(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, tensor_layer_type, node_pooling, PI_dim, sublevel_filtration_methods, device, lambda1):
        super(TASSL, self).__init__()
        self.sublevel_filtration_methods = sublevel_filtration_methods
        self.device = device
        self.num_layers = num_layers
        self.num_neighbors = 5
        self.hidden_dim = hidden_dim
        self.lambda1 = lambda1
        self.embedding_dim = hidden_dim * num_layers
        
        self.K_projection = MLP(2, 2048, 4096, 10)
        self.sim_module = CCA(10)
        
        
        self.init_emb()
        self.GCNs = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.GCNs.append(GCNConv(input_dim,hidden_dim))
            else:
                self.GCNs.append(GCNConv(hidden_dim**2,hidden_dim))
            self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim**2))
        tensor_input_shape = (self.num_layers-1,hidden_dim,hidden_dim)
        tensor_hidden_shape = [32,32,32]
        if tensor_layer_type == 'TCL':
            self.GCN_tensor_layer = TCL(tensor_input_shape,tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.GCN_tensor_layer = TRL(tensor_input_shape,tensor_hidden_shape)

        self.cnn = CNN(len(sublevel_filtration_methods),hidden_dim,kernel_size=2,stride=2,num_layers=2)
        cnn_output_shape = self.cnn.cnn_output_dim(PI_dim)
        tensor_input_shape = (hidden_dim,cnn_output_shape,cnn_output_shape)
        tensor_hidden_shape = [32,32,32]

        if tensor_layer_type == 'TCL':
            self.PI_tensor_layer = TCL(tensor_input_shape,tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.PI_tensor_layer = TRL(tensor_input_shape,tensor_hidden_shape)

        self.attend = nn.Linear(2*32, 1)
        tensor_input_shape = (2*32,32,32)

        tensor_hidden_shape = [2*32,32,32]

        if tensor_layer_type == 'TCL':
            self.output_tensor_layer = TCL(tensor_input_shape,tensor_hidden_shape)
        elif tensor_layer_type == 'TRL':
            self.output_tensor_layer = TRL(tensor_input_shape,tensor_hidden_shape)
        self.output = MLP_output(32,output_dim,final_dropout)
    
    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def compute_batch_feat(self, batch_graph):
        edge_attr = None
        edge_mat_list = []
        start_idx = [0]
        pooled_x = []
        pooled_graph_sizes = []

        for i, graph in enumerate(batch_graph):
            x = graph.x.to(self.device)
            edge_index = graph.edge_index.to(self.device)
            
            start_idx.append(start_idx[i] + x.size(0))
            edge_mat_list.append(edge_index + start_idx[i]) 
            pooled_x.append(x)
            pooled_graph_sizes.append(x.size(0))

        pooled_X_concat = torch.cat(pooled_x, 0).to(self.device)
        Adj_block_idx = torch.cat(edge_mat_list, 1).to(self.device)
        return Adj_block_idx, pooled_X_concat, pooled_graph_sizes

    def GCN_layer(self, h, edge_index, layer):
        h = self.GCNs[layer](h, edge_index)
        h = self.mlps[layer](h)
        return h
            



    def forward(self, batch_graph1, batch_graph2, PIs_aug1, PIs_aug2, mode='predict'):
        if (mode == 'cl'):
            Adj_block_idx1, pooled_X_concat1, pooled_graph_sizes1 = self.compute_batch_feat(batch_graph1)
            Adj_block_idx2, pooled_X_concat2, pooled_graph_sizes2 = self.compute_batch_feat(batch_graph2)

            PIs_aug1 = PIs_aug1.to(self.device)
            PIs_aug2 = PIs_aug2.to(self.device)
            PI_emb1 = self.cnn(PIs_aug1)
            PI_hidden1 = checkpoint(lambda x: self.PI_tensor_layer(x), PI_emb1).to(self.device)
            PI_emb2 = self.cnn(PIs_aug2)
            PI_hidden2 = checkpoint(lambda x: self.PI_tensor_layer(x), PI_emb2).to(self.device)

            hidden_rep1 = []
            h1 = pooled_X_concat1
            edge_index1 = Adj_block_idx1

            hidden_rep2 = []
            h2 = pooled_X_concat2
            edge_index2 = Adj_block_idx2

            for layer in range(self.num_layers - 1):
                h1 = checkpoint(lambda inp: self.GCN_layer(inp, edge_index1, layer), h1)
                hidden_rep1.append(h1)

                h2 = checkpoint(lambda inp: self.GCN_layer(inp, edge_index2, layer), h2)
                hidden_rep2.append(h2)

            hidden_rep1 = torch.stack(hidden_rep1).transpose(0, 1)
            graph_sizes1 = pooled_graph_sizes1
            node_embeddings1 = torch.split(hidden_rep1, graph_sizes1, dim=0)

            hidden_rep2 = torch.stack(hidden_rep2).transpose(0, 1)
            graph_sizes2 = pooled_graph_sizes2
            node_embeddings2 = torch.split(hidden_rep2, graph_sizes2, dim=0)

            gcn_tensor1 = torch.zeros(len(graph_sizes1), 32, 32, 32, device=self.device)
            gcn_tensor2 = torch.zeros(len(graph_sizes1), 32, 32, 32, device=self.device)

            pi_tensor1 = torch.zeros(len(graph_sizes1), 32, 32, 32, device=self.device)
            pi_tensor2 = torch.zeros(len(graph_sizes1), 32, 32, 32, device=self.device)

            for g_i in range(len(graph_sizes1)):
                cur_node_embeddings1 = node_embeddings1[g_i]
                cur_node_embeddings1 = cur_node_embeddings1.view(-1, self.num_layers - 1, self.hidden_dim, self.hidden_dim)
                cur_node_embeddings_hidden1 = checkpoint(lambda x: self.GCN_tensor_layer(x), cur_node_embeddings1).to(self.device)
                cur_graph_tensor_hidden1 = torch.mean(cur_node_embeddings_hidden1, dim=0)
                cur_node_embeddings2 = node_embeddings2[g_i]
                cur_node_embeddings2 = cur_node_embeddings2.view(-1, self.num_layers - 1, self.hidden_dim, self.hidden_dim)
                cur_node_embeddings_hidden2 = checkpoint(lambda x: self.GCN_tensor_layer(x), cur_node_embeddings2).to(self.device)
                cur_graph_tensor_hidden2 = torch.mean(cur_node_embeddings_hidden2, dim=0)

                gcn_tensor1[g_i] = cur_graph_tensor_hidden1
                gcn_tensor2[g_i] = cur_graph_tensor_hidden2


                cur_PI_tensor_hidden1 = PI_hidden1[g_i]
                cur_PI_tensor_hidden2 = PI_hidden2[g_i]
                pi_tensor1[g_i] = cur_PI_tensor_hidden1
                pi_tensor2[g_i] = cur_PI_tensor_hidden2
            

            gcn_tensor1_flat = gcn_tensor1.reshape(gcn_tensor1.size(0), -1)
            gcn_tensor2_flat = gcn_tensor2.reshape(gcn_tensor2.size(0), -1)

            gcn_tensor_loss = self.loss_cal(gcn_tensor1_flat, gcn_tensor2_flat)

            pi_tensor1_flat = pi_tensor1.reshape(pi_tensor1.size(0), -1)
            pi_tensor2_flat = pi_tensor2.reshape(pi_tensor2.size(0), -1)

            pi_tensor_loss = self.loss_cal(pi_tensor1_flat, pi_tensor2_flat) 

            loss = gcn_tensor_loss + self.lambda1 * pi_tensor_loss

            return loss

        
        else:
            Adj_block_idx, pooled_X_concat, pooled_graph_sizes = self.compute_batch_feat(batch_graph1)
        
            PIs_aug1 = PIs_aug1.to(self.device)

            PI_emb = self.cnn(PIs_aug1)
            PI_hidden = checkpoint(lambda x: self.PI_tensor_layer(x), PI_emb).to(self.device)

            hidden_rep = []
            h = pooled_X_concat
            edge_index = Adj_block_idx
            for layer in range(self.num_layers-1):
                h = checkpoint(lambda inp: self.GCN_layer(inp, edge_index, layer), h)
                hidden_rep.append(h)
            hidden_rep = torch.stack(hidden_rep).transpose(0,1)

            graph_sizes = pooled_graph_sizes
            batch_graph_tensor = torch.zeros(len(graph_sizes), 2 * 32, 32, 32, device=self.device)
            node_embeddings = torch.split(hidden_rep, graph_sizes, dim=0)

            for g_i in range(len(graph_sizes)):
                cur_node_embeddings = node_embeddings[g_i]
                cur_node_embeddings = cur_node_embeddings.view(-1,self.num_layers-1,self.hidden_dim,self.hidden_dim)
                cur_node_embeddings_hidden = checkpoint(lambda x: self.GCN_tensor_layer(x), cur_node_embeddings).to(self.device)
                cur_graph_tensor_hidden = torch.mean(cur_node_embeddings_hidden,dim=0)
                
                cur_PI_tensor_hidden = PI_hidden[g_i]
                cur_tensor_hidden = torch.cat([cur_graph_tensor_hidden, cur_PI_tensor_hidden], dim=0)
                batch_graph_tensor[g_i] = cur_tensor_hidden


            batch_graph_tensor = self.output_tensor_layer(batch_graph_tensor).transpose(1, 3)
            batch_graph_attn = self.attend(batch_graph_tensor).squeeze()
            score = self.output(batch_graph_attn)

            return score
    
    
    def loss_cal(self, x_aug1, x_aug2):
        T = 0.2
        batch_size = x_aug1.size(0)
        feature_size = x_aug1.size(1)
        eps = 1e-8
        x_abs = x_aug1.norm(dim=1).clamp_min(eps)
        x_aug_abs = x_aug2.norm(dim=1).clamp_min(eps)

        sim_matrix = torch.einsum('ik,jk->ij', x_aug1, x_aug2) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = torch.diag(sim_matrix)
        denom = (sim_matrix.sum(dim=1) - pos_sim).clamp_min(eps)
        loss = pos_sim / denom
        loss = - torch.log(loss).mean()

        return loss
    
    def get_cnn_features(self, batch_PI):
        batch_PI = batch_PI.to(self.device)

        PI_emb = self.cnn(batch_PI)
        return PI_emb
    
    def loss(self, pred, label):
        criterion = torch.nn.CrossEntropyLoss()
        label = label.to(self.device)
        pred = pred.to(self.device)
        loss = criterion(pred, label)
        return loss

class Localmodel(nn.Module):
    def __init__(self, feature_extractor, head):
        super(Localmodel, self).__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, batch_graph, feat=False):
        out = self.feature_extractor(batch_graph)
        if feat:
            return out
        else:
            out = self.head(out)
            return out

class Split_model(nn.Module):
    def __init__(self, base, head):
        super(Split_model, self).__init__()

        self.base = base
        self.head = head

        hidden_dim = 32
        self.linear = MLP(2,hidden_dim**3,4096,2048)
        self.K = 10
        self.T = 0.1
        self.projection_size = 2048
        self.register_buffer("queue", torch.randn(self.projection_size, self.projection_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_len = 1024

    def forward(self, batch_graph1, batch_graph2,PI1,PI2,mode='predict'):
        feature = self.base(batch_graph1,batch_graph2,PI1,PI2,mode)
        out = self.head(feature)

        return feature, out

    def loss(self, pred, label):
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(pred, label)
        return loss
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.queue_len

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x, device):
        idx_shuffle = torch.randperm(x.shape[0]).to(device)

        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    def contrastive_loss(self, q, k, device):
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            k, idx_unshuffle = self._batch_shuffle_single_gpu(k, device)
            k = nn.functional.normalize(k, dim=1)

            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        loss = nn.CrossEntropyLoss().to(device)(logits, labels)

        return loss, q, k
    
    def distillation(self, rep_s, neibor_reps, device):
        rep_s = rep_s.float()
        neibor_reps = neibor_reps.float()
        K_result = torch.zeros(rep_s.size(0), neibor_reps.size(1), self.K, device=device, dtype=torch.float32)
        for i in range(neibor_reps.size(1)):
            tmp_client_result = neibor_reps[:, i, :].squeeze(dim=1).float()
            K_result[:, i, :] = self.base.K_projection(tmp_client_result)
        K_online_pred = self.base.K_projection(rep_s).unsqueeze(dim=1)
        weight = self.base.sim_module(K_online_pred, K_result)
        teacher_q = torch.sum(weight.unsqueeze(2) * neibor_reps, dim=1)
        loss_distill, _, _ = self.contrastive_loss(teacher_q, rep_s, device)
        return loss_distill