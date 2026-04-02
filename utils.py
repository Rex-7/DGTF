import torch
from torch_geometric.utils import to_networkx, degree, to_scipy_sparse_matrix
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy import sparse as sp

def convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes)) 

    new_graphs = []
    for i, tuple in enumerate(graph_infos):
        idx, x = tuple[0].edge_index[0], tuple[0].x
        deg = degree(idx, tuple[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tuple[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs

def get_maxDegree(graphs):
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree

    return maxdegree

def split_data(graphs, train=0.8, test=0.2, shuffle=True, seed=None):
    y = [g.y.item() for g in graphs]
    
    for g in graphs:
        if hasattr(g, 'y') and g.y.is_cuda:
            g.y = g.y.cpu()
        if hasattr(g, 'x') and g.x.is_cuda:
            g.x = g.x.cpu()
        if hasattr(g, 'edge_index') and g.edge_index.is_cuda:
            g.edge_index = g.edge_index.cpu()
        if hasattr(g, 'pi') and g.pi.is_cuda:
            g.pi = g.pi.cpu()
    
    graphs_tv, graphs_test = train_test_split(graphs, train_size=train, test_size=test, 
                                             stratify=y, shuffle=shuffle, random_state=seed)
    return graphs_tv, graphs_test

def remove_labels(graph_list):
    new_graph_list = []

    for item in graph_list:
        if isinstance(item, tuple):
            graph = item[0]
        else:
            graph = item

        new_graph = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr if 'edge_attr' in graph else None,
            pos=graph.pos if 'pos' in graph else None
        )
        new_graph_list.append((new_graph,) + item[1:]) if isinstance(item, tuple) else new_graph_list.append(new_graph)

    return new_graph_list

def get_numGraphLabels(dataset):
    s = set()
    for g in dataset:
        s.add(g.y.item())
    return len(s)


def _get_avg_nodes_edges(graphs):
    numNodes = 0.
    numEdges = 0.
    numGraphs = len(graphs)
    for g in graphs:
        numNodes += g.num_nodes
        numEdges += g.num_edges / 2.
    return numNodes/numGraphs, numEdges/numGraphs


def get_stats(df, ds, graphs_train, graphs_val=None, graphs_test=None):
    df.loc[ds, "#graphs_train"] = len(graphs_train)
    avgNodes, avgEdges = _get_avg_nodes_edges(graphs_train)
    df.loc[ds, 'avgNodes_train'] = avgNodes
    df.loc[ds, 'avgEdges_train'] = avgEdges

    if graphs_val:
        df.loc[ds, '#graphs_val'] = len(graphs_val)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_val)
        df.loc[ds, 'avgNodes_val'] = avgNodes
        df.loc[ds, 'avgEdges_val'] = avgEdges

    if graphs_test:
        df.loc[ds, '#graphs_test'] = len(graphs_test)
        avgNodes, avgEdges = _get_avg_nodes_edges(graphs_test)
        df.loc[ds, 'avgNodes_test'] = avgNodes
        df.loc[ds, 'avgEdges_test'] = avgEdges

    return df



import networkx as nx
import numpy as np
import gudhi as gd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def apply_graph_extended_persistence(A, filtration_val):
    num_vertices = A.shape[0]
    (xs, ys) = np.where(np.triu(A))
    st = gd.SimplexTree()
    for i in range(num_vertices):
        st.insert([i], filtration=-1e10)
    for idx, x in enumerate(xs):
        st.insert([x, ys[idx]], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])

    st.make_filtration_non_decreasing()
    st.extend_filtration()
    LD = st.extended_persistence()
    dgmOrd0, dgmRel1, dgmExt0, dgmExt1 = LD[0], LD[1], LD[2], LD[3]
    dgmOrd0 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmOrd0 if p[0] == 0]) if len(
        dgmOrd0) else np.empty([0, 2])
    dgmRel1 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmRel1 if p[0] == 1]) if len(
        dgmRel1) else np.empty([0, 2])
    dgmExt0 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt0 if p[0] == 0]) if len(
        dgmExt0) else np.empty([0, 2])
    dgmExt1 = np.vstack(
        [np.array([[min(p[1][0], p[1][1]), max(p[1][0], p[1][1])]]) for p in dgmExt1 if p[0] == 1]) if len(
        dgmExt1) else np.empty([0, 2])
    final_dgm = np.concatenate([dgmOrd0, dgmExt0, dgmRel1, dgmExt1], axis=0)
    return final_dgm

def persistence_images(dgm, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1.):
    PXs, PYs = dgm[:, 0], dgm[:, 1]
    if PXs.shape[0]==0 and PYs.shape[0]==0:
        norm_output = np.zeros((resolution)) + 1e-6
    else:
        xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
        x = np.linspace(xm, xM, resolution[0])
        y = np.linspace(ym, yM, resolution[1])
        X, Y = np.meshgrid(x, y)
        Zfinal = np.zeros(X.shape)
        X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]


        P0, P1 = np.reshape(dgm[:, 0], [1, 1, -1]), np.reshape(dgm[:, 1], [1, 1, -1])
        weight = np.abs(P1 - P0)
        distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

        if return_raw:
            lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
            lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
        else:
            weight = weight ** power
            Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

        output = [lw, lsum] if return_raw else Zfinal

        if normalization:
            norm_output = (output - np.min(output)) / (np.max(output) - np.min(output))
        else:
            norm_output = output

        if np.sum(np.isnan(norm_output))>0:
            norm_output = np.zeros((resolution)) + 1e-6
    return norm_output


def sublevel_persistence_diagram(net, method):
    assert method in ['degree','betweenness','communicability','eigenvector','closeness']

    if method == 'degree':
        return np.array(list(nx.degree_centrality(net).values()))
    elif method == 'betweenness':
        return np.array(list(nx.betweenness_centrality(net).values()))
    elif method == 'communicability':
        return np.array(list(nx.communicability_betweenness_centrality(net).values()))
    elif method == 'eigenvector':
        return np.array(list(nx.eigenvector_centrality(net, max_iter=50000, tol=1e-06).values()))
    elif method == 'closeness':
        return np.array(list(nx.closeness_centrality(net).values()))


def compute_PI_tensor(graphs, PI_dim, filtration_methods):
    all_PI_tensors = []
    
    for graph in graphs:
        if hasattr(graph, 'edge_index') and graph.edge_index.is_cuda:
            graph.edge_index = graph.edge_index.cpu()
        if hasattr(graph, 'x') and graph.x.is_cuda:
            graph.x = graph.x.cpu()
            
        edge_index = graph.edge_index.numpy().transpose()
        net = nx.from_edgelist(edge_index)
        adj = nx.adjacency_matrix(net).toarray()
        
        PI_list_i = []
        
        for method in filtration_methods:
            scores = sublevel_persistence_diagram(net, method)
            dgm = apply_graph_extended_persistence(A=adj, filtration_val=scores)
            pi = torch.FloatTensor(persistence_images(dgm, resolution=[PI_dim]*2))
            PI_list_i.append(pi)
            
        PI_tensor_i = torch.stack(PI_list_i)
        all_PI_tensors.append(PI_tensor_i)
    
    PI_concat = torch.stack(all_PI_tensors)
    return PI_concat



def match_nodes(original_data, data_aug, device, aug):
    if aug in ['dnodes', 'subgraph', 'random2', 'random3', 'random4']:
        if not original_data.x.is_cuda and 'cuda' in device:
            original_data = original_data.to(device)
        if not data_aug.x.is_cuda and 'cuda' in device:
            data_aug = data_aug.to(device)
        
        node_num, _ = original_data.x.size()
        
        edge_index = data_aug.edge_index
        node_mask = torch.zeros(node_num, dtype=torch.bool, device=device)
        
        node_mask.scatter_(0, edge_index[0], True)
        node_mask.scatter_(0, edge_index[1], True)
        
        idx_not_missing = torch.nonzero(node_mask).squeeze()
        
        data_aug.x = data_aug.x[idx_not_missing]
        
        if original_data.batch is not None:
            data_aug.batch = original_data.batch[idx_not_missing]
        
        node_num_aug = idx_not_missing.size(0)
        mapping = torch.full((node_num,), -1, dtype=torch.long, device=device)
        mapping[idx_not_missing] = torch.arange(node_num_aug, device=device)
        
        edge_mask = (edge_index[0] != edge_index[1])
        edge_index_filtered = edge_index[:, edge_mask]
        new_edge_index = mapping[edge_index_filtered]
        
        valid_edges = (new_edge_index[0] >= 0) & (new_edge_index[1] >= 0)
        data_aug.edge_index = new_edge_index[:, valid_edges]
        
    return data_aug


def pad_graphs(graphs, max_nodes):
    batch_size = len(graphs)
    if batch_size == 0:
        return graphs
    
    feat_dim = graphs[0].x.size(1)
    device = graphs[0].x.device
    
    padding_cache = {}
    
    for graph in graphs:
        num_nodes = graph.x.size(0)
        if num_nodes < max_nodes:
            pad_size = max_nodes - num_nodes
            if pad_size not in padding_cache:
                padding_cache[pad_size] = torch.zeros(pad_size, feat_dim, device=device)
            
            graph.x = torch.cat([graph.x, padding_cache[pad_size]], dim=0)
    
    return graphs



def pass_data_iteratively(model, graphs, PIs, args, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    args.mode = 'eval'

    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch_graph = [graphs[j] for j in sampled_idx]
        batch_PI = torch.stack([PIs[j] for j in sampled_idx])
        embeddings = model(batch_graph, None, batch_PI, None, args)
        output.append(embeddings.detach())
    return torch.cat(output, 0)


def pass_data_iteratively(model, graphs, PIs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))

    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        batch_graph = [graphs[j] for j in sampled_idx]
        batch_PI = torch.stack([PIs[j] for j in sampled_idx])
        _,out = model(batch_graph, None, batch_PI, None,mode='eval')
        output.append(out.detach())
    return torch.cat(output, 0)