import copy
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import OneHotDegree
from model import TASSL, Split_model
from client import Client_GC
from utils import get_stats, split_data, get_numGraphLabels, compute_PI_tensor, remove_labels
import networkx as nx
from aug import TUDataset_aug
from copy import deepcopy


def get_dataset_specific_params(dataset_name):
    dataset_params = {
        "AIDS": (4, 64),
        "BZR": (4, 16),
        "COX2": (3, 64),
        "DHFR": (5, 64),
        "IMDB-BINARY": (5, 64),
        "IMDB-MULTI": (3, 16),
        "KKI": (5, 16),
        "OHSU": (4, 16),
        "Peking_1": (3, 32),
        "PROTEINS": (3, 32),
        "Letter-high": (5, 64),
        "Letter-med": (4, 64),
        "Letter-low": (3, 64),
        "MUTAG": (4, 32),
        "NCI1": (5, 64),
        "PTC_MR": (3, 32)
    }
    
    if dataset_name in dataset_params:
        return dataset_params[dataset_name]
    else:
        raise ValueError(f"Dataset {dataset_name} parameters not defined")


def add_noise_to_PI(pi_tensor, noise_level=0.1):
    noise = torch.randn(pi_tensor.size(), device=pi_tensor.device) * noise_level  
    noisy_pi_tensor = pi_tensor + noise
    return noisy_pi_tensor

def get_optimal_augmentations(dataset_name):
    augmentation_mapping = {
        "NCI1": ("dnodes", "mask_nodes"),
        "PROTEINS": ("dnodes", "pedges"),
        "DD": ("dnodes", "subgraph"),
        "MUTAG": ("dnodes", "pedges"),
        "DHFR": ("pedges", "none"),
        "BZR": ("dnodes", "mask_nodes"),
        "COX2": ("dnodes", "subgraph"),
        "PTC_MR": ("dnodes", "pedges"),
        "PTC_FM": ("dnodes", "pedges"),
        "IMDB-BINARY": ("subgraph", "none"),
        "IMDB-MULTI": ("subgraph", "none"),
        "REDDIT-BINARY": ("subgraph", "none"),
        "AIDS": ("dnodes", "pedges"),
        "Peking_1": ("dnodes", "none"),
        "KKI": ("dnodes", "none"),
        "OHSU": ("dnodes", "none"),
        "Letter-high": ("pedges", "none"),
        "Letter-med": ("pedges", "none"),
        "Letter-low": ("pedges", "none")
    }
    
    if dataset_name in augmentation_mapping:
        return augmentation_mapping[dataset_name]
    
    if "IMDB" in dataset_name:
        return ("subgraph", "none")
    elif any(x in dataset_name for x in ["Letter", "REDDIT"]):
        return ("subgraph", "pedges")
    elif any(x in dataset_name for x in ["PROTEINS", "ENZYMES"]):
        return ("dnodes", "pedges")
    else:
        return ("dnodes", "pedges")

def prepareData_multiDS(args, datapath, group='sm', batchSize=128, seed=None, save_dir='./'):
    processed_data_dir = os.path.join(save_dir, 'processed_graphs')
    os.makedirs(processed_data_dir, exist_ok=True)


    if group == 'sm':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    elif group == 'bio':
        datasets = ["KKI", "OHSU","Peking_1", "PROTEINS"]
    elif group == 'sn':
        datasets = ["IMDB-MULTI", "IMDB-BINARY"]
    elif group == 'cv':
        datasets = ["Letter-high", "Letter-med", "Letter-low"]
    elif group == 'sm-bio':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"] + ["KKI", "OHSU","Peking_1", "PROTEINS"]
    elif group == 'sm-bio-cv':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"] + ["KKI", "OHSU","Peking_1", "PROTEINS"] + ["Letter-high", "Letter-med", "Letter-low"]
    elif group == 'sm-bio-cv-sn':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"] + ["KKI", "OHSU","Peking_1", "PROTEINS"] + ["Letter-high", "Letter-med", "Letter-low"] + ["IMDB-MULTI", "IMDB-BINARY"]
    else:
        datasets = []
        datasets.append(group)


    splitedData = {}
    df = pd.DataFrame()
    
    filtration_methods = ['degree','betweenness','eigenvector','closeness']
    
    for data in datasets:
        aug1_type, aug2_type = get_optimal_augmentations(data)
        if aug1_type in ['dnodes', 'subgraph'] or aug2_type in ['dnodes', 'subgraph']:
            aug='dnodes'
        else:
            aug = None
        print(f"Selected augmentation strategy for dataset {data}: {aug1_type} + {aug2_type}")
        
        processed_file = os.path.join(processed_data_dir, f"{data}_processed.pt")
        aug1_file = os.path.join(processed_data_dir, f"{data}_aug1_processed.pt")
        aug2_file = os.path.join(processed_data_dir, f"{data}_aug2_processed.pt")
        
        if os.path.exists(processed_file) and os.path.exists(aug1_file) and (aug2_type == 'none' or os.path.exists(aug2_file)):
            print(f"Loading preprocessed data for dataset {data}...")
            graphs_orig = torch.load(processed_file)
            aug_graphs1 = torch.load(aug1_file)
            if aug2_type != 'none':
                aug_graphs2 = torch.load(aug2_file)
            else:
                aug_graphs2 = None
            print(f"Successfully loaded preprocessed data for dataset {data}")
        else:
            print(f"Processing dataset {data} and saving...")
            if data == "IMDB-BINARY":
                transform = OneHotDegree(135, cat=False)
                datasets_orig = TUDataset_aug(f"{datapath}/TUDataset", data, pre_transform=transform, aug='none')
                datasets_aug1 = TUDataset_aug(f"{datapath}/TUDataset", data, pre_transform=transform, aug=aug1_type)
                datasets_aug2 = TUDataset_aug(f"{datapath}/TUDataset", data, pre_transform=transform, aug=aug2_type) if aug2_type != 'none' else None
            elif data == "IMDB-MULTI":
                transform = OneHotDegree(88, cat=False)
                datasets_orig = TUDataset_aug(f"{datapath}/TUDataset", data, pre_transform=transform, aug='none')
                datasets_aug1 = TUDataset_aug(f"{datapath}/TUDataset", data, pre_transform=transform, aug=aug1_type)
                datasets_aug2 = TUDataset_aug(f"{datapath}/TUDataset", data, pre_transform=transform, aug=aug2_type) if aug2_type != 'none' else None
            elif "Letter" in data:
                datasets_orig = TUDataset_aug(f"{datapath}/TUDataset", data, use_node_attr=True, aug='none')
                datasets_aug1 = TUDataset_aug(f"{datapath}/TUDataset", data, use_node_attr=True, aug=aug1_type)
                datasets_aug2 = TUDataset_aug(f"{datapath}/TUDataset", data, use_node_attr=True, aug=aug2_type) if aug2_type != 'none' else None
            else:
                datasets_orig = TUDataset_aug(f"{datapath}/TUDataset", data, aug='none')
                datasets_aug1 = TUDataset_aug(f"{datapath}/TUDataset", data, aug=aug1_type)
                datasets_aug2 = TUDataset_aug(f"{datapath}/TUDataset", data, aug=aug2_type) if aug2_type != 'none' else None

            graphs_orig = [data for data, data_aug in datasets_orig]
            aug_graphs1 = [data_aug for data, data_aug in datasets_aug1]
            aug_graphs2 = [data_aug for data, data_aug in datasets_aug2] if datasets_aug2 is not None else None


        pi_file = os.path.join(processed_data_dir, f"{data}_{args.PI_dim}_PI.pt")
        if os.path.exists(pi_file):
            PIs = torch.load(pi_file, map_location=args.device)
            print(f'Successfully loaded PI tensor for dataset {data}, PI_dim = {args.PI_dim}')
        else:
            PIs = compute_PI_tensor(graphs_orig, args.PI_dim, filtration_methods)
            PIs = PIs.to(args.device)
            torch.save(PIs, pi_file)
            print(f'Computed and saved PI tensor for dataset {data}, PI_dim = {args.PI_dim}')
        
        noisy_PIs = add_noise_to_PI(PIs, noise_level=0.1)
        
        pi_aug1_file = os.path.join(processed_data_dir, f"{data}_{args.PI_dim}_{aug1_type}_PI.pt")
        if os.path.exists(pi_aug1_file):
            PIs_aug1 = torch.load(pi_aug1_file, map_location=args.device)
            print(f'Successfully loaded {aug1_type} augmented PI tensor for dataset {data}, PI_dim = {args.PI_dim}')
        else:
            PIs_aug1 = compute_PI_tensor(aug_graphs1, args.PI_dim, filtration_methods)
            PIs_aug1 = PIs_aug1.to(args.device)
            torch.save(PIs_aug1, pi_aug1_file)
            print(f'Computed and saved {aug1_type} augmented PI tensor for dataset {data}, PI_dim = {args.PI_dim}')
        
        noisy_PIs_aug1 = add_noise_to_PI(PIs_aug1, noise_level=0.1)
        
        if aug_graphs2:
            pi_aug2_file = os.path.join(processed_data_dir, f"{data}_{args.PI_dim}_{aug2_type}_PI.pt")
            if os.path.exists(pi_aug2_file):
                PIs_aug2 = torch.load(pi_aug2_file, map_location=args.device)
                print(f'Successfully loaded {aug2_type} augmented PI tensor for dataset {data}, PI_dim = {args.PI_dim}')
            else:
                PIs_aug2 = compute_PI_tensor(aug_graphs2, args.PI_dim, filtration_methods)
                PIs_aug2 = PIs_aug2.to(args.device)
                torch.save(PIs_aug2, pi_aug2_file)
                print(f'Computed and saved {aug2_type} augmented PI tensor for dataset {data}, PI_dim = {args.PI_dim}')
            
            noisy_PIs_aug2 = add_noise_to_PI(PIs_aug2, noise_level=0.1)
        else:
            PIs_aug2 = PIs
            noisy_PIs_aug2 = noisy_PIs

        print("  **", data, len(graphs_orig))
        graphs_without_labels = remove_labels(graphs_orig)
        dataset_aug1_without_labels = remove_labels(aug_graphs1)
        if aug_graphs2:
            dataset_aug2_without_labels = remove_labels(aug_graphs2) 
        else:
            dataset_aug2_without_labels = graphs_without_labels
        graphs_train, graphs_test , train_PIs, test_PIs= train_test_split(graphs_orig,noisy_PIs, test_size=0.1,random_state=args.seed)
        
        num_node_features = graphs_orig[0].num_node_features
        num_graph_labels = len(set([g.y.item() for g in graphs_orig]))

        splitedData[data] = ({
            'train_finetune': graphs_train,  
            'test_finetune': graphs_test,
            'PI_train': train_PIs,
            'PI_test': test_PIs,
            'train_ori': graphs_without_labels,
            'train_aug1': dataset_aug1_without_labels,
            'train_aug2': dataset_aug2_without_labels,
            'PI_aug1': noisy_PIs_aug1,
            'PI_aug2': noisy_PIs_aug2,
            
        }, num_node_features, num_graph_labels, len(graphs_train),aug)

        df = get_stats(df, data, graphs_train, graphs_test=graphs_test)

    return splitedData, df


def setup_devices_TTG(splitedData, args):
    idx_clients = {}
    clients = []
        
    for idx, ds in enumerate(splitedData.keys()):
        idx_clients[idx] = ds
        dataloaders, num_node_features, num_graph_labels, train_size ,aug= splitedData[ds]

        if True:
            filtration_methods = ['degree','betweenness','eigenvector','closeness']
            
            dataset_num_layers, dataset_hidden_dim = get_dataset_specific_params(ds)
            former = TASSL(dataset_num_layers, args.num_mlp_layers, num_node_features, 
                            dataset_hidden_dim, num_graph_labels, args.final_dropout,
                            args.tensor_layer_type, args.node_pooling, 
                            args.PI_dim, filtration_methods, args.device, args.lambda1)   
            head = former.output
            former.output = nn.Identity()
            basicModel = Split_model(former, head)
            cmodel_gc = copy.deepcopy(basicModel)
            cmodel_gc = cmodel_gc.to("cpu")
            
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
        clients.append(Client_GC(copy.deepcopy(cmodel_gc), idx, ds, train_size, dataloaders, optimizer, args,aug))

    server = None
    return clients, server, idx_clients

