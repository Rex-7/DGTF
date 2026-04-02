import os
import argparse
import random
import copy
import setupGC
from training import *
import torch
from tensorboardX import SummaryWriter
from pathlib import Path
import numpy as np
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def process_DGTF(args, clients, server, summary_writer, all_data_loader=None):
    print("\nDone setting up DGTF devices.")

    print("Running DGTF ...")
    frame = run_DGTF_joint(args, clients, server, args.num_rounds, args.local_epoch, samp=None,
                       summary_writer=summary_writer)

    if args.repeat is None:
        outfile = os.path.join(outpath, f'acc-seed{args.seed}_ftlr{args.finetune_lr}.csv')
    else:
        outfile = os.path.join(outpath, f'{args.repeat}_accuracy_DGTF_GC.csv')

    file_exists = os.path.isfile(outfile)

    frame.to_csv(outfile, mode='a', header=not file_exists)

    print(f"Wrote to file: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CPU / GPU device.')
    parser.add_argument('--alg', type=str, default='joint',
                        help='Name of algorithms.')
    parser.add_argument('--alpha', type=float, default=4,
                        help='alpha for distillation loss.')
    parser.add_argument('--lambda1',type=float,default=0.3,
                        help='lambda for TRM loss')                        
    parser.add_argument('--num_rounds', type=int, default=200,
                        help='number of rounds to simulate;')
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='number of local epochs;')
    parser.add_argument('--lr', type=float, default=0.0001, help='lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--finetune_lr', type=float, default=0.01, help='lr for finetune')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=789)
    parser.add_argument('--datapath', type=str, default='./Data1',
                        help='The input path of data.')
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    
    parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='sm')

    parser.add_argument('--sublevel_filtration_methods', nargs='+', type=str, default=['degree','betweenness','closeness','eigenvector'],
    					help='Methods for sublevel filtration on PDs')
    parser.add_argument('--PI_dim', type=int, default=50,
                        help='PI size: PI_dim * PI_dim')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GCN layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--tensor_layer_type', type = str, default = "TCL", choices=["TCL","TRL"],
                                        help='Tensor layer type: TCL/TRL')
    parser.add_argument('--node_pooling', action="store_false",
    					help='node pooling based on node scores')
    parser.add_argument('--save_dir', type=str, default='./',
                       help='Directory path for saving datasets')


    
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    seed_dataSplit = 123
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    outpath = os.path.join(args.outbase, 'raw', args.data_group)
    Path(outpath).mkdir(parents=True, exist_ok=True)

    print("Preparing data ...")
    
    splitedData, df_stats = setupGC.prepareData_multiDS(args, args.datapath, args.data_group, args.batch_size, 
                                                      seed=seed_dataSplit, save_dir=args.save_dir)
    print("Done")
    init_clients, init_server, init_idx_clients = setupGC.setup_devices_TTG(splitedData, args)
    
    print("\nDone setting up devices.")

    sw_path = os.path.join(args.outbase, 'raw', 'tensorboard', f'{args.data_group}_{args.alg}__{args.repeat}_{time.strftime("%Y%m%d_%H%M%S")}')
    summary_writer = SummaryWriter(sw_path)

    process_DGTF(args, clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), summary_writer=summary_writer)


