# DGTF: Cross-Domain Decentralized Graph Learning with Topology-Aware Knowledge Fusion

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-b31b1b)](#citation)

Official code for the AAAI 2026 paper.

> Paper link: **TODO** (add the proceedings or arXiv URL).

## Overview

DGTF trains a graph classifier on TUDatasets in a cross-dataset setting. `setupGC.setup_devices_TTG` builds one `Client_GC` per dataset and sets the server to `None`. The training loop does not average client parameters. From the second communication round, each client is given the other clients' persistence-image encoder modules (`cnn` and `PI_tensor_layer`) for that round and adds a distillation loss on those modules.

For every graph, `utils.compute_PI_tensor` builds a 4-channel persistence image:

1. Node scores from degree, betweenness, eigenvector, and closeness centrality (`networkx`).
2. Extended persistence via Gudhi (`gudhi.SimplexTree.extend_filtration` / `extended_persistence`).
3. A persistence image of size `PI_dim` × `PI_dim` (default 50).

Channels are stacked into one tensor. `add_noise_to_PI` then adds Gaussian noise with scale `0.1` to the tensors that enter training and evaluation. The tensor written to disk is the clean one.

The encoder is `TASSL` (`model.py`), wrapped by `Split_model`:

- A stack of `GCNConv` layers, each followed by an `MLP`. `TASSL` builds `num_layers - 1` convolutions. Depth and width are the per-dataset pair from `get_dataset_specific_params`, not the `--num_layers` / `--hidden_dim` flags.
- Node embeddings go through a tensor layer (`tltorch.TCL` or `tltorch.TRL`, flag `--tensor_layer_type`) and are averaged over nodes.
- The persistence image goes through `CNN` (two conv–batch-norm–ReLU–max-pool blocks) and a second tensor layer.
- Contrastive mode (`mode='cl'`) encodes two augmented views. `TASSL.loss_cal` is an in-batch contrastive loss (temperature `0.2`) on the GCN tensor and on the PI tensor. The sum is `gcn_tensor_loss + lambda1 * pi_tensor_loss`.
- Prediction mode concatenates the two tensors, passes them through `output_tensor_layer` and a linear attention map (`attend`), then `MLP_output` (batch-norm, ReLU, dropout, linear) for class logits.

Augmentations come from `aug.py` (`drop_nodes`, `permute_edges`, `subgraph`, `mask_nodes`). The pair used for each dataset is fixed in `get_optimal_augmentations`.

`training.run_DGTF_joint` loops for `--num_rounds` (default 200). Each round runs `--local_epoch` epochs of `Client_GC.joint_train` (default 1). Round 1 leaves neighbor modules empty, so the loss is contrastive only. Later rounds set `neibor_models` and add `alpha * distillation`, where `Split_model.distillation` projects neighbor PI features, weights them with the `CCA` module, and applies `contrastive_loss`. After each round the code fine-tunes `head`, `attend`, and `output_tensor_layer` for 40 epochs (`linear_probe`), logs test accuracy and macro-F1, and restores those parameters. After the last round it fine-tunes for 100 epochs and keeps that run. Inside `linear_probe`, the epoch with the highest test accuracy is the one whose weights are kept for the metric (and, on the final call, for the returned model).

## Framework

<!-- TODO: add the framework figure at assets/framework.png, then uncomment the image. -->

<!-- ![DGTF framework](assets/framework.png) -->

`assets/` is reserved for that figure. The image is not in the repository yet.

## Repository structure

```text
main.py            CLI, seeds, output paths, call to run_DGTF_joint
setupGC.py         Data groups, TU loading, PI cache, one client per dataset
training.py        Communication rounds, fine-tune, CSV rows, TensorBoard scalars
client.py          Client_GC joint training, distillation, fine-tune, evaluation
model.py           TASSL, Split_model, CCA, TCL/TRL fusion
cnn.py             CNN over persistence images
mlp.py             MLP blocks and MLP_output classifier
utils.py           Persistence diagrams and images, padding, split stats
aug.py             TUDataset_aug loader and graph augmentations
requirements.txt   Pinned Python packages (install needs the cu124 index)
LICENSE            MIT license; copyright holder is a TODO placeholder
README.md          This file
.gitignore         Python caches, training outputs, processed dataset files
assets/.gitkeep    Placeholder directory for assets/framework.png
Data1/TUDataset/   Committed TU raw text, one file per line below. PyG processed/ caches are gitignored
```

<details>
<summary>Committed raw files under <code>Data1/TUDataset</code></summary>

- `Data1/TUDataset/AIDS/raw/AIDS_A.txt`
- `Data1/TUDataset/AIDS/raw/AIDS_edge_labels.txt`
- `Data1/TUDataset/AIDS/raw/AIDS_graph_indicator.txt`
- `Data1/TUDataset/AIDS/raw/AIDS_graph_labels.txt`
- `Data1/TUDataset/AIDS/raw/AIDS_label_readme.txt`
- `Data1/TUDataset/AIDS/raw/AIDS_node_attributes.txt`
- `Data1/TUDataset/AIDS/raw/AIDS_node_labels.txt`
- `Data1/TUDataset/BZR/raw/BZR_A.txt`
- `Data1/TUDataset/BZR/raw/BZR_graph_indicator.txt`
- `Data1/TUDataset/BZR/raw/BZR_graph_labels.txt`
- `Data1/TUDataset/BZR/raw/BZR_node_attributes.txt`
- `Data1/TUDataset/BZR/raw/BZR_node_labels.txt`
- `Data1/TUDataset/BZR/raw/README.txt`
- `Data1/TUDataset/COX2/raw/COX2_A.txt`
- `Data1/TUDataset/COX2/raw/COX2_graph_indicator.txt`
- `Data1/TUDataset/COX2/raw/COX2_graph_labels.txt`
- `Data1/TUDataset/COX2/raw/COX2_node_attributes.txt`
- `Data1/TUDataset/COX2/raw/COX2_node_labels.txt`
- `Data1/TUDataset/COX2/raw/README.txt`
- `Data1/TUDataset/DD/raw/DD_A.txt` — present on disk; not in any named `data_group`
- `Data1/TUDataset/DD/raw/DD_graph_indicator.txt`
- `Data1/TUDataset/DD/raw/DD_graph_labels.txt`
- `Data1/TUDataset/DD/raw/DD_node_labels.txt`
- `Data1/TUDataset/DD/raw/README.txt`
- `Data1/TUDataset/DHFR/raw/DHFR_A.txt`
- `Data1/TUDataset/DHFR/raw/DHFR_graph_indicator.txt`
- `Data1/TUDataset/DHFR/raw/DHFR_graph_labels.txt`
- `Data1/TUDataset/DHFR/raw/DHFR_node_attributes.txt`
- `Data1/TUDataset/DHFR/raw/DHFR_node_labels.txt`
- `Data1/TUDataset/DHFR/raw/README.txt`
- `Data1/TUDataset/ENZYMES/raw/ENZYMES_A.txt` — present on disk; not in any named `data_group`
- `Data1/TUDataset/ENZYMES/raw/ENZYMES_graph_indicator.txt`
- `Data1/TUDataset/ENZYMES/raw/ENZYMES_graph_labels.txt`
- `Data1/TUDataset/ENZYMES/raw/ENZYMES_node_attributes.txt`
- `Data1/TUDataset/ENZYMES/raw/ENZYMES_node_labels.txt`
- `Data1/TUDataset/ENZYMES/raw/README.txt`
- `Data1/TUDataset/IMDB-BINARY/raw/IMDB-BINARY_A.txt`
- `Data1/TUDataset/IMDB-BINARY/raw/IMDB-BINARY_graph_indicator.txt`
- `Data1/TUDataset/IMDB-BINARY/raw/IMDB-BINARY_graph_labels.txt`
- `Data1/TUDataset/IMDB-MULTI/raw/IMDB-MULTI_A.txt`
- `Data1/TUDataset/IMDB-MULTI/raw/IMDB-MULTI_graph_indicator.txt`
- `Data1/TUDataset/IMDB-MULTI/raw/IMDB-MULTI_graph_labels.txt`
- `Data1/TUDataset/KKI/raw/KKI_A.txt`
- `Data1/TUDataset/KKI/raw/KKI_graph_indicator.txt`
- `Data1/TUDataset/KKI/raw/KKI_graph_labels.txt`
- `Data1/TUDataset/KKI/raw/KKI_node_labels.txt`
- `Data1/TUDataset/KKI/raw/readme.txt`
- `Data1/TUDataset/Letter-high/raw/Letter-high_A.txt`
- `Data1/TUDataset/Letter-high/raw/Letter-high_graph_indicator.txt`
- `Data1/TUDataset/Letter-high/raw/Letter-high_graph_labels.txt`
- `Data1/TUDataset/Letter-high/raw/Letter-high_label_readme.txt`
- `Data1/TUDataset/Letter-high/raw/Letter-high_node_attributes.txt`
- `Data1/TUDataset/Letter-low/raw/Letter-low_A.txt`
- `Data1/TUDataset/Letter-low/raw/Letter-low_graph_indicator.txt`
- `Data1/TUDataset/Letter-low/raw/Letter-low_graph_labels.txt`
- `Data1/TUDataset/Letter-low/raw/Letter-low_label_readme.txt`
- `Data1/TUDataset/Letter-low/raw/Letter-low_node_attributes.txt`
- `Data1/TUDataset/Letter-med/raw/Letter-med_A.txt`
- `Data1/TUDataset/Letter-med/raw/Letter-med_graph_indicator.txt`
- `Data1/TUDataset/Letter-med/raw/Letter-med_graph_labels.txt`
- `Data1/TUDataset/Letter-med/raw/Letter-med_label_readme.txt`
- `Data1/TUDataset/Letter-med/raw/Letter-med_node_attributes.txt`
- `Data1/TUDataset/MUTAG/raw/MUTAG_A.txt`
- `Data1/TUDataset/MUTAG/raw/MUTAG_edge_labels.txt`
- `Data1/TUDataset/MUTAG/raw/MUTAG_graph_indicator.txt`
- `Data1/TUDataset/MUTAG/raw/MUTAG_graph_labels.txt`
- `Data1/TUDataset/MUTAG/raw/MUTAG_node_labels.txt`
- `Data1/TUDataset/MUTAG/raw/README.txt`
- `Data1/TUDataset/NCI1/raw/NCI1_A.txt`
- `Data1/TUDataset/NCI1/raw/NCI1_graph_indicator.txt`
- `Data1/TUDataset/NCI1/raw/NCI1_graph_labels.txt`
- `Data1/TUDataset/NCI1/raw/NCI1_node_labels.txt`
- `Data1/TUDataset/NCI1/raw/README.txt`
- `Data1/TUDataset/OHSU/raw/OHSU_A.txt`
- `Data1/TUDataset/OHSU/raw/OHSU_graph_indicator.txt`
- `Data1/TUDataset/OHSU/raw/OHSU_graph_labels.txt`
- `Data1/TUDataset/OHSU/raw/OHSU_node_labels.txt`
- `Data1/TUDataset/OHSU/raw/readme.txt`
- `Data1/TUDataset/PROTEINS/raw/PROTEINS_A.txt`
- `Data1/TUDataset/PROTEINS/raw/PROTEINS_graph_indicator.txt`
- `Data1/TUDataset/PROTEINS/raw/PROTEINS_graph_labels.txt`
- `Data1/TUDataset/PROTEINS/raw/PROTEINS_node_attributes.txt`
- `Data1/TUDataset/PROTEINS/raw/PROTEINS_node_labels.txt`
- `Data1/TUDataset/PROTEINS/raw/README.txt`
- `Data1/TUDataset/PTC_MR/raw/PTC_MR_A.txt`
- `Data1/TUDataset/PTC_MR/raw/PTC_MR_edge_labels.txt`
- `Data1/TUDataset/PTC_MR/raw/PTC_MR_graph_indicator.txt`
- `Data1/TUDataset/PTC_MR/raw/PTC_MR_graph_labels.txt`
- `Data1/TUDataset/PTC_MR/raw/PTC_MR_node_labels.txt`
- `Data1/TUDataset/PTC_MR/raw/README.txt`
- `Data1/TUDataset/Peking_1/raw/Peking_1_A.txt`
- `Data1/TUDataset/Peking_1/raw/Peking_1_graph_indicator.txt`
- `Data1/TUDataset/Peking_1/raw/Peking_1_graph_labels.txt`
- `Data1/TUDataset/Peking_1/raw/Peking_1_node_labels.txt`
- `Data1/TUDataset/Peking_1/raw/readme.txt`

</details>

## Environment

The dependency pins target **Python 3.9** (the stack was specified with `numpy==1.24.3` and `scipy==1.10.1`, which are not published for Python 3.12). `python==3.9.21` is not a pip package and is not listed in `requirements.txt`.

`torch==2.4.0+cu124` is the CUDA 12.4 wheel. It is not on PyPI. `requirements.txt` sets `--extra-index-url https://download.pytorch.org/whl/cu124` so a normal `pip install -r` can resolve it. The default `--device` is `cuda:0`.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`tensorly-torch` provides `tltorch` (`TCL`, `TRL`). `gudhi` is imported by `utils.py` at import time, including for `python main.py --help`.

TensorBoard event files are written with `tensorboardX`. A viewer is not pinned; install `tensorboard` separately if you want the UI.

## Data

Raw TU files are already under `Data1/TUDataset/<name>/raw/` (`--datapath` defaults to `./Data1`). The loader asks for `*_A.txt` and `*_graph_indicator.txt`. Graph labels, and node or edge labels and attributes when present, ship in the same `raw/` directories. The first run writes PyG caches to `Data1/TUDataset/<name>/processed/` and PI tensors under `--save_dir` (default `./processed_graphs/`).

`prepareData_multiDS` keeps a 90/10 train/test split (`test_size=0.1`, `random_state=args.seed`, not stratified). It does not build a validation loader. The `seed=` argument of `prepareData_multiDS` is unused; `main.py` passes `seed_dataSplit = 123`, and the split follows `--seed` (default 789).

Feature construction in `prepareData_multiDS`:

| Datasets | Node features |
| --- | --- |
| `IMDB-BINARY` | `OneHotDegree(135)` |
| `IMDB-MULTI` | `OneHotDegree(88)` |
| `Letter-high`, `Letter-med`, `Letter-low` | `use_node_attr=True` |
| Every other name in the groups below | Default `read_tu_data` features |

Named groups (`--data_group`, default `sm`):

| `data_group` | Datasets, in load order |
| --- | --- |
| `sm` | MUTAG, BZR, COX2, DHFR, PTC_MR, AIDS, NCI1 |
| `bio` | KKI, OHSU, Peking_1, PROTEINS |
| `sn` | IMDB-MULTI, IMDB-BINARY |
| `cv` | Letter-high, Letter-med, Letter-low |
| `sm-bio` | `sm` then `bio` |
| `sm-bio-cv` | `sm`, `bio`, then `cv` |
| `sm-bio-cv-sn` | `sm`, `bio`, `cv`, then `sn` |

Any other string is treated as a single dataset name (`datasets = [group]`). That path only finishes setup for names listed in `get_dataset_specific_params` (the 16 datasets in the table below).

`Data1/TUDataset` also contains **DD** and **ENZYMES**. They are not members of any named group, and they have no row in `get_dataset_specific_params`, so `--data_group DD` or `--data_group ENZYMES` raises `ValueError` when clients are built. `get_optimal_augmentations` does contain entries for `DD`, `PTC_FM`, and `REDDIT-BINARY`; those two extra names are not in this checkout.

Per-dataset depth and width passed into `TASSL` as `(num_layers, hidden_dim)`, and the augmentation pair `(aug1, aug2)`:

| Dataset | `num_layers` | `hidden_dim` | aug1 | aug2 |
| --- | --- | --- | --- | --- |
| AIDS | 4 | 64 | dnodes | pedges |
| BZR | 4 | 16 | dnodes | mask_nodes |
| COX2 | 3 | 64 | dnodes | subgraph |
| DHFR | 5 | 64 | pedges | none |
| IMDB-BINARY | 5 | 64 | subgraph | none |
| IMDB-MULTI | 3 | 16 | subgraph | none |
| KKI | 5 | 16 | dnodes | none |
| Letter-high | 5 | 64 | pedges | none |
| Letter-low | 3 | 64 | pedges | none |
| Letter-med | 4 | 64 | pedges | none |
| MUTAG | 4 | 32 | dnodes | pedges |
| NCI1 | 5 | 64 | dnodes | mask_nodes |
| OHSU | 4 | 16 | dnodes | none |
| PROTEINS | 3 | 32 | dnodes | pedges |
| PTC_MR | 3 | 32 | dnodes | pedges |
| Peking_1 | 3 | 32 | dnodes | none |

If either augmentation is `dnodes` or `subgraph`, the client is constructed with `aug='dnodes'` so `match_nodes` runs. Otherwise `aug` is `None`. `pedges` is `permute_edges`. When `aug2` is `none`, the second view and its PI tensor fall back to the original graph and the original PI.

## Usage

From the repository root, with the environment above and raw data at `./Data1/TUDataset` (already committed):

```bash
# small molecules (default --data_group sm)
python main.py

python main.py --data_group sm
python main.py --data_group bio
python main.py --data_group sn
python main.py --data_group cv
python main.py --data_group sm-bio
python main.py --data_group sm-bio-cv
python main.py --data_group sm-bio-cv-sn

# one dataset that has a row in get_dataset_specific_params
python main.py --data_group MUTAG
```

The default run is 200 rounds on `cuda:0` and computes persistence images for every graph in the group before training. `sm` includes NCI1.

With the pinned `torch_geometric==2.5.2`, `read_tu_data` returns `(data, slices, sizes)`. In `aug.py`, the `MUTAG` branch unpacks two values and every other dataset unpacks three. `python main.py` loads MUTAG first, raises `ValueError: too many values to unpack (expected 2)` inside `TUDataset_aug.process`, and does not start a training round. The Python files are unchanged.

### Arguments

Defaults are the `argparse` defaults in `main.py`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--device` | `cuda:0` | Device string. PI tensors are moved here during preprocessing. Joint training and fine-tuning move the client module to this device. |
| `--alg` | `joint` | Written into log file names and TensorBoard tags. `main.py` always calls `run_DGTF_joint`. |
| `--alpha` | `4` | Coefficient on the distillation loss: contrastive loss + `alpha` × distillation loss. |
| `--lambda1` | `0.3` | Coefficient on the PI contrastive term inside `TASSL` (`gcn_tensor_loss + lambda1 * pi_tensor_loss`). The help string says "lambda for TRM loss". |
| `--num_rounds` | `200` | Communication rounds. |
| `--local_epoch` | `1` | Local joint-training epochs per round. The 40-epoch per-round fine-tune and the 100-epoch final fine-tune are constants in `training.py`. |
| `--lr` | `0.0001` | Adam learning rate for joint training. `Client_GC` builds that Adam itself (`weight_decay` left at the PyTorch default, 0). |
| `--weight_decay` | `5e-4` | Passed to an Adam constructed in `setup_devices_TTG`. `Client_GC.__init__` replaces `self.optimizer` and does not use this value. |
| `--finetune_lr` | `0.01` | Adam learning rate for the fine-tune in `linear_probe`. Also part of the default CSV file name. |
| `--batch_size` | `64` | Batch size for joint training and fine-tuning. The contrastive loop slices each batch into micro-batches of 8. |
| `--seed` | `789` | Seed for `random`, NumPy, and PyTorch, and `random_state` of the 90/10 split. |
| `--datapath` | `./Data1` | Directory that contains `TUDataset/`. |
| `--outbase` | `./outputs` | Root for the CSV and TensorBoard directories. |
| `--repeat` | `None` | If omitted, the CSV is `acc-seed<seed>_ftlr<finetune_lr>.csv`. If set, the CSV is `<repeat>_accuracy_DGTF_GC.csv`. The value is also embedded in the TensorBoard run directory name. |
| `--data_group` | `sm` | Group name, or a single dataset name. See Data. |
| `--sublevel_filtration_methods` | `degree betweenness closeness eigenvector` | Parsed only. PI computation and `TASSL` use the hardcoded list `degree`, `betweenness`, `eigenvector`, `closeness`. |
| `--PI_dim` | `50` | Height and width of each persistence image. Also part of the PI cache file name. |
| `--num_layers` | `3` | Parsed only. Help text says the default is 5. Depth comes from `get_dataset_specific_params`. |
| `--num_mlp_layers` | `2` | `MLP` depth excluding the input layer (1 means a single linear layer). This flag is passed into `TASSL`. |
| `--hidden_dim` | `16` | Parsed only. Help text says the default is 64. Width comes from `get_dataset_specific_params`. |
| `--final_dropout` | `0.5` | Dropout inside `MLP_output`. |
| `--tensor_layer_type` | `TCL` | `TCL` or `TRL` (choices are enforced). |
| `--node_pooling` | `True` | `action="store_false"`, so passing the flag sets it to `False`. `TASSL.__init__` accepts the value and never reads it. |
| `--save_dir` | `./` | Directory for `processed_graphs/` and the round-results text log. |

### Outputs

CSV (one row per client plus a `mean` row, columns `test_acc` and `test_f1_macro`):

- `--repeat` omitted: `outputs/raw/<data_group>/acc-seed<seed>_ftlr<finetune_lr>.csv`
- `--repeat` set: `outputs/raw/<data_group>/<repeat>_accuracy_DGTF_GC.csv`

The CSV is appended (`mode='a'`). A header is written when the file does not already exist.

TensorBoard run directory:

```text
outputs/raw/tensorboard/<data_group>_<alg>__<repeat>_<YYYYmmdd_HHMMSS>/
```

Scalars: `Round_Finetune/Mean_Acc_<alg>`, `Round_Finetune/Mean_F1_Macro_<alg>`, and `Test/Loss/Mean_<alg>` (the mean joint-training loss of that round).

```bash
tensorboard --logdir outputs/raw/tensorboard
```

Round log (overwritten each run, default path is the repository root because `--save_dir` defaults to `./`):

```text
<save_dir>/<alg>_<data_group>_local-epoch_<local_epoch>-a<alpha>_la<lambda1>_round_results_seed<seed>.txt
```

`processed_graphs/` holds `{name}_{PI_dim}_PI.pt` and `{name}_{PI_dim}_{aug}_PI.pt`. The loader also looks for `{name}_processed.pt`, `{name}_aug1_processed.pt`, and `{name}_aug2_processed.pt`, but the branch that builds graphs does not `torch.save` those three files.

## Results

<!-- TODO: fill this table from the paper. Do not treat the cells below as measurements. -->

| Setting | Mean accuracy | Mean macro-F1 |
| --- | --- | --- |
| `sm` | TODO | TODO |
| `bio` | TODO | TODO |
| `sn` | TODO | TODO |
| `cv` | TODO | TODO |
| `sm-bio` | TODO | TODO |
| `sm-bio-cv` | TODO | TODO |
| `sm-bio-cv-sn` | TODO | TODO |

## Citation

<!-- TODO: replace author and pages, and add the paper URL, once the camera-ready bibliographic record is known. -->

```bibtex
@inproceedings{dgtf2026,
  title     = {DGTF: Cross-Domain Decentralized Graph Learning with Topology-Aware Knowledge Fusion},
  author    = {TODO},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  pages     = {TODO},
  url       = {TODO}
}
```

## License

This repository is released under the [MIT License](LICENSE). The copyright line is `Copyright (c) 2026 TODO: copyright holder (paper authors)`. Confirm the license and replace the copyright holder before relying on it.

## Acknowledgements

Our implementation builds upon FedStar and FedSSP. We thank the authors for releasing their code.

- FedStar: [Federated Learning on Non-IID Graphs via Structural Knowledge Sharing](https://ojs.aaai.org/index.php/AAAI/article/view/26187), AAAI 2023. [arXiv:2211.13009](https://arxiv.org/abs/2211.13009). Code: [yuetan031/FedStar](https://github.com/yuetan031/FedStar).
- FedSSP: [FedSSP: Federated Graph Learning with Spectral Knowledge and Personalized Preference](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3d226fb8fbd6ee6ec70d0427f1319707-Abstract-Conference.html), NeurIPS 2024. [arXiv:2410.20105](https://arxiv.org/abs/2410.20105). Code: [OakleyTan/FedSSP](https://github.com/OakleyTan/FedSSP).
