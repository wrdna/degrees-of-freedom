# How many degrees of freedom do we need to train deep networks?

This repository contains source code for the ICLR 2022 paper [***How many degrees of freedom do we need to train deep networks: a loss landscape perspective***](https://openreview.net/forum?id=ChMLTGRjFcU) by Brett W. Larsen, Sanislav Fort, Nic Becker, and Surya Ganguli ([*arXiv version*](https://arxiv.org/abs/2107.05802)). 

This code was developed and tested using `JAX v0.1.74`, `JAXlib v0.1.52`, and `Flax v0.2.0`. The authors intend to update the repository in the future with additional versions of the script that work with the `flax.linen` module.

---

## Extended Experiments

This fork includes additional experiments exploring lottery subspace properties:

### Cross-Initialization Transfer Experiment

**Key Question:** Are lottery subspaces architecture-intrinsic or initialization-specific?

This experiment tests whether the low-dimensional training subspace discovered by one network initialization can be used to train a network with a *different* initialization. See [`EXPERIMENT_CROSS_INIT_TRANSFER.md`](EXPERIMENT_CROSS_INIT_TRANSFER.md) for the full experimental design and theoretical motivation.

```bash
# Run the full transfer experiment
./run_transfer_experiment.sh

# Quick test (fewer dimensions)
./run_transfer_experiment.sh --quick
```

### Conv-2 (TinyCNN) Reproduction

Faster reproduction experiments using a smaller architecture:

```bash
./run_conv2_experiments.sh
```

---

## Scripts

### Core Experiment Scripts

| Script | Description |
|--------|-------------|
| `burn_in_subspace.py` | Random affine subspace and burn-in affine subspace experiments. Set `init_iters=0` for random subspaces. |
| `lottery_subspace.py` | Lottery subspace experiments - trains networks and computes SVD of trajectory |
| `lottery_ticket.py` | Lottery ticket (sparse pruning) experiments |
| `lottery_subspace_transfer.py` | Cross-initialization transfer experiment - tests if lottery subspaces transfer across initializations |

### Automation Scripts

| Script | Description |
|--------|-------------|
| `run_transfer_experiment.sh` | Full pipeline: generate donor subspace → transfer experiment → baseline → plots |
| `run_conv2_experiments.sh` | Conv-2/TinyCNN reproduction: random subspace, burn-in, lottery subspace, lottery ticket |
| `reproduce_conv3.py` | Conv-3/SmallCNN reproduction script for all main paper figures |

### Plotting Scripts

| Script | Description |
|--------|-------------|
| `plot_results.py` | Generate figures for random/burn-in/lottery experiments (Fig 2-5) |
| `plot_transfer_experiment.py` | Generate figures comparing transfer vs baseline performance |

### Support Modules

| Module | Description |
|--------|-------------|
| `architectures.py` | Model definitions (TinyCNN, SmallCNN, ResNet, WideResNet) |
| `data_utils.py` | Functions for saving/loading data |
| `generate_data.py` | Dataset setup (MNIST, FashionMNIST, CIFAR-10/100, SVHN) |
| `logging_tools.py` | Logger setup with automatic timestamped experiment names |
| `training_utils.py` | Subspace projection and constrained training utilities |

---

## Data Directories

| Directory | Contents |
|-----------|----------|
| `lottery-subspace-data/` | Results from standard experiments (burn-in, lottery subspace, lottery ticket) |
| `cross-init-lottery-subspace-data/` | Results from cross-initialization transfer experiments |
| `figures/` | Generated plots |

---

## Quick Start

```bash
# 1. Setup environment
conda create -n dof python=3.8
conda activate dof
pip install jax==0.1.74 jaxlib==0.1.52 flax==0.2.0 tensorflow matplotlib scipy

# 2. Run Conv-2 experiments (fastest, ~1 hour)
./run_conv2_experiments.sh

# 3. View results
python plot_results.py --data_dir lottery-subspace-data/ --show

# 4. Run transfer experiment (optional, ~2 hours)
./run_transfer_experiment.sh
python plot_transfer_experiment.py --data_dir cross-init-lottery-subspace-data/ --show
```

---

## Documentation

- [`EXPERIMENT_CROSS_INIT_TRANSFER.md`](EXPERIMENT_CROSS_INIT_TRANSFER.md) - Detailed documentation of the cross-initialization transfer experiment
- [`freedomResearchPaper.pdf`](freedomResearchPaper.pdf) - Original paper

---

## Citation

```
@inproceedings{LaFoBeGa22,
	title={How many degrees of freedom do we need to train deep networks: a loss landscape perspective},
	author={Brett W. Larsen and Stanislav Fort and Nic Becker and Surya Ganguli},
	booktitle={International Conference on Learning Representations},
	year={2022},
	url={https://openreview.net/forum?id=ChMLTGRjFcU}
}
```
