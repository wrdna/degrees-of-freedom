#!/usr/bin/env python3
"""
Reproduction Script for Conv-3 Results from:
"How Many Degrees of Freedom Do We Need to Train Deep Networks: A Loss Landscape Perspective"

This script reproduces all main figures for Conv-3 (SmallCNN) on MNIST and CIFAR-10.

Figures Covered:
- Fig 2: Phase transition in random subspaces
- Fig 3: Effect of burn-in iterations  
- Fig 4: Threshold training dimension vs desired accuracy
- Fig 5: Comparison of all methods (random, burn-in, lottery subspace, lottery ticket)
- Fig 8: Burn-in with varying sparsity
- Fig 9-10: Extended results

Usage:
    python reproduce_conv3.py --experiment all
    python reproduce_conv3.py --experiment random_subspace --dataset MNIST
    python reproduce_conv3.py --experiment burn_in --dataset cifar10
"""

import subprocess
import argparse
import os

# Conv-3 = SmallCNN in the codebase (channels [32, 64, 64])
MODEL = "SmallCNN"

# Dimensions to explore for phase transition plots
# These are powers of 2 from 1 to 8192, plus some intermediate values
DIMS_FINE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DIMS_COARSE = [2**x for x in range(13)]  # 1 to 4096

# Burn-in iterations to test
BURN_IN_ITERS = [0, 4, 16, 64, 256, 1024]

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*80)
    subprocess.run(cmd)


def run_random_subspace(dataset, epochs=10, points=5):
    """
    Experiment 1: Random Affine Subspace Training (Fig 2, 4)
    
    This reproduces the phase transition curve showing how success probability
    goes from 0 to 1 as training dimension increases past threshold.
    """
    print(f"\n[Random Subspace] Dataset: {dataset}")
    
    cmd = [
        "python", "burn_in_subspace.py",
        "--model", MODEL,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--points_to_collect", str(points),
        "--init_iters", "0",  # 0 = random subspace (no burn-in)
        "--ds_to_explore", *[str(d) for d in DIMS_FINE],
        "--lr", "0.05",
        "--opt_alg", "Adam",
    ]
    
    run_command(cmd, f"Random Subspace - {dataset}")


def run_burn_in_subspace(dataset, epochs=10, points=3):
    """
    Experiment 2: Burn-in Subspace Training (Fig 3, 4, 8)
    
    Tests how a few initial training steps before projecting to subspace
    affects the threshold training dimension.
    """
    print(f"\n[Burn-in Subspace] Dataset: {dataset}")
    
    for init_iters in BURN_IN_ITERS:
        print(f"\n  Testing burn-in iterations: {init_iters}")
        
        cmd = [
            "python", "burn_in_subspace.py",
            "--model", MODEL,
            "--dataset", dataset,
            "--epochs", str(epochs),
            "--points_to_collect", str(points),
            "--init_iters", str(init_iters),
            "--ds_to_explore", *[str(d) for d in DIMS_COARSE],
            "--lr", "0.05",
            "--opt_alg", "Adam",
        ]
        
        run_command(cmd, f"Burn-in Subspace - {dataset} - init_iters={init_iters}")


def run_lottery_subspace(dataset, epochs=10, points=3):
    """
    Experiment 3: Lottery Subspace (Fig 5)
    
    Uses gradients from the training trajectory to form an optimal subspace
    for training, rather than random directions.
    """
    print(f"\n[Lottery Subspace] Dataset: {dataset}")
    
    cmd = [
        "python", "lottery_subspace.py",
        "--model", MODEL,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--points_to_collect", str(points),
        "--ds_to_explore", *[str(d) for d in DIMS_COARSE],
        "--traj_steps", "5",
        "--lr", "0.05",
        "--opt_alg", "Adam",
    ]
    
    run_command(cmd, f"Lottery Subspace - {dataset}")


def run_lottery_ticket(dataset, epochs=10, points=3):
    """
    Experiment 4: Lottery Ticket (Fig 5, 10)
    
    Traditional magnitude-based pruning after training to find
    sparse trainable subnetworks.
    """
    print(f"\n[Lottery Ticket] Dataset: {dataset}")
    
    cmd = [
        "python", "lottery_ticket.py",
        "--model", MODEL,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--points_to_collect", str(points),
    ]
    
    run_command(cmd, f"Lottery Ticket - {dataset}")


def run_all_experiments(datasets):
    """Run all experiments for specified datasets."""
    for dataset in datasets:
        print(f"\n{'#'*80}")
        print(f"# Running all experiments for {dataset}")
        print('#'*80)
        
        # Run each experiment type
        run_random_subspace(dataset)
        run_burn_in_subspace(dataset)
        run_lottery_subspace(dataset)
        run_lottery_ticket(dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Conv-3 results from the Degrees of Freedom paper"
    )
    parser.add_argument(
        "--experiment", 
        choices=["all", "random_subspace", "burn_in", "lottery_subspace", "lottery_ticket"],
        default="all",
        help="Which experiment to run"
    )
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "cifar10", "both"],
        default="both",
        help="Dataset to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per run"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=5,
        help="Number of random seeds/repetitions per dimension"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer epochs/points"
    )
    
    args = parser.parse_args()
    
    # Set datasets
    if args.dataset == "both":
        datasets = ["MNIST", "cifar10"]
    else:
        datasets = [args.dataset]
    
    # Quick mode for testing
    if args.quick:
        args.epochs = 3
        args.points = 2
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("="*80)
    print("Reproducing Conv-3 Results")
    print(f"Model: {MODEL} (Conv-3 = 3 conv layers with channels [32, 64, 64])")
    print(f"Datasets: {datasets}")
    print(f"Epochs: {args.epochs}, Points per dimension: {args.points}")
    print("="*80)
    
    # Run selected experiments
    if args.experiment == "all":
        run_all_experiments(datasets)
    elif args.experiment == "random_subspace":
        for ds in datasets:
            run_random_subspace(ds, args.epochs, args.points)
    elif args.experiment == "burn_in":
        for ds in datasets:
            run_burn_in_subspace(ds, args.epochs, args.points)
    elif args.experiment == "lottery_subspace":
        for ds in datasets:
            run_lottery_subspace(ds, args.epochs, args.points)
    elif args.experiment == "lottery_ticket":
        for ds in datasets:
            run_lottery_ticket(ds, args.epochs, args.points)
    
    print("\n" + "="*80)
    print("All experiments complete!")
    print(f"Results saved to: ../lottery-subspace-data/")
    print("="*80)


if __name__ == "__main__":
    main()

