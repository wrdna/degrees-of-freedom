#!/usr/bin/env python3
"""
Plotting Script for Conv-2 Reproduction Results

Generates plots matching the figures from:
"How Many Degrees of Freedom Do We Need to Train Deep Networks"

Usage:
    python plot_results.py --data_dir ../lottery-subspace-data/
"""

import os
import glob
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
# plt.style.use('seaborn-v0_8-whitegrid')

# Consistent color scheme across all plots
COLORS = {
    'init0': '#1f77b4',      # Blue - Random subspace
    'init4': '#ff7f0e',      # Orange - Burn-in 4
    'init16': '#2ca02c',     # Green - Burn-in 16
    'init64': '#d62728',     # Red - Burn-in 64
    'lottery_subspace': '#9467bd',  # Purple - Lottery subspace
    'lottery_ticket_test': '#8c564b',  # Brown - Lottery ticket test
    'lottery_ticket_train': '#e377c2', # Pink - Lottery ticket train
}


def load_pickle(filepath):
    """Load a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def find_result_files(data_dir, pattern):
    """Find result files matching a pattern."""
    files = glob.glob(os.path.join(data_dir, f"*{pattern}*_results.pkl"))
    return sorted(files)


def extract_phase_transition_data(result):
    """Extract data for phase transition plot from a results dict."""
    data = result['data']
    ds = np.array(data['d'])
    test_accs = np.array([float(x) for x in data['test_acc']])
    train_accs = np.array([float(x) for x in data['full_train_acc']])
    
    # Group by dimension
    unique_ds = np.unique(ds)
    mean_test_acc = []
    std_test_acc = []
    mean_train_acc = []
    
    for d in unique_ds:
        mask = ds == d
        mean_test_acc.append(np.mean(test_accs[mask]))
        std_test_acc.append(np.std(test_accs[mask]))
        mean_train_acc.append(np.mean(train_accs[mask]))
    
    return unique_ds, np.array(mean_test_acc), np.array(std_test_acc), np.array(mean_train_acc)


def extract_raw_accuracies(result):
    """Extract raw accuracy data grouped by dimension."""
    data = result['data']
    ds = np.array(data['d'])
    test_accs = np.array([float(x) for x in data['test_acc']])
    
    unique_ds = np.unique(ds)
    acc_by_dim = {}
    for d in unique_ds:
        mask = ds == d
        acc_by_dim[d] = test_accs[mask]
    
    return acc_by_dim


def compute_success_probability(acc_by_dim, accuracy_thresholds):
    """Compute probability of achieving each accuracy threshold for each dimension."""
    dims = sorted(acc_by_dim.keys())
    prob_matrix = np.zeros((len(accuracy_thresholds), len(dims)))
    
    for j, d in enumerate(dims):
        accs = acc_by_dim[d]
        for i, thresh in enumerate(accuracy_thresholds):
            prob_matrix[i, j] = np.mean(accs >= thresh)
    
    return prob_matrix, dims


def plot_phase_transition_heatmap(data_dir, dataset="MNIST", save_dir=None):
    """
    Plot Figure 2 style: Heatmap of success probability vs training dimension.
    X-axis: Training dimension
    Y-axis: Accuracy threshold  
    Color: Probability of achieving that accuracy (black=0, white=1)
    """
    # Methods to plot
    methods = [
        (f"burn_in_subspace_TinyCNN_{dataset}_init0", "Random\nSubspace"),
        (f"burn_in_subspace_TinyCNN_{dataset}_init4", "Burn-in\n4 steps"),
        (f"burn_in_subspace_TinyCNN_{dataset}_init16", "Burn-in\n16 steps"),
        (f"burn_in_subspace_TinyCNN_{dataset}_init64", "Burn-in\n64 steps"),
    ]
    
    # Accuracy thresholds for y-axis
    accuracy_thresholds = np.linspace(0.1, 1.0, 50)
    
    # Find how many methods have data
    valid_methods = []
    for pattern, label in methods:
        files = find_result_files(data_dir, pattern)
        if files:
            valid_methods.append((pattern, label, files[0]))
    
    if not valid_methods:
        print("No data found for heatmap")
        return
    
    n_methods = len(valid_methods)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(3 * (n_methods + 1), 5), 
                              gridspec_kw={'width_ratios': [1]*n_methods + [1.2]})
    
    threshold_curves = []
    
    for idx, (pattern, label, filepath) in enumerate(valid_methods):
        ax = axes[idx]
        result = load_pickle(filepath)
        acc_by_dim = extract_raw_accuracies(result)
        prob_matrix, dims = compute_success_probability(acc_by_dim, accuracy_thresholds)
        
        # Plot heatmap
        extent = [0, len(dims), accuracy_thresholds[0], accuracy_thresholds[-1]]
        im = ax.imshow(prob_matrix, aspect='auto', origin='lower', 
                       extent=extent, cmap='gray', vmin=0, vmax=1)
        
        # Overlay the mean accuracy curve
        ds_arr, mean_acc, _, _ = extract_phase_transition_data(result)
        # Map dimensions to x positions
        x_positions = [list(dims).index(d) + 0.5 for d in ds_arr if d in dims]
        mean_acc_filtered = [mean_acc[i] for i, d in enumerate(ds_arr) if d in dims]
        
        colors = [COLORS['init0'], COLORS['init4'], COLORS['init16'], COLORS['init64']]
        ax.plot(x_positions, mean_acc_filtered, color=colors[idx], linewidth=2)
        threshold_curves.append((x_positions, mean_acc_filtered, colors[idx], label.replace('\n', ' ')))
        
        # Labels
        ax.set_xticks(np.arange(len(dims)) + 0.5)
        ax.set_xticklabels([str(d) for d in dims], rotation=45, fontsize=8)
        ax.set_xlabel('Training Dimension', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(label, fontsize=11)
        ax.axhline(y=0.9, color='white', linestyle='--', alpha=0.7, linewidth=0.5)
    
    # Right panel: threshold curves overlay
    ax_right = axes[-1]
    for x_pos, mean_acc, color, label in threshold_curves:
        ax_right.plot(x_pos, mean_acc, color=color, linewidth=2, marker='o', 
                     markersize=3, label=label)
    
    ax_right.set_xticks(np.arange(len(dims)) + 0.5)
    ax_right.set_xticklabels([str(d) for d in dims], rotation=45, fontsize=8)
    ax_right.set_xlabel('Training Dimension', fontsize=10)
    ax_right.set_ylabel('Accuracy', fontsize=12)
    ax_right.set_title('Threshold\nComparison', fontsize=11)
    ax_right.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax_right.legend(fontsize=8, loc='lower right')
    ax_right.set_ylim(0, 1.05)
    ax_right.grid(True, alpha=0.3)
    
    plt.suptitle(f'Phase Transitions in Achievable Accuracy vs Training Dimension\n{dataset}', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'fig2_phase_heatmap_{dataset}.png'), 
                   dpi=150, bbox_inches='tight')
    plt.show()


def plot_phase_transition(data_dir, dataset="MNIST", save_dir=None):
    """
    Plot Figure 2: Phase transition in success probability vs training dimension.
    Shows the sharp transition from 0 to 1 as dimension increases.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Find random subspace files (init0 = no burn-in)
    pattern = f"burn_in_subspace_TinyCNN_{dataset}_init0"
    files = find_result_files(data_dir, pattern)
    
    if not files:
        print(f"No files found for pattern: {pattern}")
        return
    
    for f in files:
        result = load_pickle(f)
        ds, mean_acc, std_acc, _ = extract_phase_transition_data(result)
        
        ax.errorbar(ds, mean_acc, yerr=std_acc, marker='o', capsize=3,
                   label=os.path.basename(f))
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels([str(int(d)) for d in ds])
    ax.set_xlabel('Training Dimension (d)', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title(f'Phase Transition: Random Subspace Training\nConv-2 on {dataset}', fontsize=16)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% accuracy threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'fig2_phase_transition_{dataset}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_burn_in_effect(data_dir, dataset="MNIST", save_dir=None):
    """
    Plot Figure 3: Effect of burn-in iterations on threshold training dimension.
    Order: Random -> Burn-in 4 -> Burn-in 16 -> Burn-in 64
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Consistent color scheme and order
    methods = [
        (0, "Random Subspace", '#1f77b4', 'o'),
        (4, "Burn-in (4 steps)", '#ff7f0e', 's'),
        (16, "Burn-in (16 steps)", '#2ca02c', '^'),
        (64, "Burn-in (64 steps)", '#d62728', 'p'),
    ]
    
    all_dims = set()
    
    for init_iters, label, color, marker in methods:
        pattern = f"burn_in_subspace_TinyCNN_{dataset}_init{init_iters}"
        files = find_result_files(data_dir, pattern)
        
        if files:
            result = load_pickle(files[0])
            ds, mean_acc, std_acc, _ = extract_phase_transition_data(result)
            all_dims.update(ds)
            ax.plot(ds, mean_acc, marker=marker, color=color, label=label, linewidth=2)
    
    all_dims_sorted = sorted(all_dims)
    ax.set_xscale('log', base=2)
    if all_dims_sorted:
        ax.set_xticks(all_dims_sorted)
        ax.set_xticklabels([str(int(d)) for d in all_dims_sorted], rotation=45)
    ax.set_xlabel('Training Dimension (d)', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title(f'Effect of Burn-in Iterations\nConv-2 on {dataset}', fontsize=16)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'fig3_burn_in_effect_{dataset}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_method_comparison(data_dir, dataset="MNIST", save_dir=None):
    """
    Plot Figure 5: Comparison of all methods.
    Order: Random -> Burn-ins -> Lottery Subspace (left), Lottery Ticket (right)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Consistent color scheme
    colors = {
        'random': '#1f77b4',       # Blue
        'burn_in_4': '#ff7f0e',    # Orange
        'burn_in_16': '#2ca02c',   # Green
        'burn_in_64': '#d62728',   # Red
        'lottery': '#9467bd',      # Purple
    }
    
    # Left plot: Subspace methods in order
    methods = [
        ("burn_in_subspace_TinyCNN_{}_init0".format(dataset), "Random Subspace", colors['random'], 'o'),
        ("burn_in_subspace_TinyCNN_{}_init4".format(dataset), "Burn-in (4 steps)", colors['burn_in_4'], 's'),
        ("burn_in_subspace_TinyCNN_{}_init16".format(dataset), "Burn-in (16 steps)", colors['burn_in_16'], '^'),
        ("burn_in_subspace_TinyCNN_{}_init64".format(dataset), "Burn-in (64 steps)", colors['burn_in_64'], 'p'),
        ("lottery_subspace_TinyCNN_{}".format(dataset), "Lottery Subspace", colors['lottery'], 'D'),
    ]
    
    all_dims = set()
    for pattern, label, color, marker in methods:
        files = find_result_files(data_dir, pattern.replace(f"_{dataset}", f"_{dataset}"))
        if files:
            result = load_pickle(files[0])
            ds, mean_acc, std_acc, _ = extract_phase_transition_data(result)
            all_dims.update(ds)
            ax1.plot(ds, mean_acc, marker=marker, color=color, label=label, linewidth=2)
    
    all_dims_sorted = sorted(all_dims)
    ax1.set_xscale('log', base=2)
    if all_dims_sorted:
        ax1.set_xticks(all_dims_sorted)
        ax1.set_xticklabels([str(int(d)) for d in all_dims_sorted], rotation=45)
    ax1.set_xlabel('Training Dimension (d)', fontsize=14)
    ax1.set_ylabel('Test Accuracy', fontsize=14)
    ax1.set_title(f'Subspace Methods Comparison\nConv-2 on {dataset}', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Right plot: Lottery ticket (fraction on x-axis)
    pattern = f"lottery_ticket_TinyCNN_{dataset}"
    files = find_result_files(data_dir, pattern)
    
    if files:
        result = load_pickle(files[0])
        fracs = result['fracs_on_np'].mean(axis=0)
        test_accs = result['test_accs_np'].mean(axis=0)
        train_accs = result['train_accs_np'].mean(axis=0)
        
        ax2.plot(fracs, test_accs, 'o-', color=COLORS['lottery_ticket_test'], label='Test Acc')
        ax2.plot(fracs, train_accs, 's--', color=COLORS['lottery_ticket_train'], label='Train Acc', alpha=0.7)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Fraction of Weights Kept', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.set_title(f'Lottery Ticket Pruning\nConv-2 on {dataset}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'fig5_method_comparison_{dataset}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_dimension(data_dir, accuracy_threshold=0.9, save_dir=None):
    """
    Plot Figure 4: Threshold training dimension as function of method.
    Find the minimum dimension needed to reach a target accuracy.
    Order: Random -> Burn-ins -> Lottery Subspace
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = ['MNIST']
    
    # Define methods in order with their colors
    methods_order = [
        ('Random', '#1f77b4'),
        ('Burn-in 4', '#ff7f0e'),
        ('Burn-in 16', '#2ca02c'),
        ('Burn-in 64', '#d62728'),
        ('Lottery Subspace', '#9467bd'),
    ]
    
    method_results = {}
    
    for dataset in datasets:
        method_results[dataset] = {}
        
        # Random subspace
        pattern = f"burn_in_subspace_TinyCNN_{dataset}_init0"
        files = find_result_files(data_dir, pattern)
        if files:
            result = load_pickle(files[0])
            ds, mean_acc, _, _ = extract_phase_transition_data(result)
            above_thresh = ds[mean_acc >= accuracy_threshold]
            if len(above_thresh) > 0:
                method_results[dataset]['Random'] = above_thresh[0]
        
        # Burn-in subspace
        for init_iters in [4, 16, 64]:
            pattern = f"burn_in_subspace_TinyCNN_{dataset}_init{init_iters}"
            files = find_result_files(data_dir, pattern)
            if files:
                result = load_pickle(files[0])
                ds, mean_acc, _, _ = extract_phase_transition_data(result)
                above_thresh = ds[mean_acc >= accuracy_threshold]
                if len(above_thresh) > 0:
                    method_results[dataset][f'Burn-in {init_iters}'] = above_thresh[0]
        
        # Lottery subspace
        pattern = f"lottery_subspace_TinyCNN_{dataset}"
        files = find_result_files(data_dir, pattern)
        if files:
            result = load_pickle(files[0])
            ds, mean_acc, _, _ = extract_phase_transition_data(result)
            above_thresh = ds[mean_acc >= accuracy_threshold]
            if len(above_thresh) > 0:
                method_results[dataset]['Lottery Subspace'] = above_thresh[0]
    
    # Filter to only methods with data, but keep order
    available_methods = []
    available_colors = []
    for method_name, color in methods_order:
        if any(method_name in method_results[d] for d in datasets):
            available_methods.append(method_name)
            available_colors.append(color)
    
    x = np.arange(len(available_methods))
    width = 0.6
    
    # Get values for each method in order
    for dataset in datasets:
        values = [method_results[dataset].get(m, np.nan) for m in available_methods]
        ax.bar(x, values, width, color=available_colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Threshold Training Dimension', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_title(f'Threshold Dimension to Reach {accuracy_threshold*100:.0f}% Test Accuracy\nConv-2', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(available_methods, rotation=45, ha='right')
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'fig4_threshold_dimension.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_all(data_dir, save_dir=None):
    """Generate all plots."""
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("Generating Phase Transition Heatmap plots (Fig 2 style)...")
    try:
        plot_phase_transition_heatmap(data_dir, 'MNIST', save_dir)
    except Exception as e:
        print(f"  Skipping heatmap: {e}")
    
    print("Generating Phase Transition plots (Fig 2)...")
    try:
        plot_phase_transition(data_dir, 'MNIST', save_dir)
    except Exception as e:
        print(f"  Skipping: {e}")
    
    print("\nGenerating Burn-in Effect plots (Fig 3)...")
    try:
        plot_burn_in_effect(data_dir, 'MNIST', save_dir)
    except Exception as e:
        print(f"  Skipping: {e}")
    
    print("\nGenerating Method Comparison plots (Fig 5)...")
    try:
        plot_method_comparison(data_dir, 'MNIST', save_dir)
    except Exception as e:
        print(f"  Skipping: {e}")
    
    print("\nGenerating Threshold Dimension plot (Fig 4)...")
    try:
        plot_threshold_dimension(data_dir, save_dir=save_dir)
    except Exception as e:
        print(f"  Skipping: {e}")
    
    print("\nDone! Plots saved to:", save_dir or "display only")


def main():
    parser = argparse.ArgumentParser(description="Plot reproduction results")
    parser.add_argument("--data_dir", default="../lottery-subspace-data/",
                       help="Directory containing result pickle files")
    parser.add_argument("--save_dir", default=None,
                       help="Directory to save plots (displays if not set)")
    parser.add_argument("--dataset", choices=["MNIST", "all"], default="all",
                       help="Which dataset to plot")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        plot_all(args.data_dir, args.save_dir)
    else:
        plot_phase_transition(args.data_dir, args.dataset, args.save_dir)
        plot_burn_in_effect(args.data_dir, args.dataset, args.save_dir)
        plot_method_comparison(args.data_dir, args.dataset, args.save_dir)


if __name__ == "__main__":
    main()
