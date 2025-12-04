#!/usr/bin/env python3
"""
Plotting Script for Conv-2 Reproduction Results
(No Pickle Version - Parses log files directly)

Generates plots matching the figures from:
"How Many Degrees of Freedom Do We Need to Train Deep Networks"

Usage:
    python plot_results_no_pickle.py --data_dir ./lottery-subspace-data/
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Consistent color scheme across all plots
COLORS = {
    'init0': '#1f77b4',      # Blue - Random subspace
    'init4': '#ff7f0e',      # Orange - Burn-in 4
    'init16': '#2ca02c',     # Green - Burn-in 16
    'init64': '#d62728',     # Red - Burn-in 64
    'lottery_subspace': '#9467bd',  # Purple - Lottery subspace
    'baseline': '#3498db',   # Blue - Baseline lottery
    'transfer': '#e74c3c',   # Red - Transfer lottery
}


def parse_log_file(filepath):
    """
    Parse a training log file and extract accuracy data.
    
    Handles two formats:
    1. Transfer logs: "Run Number X | Seed Y" followed by "Number of params = Z   subspace d=D"
    2. Burn-in logs: "Run Number X" followed by "Number of params = Z   subspace d=D"
    
    Returns a dict matching the pickle format:
    {
        'data': {
            'd': [...],           # subspace dimensions
            'test_acc': [...],    # final test accuracies
            'full_train_acc': [...],  # final train accuracies
        },
        'transfer_from': ...,
        'seed_offset': ...,
    }
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    result = {
        'data': {
            'd': [],
            'test_acc': [],
            'full_train_acc': [],
            'seed': [],
            'point_id': [],
        },
        'transfer_from': None,
        'seed_offset': None,
    }
    
    # Extract transfer info
    transfer_match = re.search(r'Donor subspace file:\s*(\S+)', content)
    if transfer_match:
        result['transfer_from'] = transfer_match.group(1)
    
    seed_offset_match = re.search(r'Seed offset:\s*(\d+)', content)
    if seed_offset_match:
        result['seed_offset'] = int(seed_offset_match.group(1))
    
    # Try transfer log format first: "Run Number X | Seed Y"
    run_pattern_transfer = r'Run Number\s+(\d+)\s*\|\s*Seed\s+(\d+)\s*\n.*?Number of params\s*=\s*\d+\s+subspace d=(\d+)'
    run_matches = list(re.finditer(run_pattern_transfer, content))
    
    if run_matches:
        # Transfer format
        for i, match in enumerate(run_matches):
            point_id = int(match.group(1))
            seed = int(match.group(2))
            d = int(match.group(3))
            
            start_pos = match.end()
            if i + 1 < len(run_matches):
                end_pos = run_matches[i + 1].start()
            else:
                end_pos = len(content)
            
            run_content = content[start_pos:end_pos]
            final_test_acc, final_train_acc = _extract_final_accuracies(run_content)
            
            if final_test_acc is not None:
                result['data']['d'].append(d)
                result['data']['test_acc'].append(str(final_test_acc))
                result['data']['full_train_acc'].append(str(final_train_acc))
                result['data']['seed'].append(seed)
                result['data']['point_id'].append(point_id)
    else:
        # Try burn-in format: "Run Number X" (no seed)
        run_pattern_burnin = r'Run Number\s+(\d+)\s*\nNumber of params\s*=\s*\d+\s+subspace d=(\d+)'
        run_matches = list(re.finditer(run_pattern_burnin, content))
        
        for i, match in enumerate(run_matches):
            point_id = int(match.group(1))
            d = int(match.group(2))
            
            start_pos = match.end()
            if i + 1 < len(run_matches):
                end_pos = run_matches[i + 1].start()
            else:
                end_pos = len(content)
            
            run_content = content[start_pos:end_pos]
            final_test_acc, final_train_acc = _extract_final_accuracies(run_content)
            
            if final_test_acc is not None:
                result['data']['d'].append(d)
                result['data']['test_acc'].append(str(final_test_acc))
                result['data']['full_train_acc'].append(str(final_train_acc))
                result['data']['seed'].append(point_id)  # Use point_id as seed placeholder
                result['data']['point_id'].append(point_id)
    
    return result


def _extract_final_accuracies(run_content):
    """Extract the final test and train accuracy from a run block."""
    # Pattern for data lines with actual test accuracy (7 columns, all numbers)
    data_line_pattern = r'^(\d+\.?\d*)\s+(\d+)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*$'
    
    final_test_acc = None
    final_train_acc = None
    
    for line in run_content.split('\n'):
        line = line.strip()
        match_line = re.match(data_line_pattern, line)
        if match_line:
            train_acc = float(match_line.group(5))
            test_acc = float(match_line.group(7))
            final_train_acc = train_acc
            final_test_acc = test_acc
    
    return final_test_acc, final_train_acc


def find_log_files(data_dir, pattern):
    """Find log files matching a pattern."""
    files = glob.glob(os.path.join(data_dir, f"*{pattern}*_log.txt"))
    return sorted(files)


def extract_phase_transition_data(result):
    """Extract data for phase transition plot from a results dict."""
    data = result['data']
    if not data['d']:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    ds = np.array(data['d'])
    test_accs = np.array([float(x) for x in data['test_acc']])
    train_accs = np.array([float(x) for x in data['full_train_acc']])
    
    # Group by dimension
    unique_ds = sorted(np.unique(ds))
    mean_test_acc = []
    std_test_acc = []
    mean_train_acc = []
    
    for d in unique_ds:
        mask = ds == d
        mean_test_acc.append(np.mean(test_accs[mask]))
        std_test_acc.append(np.std(test_accs[mask]))
        mean_train_acc.append(np.mean(train_accs[mask]))
    
    return np.array(unique_ds), np.array(mean_test_acc), np.array(std_test_acc), np.array(mean_train_acc)


def extract_raw_accuracies(result):
    """Extract raw accuracy data grouped by dimension."""
    data = result['data']
    if not data['d']:
        return {}
    
    ds = np.array(data['d'])
    test_accs = np.array([float(x) for x in data['test_acc']])
    
    unique_ds = sorted(np.unique(ds))
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
        (f"burn_in_subspace_TinyCNN_{dataset}_init0", "Random\nSubspace", COLORS['init0']),
        (f"burn_in_subspace_TinyCNN_{dataset}_init4", "Burn-in\n4 steps", COLORS['init4']),
        (f"burn_in_subspace_TinyCNN_{dataset}_init16", "Burn-in\n16 steps", COLORS['init16']),
        (f"burn_in_subspace_TinyCNN_{dataset}_init64", "Burn-in\n64 steps", COLORS['init64']),
        (f"lottery_subspace_TinyCNN_{dataset}", "Lottery\nSubspace", COLORS['lottery_subspace']),
    ]
    
    # Accuracy thresholds for y-axis
    accuracy_thresholds = np.linspace(0.1, 1.0, 50)
    
    # Find how many methods have data
    valid_methods = []
    for pattern, label, color in methods:
        files = find_log_files(data_dir, pattern)
        if files:
            result = parse_log_file(sorted(files)[0])
            if result['data']['d']:
                valid_methods.append((pattern, label, color, result))
    
    if not valid_methods:
        print("No data found for heatmap")
        return
    
    n_methods = len(valid_methods)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(3 * (n_methods + 1), 5), 
                              gridspec_kw={'width_ratios': [1]*n_methods + [1.2]})
    
    threshold_curves = []
    largest_dims = None
    
    for idx, (pattern, label, color, result) in enumerate(valid_methods):
        ax = axes[idx]
        acc_by_dim = extract_raw_accuracies(result)
        prob_matrix, dims = compute_success_probability(acc_by_dim, accuracy_thresholds)
        
        if largest_dims is None or len(dims) > len(largest_dims):
            largest_dims = dims
        
        # Plot heatmap
        extent = [0, len(dims), accuracy_thresholds[0], accuracy_thresholds[-1]]
        im = ax.imshow(prob_matrix, aspect='auto', origin='lower', 
                       extent=extent, cmap='gray', vmin=0, vmax=1)
        
        # Overlay the mean accuracy curve
        ds_arr, mean_acc, _, _ = extract_phase_transition_data(result)
        # Map dimensions to x positions
        x_positions = [list(dims).index(d) + 0.5 for d in ds_arr if d in dims]
        mean_acc_filtered = [mean_acc[i] for i, d in enumerate(ds_arr) if d in dims]
        
        ax.plot(x_positions, mean_acc_filtered, color=color, linewidth=2)
        threshold_curves.append((x_positions, mean_acc_filtered, color, label.replace('\n', ' ')))
        
        # Labels
        ax.set_xticks(np.arange(len(dims)) + 0.5)
        ax.set_xticklabels([str(int(d)) for d in dims], rotation=45, fontsize=8)
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
    
    if largest_dims:
        ax_right.set_xticks(np.arange(len(largest_dims)) + 0.5)
        ax_right.set_xticklabels([str(int(d)) for d in largest_dims], rotation=45, fontsize=8)
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
    files = find_log_files(data_dir, pattern)
    
    if not files:
        print(f"No files found for pattern: {pattern}")
        return
    
    for f in files:
        result = parse_log_file(f)
        ds, mean_acc, std_acc, _ = extract_phase_transition_data(result)
        
        if len(ds) > 0:
            ax.errorbar(ds, mean_acc, yerr=std_acc, marker='o', capsize=3,
                       label=os.path.basename(f))
    
    if len(ds) > 0:
        ax.set_xscale('log', base=2)
        ax.set_xticks(ds)
        ax.set_xticklabels([str(int(d)) for d in ds])
    ax.set_xlabel('Training Dimension (d)', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title(f'Phase Transition: Random Subspace Training\nTinyCNN on {dataset}', fontsize=16)
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
        (0, "Random Subspace", COLORS['init0'], 'o'),
        (4, "Burn-in (4 steps)", COLORS['init4'], 's'),
        (16, "Burn-in (16 steps)", COLORS['init16'], '^'),
        (64, "Burn-in (64 steps)", COLORS['init64'], 'p'),
    ]
    
    all_dims = set()
    
    for init_iters, label, color, marker in methods:
        pattern = f"burn_in_subspace_TinyCNN_{dataset}_init{init_iters}"
        files = find_log_files(data_dir, pattern)
        
        if files:
            result = parse_log_file(files[0])
            ds, mean_acc, std_acc, _ = extract_phase_transition_data(result)
            if len(ds) > 0:
                all_dims.update(ds)
                ax.plot(ds, mean_acc, marker=marker, color=color, label=label, linewidth=2)
    
    all_dims_sorted = sorted(all_dims)
    ax.set_xscale('log', base=2)
    if all_dims_sorted:
        ax.set_xticks(all_dims_sorted)
        ax.set_xticklabels([str(int(d)) for d in all_dims_sorted], rotation=45)
    ax.set_xlabel('Training Dimension (d)', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title(f'Effect of Burn-in Iterations\nTinyCNN on {dataset}', fontsize=16)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'fig3_burn_in_effect_{dataset}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_method_comparison(data_dir, dataset="MNIST", save_dir=None):
    """
    Plot Figure 5: Comparison of all subspace methods.
    Order: Random -> Burn-ins -> Lottery Subspace
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Methods to plot
    methods = [
        (f"burn_in_subspace_TinyCNN_{dataset}_init0", "Random Subspace", COLORS['init0'], 'o'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init4", "Burn-in (4 steps)", COLORS['init4'], 's'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init16", "Burn-in (16 steps)", COLORS['init16'], '^'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init64", "Burn-in (64 steps)", COLORS['init64'], 'p'),
        (f"lottery_subspace_TinyCNN_{dataset}", "Lottery Subspace", COLORS['lottery_subspace'], 'D'),
    ]
    
    all_dims = set()
    for pattern, label, color, marker in methods:
        files = find_log_files(data_dir, pattern)
        if files:
            result = parse_log_file(files[0])
            ds, mean_acc, std_acc, _ = extract_phase_transition_data(result)
            if len(ds) > 0:
                all_dims.update(ds)
                ax.plot(ds, mean_acc, marker=marker, color=color, label=label, linewidth=2)
    
    all_dims_sorted = sorted(all_dims)
    ax.set_xscale('log', base=2)
    if all_dims_sorted:
        ax.set_xticks(all_dims_sorted)
        ax.set_xticklabels([str(int(d)) for d in all_dims_sorted], rotation=45)
    ax.set_xlabel('Training Dimension (d)', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title(f'Subspace Methods Comparison\nTinyCNN on {dataset}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'fig5_method_comparison_{dataset}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_dimension(data_dir, dataset="MNIST", accuracy_threshold=0.9, save_dir=None):
    """
    Plot Figure 4: Threshold training dimension as function of method.
    Find the minimum dimension needed to reach a target accuracy.
    Order: Random -> Burn-ins -> Lottery Subspace
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define methods in order with their colors
    methods_order = [
        ('Random', f"burn_in_subspace_TinyCNN_{dataset}_init0", COLORS['init0']),
        ('Burn-in 4', f"burn_in_subspace_TinyCNN_{dataset}_init4", COLORS['init4']),
        ('Burn-in 16', f"burn_in_subspace_TinyCNN_{dataset}_init16", COLORS['init16']),
        ('Burn-in 64', f"burn_in_subspace_TinyCNN_{dataset}_init64", COLORS['init64']),
        ('Lottery Subspace', f"lottery_subspace_TinyCNN_{dataset}", COLORS['lottery_subspace']),
    ]
    
    method_results = {}
    
    for method_name, pattern, color in methods_order:
        files = find_log_files(data_dir, pattern)
        if files:
            result = parse_log_file(files[0])
            ds, mean_acc, _, _ = extract_phase_transition_data(result)
            if len(ds) > 0:
                above_thresh = ds[mean_acc >= accuracy_threshold]
                if len(above_thresh) > 0:
                    method_results[method_name] = (above_thresh[0], color)
    
    # Filter to only methods with data, but keep order
    available_methods = []
    available_colors = []
    available_values = []
    for method_name, _, color in methods_order:
        if method_name in method_results:
            available_methods.append(method_name)
            available_values.append(method_results[method_name][0])
            available_colors.append(method_results[method_name][1])
    
    x = np.arange(len(available_methods))
    width = 0.6
    
    ax.bar(x, available_values, width, color=available_colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Threshold Training Dimension', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_title(f'Threshold Dimension to Reach {accuracy_threshold*100:.0f}% Test Accuracy\nTinyCNN on {dataset}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(available_methods, rotation=45, ha='right')
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'fig4_threshold_dimension_{dataset}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_train_test_comparison(data_dir, dataset="MNIST", save_dir=None):
    """
    Plot train vs test accuracy comparison across all methods.
    X-axis: Training dimension (d)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Methods to plot (subspace methods)
    subspace_methods = [
        (f"burn_in_subspace_TinyCNN_{dataset}_init0", "Random Subspace", COLORS['init0'], 'o'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init4", "Burn-in 4", COLORS['init4'], 's'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init16", "Burn-in 16", COLORS['init16'], '^'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init64", "Burn-in 64", COLORS['init64'], 'D'),
        (f"lottery_subspace_TinyCNN_{dataset}", "Lottery Subspace", COLORS['lottery_subspace'], 'p'),
    ]
    
    all_dims = set()
    
    # Plot subspace methods
    for pattern, label, color, marker in subspace_methods:
        files = find_log_files(data_dir, pattern)
        if files:
            result = parse_log_file(files[0])
            data = result['data']
            if data['d']:
                ds = np.array(data['d'])
                test_accs = np.array([float(x) for x in data['test_acc']])
                train_accs = np.array([float(x) for x in data['full_train_acc']])
                
                # Group by dimension
                unique_ds = sorted(np.unique(ds))
                all_dims.update(unique_ds)
                
                mean_test = []
                std_test = []
                mean_train = []
                std_train = []
                
                for d in unique_ds:
                    mask = ds == d
                    mean_test.append(np.mean(test_accs[mask]))
                    std_test.append(np.std(test_accs[mask]))
                    mean_train.append(np.mean(train_accs[mask]))
                    std_train.append(np.std(train_accs[mask]))
                
                # Left: Train accuracy
                ax1.errorbar(unique_ds, mean_train, yerr=std_train, 
                            marker=marker, color=color, label=label, 
                            linewidth=2, capsize=3, markersize=6)
                
                # Right: Test accuracy
                ax2.errorbar(unique_ds, mean_test, yerr=std_test,
                            marker=marker, color=color, label=label,
                            linewidth=2, capsize=3, markersize=6)
    
    # Formatting
    all_dims_sorted = sorted(all_dims)
    for ax, title in [(ax1, 'Full Train Accuracy'), (ax2, 'Test Accuracy')]:
        ax.set_xscale('log', base=2)
        if all_dims_sorted:
            ax.set_xticks(all_dims_sorted)
            ax.set_xticklabels([str(int(d)) for d in all_dims_sorted], rotation=45)
        ax.set_xlabel('Training Dimension (d)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower right', fontsize=9)
    
    ax1.set_ylabel('Accuracy', fontsize=12)
    
    plt.suptitle(f'Train vs Test Accuracy\nTinyCNN on {dataset}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'fig6_train_test_comparison_{dataset}.png'), 
                   dpi=150, bbox_inches='tight')
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
    
    print("\nGenerating Train/Test Comparison plot (Fig 6)...")
    try:
        plot_train_test_comparison(data_dir, 'MNIST', save_dir)
    except Exception as e:
        print(f"  Skipping: {e}")
    
    print("\nGenerating Threshold Dimension plot (Fig 4)...")
    try:
        plot_threshold_dimension(data_dir, 'MNIST', save_dir=save_dir)
    except Exception as e:
        print(f"  Skipping: {e}")
    
    print("\nDone! Plots saved to:", save_dir or "display only")


def main():
    parser = argparse.ArgumentParser(description="Plot reproduction results from log files (no pickle needed)")
    parser.add_argument("--data_dir", default="./lottery-subspace-data/",
                       help="Directory containing log files")
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

