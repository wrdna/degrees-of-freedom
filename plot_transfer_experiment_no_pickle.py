#!/usr/bin/env python3
"""
Plotting Script for Cross-Initialization Lottery Subspace Transfer Experiment
(No Pickle Version - Parses log files directly)

Compares:
- Transfer: Training with donor seed's lottery subspace
- Baseline: Training with each seed's own lottery subspace

Usage:
    python plot_transfer_experiment_no_pickle.py --data_dir ../cross-init-lottery-subspace-data/
    
    # Or specify files directly:
    python plot_transfer_experiment_no_pickle.py \
        --transfer_file ../cross-init-lottery-subspace-data/artifact_lottery_subspace_transfer_..._log.txt \
        --baseline_file ../cross-init-lottery-subspace-data/artifact_lottery_subspace_transfer_..._log.txt
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Color scheme
COLORS = {
    'transfer': '#e74c3c',    # Red - transferred subspace
    'baseline': '#3498db',    # Blue - own subspace  
    'random': '#95a5a6',      # Gray - random subspace reference
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


def find_transfer_log_files(data_dir):
    """Find transfer experiment log files."""
    transfer_files = glob.glob(os.path.join(data_dir, "*_transfer_*_log.txt"))
    baseline_files = glob.glob(os.path.join(data_dir, "*_baseline_*_log.txt"))
    return sorted(transfer_files), sorted(baseline_files)


def extract_accuracy_data(result):
    """Extract accuracy data grouped by dimension."""
    data = result['data']
    ds = np.array(data['d'])
    test_accs = np.array([float(x) for x in data['test_acc']])
    train_accs = np.array([float(x) for x in data['full_train_acc']])
    
    unique_ds = sorted(np.unique(ds))
    
    test_by_dim = {}
    train_by_dim = {}
    
    for d in unique_ds:
        mask = ds == d
        test_by_dim[d] = test_accs[mask]
        train_by_dim[d] = train_accs[mask]
    
    return unique_ds, test_by_dim, train_by_dim


def compute_stats(acc_by_dim, dims):
    """Compute mean and std for each dimension."""
    means = []
    stds = []
    for d in dims:
        accs = acc_by_dim[d]
        means.append(np.mean(accs))
        stds.append(np.std(accs))
    return np.array(means), np.array(stds)


def plot_transfer_comparison(transfer_result, baseline_result, save_path=None):
    """
    Main comparison plot: Test accuracy vs dimension for transfer vs baseline.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Extract data
    dims_t, test_t, train_t = extract_accuracy_data(transfer_result)
    dims_b, test_b, train_b = extract_accuracy_data(baseline_result)
    
    # Use common dimensions
    common_dims = sorted(set(dims_t) & set(dims_b))
    
    # Compute stats
    mean_t, std_t = compute_stats(test_t, common_dims)
    mean_b, std_b = compute_stats(test_b, common_dims)
    
    # Plot with error bands
    ax.fill_between(common_dims, mean_t - std_t, mean_t + std_t, 
                    alpha=0.2, color=COLORS['transfer'])
    ax.fill_between(common_dims, mean_b - std_b, mean_b + std_b, 
                    alpha=0.2, color=COLORS['baseline'])
    
    ax.plot(common_dims, mean_t, 'o-', color=COLORS['transfer'], 
            label='Transfer (donor subspace)', linewidth=2, markersize=6)
    ax.plot(common_dims, mean_b, 's-', color=COLORS['baseline'], 
            label='Baseline (own subspace)', linewidth=2, markersize=6)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(common_dims)
    ax.set_xticklabels([str(d) for d in common_dims])
    ax.set_xlabel('Subspace Dimension (d)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Cross-Initialization Lottery Subspace Transfer', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add metadata
    n_transfer = len(list(test_t.values())[0]) if test_t else 0
    n_baseline = len(list(test_b.values())[0]) if test_b else 0
    ax.text(0.02, 0.98, f'Transfer: {n_transfer} seeds | Baseline: {n_baseline} seeds',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_transfer_gap(transfer_result, baseline_result, save_path=None):
    """
    Plot the accuracy gap: (baseline - transfer) vs dimension.
    Positive = baseline wins, Negative = transfer wins.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # Extract data
    dims_t, test_t, _ = extract_accuracy_data(transfer_result)
    dims_b, test_b, _ = extract_accuracy_data(baseline_result)
    
    common_dims = sorted(set(dims_t) & set(dims_b))
    
    mean_t, std_t = compute_stats(test_t, common_dims)
    mean_b, std_b = compute_stats(test_b, common_dims)
    
    gap = mean_b - mean_t
    gap_std = np.sqrt(std_t**2 + std_b**2)  # Combined uncertainty
    
    # Plot gap
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between(common_dims, gap - gap_std, gap + gap_std, alpha=0.3, color='purple')
    ax.plot(common_dims, gap, 'o-', color='purple', linewidth=2, markersize=6)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(common_dims)
    ax.set_xticklabels([str(d) for d in common_dims])
    ax.set_xlabel('Subspace Dimension (d)', fontsize=12)
    ax.set_ylabel('Accuracy Gap (Baseline - Transfer)', fontsize=12)
    ax.set_title('Transfer Penalty: How much accuracy is lost by using donor subspace?', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Annotate interpretation
    ax.text(0.98, 0.98, '↑ Baseline better\n↓ Transfer better',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_individual_runs(transfer_result, baseline_result, save_path=None):
    """
    Scatter plot showing individual runs for each condition.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    dims_t, test_t, _ = extract_accuracy_data(transfer_result)
    dims_b, test_b, _ = extract_accuracy_data(baseline_result)
    
    common_dims = sorted(set(dims_t) & set(dims_b))
    
    # Plot individual points with jitter
    for d in common_dims:
        jitter = 0.05
        
        # Transfer points
        n_t = len(test_t[d])
        x_t = d * (1 - jitter + np.random.rand(n_t) * jitter * 2)
        ax.scatter(x_t, test_t[d], color=COLORS['transfer'], alpha=0.6, s=40, 
                   label='Transfer' if d == common_dims[0] else '')
        
        # Baseline points
        n_b = len(test_b[d])
        x_b = d * (1 + jitter + np.random.rand(n_b) * jitter * 2)
        ax.scatter(x_b, test_b[d], color=COLORS['baseline'], alpha=0.6, s=40,
                   label='Baseline' if d == common_dims[0] else '')
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(common_dims)
    ax.set_xticklabels([str(d) for d in common_dims])
    ax.set_xlabel('Subspace Dimension (d)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Individual Runs: Transfer vs Baseline', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def print_summary(transfer_result, baseline_result):
    """Print summary statistics."""
    dims_t, test_t, _ = extract_accuracy_data(transfer_result)
    dims_b, test_b, _ = extract_accuracy_data(baseline_result)
    
    common_dims = sorted(set(dims_t) & set(dims_b))
    
    print("\n" + "="*70)
    print("CROSS-INITIALIZATION TRANSFER EXPERIMENT SUMMARY")
    print("="*70)
    
    print(f"\nTransfer from: {transfer_result.get('transfer_from', 'N/A')}")
    print(f"Transfer seeds: {transfer_result.get('seed_offset', 'N/A')}")
    print(f"Points per dimension: Transfer={len(list(test_t.values())[0]) if test_t else 0}, "
          f"Baseline={len(list(test_b.values())[0]) if test_b else 0}")
    
    print("\n{:>10} | {:>15} | {:>15} | {:>10}".format(
        "Dim", "Transfer Acc", "Baseline Acc", "Gap"))
    print("-"*60)
    
    for d in common_dims:
        mean_t = np.mean(test_t[d])
        mean_b = np.mean(test_b[d])
        gap = mean_b - mean_t
        gap_str = f"{gap:+.3f}"
        print(f"{d:>10} | {mean_t:>15.3f} | {mean_b:>15.3f} | {gap_str:>10}")
    
    # Overall summary
    all_gaps = []
    for d in common_dims:
        all_gaps.append(np.mean(test_b[d]) - np.mean(test_t[d]))
    
    print("-"*60)
    print(f"\nMean gap across dimensions: {np.mean(all_gaps):+.4f}")
    print(f"Max gap (baseline advantage): {np.max(all_gaps):+.4f}")
    print(f"Min gap (transfer advantage): {np.min(all_gaps):+.4f}")
    
    # Interpretation
    print("\n" + "-"*70)
    if np.mean(all_gaps) < 0.02:
        print("RESULT: Transfer works well! Lottery subspaces appear architecture-intrinsic.")
    elif np.mean(all_gaps) < 0.05:
        print("RESULT: Moderate transfer penalty. Subspaces partially transfer.")
    else:
        print("RESULT: Significant transfer penalty. Subspaces appear init-specific.")
    print("-"*70 + "\n")


def parse_burn_in_log_file(filepath):
    """
    Parse a burn-in subspace log file (same format as transfer logs).
    """
    return parse_log_file(filepath)


def plot_burn_in_with_transfer(transfer_result, original_data_dir='../lottery-subspace-data/',
                                dataset="MNIST", save_path=None):
    """
    Burn-in effect plot including transfer method.
    Order: Random -> Burn-ins -> Lottery Subspace -> Transfer
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Consistent color scheme
    colors = {
        'random': '#1f77b4',       # Blue
        'burn_in_4': '#ff7f0e',    # Orange
        'burn_in_16': '#2ca02c',   # Green
        'burn_in_64': '#d62728',   # Red
        'lottery': '#9467bd',      # Purple
        'transfer': '#17becf',     # Cyan
    }
    
    all_dims = set()
    
    # Methods in order: Random, Burn-in 4, 16, 64, Lottery Subspace, Transfer
    methods_order = [
        (f"burn_in_subspace_TinyCNN_{dataset}_init0", "Random Subspace", colors['random'], 'o'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init4", "Burn-in (4 steps)", colors['burn_in_4'], 's'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init16", "Burn-in (16 steps)", colors['burn_in_16'], '^'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init64", "Burn-in (64 steps)", colors['burn_in_64'], 'p'),
        (f"lottery_subspace_TinyCNN_{dataset}", "Lottery Subspace", colors['lottery'], 'D'),
    ]
    
    for pattern, label, color, marker in methods_order:
        files = glob.glob(os.path.join(original_data_dir, f"*{pattern}*_log.txt"))
        if files:
            result = parse_log_file(sorted(files)[0])
            data = result['data']
            if data['d']:
                ds = np.array(data['d'])
                test_accs = np.array([float(x) for x in data['test_acc']])
                unique_ds = sorted(np.unique(ds))
                all_dims.update(unique_ds)
                mean_acc = [np.mean(test_accs[ds == d]) for d in unique_ds]
                ax.plot(unique_ds, mean_acc, marker=marker, color=color, label=label, linewidth=2)
    
    # Transfer - plotted last
    dims_t, test_t, _ = extract_accuracy_data(transfer_result)
    all_dims.update(dims_t)
    mean_t, _ = compute_stats(test_t, dims_t)
    ax.plot(dims_t, mean_t, 'v-', color=colors['transfer'], 
           label='Transfer Lottery Subspace', linewidth=2.5, markersize=8)
    
    all_dims_sorted = sorted(all_dims)
    ax.set_xscale('log', base=2)
    ax.set_xticks(all_dims_sorted)
    ax.set_xticklabels([str(int(d)) for d in all_dims_sorted], rotation=45)
    ax.set_xlabel('Training Dimension (d)', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title(f'Subspace Methods Comparison\nTinyCNN on {dataset}', fontsize=14)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_all_methods_comparison(transfer_result, baseline_result=None, 
                                 original_data_dir='../lottery-subspace-data/',
                                 dataset="MNIST", save_path=None):
    """
    Combined plot showing all subspace methods including transfer.
    Order: Random -> Burn-ins -> Lottery Subspace -> Transfer
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Consistent color scheme
    colors = {
        'random': '#1f77b4',       # Blue
        'burn_in_4': '#ff7f0e',    # Orange
        'burn_in_16': '#2ca02c',   # Green
        'burn_in_64': '#d62728',   # Red
        'lottery': '#9467bd',      # Purple
        'transfer': '#17becf',     # Cyan
    }
    
    # Methods in consistent order
    methods_order = [
        (f"burn_in_subspace_TinyCNN_{dataset}_init0", "Random Subspace", colors['random'], 'o'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init4", "Burn-in (4 steps)", colors['burn_in_4'], 's'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init16", "Burn-in (16 steps)", colors['burn_in_16'], '^'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init64", "Burn-in (64 steps)", colors['burn_in_64'], 'p'),
        (f"lottery_subspace_TinyCNN_{dataset}", "Lottery Subspace", colors['lottery'], 'D'),
    ]
    
    all_dims = set()
    
    for pattern, label, color, marker in methods_order:
        files = glob.glob(os.path.join(original_data_dir, f"*{pattern}*_log.txt"))
        if files:
            result = parse_log_file(sorted(files)[0])
            data = result['data']
            if data['d']:
                ds = np.array(data['d'])
                test_accs = np.array([float(x) for x in data['test_acc']])
                
                unique_ds = sorted(np.unique(ds))
                all_dims.update(unique_ds)
                mean_acc = [np.mean(test_accs[ds == d]) for d in unique_ds]
                
                ax.plot(unique_ds, mean_acc, marker=marker, color=color, 
                       label=label, linewidth=2, markersize=7)
    
    # Transfer - plotted last
    dims_t, test_t, _ = extract_accuracy_data(transfer_result)
    all_dims.update(dims_t)
    mean_t, std_t = compute_stats(test_t, dims_t)
    ax.plot(dims_t, mean_t, 'v-', color=colors['transfer'], 
           label='Transfer Lottery Subspace', linewidth=2.5, markersize=9)
    
    # Formatting
    ax.set_xscale('log', base=2)
    all_dims_sorted = sorted(all_dims)
    ax.set_xticks(all_dims_sorted)
    ax.set_xticklabels([str(int(d)) for d in all_dims_sorted], rotation=45)
    ax.set_xlabel('Subspace Dimension (d)', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title(f'All Subspace Methods Comparison\nTinyCNN on {dataset}', fontsize=14)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_phase_transition_with_transfer(transfer_result, baseline_result,
                                         original_data_dir='../lottery-subspace-data/',
                                         dataset="MNIST", save_path=None):
    """
    Phase transition plot: All methods including baseline + transfer.
    Order: Random -> Burn-ins -> Lottery Subspace -> Baseline -> Transfer
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color scheme for all methods
    colors = {
        'random': '#1f77b4',       # Blue
        'burn_in_4': '#ff7f0e',    # Orange
        'burn_in_16': '#2ca02c',   # Green
        'burn_in_64': '#d62728',   # Red
        'lottery': '#9467bd',      # Purple
        'baseline': '#17becf',     # Cyan
        'transfer': '#e74c3c',     # Red (different shade)
    }
    
    all_dims = set()
    
    # Methods from original data dir (in order)
    methods_order = [
        (f"burn_in_subspace_TinyCNN_{dataset}_init0", "Random Subspace", colors['random'], 'o'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init4", "Burn-in (4 steps)", colors['burn_in_4'], 's'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init16", "Burn-in (16 steps)", colors['burn_in_16'], '^'),
        (f"burn_in_subspace_TinyCNN_{dataset}_init64", "Burn-in (64 steps)", colors['burn_in_64'], 'p'),
        (f"lottery_subspace_TinyCNN_{dataset}", "Lottery Subspace", colors['lottery'], 'D'),
    ]
    
    for pattern, label, color, marker in methods_order:
        files = glob.glob(os.path.join(original_data_dir, f"*{pattern}*_log.txt"))
        if files:
            result = parse_log_file(sorted(files)[0])
            data = result['data']
            if data['d']:
                ds = np.array(data['d'])
                test_accs = np.array([float(x) for x in data['test_acc']])
                unique_ds = sorted(np.unique(ds))
                all_dims.update(unique_ds)
                mean_acc = [np.mean(test_accs[ds == d]) for d in unique_ds]
                std_acc = [np.std(test_accs[ds == d]) for d in unique_ds]
                ax.errorbar(unique_ds, mean_acc, yerr=std_acc, marker=marker, color=color,
                           label=label, capsize=3, linewidth=2, markersize=6)
    
    # Baseline (own subspace)
    dims_b, test_b, _ = extract_accuracy_data(baseline_result)
    if dims_b:
        all_dims.update(dims_b)
        mean_b, std_b = compute_stats(test_b, dims_b)
        ax.errorbar(dims_b, mean_b, yerr=std_b, marker='h', color=colors['baseline'],
                    label='Baseline Lottery', capsize=3, linewidth=2.5, markersize=7)
    
    # Transfer (donor subspace)
    dims_t, test_t, _ = extract_accuracy_data(transfer_result)
    if dims_t:
        all_dims.update(dims_t)
        mean_t, std_t = compute_stats(test_t, dims_t)
        ax.errorbar(dims_t, mean_t, yerr=std_t, marker='v', color=colors['transfer'],
                    label='Transfer Lottery', capsize=3, linewidth=2.5, markersize=8)
    
    all_dims_sorted = sorted(all_dims)
    ax.set_xscale('log', base=2)
    ax.set_xticks(all_dims_sorted)
    ax.set_xticklabels([str(int(d)) for d in all_dims_sorted], rotation=45)
    ax.set_xlabel('Training Dimension (d)', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title(f'Phase Transition: All Methods Comparison\nTinyCNN on {dataset}', fontsize=14)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def extract_raw_accuracies(result):
    """Extract raw accuracy data grouped by dimension."""
    data = result['data']
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


def plot_phase_transition_heatmap_with_transfer(transfer_result, baseline_result,
                                                 original_data_dir='lottery-subspace-data/',
                                                 dataset="MNIST", save_path=None):
    """
    Plot Figure 2 style: Heatmap of success probability vs training dimension.
    X-axis: Training dimension
    Y-axis: Accuracy threshold  
    Color: Probability of achieving that accuracy (black=0, white=1)
    
    Includes: Random, Burn-ins, Lottery Subspace, Baseline, Transfer
    """
    # Color scheme
    method_colors = {
        'random': '#1f77b4',       # Blue
        'burn_in_4': '#ff7f0e',    # Orange
        'burn_in_16': '#2ca02c',   # Green
        'burn_in_64': '#d62728',   # Red
        'lottery': '#9467bd',      # Purple
        'baseline': '#3498db',     # Blue (baseline)
        'transfer': '#e74c3c',     # Red (transfer)
    }
    
    # Methods to plot (from log files)
    methods = [
        (f"burn_in_subspace_TinyCNN_{dataset}_init0", "Random\nSubspace", method_colors['random']),
        (f"burn_in_subspace_TinyCNN_{dataset}_init4", "Burn-in\n4 steps", method_colors['burn_in_4']),
        (f"burn_in_subspace_TinyCNN_{dataset}_init16", "Burn-in\n16 steps", method_colors['burn_in_16']),
        (f"burn_in_subspace_TinyCNN_{dataset}_init64", "Burn-in\n64 steps", method_colors['burn_in_64']),
        (f"lottery_subspace_TinyCNN_{dataset}", "Lottery\nSubspace", method_colors['lottery']),
    ]
    
    # Accuracy thresholds for y-axis
    accuracy_thresholds = np.linspace(0.1, 1.0, 50)
    
    # Find how many methods have data
    valid_methods = []
    for pattern, label, color in methods:
        files = glob.glob(os.path.join(original_data_dir, f"*{pattern}*_log.txt"))
        if files:
            result = parse_log_file(sorted(files)[0])
            if result['data']['d']:
                valid_methods.append((pattern, label, color, result))
    
    # Add baseline and transfer
    if baseline_result['data']['d']:
        valid_methods.append(('baseline', 'Baseline\nLottery', method_colors['baseline'], baseline_result))
    if transfer_result['data']['d']:
        valid_methods.append(('transfer', 'Transfer\nLottery', method_colors['transfer'], transfer_result))
    
    if not valid_methods:
        print("No data found for heatmap")
        return
    
    n_methods = len(valid_methods)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(2.5 * (n_methods + 1), 5), 
                              gridspec_kw={'width_ratios': [1]*n_methods + [1.2]})
    
    threshold_curves = []
    all_dims = set()
    
    for idx, (pattern, label, color, result) in enumerate(valid_methods):
        ax = axes[idx]
        acc_by_dim = extract_raw_accuracies(result)
        prob_matrix, dims = compute_success_probability(acc_by_dim, accuracy_thresholds)
        all_dims.update(dims)
        
        # Plot heatmap
        extent = [0, len(dims), accuracy_thresholds[0], accuracy_thresholds[-1]]
        im = ax.imshow(prob_matrix, aspect='auto', origin='lower', 
                       extent=extent, cmap='gray', vmin=0, vmax=1)
        
        # Overlay the mean accuracy curve
        unique_ds, test_by_dim, _ = extract_accuracy_data(result)
        mean_acc, _ = compute_stats(test_by_dim, unique_ds)
        
        # Map dimensions to x positions
        x_positions = [list(dims).index(d) + 0.5 for d in unique_ds if d in dims]
        mean_acc_filtered = [mean_acc[i] for i, d in enumerate(unique_ds) if d in dims]
        
        ax.plot(x_positions, mean_acc_filtered, color=color, linewidth=2)
        threshold_curves.append((x_positions, mean_acc_filtered, color, label.replace('\n', ' ')))
        
        # Labels
        ax.set_xticks(np.arange(len(dims)) + 0.5)
        ax.set_xticklabels([str(int(d)) for d in dims], rotation=45, fontsize=7)
        ax.set_xlabel('Training Dim', fontsize=9)
        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(label, fontsize=10)
        ax.axhline(y=0.9, color='white', linestyle='--', alpha=0.7, linewidth=0.5)
    
    # Right panel: threshold curves overlay
    ax_right = axes[-1]
    
    # Get max dims for x-axis
    max_dims = max(len(tc[0]) for tc in threshold_curves)
    
    for x_pos, mean_acc, color, label in threshold_curves:
        ax_right.plot(x_pos, mean_acc, color=color, linewidth=2, marker='o', 
                     markersize=3, label=label)
    
    # Use the largest dimension set for x-axis
    largest_dims = None
    for _, _, _, result in valid_methods:
        dims = sorted(np.unique(result['data']['d']))
        if largest_dims is None or len(dims) > len(largest_dims):
            largest_dims = dims
    
    if largest_dims:
        ax_right.set_xticks(np.arange(len(largest_dims)) + 0.5)
        ax_right.set_xticklabels([str(int(d)) for d in largest_dims], rotation=45, fontsize=7)
    ax_right.set_xlabel('Training Dim', fontsize=9)
    ax_right.set_ylabel('Accuracy', fontsize=12)
    ax_right.set_title('Threshold\nComparison', fontsize=10)
    ax_right.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax_right.legend(fontsize=7, loc='lower right')
    ax_right.set_ylim(0, 1.05)
    ax_right.grid(True, alpha=0.3)
    
    plt.suptitle(f'Phase Transitions in Achievable Accuracy vs Training Dimension\n{dataset}', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_threshold_dimension_with_transfer(transfer_result, original_data_dir='../lottery-subspace-data/',
                                            dataset="MNIST", accuracy_threshold=0.9, save_path=None):
    """
    Bar chart: Threshold dimension for each method including transfer.
    Order: Random -> Burn-ins -> Lottery Subspace -> Transfer
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Methods in order with colors
    methods_order = [
        ('Random', '#1f77b4'),
        ('Burn-in 4', '#ff7f0e'),
        ('Burn-in 16', '#2ca02c'),
        ('Burn-in 64', '#d62728'),
        ('Lottery Subspace', '#9467bd'),
        ('Transfer', '#17becf'),
    ]
    
    method_results = {}
    
    # Random subspace
    pattern = f"burn_in_subspace_TinyCNN_{dataset}_init0"
    files = glob.glob(os.path.join(original_data_dir, f"*{pattern}*_log.txt"))
    if files:
        result = parse_log_file(sorted(files)[0])
        data = result['data']
        if data['d']:
            ds = np.array(data['d'])
            test_accs = np.array([float(x) for x in data['test_acc']])
            unique_ds = sorted(np.unique(ds))
            mean_acc = np.array([np.mean(test_accs[ds == d]) for d in unique_ds])
            above_thresh = np.array(unique_ds)[mean_acc >= accuracy_threshold]
            if len(above_thresh) > 0:
                method_results['Random'] = above_thresh[0]
    
    # Burn-in subspaces
    for init_iters in [4, 16, 64]:
        pattern = f"burn_in_subspace_TinyCNN_{dataset}_init{init_iters}"
        files = glob.glob(os.path.join(original_data_dir, f"*{pattern}*_log.txt"))
        if files:
            result = parse_log_file(sorted(files)[0])
            data = result['data']
            if data['d']:
                ds = np.array(data['d'])
                test_accs = np.array([float(x) for x in data['test_acc']])
                unique_ds = sorted(np.unique(ds))
                mean_acc = np.array([np.mean(test_accs[ds == d]) for d in unique_ds])
                above_thresh = np.array(unique_ds)[mean_acc >= accuracy_threshold]
                if len(above_thresh) > 0:
                    method_results[f'Burn-in {init_iters}'] = above_thresh[0]
    
    # Lottery subspace
    pattern = f"lottery_subspace_TinyCNN_{dataset}"
    files = glob.glob(os.path.join(original_data_dir, f"*{pattern}*_log.txt"))
    if files:
        result = parse_log_file(sorted(files)[0])
        data = result['data']
        if data['d']:
            ds = np.array(data['d'])
            test_accs = np.array([float(x) for x in data['test_acc']])
            unique_ds = sorted(np.unique(ds))
            mean_acc = np.array([np.mean(test_accs[ds == d]) for d in unique_ds])
            above_thresh = np.array(unique_ds)[mean_acc >= accuracy_threshold]
            if len(above_thresh) > 0:
                method_results['Lottery Subspace'] = above_thresh[0]
    
    # Transfer
    dims_t, test_t, _ = extract_accuracy_data(transfer_result)
    mean_t, _ = compute_stats(test_t, dims_t)
    above_thresh = np.array(dims_t)[np.array(mean_t) >= accuracy_threshold]
    if len(above_thresh) > 0:
        method_results['Transfer'] = above_thresh[0]
    
    # Filter to available methods, keep order
    available_methods = []
    available_colors = []
    available_values = []
    for method_name, color in methods_order:
        if method_name in method_results:
            available_methods.append(method_name)
            available_colors.append(color)
            available_values.append(method_results[method_name])
    
    x = np.arange(len(available_methods))
    width = 0.6
    
    ax.bar(x, available_values, width, color=available_colors, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Threshold Training Dimension', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_title(f'Threshold Dimension to Reach {accuracy_threshold*100:.0f}% Test Accuracy\nTinyCNN on {dataset}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(available_methods, rotation=45, ha='right')
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description='Plot transfer experiment results from log files (no pickle needed)')
    parser.add_argument('--data_dir', type=str, default='cross-init-lottery-subspace-data/',
                        help='Directory containing log files')
    parser.add_argument('--transfer_file', type=str, default=None,
                        help='Specific transfer log file')
    parser.add_argument('--baseline_file', type=str, default=None,
                        help='Specific baseline log file')
    parser.add_argument('--original_data_dir', type=str, default='lottery-subspace-data/',
                        help='Directory containing burn-in/lottery log files for comparison plots')
    parser.add_argument('--save_dir', type=str, default='figures/',
                        help='Directory to save figures')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    args = parser.parse_args()
    
    # Find or use specified files
    if args.transfer_file and args.baseline_file:
        transfer_files = [args.transfer_file]
        baseline_files = [args.baseline_file]
    else:
        transfer_files, baseline_files = find_transfer_log_files(args.data_dir)
    
    if not transfer_files:
        print(f"No transfer log files found in {args.data_dir}")
        print("Looking for files matching: *_transfer_*_log.txt")
        return
    
    if not baseline_files:
        print(f"No baseline log files found in {args.data_dir}")
        print("Looking for files matching: *_baseline_*_log.txt")
        return
    
    # Use most recent files
    transfer_file = transfer_files[-1]
    baseline_file = baseline_files[-1]
    
    print(f"Parsing transfer log: {transfer_file}")
    print(f"Parsing baseline log: {baseline_file}")
    
    transfer_result = parse_log_file(transfer_file)
    baseline_result = parse_log_file(baseline_file)
    
    print(f"  Transfer: found {len(transfer_result['data']['d'])} data points")
    print(f"  Baseline: found {len(baseline_result['data']['d'])} data points")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print summary
    print_summary(transfer_result, baseline_result)
    
    # Generate plots
    plot_transfer_comparison(
        transfer_result, baseline_result,
        save_path=os.path.join(args.save_dir, 'fig_transfer_comparison.png')
    )
    
    plot_transfer_gap(
        transfer_result, baseline_result,
        save_path=os.path.join(args.save_dir, 'fig_transfer_gap.png')
    )
    
    plot_individual_runs(
        transfer_result, baseline_result,
        save_path=os.path.join(args.save_dir, 'fig_transfer_individual.png')
    )
    
    # Burn-in effect with transfer
    plot_burn_in_with_transfer(
        transfer_result,
        original_data_dir=args.original_data_dir,
        dataset='MNIST',
        save_path=os.path.join(args.save_dir, 'fig_burn_in_with_transfer.png')
    )
    
    # Combined comparison with all methods
    plot_all_methods_comparison(
        transfer_result, baseline_result,
        original_data_dir=args.original_data_dir,
        dataset='MNIST',
        save_path=os.path.join(args.save_dir, 'fig_all_methods_comparison.png')
    )
    
    # Phase transition with transfer
    plot_phase_transition_with_transfer(
        transfer_result,
        baseline_result,
        original_data_dir=args.original_data_dir,
        dataset='MNIST',
        save_path=os.path.join(args.save_dir, 'fig_phase_transition_with_transfer.png')
    )
    
    # Threshold dimension bar chart with transfer
    plot_threshold_dimension_with_transfer(
        transfer_result,
        original_data_dir=args.original_data_dir,
        dataset='MNIST',
        save_path=os.path.join(args.save_dir, 'fig_threshold_dimension_with_transfer.png')
    )
    
    # Phase transition heatmap with transfer
    plot_phase_transition_heatmap_with_transfer(
        transfer_result,
        baseline_result,
        original_data_dir=args.original_data_dir,
        dataset='MNIST',
        save_path=os.path.join(args.save_dir, 'fig_phase_heatmap_with_transfer.png')
    )
    
    if args.show:
        plt.show()
    
    print("\nDone! Figures saved to:", args.save_dir)


if __name__ == "__main__":
    main()

