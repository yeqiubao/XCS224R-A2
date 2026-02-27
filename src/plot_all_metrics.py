#!/usr/bin/env python3
"""
Plot all training metrics from TensorBoard logs
"""
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import sys
import os
from pathlib import Path

def find_latest_logdir(base_dir='logdir'):
    """Find the latest run directory under logdir"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None
    
    # Find all directories matching run_* pattern
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    
    if not run_dirs:
        return None
    
    # Sort by modification time and get the latest
    latest_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
    tb_dir = latest_dir / 'tb'
    
    if tb_dir.exists():
        return str(tb_dir)
    else:
        # If tb subdirectory doesn't exist, return the run directory itself
        return str(latest_dir)

def extract_tensorboard_data(logdir, tags):
    """Extract scalar data from TensorBoard event file for multiple tags"""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    
    if 'scalars' not in ea.Tags():
        print("No scalar data found in TensorBoard logs")
        return {}
    
    available_tags = ea.Tags()['scalars']
    print(f"Available tags: {available_tags}\n")
    
    data = {}
    for tag in tags:
        if tag in available_tags:
            scalars = ea.Scalars(tag)
            steps = [s.step for s in scalars]
            values = [s.value for s in scalars]
            data[tag] = (steps, values)
            print(f"✓ Found {tag}: {len(steps)} data points")
        else:
            print(f"✗ Tag '{tag}' not found")
            data[tag] = None
    
    return data

def plot_all_metrics(data, output_path=None):
    """Plot all metrics in subplots"""
    
    # Filter out None values
    valid_data = {k: v for k, v in data.items() if v is not None}
    
    if not valid_data:
        print("No valid data to plot!")
        return
    
    # Create subplots - 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Define tags and their display names
    # Based on logger.py: metrics are logged as {ty}/{key}
    # actor metrics: actor/actor_loss, actor/actor_q_min
    # critic metrics: critic/critic_loss, critic/target_q_mean, critic/q_mean
    tags = ['pretrain/bc_loss', 'actor/actor_loss', 'actor/actor_q_min', 
            'critic/critic_loss', 'critic/target_q_mean', 'critic/q_mean']
    display_names = ['BC Loss', 'Actor Loss', 'Actor Q Mean', 
                     'Critic Loss', 'Target Q Mean', 'Q Mean']
    
    for idx, (tag, display_name) in enumerate(zip(tags, display_names)):
        ax = axes[idx]
        
        if tag in valid_data:
            steps, values = valid_data[tag]
            ax.plot(steps, values, linewidth=1.5, label=display_name, color='blue')
            ax.set_title(display_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Environment Steps', fontsize=10)
            ax.set_ylabel(display_name, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis to show steps in thousands
            if len(steps) > 0 and max(steps) > 1000:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
            
            # Print summary
            if len(values) > 0:
                print(f"{display_name}: min={min(values):.4f}, max={max(values):.4f}, final={values[-1]:.4f}")
        else:
            ax.text(0.5, 0.5, f'No data for\n{display_name}\n\n(Training may not have run yet)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title(display_name, fontsize=12)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Metrics (num_critics=2, utd=1)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.savefig('all_metrics_plot.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved to: all_metrics_plot.png")
    
    plt.close()

if __name__ == '__main__':
    # Allow manual override via command line argument
    if len(sys.argv) > 1:
        logdir = sys.argv[1]
    else:
        # Automatically find the latest log directory
        logdir = find_latest_logdir()
        if logdir is None:
            print("Error: No log directories found in 'logdir/'")
            print("Please specify a log directory manually or ensure training has been run.")
            sys.exit(1)
    
    if not os.path.exists(logdir):
        print(f"Error: Log directory not found: {logdir}")
        sys.exit(1)
    
    # Tags to extract
    # Based on logger.py: metrics are logged as {ty}/{key}
    # actor metrics: actor/actor_loss, actor/actor_q_min
    # critic metrics: critic/critic_loss, critic/target_q_mean, critic/q_mean
    tags = [
        'pretrain/bc_loss',
        'actor/actor_loss',
        'actor/actor_q_min',
        'critic/critic_loss',
        'critic/target_q_mean',
        'critic/q_mean'
    ]
    
    print(f"Extracting data from: {logdir}\n")
    data = extract_tensorboard_data(logdir, tags)
    
    if data:
        print("\n" + "="*60)
        plot_all_metrics(data, output_path='all_metrics_plot.png')
    else:
        print("No data extracted!")
        sys.exit(1)
