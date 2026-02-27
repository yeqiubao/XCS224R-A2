#!/usr/bin/env python3
"""
Simple script to extract and print TensorBoard data without needing the full tensorboard command
"""
import sys
import os

# Try to import tensorboard components
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError as e:
    print(f"Error importing tensorboard: {e}")
    print("\nTry installing: pip install tensorboard")
    sys.exit(1)

def extract_and_display(logdir, tag='eval/episode_success'):
    """Extract scalar data and display summary"""
    print(f"Loading TensorBoard data from: {logdir}")
    
    try:
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()
        
        if 'scalars' not in ea.Tags():
            print("No scalar data found in TensorBoard logs")
            return
        
        available_tags = ea.Tags()['scalars']
        print(f"\nAvailable tags: {available_tags}")
        
        if tag not in available_tags:
            print(f"\nTag '{tag}' not found!")
            return
        
        scalars = ea.Scalars(tag)
        
        if not scalars:
            print(f"No data found for tag '{tag}'")
            return
        
        print(f"\n{'='*60}")
        print(f"Data for '{tag}':")
        print(f"{'='*60}")
        print(f"{'Step':<12} {'Value':<10} {'Wall Time'}")
        print(f"{'-'*60}")
        
        # Show first 5, last 5, and key points
        for i, scalar in enumerate(scalars):
            if i < 5 or i >= len(scalars) - 5 or scalar.value >= 0.9:
                print(f"{int(scalar.step):<12} {scalar.value:<10.4f} {scalar.wall_time:.2f}")
        
        # Find 90% threshold
        df_90 = [s for s in scalars if s.value >= 0.9]
        if df_90:
            step_90 = min(s.step for s in df_90)
            print(f"\n{'='*60}")
            print(f"✓ 90% success rate reached at step: {int(step_90):,}")
        else:
            print(f"\n{'='*60}")
            print("✗ 90% success rate NOT reached")
        
        print(f"\nTotal data points: {len(scalars)}")
        print(f"Max success rate: {max(s.value for s in scalars):.4f}")
        print(f"Final success rate: {scalars[-1].value:.4f}")
        print(f"Steps range: {int(scalars[0].step):,} - {int(scalars[-1].step):,}")
        
    except Exception as e:
        print(f"Error processing TensorBoard data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    logdir = 'logdir/run_204837_agent.num_critics=2,utd=1/tb'
    
    if len(sys.argv) > 1:
        logdir = sys.argv[1]
    
    if not os.path.exists(logdir):
        print(f"Error: Log directory not found: {logdir}")
        sys.exit(1)
    
    extract_and_display(logdir, tag='eval/episode_success')
