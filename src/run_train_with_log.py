#!/usr/bin/env python3
"""
Python script to run training and save all output to a log file
"""
import sys
import subprocess
import datetime
import os

def main():
    # Get timestamp for unique log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"training_log_{timestamp}.txt"
    
    # Default arguments
    args = ["agent.num_critics=2", "utd=1"]
    
    # Allow command line arguments to override defaults
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    
    # Build command
    cmd = ["python", "train.py"] + args
    
    print("Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    print("Output will be saved to log file only (not shown in terminal)")
    print("=" * 60)
    
    # Open log file for writing
    with open(log_file, 'w', encoding='utf-8') as log:
        # Write header to log file
        log.write(f"Training Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Command: {' '.join(cmd)}\n")
        log.write("=" * 60 + "\n\n")
        log.flush()
        
        # Run training and capture output
        # Redirect all output to log file only (no terminal output)
        process = subprocess.Popen(
            cmd,
            stdout=log,  # Direct stdout to log file
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            universal_newlines=True,
            bufsize=1,  # Line buffered
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Wait for process to complete
        try:
            exit_status = process.wait()
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            log.write("\n\nTraining interrupted by user\n")
            process.terminate()
            exit_status = process.wait()
    
    print("\n" + "=" * 60)
    print(f"Training completed with exit status: {exit_status}")
    print(f"All output saved to: {log_file}")
    print(f"You can view the log with: tail -f {log_file}")
    
    return exit_status

if __name__ == "__main__":
    sys.exit(main())
