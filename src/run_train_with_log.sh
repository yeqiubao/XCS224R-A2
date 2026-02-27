#!/bin/bash
# Script to run training and save all output to a log file (no terminal output)

# Get the current timestamp for unique log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="training_log_${TIMESTAMP}.txt"

# Default arguments
ARGS="agent.num_critics=2 utd=1"

# Allow command line arguments to override defaults
if [ $# -gt 0 ]; then
    ARGS="$@"
fi

echo "Starting training..."
echo "Arguments: $ARGS"
echo "Log file: $LOG_FILE"
echo "Output will be saved to log file only (not shown in terminal)"
echo "========================================"

# Run training and save all output (stdout and stderr) to log file only
# Redirect both stdout and stderr to the log file
python train.py $ARGS > "$LOG_FILE" 2>&1

# Get exit status
EXIT_STATUS=$?

echo ""
echo "========================================"
echo "Training completed with exit status: $EXIT_STATUS"
echo "All output saved to: $LOG_FILE"
echo "You can view the log with: tail -f $LOG_FILE"

exit $EXIT_STATUS
