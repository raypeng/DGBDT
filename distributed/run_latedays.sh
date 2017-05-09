#!/bin/bash

# Max allowed nodes.
MAX_ALLOWED_NODES=12

# 2 minute time limit.
WALLTIME=4

# Ensure 2 arguments for nodes and processors per node.
if [ $# -ne 1 ]; then
  echo "Usage: $(basename $0) nodes processors_per_node"
  exit $E_BADARGS
fi

# Get command line arguments.
NODES=$1

# Validate arguments.
if [ $NODES -le 0 ] || [ $NODES -gt $MAX_ALLOWED_NODES ]; then
    echo "ERROR: Only $MAX_ALLOWED_NODES nodes allowed."
    exit $E_BADARGS
fi
if [ ! -f "./main" ]; then
    echo "ERROR: ./main program does not exist."
    exit $E_BADARGS
fi

# Submit the job.  No need to modify this.
qsub -l walltime=0:$WALLTIME:00,nodes=$NODES:ppn=24 -F "$NODES" latedays.qsub
