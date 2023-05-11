#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=0-00:30:00

module load gcccore/10.2.0 python/3.8.6
source ~/virtualenv/python3.8.6/bin/activate

# Launch multiple process python code
echo "Running"
python ./run_all.py
