#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=0-00:30:00

# Load required modules
module load gcc/8.3.0; module load scipy-bundle/2019.10-python-3.7.4
source ~/virtualenv/py37/bin/activate

# Launch multiple process python code
echo "Running"
python ./run_all.py
