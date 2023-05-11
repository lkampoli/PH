#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00

# Load required modules
module load gcc/8.3.0; module load scipy-bundle/2019.10-python-3.7.4
source ~/virtualenv/py37/bin/activate

# Launch multiple process python code
echo "Running"
python ./eve3_study.py
