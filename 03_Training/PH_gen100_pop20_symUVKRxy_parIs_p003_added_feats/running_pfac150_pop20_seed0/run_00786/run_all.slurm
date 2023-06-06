#!/bin/bash

#SBATCH --job-name="PH_gen100_pop20_symUVKRxy_parIs_p003_added_feats_child"
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0-00:30:00

# Load required modules
module load gcccore/10.2.0 python/3.8.6
source ~/virtualenv/python3.8.6/bin/activate

# Launch multiple process python code
echo "Running"
python ./run_all.py
