#!/bin/bash

#SBATCH --job-name="PH_gen100_pop12_symUVKRxy_parIs_child_clusters_012"
#SBATCH --nodes=1
#SBATCH --ntasks=1 # 4
#SBATCH --time=0-00:30:00
#SBATCH --mem=2G

# Load required modules
module load gcccore/10.2.0 python/3.8.6
source ~/virtualenv/python3.8.6/bin/activate

# Launch multiple process python code
echo "Running"
python ./run_all.py
