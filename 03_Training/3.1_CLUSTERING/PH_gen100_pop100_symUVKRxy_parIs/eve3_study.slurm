#!/bin/bash

#SBATCH --job-name="PH_gen100_pop12_symUVKRxy_parIs_cluster012"
#SBATCH -p interactive #physical
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorenzo.campoli@unimelb.edu.au

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=2-00:00:00

#SBATCH --mem=2G

module load gcccore/10.2.0 python/3.8.6
source ~/virtualenv/python3.8.6/bin/activate

echo "Running"
python ./eve3_study.py
