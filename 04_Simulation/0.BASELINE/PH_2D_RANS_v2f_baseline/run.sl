#!/bin/sh

#SBATCH --job-name="kEpsilonPhitF_BSL"
#SBATCH -p interactive
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorenzo.campoli@unimelb.edu.au

./Allrun
