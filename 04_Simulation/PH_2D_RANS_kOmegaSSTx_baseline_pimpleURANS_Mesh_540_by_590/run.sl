#!/bin/sh

#SBATCH --job-name="kOmegaSST_BSL_pimpleURANS_Mesh_540_by_590"
#SBATCH -p interactive
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorenzo.campoli@unimelb.edu.au

./pimpleAllrun
