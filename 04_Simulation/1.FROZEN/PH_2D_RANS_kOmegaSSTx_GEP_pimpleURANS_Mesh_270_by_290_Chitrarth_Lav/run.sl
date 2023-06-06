#!/bin/sh

#SBATCH --job-name="kOmegaSST_GEP_pimpleURANS_Mesh_270_by_290_Chitrarth_Lav"
#SBATCH -p interactive
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorenzo.campoli@unimelb.edu.au

./pimpleAllrun
