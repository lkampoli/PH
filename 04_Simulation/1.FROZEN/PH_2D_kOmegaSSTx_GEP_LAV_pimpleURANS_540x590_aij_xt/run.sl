#!/bin/sh

#SBATCH --job-name="PH_2D_kOmegaSSTx_GEP_LAV_pimpleURANS_540x590_aij_xt"
#SBATCH -p physical
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorenzo.campoli@unimelb.edu.au

./pimpleAllrun
