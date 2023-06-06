#!/bin/sh

#SBATCH --job-name="preproc"
#SBATCH -p interactive
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=48G
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorenzo.campoli@unimelb.edu.au

blockMesh
transformPoints -scale '(35.7142857 35.7142857 35.7142857)'
topoSet 
#mapFields -consistent  ../PH_2D_RANS_kOmegaSSTx_baseline/ -sourceTime 'latestTime'
