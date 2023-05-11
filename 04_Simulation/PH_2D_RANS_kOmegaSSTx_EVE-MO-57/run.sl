#!/bin/bash

#SBATCH --job-name="PH_2D_RANS_kOmegaSSTx_EVE-MO-57"
#SBATCH -p physical
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorenzo.campoli@unimelb.edu.au

source /usr/local/module/spartan_old.sh
module load OpenFOAM/2.4.0-intel-2017.u2
source $FOAM_BASH

##decomposePar > log.decomposePar
##srun --export=all -n 4 simpleFoam -parallel > log.simpleFoam
##reconstructPar > log.reconstructPar

##simpleFoam | tee simpleFoam.log
##mpiexec -n 1 sample -latestTime
./Allrun
