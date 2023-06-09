#!/bin/sh

#SBATCH --job-name="case_1p0_refined_kOmegaSST"
#SBATCH -p interactive
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorenzo.campoli@unimelb.edu.au

# Load OpenFOAM v2.4
source /usr/local/module/spartan_old.sh
module load OpenFOAM/2.4.0-intel-2017.u2
source $FOAM_BASH

# Decompose case
#decomposePar -force > log.decomposePar

# Launch OpenFOAM solver
#mpirun -np 8 simpleFoam -parallel > log.simpleFoam
simpleFoam > log.simpleFoam

# Reconstruct case
#reconstructPar > log.reconstructPar

# Postprocessing
#sample > log.sample

# Remove multi-processing folders
#rm -r processor*
