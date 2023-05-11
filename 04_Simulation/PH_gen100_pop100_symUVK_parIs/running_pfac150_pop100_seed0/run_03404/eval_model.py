# Imports
import os
import subprocess

# Settings
slurm_file = "run_all.slurm"

# Get run folder
run_folder = os.path.dirname(os.path.abspath(__file__))

# Submit job
subprocess.call(["sbatch", "--wait", "--chdir", run_folder, os.path.join(run_folder, slurm_file)])
