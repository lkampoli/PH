"""EVE3 training on LES data of periodic hill flow at Re=10,595."""

# General imports
import numpy as np
import subprocess
import os

# Import models
from eve3_model import Model as PHModel

# Define parameters ranges
pop_sizes = [100]
pfacs = [1.5]
seeds = [0, 1, 2]
algos = [1]

# Define parameters
ngens = 250
tsize = 2
msize_fac = 0.5
res_gens = [-1]
res_vars = ['gen', 'minFit', 'best_run_id', 'pop_size', 'tsize', 'msize', 'pmut', 'ponec', 'ptwoc', 'pinv', 'pblo', 'pgswap', 'piswap', 'seed']
save_gens = [-1]
restart = 0
restart_file = None
quiet = True
num_proc = None
num_thr = 12

# Standard probabilites for genetic operations
pmut = 0.2
ponec = 0.3
ptwoc = 0.3
pinv = 0.0
pblo = 0.0
evo_pressure = 1

# Start EVE3 runs
for pop_size in pop_sizes:
    for pfac in pfacs:
        for seed in seeds:
            for algo in algos:

                # Create run folder
                run_folder = ("running_pfac%.2f_pop%d_seed%d" % (pfac, pop_size, seed)).replace('.','')
                if restart == 0:
                    subprocess.call(["cp", "-r", "running_template", run_folder])

                # Define save and result file
                save_file = "./" + run_folder + "/opt_dat.pkl"
                res_file = "./study_log_data_algo" + str(algo) + ".csv"

                # Train PH model
                model = PHModel(
                    pmut=np.round(pfac*pmut,3), 
                    ponec=np.round(pfac*ponec,3), 
                    ptwoc=np.round(pfac*ptwoc,3), 
                    pinv=np.round(pfac*pinv,3), 
                    pblo=np.round(pfac*pblo,3), 
                    evo_pressure=evo_pressure, 
                    pop_size=pop_size, 
                    seed=seed, 
                    ext_eval_path=run_folder, 
                    res_file=res_file, 
                    res_gens=res_gens, 
                    res_vars=res_vars,
                    save_file=save_file, 
                    save_gens=save_gens, 
                    restart=restart, 
                    restart_file=restart_file, 
                    tsize=tsize, 
                    msize=int(msize_fac * pop_size), 
                    ngens=ngens, 
                    quiet=quiet, 
                    algorithm_choice=algo,
                    num_proc=num_proc,
                    num_thr=num_thr,
                    )
                model.train()

                # Reset restart flags
                restart = 0
                restart_file = None

# Finish
print("The End.")
            

