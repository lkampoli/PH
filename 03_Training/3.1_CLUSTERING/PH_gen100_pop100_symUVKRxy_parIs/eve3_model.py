"""EVE3 training on LES data of periodic hill flow at Re=10,595."""

# General imports
import numpy as np
import argparse
import os
import joblib
import subprocess
from collections.abc import Iterable

# Import EVE3 packages
import ea
import we
import ns

# Default options values

# Path to training data (called frozens in CWorld)
data_path = "./training_data"

# Path to running folder for external evaluation
ext_eval_path = "./running"

# Number of generations and convergence criterion
ngens = 50
stopping_tolerance = 1e-10

# Population size and natural selection options
pop_size = 30
tsize = 2
msize = 10
                   
# Seed for random function
seed = 0

# Probabilites for genetic operations
pmut = 0.2
ponec = 0.3
ptwoc = 0.3
pinv = 0.0
pblo = 0.0
pgswap = 0.0
piswap = 0.0
evo_pressure = 0

# Output options
res_file = None
res_vars = ['gen', 'minFit', 'pop_size', 'tsize', 'msize', 'pmut', 'ponec', 'ptwoc', 'pinv', 'pblo', 'pgswap', 'piswap', 'seed']
res_gens = [0]
quiet = False
vquiet = False

# Restart options
restart = 0
restart_file = None
save_file = None
save_gens = [0]

# Multiprocessing/threading options
num_proc = None
num_thr = None

################
num_clusters = 3
################

# Number of genes per expression 
ngenes = [4]*6*num_clusters

# Algorithm switch: 1 for GEP, 2 for DG
algorithm_choice = 1

# GEP specific options
head_length = [5]*6*num_clusters

# DG specific options
max_length = [12]*6*num_clusters

class Model(object):
    """EVE3 model."""
    
    def __init__(self, **kwargs):
        """Initializes EVE3 model."""

        # Set all attributes     
        try:
            self.data_path = kwargs['data_path']
        except:
            self.data_path = data_path
        
        try:
            self.ext_eval_path = kwargs['ext_eval_path']
        except:
            self.ext_eval_path = ext_eval_path
        
        try:
            self.ngens = kwargs['ngens']
        except:
            self.ngens = ngens
        
        try:
            self.eps = kwargs['eps']
        except:
            self.eps = stopping_tolerance
        
        try:
            self.pop_size = kwargs['pop_size']
        except:
            self.pop_size = pop_size
        
        try:
            self.tsize = kwargs['tsize']
        except:
            self.tsize = tsize
        
        try:
            self.msize = kwargs['msize']
        except:
            self.msize = msize
        
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = seed
        
        try:
            self.pmut = kwargs['pmut']
        except:
            self.pmut = pmut
        
        try:
            self.ponec = kwargs['ponec']
        except:
            self.ponec = ponec
        
        try:
            self.ptwoc = kwargs['ptwoc']
        except:
            self.ptwoc = ptwoc
        
        try:
            self.pinv = kwargs['pinv']
        except:
            self.pinv = pinv
        
        try:
            self.pblo = kwargs['pblo']
        except:
            self.pblo = pblo

        try:
            self.piswap = kwargs['piswap']
        except:
            self.piswap = piswap

        try:
            self.pgswap = kwargs['pgswap']
        except:
            self.pgswap = pgswap

        try:
            self.evo_pressure = kwargs['evo_pressure']
        except:
            self.evo_pressure = evo_pressure
        
        try:
            self.res_file = kwargs['res_file']
        except:
            self.res_file = res_file
        
        try:
            self.res_vars = kwargs['res_vars']
        except:
            self.res_vars = res_vars
        
        try:
            self.res_gens = kwargs['res_gens']
        except:
            self.res_gens = res_gens

        try:
            self.restart = kwargs['restart']
        except:
            self.restart = restart

        try:
            self.algorithm_choice = kwargs['algorithm_choice']
        except:
            self.algorithm_choice = algorithm_choice

        try:
            self.restart_file = kwargs['restart_file']
        except:
            self.restart_file = restart_file

        try:
            self.save_file = kwargs['save_file']
        except:
            self.save_file = save_file

        try:
            self.save_gens = kwargs['save_gens']
        except:
            self.save_gens = save_gens

        try:
            self.num_proc = kwargs['num_proc']
        except:
            self.num_proc = num_proc

        try:
            self.num_thr = kwargs['num_thr']
        except:
            self.num_thr = num_thr
        
        try:
            self.head_length = kwargs['head_length']
        except:
            self.head_length = head_length

        try:
            self.ngenes = kwargs['ngenes']
        except:
            self.ngenes = ngenes

        try:
            self.quiet = kwargs['quiet']
        except:
            self.quiet = quiet

        try:
            self.vquiet = kwargs['vquiet']
        except:
            self.vquiet = vquiet

        # Ensure quietness
        if self.vquiet:
            self.quiet = True


    def train(self):
        """Trains EVE3 model."""

        # Check restart
        if self.restart != 1:

            # Define cost function
            #cost_fun1 = "sum(1.0/DATALEN/1.0*(U1-U0)**2)"
            #cost_fun2 = "sum(1.0/DATALEN/1.0*(V1-V0)**2)"
            #cost_fun3 = "sum(1.0/DATALEN/1.0*(K1-K0)**2)"
            #cost_fun4 = "sum(1.0/DATALEN/1.0*(Rxy1-Rxy0)**2)"
            #cost_fun = [cost_fun1, cost_fun2, cost_fun3, cost_fun4]
            
            #cost_fun1_cluster0 = "sum(1.0/DATALEN/1.0*(U01-U00)**2)"
            #cost_fun2_cluster0 = "sum(1.0/DATALEN/1.0*(V01-V00)**2)"
            #cost_fun3_cluster0 = "sum(1.0/DATALEN/1.0*(K01-K00)**2)"
            #cost_fun4_cluster0 = "sum(1.0/DATALEN/1.0*(Rxy01-Rxy00)**2)"
            
            #cost_fun1_cluster1 = "sum(1.0/DATALEN/1.0*(U11-U10)**2)"
            #cost_fun2_cluster1 = "sum(1.0/DATALEN/1.0*(V11-V10)**2)"
            #cost_fun3_cluster1 = "sum(1.0/DATALEN/1.0*(K11-K10)**2)"
            #cost_fun4_cluster1 = "sum(1.0/DATALEN/1.0*(Rxy11-Rxy10)**2)"

            #cost_fun1_cluster2 = "sum(1.0/DATALEN/1.0*(U21-U20)**2)"
            #cost_fun2_cluster2 = "sum(1.0/DATALEN/1.0*(V21-V20)**2)"
            #cost_fun3_cluster2 = "sum(1.0/DATALEN/1.0*(K21-K20)**2)"
            #cost_fun4_cluster2 = "sum(1.0/DATALEN/1.0*(Rxy21-Rxy20)**2)"
            
            #cost_fun1 = cost_fun1_cluster0 + cost_fun1_cluster1 + cost_fun1_cluster2
            #cost_fun2 = cost_fun2_cluster0 + cost_fun2_cluster1 + cost_fun2_cluster2
            #cost_fun3 = cost_fun3_cluster0 + cost_fun3_cluster1 + cost_fun3_cluster2
            #cost_fun4 = cost_fun4_cluster0 + cost_fun4_cluster1 + cost_fun4_cluster2

            cost_fun1 = "sum(1.0/DATALEN/1.0*(U01-U00)**2) + sum(1.0/DATALEN/1.0*(U11-U10)**2) + sum(1.0/DATALEN/1.0*(U21-U20)**2)"
            cost_fun2 = "sum(1.0/DATALEN/1.0*(V01-V00)**2) + sum(1.0/DATALEN/1.0*(V11-V10)**2) + sum(1.0/DATALEN/1.0*(V21-V20)**2)"
            cost_fun3 = "sum(1.0/DATALEN/1.0*(K01-K00)**2) + sum(1.0/DATALEN/1.0*(K11-K10)**2) + sum(1.0/DATALEN/1.0*(K21-K20)**2)"
            cost_fun4 = "sum(1.0/DATALEN/1.0*(Rxy01-Rxy00)**2) + sum(1.0/DATALEN/1.0*(Rxy11-Rxy10)**2) + sum(1.0/DATALEN/1.0*(Rxy21-Rxy20)**2)"

            cost_fun = [cost_fun1, cost_fun2, cost_fun3, cost_fun4]

            # Check restart option
            if self.restart != 2:

                # Set starting generation
                start_gen = 1

                # Set random seed
                np.random.seed(self.seed)

                ### Model definition

                # Number of trained expressions
                nexpr = 6*num_clusters

                # Evaluation via numexpr or eval()
                numexpr = True

                # Define symbols
                # Frozens
                U00sym   = ea.EvalSymbol("U00",   0)
                V00sym   = ea.EvalSymbol("V00",   0)
                K00sym   = ea.EvalSymbol("K00",   0)
                Rxy00sym = ea.EvalSymbol("Rxy00", 0)
                U10sym   = ea.EvalSymbol("U10",   0)
                V10sym   = ea.EvalSymbol("V10",   0)
                K10sym   = ea.EvalSymbol("K10",   0)
                Rxy10sym = ea.EvalSymbol("Rxy10", 0)
                U20sym   = ea.EvalSymbol("U20",   0)
                V20sym   = ea.EvalSymbol("V20",   0)
                K20sym   = ea.EvalSymbol("K20",   0)
                Rxy20sym = ea.EvalSymbol("Rxy20", 0)

                # Actives
                U01sym   = ea.EvalSymbol("U01",   0)
                V01sym   = ea.EvalSymbol("V01",   0)
                K01sym   = ea.EvalSymbol("K01",   0)
                Rxy01sym = ea.EvalSymbol("Rxy01", 0)
                U11sym   = ea.EvalSymbol("U11",   0)
                V11sym   = ea.EvalSymbol("V11",   0)
                K11sym   = ea.EvalSymbol("K11",   0)
                Rxy11sym = ea.EvalSymbol("Rxy11", 0)
                U21sym   = ea.EvalSymbol("U21",   0)
                V21sym   = ea.EvalSymbol("V21",   0)
                K21sym   = ea.EvalSymbol("K21",   0)
                Rxy21sym = ea.EvalSymbol("Rxy21", 0)

                # Parameters
                I01sym = ea.EvalSymbol("I01", 0)
                I02sym = ea.EvalSymbol("I02", 0)

                # Math symbols
                times = ea.Symbol("*", 2)
                div = ea.Symbol("/", 2)
                plus = ea.Symbol("+", 2)
                minus = ea.Symbol("-", 2)
                e = ea.Symbol("exp", 1, nice_name="e")
                log = ea.Symbol("log", 1, nice_name="log")
                tanh = ea.Symbol("tanh", 1, nice_name="tanh")
                sqrt = ea.Symbol("sqrt", 1, nice_name="sqrt")

                # Numbers
                two = ea.Symbol("2.0", 0)
                one = ea.Symbol("1.0", 0)
                zero = ea.Symbol("0.0", 0)
                mone = ea.Symbol("-1.0", 0)

                # Random numbers
                rmin_ = -1
                rmax_ = 1
                num_rnc = 5
                rncs = np.random.uniform(rmin_, rmax_, num_rnc)
                rnc_syms = [ea.Symbol(str(rnc), 0, nice_name=str(rnc)[0:5]) for rnc in rncs]

                # Grouping of symbols

                # Arity 0 symbols
                terms_exp1 = [I01sym, I02sym, one, two, mone] + rnc_syms

                # Arity 0 weights
                term_exp1_weights = list(0.6/2*np.ones(2)) + \
                                    list(0.2/3*np.ones(3)) + \
                                    list(0.2/num_rnc*np.ones(num_rnc))

                # All symbols (separated with respect to arity)
                syms_exp1 = [terms_exp1, [], [times, plus, minus]]

                # Probabilities and weights for different arities
                pterms = [0.5]*nexpr
                pones = [0.0]*nexpr
                ptwos = [0.5]*nexpr
                psyms = [pterms, pones, ptwos]
                sym_ar2_weights = [1./3, 1./3, 1./3]

                # All weights
                sym_exp1_weights = [term_exp1_weights, [], sym_ar2_weights]

                # Link symbols
                link_syms = [plus, minus, times]

                # Blocking symbols
                block_syms = []

                # Symbols to load
                frozens = [U00sym, V00sym, K00sym, Rxy00sym, U10sym, V10sym, K10sym, Rxy10sym, U20sym, V20sym, K20sym, Rxy20sym]
                actives = [U01sym, V01sym, K01sym, Rxy01sym, U11sym, V11sym, K11sym, Rxy11sym, U21sym, V21sym, K21sym, Rxy21sym]

                # Additional kwargs dict for indi creation
                indi_kwargs_exp1 = {}

                # Summarize for all expressions
                terms = [terms_exp1]*nexpr
                syms = [syms_exp1]*nexpr
                sym_weights = [sym_exp1_weights]*nexpr
                links = [link_syms]*nexpr
                blockers = [block_syms]*nexpr
                seeds = [self.seed]*nexpr
                indi_kwargs = [indi_kwargs_exp1]*nexpr

                ### Automated from here on
            
                # Model creation
                world = we.CWorld(
                    frozens=frozens,
                    actives=actives,
                    datapath_frozens=self.data_path,
                    datapath_actives=self.ext_eval_path,
                    quiet=self.quiet,
                    comp=True,
                    numexpr=numexpr,
                )

                mofit = ns.CFitPack
                mofit.set_cost_fun(cost_fun)
                if mofit.get_ncost() > 1:
                    pareto_hist = {}

                # GEP algorithm
                if self.algorithm_choice == 1:

                    # Convert symbol and weight format
                    for i in range(nexpr):
                        syms[i] = [sym for arlist in syms[i] for sym in arlist]
                        sym_weights[i] = [sym_weights[i][j][k] * psyms[j][i] for j in range(len(sym_weights[i])) for k in range(len(sym_weights[i][j]))]

                    # Create factory
                    factory = ea.CFactory(
                        # input for MEFactory
                        ftypes=ea.GEPFactory,                               # define a factory for each expression or if only one stated, use for all expressions
                        nexpr=nexpr,                                        # save number of expressions, might be useful -> give it to world and population as well?
                        world=world,                                        # refence to the world individuals will belong to
                        fitpack=mofit,                                      # optimisation method

                        # input for sub-factories: define everything that shall be passed to individual factories
                        symbols=syms,                                       # list of symbols for each trained expression
                        sweights=sym_weights,                               # weights of each symbol
                        terms=terms,                                        # arity 0's
                        links=links,                                        # functions to link genes
                        indi_kwargs=indi_kwargs,                            # additional kwargs for indi creation
                        blockers=blockers,                                  # include blockers
                        head=self.head_length,                              # length of head for each expression
                        ngenes=self.ngenes,                                 # number of genes for each expression
                        seed=seeds,                                         # seed needs to be passed to all namespaces
                    )

                # DG algorithm
                elif self.algorithm_choice == 2:

                    factory = ea.CFactory(
                        # input for MEFactory
                        ftypes=ea.DGFactory,                                # define a factory for each expression or if only one stated, use for all expressions
                        nexpr=nexpr,                                        # save number of expressions, might be useful -> give it to world and population as well?
                        world=world,                                        # refence to the world individuals will belong too
                        fitpack=mofit,                                      # optimisation method

                        # input for sub-factories: define everything that shall be passed to individual factories
                        symbols=syms,                                       # list of symbols for each trained expression
                        sweights=sym_weights,                               # weights of each symbol
                        pterm=pterms,                                       # probability of selecting arity 0 symbol
                        pone=pones,                                         # probability of selecting arity 1 symbol
                        ptwo=ptwos,                                         # probability of selecting arity 0 symbol
                        length=max_length,                                  # maximum genotype length
                        itype=[ea.TreeIndividual]*nexpr,                    # type of individual
                        indi_kwargs=indi_kwargs,                            # additional kwargs for indi creation
                        links=links,                                        # functions to link genes
                        blockers=blockers,                                  # include blockers
                        ngenes=ngenes,                                      # number of genes for each expression
                        seed=seeds,                                         # seed needs to be passed to all namespaces
                    )

                go = ns.CGos(rseed=self.seed, quiet=self.quiet)

                natural_selector = ns.EliteTournamentSelection(
                    tsize = self.tsize,
                    msize = self.msize,
                    rseed = self.seed,
                    quiet = self.quiet
                )

                # Check restart option one last time
                if self.restart != 3:
                
                    pop = ea.CPopulation(
                        n_colos=self.pop_size,
                        factory=factory,
                        rseed=self.seed,
                        quiet=self.quiet
                    )
                    pop.build(self.pop_size)

                    # Add referee
                    referee = ns.Referee(pop, drop_dup=True)
                    mofit.set_ref(referee)

        # Restart settings
        # Options:  1 - Full restart: all required information are taken from restart file
        #           2 - Adapted restart: load objects and RNG state only
        #           3 - Adapted restart: load population and referee only
        if self.restart:

            # Load optimization status from restart file
            restart_state = joblib.load(self.restart_file)

            # Full restart
            if self.restart == 1:

                # Get referee
                referee = restart_state['referee']

                # Build population
                pop = referee._pop

                # Estabilish factory, world and fitpack
                factory = pop.fac
                world = pop.fac._world
                mofit = pop.fac._fitpack

                # Read external evaluation data after last save
                if world._actives is not None:
                    run_folders = os.listdir(world._datapath_actives)
                    run_id = str(world._run_index + 1).zfill(5)
                    run_folder = "run_" + run_id
                    
                    while run_folder in run_folders:
                        
                        # Read expressions
                        f = open(os.path.join(world._datapath_actives, run_folder, "input_" + run_id), 'r')
                        run_expressions = f.readlines()
                        run_expressions = [expr.replace('\n','') for expr in run_expressions]
                        f.close()
                        
                        # Create new register entry
                        world._register['<&&&>'.join(run_expressions)] = run_id
                        world.actives_db[run_id] = {}

                        # Load external evaluation data
                        for sym in world._actives:
                            world.actives_db[run_id][sym.nice_name] = world.load_data(sym, datapath=os.path.join(world._datapath_actives, run_folder, "output"))
                        
                        # Increase run index
                        world._run_index += 1
                        run_id = str(world._run_index + 1).zfill(5)
                        run_folder = "run_" + run_id

                # Initialize natural selector and genetic operators
                natural_selector = restart_state['natural_selector']
                go = restart_state['go']

                # Set optimization state
                start_gen = restart_state['gen'] + 1
                self.ngens = restart_state['ngens']
                self.eps = restart_state['stopping_tolerance']
                cost_fun = restart_state['cost_fun']
                self.pmut = restart_state['pmut']
                self.ponec = restart_state['pone']
                self.ptwoc = restart_state['ptwo']
                self.pinv = restart_state['pinv']
                self.pblo = restart_state['pblo']
                self.pgswap = restart_state['pgswap']
                self.piswap = restart_state['piswap']
                self.evo_pressure = restart_state['evo_pressure']

                # Set cost function and link database
                mofit.set_cost_fun(cost_fun)
                mofit.set_ref(referee)
                if mofit.get_ncost() > 1:
                    pareto_hist = restart_state['pareto_hist']

                # Set state of random number generator
                rng_state = restart_state['rng_state'] 
                np.random.set_state(rng_state)

            # Load objects
            elif self.restart == 2:
                
                # Get referee
                referee = restart_state['referee']

                # Build population
                pop = referee._pop

                # Estabilish factory, world and fitpack
                factory = pop.fac
                world = pop.fac._world
                mofit = pop.fac._fitpack

                # Initialize natural selector and genetic operators
                natural_selector = restart_state['natural_selector']
                go = restart_state['go']

                # Set cost function and link referee
                mofit.set_cost_fun(cost_fun)
                mofit.set_ref(referee)
                if mofit.get_ncost() > 1:
                    pareto_hist = restart_state['pareto_hist']

                # Set start generation number
                start_gen = restart_state['gen'] + 1

                # Set state of random number generator
                rng_state = restart_state['rng_state'] 
                np.random.set_state(rng_state)

            # Load population
            elif self.restart == 3:

                # Get referee
                referee = restart_state['referee']

                # Build population
                pop = referee._pop  

                # Map correct objects
                pop.fac = factory

                for i in range(len(pop)):
                    pop[i].fitpack._world = world

        # Calculate initial fitness values  
        pop.fitness(num_proc=self.num_proc, num_thr=self.num_thr)
        
        # Calculate stats
        fits = [np.mean(p.fitness(cost_id=range(mofit.get_ncost()))) for p in pop]
        minFit = np.nanmin(fits)
        minIndex = fits.index(minFit)
        
        if start_gen == 1 and mofit.get_ncost() > 1:
            pareto_hist[0] = mofit.get_ref().db[mofit.get_ref().db.rnk < 2].loc[:,'rnk':'fit%s' % str(mofit.get_ncost()-1).zfill(2)].to_numpy()


        # Save optimization
        if self.save_file and (start_gen in self.save_gens):

            # Save objects
            save_state = {}
            save_state['referee'] = referee
            save_state['go'] = go
            save_state['natural_selector'] = natural_selector

            # Save status of optimization run
            save_state['rng_state'] = np.random.get_state()
            save_state['gen'] = start_gen - 1
            save_state['ngens'] = self.ngens
            save_state['stopping_tolerance'] = self.eps
            save_state['cost_fun'] = cost_fun
            save_state['pmut'] = self.pmut
            save_state['pone'] = self.ponec
            save_state['ptwo'] = self.ptwoc
            save_state['pinv'] = self.pinv
            save_state['pblo'] = self.pblo
            save_state['pgswap'] = self.pgswap
            save_state['piswap'] = self.piswap
            save_state['evo_pressure'] = self.evo_pressure
            if mofit.get_ncost() > 1:
                save_state['pareto_hist'] = pareto_hist

            # Save all information to pickle file
            joblib.dump(save_state, self.save_file)

            # Add-on: Reduce number of files on cluster drive
            subprocess.call(["./zip_running.sh"])

        # Initialize tracking of change
        gens_unchanged = 0
        minFit_prev = minFit

        # Start optimization from generation 1
        for gen in range(start_gen, self.ngens+1):
            
            # Report generation
            if not self.vquiet:
                print(("Generation %s" % gen))
        
            # Natural selection
            natural_selector.compete(pop, world)
            
            # Genetic operations including evoluationary pressure
            if self.evo_pressure:

                if self.ngens - (gen - gens_unchanged) != 0:
                    gen_frac = float(gens_unchanged)/(self.ngens - (gen - gens_unchanged))
                else:
                    gen_frac = 0

                ppmut = self.pmut
                pponec = self.ponec
                pptwoc = self.ptwoc
                ppinv = self.pinv
                ppblo = self.pblo
                ppgswap = self.pgswap
                ppiswap = self.piswap

                if not self.vquiet:
                    print(("Generations without improvement: %d (%.2f)" % (gens_unchanged,gen_frac)))

                if gen_frac > 0.3:
                    p_frac = (gen_frac - 0.3) / (0.9 - 0.3)

                    if self.pmut:
                        ppmut = min(1.0, self.pmut + p_frac * (1.0 - self.pmut))

                    if self.ponec:
                        pponec = min(1.0, self.ponec + p_frac * (1.0 - self.ponec))

                    if self.ptwoc:
                        pptwoc = min(1.0, self.ptwoc + p_frac * (1.0 - self.ptwoc))

                    if self.pinv:
                        ppinv = min(1.0, self.pinv + p_frac * (1.0 - self.pinv))

                    if self.pblo:
                        ppblo = min(1.0, self.pblo + p_frac * (1.0 - self.pblo))

                    if self.pgswap:
                        ppgswap = min(1.0, self.pgswap + p_frac * (1.0 - self.pgswap))

                    if self.piswap:
                        ppiswap = min(1.0, self.piswap + p_frac * (1.0 - self.piswap))

                if not self.vquiet:
                    print(("Used probs: %f, %f, %f, %f, %f, %f, %f" % (ppmut, pponec, pptwoc, ppinv, ppblo, ppgswap, ppiswap)))
                pop.update(go, pmut=ppmut, pone=pponec, ptwo=pptwoc, pinv=ppinv, pblo=ppblo, pgswap=ppgswap, piswap=ppiswap)
            else:
                pop.update(go, pmut=self.pmut, pone=self.ponec, ptwo=self.ptwoc, pinv=self.pinv, pblo=self.pblo, pgswap=self.pgswap, piswap=self.piswap)

            # Rank population
            pop.fitness(num_proc=self.num_proc, num_thr=self.num_thr)

            # Merge population
            pop.merge(drop_dup=True)

            # Calculate stats
            fits = [np.mean(p.fitness(cost_id=range(mofit.get_ncost()))) for p in pop]
            minFit = np.nanmin(fits)
            minIndex = fits.index(minFit)
            if mofit.get_ncost() > 1:
                pareto_hist[gen] = mofit.get_ref().db[mofit.get_ref().db.rnk < 2].loc[:,'rnk':'fit%s' % str(mofit.get_ncost()-1).zfill(2)].to_numpy()
            
            # Track generations without improvement
            if not minFit_prev == minFit:
                gens_unchanged = 0
                minFit_prev = minFit
            else:
                gens_unchanged += 1
            
            # Calculate stats
            sumFit = np.nansum(fits)
            stdFit = np.nanstd(fits)
            meanFit = np.nanmean(fits)
            best_run_id = world.get_run_id(pop[minIndex])

            # Report stats
            if not self.vquiet:
                print(("Best colony has index %d and fitness value of %f" % (minIndex, minFit)))
                for i in range(factory._nexpr):
                    print(("Expression: %d \t%s" % (i, pop[minIndex].print_phenotype(nice=True)[i])))
                print("Stats of entire population")
                print(("Fitness value sum: \t" + str(sumFit)))
                print(("Fitness value mean: \t" +str(meanFit)))
                print(("Fitness value std: \t" +str(stdFit)))
                print(("Number of NaN values: \t%d" % sum([np.isnan(fit) for fit in fits])))
                if best_run_id is not None:
                    print(("Best colony run ID: %s" % best_run_id))
                if mofit.get_ref() is not None:
                    print(("Unique Pareto front colonies: %d" % np.sum(~mofit.get_ref().db[mofit.get_ref().db.rnk < 2].loc[:,'fit00':'fit%s' % str(mofit.get_ncost()-1).zfill(2)].duplicated())))
                print('\n')

            # Write result file
            if self.res_file and ((gen in self.res_gens or gen==1 or gen==self.ngens) or ((-1 in self.res_gens) and gens_unchanged==0)):

                # Build header
                header = ','.join(self.res_vars) + '\n'

                # Fetch variable values
                res_string = ''
                for var in self.res_vars:
                    if var in self.__dict__:
                        res_string += str(self.__dict__[var]) + ','
                    else:
                        res_string += str(eval(var)) + ','
                res_string = res_string[:-1] + '\n'

                # Check if result file already exists
                if os.path.exists(self.res_file):

                    # Check if output variables align
                    with open(self.res_file, 'r') as f:
                        first_line = f.readline()

                    if not first_line == header:
                        raise RuntimeError("ERROR: output variables do not match")

                    with open(self.res_file, 'a') as f:
                        f.write(res_string)

                else:
                    with open(self.res_file, 'w') as f:
                        f.write(header)
                        f.write(res_string)
            
            # Save optimization
            if self.save_file and ((gen in self.save_gens or gen==1 or gen==self.ngens) or ((-1 in self.save_gens) and gens_unchanged==0)):
                           
                # Save objects
                save_state = {}
                save_state['referee'] = referee
                save_state['go'] = go
                save_state['natural_selector'] = natural_selector

                # Save status of optimization run
                save_state['rng_state'] = np.random.get_state()
                save_state['gen'] = gen
                save_state['ngens'] = self.ngens
                save_state['stopping_tolerance'] = self.eps
                save_state['cost_fun'] = cost_fun
                save_state['pmut'] = self.pmut
                save_state['pone'] = self.ponec
                save_state['ptwo'] = self.ptwoc
                save_state['pinv'] = self.pinv
                save_state['pblo'] = self.pblo
                save_state['pgswap'] = self.pgswap
                save_state['piswap'] = self.piswap
                save_state['evo_pressure'] = self.evo_pressure
                if mofit.get_ncost() > 1:
                    save_state['pareto_hist'] = pareto_hist

                # Save all information to pickle file
                joblib.dump(save_state, self.save_file)
            
                # Add-on: Reduce number of files on cluster drive
                subprocess.call(["./zip_running.sh"])

            # Check convergence
            if minFit < 0 + self.eps:
                
                if not self.vquiet:
                    print("Convergence reached!")
                    break

        # Get run_id of best result
        best_run_id = world.get_run_id(pop[minIndex])

        # Output results
        if not self.vquiet:
            print(("After %d generations, best colony has index %d and fitness value of %f" % (self.ngens, minIndex, minFit)))
            for i in range(factory._nexpr):
                print(("Expression: %d \t%s" % (i, pop[minIndex].print_phenotype(nice=True)[i])))
        
            if best_run_id is not None:
                print(("Run ID: %s" % best_run_id))

        # Save best external evaluation if available
        if best_run_id is not None:
            subprocess.call(["cp", "-r", self.ext_eval_path + '/' + "run_" + best_run_id, self.ext_eval_path + '/' + "run_best"])
                

if __name__ == "__main__":        

    # Reading command line arguments
    parser = argparse.ArgumentParser(description='Training specified EVE model')
    parser.add_argument('--data_path', type=str, default=data_path, help='Specify path to training data')
    parser.add_argument('--ext_eval_path', type=str, default=ext_eval_path, help='Specify path to running folder')
    parser.add_argument('--ngens', type=int, default=ngens, help='Set number of generations')
    parser.add_argument('--eps', type=float, default=stopping_tolerance, help='Set convergence criteria')
    parser.add_argument('-pop','--pop_size', type=int, default=pop_size, help='Set size of population')
    parser.add_argument('--tsize', type=int, default=tsize, help='Set number of tournament participants')
    parser.add_argument('--msize', type=int, default=msize, help='Set number of tournaments = size of mating pool')
    parser.add_argument('--seed', type=int, default=seed, help='Set seed for random function')
    parser.add_argument('--pmut', type=float, default=pmut, help='Set probability for mutation')
    parser.add_argument('--ponec', type=float, default=ponec, help='Set probability for one-point crossover')
    parser.add_argument('--ptwoc', type=float, default=ptwoc, help='Set probability for two-point crossover')
    parser.add_argument('--pinv', type=float, default=pinv, help='Set probability for inversion')
    parser.add_argument('--pblo', type=float, default=pblo, help='Set probability for blocking')
    parser.add_argument('--pgswap', type=float, default=pgswap, help='Set probability for gene swapping')
    parser.add_argument('--piswap', type=float, default=piswap, help='Set probability for indi swapping')
    parser.add_argument('-evo','--evo_pressure', action='store_true', help='Activate evolutionary pressure')
    parser.add_argument('--head_length', type=int, default=head_length, nargs="+", help='Set length of head for each expression using GEP')
    parser.add_argument('--ngenes', type=int, default=ngenes, nargs="+", help='Set number of genes for each expression using GEP')
    parser.add_argument('--quiet', action='store_true', help='Suppress most command line output')
    parser.add_argument('--vquiet', action='store_true', help='Suppress all command line output')
    parser.add_argument('-res', '--res_file', type=str, default=res_file, help='Specify result file')
    parser.add_argument('--res_vars', type=str, default=res_vars, nargs="+", help='Specify variables to output in result file')
    parser.add_argument('--res_gens', type=int, default=res_gens, nargs="+", help='Specify generations after which output is written')
    parser.add_argument('--restart', type=int, default=restart, help='Set restart option')
    parser.add_argument('--restart_file', type=str, default=restart_file, help='Specify restart file')
    parser.add_argument('--save_file', type=str, default=save_file, help='Specify save file')
    parser.add_argument('--save_gens', type=int, default=save_gens, nargs="+", help='Specify generations after which optimization is saved')
    parser.add_argument('-algo','--algorithm_choice', type=int, default=algorithm_choice, help='Select algorithm: 1 for GEP, 2 for DG')
    parser.add_argument('--num_proc', type=int, default=num_proc, help='Specify number of processes to run external evaluations')
    parser.add_argument('--num_thr', type=int, default=num_thr, help='Specify number of threads to run external evaluations')

    opt = parser.parse_args()

    # Check if lists are lists
    check_vars = ['head_length', 'ngenes', 'res_gens', 'save_gens']
    for var in check_vars:
        if not isinstance(opt.__dict__[var], Iterable):
            opt.__dict__[var] = [opt.__dict__[var]]

    # Set parameters
    model = Model(**opt.__dict__)

    # Run training
    model.train()
