import os
import numpy as np
import subprocess
import signal
import sympy as sp

# Change working directory
os.chdir(os.path.dirname(__file__))

# Get run_id
run_id = os.path.basename(os.getcwd()).split('_')[-1]

# Load expressions
expressions = []

with open("input_" + run_id, 'r') as f:
    for line in f.readlines():
        expressions.append(line.replace('\n',''))

# Simplify expressions
input_symbols = ["I01", "I02"]
I01, I02 = sp.symbols(' '.join(input_symbols))

sympy_precision = 8
sympy_err = 0

sim_err  = 0
sim_err_ = 0

short_exprs = []

# Python power (**) does not work in c++
# Source: https://stackoverflow.com/questions/14264431/expanding-algebraic-powers-in-python-sympy
# Modified as expr.subs() did not work
def pow_to_mul_to_str(expr):
    """
    Convert integer powers in an expression to Muls, like a**2 => a*a.
    """
    pows = list(expr.atoms(sp.Pow))
    if any(not e.is_Integer for b, e in (i.as_base_exp() for i in pows)):
        raise ValueError("A power contains a non-integer exponent")
    str_expr = str(expr)
    for pow in pows:
        b, e = pow.as_base_exp()
        mul = sp.Mul(*[b]*e,evaluate=False)
        str_expr = str_expr.replace(str(pow), str(mul))
    return str_expr

def signal_handler(signum, frame):
    raise OverflowError("ERROR: Max. processing time reached.")

signal.signal(signal.SIGALRM, signal_handler)

for i, expr in enumerate(expressions):

    # Set alarm
    signal.alarm(30)

    try:
        short_expr = sp.simplify(expr)
        short_expr = short_expr.xreplace({n : round(n, sympy_precision) for n in short_expr.atoms(sp.Number)})
        str_expr = pow_to_mul_to_str(short_expr)
        short_exprs.append(str_expr)

    except:
        sympy_err = 1
    
    # Disable alarm
    signal.alarm(0)

if not sympy_err:

    # Create nonLinearModel.H
    compile_folder = "./kOmegaSSTx"
    model_template = "./modelTemp.H"
    model_file = "./nonLinearModel.H"
    
    model_exprs = []

    os.chdir(compile_folder)
    with open(model_template, 'r') as f:
        for line in f.readlines():
            model_exprs.append(line)

    os.remove(model_template)

    with open(model_file, 'w') as f:
        for line in model_exprs:
            for i, expr in enumerate(short_exprs):
                line = line.replace("EXPR%s" % str(i).zfill(2), expr)
            f.write("%s" % line)

    # Compile turbulence model
    compile_command = "./Allwmake"

    subprocess.call(compile_command)
    os.chdir("../.")

    # Run OpenFOAM
    run_command = "./Allrun"
    run_folder = "./PH_2D_RANS_kOmegaSSTx"

    os.chdir(run_folder)
    subprocess.call(run_command)

    # Read OpenFOAM results
    post_folder = "./postProcessing/sets/"
    #post_vars = ['U', 'k', 'R']
    #var_dict = {'U': 1, 'k': 1, 'R' : 2}
    #x_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    cwd = os.getcwd()
    print(cwd)

    try:
        res_folders = os.listdir(post_folder)
        max_iter = np.max([int(dir) for dir in res_folders])

        if (max_iter==0) | (max_iter==20000):
            sim_err = 1
        else:
            #for var in post_vars:
            #    
            #    var_data = []

            #    for pos in x_positions:
            #        post_file = 'X' + str(int(pos)) + 'H_' + var + '.csv'
            #        data = np.loadtxt(os.path.join(post_folder, str(max_iter), post_file),delimiter=',', skiprows=1)
            #        var_data = np.append(var_data, data[:,var_dict[var]])
            #    if (var=='R'):
            #        save_file = var.upper() + 'xy1.edf.gz'
            #        np.savetxt("../output/" + save_file, var_data)
            #    else:
            #        save_file = var.upper() + '1.edf.gz'
            #        np.savetxt("../output/" + save_file, var_data)
    
            #post_file = 'cluster0_k.xy' # k
            #data = np.loadtxt(post_file)
            #np.savetxt("../output/" + save_file, data[:,3])
            #
            #post_file = 'cluster0_U.xy' # U
            #data = np.loadtxt(post_file)
            #np.savetxt("../output/" + save_file, data[:,3])
            #
            #post_file = 'cluster0_U.xy' # V
            #data = np.loadtxt(post_file)
            #np.savetxt("../output/" + save_file, data[:,4])
            #
            #post_file = 'cluster0_R.xy' # Rxy
            #data = np.loadtxt(post_file)
            #np.savetxt("../output/" + save_file, data[:,4])

            # cluster 0
            post_file = post_folder + str(max_iter) + '/cluster0_k.xy' # k
            data = np.loadtxt(post_file)
            np.savetxt("../output/K01.edf.gz", data[:,3])

            post_file = post_folder + str(max_iter) + '/cluster0_U.xy' # U
            data = np.loadtxt(post_file)
            np.savetxt("../output/U01.edf.gz", data[:,3])

            post_file = post_folder + str(max_iter) + '/cluster0_U.xy' # V
            data = np.loadtxt(post_file)
            np.savetxt("../output/V01.edf.gz", data[:,4])

            post_file = post_folder + str(max_iter) + '/cluster0_R.xy' # Rxy
            data = np.loadtxt(post_file)
            np.savetxt("../output/Rxy01.edf.gz", data[:,4])
            
            # cluster 1
            post_file = post_folder + str(max_iter) + '/cluster1_k.xy' # k
            data = np.loadtxt(post_file)
            np.savetxt("../output/K11.edf.gz", data[:,3])

            post_file = post_folder + str(max_iter) + '/cluster1_U.xy' # U
            data = np.loadtxt(post_file)
            np.savetxt("../output/U11.edf.gz", data[:,3])

            post_file = post_folder + str(max_iter) + '/cluster1_U.xy' # V
            data = np.loadtxt(post_file)
            np.savetxt("../output/V11.edf.gz", data[:,4])

            post_file = post_folder + str(max_iter) + '/cluster1_R.xy' # Rxy
            data = np.loadtxt(post_file)
            np.savetxt("../output/Rxy11.edf.gz", data[:,4])

            # cluster 2
            post_file = post_folder + str(max_iter) + '/cluster2_k.xy' # k
            data = np.loadtxt(post_file)
            np.savetxt("../output/K21.edf.gz", data[:,3])

            post_file = post_folder + str(max_iter) + '/cluster2_U.xy' # U
            data = np.loadtxt(post_file)
            np.savetxt("../output/U21.edf.gz", data[:,3])

            post_file = post_folder + str(max_iter) + '/cluster2_U.xy' # V
            data = np.loadtxt(post_file)
            np.savetxt("../output/V21.edf.gz", data[:,4])

            post_file = post_folder + str(max_iter) + '/cluster2_R.xy' # Rxy
            data = np.loadtxt(post_file)
            np.savetxt("../output/Rxy21.edf.gz", data[:,4])

    except:
        sim_err = 1
    
#    post_vars_ = ['U']
#    var_dict = {'U': 2}
#    x_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#
#    try:
#        res_folders = os.listdir(post_folder)
#        max_iter = np.max([int(dir) for dir in res_folders])
#
#        if (max_iter==0) | (max_iter==20000):
#            sim_err_ = 1
#        else:
#            for var in post_vars_:
#                
#                var_data = []
#
#                for pos in x_positions:
#                    post_file = 'X' + str(int(pos)) + 'H_' + var + '.csv'
#                    data = np.loadtxt(os.path.join(post_folder, str(max_iter), post_file),delimiter=',', skiprows=1)
#                    var_data = np.append(var_data, data[:,var_dict[var]])
#
#                save_file = 'V1.edf.gz'
#                np.savetxt("../output/" + save_file, var_data)
#    except:
#        sim_err_ = 1

    os.chdir("../.")

# Handle errors
if (sympy_err + sim_err) > 0:
    
    # Get length of training data
    #train_folder = "../../training_data/"
    #train_files = os.listdir(train_folder)
    #data_length = np.loadtxt(os.path.join(train_folder, train_files[0])).shape[0]
    #print(data_length)

    train_folder = "../../training_data/cluster0/"
    train_files = os.listdir(train_folder)
    data_length = np.loadtxt(os.path.join(train_folder, train_files[0])).shape[0]
    print(data_length)
    
    np.savetxt("./output/K01.edf.gz", [np.nan]*data_length)
    np.savetxt("./output/U01.edf.gz", [np.nan]*data_length)
    np.savetxt("./output/V01.edf.gz", [np.nan]*data_length)
    np.savetxt("./output/Rxy01.edf.gz", [np.nan]*data_length)

    train_folder = "../../training_data/cluster1/"
    train_files = os.listdir(train_folder)
    data_length = np.loadtxt(os.path.join(train_folder, train_files[0])).shape[0]
    print(data_length)
    
    np.savetxt("./output/K11.edf.gz", [np.nan]*data_length)
    np.savetxt("./output/U11.edf.gz", [np.nan]*data_length)
    np.savetxt("./output/V11.edf.gz", [np.nan]*data_length)
    np.savetxt("./output/Rxy11.edf.gz", [np.nan]*data_length)

    train_folder = "../../training_data/cluster2/"
    train_files = os.listdir(train_folder)
    data_length = np.loadtxt(os.path.join(train_folder, train_files[0])).shape[0]
    print(data_length)
    
    np.savetxt("./output/K21.edf.gz", [np.nan]*data_length)
    np.savetxt("./output/U21.edf.gz", [np.nan]*data_length)
    np.savetxt("./output/V21.edf.gz", [np.nan]*data_length)
    np.savetxt("./output/Rxy21.edf.gz", [np.nan]*data_length)
    
    # Set NaNs
    #for var in post_vars:
    #    if (var=='R'):
    #        save_file = var.upper() + 'xy1.edf.gz'
    #        np.savetxt("./output/" + save_file, [np.nan]*data_length)
    #    else:
    #        save_file = var.upper() + '1.edf.gz'
    #        np.savetxt("./output/" + save_file, [np.nan]*data_length)

#if (sim_err_) > 0:
#    
#    # Get length of training data
#    train_folder = "../../training_data/"
#    train_files = os.listdir(train_folder)
#    data_length = np.loadtxt(os.path.join(train_folder, train_files[0])).shape[0]
#    print(data_length)
#
#    # Set NaNs
#    for var in post_vars_:
#        save_file = 'V1.edf.gz'
#        np.savetxt("./output/" + save_file, [np.nan]*data_length)

