from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.rc('font', family='serif',size=12)
import numpy as np
import h5py
import os

def set_numpy_threads(nthreads=1):
    # see also https://codereview.stackexchange.com/questions/206736/better-way-to-set-number-of-threads-used-by-numpy  # noqa
    import os
    try:
        import mkl
        mkl.set_num_threads(nthreads)
        return 0
    except:  # noqa=E722
        pass
    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            import ctypes
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
            return 0
        except:  # noqa=E722
            pass
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)

def run_stochastic_simulation(args):
    i, method, dt = args
    from pydemic.models import SEIRModelSimulation
    t_span = [0., 10.]
    y0 = {
        'susceptible': 1.e3,
        'exposed': 6.,
        'infectious': 14.,
        'removed': 0 
    }
    print(" - running stochastic ({0:s}) simulation {1:d}".format(method, i+1))
    simulation = SEIRModelSimulation()
    sresult = simulation(t_span, y0, stochastic_method=method, dt=dt)
    packed_data = np.zeros((5, sresult.t.shape[0]))
    packed_data[0,:] = sresult.t
    packed_data[1,:] = sresult.y['susceptible']
    packed_data[2,:] = sresult.y['exposed']
    packed_data[3,:] = sresult.y['infectious']
    packed_data[4,:] = sresult.y['removed']
    return packed_data

def generate_stochastic_data(n_sims, method, datafile_name, dt=0.005):
    n_cores = int(cpu_count()*0.9)
    p = Pool(n_cores)
    print(" - generating stochastic samples with {0:d} cores".format(n_cores))
    args_vec = [(i, method, dt) for i in range(n_sims)]
    stochastic_results = p.map(run_stochastic_simulation, args_vec)
    hfp = h5py.File(datafile_name, 'w')
    hfp['n_sims'] = n_sims
    for i in range(n_sims):
        hfp["{0:d}".format(i)] = stochastic_results[i]
    hfp.close()
    

def load_tauleap(fname, keys, force=False, n_sims=100):
    if force or not os.path.exists(fname):
        generate_stochastic_data(n_sims, 'tau_leap', fname)
    data = {}
    for key in keys:
        data[key] = []
    hfp = h5py.File(fname,'r')
    n_sims = hfp['n_sims'][()]
    for i in range(n_sims):
        fkey = "{0:d}".format(i)
        run_data = np.array(hfp[fkey])
        for c in range(5):
            data[keys[c]].append(run_data[c,:])
    hfp.close()
    return data

def load_direct(fname, keys, times, force=False, n_sims=100):
    if force or not os.path.exists(fname):
        generate_stochastic_data(n_sims, 'direct', fname)
    data = {}
    for key in keys:
        data[key] = []
    hfp = h5py.File(fname,'r')
    n_sims = hfp['n_sims'][()]
    for i in range(n_sims):
        fkey = "{0:d}".format(i)
        temp_data = np.array(hfp[fkey])
        for c in range(1,5):
            key = keys[c]
            f = interp1d(temp_data[0,:], temp_data[c,:], bounds_error=False, fill_value=temp_data[c,:][-1])
            data[key].append(f(times))
    hfp.close()
    return data

if __name__ == "__main__":

    # load (or generate) data from tauleap and gillespie direct methods
    keys = ['t', 'susceptible', 'exposed', 'infectious', 'removed']
    tauleap_data = load_tauleap("data/stochastic_runs_tauleap.h5", keys, force=True, n_sims=100)
    times = tauleap_data['t'][0]
    direct_data = load_direct("data/stochastic_runs_direct.h5", keys, times, force=True, n_sims=100)

    # translate to quantiles
    quantiles = [ 0.0455, 0.3173, 0.5, 0.6827, 0.9545 ]
    tauleap_quantiles = {}
    direct_quantiles = {}
    for i in range(len(keys[1:])):
        key = keys[i+1]
        tauleap_quantiles[key] = []
        direct_quantiles[key] = []
        for quantile in quantiles:
            tauleap_quantiles[key].append(np.quantile(np.array(tauleap_data[key]), quantile, axis=0))
            direct_quantiles[key].append(np.quantile(np.array(direct_data[key]), quantile, axis=0))

    # formatting options and names for plot
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple']
    
    # plot
    plt.figure(figsize=(10,10))
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)
    for i in range(len(keys[1:])):
        key = keys[i+1]
        color = colors[i]
        # plot for tau leap
        ax1.fill_between(times, tauleap_quantiles[key][1], tauleap_quantiles[key][3], color=color, alpha=0.5)
        ax1.fill_between(times, tauleap_quantiles[key][0], tauleap_quantiles[key][4], color=color, alpha=0.2)
        ax1.plot(times, tauleap_quantiles[key][2], '-', c=color, lw=1, label=key)
        # plot for gillespie
        ax2.fill_between(times, direct_quantiles[key][1], direct_quantiles[key][3], color=color, alpha=0.5)
        ax2.fill_between(times, direct_quantiles[key][0], direct_quantiles[key][4], color=color, alpha=0.2)
        ax2.plot(times, direct_quantiles[key][2], '-', c=color, lw=1, label=key)
        # plot differences
        style = [':', '--', '-', '--', ':']
        for q in range(5):
            ax3.plot(times, np.abs(1.-tauleap_quantiles[key][q]/direct_quantiles[key][q]), style[q], c=color)

    # format and save
    PLOT_LOG = True
    ax1.set_title('tau leap method')
    ax2.set_title('gillespie direct method')
    if PLOT_LOG:
        ax1.set_ylim(0.8,2.e3)
        ax2.set_ylim(0.8,2.e3)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    else:
        ax1.set_ylim(0.8,1.e3)
        ax2.set_ylim(0.8,1.e3)
    ax1.set_xlim(-0.2,10.2)
    ax2.set_xlim(-0.2,10.2)
    #ax3.set_ylim(0.01, 2.)
    ax3.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('count (persons)')
    ax2.set_ylabel('count (persons)')
    ax3.set_ylabel('relative error')
    ax3.set_xlabel('time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/stochastic_resolution_study.png')
    
