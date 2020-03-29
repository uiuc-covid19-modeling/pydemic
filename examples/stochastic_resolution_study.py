from scipy.interpolate import interp1d
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

def generate_stochastic_data(n_sims, method, dt=0.005):
    from pydemic.models import SEIRModelSimulation
    t_span = [0., 10.]
    y0 = {
        'susceptible': 1.e3,
        'exposed': 6.,
        'infectious': 14.,
        'removed': 0 
    }
    stochastic_results = []
    n_sims = 100
    for i in range(n_sims):
        print(" - running stochastic ({0:s}) sample {1:d} of {2:d}".format(method, i+1, n_sims))
        sresult = simulation(t_span, y0, lambda x:x, stochastic_method=stochastic_method, dt=timestep) 
        stochastic_results.append(sresult)
    pass


def load_tauleap(fname, keys):
    data = {}
    for key in keys:
        data[key] = []
    hfp = h5py.File(fname,'r')
    for i in range(100):
        for key in keys:
            fkey = "{0:d}_{1:s}".format(i, key)
            data[key].append(np.array(hfp[fkey]))
    hfp.close()
    return data

def load_direct(fname, keys, times):
    data = {}
    for key in keys:
        data[key] = []
    hfp = h5py.File(fname,'r')
    for i in range(100):
        temp_data = {}
        for key in keys:
            fkey = "{0:d}_{1:s}".format(i, key)
            temp_data[key] = np.array(hfp[fkey])
            if key != 't':
                f = interp1d(temp_data['t'], temp_data[key], bounds_error=False, fill_value=temp_data[key][-1])
                data[key].append(f(times))
    hfp.close()
    return data

if __name__ == "__main__":


    generate_stochastic_data(1000, 'direct')


    exit()

    # load data from tauleap and gillespie direct methods
    keys = ["t", "susceptible", "exposed", "infectious", "removed"]
    tauleap_data = load_tauleap("DATA_TAU_LEAP.h5", keys)
    times = tauleap_data['t'][0]
    direct_data = load_direct("DATA_DIRECT.h5", keys, times)

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
    plt.savefig('stochastic_comparison.png')

