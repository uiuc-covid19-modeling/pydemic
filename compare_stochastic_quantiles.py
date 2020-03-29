from scipy.interpolate import interp1d
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.rc('font', family='serif',size=12)
import numpy as np
import h5py

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

    # load data from tauleap and gillespie direct methods
    keys = ["t", "susceptible", "exposed", "infectious", "removed"]
    tauleap_data = load_tauleap("DATA_TAU_LEAP.h5", keys)
    times = tauleap_data['t'][0]
    direct_data = load_direct("DATA_DIRECT.h5", keys, times)

    # formatting options and names for plot
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple']
    quantiles = [ 0.0455, 0.3173, 0.5, 0.6827, 0.9545 ]
    
    # plot
    plt.figure(figsize=(10,10))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    for i in range(len(keys[1:])):
        key = keys[i+1]
        color = colors[i]
        # plot for tau leap
        all_data = np.array(tauleap_data[key])
        quantile_data = []
        for quantile in quantiles:
            quantile_data.append(np.quantile(all_data, quantile, axis=0))
        ax1.fill_between(times, quantile_data[1], quantile_data[3], color=color, alpha=0.5)
        ax1.fill_between(times, quantile_data[0], quantile_data[4], color=color, alpha=0.2)
        ax1.plot(times, quantile_data[2], '-', c=color, lw=1, label=key)
        # plot for gillespie
        all_data = np.array(direct_data[key])
        quantile_data = []
        for quantile in quantiles:
            quantile_data.append(np.quantile(all_data, quantile, axis=0))
        ax2.fill_between(times, quantile_data[1], quantile_data[3], color=color, alpha=0.5)
        ax2.fill_between(times, quantile_data[0], quantile_data[4], color=color, alpha=0.2)
        ax2.plot(times, quantile_data[2], '-', c=color, lw=1, label=key)

    # format and save
    PLOT_LOG = False
    ax1.set_title('tau leap method')
    ax2.set_title('gillespie direct method')
    if PLOT_LOG:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax1.set_xlim(-0.2,10.2)
        ax1.set_ylim(0.8,2.e3)
        ax2.set_xlim(-0.2,10.2)
        ax2.set_ylim(0.8,2.e3)
    else:
        ax1.set_xlim(-0.2,10.2)
        ax1.set_ylim(0.8,1.e3)
        ax2.set_xlim(-0.2,10.2)
        ax2.set_ylim(0.8,1.e3)
    ax1.legend()
    ax1.set_ylabel('count (persons)')
    ax2.set_ylabel('count (persons)')
    ax2.set_xlabel('time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('stochastic_comparison.png')

