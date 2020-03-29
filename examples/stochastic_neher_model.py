"""from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.rc('font', family='serif', size=12)
import numpy as np
import h5py  # pylint: disable=E0401
import os"""

import numpy as np
from datetime import datetime, timedelta


from pydemic import (PopulationModel, AgeDistribution, SeverityModel,
                     EpidemiologyModel, ContainmentModel, QuantileLogger)
from pydemic.models import NeherModelSimulation

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

"""
def run_stochastic_simulation(args):
    i, method, dt, num_age_groups = args
    from pydemic.models import SEIRModelSimulation
    t_span = [0., 10.]
    y0 = {
        'susceptible': np.array([1.e3, 2.e3]),
        'exposed': np.array([6, 4]),
        'infectious': np.array([14, 18]),
        'removed': np.array([0, 0]),
    }
    if num_age_groups == 1:
        y0 = {key: val[0] for key, val in y0.items()}

    print(" - running stochastic ({0:s}) simulation {1:d}".format(method, i+1))
    simulation = SEIRModelSimulation()
    sresult = simulation(t_span, y0, stochastic_method=method, dt=dt)
    packed_data = np.zeros((5,) + sresult.y['susceptible'].shape)
    packed_data[0, :, 0] = sresult.t
    packed_data[1, :] = sresult.y['susceptible']
    packed_data[2, :] = sresult.y['exposed']
    packed_data[3, :] = sresult.y['infectious']
    packed_data[4, :] = sresult.y['removed']
    return packed_data


def generate_stochastic_data(n_sims, method, datafile_name, dt=0.005):
    n_cores = int(cpu_count()*0.9)
    p = Pool(n_cores)
    num_age_groups = 2
    print(" - generating stochastic samples with {0:d} cores".format(n_cores))
    args_vec = [(i, method, dt, num_age_groups) for i in range(n_sims)]
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
    hfp = h5py.File(fname, 'r')
    n_sims = hfp['n_sims'][()]
    for i in range(n_sims):
        fkey = "{0:d}".format(i)
        run_data = np.array(hfp[fkey])
        data['t'].append(run_data[0, :, 0])
        for c in range(1, 5):
            data[keys[c]].append(run_data[c, :])
    hfp.close()
    return data


def load_direct(fname, keys, times, force=False, n_sims=100):
    if force or not os.path.exists(fname):
        generate_stochastic_data(n_sims, 'direct', fname)
    data = {}
    for key in keys:
        data[key] = []
    hfp = h5py.File(fname, 'r')
    n_sims = hfp['n_sims'][()]
    for i in range(n_sims):
        fkey = "{0:d}".format(i)
        temp_data = np.array(hfp[fkey])
        t = temp_data[0, :, 0]
        for c in range(1, 5):
            key = keys[c]
            f = interp1d(t, temp_data[c, :], bounds_error=False,
                         fill_value=temp_data[c, :][-1], axis=0)
            data[key].append(f(times))
    hfp.close()
    return data
"""

if __name__ == "__main__":

    # define containment event
    containment_date = (2020, 3, 20)
    containment_factor = 1.0

    # set some base model stats for the neher model
    compartments = ["susceptible", "exposed", "infectious", "recovered", "hospitalized", "critical", "dead"]
    n_age_groups = 9
    start_date = (2020, 3, 1, 0, 0, 0)
    end_date = (2020, 9, 1, 0, 0, 0)
    population = PopulationModel(
        country='United States of America',
        cases='USA-Illinois',
        population_served=12659682,
        suspected_cases_today=215,
        ICU_beds=1e10,  # originally 1055
        hospital_beds=1e10,  # originally 31649
        imports_per_day=5.0,
    )
    age_distribution = AgeDistribution(
        bin_edges=np.arange(0, 90, 10),
        counts=[39721484, 42332393, 46094077, 44668271, 40348398, 42120077,
                38488173, 24082598, 13147180]
    )
    severity = SeverityModel(
        id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
        age_group=np.arange(0., 90., 10),
        isolated=np.array([0., 0., 20., 10., 0., 0., 50., 90., 0.]),
        confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
        severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
        critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
        fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
    )
    epidemiology = EpidemiologyModel(
        r0=3.7,
        incubation_time=5,
        infectious_period=3,
        length_hospital_stay=4,
        length_ICU_stay=14,
        seasonal_forcing=0.2,
        peak_month=0,
        overflow_severity=2
    )
    containment = ContainmentModel(start_date, end_date, is_in_days=True)
    containment.add_sharp_event(containment_date, containment_factor)

    # generate neher model simulation
    simulation = NeherModelSimulation(
        epidemiology, severity, population.imports_per_day,
        n_age_groups, containment
    )
    y0 = simulation.get_initial_population(population, age_distribution)

    # run simulation
    logger = QuantileLogger()
    result = simulation([start_date, end_date], y0, 0.25, samples=10000, stochastic_method='tau_leap', logger=logger)
    dates = [datetime(2020, 1, 1)+timedelta(x) for x in result.t]



    import matplotlib as mpl ; mpl.use('agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10,8))

    ax1 = plt.subplot(1,1,1)


    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    for i in range(len(compartments)):
        compartment = compartments[i]
        ax1.plot(dates, result.quantile_data[compartment][2, ...].sum(axis=-1), color=colors[i], label=compartment, lw=2)
        ax1.fill_between(dates, result.quantile_data[compartment][1, ...].sum(axis=-1), 
                         result.quantile_data[compartment][3, ...].sum(axis=-1), color=colors[i], alpha=0.5)
        ax1.fill_between(dates, result.quantile_data[compartment][0, ...].sum(axis=-1), 
                         result.quantile_data[compartment][4, ...].sum(axis=-1), color=colors[i], alpha=0.2)


    ax1.set_yscale('log')

    ax1.set_ylim(ymin=0.8)

    ax1.legend()
    ax1.set_xlabel('time')
    ax1.set_ylabel('count (persons)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/stochastic_neher_model.png')





    exit()
    n_sims = 100
    force_regen = False
    # load (or generate) data from tauleap and gillespie direct methods
    keys = ['t', 'susceptible', 'exposed', 'infectious', 'removed']
    tauleap_data = load_tauleap("data/stochastic_runs_tauleap.h5", keys,
                                force=force_regen, n_sims=n_sims)
    times = tauleap_data['t'][0]
    direct_data = load_direct("data/stochastic_runs_direct.h5", keys, times,
                              force=force_regen, n_sims=n_sims)

    # translate to quantiles
    quantiles = [0.0455, 0.3173, 0.5, 0.6827, 0.9545]
    tauleap_quantiles = {}
    direct_quantiles = {}
    for key in keys[1:]:
        tauleap_quantiles[key] = [
            np.quantile(np.array(tauleap_data[key]), quantile, axis=0)
            for quantile in quantiles
        ]
        direct_quantiles[key] = [
            np.quantile(np.array(direct_data[key]), quantile, axis=0)
            for quantile in quantiles
        ]

    # formatting options and names for plot
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple']
    style = [':', '--', '-', '--', ':']

    # plot
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    for i, key in enumerate(keys[1:]):
        shape = tauleap_quantiles[key][1].shape[1:]
        from itertools import product
        slices = list(product(*[range(n) for n in shape]))

        color = colors[i]
        for slc in slices:
            slc = (slice(None, None),) + slc

            for ax, data in zip((ax1, ax2), (tauleap_quantiles, direct_quantiles)):
                ax.fill_between(times, data[key][1][slc], data[key][3][slc],
                                color=color, alpha=0.5)
                ax.fill_between(times, data[key][0][slc], data[key][4][slc],
                                color=color, alpha=0.2)
                ax.plot(times, data[key][2][slc], '-', c=color, lw=1, label=key)

            # plot differences
            for q in range(5):
                relerr = np.abs(
                    1-tauleap_quantiles[key][q][slc] / direct_quantiles[key][q][slc]
                )
                ax3.plot(times, relerr, style[q], c=color)

    # format and save
    PLOT_LOG = True
    ax1.set_title('tau leap method')
    ax2.set_title('gillespie direct method')
    if PLOT_LOG:
        ax1.set_ylim(0.8, 2.e3)
        ax2.set_ylim(0.8, 2.e3)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    else:
        ax1.set_ylim(0.8, 1.e3)
        ax2.set_ylim(0.8, 1.e3)
    ax1.set_xlim(-0.2, 10.2)
    ax2.set_xlim(-0.2, 10.2)
    #ax3.set_ylim(0.01, 2.)
    ax3.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('count (persons)')
    ax2.set_ylabel('count (persons)')
    ax3.set_ylabel('relative error')
    ax3.set_xlabel('time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/stochastic_resolution_study.png')
