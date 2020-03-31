import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.rc('font', family='serif', size=12)
from datetime import date, datetime, timedelta
import matplotlib.gridspec as gridspec

import emcee
from multiprocessing import Pool, cpu_count

import models.neher as neher
from pydemic.load import get_case_data
# from pydemic.plot import plot_quantiles, plot_deterministic
from plutil import plot_data, format_axis


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


set_numpy_threads(1)
import numpy as np

# define posterior parameters
parameter_names = ['r0', 'start_day', 'mitigation', 'mitigation_day', 'mitigation_width']
centered_guesses = [3., 30, 0.9, 60, 10]
guess_uncertainties = [0.2, 2, 0.1, 2, 2]
parameter_priors = [[1., 5.], [20, 40], [0.05, 1.0], [30,88], [0.05, 20]]


def not_within(x, xrng):
    if x < xrng[0] or x > xrng[1]:
        return True
    return False


def log_probability(theta, cases):
    for i in range(len(theta)):
        if not_within(theta[i], parameter_priors[i]):
            return -np.inf
    model_params = {
        'r0': theta[0],
        'start_day': theta[1],
        'mitigation': theta[2],
        'mitigation_day': max(theta[3], theta[1]+1),
        'mitigation_width': theta[4],
        'end_day': 88
    }
    likelihood = neher.calculate_likelihood_for_model(
        model_params, cases.dates, cases.deaths)
    #print(model_params, likelihood)
    return likelihood


if __name__ == "__main__":
    # load reported data
    cases = get_case_data("Italy")
    cases = get_case_data("ITA-Lombardia")
    target_date = date(*cases.last_date)

    # define sampler parameters
    n_walkers = 72
    n_steps = 1000

    # get pool for multi-processing
    num_workers = cpu_count() // 1
    print(" - trying with {0:d} workers".format(num_workers))
    pool = Pool(num_workers)

    # generate emcee sampler
    n_dims = len(parameter_names)
    inital_position = np.array(centered_guesses) + \
        np.random.randn(n_walkers, n_dims)*np.array(guess_uncertainties)
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dims, log_probability, args=([cases]), pool=pool)

    # run sampler
    sampler.run_mcmc(inital_position, n_steps, progress=True)

    # get summary statistics
    # tau = sampler.get_autocorr_time()
    # print("autocorrelation time:", tau)
    tau = 20

    discard_n = 200
    flat_samples = sampler.get_chain(discard=discard_n, thin=10, flat=True)

    import corner
    plt.close('all')
    fig = corner.corner(flat_samples, labels=parameter_names)
    fig.savefig('imgs/italy_lombadia_neher_emcee_samples.png')

    for i in range(n_dims):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(mcmc[1], q[0], q[1])
        if i == 0:
            r0_best = mcmc[1]
        elif i == 1:
            start_day_best = mcmc[1]
        elif i == 2:
            mitigation_best = mcmc[1]
        elif i == 3:
            mitigation_day_best = mcmc[1]
        elif i == 4:
            mitigation_width_best = mcmc[1]

    best_params = {
        'r0': r0_best,
        'start_day': start_day_best,
        'mitigation': mitigation_best,
        'mitigation_day': mitigation_day_best,
        'mitigation_width': mitigation_width_best,
        'end_day': 88
    }
    print(best_params)

    plt.close('all')
    fig = plt.figure(figsize=(10, 8))
    gspec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
    ax1 = fig.add_subplot(gspec[:2,0])
    ax2 = fig.add_subplot(gspec[2,0])
    deterministic = neher.get_model_result(best_params)
    model_dates = deterministic.t
    model_deaths = deterministic.quantile_data[2, :]
    dates = [datetime(2020, 1, 1)+timedelta(x) for x in deterministic.t]
    ax1.fill_between(
        dates, deterministic.quantile_data[1, :], deterministic.quantile_data[3, :])
    ax1.plot(dates, model_deaths, '-k')
    plot_data(ax1, cases.dates, cases.deaths, target_date)
    format_axis(fig, ax1)
    containment = neher.get_containment_for_model(best_params)
    ax2.grid()
    ax2.plot(dates, containment(deterministic.t), '-b')
    ax2.set_xlabel('time')
    ax1.set_ylabel('count (persons)')
    ax2.set_ylabel('mitigation factor')
    plt.suptitle(" ".join(["{0:s}={1:.1f}".format(
        x, best_params[x]) for x in best_params]))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/italy_lombardia_neher_emcee_best.png')

