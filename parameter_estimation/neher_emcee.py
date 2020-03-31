import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.rc('font', family='serif',size=12)
from datetime import date, datetime, timedelta
import emcee
from multiprocessing import Pool, cpu_count

import models.neher as neher
from pydemic.load import get_case_data
from pydemic.plot import plot_quantiles, plot_deterministic
from plutil import plot_model, plot_data, format_axis


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

set_numpy_threads(2)
import numpy as np

# define posterior parameters
parameter_names = ['r0', 'start_day']
centered_guesses = [3., 50]
guess_uncertainties = [0.2, 2]
parameter_priors = [ [2., 4.], [40,60] ]


def not_within(x, xrng):
    if x < xrng[0] or x > xrng[1]: return True
    return False

def log_probability(theta, cases):
    for i in range(len(theta)):
        if not_within(theta[i], parameter_priors[i]): 
            return -np.inf
    model_params = {
        'r0': theta[0],
        'start_day': theta[1],
        'end_day': 88
    }
    likelihood = neher.calculate_likelihood_for_model(model_params, cases.dates, cases.deaths, n_samples=200)
    print(model_params, likelihood)
    return likelihood

if __name__ == "__main__":

    # load reported data
    cases = get_case_data("USA-Illinois")
    target_date = date(*cases.last_date)
    
    # tests
    """
    theta = [ 2.9596679309359946, 54.348166086957605 ]
    print( theta, log_probability(theta, y_data) )
    theta = [ 3.12, 54. ]
    print( theta, log_probability(theta, y_data) )
    theta = [ 3.15, 55. ]
    print( theta, log_probability(theta, y_data) )
    """

    # define sampler parameters
    n_walkers = 36
    n_steps = 500
    
    # get pool for multi-processing
    num_workers = cpu_count() // 1
    print(" - trying with {0:d} workers".format(num_workers))
    pool = Pool(num_workers)

    # generate emcee sampler
    n_dims = len(parameter_names)
    inital_position = np.array(centered_guesses) + \
        np.random.randn(n_walkers, n_dims)*np.array(guess_uncertainties)
    sampler = emcee.EnsembleSampler(n_walkers, n_dims, log_probability, 
        args=([cases]), pool=pool)

    # run sampler
    sampler.run_mcmc(inital_position, n_steps, progress=True)

    # get summary statistics
    # tau = sampler.get_autocorr_time()
    # print("autocorrelation time:", tau)
    tau = 20

    discard_n = 100
    flat_samples = sampler.get_chain(discard=discard_n, thin=10, flat=True)

    import corner
    plt.close('all')
    fig = corner.corner( flat_samples, labels=parameter_names )
    fig.savefig('imgs/neher_emcee_samples.png')

    for i in range(n_dims):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(mcmc[1], q[0], q[1])
        if i==0:
            r0_best = mcmc[1]
        elif i==1:
            start_day_best = mcmc[2]

    best_params = {
        'r0': r0_best,
        'start_day': start_day_best,
        'end_day': 88
    }
    print(best_params)

    plt.close('all')
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)
    if False:
      quantiles_result = neher.get_model_result(best_params, 0.05, run_stochastic=True)
      plot_quantiles(ax1, quantiles_result)
    else:
      quantiles_result = neher.get_model_result(best_params, 0.05)
      dates = [datetime(2020, 1, 1)+timedelta(x) for x in quantiles_result.t]
      ax1.fill_between(dates, quantiles_result.quantile_data[1,:], quantiles_result.quantile_data[3,:])
      ax1.plot(dates, quantiles_result.quantile_data[2,:], '-k')
    plot_data(ax1, cases.dates, cases.deaths, target_date)
    format_axis(fig, ax1)
    #plt.suptitle("fit for incubation ~ 5 & infectious ~ 3: R0 ~ {0:.1f}".format(best_params['r0']))
    plt.suptitle(" ".join(["{0:s}={1:.1f}".format(x,best_params[x]) for x in best_params]))
    plt.ylabel("count (persons)")
    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/neher_emcee_best.png')





