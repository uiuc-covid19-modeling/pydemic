import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.rc('font', family='serif',size=12)
from datetime import date, datetime, timedelta
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
#set_numpy_threads()

import numpy as np

def wrapper(args):
    model_params, cases = args
    likelihood = neher.calculate_likelihood_for_model(model_params, cases.dates, cases.deaths, n_samples=200)
    print(model_params, likelihood)
    return likelihood

if __name__ == "__main__":

    # load reported data
    cases = get_case_data("USA-Illinois")
    target_date = date(*cases.last_date)
    
    # define parameter space
    n1 = 21
    n2 = 21
    R0s = np.linspace(2.,4.,n1)
    start_days = np.linspace(40,60,n2)
    params_1, params_2 = np.meshgrid(R0s, start_days)

    # generate parameters over which to iterate
    likelihoods = np.zeros(params_1.shape)
    data_params = []
    for i in range(params_1.shape[0]):
        for j in range(params_1.shape[1]):
            p1 = params_1[i,j]
            p2 = params_2[i,j]
            model_params = {
                'r0': p1,
                'start_day': p2,
                'end_day': (date(*cases.last_date)-date(2020,1,1)).days,
            }
            data_params.append((model_params, cases))
            
    # run in parallel
    p = Pool(int(cpu_count()*0.9375))
    likelihoods = np.array(p.map(wrapper, data_params)).reshape(n1,n2)

    # find best likelihood
    best_likelihood = -np.inf
    for i in range(params_1.shape[0]):
        for j in range(params_1.shape[1]):
            if likelihoods[j,i] > best_likelihood:
                best_likelihood = likelihoods[j,i]
                p1 = params_1[j,i]
                p2 = params_2[j,i]
                best_params = { 
                    'r0': p1,
                    'start_day': p2,
                    'end_day': (date(*cases.last_date)-date(2020,1,1)).days,
                }

    print("best parameters with likelihood", best_likelihood)
    print(best_params)
    params_string = " ".join(["{0:s}={1:g}".format(x,best_params[x]) for x in best_params])

    # save likelihood data
    import h5py
    hfp = h5py.File('neher_naive.h5','w')
    hfp['r0'] = R0s
    hfp['start_days'] = start_days
    hfp['likelihoods'] = likelihoods
    hfp.close()

    # plot grid of parameter space
    plt.close('all')
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)
    d1 = R0s[1]-R0s[0]
    d2 = start_days[1] - start_days[0]
    R0s = np.linspace(R0s[0]-d1,R0s[-1]+d1,len(R0s)+1)
    start_days = np.linspace(start_days[0]-d2,start_days[-1]+d2,len(start_days)+1)
    ax1.pcolormesh(R0s, start_days, np.exp(likelihoods))
    ax1.set_xlabel('r0')
    ax1.set_ylabel('start day')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/neher_naive_likelihoods.png')

    # plot best-fit model
    plt.close('all')
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)
    deterministic = neher.get_model_result(best_params)
    model_dates = deterministic.t
    model_deaths = deterministic.quantile_data[2,:]
    dates = [datetime(2020, 1, 1)+timedelta(x) for x in deterministic.t]
    ax1.fill_between(dates, deterministic.quantile_data[1,:], deterministic.quantile_data[3,:])
    ax1.plot(dates, model_deaths, '-k')
    plot_data(ax1, cases.dates, cases.deaths, target_date)
    format_axis(fig, ax1)
    #plt.suptitle("fit for incubation ~ 5 & infectious ~ 3: R0 ~ {0:.1f}".format(best_params['r0']))
    plt.suptitle(" ".join(["{0:s}={1:.1f}".format(x,best_params[x]) for x in best_params]))
    plt.ylabel("count (persons)")
    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/neher_naive_best.png')





    """
    # generate params
    likelihoods = np.zeros(params_1.shape)
    data_params = []
    for i in range(params_1.shape[0]):
        for j in range(params_1.shape[1]):
            p1 = params_1[i,j]
            p2 = params_2[i,j]
            model_params = {
                'r0': p1,
                'start_day': p2,
                'end_day': (date(*cases.last_date)-date(2020,1,1)).days,
            }
            data_params.append((model_params, y_data))
            
    # run in parallel
    p = Pool(int(cpu_count()*0.9375))
    likelihoods = np.array(p.map(wrapper, data_params)).reshape(n1,n2)

    best_likelihood = -np.inf
    for i in range(params_1.shape[0]):
        for j in range(params_1.shape[1]):
            if likelihoods[j,i] > best_likelihood:
                best_likelihood = likelihoods[j,i]
                p1 = params_1[j,i]
                p2 = params_2[j,i]
                best_params = { 
                    'r0': p1,
                    'start_day': p2,
                    'end_day': (date(*cases.last_date)-date(2020,1,1)).days,
                }

    print("best parameters with likelihood", best_likelihood)
    print(best_params)
    params_string = " ".join(["{0:s}={1:g}".format(x,best_params[x]) for x in best_params])

    # save likelihood data
    import h5py
    hfp = h5py.File('neher_naive.h5','w')
    hfp['r0'] = R0s
    hfp['start_days'] = start_days
    hfp['likelihoods'] = likelihoods
    hfp.close()

    # plot grid of parameter space
    plt.close('all')
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)
    d1 = R0s[1]-R0s[0]
    d2 = start_days[1] - start_days[0]
    R0s = np.linspace(R0s[0]-d1,R0s[-1]+d1,len(R0s)+1)
    start_days = np.linspace(start_days[0]-d2,start_days[-1]+d2,len(start_days)+1)
    ax1.pcolormesh(R0s, start_days, np.exp(likelihoods))
    ax1.set_xlabel('r0')
    ax1.set_ylabel('start day')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/neher_naive_likelihoods.png')


    # plot best-fit model
    plt.close('all')
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)
    if False:
      quantiles_result = neher.get_model_result(best_params, 0.05, run_stochastic=True)
      plot_quantiles(ax1, quantiles_result)
    else:
      quantiles_result = neher.get_model_result(best_params, 0.05)
      dates = [datetime(2020, 1, 2, 0)+timedelta(x) for x in quantiles_result.t]
      ax1.fill_between(dates, quantiles_result.quantile_data[1,:], quantiles_result.quantile_data[3,:])
      ax1.plot(dates, quantiles_result.quantile_data[2,:], '-k')
    plot_data(ax1, cases.dates, cases.deaths, target_date)
    format_axis(fig, ax1)
    #plt.suptitle("fit for incubation ~ 5 & infectious ~ 3: R0 ~ {0:.1f}".format(best_params['r0']))
    plt.suptitle(" ".join(["{0:s}={1:.1f}".format(x,best_params[x]) for x in best_params]))
    plt.ylabel("count (persons)")
    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/neher_naive_best.png')
    """


