import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import numpy as np

import models.neher as neher
from pydemic.load import get_case_data
from pydemic.plot import plot_quantiles, plot_deterministic
from plutil import plot_model, plot_data, format_axis


if __name__ == "__main__":

    # load reported data
    cases = get_case_data("USA-Illinois")
    target_date = date(*cases.last_date)

    # define parameter space
    R0s = np.linspace(2.5,5.5,11)
    start_days = np.linspace(50,70,11)
    params_1, params_2 = np.meshgrid(R0s, start_days)

    # run over the grid
    best_params = None
    best_likelihood = -np.inf
    likelihoods = np.zeros(params_1.shape)
    for i in range(params_1.shape[0]):
        for j in range(params_1.shape[1]):
            p1 = params_1[i,j]
            p2 = params_2[i,j]
            model_params = {
                'r0': p1,
                'start_day': int(p2),
                'end_day': (date(*cases.last_date)-date(2020,1,1)).days,
            }
            likelihood = neher.calculate_likelihood_for_model(model_params, cases.deaths[2:])
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_params = model_params
            print(p1, p2, likelihood)
            likelihoods[i, j] = likelihood 

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
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/neher_naive_likelihoods.png')
  
    # plot best-fit model
    plt.close('all')
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)
    quantiles_result = neher.get_model_result(best_params, 0.05)
    plot_quantiles(ax1, quantiles_result)
    plot_data(ax1, cases.dates, cases.deaths, target_date)
    format_axis(fig, ax1)
    plt.suptitle("likelihood = {0:.2g}".format(best_likelihood))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/neher_naive_best.png')
