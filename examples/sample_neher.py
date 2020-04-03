__copyright__ = """
Copyright (C) 2020 George N Wong
Copyright (C) 2020 Zachary J Weiner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.rc('font', family='serif', size=12)
import emcee
from multiprocessing import Pool, cpu_count

if __name__ == "__main__":
    # load reported data
    population = "USA-Illinois"
    age_dist_pop = "United States of America"

    from pydemic.data.us import get_case_data
    cases = get_case_data('IL')

    i_start = np.searchsorted(cases.y['death'], .1)
    i_end = np.searchsorted(cases.t, 90)

    # start at day of first death
    data = {'t': cases.t[i_start:i_end], 'dead': cases.y['death'][i_start:i_end]}

    from pydemic.models import SampleParameter

    fit_parameters = [
        SampleParameter('r0', (1, 5), 3, .2),
        SampleParameter('start_day', (40, 60), 50, 2),
        # SampleParameter('mitigation_factor', (.05, 1), .9, .1),
        # SampleParameter('mitigation_day', (60, 88), 80, 2),
        # SampleParameter('mitigation_width', (.05, 20), 10, 2),
    ]

    labels = {
        'r0': r'$R_0$',
        'start_day': 'start day',
        'mitigation_factor': 'mitigation factor',
        'mitigation_day': 'mitigation day',
        'mitigation_width': 'mitigation width',
    }

    fixed_values = dict(
        end_day=np.max(data['t']) + 2,
        population=population,
        age_dist_pop=age_dist_pop,
        initial_cases=10.,
        imports_per_day=1.1,
        mitigation_factor=1,
        mitigation_day=80,
        mitigation_width=1,
    )

    from pydemic.models.neher import NeherModelEstimator
    estimator = NeherModelEstimator(fit_parameters, fixed_values, data)

    num_workers = cpu_count()
    pool = Pool(num_workers)

    # run uniform sampling
    nsamples = 25
    uniform_values, uniform_likelihoods = estimator.sample_uniform(nsamples, pool)

    uniform_likelihoods = uniform_likelihoods.reshape(nsamples, nsamples)
    r0_vals = uniform_values[:, 0].reshape(nsamples, nsamples)
    start_day_vals = uniform_values[:, 1].reshape(nsamples, nsamples)

    max_loc = np.where(uniform_likelihoods == uniform_likelihoods.max())
    uniform_best_fit = dict(zip(estimator.fit_names,
                                [r0_vals[max_loc][0], start_day_vals[max_loc][0]]))
    print('uniform best fit:', uniform_best_fit)

    fig, ax = plt.subplots()
    ax.pcolormesh(r0_vals, start_day_vals, np.exp(uniform_likelihoods))
    ax.set_xlabel('r0')
    ax.set_ylabel('start day')
    fig.savefig('neher_uniform_samples.png')

    # run MCMC
    n_walkers = 32
    n_steps = 200

    initial_positions = estimator.get_initial_positions(n_walkers)
    n_dims = initial_positions.shape[-1]

    num_workers = cpu_count()
    pool = Pool(num_workers)

    sampler = emcee.EnsembleSampler(n_walkers, n_dims, estimator, pool=pool)
    sampler.run_mcmc(initial_positions, n_steps, progress=True)
    # pool.terminate()

    discard = 100
    thin = 10
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    likelihoods = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
    import corner
    fig = corner.corner(flat_samples,
                        labels=[labels[key] for key in estimator.fit_names],
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 12})
    fig.savefig('neher_emcee_samples.png')

    mcmc = np.percentile(flat_samples, [16, 50, 84], axis=0)
    q = np.diff(mcmc, axis=0)
    best_fit = dict(zip(estimator.fit_names, mcmc[1, :]))
    uncertainties = dict(zip(estimator.fit_names, q.T))
    print('50%% best fit:', best_fit)
    print('50%% uncertainties:', uncertainties)

    flat_samples[np.argmax(likelihoods)]
    best_fit = dict(zip(estimator.fit_names, flat_samples[np.argmax(likelihoods)]))
    best_parameters = {**best_fit, **estimator.fixed_values}
    print('fit of maximum L:', best_fit)

    def days_to_dates(days):
        from datetime import datetime, timedelta
        return [datetime(2020, 1, 1) + timedelta(float(x)) for x in days]

    tt = np.linspace(best_parameters['start_day']+1,
                     best_parameters['end_day'], 1000)
    result = estimator.get_model_data(tt, **best_parameters)

    fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    ax[0].semilogy(days_to_dates(data['t']), np.diff(data['dead'], prepend=0),
                   'x', c='r', ms=4, markeredgewidth=1,
                   label='reported')
    ax[0].semilogy(days_to_dates(result.t), result.y['dead'].sum(axis=-1),
                   '-', linewidth=1.1, color='k',
                   label='deterministic')
    ax[0].set_ylabel("daily deaths")
    ax[0].set_ylim(.9, .5 * ax[0].get_ylim()[1])

    cumulative_estimator = NeherModelEstimator(fit_parameters, fixed_values, data,
                                               fit_cumulative=True)
    tt = np.linspace(best_parameters['start_day'], best_parameters['end_day'], 1000)
    result = cumulative_estimator.get_model_data(tt, **best_parameters)

    ax[1].semilogy(days_to_dates(data['t']), data['dead'],
                   'x', c='r', ms=4, markeredgewidth=1,
                   label='reported deaths')

    ax[1].semilogy(days_to_dates(result.t), result.y['dead'].sum(axis=-1),
                   '-', linewidth=1.1, color='k',
                   label='deterministic')

    ax[1].set_ylabel("cumulative deaths")
    ax[1].set_ylim(.95, .5 * ax[1].get_ylim()[1])

    ax[0].legend()
    ax[0].grid()
    ax[1].grid()

    model_deaths = result.y['dead'].sum(axis=-1)
    uncert = np.sqrt(model_deaths)
    ax[1].fill_between(
        days_to_dates(result.t),
        model_deaths + uncert,
        model_deaths - uncert,
        alpha=.3, color='b',
    )

    fig.tight_layout()
    title = '\n'.join(
        [labels[key]+' = '+('%.3f' % val) for key, val in best_fit.items()]
    )
    fig.suptitle(title, va='baseline', y=1.)
    fig.savefig('neher_best_fit_to_deaths.png', bbox_inches='tight')
