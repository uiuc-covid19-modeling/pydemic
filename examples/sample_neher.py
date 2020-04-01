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
from datetime import date, datetime, timedelta
import emcee
from multiprocessing import Pool, cpu_count


if __name__ == "__main__":
    # load reported data
    from pydemic.load import get_case_data
    cases = get_case_data("USA-Illinois")
    target_date = date(*cases.last_date)
    death_counts = np.array(cases.deaths)
    data_deaths_gtr_1 = (death_counts > 1)
    death_counts = death_counts[data_deaths_gtr_1]
    data = {'t': np.array(cases.dates)[data_deaths_gtr_1], 'dead': death_counts}

    from pydemic.models import SampleParameter

    fit_parameters = [
        SampleParameter('r0', (2, 4), 3, .2),
        SampleParameter('start_day', (40, 60), 50, 2),
    ]

    labels = {
        'r0': r'$R_0$',
        'start_day': 'start day',
        'mitigation_factor': 'mitigation factor',
        'mitigation_day': 'mitigation day',
        'mitigation_width': 'mitigation width',
    }

    fixed_values = dict(
        end_day=88,
        country='USA-Illinois',
        subregion="United States of America",
        mitigation_factor=1,
        mitigation_day=80,
        mitigation_width=.05,
    )

    from pydemic.models.neher import NeherModelEstimator
    estimator = NeherModelEstimator(fit_parameters, fixed_values, data)

    n_walkers = 32
    n_steps = 100

    initial_positions = estimator.get_initial_positions(n_walkers)
    n_dims = initial_positions.shape[-1]

    num_workers = cpu_count()
    pool = Pool(num_workers)

    sampler = emcee.EnsembleSampler(n_walkers, n_dims, estimator, pool=pool)
    sampler.run_mcmc(initial_positions, n_steps, progress=True)
    # pool.terminate()

    flat_samples = sampler.get_chain(discard=10, thin=10, flat=True)
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
    print('best fit:', best_fit)
    print('uncertainties:', uncertainties)

    fig, ax = plt.subplots(1, 1)

    best_parameters = {**best_fit, **estimator.fixed_values}

    tt = np.linspace(best_parameters['start_day'], best_parameters['end_day'], 1000)

    result = estimator.get_model_data(tt, **best_parameters)
    model_dates = result.t
    model_deaths = result.y['dead'].sum(axis=-1)

    def days_to_dates(days):
        return [datetime(2020, 1, 1) + timedelta(x) for x in days]

    ax.semilogy(days_to_dates(cases.dates), cases.deaths,
                'x', c='r', ms=4, markeredgewidth=1,
                label='reported deaths')

    ax.semilogy(days_to_dates(model_dates), model_deaths,
                '-', linewidth=1.1, color='k', label='deterministic')
    ax.fill_between(
        days_to_dates(model_dates),
        model_deaths + np.sqrt(model_deaths),
        model_deaths - np.sqrt(model_deaths),
        alpha=.3, color='b',
    )
    ax.set_ylabel("count (persons)")
    ax.set_ylim(.95, .5 * ax.get_ylim()[1])
    ax.legend()

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.autofmt_xdate()
    fig.savefig('neher_best_fit.png')
