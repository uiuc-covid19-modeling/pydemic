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
    # define posterior parameters
    fit_priors = {
        'r0': (2, 4),
        'start_day': (40, 60),
    }
    fit_guesses = {
        'r0': 3.,
        'start_day': 50,
    }
    guess_uncertainties = {
        'r0': .2,
        'start_day': 2,
    }
    fixed_parameters = dict(
        end_day=89,
        country='USA-Illinois',
        subregion="United States of America",
        mitigation_factor=1,
        mitigation_day=70,
        mitigation_width=.05,
    )

    # load reported data
    from pydemic.load import get_case_data
    cases = get_case_data("USA-Illinois")
    target_date = date(*cases.last_date)
    data = {'t': np.array(cases.dates), 'deaths': np.array(cases.deaths)}

    from pydemic.models.neher import NeherModelEstimator
    estimator = NeherModelEstimator(fit_priors, fixed_parameters, data)

    # define sampler parameters
    n_walkers = 64
    n_steps = 200

    # get pool for multi-processing
    num_workers = cpu_count()
    pool = Pool(num_workers)

    # generate emcee sampler
    initial_positions = np.array(
        [guess + np.random.randn(n_walkers) * guess_uncertainties[key]
        for key, guess in fit_guesses.items()]
    ).T
    n_dims = initial_positions.shape[-1]

    sampler = emcee.EnsembleSampler(n_walkers, n_dims, estimator, pool=pool)
    sampler.run_mcmc(initial_positions, n_steps, progress=True)
    pool.terminate()

    discard_n = 100
    flat_samples = sampler.get_chain(discard=discard_n, thin=10, flat=True)

    import corner
    fig = corner.corner(flat_samples, labels=list(fit_priors.keys()))
    fig.savefig('neher_emcee_samples.png')

    mcmc = np.percentile(flat_samples, [16, 50, 84], axis=0)
    q = np.diff(mcmc, axis=0)
    best_fit = dict(zip(fit_guesses.keys(), mcmc[1, :]))
    uncertainties = dict(zip(fit_guesses.keys(), q.T))
    print(best_fit)
    print(uncertainties)

    fig, ax = plt.subplots(1, 1)

    deterministic = estimator.get_model_result(
        **best_fit, **estimator.fixed_parameters
    )
    model_dates = deterministic.t
    model_deaths = deterministic.quantile_data[2, :]

    def days_to_dates(days):
        return [datetime(2020, 1, 1) + timedelta(x) for x in days]

    ax.semilogy(days_to_dates(model_dates), model_deaths,
                '-', color='k', label='deterministic')
    ax.fill_between(
        days_to_dates(model_dates),
        deterministic.quantile_data[1, :],
        deterministic.quantile_data[3, :], alpha=.5
    )
    ax.semilogy(days_to_dates(cases.dates), cases.deaths,
                'x', c='k', ms=4, markeredgewidth=2,
                label='reported deaths')

    ax.set_ylabel("count (persons)")
    # ax.set_xlabel("date")
    ax.set_ylim(.95, .5 * ax.get_ylim()[1])
    ax.legend()

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.autofmt_xdate()
    fig.savefig('neher_best_fit.png')
