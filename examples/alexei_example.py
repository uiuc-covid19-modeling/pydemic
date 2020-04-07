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

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from pydemic import TrackedSimulation
from pydemic import days_to_dates


if __name__ == "__main__":

    tspan = (55., 150.)


    if False:

        simulation = TrackedSimulation(tspan, dt=1.)

        import numpy as np
        n_tracks = 6
        n_demographics = 1
        y0 = np.zeros((n_tracks, n_demographics))

        track_names = list(simulation.tracks.keys())
        for i in range(len(track_names)):
            if track_names[i] == "population":
                y0[i,:] = 1.e6
            elif track_names[i] == "susceptible":
                y0[i,:] = 1.e6 - 1
            elif track_names[i] == "infected":
                y0[i,:] = 1
        
        import time
        then = time.time()
        result = simulation(tspan, y0)
        now = time.time()
        print("dt = {0:g} s".format(now - then))



    if True:
        ## FIXME: doing a naive study of dt=1., 0.5, 0.25, 0.1, 0.05, 0.025, 0.01
        ##        suggests that 0.05 is sufficient, but a more systematic study
        ##        is probably worthwhile. it might also be possible to automatically
        ##        translate some reactions into passive "after the fact" translations
        simulation = TrackedSimulation(tspan, dt=0.05)
        

        import numpy as np
        n_tracks = 6
        n_demographics = 1

        y0 = simulation.get_y0(population=1.e6, infected=1.)
        
        import time
        then = time.time()
        result = simulation(tspan, y0)
        now = time.time()
        print("dt = {0:g} s".format(now - then))

        #print(result.tracks)

        data = {}
        data['susceptible'] = result.tracks['susceptible'].sum(axis=0) # sum over demographics
        data['infected'] = np.cumsum(result.tracks['infected'].sum(axis=0))
        data['symptomatic'] = np.cumsum(result.tracks['symptomatic'].sum(axis=0))
        #susceptible = result.tracks['susceptible']

        print(result.t.shape)


        plt.figure(figsize=(8,6))
        ax1 = plt.subplot(1,1,1)

        for key in data:
            #ax1.plot(days_to_dates(result.t), result.tracks[key], label=key)
            ax1.plot(days_to_dates(result.t), data[key], label=key)
            print(key, data[key].shape)

        ax1.legend()

        ax1.set_yscale('log')
        ax1.set_ylim(ymin=0.8, ymax=2.e6)

        plt.savefig('alexei_example.png')

    if False:

        # this would take more time to code up than is worth it, probably

        simulation = TrackedSimulation(tspan)

        import numpy as np
        n_tracks = 6
        n_demographics = 1
        y0 = np.zeros((n_tracks, n_demographics))

        print(simulation.tracks.keys())

        track_names = list(simulation.tracks.keys())
        for i in range(len(track_names)):
            if track_names[i] == "population":
                y0[i,:] = 1.e6
            elif track_names[i] == "susceptible":
                y0[i,:] = 1.e6 - 1
            elif track_names[i] == "infected":
                y0[i,:] = 1

        result = simulation(tspan, y0, dt=1.)




    """
    population = "USA-Illinois"
    age_dist_pop = "United States of America"
    from pydemic.data.us import get_case_data
    data = get_case_data('IL')

    i_end = np.searchsorted(data.t, 90)
    data.t = data.t[:i_end]
    data.y = {'dead': np.array(data.y['death'][:i_end]),
              'cases': np.array(data.y['positive'][:i_end])}

    from pydemic.sampling import SampleParameter

    fit_parameters = [
        SampleParameter('r0', (1, 5), 3, .2),
        SampleParameter('start_day', (40, 60), 50, 2),
    ]

    fixed_values = dict(
        end_day=data.t[-1] + 2,
        population=population,
        age_dist_pop=age_dist_pop,
        initial_cases=10.,
        imports_per_day=1.1,
        mitigation_day=81,
        mitigation_width=3,
        mitigation_factor=1,
        hospital_case_ratio=1.2,
    )

    from pydemic.models.neher import NeherModelEstimator
    estimator = NeherModelEstimator(
        fit_parameters, fixed_values, data,
        fit_cumulative=True,
        weights={'dead': 1, 'cases': 0, 'critical': 0}
    )

    from multiprocessing import Pool
    pool = Pool(32)

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

    from pydemic import days_to_dates
    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt

    plt.rc('font', family='serif', size=12)

    labels = {
        'r0': r'$R_0$',
        'start_day': 'start day',
        'mitigation_factor': 'mitigation factor',
        'mitigation_day': 'mitigation day',
        'mitigation_width': 'mitigation width',
        'hospital_case_ratio': 'hospital/case ratio'
    }

    fig, ax = plt.subplots()
    ax.pcolormesh(r0_vals, start_day_vals, np.exp(uniform_likelihoods))
    ax.set_xlabel('r0')
    ax.set_ylabel('start day')
    fig.savefig('neher_uniform_samples.png')

    # run MCMC
    walkers = 64
    steps = 200
    sampler = estimator.sample_emcee(steps, walkers=walkers, pool=pool)
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

    def scatter(ax, x, y, color='r', ms=4, markeredgewidth=1, label=None):
        ax.semilogy(x, y,
                    'x', color=color, ms=ms, markeredgewidth=markeredgewidth,
                    label=label)

    def plot_with_quantiles(ax, x, y, quantiles=True, label=None):
        ax.semilogy(x, y,
                    '-', linewidth=1.1, color='k',
                    label=label)

        if quantiles:
            ax.fill_between(
                x, y + np.sqrt(y), y - np.sqrt(y),
                alpha=.3, color='b'
            )

    def get_data(params):
        tt = np.linspace(params['start_day']+1, params['end_day'], 1000)
        # will become diff data
        model_data = NeherModelEstimator.get_model_data(tt, **params)
        # will be actual data
        model_data_1 = NeherModelEstimator.get_model_data(tt-1, **params)
        for key in model_data.y.keys():
            model_data.y[key] -= model_data_1.y[key]

        return model_data_1, model_data

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)

    result, diff = get_data(best_parameters)

    # plot daily results
    dead = diff.y['dead'].sum(axis=-1)
    cases = (best_parameters['hospital_case_ratio']
             * diff.y['hospitalized_tracker'].sum(axis=-1))

    scatter(ax[0, 0], days_to_dates(data.t), np.diff(data.y['dead'], prepend=0))
    plot_with_quantiles(ax[0, 0], days_to_dates(diff.t), dead, False)
    ax[0, 0].set_ylabel("daily new deaths")

    scatter(ax[0, 1], days_to_dates(data.t), np.diff(data.y['cases'], prepend=0))
    plot_with_quantiles(ax[0, 1], days_to_dates(diff.t), cases, False)
    ax[0, 1].set_ylabel("daily new cases")

    # plot cumulative results
    dead = result.y['dead'].sum(axis=-1)
    cases = (best_parameters['hospital_case_ratio']
             * result.y['hospitalized_tracker'].sum(axis=-1))

    scatter(ax[1, 0], days_to_dates(data.t), data.y['dead'])
    plot_with_quantiles(ax[1, 0], days_to_dates(result.t), dead, True)
    ax[1, 0].set_ylabel("cumulative deaths")

    scatter(ax[1, 1], days_to_dates(data.t), data.y['cases'])
    plot_with_quantiles(ax[1, 1], days_to_dates(result.t), cases, True)
    ax[1, 1].set_ylabel("cumulative cases")

    for a in ax.reshape(-1):
        a.grid()
        a.set_ylim(.9, .5 * a.get_ylim()[1])

    fig.tight_layout()
    title = '\n'.join([labels[key]+' = '+('%.3f' % val)
                       for key, val in best_fit.items()])
    fig.suptitle(title, va='baseline', y=1.)
    fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0)
    fig.savefig('neher_best_fit_to_deaths.png', bbox_inches='tight')

    pool.close()
    pool.join()
    """
