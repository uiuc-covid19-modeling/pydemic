import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np

pool_size = 72
walkers = 144
steps = 2000
discard = 200
thin = 20
population = "USA-Illinois"
age_dist_pop = "United States of America"
from pydemic.data import united_states
data = united_states.get_case_data('IL')
fit_cumulative = False


from pydemic.sampling import SampleParameter
from pydemic.models.nonmarkovian import NonMarkovianModelEstimator
from multiprocessing import Pool
from common import all_labels, plot_deaths_and_positives
import corner


fit_parameters = [
    SampleParameter('r0', (0.5, 8), 3, .2),
    SampleParameter('start_day', (20, 60), 50, 2),
    SampleParameter('p_positive', (0., 1.), 0.5, 0.1),
    SampleParameter('p_dead', (0., 1.), 0.5, 0.1),
    SampleParameter('positive_mean', (4., 11.), 7.5, 1.),
    SampleParameter('icu_mean', (8., 10.), 10., 0.2),
    SampleParameter('dead_mean', (7., 15.), 11, 1.),
    SampleParameter('dead_k', (1., 4.), 2.5, 0.4),
    SampleParameter('mitigation_factor_1', (.01, 1), .5, .1),
    SampleParameter('mitigation_factor_2', (.01, 1), .5, .1),
    SampleParameter('mitigation_factor_3', (.01, 1), .5, .1),
    SampleParameter('mitigation_factor_4', (.01, 1), .5, .1),
]

fixed_values = dict(
    end_day=data.t[-1] + 2,
    #mitigation_t=([60, 67, 74, 81, 88]),
    #mitigation_t=([60, 65, 70, 75, 80, 85]),
    mitigation_t=([65, 70, 75, 80, 85]),
    mitigation_factor_0=1,
    population=population,
    age_dist_pop=age_dist_pop,
    initial_cases=10.,
    imports_per_day=1.1,  # FIXME: currently doesn't do anything
    #length_ICU_stay=14,
)

estimator = NonMarkovianModelEstimator(
    fit_parameters, fixed_values, data, {'dead': 1, 'positive': 1},
    fit_cumulative=fit_cumulative
)

pool = Pool(pool_size)
sampler = estimator.sample_emcee(
    steps, walkers=walkers,  # pool=pool,
    backend_filename="nonmarkovian.h5"
)
pool.close()
pool.join()


flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
likelihoods = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
best_fit = dict(zip(estimator.fit_names, flat_samples[np.argmax(likelihoods)]))

fig = corner.corner(flat_samples,
                    labels=[all_labels[key] for key in estimator.fit_names],
                    quantiles=[0.16, 0.5, 0.84],
                    bins=30,
                    show_titles=True, title_kwargs={"fontsize": 12})
fig.set_size_inches((18, 18))
fig.savefig("imgs/nonmarkovian.png", bbox_inches='tight')

best_parameters = {**best_fit, **estimator.fixed_values}

fig = plot_deaths_and_positives(data, best_fit, fixed_values)
fig.savefig("imgs/nonmarkovian-best-fit.png", bbox_inches='tight')
