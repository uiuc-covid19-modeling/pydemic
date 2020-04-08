import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np

pool_size = 4
walkers = 8
steps = 500
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
    #SampleParameter('mitigation_factor', (.01, 1), .9, .1),
    #SampleParameter('mitigation_day', (70, 95), 80, 2),
    #SampleParameter('mitigation_width', (.05, 20), 10, 2),
    #SampleParameter('fraction_hospitalized', (.05, 10), 5, 3),
]

fixed_values = dict(
    end_day=data.t[-1] + 2,
    #population=population,
    #age_dist_pop=age_dist_pop,
    #initial_cases=10.,
    #imports_per_day=1.1,
    #length_ICU_stay=14,
)

estimator = NonMarkovianModelEstimator(
    fit_parameters, fixed_values, data, {'dead': 1, 'positive': 0},
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
