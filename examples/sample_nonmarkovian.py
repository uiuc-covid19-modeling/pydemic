import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np

pool_size = 4
walkers = 8
steps = 200
discard = 100
thin = 10
population = "USA-Illinois"
age_dist_pop = "United States of America"
from pydemic.data import united_states
data = united_states.get_case_data('IL')
fit_cumulative = False

from pydemic.sampling import SampleParameter
from pydemic import days_to_dates

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

from pydemic.models.nonmarkovian import NonMarkovianModelEstimator
estimator = NonMarkovianModelEstimator(
    fit_parameters, fixed_values, data, {'dead': 1, 'positive': 0},
    fit_cumulative=fit_cumulative
)


# for sday in range(40,60): #range(30,45): #[30, 45]:
#     parameters = {
#         'r0': 2.8,
#         'start_day': sday
#     }
#     x = estimator.get_log_likelihood(parameters)
#     print(sday, x)

# best_fit = {
#     'r0': 2.8,
#     'start_day': 52
# }




# best_fit = {
#     'r0': 2.8,
#     'start_day': 55.
# }
# tt = np.linspace(best_fit['start_day']+1, fixed_values['end_day'], 1000)
# data = estimator.get_model_data(tt-1, **best_fit, **fixed_values)

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,8))
# ax = plt.subplot(1,1,1)
# #ax2 = plt.subplot(2,1,2)
# for key in data.y:
#     ax.plot(days_to_dates(data.t), data.y[key], label=key)

# fig.autofmt_xdate()
# #ax.plot()
# ax.legend()
# ax.set_ylim(ymin=0.8, ymax=2.e7)
# ax.set_yscale('log')
# plt.savefig('imgs/test.png')

# exit()
# data2 = estimator.get_model_data(tt, **best_fit, **fixed_values)

# import matplotlib.pyplot as plt
# ax1 = plt.subplot(2,1,1)
# ax2 = plt.subplot(2,1,2)
# #ax3 = plt.subplot(3,1,3)
# ax1.plot(data.t, data.y['infected'])
# ax2.plot(data.t, data.y['positive'])
# ax1.set_yscale('log')
# ax2.set_yscale('log')
# plt.savefig('imgs/test.png')
# print(data)

# exit()

# fig = plot_deaths_and_positives(data, best_fit, fixed_values)
# fig.savefig("imgs/nonmarkovian-best-fit.pdf", bbox_inches='tight')



from multiprocessing import Pool
pool = Pool(pool_size)
sampler = estimator.sample_emcee(
    steps, walkers=walkers, #pool=pool,
    backend_filename="nonmarkovian.h5"
)
pool.close()
pool.join()


flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
likelihoods = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
best_fit = dict(zip(estimator.fit_names, flat_samples[np.argmax(likelihoods)]))

from common import all_labels, plot_deaths_and_positives

import corner
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


