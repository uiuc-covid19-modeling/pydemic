import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np

pool_size = 4
walkers = 16
steps = 200
discard = 50
thin = 10
population = "USA-Illinois"
age_dist_pop = "United States of America"
from pydemic.data import united_states
data = united_states.get_case_data('IL')
fit_cumulative = False


from pydemic.sampling import SampleParameter
from pydemic.models.nonmarkovian import NonMarkovianModelEstimator
from multiprocessing import Pool
from common import all_labels, plot_deaths_and_positives, get_data
from common import plot_deaths_and_positives_with_ax
import corner


fit_parameters = [
    SampleParameter('r0', (0.5, 8), 3, .2),
    SampleParameter('start_day', (20, 60), 50, 2),
    SampleParameter('p_positive', (0., 1.), 0.5, 0.1),
    SampleParameter('p_dead', (0., 1.), 0.5, 0.1),
    SampleParameter('mitigation_factor_1', (.01, 1), .5, .1),
    SampleParameter('mitigation_factor_2', (.01, 1), .5, .1),
    SampleParameter('mitigation_factor_3', (.01, 1), .5, .1),
    SampleParameter('mitigation_factor_4', (.01, 1), .5, .1),
]

fixed_values = dict(
    end_day=data.t[-1] + 2,
    mitigation_t=([60, 67, 74, 81, 88]),
    mitigation_factor_0=1,
    population=population,
    age_dist_pop=age_dist_pop,
    initial_cases=10.,
    imports_per_day=1.1, # FIXME: currently doesn't do anything
)

# just plot a model
best_fit = { 
    'r0': 2.,
    'start_day': 50.,
    'p_positive': 0.1,
    'p_dead': 1.0,
    'mitigation_factor_1': 1.,
    'mitigation_factor_2': 1.,
    'mitigation_factor_3': 1.,
    'mitigation_factor_4': 1.,
}


from common import scatter, plot_with_quantiles
from pydemic import days_to_dates
import matplotlib.pyplot as plt



# best_fit['p_dead'] = 0.01
# result, diff = get_data(**best_fit, **fixed_values)
# dead = diff.y['dead'].sum(axis=-1)
# scatter(ax[0, 0], days_to_dates(data.t), np.diff(data.y['dead'], prepend=0))
# plot_with_quantiles(ax[0, 0], days_to_dates(diff.t), dead, False)
# ax[0, 0].set_ylabel("daily new deaths")

# best_fit['p_dead'] = 0.02
# result, diff = get_data(**best_fit, **fixed_values)
# dead = diff.y['dead'].sum(axis=-1)
# scatter(ax[0, 0], days_to_dates(data.t), np.diff(data.y['dead'], prepend=0))
# plot_with_quantiles(ax[0, 0], days_to_dates(diff.t), dead, False)
# ax[0, 0].set_ylabel("daily new deaths")

if True:
    # simple
    fig = plot_deaths_and_positives(data, best_fit, fixed_values)
    fig.savefig('imgs/explorations.png')
    exit()


def set_all_after(index, value, d):
    for i in range(index,5):
        d['mitigation_factor_{0:d}'.format(i)] = value
    return d

# p_dead
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
ax = ax.T
best_fit['p_dead'] = 0.1
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='k')
best_fit['p_dead'] = 0.25
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='r')
best_fit['p_dead'] = 0.5
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='g')
best_fit['p_dead'] = 1.0
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='b')
best_fit['p_dead'] = 1.0
ax[0,0].set_ylim(0.8, 100)
ax[1,0].set_ylim(0.8, 800)
ax[0,1].set_ylim(0.8, 2000)
ax[1,1].set_ylim(0.8, 30000)
fig.savefig("imgs/exploration_nonmarkovian_pdead.png", bbox_inches='tight')


# p_dead
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
ax = ax.T
best_fit = set_all_after(4, 1., best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='k', quantiles=False)
best_fit = set_all_after(4, 0.75, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='r', quantiles=False)
best_fit = set_all_after(4, 0.5, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='g', quantiles=False)
best_fit = set_all_after(4, 0.25, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='b', quantiles=False)
ax[0,0].set_ylim(0.8, 100)
ax[1,0].set_ylim(0.8, 800)
ax[0,1].set_ylim(0.8, 2000)
ax[1,1].set_ylim(0.8, 30000)
for i in range(1,5):
    date = fixed_values['mitigation_t'][i]
    ax[0,0].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[1,0].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[0,1].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[1,1].axvline(x=days_to_dates([date])[0], color='k', ls=':')
best_fit = set_all_after(4, 1., best_fit)
fig.savefig("imgs/exploration_nonmarkovian_m_4.png", bbox_inches='tight')

# p_dead
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
ax = ax.T
best_fit = set_all_after(3, 1., best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='k', quantiles=False)
best_fit = set_all_after(3, 0.75, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='r', quantiles=False)
best_fit = set_all_after(3, 0.5, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='g', quantiles=False)
best_fit = set_all_after(3, 0.25, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='b', quantiles=False)
ax[0,0].set_ylim(0.8, 100)
ax[1,0].set_ylim(0.8, 800)
ax[0,1].set_ylim(0.8, 2000)
ax[1,1].set_ylim(0.8, 30000)
for i in range(1,5):
    date = fixed_values['mitigation_t'][i]
    ax[0,0].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[1,0].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[0,1].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[1,1].axvline(x=days_to_dates([date])[0], color='k', ls=':')
best_fit = set_all_after(3, 1., best_fit)
fig.savefig("imgs/exploration_nonmarkovian_m_3.png", bbox_inches='tight')

# p_dead
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
ax = ax.T
best_fit = set_all_after(2, 1., best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='k', quantiles=False)
best_fit = set_all_after(2, 0.75, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='r', quantiles=False)
best_fit = set_all_after(2, 0.5, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='g', quantiles=False)
best_fit = set_all_after(2, 0.25, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='b', quantiles=False)
ax[0,0].set_ylim(0.8, 100)
ax[1,0].set_ylim(0.8, 800)
ax[0,1].set_ylim(0.8, 2000)
ax[1,1].set_ylim(0.8, 30000)
for i in range(1,5):
    date = fixed_values['mitigation_t'][i]
    ax[0,0].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[1,0].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[0,1].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[1,1].axvline(x=days_to_dates([date])[0], color='k', ls=':')
best_fit = set_all_after(2, 1., best_fit)
fig.savefig("imgs/exploration_nonmarkovian_m_2.png", bbox_inches='tight')

# p_dead
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
ax = ax.T
best_fit = set_all_after(1, 1., best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='k', quantiles=False)
best_fit = set_all_after(1, 0.75, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='r', quantiles=False)
best_fit = set_all_after(1, 0.5, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='g', quantiles=False)
best_fit = set_all_after(1, 0.25, best_fit)
fig = plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, fmt='b', quantiles=False)
ax[0,0].set_ylim(0.8, 100)
ax[1,0].set_ylim(0.8, 800)
ax[0,1].set_ylim(0.8, 2000)
ax[1,1].set_ylim(0.8, 30000)
for i in range(1,5):
    date = fixed_values['mitigation_t'][i]
    ax[0,0].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[1,0].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[0,1].axvline(x=days_to_dates([date])[0], color='k', ls=':')
    ax[1,1].axvline(x=days_to_dates([date])[0], color='k', ls=':')
best_fit = set_all_after(2, 1., best_fit)
fig.savefig("imgs/exploration_nonmarkovian_m_1.png", bbox_inches='tight')




exit()


exit()
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
