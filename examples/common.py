import numpy as np
import matplotlib.pyplot as plt


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nthreads", default=None, type=int,
                    help="how many threads to use")
parser.add_argument("--walkers", default=None, type=int,
                    help="how many walkers to use")
parser.add_argument("--steps", default=2000, type=int,
                    help="how many steps to take")
parser.add_argument("--output", type=str,
                    help="name for output files")
parser.add_argument("--end", type=int, default=1e10,
                    help="final day whose data should be fitted")


def create_unique_filename(name, ext, directory='.'):
    import os
    full_path = os.path.join(directory, name + ext)
    if os.path.exists(full_path):
        i = 2
        full_path = os.path.join(directory, name + '-{i}' + ext)
        while os.path.exists(full_path.format(i=i)):
            i += 1
        return name + '-' + str(i) + ext
    else:
        return name + ext


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


def get_data(Estimator=None, **kwargs):
    if Estimator is None:
        from pydemic.models.neher import NeherModelEstimator
        Estimator = NeherModelEstimator

    tt = np.linspace(kwargs['start_day']+1, kwargs['end_day'], 1000)
    # will become diff data
    model_data = Estimator.get_model_data(tt, **kwargs)
    # will be actual data
    model_data_1 = Estimator.get_model_data(tt-1, **kwargs)
    for key in model_data.y.keys():
        model_data.y[key] -= model_data_1.y[key]

    return model_data_1, model_data


all_labels = {
    'r0': r'$R_0$',
    'start_day': 'start day',
    'fraction_hospitalized': 'hospitalization fraction',
    'p_positive': r'$P_\mathrm{positive}$',
    'p_dead': r'$P_\mathrm{dead}$',
    'incubation_mean': 'incubation delay',
    'incubation_k': r'$k_\mathrm{incubation}$',
    'positive_mean': 'positive delay',
    'positive_k': r'$k_\mathrm{positive}$',
    'icu_mean': 'icu delay',
    'icu_k': r'$k_\mathrm{ICU}$',
    'dead_mean': 'death delay',
    'dead_k': r'$k_\mathrm{death}$',
}

for i in range(10):
    all_labels['mitigation_factor_%d' % i] = r'$M(t_%d)$' % i
    all_labels['mitigation_t_%d' % i] = r'$t_%d$' % i


def plot_deaths_and_positives(data, best_fit, fixed_values,
                              Estimator=None, labels=all_labels):
    from pydemic import days_to_dates

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)

    result, diff = get_data(Estimator, **best_fit, **fixed_values)

    # plot daily results
    dead = diff.y['dead'].sum(axis=-1)
    scatter(ax[0, 0], days_to_dates(data.t), np.diff(data.y['dead'], prepend=0))
    plot_with_quantiles(ax[0, 0], days_to_dates(diff.t), dead, False)
    ax[0, 0].set_ylabel("daily new deaths")

    positive = diff.y['positive'].sum(axis=-1)
    scatter(ax[0, 1], days_to_dates(data.t), np.diff(data.y['positive'], prepend=0))
    plot_with_quantiles(ax[0, 1], days_to_dates(diff.t), positive, False)
    ax[0, 1].set_ylabel("daily new positive")

    # plot cumulative results
    dead = result.y['dead'].sum(axis=-1)
    scatter(ax[1, 0], days_to_dates(data.t), data.y['dead'])
    plot_with_quantiles(ax[1, 0], days_to_dates(result.t), dead, True)
    ax[1, 0].set_ylabel("cumulative deaths")

    positive = result.y['positive'].sum(axis=-1)
    scatter(ax[1, 1], days_to_dates(data.t), data.y['positive'])
    plot_with_quantiles(ax[1, 1], days_to_dates(result.t), positive, True)
    ax[1, 1].set_ylabel("cumulative positive")

    for a in ax.reshape(-1):
        a.grid()
        a.set_ylim(.9, 1.1 * a.get_ylim()[1])

    fig.tight_layout()
    title = '\n'.join([labels[key]+' = '+('%.3f' % val)
                       for key, val in best_fit.items()])
    fig.suptitle(title, va='baseline', y=1.)
    fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0)

    return fig


def get_mitigation_model(**kwargs):
    mitigation_keys = sorted([key for key in kwargs.keys()
                              if key.startswith('mitigation_factor')])
    factors = np.array([kwargs.get(key) for key in mitigation_keys])

    from pydemic.containment import MitigationModel
    return MitigationModel(kwargs['start_day'], kwargs['end_day'],
                           kwargs['mitigation_t'], factors)
