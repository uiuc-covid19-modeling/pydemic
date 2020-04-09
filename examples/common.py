import numpy as np
import matplotlib.pyplot as plt


def scatter(ax, x, y, color='r', ms=4, markeredgewidth=1, label=None):
    ax.semilogy(x, y,
                'x', color=color, ms=ms, markeredgewidth=markeredgewidth,
                label=label)


def plot_with_quantiles(ax, x, y, quantiles=True, label=None, fmt="k"):
    ax.semilogy(x, y,
                '-', linewidth=1.1, color=fmt,
                label=label)

    if quantiles:
        ax.fill_between(
            x, y + np.sqrt(y), y - np.sqrt(y),
            alpha=.3, color='b'
        )


def get_data(Estimator=None, **kwargs):
    if Estimator is None:
        from pydemic.models.nonmarkovian import NonMarkovianModelEstimator
        Estimator = NonMarkovianModelEstimator

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
    'mitigation_factor': 'mitigation factor',
    'mitigation_day': 'mitigation day',
    'mitigation_width': 'mitigation width',
    'fraction_hospitalized': 'hospitalization fraction',
    'p_positive': 'positive test ratio',
    'p_dead': 'death ratio',
    'positive_mean': 'positive delay',
    'icu_mean': 'icu delay',
    'dead_mean': 'death delay'
}
for i in range(20):
    all_labels['mitigation_factor_%d' % i] = r'$M_{%d}$' % i


def plot_deaths_and_positives(data, best_fit, fixed_values, labels=None):

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)

    return plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, labels=labels)


def plot_deaths_and_positives_with_ax(fig, ax, data, best_fit, fixed_values, labels=None, fmt="k", quantiles=True):
    if labels is None:
        labels = all_labels

    from pydemic import days_to_dates

    result, diff = get_data(**best_fit, **fixed_values)

    # plot daily results
    dead = diff.y['dead'].sum(axis=-1)
    scatter(ax[0, 0], days_to_dates(data.t), np.diff(data.y['dead'], prepend=0))
    plot_with_quantiles(ax[0, 0], days_to_dates(diff.t), dead, False, fmt=fmt)
    ax[0, 0].set_ylabel("daily new deaths")

    positive = diff.y['positive'].sum(axis=-1)
    scatter(ax[0, 1], days_to_dates(data.t), np.diff(data.y['positive'], prepend=0))
    plot_with_quantiles(ax[0, 1], days_to_dates(diff.t), positive, False, fmt=fmt)
    ax[0, 1].set_ylabel("daily new positive")

    # plot cumulative results
    dead = result.y['dead'].sum(axis=-1)
    scatter(ax[1, 0], days_to_dates(data.t), data.y['dead'])
    plot_with_quantiles(ax[1, 0], days_to_dates(result.t), dead, quantiles, fmt=fmt)
    ax[1, 0].set_ylabel("cumulative deaths")

    positive = result.y['positive'].sum(axis=-1)
    scatter(ax[1, 1], days_to_dates(data.t), data.y['positive'])
    plot_with_quantiles(ax[1, 1], days_to_dates(result.t), positive, quantiles, fmt=fmt)
    ax[1, 1].set_ylabel("cumulative positive")

    for a in ax.reshape(-1):
        a.grid()
        a.set_ylim(.9, .5 * a.get_ylim()[1])

    fig.tight_layout()
    title = '\n'.join([labels[key]+' = '+('%.3f' % val)
                       for key, val in best_fit.items()])
    fig.suptitle(title, va='baseline', y=1.)
    fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0)

    return fig
