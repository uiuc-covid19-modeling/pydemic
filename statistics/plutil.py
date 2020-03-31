import matplotlib as mpl
mpl.use('agg')
import matplotlib.dates as mdates
from datetime import timedelta


def get_date_before(target_date, x):
    # useful for plotting purposes
    return target_date + timedelta(x)


def plot_data(ax, x, y, target_date, c='k'):
    dates = [get_date_before(target_date, i) for i in x]
    ax.plot(dates, y, 'x', c=c, ms=4, markeredgewidth=2, label='reported deaths')


def format_axis(fig, ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylim(ymin=0.8)
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.tick_params(labelright=True, right=True)
    ax.grid()
    fig.autofmt_xdate()


def plot_model(ax, data_mean, data_abv, data_bel, target_date, c='k'):
    dates = [get_date_before(target_date, -x)
                             for x in [x for x in range(data_mean.shape[0])][::-1]]
    ax.plot(dates, data_mean, '-', lw=2, c=c)
    if data_abv is not None and data_bel is not None:
        ax.fill_between(dates, data_abv, data_bel, color=c, alpha=0.2)
