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

from datetime import datetime, timedelta


def plot_quantiles(ax, quantile_logger, log=True, legend=False):

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    compartments = quantile_logger.compartments
    dates = [datetime(2020, 1, 1)+timedelta(x) for x in quantile_logger.t]

    maxpop = sum([quantile_logger.quantile_data[x][-1, 0, :].sum()
                  for x in quantile_logger.quantile_data])

    for i in range(len(compartments)):
        color = colors[i]
        compartment = compartments[i]
        ax.plot(dates,
                quantile_logger.quantile_data[compartment][2, ...].sum(axis=-1),
                color=color, label=compartment, lw=2)
        ax.fill_between(
            dates,
            quantile_logger.quantile_data[compartment][1, ...].sum(axis=-1),
            quantile_logger.quantile_data[compartment][3, ...].sum(axis=-1),
            color=colors[i], alpha=0.5
        )
        ax.fill_between(
            dates,
            quantile_logger.quantile_data[compartment][0, ...].sum(axis=-1),
            quantile_logger.quantile_data[compartment][4, ...].sum(axis=-1),
            color=colors[i], alpha=0.2
        )

    if legend:
        ax.legend(loc='upper right')

    if log:
        ax.set_ylim(ymin=0.8, ymax=2.*maxpop)
        ax.set_yscale('log')


def plot_deterministic(ax, state_logger, log=True, legend=False, force_color=None):

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    compartments = state_logger.compartments
    dates = [datetime(2020, 1, 1)+timedelta(x) for x in state_logger.t]

    maxpop = sum([state_logger.y[x][0, :].sum() for x in state_logger.y])

    for i in range(len(compartments)):
        if force_color is None:
            color = colors[i]
        else:
            color = force_color
        compartment = compartments[i]
        if legend:
            ax.plot(dates, state_logger.y[compartment].sum(axis=-1),
                    color=color, label=compartment)
        else:
            ax.plot(dates, state_logger.y[compartment].sum(axis=-1),
                    color=color)

    if legend:
        ax.legend(loc='upper right')

    if log:
        ax.set_ylim(ymin=0.8, ymax=2.*maxpop)
        ax.set_yscale('log')
