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

import numpy as np
import pandas as pd
from pydemic.models import SEIRPlusPlusSimulation
from pydemic import MitigationModel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.family'] = [u'serif']
plt.rcParams['font.size'] = 16

state = "Illinois"
from pydemic.data.united_states import nyt, get_population, get_age_distribution
data = nyt(state)
total_population = get_population(state)
age_distribution = get_age_distribution()

tspan = ('2020-02-15', '2020-05-30')

from pydemic.distributions import GammaDistribution

parameters = dict(
    ifr=.003,
    r0=2.3,
    serial_dist=GammaDistribution(mean=4, std=3.25),
    seasonal_forcing_amp=.1,
    peak_day=15,
    incubation_dist=GammaDistribution(5.5, 2),
    p_symptomatic=np.array([0.057, 0.054, 0.294, 0.668, 0.614, 0.83,
                            0.99, 0.995, 0.999]),
    # p_positive=1.5,
    hospitalized_dist=GammaDistribution(6.5, 1.6),
    p_hospitalized=np.array([0.001, 0.003, 0.012, 0.032, 0.049, 0.102,
                             0.166, 0.243, 0.273]),
    discharged_dist=GammaDistribution(9, 6),
    critical_dist=GammaDistribution(3, 1),
    p_critical=.9 * np.array([0.05, 0.05, 0.05, 0.05, 0.063, 0.122,
                              0.274, 0.432, 0.709]),
    dead_dist=GammaDistribution(7.5, 5.),
    p_dead=1.2 * np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5]),
    recovered_dist=GammaDistribution(9, 2.2),
    all_dead_dist=GammaDistribution(3, 3),
    all_dead_multiplier=1.,
)

t0, tf = 50, 140
times = [70, 80]
factors = [1, .48]
mitigation = MitigationModel(t0, tf, times, factors)
fig, ax = plt.subplots(figsize=(12, 4))
_t = np.linspace(t0, tf, 1000)
ax.plot(_t, mitigation(_t))

sim = SEIRPlusPlusSimulation(total_population, age_distribution,
                             mitigation=mitigation, **parameters)

initial_cases = 10
y0 = {}
y0['infected'] = initial_cases * np.array(age_distribution)
y0['susceptible'] = (
    total_population * np.array(age_distribution) - y0['infected']
)

result = sim(tspan, y0, .05)
_t = result.t
result.t = pd.to_datetime(result.t, unit='D', origin='2020-01-01')

plot_compartments = ['infected', 'positive', 'all_dead', 'hospitalized']

fig, ax = plt.subplots(figsize=(10, 6))
for name in plot_compartments:
    ax.plot(result.t, result.y[name].sum(axis=1), label=name)

ax.plot()
ax.set_yscale('log')
ax.set_ylim(ymin=0.8)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
fig.autofmt_xdate()

ax.legend(loc='center left', bbox_to_anchor=(1, .5))
ax.set_xlabel('time')
ax.set_ylabel('count (persons)')

fig = plt.figure(figsize=(12, 8))

import matplotlib.gridspec as gridspec
gspec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
ax = [fig.add_subplot(gspec[:2, 0]), fig.add_subplot(gspec[2, 0])]

ax[0].semilogy(data.index, data.all_dead,
               'x', c='r', ms=4, markeredgewidth=2,
               label='reported daily deaths')

ax[0].semilogy(result.t, result.y['all_dead'].sum(axis=1),
               '-', linewidth=1.5, label='simulated daily deaths')

ax[1].plot(_t, parameters['r0'] * mitigation(_t), '-', linewidth=1.5)

ax[0].set_ylabel("count (persons)")
ax[1].set_ylabel(r'$R_t$')
ax[0].set_ylim(.95, .5 * ax[0].get_ylim()[1])
ax[0].legend()

fig.autofmt_xdate()
ax[0].grid()
ax[1].grid()
fig.subplots_adjust(hspace=0, wspace=0)
# fig.savefig('seirpp_best_fit.png')
