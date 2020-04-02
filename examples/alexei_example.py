import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from pydemic.models import AlexeiModelSimulation
import numpy as np


def save_data(result, filename):
    from datetime import datetime, timedelta
    dates = [ datetime(2020,1,1)+timedelta(days=x) for x in result.t ]
    compartments = {}
    fp = open(filename, 'w')
    fp.write("time\t")
    for compartment in result.compartments:
        compartments[compartment] = result.y[compartment] 
        fp.write(compartment + "\t")
    fp.write("\n")
    for i in range(len(dates)):
        fp.write(dates[i].strftime("%y-%m-%d")+"\t")
        for compartment in compartments:
            fp.write("{0:g}\t".format(compartments[compartment][i].sum()))
        fp.write("\n")
    fp.close()


if __name__ == "__main__":

    """
        
        From blog post: dD/dt = p_d [ICU] / tau

        where we must specify tau : the average ICU stay length (7-9 days for COVID)
                             p_d  : the probability of death in the ICU (from 60% to 80% according to data Alexei has referenced)

        coefficient is tau/p_d ~ 10 days

    """

    tspan = [ (2020,3,1), (2020,6,1) ]
    total_population = 1.e5
    

    # generate simulation
    sim = AlexeiModelSimulation(r0=2.7)
    y0 = sim.get_initial_population(total=total_population)

    # run the simulation
    dense_result = sim.solve_deterministic(tspan, y0)
    times = np.arange(dense_result.t[0], dense_result.t[-1])
    dates = [ datetime(2020,1,1)+timedelta(days=x) for x in times ]
    result = sim.dense_to_logger(dense_result, times)

    # save result
    save_data(result, "data.txt")

    compartments_to_plot = [
        "susceptible",
        "exposed",
        "infectious",
        "removed",
        "hospitalized_died",
        #"hospitalized_cases_base",
        #"hospitalized_died",
    ]

    # plot result
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot(1,1,1)
    for compartment in compartments_to_plot:
        ax1.plot(dates, result.y[compartment], label=compartment)
        print(compartment, result.y[compartment][-1])
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=0.8)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.autofmt_xdate()

    plt.savefig('alexei.png')




"""

fig = plt.figure(figsize=(14, 8))


gspec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
ax = [fig.add_subplot(gspec[:2,0]), fig.add_subplot(gspec[2,0])]

def days_to_dates(days):
    return [datetime(2020, 1, 1) + timedelta(x) for x in days]

ax[0].semilogy(days_to_dates(cases.dates), cases.deaths,
            'x', c='r', ms=6, markeredgewidth=2,
            label='reported deaths')

for label, (result, contain) in results.items():
    ax[0].semilogy(days_to_dates(result.t), result.y['dead'].sum(axis=-1),
                '-', linewidth=1.5, label=label)
    
    ax[1].plot(days_to_dates(result.t), 2.7 * contain(result.t),
                '-', linewidth=1.5, label=label)

ax[0].set_ylabel("count (persons)")
ax[1].set_ylabel(r'$R_0$')
ax[0].set_ylim(.95, .5 * ax[0].get_ylim()[1])
ax[0].legend(bbox_to_anchor=(1, 0), loc='center left')

ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
fig.autofmt_xdate()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('standalone.png')
"""