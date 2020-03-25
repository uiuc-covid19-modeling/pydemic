import requests
import json
import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

from pydemic import PopulationModel, AgeDistribution, SeverityModel, EpidemiologyModel, ContainmentModel
from pydemic import Simulation
from pydemic.load import get_country_population_model, get_age_distribution_model


URL = "http://localhost:8081"

if __name__ == "__main__":

    ## define simulation parameters
    start_time = (2020, 3, 1, 0, 0, 0)
    end_time = (2020, 9, 1, 0, 0, 0)

    ## load population from remote data
    POPULATION_NAME = "USA-Illinois"
    AGE_DATA_NAME = "United States of America"
    population = get_country_population_model(POPULATION_NAME)
    age_distribution = get_age_distribution_model(AGE_DATA_NAME)

    ## set severity model
    n_age_groups = 9
    severity = SeverityModel(
        id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
        age_group=np.arange(0., 90., 10),
        isolated=np.zeros(n_age_groups),
        confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
        severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
        critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
        fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
    )

    ## set epidemiology model
    epidemiology = EpidemiologyModel(
        r0=2.7,
        incubation_time=5,
        infectious_period=3,
        length_hospital_stay=4,
        length_ICU_stay=14,
        seasonal_forcing=0.2,
        peak_month=0,
        overflow_severity=2
    )

    ## set containment model
    containment = ContainmentModel(start_time, end_time)
    containment.add_sharp_event([2020,3,15], 0.5)


    ## the node/javascript wrapper expects some things in a certain foramt
    simulation = {
        "start": list(start_time),
        "end": list(end_time)
    }
    population.populations_by_decade = age_distribution.counts
    containment_dict = containment.get_dictionary()


    ## generate and POST request to javascript api
    body = { "simulation":simulation, "population":population, "containment":containment_dict, "epidemiology":epidemiology }
    r = requests.post(url=URL, data=json.dumps(body))
    data = r.json()
    dkeys = [ 'times', 'suspectible', 'exposed', 'infectious', 'recovered', 'hospitalized', 'critical', 'overflow', 'discharged', 'intensive', 'dead' ]
    dates = [ datetime.utcfromtimestamp(x//1000) for x in data['times'] ]

    ## make figure
    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs[:2,0])
    ax2 = fig.add_subplot(gs[2,0], sharex=ax1)

    for key in dkeys[1:]:
      ax1.plot(dates, data[key], label=key)

    # plot nice hint data
    ax1.axhline(y=population['hospital_beds'],ls=':',c='#999999')
    ax1.axhline(y=population['ICU_beds'],ls=':',c='#999999')

    # plot containment
    mitigation_dates = [ datetime(*x[:-2]) for x in containment["times"] ]
    ax2.plot(mitigation_dates, containment["factors"], 'ok-', lw=2)
    ax2.set_ylim(0,1.2)

    # plot on y log scale
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=1)

    # plot x axis as dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    #ax1.set_xlim(dates[0],dates[-1])
    fig.autofmt_xdate()


    # formatting hints
    ax1.legend()
    ax2.set_xlabel('time')
    ax2.set_ylabel('mitigation factor')
    ax1.set_ylabel('count (persons)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('example.png')





