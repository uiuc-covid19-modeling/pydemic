import requests  # pylint: disable=E0401
import json
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

from pydemic import SeverityModel, EpidemiologyModel, ContainmentModel, date_to_ms
from pydemic.data.neher_load import get_population_model, get_age_distribution_model

import os
import sys
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cwd, "../test/"))
from neher_port import NeherPortSimulation as Simulation  # pylint: disable=E0401

URL = "http://localhost:8081"

if __name__ == "__main__":
    # define simulation parameters
    start_time = (2020, 3, 1, 0, 0, 0)
    end_time = (2020, 9, 1, 0, 0, 0)

    # load population from remote data
    POPULATION_NAME = "USA-Illinois"
    AGE_DATA_NAME = "United States of America"
    population = get_population_model(POPULATION_NAME)
    age_distribution = get_age_distribution_model(AGE_DATA_NAME)

    # set severity model
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

    # set epidemiology model
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

    # set containment model
    containment = ContainmentModel(start_time, end_time)
    containment.add_sharp_event((2020, 3, 15), 0.6)

    # the node/javascript wrapper expects some things in a certain foramt
    simulation = {
        "start": list(start_time),
        "end": list(end_time)
    }
    population.populations_by_decade = age_distribution.counts
    containment_dict = containment.get_dictionary()
    severities = []
    severity_labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69",
                       "70-79", "80+"]
    for i in range(len(severity.id)):
        dobj = {
            "id": int(severity.id[i]),
            "ageGroup": severity_labels[i],
            "isolated": severity.isolated[i],
            "confirmed": severity.confirmed[i],
            "severe": severity.severe[i],
            "critical": severity.critical[i],
            "fatal": severity.fatal[i]
        }
        severities.append(dobj)

    # generate and POST request to javascript api
    body = {
        "simulation": simulation,
        "population": population,
        "containment": containment_dict,
        "epidemiology": epidemiology,
        "severities": severities
    }
    r = requests.post(url=URL, data=json.dumps(body))
    data = r.json()
    dkeys = ['time', 'susceptible', 'exposed', 'infectious', 'recovered',
             'hospitalized', 'critical', 'overflow', 'discharged', 'intensive',
             'dead']
    dates = [datetime.utcfromtimestamp(x//1000) for x in data['time']]

    # define simulation parameters
    # Differs from above because javascript/node (above) does not expect UTC.
    start_date = (2020, 3, 1, 6, 0, 0)
    end_date = (2020, 9, 1, 0, 0, 0)

    # create simulation object
    # note: broken now that Simulation -> NeherModelSimulation and isn't in pydemic
    sim = Simulation(population, epidemiology, severity, age_distribution,
                     containment)

    start_time = date_to_ms(start_date)
    end_time = date_to_ms(end_date)
    result = sim(start_time, end_time, lambda x: x)

    data2 = {}
    dates2 = [datetime.utcfromtimestamp(x//1000) for x in result.t]

    for key in dkeys[1:]:
        data2[key] = np.sum(result.y[key], axis=-1)

    # make figure
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)

    for key in dkeys[1:]:
        ax1.plot(dates, data[key], '-', label=key)
    for key in dkeys[1:]:
        ax1.plot(dates2, data2[key], '--')

    # plot on y log scale
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=1)

    # plot x axis as dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.set_xlim(dates[0], dates[-1])
    fig.autofmt_xdate()

    # formatting hints
    ax1.legend()
    ax2.set_xlabel('time')
    ax2.set_ylabel('mitigation factor')
    ax1.set_ylabel('count (persons)')

    # plot containment
    mitigation_dates = [datetime(*x[:-2]) for x in containment_dict["times"]]
    ax2.plot(mitigation_dates, containment_dict["factors"], 'ok-', lw=2)
    ax2.set_ylim(0, 1.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparison.png')

    print(" - please check comparison plot: \"comparison.png\"")
