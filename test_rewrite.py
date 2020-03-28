import numpy as np
import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from pydemic import PopulationModel, AgeDistribution, SeverityModel, EpidemiologyModel, ContainmentModel, date_to_ms
from pydemic import Simulation
from pydemic.load import get_country_population_model, get_age_distribution_model
from pydemic.models import NeherModelSimulation


if __name__ == "__main__":

    ### initial conditions
    n_age_groups = 9
    start_date = (2020, 3, 1, 0, 0, 0)
    end_date = (2020, 9, 1, 0, 0, 0)
    POPULATION_NAME = "USA-Illinois"
    AGE_DATA_NAME = "United States of America"
    population = get_country_population_model(POPULATION_NAME)
    population.ICU_beds = int(1.e10) ; population.hospital_beds = int(1.e10)
    age_distribution = get_age_distribution_model(AGE_DATA_NAME)
    severity = SeverityModel(
        id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
        age_group=np.arange(0., 90., 10),
        isolated=np.zeros(n_age_groups),
        confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
        severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
        critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
        fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
    )
    epidemiology = EpidemiologyModel(
        r0=2.7,
        incubation_time=5,
        infectious_period=3,
        length_hospital_stay=4,
        length_ICU_stay=14,
        seasonal_forcing=0.,
        peak_month=0,
        overflow_severity=2
    )
    containment = ContainmentModel(start_date, end_date)
    containment.add_sharp_event((2020, 3, 15), 1.0)


    ### generate, run, and aggregate results for old pydemic model version
    sim = Simulation(population, epidemiology, severity, age_distribution,
                     containment)
    start_time = date_to_ms(start_date)
    end_time = date_to_ms(end_date)
    result = sim(start_time, end_time, lambda x: x)
    og_data = {}
    dkeys = [ 'time', 'susceptible', 'exposed', 'infectious', 'recovered', 'dead' ]
    dates = [ datetime.utcfromtimestamp(x//1000) for x in result['time'] ]
    for key in dkeys:
        og_data[key] = np.sum(result[key], axis=-1)


    ### generate, run, and aggregate results for new pydemic model version
    simulation = NeherModelSimulation(epidemiology, severity, population.imports_per_day, population.population_served, n_age_groups)
    N = population.population_served


    ## working
    y0 = {
        'susceptible': np.array([ int(np.round(x)) for x in np.array(age_distribution.counts)*N/sum(age_distribution.counts) ]),
        'exposed': np.zeros(n_age_groups),
        'infectious': np.zeros(n_age_groups),
        'recovered': np.zeros(n_age_groups),
        'hospitalized': np.zeros(n_age_groups),
        'critical': np.zeros(n_age_groups),
        'dead': np.zeros(n_age_groups)
    }
    i_middle = round(n_age_groups / 2) + 1
    y0['susceptible'][i_middle] -= population.suspected_cases_today
    y0['exposed'][i_middle] += population.suspected_cases_today * 0.7
    y0['infectious'][i_middle] += population.suspected_cases_today * 0.3
    tspan = (0, 30)
    dt = 0.25
    new_result = simulation(tspan, y0, lambda x: x, dt=dt)
    new_dates = [ date(*start_date)+timedelta(x) for x in new_result['time'] ]


    ### make figure
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)

    for key in dkeys[1:]:
        ax1.plot(dates, og_data[key], label=key)
        ax1.plot(new_dates, new_result[key].sum(axis=1))


    # plot on y log scale
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=1)

    # plot x axis as dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.autofmt_xdate()

    # formatting hints
    ax1.legend()
    ax1.set_xlabel('time')
    ax1.set_ylabel('count (persons)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('compare_rewrite.png')










