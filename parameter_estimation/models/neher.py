import numpy as np
from datetime import datetime, timedelta
from pydemic.models import NeherModelSimulation
from pydemic import (PopulationModel, AgeDistribution, SeverityModel,
                     EpidemiologyModel, ContainmentModel, QuantileLogger, StateLogger)
from pydemic.load import get_age_distribution_model, get_country_population_model
from scipy.interpolate import interp1d


def get_model_result(model_parameters, dt=0.1, n_samples=100, run_stochastic=False):
    # set start date and end date based on offset (for single parameter)
    start_date = datetime(2020, 1, 1) + timedelta(model_parameters['start_day'])
    end_date = datetime(2020, 1, 1) + timedelta(model_parameters['end_day'])
    start_date = (2020, start_date.month, start_date.day,
                  start_date.hour, start_date.minute, start_date.second)
    end_date = (2020, end_date.month, end_date.day,
                end_date.hour, end_date.minute, end_date.second)

    # define containment event
    containment = ContainmentModel((2019,1,1), (2020,12,1), is_in_days=True)
    cdate = datetime(2020,1,1) + timedelta(days=model_parameters['mitigation_day'])
    containment_date = (cdate.year, cdate.month, cdate.day)
    containment_factor = 1.0
    containment_width = 0.05
    if 'mitigation' in model_parameters:
        containment_factor = model_parameters['mitigation']
    if 'mitigation_width' in model_parameters:
        containment_width = model_parameters['mitigation_width']
    containment.add_sharp_event(containment_date, containment_factor, dt_days=containment_width)

    # set parameters
    country = "Italy"
    epidemiology, severity, population, age_distribution = get_default_parameters(population_name=country, age_distribution_name=country)
    n_age_groups = len(age_distribution.counts)

    # load parameters from model_params
    epidemiology.r0 = model_parameters['r0']
    # FIXME: do this better

    # generate neher model simulation
    simulation = NeherModelSimulation(
        epidemiology, severity, population.imports_per_day,
        n_age_groups, containment
    )
    y0 = simulation.get_initial_population(population, age_distribution)

    if False or run_stochastic:
        # run simulation
        logger = QuantileLogger()
        return simulation([start_date, end_date], y0, dt, samples=n_samples, stochastic_method='tau_leap', logger=logger)

    if True:
        # deterministic
        deterministic = simulation([start_date, end_date], y0, dt)
        quantile_data = np.zeros((5, deterministic.t.shape[0]))
        mean = deterministic.y['dead'].sum(axis=1)
        std_dev = np.sqrt(mean)
        quantile_data[1, :] = mean - std_dev
        quantile_data[2, :] = mean
        quantile_data[3, :] = mean + std_dev
        deterministic.quantile_data = quantile_data
        return deterministic

    if False:
        # DeterministicSimulation class
        result = simulation([start_date, end_date], y0, dt)
        t = result.t
        times = np.arange(t.min(), t.max()+0.1, step=0.25)
        determ_sol = {comp: result.sol(
            times)[i] for i, comp in enumerate(simulation.compartments)}
        quantile_data = np.zeros((5, times.shape[0]))
        mean = determ_sol['dead']
        std_dev = np.sqrt(mean)
        quantile_data[1, :] = mean - std_dev
        quantile_data[2, :] = mean
        quantile_data[3, :] = mean + std_dev
        logger = StateLogger()
        logger.quantile_data = quantile_data
        return logger


def align_pad_left(a1, a2):
    l1 = len(a1)
    l2 = len(a2)
    if l1 < l2:
        return np.pad(a1, (l2-l1, 0), 'constant'), a2
    else:
        return a1, np.pad(a2, (l1-l2, 0), 'constant')


def align_right(arr, length):
    mylen = len(arr)
    if mylen > length:
        return arr[-length:]
    return np.pad(arr, (length-mylen, 0), 'constant')


def calculate_likelihood_for_model(model_parameters, dates, deaths, n_samples=100):
    deterministic = get_model_result(model_parameters)

    model_dates = deterministic.t
    model_deaths = deterministic.quantile_data[2, :]

    # interpolate dates to exact times
    ifunc = interp1d(model_dates, model_deaths,
                      bounds_error=False, fill_value=(0, 0))
    model_deaths = ifunc(dates)

    # cut off deaths < 2
    i_to_cut = np.argmax(deaths > 1)
    deaths = deaths[i_to_cut:]
    model_deaths = model_deaths[i_to_cut:]

    return - 0.5 * np.sum(np.power(np.log(deaths)-np.log(model_deaths), 2.))


def get_default_parameters(population_name='USA-Illinois',
                           age_distribution_name='United States of America'):

    population_name = "ITA-Lombardia"
    population = get_country_population_model(population_name)
    population['suspected_cases_today'] = 10
    population['imports_per_day'] = 1.1
    population['ICU_beds'] = 1.e10
    population['hospital_beds'] = 1.e10

    age_distribution = get_age_distribution_model(age_distribution_name)
    n_age_groups = len(age_distribution.counts)

    compartments = ["susceptible", "exposed", "infectious",
        "recovered", "hospitalized", "critical", "dead"]

    # agrees with covid19-scenarios.now.sh as of 2020.03.30
    severity = SeverityModel(
        id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
        age_group=np.arange(0., 90., 10),
        isolated=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
        severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
        critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
        fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
    )

    epidemiology = EpidemiologyModel(
        r0=3.7,
        incubation_time=1,
        infectious_period=5,
        length_hospital_stay=7,
        length_ICU_stay=7,
        seasonal_forcing=0.2,
        peak_month=0,
        overflow_severity=2
    )

    return epidemiology, severity, population, age_distribution

