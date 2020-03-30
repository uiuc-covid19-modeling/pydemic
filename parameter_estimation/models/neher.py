import numpy as np
from datetime import datetime, timedelta 
from pydemic.models import NeherModelSimulation
from pydemic import (PopulationModel, AgeDistribution, SeverityModel,
                     EpidemiologyModel, ContainmentModel, QuantileLogger)


def get_model_result(model_parameters, dt, n_samples=100):

    # set start date and end date based on offset (for single parameter)
    start_date = datetime(2020,1,1) + timedelta(model_parameters['start_day'])
    end_date = datetime(2020,1,1) + timedelta(model_parameters['end_day'])
    start_date = (2020, start_date.month, start_date.day, 0, 0, 0)
    end_date = (2020, end_date.month, end_date.day, 0, 0, 0)

    ## TODO
    # define containment event
    containment_date = (2020, 3, 20)
    containment_factor = 1.0
    containment = ContainmentModel(start_date, end_date, is_in_days=True)
    containment.add_sharp_event(containment_date, containment_factor)

    # set parameters
    epidemiology, severity, population, age_distribution = get_default_parameters()
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

    # run simulation
    logger = QuantileLogger()
    return simulation([start_date, end_date], y0, dt, samples=n_samples, stochastic_method='tau_leap', logger=logger)

def align_pad_left(a1, a2):
    l1 = len(a1)
    l2 = len(a2)
    if l1 < l2:
        return np.pad(a1, (l2-l1,0), 'constant'), a2
    else:
        return a1, np.pad(a2, (l1-l2,0), 'constant')

def align_right(arr, length):
    mylen = len(arr)
    if mylen > length:
        return arr[-length:]
    return np.pad(arr, (length-mylen,0), 'constant')


def calculate_likelihood_for_model(model_parameters, y_data, n_samples=100):
    # needed for resolution
    dt = 0.05
    skip = int(1./dt)

    # run model and get result
    result = get_model_result(model_parameters, dt, n_samples=n_samples)
    dead_quantiles = result.quantile_data['dead'].sum(axis=2)

    # cut model appropriately
    y_below = dead_quantiles[0,::skip]
    y_model = dead_quantiles[2,::skip]
    y_above = dead_quantiles[4,::skip]

    # we only want to compute the likelihood for the set of values in
    # which either of the y_model or y_data are non-zero
    y_model = y_model[np.argmax(y_model > 0):]
    y_data = y_data[np.argmax(y_data > 0):]

    #print(model_zeroidx, data_zeroidx)
    maxl = max(len(y_data), len(y_model))
    y_data = np.array(align_right(y_data, maxl), dtype=np.float64)
    y_model = align_right(y_model, maxl)
    y_below = align_right(y_below, maxl)
    y_above = align_right(y_above, maxl)

    # deal with zeros and compute "standard deviation"
    LOG_ZERO_VALUE = 0.1
    y_data[y_data < LOG_ZERO_VALUE] = LOG_ZERO_VALUE
    y_below[y_below < LOG_ZERO_VALUE] = LOG_ZERO_VALUE
    y_model[y_model < LOG_ZERO_VALUE] = LOG_ZERO_VALUE
    y_above[y_above < LOG_ZERO_VALUE] = LOG_ZERO_VALUE

    # make log scale for Gaussian probability?
    y_data = np.log(y_data)
    y_model = np.log(y_model)
    y_below_diff = y_model - np.log(y_below) + 0.01   # FIXME: do this better
    y_above_diff = np.log(y_above) - y_model + 0.01   # FIXME: do this better
    y_std = np.maximum(y_above_diff,y_below_diff) / 2.

    # compute and return likelihood
    diff_sq = np.power(y_data - y_model, 2.)
    ratios = diff_sq / np.power(y_std, 2.)

    # print(y_data)
    # print(y_model)
    # print(y_std)
    # print(diff_sq)
    # print(ratios)

    return -0.5 * np.sum(ratios)


def get_default_parameters():

    """
    # set some base model stats for the neher model
    POPULATION_NAME = "USA-Illinois"
    AGE_DATA_NAME = "United States of America"
    population = get_country_population_model(self.POPULATION_NAME)
    age_distribution = get_age_distribution_model(self.AGE_DATA_NAME)
    """

    compartments = ["susceptible", "exposed", "infectious", "recovered", "hospitalized", "critical", "dead"]
    n_age_groups = 9
    population = PopulationModel(
        country='United States of America',
        cases='USA-Illinois',
        population_served=12659682,
        suspected_cases_today=215,
        ICU_beds=1e10,  # originally 1055
        hospital_beds=1e10,  # originally 31649
        imports_per_day=5.0,
    )
    age_distribution = AgeDistribution(
        bin_edges=np.arange(0, 90, 10),
        counts=[39721484, 42332393, 46094077, 44668271, 40348398, 42120077,
                38488173, 24082598, 13147180]
    )
    severity = SeverityModel(
        id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
        age_group=np.arange(0., 90., 10),
        isolated=np.array([0., 0., 20., 10., 0., 0., 50., 90., 0.]),
        confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
        severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
        critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
        fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
    )
    epidemiology = EpidemiologyModel(
        r0=3.7,
        incubation_time=5,
        infectious_period=3,
        length_hospital_stay=4,
        length_ICU_stay=14,
        seasonal_forcing=0.2,
        peak_month=0,
        overflow_severity=2
    )

    return epidemiology, severity, population, age_distribution
