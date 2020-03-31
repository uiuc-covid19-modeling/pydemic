import numpy as np
from datetime import datetime, timedelta
from pydemic.models import NeherModelSimulation
from pydemic import (PopulationModel, AgeDistribution, SeverityModel,
                     EpidemiologyModel, ContainmentModel, QuantileLogger, StateLogger)
from scipy.interpolate import interp1d


def get_model_result(model_parameters, dt=0.1, n_samples=100, run_stochastic=False):
    # set start date and end date based on offset (for single parameter)
    start_date = datetime(2020, 1, 1) + timedelta(model_parameters['start_day'])
    end_date = datetime(2020, 1, 1) + timedelta(model_parameters['end_day'])
    start_date = (2020, start_date.month, start_date.day,
                  start_date.hour, start_date.minute, start_date.second)
    end_date = (2020, end_date.month, end_date.day,
                end_date.hour, end_date.minute, end_date.second)

    # TODO
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

    # method 1
    i_func = interp1d(model_dates, model_deaths,
                      bounds_error=False, fill_value=(0, 0))
    model_deaths = i_func(dates)
    # method 2
    # model_deaths = np.interp(dates, model_dates, model_deaths)
    # method 3
    # skip = 100
    # model_deaths = model_deaths[len(model_deaths)%skip-1::skip]
    # model_deaths = align_right(model_deaths, len(deaths))

    # cut off deaths < 2
    i_to_cut = np.argmax(deaths > 1)
    deaths = deaths[i_to_cut:]
    model_deaths = model_deaths[i_to_cut:]

    return - 0.5 * np.sum(np.power(np.log(deaths)-np.log(model_deaths), 2.))

    # needed for resolution
    dt = 0.05
    skip = int(1./dt)

    # run model and get result
    #result = get_model_result(model_parameters, dt, n_samples=n_samples)
    #dead_quantiles = result.quantile_data['dead'].sum(axis=2)
    deterministic = get_model_result(model_parameters, dt, n_samples=n_samples)
    dead_quantiles = deterministic.quantile_data

    # cut model appropriately
    sidx = dead_quantiles.shape[1] % skip
    y_below = dead_quantiles[1, sidx::skip]
    y_model = dead_quantiles[2, sidx::skip]
    y_above = dead_quantiles[3, sidx::skip]

    if True:

        maxl = len(y_data)
        y_data = np.array(y_data, dtype=np.float64)
        y_model = align_right(y_model, maxl)
        y_above = align_right(y_above, maxl)
        y_below = align_right(y_below, maxl)

        arr = np.power(np.log(y_data) - np.log(y_model), 2.)

        return - 0.5 * np.sum(arr)

    if True:

        maxl = len(y_data)
        y_data = np.array(y_data, dtype=np.float64)
        y_model = align_right(y_model, maxl)
        y_above = align_right(y_above, maxl)
        y_below = align_right(y_below, maxl)
        LOG_ZERO_VALUE = -2

        y_data = np.log(y_data)
        y_model = np.log(y_model)
        y_std = np.log(y_above/y_model)

        y_data[~np.isfinite(y_data)] = LOG_ZERO_VALUE
        y_model[~np.isfinite(y_model)] = LOG_ZERO_VALUE
        y_std[~np.isfinite(y_std)] = LOG_ZERO_VALUE

        if False:
            print(y_data)
            print(y_model)
            print(y_std)

        ratio = np.power(y_model-y_data, 2.)/np.power(y_std, 2.)

        return - 0.5 * np.sum(ratio)

    if True:

        maxl = len(y_data)
        y_data = np.array(y_data, dtype=np.float64)
        y_model = align_right(y_model, maxl)
        y_above = align_right(y_above, maxl)
        y_below = align_right(y_below, maxl)

        y_top = np.power(y_model - y_data, 2.)
        y_bot = np.power(y_above - y_model, 2.)
        ratio = y_top/y_bot

        print(y_data)
        print(y_model)
        print(y_above)
        print(y_top)
        print(y_bot)

        return - 0.5 * np.sum(ratio)

    if False:

        maxl = len(y_data)
        y_data = np.array(ydata, dtype=np.float64)
        y_model = align_right(y_model, maxl)
        y_above = align_right(y_above, maxl)
        y_below = align_right(y_below, maxl)

        print(y_data)

        return -0.5 * np.sum(np.power(y_data-y_model, 2.)/np.power(y_diff, 2.)), (y_data, y_model, y_below, y_above)

    if False:

        maxl = min(len(y_data), len(y_model))
        y_data = np.array(align_right(y_data, maxl), dtype=np.float64)
        y_model = align_right(y_model, maxl)
        y_below = align_right(y_below, maxl)
        y_above = align_right(y_above, maxl)

        # log data
        y_data = np.log(y_data)
        y_model = np.log(y_model)
        y_diff_estimator = np.maximum(
            np.log((y_above+1)/(y_model+1)), np.log((y_model+1)/(y_below+1)))
        y_diff_estimator = np.mean(
            np.array([np.log((y_above+1)/(y_model+1)), np.log((y_model+1)/(y_below+1))]), axis=0)
        y_diff_estimator += 0.2  # heuristic

        print(y_diff_estimator)

        return -0.5 * np.sum(np.power(y_data-y_model, 2.)/np.power(y_diff_estimator, 2.)), (np.exp(y_data), np.exp(y_model), y_below, y_above)

    if True:

        # we only want to compute the likelihood for the set of values in
        # which either of the y_model or y_data are non-zero
        #y_model = y_model[np.argmax(y_model > 0):]
        #y_data = y_data[np.argmax(y_data > 0):]

        #print(model_zeroidx, data_zeroidx)
        #maxl = max(len(y_data), len(y_model))
        maxl = len(y_data)
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
        y_std = np.maximum(y_above_diff, y_below_diff)

        # compute and return likelihood
        diff_sq = np.power(y_data - y_model, 2.)
        ratios = diff_sq / np.power(y_std, 2.)

        return -0.5 * np.sum(ratios)


def get_default_parameters():
    """
    # set some base model stats for the neher model
    POPULATION_NAME = "USA-Illinois"
    AGE_DATA_NAME = "United States of America"
    population = get_country_population_model(self.POPULATION_NAME)
    age_distribution = get_age_distribution_model(self.AGE_DATA_NAME)
    """

    compartments = ["susceptible", "exposed", "infectious",
        "recovered", "hospitalized", "critical", "dead"]
    n_age_groups = 9
    population = PopulationModel(
        country='United States of America',
        cases='USA-Illinois',
        population_served=12659682,
        suspected_cases_today=10,  # original 215
        ICU_beds=1e10,  # originally 1055
        hospital_beds=1e10,  # originally 31649
        imports_per_day=1.1   # originally 5.0
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
        incubation_time=1,  # was 5
        infectious_period=4,  # was 3
        length_hospital_stay=4,
        length_ICU_stay=14,
        seasonal_forcing=0.2,
        peak_month=0,
        overflow_severity=2
    )

    return epidemiology, severity, population, age_distribution
