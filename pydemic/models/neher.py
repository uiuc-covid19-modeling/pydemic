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
from pydemic import Reaction, Simulation, map_to_days_if_needed
from pydemic.models import LikelihoodEstimatorBase


class NeherModelSimulation(Simulation):
    """
    Each compartment has n=9 age bins (demographics)
    ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

    Interactions between compartments are according to equations
    [TODO FIXME src/ref] in the pdf
    and are encapsulated in the reactions definition below.

    TODO Current model does not implement hospital overflow.
    """

    def beta(self, t, y):
        phase = 2. * np.pi * (t-self.peak_day)/365
        return self.avg_infection_rate * (1. + self.seasonal_forcing * np.cos(phase))

    def __init__(self, epidemiology, severity, imports_per_day,
                 n_age_groups, containment):
        self.containment = lambda t, y: containment(t)

        # translate from epidemiology/severity models into rates
        dHospital = severity.severe/100. * severity.confirmed/100.
        dCritical = severity.critical/100.
        dFatal = severity.fatal/100.

        isolated_frac = severity.isolated / 100
        exposed_infectious_rate = 1. / epidemiology.incubation_time
        infectious_hospitalized_rate = dHospital / epidemiology.infectious_period
        infectious_recovered_rate = (1.-dHospital) / epidemiology.infectious_period
        hospitalized_discharged_rate = (
            (1 - dCritical) / epidemiology.length_hospital_stay
        )
        hospitalized_critical_rate = dCritical / epidemiology.length_hospital_stay
        critical_hospitalized_rate = (1 - dFatal) / epidemiology.length_ICU_stay
        critical_dead_rate = dFatal / epidemiology.length_ICU_stay

        self.avg_infection_rate = epidemiology.r0 / epidemiology.infectious_period
        self.seasonal_forcing = epidemiology.seasonal_forcing
        self.peak_day = 30 * epidemiology.peak_month + 14.75

        reactions = (
            Reaction("susceptible", "exposed",
                     lambda t, y: ((1 - isolated_frac) * self.containment(t, y)
                                   * self.beta(t, y) * y.susceptible
                                   * y.infectious.sum() / y.sum())),
            Reaction("susceptible", "exposed",
                     lambda t, y: imports_per_day / n_age_groups),
            Reaction("exposed", "infectious",
                     lambda t, y: y.exposed * exposed_infectious_rate),
            Reaction("infectious", "hospitalized",
                     lambda t, y: y.infectious * infectious_hospitalized_rate),
            Reaction("infectious", "recovered",
                     lambda t, y: y.infectious * infectious_recovered_rate),
            Reaction("hospitalized", "recovered",
                     lambda t, y: y.hospitalized * hospitalized_discharged_rate),
            Reaction("hospitalized", "critical",
                     lambda t, y: y.hospitalized * hospitalized_critical_rate),
            Reaction("critical", "hospitalized",
                     lambda t, y: y.critical * critical_hospitalized_rate),
            Reaction("critical", "dead",
                     lambda t, y: y.critical * critical_dead_rate)
        )
        super().__init__(reactions)

    def get_initial_population(self, population, age_distribution):
        # FIXME: remove this method?
        N = population.population_served
        n_age_groups = len(age_distribution.counts)
        age_counts = age_distribution.counts
        y0 = {
            'susceptible': np.round(np.array(age_counts) * N / np.sum(age_counts)),
            'exposed': np.zeros(n_age_groups),
            'infectious': np.zeros(n_age_groups),
            'recovered': np.zeros(n_age_groups),
            'hospitalized': np.zeros(n_age_groups),
            'critical': np.zeros(n_age_groups),
            'dead': np.zeros(n_age_groups)
        }
        i_middle = round(n_age_groups / 2) + 1
        y0['susceptible'][i_middle] -= population.initial_cases
        y0['exposed'][i_middle] += population.initial_cases * 0.7
        y0['infectious'][i_middle] += population.initial_cases * 0.3
        return y0

    def __call__(self, t_span, y0, dt=.01, **kwargs):
        t_start = map_to_days_if_needed(t_span[0])
        t_end = map_to_days_if_needed(t_span[1])
        return super().__call__((t_start, t_end), y0, dt=dt, **kwargs)

    def solve_deterministic(self, t_span, y0, **kwargs):
        t_start = map_to_days_if_needed(t_span[0])
        t_end = map_to_days_if_needed(t_span[1])
        return super().solve_deterministic((t_start, t_end), y0, **kwargs)


def clipped_l2_log_norm(model, data, model_uncert):
    model = np.maximum(model, .1)
    sig = np.log(model_uncert / model)
    sig += 0.05

    top = np.power(np.log(data)-np.log(model), 2.)
    bot = np.power(sig, 2.)

    return - 1/2 * np.sum(top / bot)


def poisson_norm(model, data):
    from scipy.special import factorial
    return np.sum(- model - np.log(factorial(data)) + data * np.log(model))


class NeherModelEstimator(LikelihoodEstimatorBase):
    def __init__(self, fit_parameters, fixed_values, data, norm=None,
                 fit_daily_deaths=True):
        self.fit_daily_deaths = fit_daily_deaths

        if self.fit_daily_deaths and norm is None:
            norm = poisson_norm
        elif norm is None:
            norm = clipped_l2_log_norm

        super().__init__(fit_parameters, fixed_values, data, norm=norm)

    def get_log_likelihood(self, parameters):
        if not self.check_within_bounds(list(parameters.values())):
            return -np.inf
        if 'mitigation_day' in parameters and 'mitigation_day' in parameters:
            if parameters['mitigation_day'] < parameters['start_day']:
                return -np.inf

        model_data = self.get_model_data(
            self.data['t'], **parameters, **self.fixed_values
        )
        model_dead = model_data.y['dead'].sum(axis=-1)
        model_uncert = np.power(model_dead, .5)

        if self.fit_daily_deaths:
            return self.norm(model_dead, self.data['dead'])
        else:
            return self.norm(model_dead, self.data['dead'], model_uncert)

    def get_model_data(self, t, **kwargs):
        start_time = kwargs.pop('start_day')
        end_time = kwargs.pop('end_day')

        from pydemic.load import get_population_model, get_age_distribution_model
        pop_name = kwargs.pop('population')
        population = get_population_model(pop_name)
        if 'population_served' in kwargs:
            population.population_served = kwargs.pop('population_served')
        if 'initial_cases' in kwargs:
            population.initial_cases = kwargs.pop('initial_cases')
        if 'imports_per_day' in kwargs:
            population.imports_per_day = kwargs.pop('imports_per_day')
        population.ICU_beds = 1e10
        population.hospital_beds = 1e10

        age_dist_pop = kwargs.pop('age_dist_pop', pop_name)
        age_distribution = get_age_distribution_model(age_dist_pop)
        age_distribution = kwargs.pop('age_distribution', age_distribution)
        n_age_groups = len(age_distribution.counts)

        from pydemic import SeverityModel, EpidemiologyModel, ContainmentModel
        severity = SeverityModel(
            id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
            age_group=np.arange(0., 90., 10),
            isolated=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
            severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
            critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
            fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
        )
        severity = kwargs.pop('severity', severity)
        epidemiology = EpidemiologyModel(
            r0=kwargs.pop('r0'),
            incubation_time=kwargs.pop('incubation_time', 1),
            infectious_period=kwargs.pop('infectious_period', 5),
            length_hospital_stay=kwargs.pop('length_hospital_stay', 7),
            length_ICU_stay=kwargs.pop('length_ICU_stay', 7),
            seasonal_forcing=kwargs.pop('seasonal_forcing', .2),
            peak_month=kwargs.pop('peak_month', 0),
            overflow_severity=kwargs.pop('overflow_severity', 2),
        )

        mitigation_day = kwargs.pop('mitigation_day')
        mitigation_factor = kwargs.pop('mitigation_factor')
        mitigation_width = kwargs.pop('mitigation_width')

        containment = ContainmentModel((2019, 1, 1), (2022, 1, 1))
        containment.add_sharp_event(mitigation_day, mitigation_factor,
                                    dt_days=mitigation_width)

        from pydemic.models import NeherModelSimulation
        sim = NeherModelSimulation(
            epidemiology, severity, population.imports_per_day,
            n_age_groups, containment
        )
        y0 = sim.get_initial_population(population, age_distribution)

        result = sim.solve_deterministic((start_time, end_time), y0)

        if not self.fit_daily_deaths:
            return sim.dense_to_logger(result, t)
        else:
            model_data = sim.dense_to_logger(result, t)
            model_data_1 = sim.dense_to_logger(result, np.array(t) - 1)

            model_data.y['dead'] = model_data.y['dead'] - model_data_1.y['dead']

            return model_data
