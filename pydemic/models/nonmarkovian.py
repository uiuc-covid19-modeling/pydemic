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
from pydemic import NonMarkovianSimulation
from pydemic.sampling import LikelihoodEstimatorBase
from scipy.interpolate import interp1d


def clipped_l2_log_norm(model, data, model_uncert):
    model = np.maximum(model, .1)
    sig = np.log(model_uncert / model)
    sig += 0.05

    top = np.power(np.log(data)-np.log(model), 2.)
    bot = np.power(sig, 2.)

    return - 1/2 * np.sum(top / bot)


def poisson_norm(model, data):
    from scipy.special import gammaln  # pylint: disable=E0611
    return np.sum(- model - gammaln(data) + data * np.log(model))


class NonMarkovianModelEstimator(LikelihoodEstimatorBase):
    def __init__(self, fit_parameters, fixed_values, data, weights, norm=None,
                 fit_cumulative=False):
        self.fit_cumulative = fit_cumulative
        self.weights = weights

        if self.fit_cumulative and norm is None:
            norm = clipped_l2_log_norm
        elif norm is None:
            norm = poisson_norm

        super().__init__(fit_parameters, fixed_values, data, norm=norm)

        if not self.fit_cumulative:
            self.data.y = {
                key: np.diff(self.data.y[key], prepend=0)
                for key in self.weights
            }

    def get_log_likelihood(self, parameters):
        if not self.check_within_bounds(list(parameters.values())):
            return -np.inf

        # FIXME: add new mitigation functionality
        # if 'mitigation_day' in parameters and 'start_day' in parameters:
        #     # FIXME: doesn't check if either is fixed
        #     if parameters['mitigation_day'] < parameters['start_day']:
        #         return -np.inf

        # get model data at daily values
        # when computing diffs, datasets were prepended with 0, so there is no need
        # to evaluate at an extra data point on day earlier
        t_eval = np.arange(self.data.t[0], self.data.t[-1]+2)
        model_data = self.get_model_data(
            t_eval, **parameters, **self.fixed_values
        )
        data_t_indices = np.isin(t_eval, self.data.t)

        def get_one_likelihood(_model, data):
            if not self.fit_cumulative:
                model = np.diff(_model, prepend=0)
            else:
                model = _model

            # slice to match data time coordinates
            model = model[data_t_indices]
            # ensure no model data is smaller than .1
            model = np.maximum(.1, model)
            # only compare data points whose values are >= 1
            data_nonzero = data > .9

            if self.fit_cumulative:
                sigma = np.power(model, .5)
                return self.norm(model[data_nonzero],
                                 data[data_nonzero],
                                 sigma[data_nonzero])
            else:
                return self.norm(model[data_nonzero],
                                 data[data_nonzero])

        likelihood = 0
        for compartment, weight in self.weights.items():
            if weight > 0:
                L = get_one_likelihood(
                    model_data.y[compartment].sum(axis=-1),
                    self.data.y[compartment]
                )
                likelihood += weight * L

        return likelihood

    @classmethod
    def get_model_data(cls, t, **kwargs):

        start_time = kwargs.pop('start_day')
        end_time = kwargs.pop('end_day')

        from pydemic.data import get_population_model, get_age_distribution_model
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
        n_age_groups = len(age_distribution.counts)

        # from pydemic import SeverityModel, EpidemiologyModel, ContainmentModel
        # severity = SeverityModel(
        #     id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
        #     age_group=np.arange(0., 90., 10),
        #     isolated=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        #     confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
        #     severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
        #     critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
        #     fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
        # )
        # severity = kwargs.pop('severity', severity)
        # epidemiology = EpidemiologyModel(
        #     r0=kwargs.pop('r0'),
        #     incubation_time=kwargs.pop('incubation_time', 1),
        #     infectious_period=kwargs.pop('infectious_period', 5),
        #     length_hospital_stay=kwargs.pop('length_hospital_stay', 7),
        #     length_ICU_stay=kwargs.pop('length_ICU_stay', 7),
        #     seasonal_forcing=kwargs.pop('seasonal_forcing', .2),
        #     peak_month=kwargs.pop('peak_month', 0),
        #     overflow_severity=kwargs.pop('overflow_severity', 2),
        # )
        # fraction_hospitalized = kwargs.pop('fraction_hospitalized')

        # containment = ContainmentModel((2019, 1, 1), (2022, 1, 1))
        # containment.add_sharp_event(mitigation_day, mitigation_factor,
        #                             dt_days=mitigation_width)

        # from pydemic.models import NeherModelSimulation
        # sim = NeherModelSimulation(
        #     epidemiology, severity, population.imports_per_day,
        #     n_age_groups, containment,
        #     fraction_hospitalized=fraction_hospitalized
        # )
        # y0 = sim.get_initial_population(population, age_distribution)

        # result = sim.solve_deterministic((start_time, end_time), y0)

        # return sim.dense_to_logger(result, t)

        mitigation_keys = sorted([key for key in kwargs.keys()
                                  if key.startswith('mitigation_factor')])
        factors = np.array([kwargs.pop(key) for key in mitigation_keys])
        from pydemic.containment import MitigationModel
        mitigation = MitigationModel(
            start_time, end_time, kwargs.pop('mitigation_t'), factors
        )
        

        tspan = (start_time, end_time)
        sim = NonMarkovianSimulation(tspan, mitigation, dt=0.05, **kwargs)
        y0 = sim.get_y0(population, age_distribution)
        result = sim(tspan, y0)

        data = {}
        for track in result.tracks:
            if track not in ["susceptible", "population"]:
                data[track] = np.cumsum(result.tracks[track], axis=1)
            else:
                data[track] = result.tracks[track]

        class AdHocLogger:
            def __init__(self, t, result):
                self.t = t
                self.y = {}

        logger = AdHocLogger(t, result)
        for track in data:
            func = interp1d(result.t, data[track], axis=1)
            logger.y[track] = func(t).T

        return logger
