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


class NonMarkovianModelEstimator(LikelihoodEstimatorBase):
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

        factor_keys = sorted([key for key in kwargs.keys()
                              if key.startswith('mitigation_factor')])
        factors = np.array([kwargs.pop(key) for key in factor_keys])

        time_keys = sorted([key for key in kwargs.keys()
                            if key.startswith('mitigation_t')])
        times = np.array([kwargs.pop(key) for key in time_keys])
        # ensure times are ordered
        if (np.diff(times, prepend=start_time, append=end_time) < 0).any():
            return -np.inf
        if (np.diff(times) < kwargs.get('min_mitigation_spacing', 5)).any():
            return -np.inf
        from pydemic.containment import MitigationModel
        mitigation = MitigationModel(start_time, end_time, times, factors)

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
