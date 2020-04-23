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
import pandas as pd
from pydemic import Reaction, PassiveReaction, Simulation


def map_to_days_if_needed(time):
    if isinstance(time, (tuple, list)):
        from pydemic import days_from
        return days_from(time)
    else:
        return time


class AttrDict(dict):
    expected_kwargs = set()

    def __init__(self, *args, **kwargs):
        if not self.expected_kwargs.issubset(set(kwargs.keys())):
            raise ValueError

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, name):
        # This method is implemented to avoid pylint 'no-member' errors for
        # attribute access.
        raise AttributeError(
            "'%s' object has no attribute '%s'" % (self.__class__.__name__, name)
        )


class AgeDistribution(AttrDict):
    """
    .. attribute:: bin_edges

        A :class:`numpy.ndarray` specifying the lower end of each of the age ranges
        represented in :attr:`counts`.
        I.e., the age group ``[bin_edges[i], bin_edges[i+1])``
        has count ``counts[i]``.
        Has the same length as :attr:`counts`, i.e., the final bin is
        ``[bin_edges[-1], inf]``.

    .. attribute:: counts

        A :class:`numpy.ndarray` of the total population of the age groups specified
        by :attr:`bin_edges`.
    """

    expected_kwargs = {
        'bin_edges',
        'counts'
    }


class PopulationModel(AttrDict):
    """
    .. attribute:: country

        A :class:`string` with the name of the country.

    .. attribute:: cases

    .. attribute:: population_served

        The total population.

    .. attribute:: population_served

    .. attribute:: hospital_beds

    .. attribute:: ICU_beds

    .. attribute:: initial_cases

    .. attribute:: imports_per_day
    """

    expected_kwargs = {
        'country',
        'cases',
        'population_served',
        'hospital_beds',
        'ICU_beds',
        'initial_cases',
        'imports_per_day'
    }


class EpidemiologyModel(AttrDict):
    """
    .. attribute:: r0

        The average number of infections caused by an individual who is infected
        themselves.

    .. attribute:: incubation_time

        The number of days of incubation.

    .. attribute:: infectious_period

        The number of days an individual remains infections.

    .. attribute:: length_hospital_stay

        The average amount of time a patient remains in the hospital.

    .. attribute:: length_ICU_stay

        The average amount of time a critical patient remains in the ICU.

    .. attribute:: seasonal_forcing

        The amplitude of the seasonal modulation to the
        :meth:`Simulation.infection_rate`.

    .. attribute:: peak_month

        The month (as an integer in ``[0, 12)``) of peak
        :meth:`Simulation.infection_rate`.

    .. attribute:: overflow_severity

        The factor by which the :attr:`Simulation.overflow_death_rate`
        exceeds the ICU :attr:`Simulation.death_rate`
    """

    expected_kwargs = {
        'r0',
        'incubation_time',
        'infectious_period',
        'length_hospital_stay',
        'length_ICU_stay',
        'seasonal_forcing',
        'peak_month',
        'overflow_severity'
    }


class SeverityModel(AttrDict):
    """
    .. attribute:: id

    .. attribute:: age_group

    .. attribute:: isolated

    .. attribute:: confirmed

    .. attribute:: severe

    .. attribute:: critical

    .. attribute:: fatal
    """

    expected_kwargs = {
        'id',
        'age_group',
        'isolated',
        'confirmed',
        'severe',
        'critical',
        'fatal'
    }


class ContainmentModel:
    def __init__(self, start_time, end_time):
        self._events = [
            ['start', map_to_days_if_needed(start_time), 1],
            ['end', map_to_days_if_needed(end_time)]
        ]
        self.sort_times()
        self._regenerate()

    def add_sharp_event(self, time, factor, dt_days=0.05):
        self._events.append(['sharp', map_to_days_if_needed(time), factor, dt_days])
        # regenerate list
        self.sort_times()
        self._regenerate()

    def sort_times(self):
        self._events = sorted(self._events, key=lambda x: x[1])
        c_factor = 1.
        times = []
        factors = []
        for event in self._events:
            if event[0] == "start":
                times.append(event[1])
                factors.append(c_factor)
            elif event[0] == "end":
                times.append(event[1])
                factors.append(factors[-1])
            elif event[0] == "sharp":
                times.append(event[1]-event[3])
                factors.append(factors[-1])
                times.append(event[1])
                factors.append(event[2])
        self.times, self.factors = (
            list(l) for l in zip(*sorted(zip(times, factors)))
        )

    def _regenerate(self):
        from scipy.interpolate import interp1d
        self._interp = interp1d(self.times, self.factors)

    def get_dictionary(self):
        obj = {}
        from datetime import datetime
        dts = [datetime.utcfromtimestamp(x//1000) for x in self.times]
        obj['times'] = [[x.year, x.month, x.day, x.hour, x.minute, x.second]
                        for x in dts]
        obj['factors'] = self.factors
        return obj

    def __call__(self, time):
        return self._interp(time)


class NeherModelSimulation(Simulation):
    def beta(self, t, y):
        phase = 2 * np.pi * (t - self.peak_day) / 365
        return self.avg_infection_rate * (1 + self.seasonal_forcing * np.cos(phase))

    def __init__(self, epidemiology, severity, imports_per_day,
                 n_age_groups, containment, fraction_hospitalized=1):
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

        def I_to_H(t, y):
            return y.infectious * infectious_hospitalized_rate

        reactions = (
            Reaction("susceptible", "exposed",
                     lambda t, y: ((1 - isolated_frac) * self.containment(t, y)
                                   * self.beta(t, y) * y.susceptible
                                   * y.infectious.sum() / y.sum())),
            Reaction("susceptible", "exposed",
                     lambda t, y: imports_per_day / n_age_groups),
            Reaction("exposed", "infectious",
                     lambda t, y: y.exposed * exposed_infectious_rate),
            Reaction("infectious", "hospitalized", I_to_H),
            PassiveReaction(None, "positive",
                            lambda t, y: I_to_H(t, y) / fraction_hospitalized),
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
            'positive': np.zeros(n_age_groups),
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

    @classmethod
    def get_model_data(cls, t, **kwargs):
        if isinstance(t, pd.DatetimeIndex):
            t_eval = (t - pd.to_datetime('2020-01-01')) / pd.Timedelta('1D')
        else:
            t_eval = t

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
        age_distribution = kwargs.pop('age_distribution', age_distribution)
        n_age_groups = len(age_distribution.counts)

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
        fraction_hospitalized = kwargs.pop('fraction_hospitalized')

        factor_keys = sorted([key for key in kwargs.keys()
                              if key.startswith('mitigation_factor')])
        factors = np.array([kwargs.pop(key) for key in factor_keys])

        time_keys = sorted([key for key in kwargs.keys()
                            if key.startswith('mitigation_t')])
        times = np.array([kwargs.pop(key) for key in time_keys])
        # ensure times are ordered

        from pydemic.sampling import InvalidParametersError

        if (np.diff(times, prepend=start_time, append=end_time) < 0).any():
            raise InvalidParametersError(
                "Mitigation times must be ordered within t0 and tf."
            )
        if (np.diff(times) < kwargs.get('min_mitigation_spacing', 5)).any():
            raise InvalidParametersError(
                "Mitigation times must be spaced by at least min_mitigation_spacing."
                " Decrease min_mitigation_spacing to prevent this check."
            )

        from pydemic.containment import MitigationModel
        mitigation = MitigationModel(start_time, end_time, times, factors)

        sim = cls(
            epidemiology, severity, population.imports_per_day,
            n_age_groups, mitigation,
            fraction_hospitalized=fraction_hospitalized
        )
        y0 = sim.get_initial_population(population, age_distribution)

        result = sim.solve_deterministic((start_time, end_time), y0)

        logger = sim.dense_to_logger(result, t_eval)
        y = {key: val.sum(axis=-1) for key, val in logger.y.items()}

        _t = pd.to_datetime(t_eval, origin='2020-01-01', unit='D')
        return pd.DataFrame(y, index=_t)
