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
from scipy.interpolate import interp1d


class SimulationResult:
    def __init__(self, time, y):
        self.t = time
        self.y = y


def mean_std_to_k_theta(mean, std):
    return (mean**2 / std**2, std**2 / mean)


def convolve_pdf(t, influx, prefactor=1, mean=5, std=2):
    shape, scale = mean_std_to_k_theta(mean, std)
    from scipy.stats import gamma
    cdf = gamma.cdf(t[:] - t[0], shape, scale=scale)
    pdf = np.diff(cdf, prepend=0)

    prefactor = prefactor * np.ones_like(influx[0, ...])

    kernel = np.outer(pdf, prefactor)

    end = t.shape[0]
    from scipy.signal import fftconvolve
    result = fftconvolve(kernel, influx, mode='full', axes=0)[:end]

    return result


def convolve_survival(t, influx, prefactor=1, mean=5, std=2):
    shape, scale = mean_std_to_k_theta(mean, std)
    from scipy.stats import gamma
    survival = 1 - gamma.cdf(t - t[0], shape, scale=scale)

    prefactor = prefactor * np.ones_like(influx[0, ...])
    kernel = np.outer(survival, prefactor)

    end = t.shape[0]
    from scipy.signal import fftconvolve
    result = fftconvolve(kernel, influx, mode='full', axes=0)[:end]

    return result


def convolve_direct(t, influx, prefactor=1, mean=5, std=2):
    shape, scale = mean_std_to_k_theta(mean, std)
    from scipy.stats import gamma
    cdf = gamma.cdf(t[1:] - t[0], shape, scale=scale)
    pdf = np.diff(cdf, prepend=0)

    prefactor = prefactor * np.ones_like(influx[0, ...])

    end = t.shape[0]
    result = np.zeros_like(influx)
    for i in range(1, end):
        result[i, ...] = prefactor * np.dot(influx[i-1::-1].T, pdf[:i])

    return result


class NonMarkovianSEIRSimulationBase:
    """
    Main driver for non-Markovian simulations, used as a base class for
    SEIR++ variants.

    .. automethod:: __init__
    .. automethod:: get_y0
    .. automethod:: __call__
    .. automethod:: get_model_data
    """

    increment_keys = ('dead',)

    def set_std(self, mean, std, k):
        if k is not None:
            return np.sqrt(mean**2 / k)
        else:
            return std

    def get_gamma_pdf(self, t, shape, scale, dt):
        from scipy.stats import gamma
        cdf = gamma.cdf(t, shape, scale=scale)
        return np.diff(cdf, prepend=0) / dt

    def set_kernels(self, t, dt):
        self.kernels = {
            key: self.get_gamma_pdf(t, shape, scale, dt)
            for key, (shape, scale) in self.distribution_params.items()
        }

    def seasonal_forcing(self, t):
        phase = 2 * np.pi * (t - self.peak_day) / 365
        return (1 + self.seasonal_forcing_amp * np.cos(phase))

    def __init__(self, mitigation=None, *,
                 r0=3.2, serial_mean=4, serial_std=3.25, serial_k=None,
                 seasonal_forcing_amp=.2, peak_day=15, **kwargs):
        """
        The following keyword-only arguments are recognized:

        :arg mitigation: A function of time specifying a multiplicative factor.
            Defaults to ``lambda t: 1``.

        :arg r0: The basic reproduction number.

        :arg serial_mean: The mean of the serial interval distribution.

        :arg serial_std: The standard deviation of the serial interval distribution.

        :arg serial_k: The shape parameter :math:`k` (for a gamma distribution)
            of the delay-time distribution for infecting
            new individuals after having been infected.
            If not *None*, will overwrite ``serial_std``.

        :arg seasonal_forcing_amp: The amplitude (i.e., maximum fractional change)
            in the force of infection due to seasonal effects.

        :arg peak_day: The day of the year at which seasonal forcing is greatest.
        """

        serial_std = self.set_std(serial_mean, serial_std, serial_k)

        if mitigation is not None:
            self.mitigation = mitigation
        else:
            self.mitigation = lambda t: 1

        self.distribution_params = {
            'serial': mean_std_to_k_theta(serial_mean, serial_std),
        }
        self.seasonal_forcing_amp = seasonal_forcing_amp
        self.peak_day = peak_day
        self.r0 = r0

    def step(self, state, count, dt):
        Rt = (self.r0
              * self.mitigation(state.t[count])
              * self.seasonal_forcing(state.t[count]))
        j_m = np.dot(state.y['infected'][..., count-1::-1],
                     self.kernels['serial'][:count])
        j_i = j_m.sum()
        S_i = state.y['susceptible'][..., count-1]
        new_infected_i = dt * Rt * S_i * j_i / self.total_population
        state.y['infected'][..., count] = new_infected_i

        state.y['susceptible'][..., count] = (
            state.y['susceptible'][..., count-1] - new_infected_i
        )

    def get_y0(self, total_population, initial_cases, age_distribution):
        """
        Initializes a population with a number ``initial_cases`` of initial
        infectious individuals distributed in proportion to ``age_distribution``.

        :arg total_population: The total size of the population.

        :arg initial_cases: The total numnber of initial cases.

        :arg age_distribution: A :class:`numpy.ndarray` specifying the relative
            fraction of the population in various age groups.

        :returns: A :class:`dict` containing the initial conditions for the
            ``'infected'`` and ``'susceptible'`` compartments.
        """

        # FIXME: shouldn't be set here
        self.total_population = total_population
        self.population = total_population * np.array(age_distribution)
        n_demographics = len(age_distribution)

        y0 = {}
        for key in ('susceptible', 'infected'):
            y0[key] = np.zeros((n_demographics,))

        y0['infected'][...] = initial_cases * np.array(age_distribution)
        y0['susceptible'][...] = self.population - y0['infected']

        return y0

    def __call__(self, tspan, y0, dt=.05):
        """
        :arg tspan: A :class:`tuple` specifying the initiala and final times.

        :arg y0: A :class:`dict` with the initial values
            (as :class:`numpy.ndarray`'s) for the
            ``'infected'`` and ``'susceptible'`` compartments, e.g., as returned
            by :meth:`get_y0`.

        :arg dt: The timestep.

        :returns: A :class:`SimulationResult` with attributes ``t``, the array of
            times of evaluation, and ``y``, a :class:`dict` of results where time
            proceeds along the first axis.
        """

        start_time, end_time = tspan
        times = np.arange(start_time, end_time + dt, dt)
        n_steps = times.shape[0]  # pylint: disable=
        self.set_kernels(times[1:] - start_time, dt)

        y0_all_t = {}
        for key in y0:
            y0_all_t[key] = np.zeros(y0[key].shape + (n_steps,))
            y0_all_t[key][..., 0] = y0[key]

        influxes = SimulationResult(times, y0_all_t)

        for count in range(1, n_steps):
            self.step(influxes, count, dt)

        for key, val in influxes.y.items():
            influxes.y[key] = val.T

        return influxes

    @classmethod
    def get_model_data(cls, t, **kwargs):
        """
        A wrapper to :meth:`__init__` and :meth:`__call__` for initializing and
        running a simulation from keyword arguments only (i.e., as used by
        :class:`~pydemic.LikelihoodEstimator`.)

        :arg t: A :class:`pandas.DatetimeIndex` (or :class:`numpy.ndarray` of
            times in days since 2020/1/1) of times at which to evaluate the
            solution.

        The following keyword arguments are required:

        :arg start_day: The day (relative to 2020/1/1) at which to begin the
            simulation---i.e., the day corresponding to the initial condition
            generated by :meth:`get_y0`.

        :arg total_population: The total size of the population.

        :arg initial_cases: The total numnber of initial cases.

        :arg age_distribution: A :class:`numpy.ndarray` specifying the relative
            fraction of the population in various age groups.

        In addition, a :class:`~pydemic.MitigationModel` is created from
        passed keyword arguments via
        :meth:`~pydemic.MitigationModel.init_from_kwargs`.

        The following optional keyword arguments are also recognized:

        :arg min_mitigation_spacing: The minimum number of days of separation
            between mitigation events.
            Defaults to ``5``.

        All remaining keyword arguments are passed to :meth:`__init__`.

        :raises InvalidParametersError: if ``t`` specifies any days of evaluation
            which are not at least one day after ``start_day``.

        :raises InvalidParametersError: if mitigation events are not ordered and
            separated by ``min_mitigation_spacing``.

        :returns: A :class:`pandas.DataFrame` of simulation results.
        """

        if isinstance(t, pd.DatetimeIndex):
            t_eval = (t - pd.to_datetime('2020-01-01')) / pd.Timedelta('1D')
        else:
            t_eval = t

        t0 = kwargs.pop('start_day')
        tf = t_eval[-1] + 2

        from pydemic.sampling import InvalidParametersError

        if (t_eval < t0 + 1).any():
            raise InvalidParametersError(
                "Must start simulation at least one day before result evaluation."
            )

        try:
            from pydemic.mitigation import MitigationModel
            mitigation = MitigationModel.init_from_kwargs(t0, tf, **kwargs)
        except ValueError:  # raised by PchipInterpolator when times aren't ordered
            raise InvalidParametersError(
                "Mitigation times must be ordered within t0 and tf."
            )

        if any(np.diff(mitigation.times) < kwargs.get('min_mitigation_spacing', 5)):
            raise InvalidParametersError(
                "Mitigation times must be spaced by at least min_mitigation_spacing."
                " Decrease min_mitigation_spacing to prevent this check."
            )

        age_distribution = kwargs.pop('age_distribution')
        sim = cls(
            mitigation=mitigation, age_distribution=age_distribution, **kwargs
        )

        y0 = sim.get_y0(kwargs.pop('total_population'),
                        kwargs.pop('initial_cases'),
                        age_distribution)
        result = sim((t0, tf), y0)

        y = {}
        for key, val in result.y.items():
            y[key] = interp1d(result.t, val.sum(axis=-1), axis=0)(t_eval)

        for key in cls.increment_keys:
            if key in result.y.keys():
                spline = interp1d(result.t, result.y[key].sum(axis=-1), axis=0)
                y[key+'_incr'] = spline(t_eval) - spline(t_eval - 1)

        _t = pd.to_datetime(t_eval, origin='2020-01-01', unit='D')
        return pd.DataFrame(y, index=_t)


class SEIRPlusPlusSimulation(NonMarkovianSEIRSimulationBase):
    """
    .. automethod:: __init__
    """

    def __init__(self, mitigation=None, *,
                 r0=3.2, serial_mean=4, serial_std=3.25,
                 seasonal_forcing_amp=.2, peak_day=15,
                 incubation_mean=5.5, incubation_std=2, p_observed=1,
                 icu_mean=11, icu_std=5, p_icu=1, p_icu_prefactor=1,
                 dead_mean=7.5, dead_std=7.5, p_dead=1, p_dead_prefactor=1,
                 recovered_mean=7.5, recovered_std=7.5,
                 ifr=0.003, age_distribution=None, **kwargs):
        """
        In addition to the arguments recognized by
        :class:`~pydemic.models.seirpp.NonMarkovianSEIRSimulationBase`, the
        following keyword-only arguments are recognized:

        :arg incubation_mean:

        :arg incubation_std:

        :arg p_observed:

        :arg icu_mean:

        :arg icu_std:

        :arg p_icu:

        :arg p_icu_prefactor:

        :arg dead_mean:

        :arg dead_std:

        :arg p_dead:

        :arg p_dead_prefactor:

        :arg recovered_mean:

        :arg recovered_std:

        :arg ifr:

        :arg age_distribution:
        """

        super().__init__(
            mitigation=mitigation, r0=r0,
            serial_mean=serial_mean, serial_std=serial_std,
            seasonal_forcing_amp=seasonal_forcing_amp, peak_day=peak_day
        )

        if age_distribution is None:
            age_distribution = np.array([0.24789492, 0.13925591, 0.13494838,
                                         0.12189751, 0.12724997, 0.11627754,
                                         0.07275651, 0.03971926])

        p_symptomatic = 1.
        p_symptomatic = np.array(p_symptomatic)
        p_observed = np.array(p_observed)
        p_icu = np.array(p_icu) * p_icu_prefactor
        p_dead = np.array(p_dead) * p_dead_prefactor

        # FIXME: this is a kludge-y way to set the target ifr
        # (infection, not just symptomatic)
        p_dead_all = p_symptomatic * p_observed * p_icu * p_dead
        synthetic_ifr = (p_dead_all * age_distribution).sum()
        p_symptomatic *= ifr / synthetic_ifr

        self.readouts = {
            "observed": ('infected', p_observed * p_symptomatic,
                         incubation_mean, incubation_std),
            "icu": ('observed', p_icu, icu_mean, icu_std),
            "dead": ('icu', p_dead, dead_mean, dead_std),
            "recovered": ('icu', (1 - p_dead), recovered_mean, recovered_std),
        }

    def __call__(self, tspan, y0, dt=.05):
        influxes = super().__call__(tspan, y0, dt=dt)
        t = influxes.t

        for key, (src, prob, mean, std) in self.readouts.items():
            influxes.y[key] = convolve_pdf(t, influxes.y[src], prob, mean, std)

        sol = SimulationResult(t, {})

        for key, val in influxes.y.items():
            if key not in ["susceptible", "population"]:
                sol.y[key] = np.cumsum(val, axis=0)
            else:
                sol.y[key] = val

        sol.y['infectious'] = convolve_survival(t, influxes.y['infected'],
                                                2, 5, 2)
        sol.y["critical"] = sol.y["icu"] - sol.y["dead"] - sol.y["recovered"]
        sol.y['ventilators'] = .73 * sol.y['critical']

        i = np.searchsorted(sol.t, sol.t[0] + 5)
        sol.y["hospitalized"] = np.zeros_like(sol.y['critical'])
        sol.y["hospitalized"][:-i] = 2.7241 * sol.y['critical'][i:]
        sol.y["hospitalized"][-i:] = np.nan

        return sol


class SEIRPlusPlusSimulationOnsetAndDeath(NonMarkovianSEIRSimulationBase):
    """
    .. automethod:: __init__
    """

    def __init__(self, mitigation=None, *,
                 r0=3.2, serial_mean=4, serial_std=3.25,
                 seasonal_forcing_amp=.2, peak_day=15,
                 p_symptomatic=1.,
                 incubation_mean=5.5, incubation_std=2, p_observed=1,
                 dead_mean=7.5, dead_std=7.5, p_dead=1, p_dead_prefactor=1,
                 **kwargs):
        """
        In addition to the arguments recognized by
        :class:`~pydemic.models.seirpp.NonMarkovianSEIRSimulationBase`, the
        following keyword-only arguments are recognized:

        :arg p_symptomatic:

        :arg incubation_mean:

        :arg incubation_std:

        :arg p_observed:

        :arg dead_mean:

        :arg dead_std:

        :arg p_dead:

        :arg p_dead_prefactor:
        """

        super().__init__(
            mitigation=mitigation, r0=r0,
            serial_mean=serial_mean, serial_std=serial_std,
            seasonal_forcing_amp=seasonal_forcing_amp, peak_day=peak_day
        )

        p_symptomatic = np.array(p_symptomatic)
        p_observed = np.array(p_observed)
        p_dead = np.array(p_dead) * p_dead_prefactor

        self.readouts = {
            "observed": ('infected', p_observed * p_symptomatic,
                         incubation_mean, incubation_std),
            "dead": ('observed', p_dead, dead_mean, dead_std),
        }

    def __call__(self, tspan, y0, dt=.05):
        influxes = super().__call__(tspan, y0, dt=dt)
        t = influxes.t

        for key, (src, prob, mean, std) in self.readouts.items():
            influxes.y[key] = convolve_pdf(t, influxes.y[src], prob, mean, std)

        sol = SimulationResult(t, {})

        for key, val in influxes.y.items():
            if key not in ["susceptible", "population"]:
                sol.y[key] = np.cumsum(val, axis=0)
            else:
                sol.y[key] = val

        sol.y['infectious'] = convolve_survival(t, influxes.y['infected'],
                                                2, 5, 2)

        return sol


class SEIRPlusPlusSimulationHospitalCriticalAndDeath(NonMarkovianSEIRSimulationBase):
    """
    SEIR++ model with unconnected infectivity loop. Readout topology is::

        -> symptomatic
            -> hospitalized -> recovered
                            -> critical -> dead -> all_dead
                                        -> hospitalized -> recovered_mean

    .. automethod:: __init__
    """

    increment_keys = ('dead', 'all_dead', 'positive', 'admitted_to_hospital')

    def __init__(self, mitigation=None, *,
                 r0=3.2, serial_mean=4, serial_std=3.25, serial_k=None,
                 seasonal_forcing_amp=.2, peak_day=15,
                 ifr=0.009, incubation_mean=5.5, incubation_std=2, incubation_k=None,
                 p_symptomatic=1., p_symptomatic_prefactor=None,
                 p_positive=.5,
                 p_hospitalized=1., p_hospitalized_prefactor=1.,
                 hospitalized_mean=6.5, hospitalized_std=4., hospitalized_k=None,
                 discharged_mean=6., discharged_std=4., discharged_k=None,
                 p_critical=1., p_critical_prefactor=1.,
                 critical_mean=2., critical_std=2., critical_k=None,
                 p_dead=1., p_dead_prefactor=1.,
                 dead_mean=7.5, dead_std=7.5, dead_k=None,
                 recovered_mean=7.5, recovered_std=7.5, recovered_k=None,
                 all_dead_multiplier=1.,
                 all_dead_mean=2.5, all_dead_std=2.5, all_dead_k=None,
                 age_distribution=None, **kwargs):
        """
        In addition to the arguments recognized by
        :class:`~pydemic.models.seirpp.NonMarkovianSEIRSimulationBase`, the
        following keyword-only arguments are recognized:

        :arg ifr: The infection fatality ratio, i.e., the proportion of the infected
            population who eventually die.

        :arg age_distribution: A :class:`numpy.ndarray` specifying the relative
            fraction of the population in various age groups.

        :arg incubation_mean: The mean of the delay-time distribution
            of developing symptoms after being infected.

        :arg incubation_std: The standard deviation of the delay-time distribution
            of developing symptoms after being infected.

        :arg incubation_k: The gamma k of the delay-time distribution
            of developing symptoms after being infected. If not None, will overwrite
            incubation_std.

        :arg p_symptomatic: The distribution of the proportion of infected
            individuals who become symptomatic.

        :arg p_symptomatic_prefactor: The overall scaling of the proportion of
            infected individuals who become symptomatic.
            If not *None*, overrides the input ``ifr``; otherwise its value is set
            according to ``ifr``.

        :arg p_positive: The fraction of symptomatic individuals who are tested and
            test positive.

        :arg p_hospitalized: The distribution of the proportion of symptomatic
            individuals who enter the hospital.

        :arg p_hospitalized_prefactor: The overall scaling of the proportion of
            symptomatic individuals who enter the hospital.

        :arg hospitalized_mean: The mean of the delay-time
            distribution of entering the hospital after becoming symptomatic.

        :arg hospitalized_std: The standard derivation of the delay-time
            distribution of entering the hospital after becoming symptomatic.

        :arg hospitalized_k: The gamma k of the delay-time distribution
            of entering the hospital after becoming symptomatic. If not None,
            will overwrite hospitalized_std.

        :arg discharged_mean: The mean of the delay-time
            distribution of survivors being discharged after entering the hospital.

        :arg discharged_std: The standard derivation of the delay-time
            distribution of survivors being discharged after entering the hospital.

        :arg discharged_k: The gamma k of the delay-time distribution
            of survivors being discharged after entering the hospital.
            If not *None*, will overwrite ``discharged_std``.

        :arg p_critical: The distribution of the proportion of
            hospitalized individuals who become critical.

        :arg p_critical_prefactor: The overall scaling of the proportion of
            hospitalized individuals who become critical.

        :arg critical_mean: The mean of the delay-time
            distribution of hospitalized individuals entering the ICU.

        :arg critical_std: The standard deviation of the delay-time
            distribution of hospitalized individuals entering the ICU.

        :arg critical_k: The gamma k of the delay-time distribution
            of hospitalized individuals entering the ICU. If not None,
            will overwrite critical_std.

        :arg p_dead: The distribution of the proportion of
            ICU patients who die.

        :arg p_dead_prefactor: The overall scaling of the proportion of
            ICU patients who die.

        :arg dead_mean: The mean of the delay-time
            distribution of ICU patients dying.

        :arg dead_std: The standard deviation of the delay-time
            distribution of ICU patients dying.

        :arg dead_k: The gamma k of the delay-time distribution of ICU
            patients dying. If not None, will overwrite dead_std.

        :arg recovered_mean: The mean of the delay-time distribution
            of ICU patients recovering and returning to the general
            ward.

        :arg recovered_std: The standard deviation of the delay-time
            distribution of ICU patients recovering and returning to
            the general ward.

        :arg recovered_k: The gamma k of the delay-time distribution of ICU
            patients recovering and returning to the general ward. If not
            None, will overwrite recovered_std.

        :arg all_dead_multiplier: The ratio of total deaths to deaths occurring
            in the ICU.

        :arg all_dead_mean: The mean of the delay-time distribution
            between ICU deaths and all reported deaths.

        :arg all_dead_std: The standard deviation of the delay-time distribution
            between ICU deaths and all reported deaths.

        :arg all_dead_k: The gamma k of the delay-time distribution
            between ICU deaths and all reported deaths. If not None,
            will overwrite all_dead_std.
        """

        super().__init__(
            mitigation=mitigation, r0=r0,
            serial_mean=serial_mean, serial_std=serial_std, serial_k=serial_k,
            seasonal_forcing_amp=seasonal_forcing_amp, peak_day=peak_day
        )

        if age_distribution is None:
            # default to usa_population
            age_distribution = np.array([0.12000352, 0.12789140, 0.13925591,
                                         0.13494838, 0.12189751,  0.12724997,
                                         0.11627754, 0.07275651, 0.03971926])

        # a bit of a verbose kludge to overwrite _std if _k is passed
        serial_std = self.set_std(serial_mean, serial_std, serial_k)
        incubation_std = self.set_std(incubation_mean, incubation_std, incubation_k)
        hospitalized_std = self.set_std(hospitalized_mean, hospitalized_std,
                                        hospitalized_k)
        discharged_std = self.set_std(discharged_mean, discharged_std, discharged_k)
        critical_std = self.set_std(critical_mean, critical_std, critical_k)
        dead_std = self.set_std(dead_mean, dead_std, dead_k)
        recovered_std = self.set_std(recovered_mean, recovered_std, recovered_k)
        all_dead_std = self.set_std(all_dead_mean, all_dead_std, all_dead_k)

        # make numpy arrays first in case p_* passed as lists
        p_symptomatic = np.array(p_symptomatic)
        p_hospitalized = np.array(p_hospitalized) * p_hospitalized_prefactor
        p_critical = np.array(p_critical) * p_critical_prefactor
        p_dead = np.array(p_dead) * p_dead_prefactor

        # if p_symptomatic_prefactor is None, set according to ifr
        if p_symptomatic_prefactor is None:
            p_dead_product = p_symptomatic * p_hospitalized * p_critical * p_dead
            synthetic_ifr = (p_dead_product * age_distribution).sum()
            p_symptomatic_prefactor = ifr / synthetic_ifr

        # ... and update p_symptomatic
        p_symptomatic *= p_symptomatic_prefactor

        # now check that none of the prefactors are too large
        from pydemic.sampling import InvalidParametersError

        # first check p_symptomatic_prefactor
        top = age_distribution
        bottom = age_distribution * p_symptomatic
        if top.sum() < bottom.sum():
            raise InvalidParametersError(
                "p_symptomatic_prefactor must not be too large"
            )

        # then check p_hospitalized_prefactor
        top = bottom.copy()
        bottom *= p_hospitalized
        if top.sum() < bottom.sum():
            raise InvalidParametersError(
                "p_hospitalized_prefactor must not be too large"
            )

        # then check p_critical_prefactor
        top = bottom.copy()
        bottom *= p_critical
        if top.sum() < bottom.sum():
            raise InvalidParametersError(
                "p_critical_prefactor must not be too large"
            )

        # and finally check p_dead_prefactor
        top = bottom.copy()
        bottom *= p_dead
        if top.sum() < bottom.sum():
            raise InvalidParametersError(
                "p_dead_prefactor must not be too large"
            )

        self.readouts = {
            "symptomatic": ('infected', p_symptomatic,
                            incubation_mean, incubation_std),
            "positive": ('infected', p_positive * p_symptomatic,
                         incubation_mean, incubation_std),
            "admitted_to_hospital": ('symptomatic', p_hospitalized,
                                     hospitalized_mean, hospitalized_std),
            "icu": ('admitted_to_hospital', p_critical, critical_mean, critical_std),
            "dead": ('icu', p_dead, dead_mean, dead_std),
            "general_ward": ('icu', 1.-p_dead, recovered_mean, recovered_std),
            "hospital_recovered": ('admitted_to_hospital', 1.-p_critical,
                                   discharged_mean, discharged_std),
            "general_ward_recovered": ('general_ward', 1., discharged_mean,
                                       discharged_std),
            "all_dead": ('dead', all_dead_multiplier, all_dead_mean, all_dead_std)
        }

    def __call__(self, tspan, y0, dt=.05):
        influxes = super().__call__(tspan, y0, dt=dt)
        t = influxes.t

        for key, (src, prob, mean, std) in self.readouts.items():
            influxes.y[key] = convolve_pdf(t, influxes.y[src], prob, mean, std)

        sol = SimulationResult(t, {})

        for key, val in influxes.y.items():
            if key not in ["susceptible", "population"]:
                sol.y[key] = np.cumsum(val, axis=0)
            else:
                sol.y[key] = val

        # FIXME: something wrong with this -- infectious > infected at small time
        sol.y['infectious'] = convolve_survival(t, influxes.y['infected'], 1, 5, 2)

        sol.y['critical'] = sol.y['icu'] - sol.y['general_ward'] - sol.y['dead']
        sol.y['ventilators'] = .73 * sol.y['critical']
        sol.y['hospitalized'] = (
            sol.y['admitted_to_hospital']
            - sol.y['hospital_recovered'] - sol.y['icu']
        )
        sol.y['hospitalized'] += (sol.y['general_ward']
                                  - sol.y['general_ward_recovered'])

        return sol
