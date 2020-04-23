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

__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: SEIRPlusPlusSimulationV2
"""


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


def convolve_direct(t, influx, prefactor=1, mean=5, std=2, bad=False):
    shape, scale = mean_std_to_k_theta(mean, std)
    from scipy.stats import gamma
    if bad:
        cdf = gamma.cdf(t[:] - t[0], shape, scale=scale)  # BAD!!!!
    else:
        cdf = gamma.cdf(t[1:] - t[0], shape, scale=scale)
    pdf = np.diff(cdf, prepend=0)

    prefactor = prefactor * np.ones_like(influx[0, ...])

    end = t.shape[0]
    result = np.zeros_like(influx)
    for i in range(1, end):
        result[i, ...] = prefactor * np.dot(influx[i-1::-1].T, pdf[:i])

    return result


class NonMarkovianSIRSimulationBase:
    """
    Main driver for non-Markovian simulations.

    .. automethod:: __init__
    .. automethod:: __call__
    """

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

    def __init__(self, mitigation, *,
                 r0=3.2, serial_mean=4, serial_std=3.25,
                 seasonal_forcing_amp=.2, peak_day=15, **kwargs):
        self.mitigation = mitigation

        self.distribution_params = {
            'serial': mean_std_to_k_theta(serial_mean, serial_std),
        }
        self.seasonal_forcing_amp = seasonal_forcing_amp
        self.peak_day = peak_day
        self.r0 = r0

    def step(self, state, count, dt):
        fraction = (state.y['susceptible'][..., count-1] / self.population)
        coef = fraction * dt * self.mitigation(state.t[count])
        coef *= self.r0 * self.seasonal_forcing(state.t[count])
        update = coef * np.dot(
            state.y['infected'][..., count-1::-1],
            self.kernels['serial'][:count]
        )
        state.y['infected'][..., count] = update

        state.y['susceptible'][..., count] = (
            state.y['susceptible'][..., count-1] - update
        )

    def get_y0(self, total_population, initial_cases, age_distribution):
        """
        :arg total_population: FIXME: document

        :arg age_distribution: A :class:`dict` with key counts
            (as :class:`numpy.ndarray`'s) FIXME: document

        :returns: FIXME: document
        """

        # FIXME: shouldn't be set here
        self.population = total_population * np.array(age_distribution)
        n_demographics = len(age_distribution)

        y0 = {}
        for key in ('susceptible', 'infected'):
            y0[key] = np.zeros((n_demographics,))

        y0['infected'][...] = initial_cases / n_demographics
        y0['susceptible'][...] = self.population - y0['infected']

        return y0

    def __call__(self, tspan, y0, dt=.05):
        """
        :arg tspan: A :class:`tuple` specifying the initiala and final times.

        :arg y0: A :class:`dict` with the initial values
            (as :class:`numpy.ndarray`'s) for each of :attr:`compartments`.

        :returns: A :class:`~pydemic.simulation.StateLogger`. FIXME: maybe not?
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
        if isinstance(t, pd.DatetimeIndex):
            t_eval = (t - pd.to_datetime('2020-01-01')) / pd.Timedelta('1D')
        else:
            t_eval = t

        t0 = kwargs.pop('start_day')
        tf = t_eval[-1] + 2

        from pydemic.sampling import InvalidParametersError

        if t_eval[0] < t0 + 1:
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

        for key in ('dead',):
            if key in result.y.keys():
                spline = interp1d(result.t, result.y[key].sum(axis=-1), axis=0)
                y[key+'_incr'] = spline(t_eval) - spline(t_eval - 1)

        _t = pd.to_datetime(t_eval, origin='2020-01-01', unit='D')
        return pd.DataFrame(y, index=_t)


class SEIRPlusPlusSimulationV3(NonMarkovianSIRSimulationBase):
    """
    Main driver for non-Markovian simulations.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, mitigation, *,
                 r0=3.2, serial_mean=4, serial_std=3.25,
                 seasonal_forcing_amp=.2, peak_day=15,
                 incubation_mean=5.5, incubation_std=2, p_observed=1,
                 icu_mean=11, icu_std=5, p_icu=1, p_icu_prefactor=1,
                 dead_mean=7.5, dead_std=7.5, p_dead=1, p_dead_prefactor=1,
                 recovered_mean=7.5, recovered_std=7.5,
                 ifr=0.003, age_distribution=None, **kwargs):

        super().__init__(
            mitigation, r0=r0, serial_mean=serial_mean, serial_std=serial_std,
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
        """
        :arg tspan: A :class:`tuple` specifying the initial and final times.

        :arg y0: A :class:`dict` with the initial values
            (as :class:`numpy.ndarray`'s) for each of :attr:`compartments`.

        :returns: A :class:`~pydemic.simulation.StateLogger`. FIXME: maybe not?
        """

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
