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

__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: SEIRPlusPlusSimulation
"""


class SEIRPlusPlusSimulationState:
    def __init__(self, time, y):
        self.t = time
        self.y = y


class SEIRPlusPlusSimulation:
    """
    Main driver for tracked class model simulations.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def set_kernels(self, t):
        from scipy.stats import gamma
        self.kernels = {
            key: gamma.pdf(t, shape, scale=scale)
            for key, (shape, scale) in self.distribution_params.items()
        }

    def seasonal_forcing(self, t):
        phase = 2 * np.pi * (t - self.peak_day) / 365
        return (1 + self.seasonal_forcing_amp * np.cos(phase))

    def __init__(self, mitigation,
                 r0=3.2, serial_k=1.5, serial_mean=4.,
                 p_symptomatic=1.0, incubation_k=3., incubation_mean=5.,
                 p_positive=1.0, positive_k=1., positive_mean=5.,
                 p_dead=1., icu_k=1., icu_mean=9., dead_k=1., dead_mean=7.,
                 p_hospitalized=1., hospital_removed_k=6.,
                 hospital_removed_mean=12.,  # corresponds to mean ~ 12, std ~ 4.9
                 seasonal_forcing_amp=.2, peak_day=15,
                 **kwargs):
        self.mitigation = mitigation
        self.distribution_params = {
            'serial': (serial_k, serial_mean/serial_k),
            'incubation': (incubation_k, incubation_mean/incubation_k),
            'icu': (icu_k, icu_mean/icu_k),
            'dead': (dead_k, dead_mean/dead_k),
            'positive': (positive_k, positive_mean/positive_k),
            'hospital_removed': (hospital_removed_k,
                                 hospital_removed_mean/hospital_removed_k),
        }
        self.seasonal_forcing_amp = seasonal_forcing_amp
        self.peak_day = peak_day

        p_positive = p_positive * np.array([5, 5, 10, 15, 20, 25, 30, 40, 50]) / 100
        p_symptomatic = p_symptomatic * np.ones_like(p_positive)
        p_dead = p_dead * np.array([7.5e-6, 4.5e-5, 9.e-5, 2.025e-4, 7.2e-4,
                                    2.5e-3, 1.05e-2, 3.15e-2, 6.875e-2])
        p_hospitalized_given_positive = p_hospitalized * np.ones_like(p_positive)

        def update_infected(state, count, dt):
            fraction = (state.y['susceptible'][..., count-1]
                        / state.y['population'][..., 0])
            update = fraction * r0 * np.dot(
                state.y['infected'][..., count-1::-1],
                self.kernels['serial'][:count]
            )
            update *= dt * self.mitigation(state.t[count])
            update *= self.seasonal_forcing(state.t[count])

            # FIXME: does it make sense to update here?
            state.y['susceptible'][..., count] = (
                state.y['susceptible'][..., count-1] - update
            )
            return update  # alternatively, always update in these functions?

        def update_symptomatic(state, count, dt):
            symptomatic_source = p_symptomatic * dt * np.dot(
                state.y['infected'][..., count-1::-1],
                self.kernels['incubation'][:count]
            )
            return symptomatic_source

        def update_icu_dead(state, count, dt):
            icu_dead_source = dt * p_dead * np.dot(
                state.y['symptomatic'][..., count-1::-1],
                self.kernels['icu'][:count]
            )
            return icu_dead_source

        def update_dead(state, count, dt):
            dead_source = dt * np.dot(
                state.y['critical_dead'][..., count-1::-1],
                self.kernels['dead'][:count])
            return dead_source

        def update_removed_from_hospital(state, count, dt):
            removed_source = dt * p_hospitalized_given_positive * np.dot(
                state.y['positive'][..., count-1::-1],
                self.kernels['hospital_removed'][:count])
            return removed_source

        def update_positive(state, count, dt):
            positive_source = dt * p_positive * np.dot(
                state.y['symptomatic'][..., count-1::-1],
                self.kernels['positive'][:count]
            )
            return positive_source

        self.sources = {
            "susceptible": [],
            "infected": [update_infected],
            "symptomatic": [update_symptomatic],
            "positive": [update_positive],
            "hospital_removed": [update_removed_from_hospital],
            "critical_dead": [update_icu_dead],
            "dead": [update_dead],
            "population": []
        }

    def step(self, state, count, dt):
        for track in state.y:
            for source in self.sources[track]:
                state.y[track][..., count] = source(state, count, dt)

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
        self.set_kernels(times[1:] - start_time)

        y0_all_t = {}
        for key in y0:
            y0_all_t[key] = np.zeros(y0[key].shape + (n_steps,))
            y0_all_t[key][..., 0] = y0[key]

        state = SEIRPlusPlusSimulationState(times, y0_all_t)

        for count in range(1, n_steps):
            self.step(state, count, dt)

        for key, val in state.y.items():
            if key not in ["susceptible", "population"]:
                state.y[key] = np.cumsum(val, axis=1).T
            else:
                state.y[key] = val.T

        return state

    def get_y0(self, total_population, initial_cases, age_distribution):
        """
        :arg population: FIXME: document

        :arg age_distribution: A :class:`dict` with key counts
            (as :class:`numpy.ndarray`'s) FIXME: document

        :returns: FIXME: document
        """

        n_demographics = len(age_distribution)

        y0 = {}
        for key in self.sources.keys():
            y0[key] = np.zeros((n_demographics,))

        y0['population'][...] = np.array(age_distribution) * total_population
        y0['infected'][...] = initial_cases / n_demographics
        y0['susceptible'][...] = y0['population'] - y0['infected']

        return y0


from pydemic.sampling import (LikelihoodEstimatorBase, clipped_l2_log_norm,
                              poisson_norm)


class SEIRPlusPlusEstimator(LikelihoodEstimatorBase):
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

        # get model data at daily values
        # when computing diffs, datasets were prepended with 0, so there is no need
        # to evaluate at an extra data point on day earlier
        t_eval = np.arange(self.data.t[0], self.data.t[-1]+2)
        model_data = self.get_model_data(
            t_eval, **parameters, **self.fixed_values
        )
        if model_data == -np.inf:
            return -np.inf
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
        t0 = kwargs.pop('start_day')
        tf = kwargs.pop('end_day')

        from pydemic.containment import MitigationModel
        mitigation = MitigationModel.init_from_kwargs(t0, tf, **kwargs)

        # ensure times are ordered
        if any(np.diff(mitigation.times, prepend=t0, append=tf) < 0):
            return -np.inf
        if any(np.diff(mitigation.times) < kwargs.get('min_mitigation_spacing', 5)):
            return -np.inf

        sim = SEIRPlusPlusSimulation(mitigation=mitigation, **kwargs)
        y0 = sim.get_y0(kwargs.pop('total_population'),
                        kwargs.pop('initial_cases'),
                        kwargs.pop('age_distribution'))
        result = sim((t0, tf), y0)

        y = {}
        from scipy.interpolate import interp1d
        for key, val in result.y.items():
            y[key] = interp1d(result.t, val, axis=0)(t)

        from pydemic.data import CaseData
        result = CaseData(t, y)
        return result
