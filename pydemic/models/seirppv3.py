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
from scipy.interpolate import interp1d

__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: SEIRPlusPlusSimulationV2
"""


class SEIRPlusPlusSimulationState:
    def __init__(self, time, y):
        self.t = time
        self.y = y


class SEIRPlusPlusSimulationV3:
    """
    Main driver for tracked class model simulations.

    We track the infectious loop in the usual way with
        serial_k = 1.5, serial_mean = 4., and allow r0 to float

    Infected persons become symptomatic after
        incubation_k = 7., incubation_mean = 5.5
        and with probability
        p_symptomatic initialized to 1., but explored.

    A fraction of the symptomatic people are observed according to
        p_observed

    In the second (v3) model, some fraction of observed individuals end up
        in the icu after
        icu_mean=11., icu_std=5.
        with probability roughly proportional to ccdphcd deaths and scaled by
        p_icu_prefactor

    In this second (v3) model, some fraction of icu individuals die after
        dead_mean=7.5, dead_std=7.5.
        with uniform probability over demographics, scaled by
        p_dead_prefactor

    Tracks are thus then called:
        population (constant)
        susceptible
        infected
        observed  (= p_observed * symptomatic)
        icu
        dead
        recovered  (from icu)


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
                 serial_mean=4., serial_std=3.25, r0=3.2,
                 incubation_mean=5.5, incubation_std=2., 
                 #p_symptomatic=1.,
                 ifr=0.003,
                 p_observed=1.,
                 icu_mean=11., icu_std=5., p_icu=1., p_icu_prefactor=1.,
                 dead_mean=7.5, dead_std=7.5, p_dead=1., p_dead_prefactor=1., dead_force_exp=False,
                 recovered_mean=7.5, recovered_std=7.5, 
                 # original onset->death: mean=18, std=8
                 seasonal_forcing_amp=.2, peak_day=15,
                 age_distribution=np.array([0.24789492, 0.13925591, 0.13494838, 0.12189751, 0.12724997, 0.11627754, 0.07275651, 0.03971926]),
                 **kwargs):

        self.mitigation = mitigation

        def mean_std_to_k_theta(mean, std):
            return (mean**2 / std**2, std**2 / mean)

        if dead_force_exp:
            dead_std = dead_mean

        self.distribution_params = {
            'serial': mean_std_to_k_theta(serial_mean, serial_std),
            'incubation': mean_std_to_k_theta(incubation_mean, incubation_std),
            'icu': mean_std_to_k_theta(icu_mean, icu_std),
            'dead': mean_std_to_k_theta(dead_mean, dead_std),
            'recovered': mean_std_to_k_theta(recovered_mean, recovered_std)
        }
        self.seasonal_forcing_amp = seasonal_forcing_amp
        self.peak_day = peak_day

        p_symptomatic = 1.
        if type(p_symptomatic) != list and type(p_symptomatic) != np.ndarray:
            p_symptomatic = p_symptomatic * np.ones(8)
        if type(p_observed) != list and type(p_observed) != np.ndarray:
            p_observed = p_observed * np.ones(8)
        if type(p_icu) != list and type(p_icu) != np.ndarray:
            p_icu = p_icu * np.ones(8)
        if type(p_dead) != list and type(p_dead) != np.ndarray:
            p_dead = p_dead * np.ones(8)

        p_symptomatic = np.array(p_symptomatic)
        p_observed = np.array(p_observed)
        p_icu = np.array(p_icu) * p_icu_prefactor
        p_dead = np.array(p_dead) * p_dead_prefactor
        p_recovered = np.ones(p_dead.shape) - p_dead

        # FIXME: this is a kludge-y way to set the target ifr (infection, not just symptomatic)
        target_ifr = ifr
        p_dead_all = p_symptomatic * p_observed * p_icu * p_dead
        synthetic_ifr = (p_dead_all * age_distribution).sum()
        p_symptomatic *= target_ifr / synthetic_ifr

        def update_infected(state, count, dt):
            fraction = (state.y['susceptible'][..., count-1]
                        / state.y['population'][..., 0])
            update = fraction * r0 * np.dot(
                state.y['infected'][..., count-1::-1],
                self.kernels['serial'][:count]
            )
            update *= dt * self.mitigation(state.t[count])
            update *= self.seasonal_forcing(state.t[count])

            state.y['susceptible'][..., count] = (
                state.y['susceptible'][..., count-1] - update
            )
            return update

        def update_observed(state, count, dt):
            update = p_observed * p_symptomatic * dt * np.dot(
                state.y['infected'][..., count-1::-1],
                self.kernels['incubation'][:count]
            )
            return update

        def update_icu(state, count, dt):
            update = p_icu * dt * np.dot(
                state.y['observed'][..., count-1::-1],
                self.kernels['icu'][:count])
            return update

        def update_dead(state, count, dt):
            update = p_dead * dt * np.dot(
                state.y['icu'][..., count-1::-1],
                self.kernels['dead'][:count])
            return update

        def update_recovered(state, count, dt):
            update = p_recovered * dt * np.dot(
                state.y['icu'][..., count-1::-1],
                self.kernels['recovered'][:count])
            return update

        self.sources = {
            "susceptible": [],
            "infected": [update_infected],
            "observed": [update_observed],
            "icu": [update_icu],
            "dead": [update_dead],
            "recovered": [update_recovered],
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
        state.y["critical"] = state.y["icu"] - state.y["dead"] - state.y["recovered"]

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


from pydemic.sampling import LikelihoodEstimatorBase


class SEIRPlusPlusEstimator(LikelihoodEstimatorBase):
    @classmethod
    def get_model_data(cls, t, **kwargs):
        t0 = kwargs.pop('start_day')
        tf = kwargs.pop('end_day')

        from pydemic.containment import LinearMitigationModel
        mitigation = LinearMitigationModel.init_from_kwargs(t0, tf, **kwargs)

        # ensure times are ordered
        if any(np.diff(mitigation.times, prepend=t0, append=tf) < 0):
            return -np.inf
        if any(np.diff(mitigation.times) < kwargs.get('min_mitigation_spacing', 5)):
            # FIXME: ideally I'd like some check on this for manual models
            #raise ValueError("mitigation point spacing < min_mitigation_spacing")
            return -np.inf

        age_distribution = kwargs.pop('age_distribution')
        sim = SEIRPlusPlusSimulationV3(mitigation=mitigation, age_distribution=age_distribution, **kwargs)
        y0 = sim.get_y0(kwargs.pop('total_population'),
                        kwargs.pop('initial_cases'),
                        age_distribution)
        result = sim((t0, tf), y0)

        y = {}
        for key, val in result.y.items():
            # FIXME: maybe deal with bounds error differently? need this for the forecasting thing.
            y[key] = interp1d(result.t, val, axis=0, bounds_error=False)(t)

        from pydemic.data import CaseData
        result = CaseData(t, y)
        return result
