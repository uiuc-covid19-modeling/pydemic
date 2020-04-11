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

from scipy.stats import gamma
import numpy as np

__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: NonMarkovianSimulation
"""


class NonMarkovianSimulationState:

    def __init__(self, time, tracks):
        """
        :arg t: The current time.
        """
        self.t = time
        self.tracks = tracks


class NonMarkovianSimulation:
    """
    Main driver for tracked class model simulations.

    .. automethod:: __init__
    .. automethod:: __call__

    .. attribute:: tracks

        FIXME: rewrite The compartment names comprising the simulation state,
        inferred as the set of
        all :attr:`Reaction.lhs`'s and :attr:`Reaction.rhs`'s from the input list
        of :class:`Reaction`'s.
    """

    def __init__(self, tspan, mitigation, dt=1.,
                 r0=3.2, serial_k=1.5, serial_mean=4.,
                 p_symptomatic=1.0, incubation_k=3., incubation_mean=5.,
                 p_positive=1.0, positive_k=1., positive_mean=5.,
                 p_dead=1., icu_k=1., icu_mean=9., dead_k=1., dead_mean=7.):
        """

            # given k, theta, translate to mean, std
            # k = 9.0
            # theta = 1.0
            mean = k * theta
            std = np.sqrt(k*theta*theta)

            # given mean, std, translate to k, theta
            # mean = 10.
            # std = 1.
            theta = std*std/mean
            k = mean / theta

            # given mean, k, translate to k, theta
            # mean = 10.
            # k = 25.
            theta = mean / k
            std = np.sqrt(mean * theta)

            parameters used below from Alexei's post:
            "relevant-delays-for-our-model" on March 30th.

                default suggested ranges are:
                    serial_k            1.5 ->  2
                    serial_mean         4   ->  5
                    p_symptomatic       ??
                    incubation_k        3+
                    incubation_mean     5   ->  6
                    p_positive          ??
                    positive_k          1
                    positive_mean       5   -> 10
                    p_dead              ??
                    icu_k               1
                    icu_mean            9   -> 11
                    dead_k              1
                    dead_mean           7   ->  8

            new notes from Alexei's post:
            "Refining parameters of the model" on April 9th.

                Infection -> onset (incubation)
                    mean ~ 5-6 and sd ~ 2
                    Alexei suggests using lognormal here. Data from the link Alexei posted are
                    Log-normal: 1.621 (1.504-1.755) and 0.418 (0.271-0.542)
                    Gamma: 5.807 (3.585-13.865) and 0.948 (0.368-1.696)
                    Weibull: 2.453 (1.917-4.171) and 6.258 (5.355-7.260)
                    Erlang: 6 (3-11) and 0.880 (0.484-1.895)
                    FOR our gamma kernel, suggested to use
                        mean ~ 5.5, theta ~ 0.72, k = 7.563

                Onset -> ICU
                    mean ~ 10, k=4-6
                    Alexei notes: literature claims 10-12 days as mean/median and distribution
                    is narrow. This is dependent on onset-to-death curve. The k=4-6 value comes
                    from estimates of onset->death distributions and knowledge of below
                    ICU -> death.

                ICU -> death
                    time constant ~ 7-8 days
                    death probability ~ 50-75%, so to recover ICU (for patients who do not die),
                    we need to modify slightly. Also note timescale for non-terminal patients
                    may be longer.

                * onset -> hospitalization
                    "typically 5-7 days" with high latency which suggests relatively high k.
                    I'm going to use mean ~ 6 and k ~ 36  [6 + 1]
                    alternatively could use ~ 6 and k ~ 9

                * bureaucratic delays
                    about 3 days for reporting, Alexei suggests a form factor 2

                4+5 well-described by gamma with mean ~ 11 and SD ~ 1
                    theta ~ 0.09, k ~ 121

            notes from the "Parameter estimates" spreadsheet shared by the Cobey lab

                incubation time: mean ~ 5.1 - 6.8 with CI ~ pm 0.5 - 1.0, this
                    is roughly consistent with the above fits by Alexei

                onset -> icu:
                    Zhou(Wuhan) ~ 12 and 8-15  (mean, error)
                    Gaythorpe(Hong Kong+Japan), 5.76 & 4.22   (mean, error)
                    Wang(Wuhan), 7, 4-8
                    Wang+Zhu(5, 4-7

                icu -> death
                    Yang 7, 3-11
                    Bhatraju ~ 6.6, std = 3.4

                onset -> hospitalization
                    many different values...


        """

        self._mitigation = mitigation

        self.dt = dt
        demo_shape = (9,)
        n_bins = int((tspan[1] - tspan[0]) / dt + 2)

        # custom severity model following Neher's data from China. in particular:
        #   p_symptomatic = 1.
        #   p_positive = confirmed
        #   p_dead = confirmed * severe * critical * fatal
        #
        p_positive = np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]) / 100. * p_positive
        p_symptomatic = np.ones(demo_shape) * p_symptomatic
        p_dead = np.array([7.5e-6, 4.5e-5, 9.e-5, 2.025e-4, 7.2e-4, 2.5e-3, 1.05e-2, 3.15e-2, 6.875e-2]) * p_dead

        # FIXME: in principle we have another set of distributions for those
        # who should go from onset -> hospital (including ICU?) -> recovered,
        # but we don't have numbers for those values. we can "fake" this by
        # changing the ratios in the above class of individuals who go to the
        # ICU but don't die?

        # FIXME: can also change "population", "susceptible", and "dead"
        # into non-time series
        # arrays that just directly accumulate. one might call them
        # "observer" tracks?

        # FIXME: maybe threshold these values based on cumulative sum
        # (< some max 1./population)?

        # FIXME: it might be possible to take integrated values from the cdf
        # here instead of point sampling the pdf to achieve faster convergence

        ts = np.arange(0, n_bins) * dt
        self.kernels = [
            gamma.pdf(ts, serial_k, scale=serial_mean/serial_k),
            gamma.pdf(ts, incubation_k, scale=incubation_mean/incubation_k),
            gamma.pdf(ts, icu_k, scale=icu_mean/icu_k),
            gamma.pdf(ts, dead_k, scale=dead_mean/dead_k),
            gamma.pdf(ts, positive_k, scale=positive_mean/positive_k),
        ]

        self.tracks = {
            "susceptible": np.zeros(demo_shape+(n_bins,)),
            "infected": np.zeros(demo_shape+(n_bins,)),
            "symptomatic": np.zeros(demo_shape+(n_bins,)),
            "positive": np.zeros(demo_shape+(n_bins,)),
            "critical_dead": np.zeros(demo_shape+(n_bins,)),
            "dead": np.zeros(demo_shape+(n_bins,)),
            "population": np.zeros(demo_shape+(n_bins,))
        }

        def update_infected(state, count):
            fraction = (state.tracks['susceptible'][..., count-1]
                        / state.tracks['population'][..., 0])
            update = fraction * r0 * np.dot(
                state.tracks['infected'][..., count::-1],
                self.kernels[0][:count+1]
            )
            update *= self._mitigation(state.t[count])
            update *= self.dt

            # FIXME: does it make sense to update here?
            state.tracks['susceptible'][..., count] = (
                state.tracks['susceptible'][..., count-1] - update
            )
            return update  # alternatively, always update in these functions?

        def update_symptomatic(state, count):
            symptomatic_source = p_symptomatic * self.dt * np.dot(
                state.tracks['infected'][..., count::-1],
                self.kernels[1][:count+1]
            )
            return symptomatic_source

        def update_icu_dead(state, count):
            icu_dead_source = p_dead * \
                np.dot(state.tracks['symptomatic'][..., count::-1],
                       self.kernels[2][:count+1]) * self.dt
            return icu_dead_source

        def update_dead(state, count):
            dead_source = self.dt * np.dot(
                state.tracks['critical_dead'][..., count::-1],
                self.kernels[3][:count+1])
            return dead_source

        def update_positive(state, count):
            positive_source = p_positive * self.dt * np.dot(
                state.tracks['symptomatic'][..., count::-1],
                self.kernels[4][:count+1]
            )
            return positive_source

        self.sources = {
            "susceptible": [
            ],
            "infected": [
                update_infected
            ],
            "positive": [
                update_positive
            ],
            "symptomatic": [
                update_symptomatic
            ],
            "critical_dead": [
                update_icu_dead
            ],
            "dead": [
                update_dead
            ],
            "population": [
            ]
        }

    def step(self, state, count):
        for track in state.tracks:
            for source in self.sources[track]:
                state.tracks[track][..., count] = source(state, count)

    def __call__(self, tspan, y0):
        """
        :arg tspan: A :class:`tuple` specifying the initiala and final times.

        :arg y0: A :class:`dict` with the initial values
            (as :class:`numpy.ndarray`'s) for each of :attr:`compartments`.

        :returns: A :class:`~pydemic.simulation.StateLogger`. FIXME: maybe not?
        """

        start_time, end_time = tspan
        n_steps = int((end_time-start_time)/self.dt + 2)
        times = np.linspace(start_time, end_time, n_steps)
        state = NonMarkovianSimulationState(times, self.tracks)

        for key in y0:
            state.tracks[key][..., 0] = y0[key]

        count = 0
        time = start_time
        while time < end_time:

            # this ordering is correct!
            # state[0] corresponds to start_time
            count += 1
            time += self.dt
            self.step(state, count)

        return state

    def get_y0(self, population, age_distribution):
        """
        :arg population: FIXME: document

        :arg age_distribution: A :class:`dict` with key counts
            (as :class:`numpy.ndarray`'s) FIXME: document

        :returns: FIXME: document
        """

        n_demographics = len(age_distribution.counts)
        population_scale = population.population_served / sum(age_distribution.counts)

        y0 = {}
        for key in self.tracks:
            y0[key] = np.zeros((n_demographics))

        y0['population'][...] = np.array(age_distribution.counts) * population_scale
        y0['infected'][...] = population.initial_cases / n_demographics
        y0['susceptible'][...] = y0['population'] - y0['infected']

        return y0
