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

from datetime import datetime, timedelta
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

    def __init__(self, tspan, dt=1., 
                 R0=3.2, serial_k=1.5, serial_theta=4.,
                 p_symptomatic=0.8, incubation_k=3., incubation_theta=5.,
                 p_positive=0.8, positive_k=1., positive_theta=5.,
                 p_dead=0.05, icu_k=1., icu_theta=9., dead_k=1., dead_theta=7.):
        """

            parameters used below from Alexei's post:
            "relevant-delays-for-our-model" on March 30th.

            default suggested ranges are:
                serial_k            1.5 ->  2
                serial_theta        4   ->  5
                p_symptomatic       ??
                incubation_k        3+
                incubation_theta    5   ->  6
                p_positive          ??
                positive_k          1
                positive_theta      5   -> 10
                p_dead              ??
                icu_k               1
                icu_theta           9   -> 11
                dead_k              1
                dead_theta          7   ->  8


        """
        
        self.dt = dt
        demo_shape = (9,)
        n_bins = int((tspan[1] - tspan[0]) / dt + 1)

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
            gamma.pdf(ts, serial_k, scale=serial_theta),
            gamma.pdf(ts, incubation_k, scale=incubation_theta),
            gamma.pdf(ts, icu_k, scale=icu_theta),
            gamma.pdf(ts, dead_k, scale=dead_theta),
            gamma.pdf(ts, positive_k, scale=positive_theta),
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
            update = fraction * R0 * np.dot(
                state.tracks['infected'][..., count::-1],
                self.kernels[0][:count+1]
            )
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
        n_steps = int((end_time-start_time)/self.dt + 1)

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

    def get_y0(self, population, infected):
        # FIXME: set these shapes by n_demographics

        y0 = {}
        for key in self.tracks:
            y0[key] = np.array([0.])

        y0['population'][...] = population
        y0['susceptible'][...] = population - infected
        y0['infected'][...] = infected

        return y0
