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
.. autoclass:: TrackedSimulation
.. autoclass:: TrackedStateLogger
"""


class TrackedStateLogger:
    """
    Used to log simulation results returned by
    :meth:`Simulation.__call__`.

    .. attribute:: t

        A :class:`numpy.ndarray` of output times.

    .. attribute:: y

        A :class:`dict` whose values are :class:`numpy.ndarray`'s of the
        timeseries for each key (each of :attr:`Simulation.compartments`).
        The time axis is the first axis of the :class:`numpy.ndarray`'s.
    """

    def __init__(self, chunk_length=1000):
        self.chunk_length = chunk_length
        self.t = np.zeros(shape=(self.chunk_length,))
        self.slice = 0
        self.track_names = []
        self.y = {}

    def initialize_with_state(self, state):
        self.t[0] = state.t
        self.track_names = state.tracks.keys()
        self.y = {}
        for key in self.track_names:
            val = state.tracks[key]
            # FIXME: I think this should always force the last dimension
            # (time) to be collapsed
            ary = np.zeros(shape=(self.chunk_length,)+val.shape[:-1])
            self.y[key] = ary
            self.y[key][self.slice] = state.tracks[key].sum(axis=-1)
        self.slice = 1

    def __call__(self, state):
        if self.slice == self.t.shape[0]:
            self.add_chunk()
        self.t[self.slice] = state.t
        for key in state.tracks:
            self.y[key][self.slice] = state.tracks[key].sum(axis=-1)
        self.slice += 1

    def add_chunk(self):
        self.t = np.concatenate([self.t, np.zeros(shape=(self.chunk_length,))])
        for key, val in self.y.items():
            shape = (self.chunk_length,)+val.shape[1:]
            self.y[key] = np.concatenate([val, np.zeros(shape=shape)])

    def cleanup(self, flatten_first_axis_if_unit=True):
        self.trim(flatten_first_axis_if_unit=flatten_first_axis_if_unit)

    def trim(self, flatten_first_axis_if_unit=True):
        self.t = self.t[:self.slice]
        for key in self.y:
            if self.y[key].ndim > 1:
                if flatten_first_axis_if_unit and self.y[key].shape[1] == 1:
                    self.y[key] = self.y[key][:self.slice, 0, ...]
                else:
                    self.y[key] = self.y[key][:self.slice, ...]
            else:
                self.y[key] = self.y[key][:self.slice]

    def __repr__(self):
        text = "{0:s} with\n".format(str(type(self)))
        text += "  - t from {0:g} to {1:g}\n".format(self.t[0], self.t[-1])
        for key in self.track_names:
            text += "  - {0:s} {1:s}\n".format(key, str(self.y[key].shape))
        return text[:-1]

    def save_tsv(self, fname, dt_days=None, save_times=False):
        """
        Used to save contents in easy-to-read tsv file format.

        :arg fname: A :class:`string` path for where to save the data.

        The following keyword arguments are also recognized:

        :arg dt_days: The minimum cadence in days at which to save the data.
            Defautls to *None*, in which case data is output at all datapoints.

        :arg save_times: Whether to outputs a second column in the tsv containing
            the times (in ``HH:MM:SS`` format) for each data point.
            Defaults to *False*.
        """
        # FIXME: rewrite this to deal with internal track_name / passive classes?
        dates = [datetime(2020, 1, 1)+timedelta(days=x) for x in self.t]
        compartment_data = {}
        fp = open(fname, 'w')
        fp.write("date")
        if save_times:
            fp.write("\ttime")
        for key in self.track_names:
            fp.write("\t"+key)
            compartment_data[key] = self.y[key].sum(axis=-1)
        fp.write("\n")
        last_date = None
        for i in range(len(dates)):
            if last_date is not None and dt_days is not None:
                if (dates[i]-last_date).days < dt_days:
                    continue
            last_date = dates[i]
            fp.write(dates[i].strftime("%y-%m-%d")+"\t")
            if save_times:
                fp.write(dates[i].strftime("%H:%M:%S")+"\t")
            fp.write("\t".join("{0:g}".format(
                compartment_data[x][i]) for x in self.track_names)+"\n")
        fp.close()


class TrackedSimulationState:

    def __init__(self, time, tracks):
        """
        :arg t: The current time.
        """
        self.t = time
        self.tracks = tracks


class TrackedSimulation:
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

    def __init__(self, tspan, dt=1.):

        self.dt = dt
        n_demographics = 1

        n_bins = int((tspan[1] - tspan[0]) / dt + 1)
        print("creating simulation object with {0:d} time bins".format(n_bins))

        # FIXME: maybe threshold these values based on cumulative sum
        # (< some max 1./population)?

        # parameters used below from Alexei's post
        # "relevant-delays-for-our-model" on March 30th.

        # generate times for interval generation
        ts = np.arange(0, n_bins) * dt

        # parameters for serial interval
        R0 = 3.2
        serial_k = 1.5  # shape 1.5 -> 2.
        serial_theta = 4.  # scale 4 -> 5

        # parameters for incubation time
        p_symptomatic = 0.8  # the percentage of people who become symptomatic
        incubation_k = 3     # 3+ from alexei
        # FIXME: confirm it should be this and not 1/this  (5 -> 6)
        incubation_theta = 5

        # parameters for testing positive

        # percentage of people who become symptomatic who end up testing positive
        # these aren't used, so they're commented out!
        # p_confirmed = 0.8
        # positive_k = 1
        # positive_theta = 5.  # 5 -> 10 from Alexei

        # parameters for those who follow the symptomatic -> ICU -> death track
        p_dead = 0.05    # percentage of symptomatic individuals who die
        icu_k = 1.
        icu_theta = 9    # 9 -> 11 from Alexei
        dead_k = 1       #
        dead_theta = 7   # 7 -> 8 from Alexei

        # in principle we have another set of distributions for those
        # who should go from onset -> hospital (including ICU?) -> recovered,
        # but we don't have
        # numbers for those values. we can "fake" this by changing the ratios
        # in the above class
        # of individuals who go to the ICU but don't die?

        # FIXME: can also change "population", "susceptible", and "dead"
        # into non-time series
        # arrays that just directly accumulate. one might call them
        # "observer" tracks?

        # FIXME: it might be possible to take integrated values from the cdf
        # here instead of
        # point sampling the pdf to achieve faster convergence
        self.kernels = [
            gamma.pdf(ts, serial_k, scale=serial_theta),
            gamma.pdf(ts, incubation_k, scale=incubation_theta),
            gamma.pdf(ts, icu_k, scale=icu_theta),
            gamma.pdf(ts, dead_k, scale=dead_theta),
        ]

        self.tracks = {
            "susceptible": np.zeros((n_demographics, n_bins)),
            "infected": np.zeros((n_demographics, n_bins)),
            "symptomatic": np.zeros((n_demographics, n_bins)),
            "critical_dead": np.zeros((n_demographics, n_bins)),
            "dead": np.zeros((n_demographics, n_bins)),
            "population": np.zeros((n_demographics, n_bins))
        }

        def update_infected(state, count):
            fraction = (state.tracks['susceptible'][:, count-1]
                        / state.tracks['population'][:, 0])
            update = fraction * R0 * np.dot(
                state.tracks['infected'][0, count::-1],
                self.kernels[0][:count+1]
            )
            update *= self.dt

            # FIXME: does it make sense to update here?
            state.tracks['susceptible'][:, count] = (
                state.tracks['susceptible'][:, count-1] - update
            )
            return update  # alternatively, always update in these functions?

        def update_symptomatic(state, count):
            symptomatic_source = p_symptomatic * self.dt * np.dot(
                state.tracks['infected'][0, count::-1],
                self.kernels[1][:count+1]
            )
            return symptomatic_source

        def update_icu_dead(state, count):
            icu_dead_source = p_dead * \
                np.dot(state.tracks['symptomatic'][0, count::-1],
                       self.kernels[2][:count+1]) * self.dt
            return icu_dead_source

        def update_dead(state, count):
            dead_source = self.dt * np.dot(
                state.tracks['critical_dead'][0, count::-1],
                self.kernels[3][:count+1])
            return dead_source

        self.sources = {
            "susceptible": [
            ],
            "infected": [
                # FIXME: does not work for demographics
                # FIXME: some of these could be lambdas, if desired.
                update_infected
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
                state.tracks[track][:, count] = source(state, count)

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

        state = TrackedSimulationState(times, self.tracks)

        for key in y0:
            state.tracks[key][:, 0] = y0[key]

        count = 0
        time = start_time
        while time < end_time:

            # this ordering is correct!
            # state[0] corresponds to start_time
            count += 1
            time += self.dt
            self.step(state, count)

        return state

    # def deprecated__call__(self, tspan, y0, dt=1.):
    #     # suppose we have n_demographics number of demographics
    #     # and that we have n_tracks tracks. each track will have
    #     # shape = (n_demographics, n_steps) where
    #     # n_steps = ( (end_time - start_time) / dt ) + 1
    #     # and thus the "whole state" will have shape
    #     # n_tracks, n_demographics, n_steps

    #     # get time dimensions
    #     start_time, end_time = tspan
    #     n_steps = int((end_time - start_time) / dt + 1)

    #     # get full state object
    #     n_tracks, n_demographics = y0.shape
    #     state = np.zeros((n_tracks, n_demographics, n_steps))
    #     state[:, :, -1] = y0

    #     # get time kernels
    #     time_kernels = np.zeros((len(self.kernels), n_steps))
    #     for i in range(len(self.kernels)):
    #         time_kernels[i, :] = self.kernels[i]

    #     # get demographic kernels
    #     demographic_kernels = np.ones((n_demographics, n_demographics))

    #     # interface with c
    #     from pydemic.ctypes import cfunc
    #     cmodule = cfunc.cfunc()

    #     # FIXME: the kernels here are assumed to be separable,
    #     # which is not necessarily
    #     # the right way to deal with this...
    #     cmodule.evolve_track_simulation(state, time_kernels, demographic_kernels)

    # def bad__call__(self, tspan, y0):
    #     # get time dimensions
    #     start_time, end_time = tspan
    #     n_steps = int((end_time - start_time) / self.dt + 1)

    #     # get full state object
    #     n_tracks, n_demographics = y0.shape
    #     state = np.zeros((n_tracks, n_demographics, n_steps))
    #     state[:, :, -1] = y0

    #     # fill out each element of the state object
    #     time = start_time
    #     index = n_steps - 2
    #     while time < end_time:
    #         self.step_new(state, index, time)
    #         time += self.dt

    #     return None

    def get_y0(self, population, infected):
        # FIXME: set these shapes by n_demographics

        y0 = {}
        for key in self.tracks:
            y0[key] = np.array([0.])

        y0['population'][:] = population
        y0['susceptible'][:] = population - infected
        y0['infected'][:] = infected

        return y0
