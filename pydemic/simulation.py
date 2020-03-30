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


from scipy.stats import poisson
import numpy as np

__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: Simulation
.. autoclass:: SimulationState
.. autoclass:: StateLogger
"""


class SimulationState:
    """
    Manages the state for :class:`Simulation`'s.
    User-specified compartments are accessed as attributes, e.g.::

        >>> state = SimulationState(0., {'a': np.array([1., 0.])}, {})
        >>> state.a
        array([1., 0.])
        >>> state.t
        0.

    Note that :class:`Simulation` initializes state with an extra axis
    relative to the input data, corresponding to the number of requested
    stochastic samples (see :meth:`Simulation.__call__`).
    Any user-implemented axes occupy all but the first axis of the state arrays.

    .. automethod:: __init__
    .. automethod:: sum
    """

    def __init__(self, t, compartments, hidden_compartments):
        """
        :arg t: The current time.

        :arg compartments: A :class:`dict` of current values
            (as :class:`numpy.ndarray`'s) of all canonical compartments (the keys).

        :arg hidden_compartments: A :class:`dict` of current values
            (as :class:`numpy.ndarray`'s) of all compartments (the keys) not present
            in ``compartments`` (i.e., those used to implement
            and :class:`ErlangProcess`.)
        """

        self.t = t
        self.y = {**compartments, **hidden_compartments}
        self.compartments = list(compartments.keys())
        self.hidden_compartments = list(hidden_compartments.keys())

        self.sum_compartments = {}
        for item in self.compartments:
            self.sum_compartments[item] = [item]
            for full_key in self.hidden_compartments:
                if item == full_key.split(':')[0]:
                    self.sum_compartments[item].append(full_key)

    def __getattr__(self, item):
        return sum(self.y[key] for key in self.sum_compartments[item])

    def sum(self):
        """
        :returns: The total population across all summed compartments.
        """

        return sum(val.sum() for val in self.y.values())


class StateLogger:
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
        self.quantile_data = None

    def initialize_with_state(self, state):
        self.t[0] = state.t
        self.compartments = state.compartments.copy()
        self.y = {}
        for key in state.compartments:
            val = state.y[key]
            ary = np.zeros(shape=(self.chunk_length,)+val.shape)
            ary[0] = val
            self.y[key] = ary

    def __call__(self, state):
        self.slice += 1
        if self.slice == self.t.shape[0]:
            self.add_chunk()

        self.t[self.slice] = state.t
        for key in state.compartments:
            self.y[key][self.slice] = state.__getattr__(key)

    def add_chunk(self):
        self.t = np.concatenate([self.t, np.zeros(shape=(self.chunk_length,))])
        for key, val in self.y.items():
            shape = (self.chunk_length,)+val.shape[1:]
            self.y[key] = np.concatenate([val, np.zeros(shape=shape)])

    def cleanup(self, flatten_first_axis_if_unit=True):
        self.trim(flatten_first_axis_if_unit=flatten_first_axis_if_unit)

    def trim(self, flatten_first_axis_if_unit=True):
        self.t = self.t[:self.slice]
        for key in self.y.keys():
            if flatten_first_axis_if_unit and self.y[key].shape[1] == 1:
                self.y[key] = self.y[key][:self.slice, 0, ...]
            else:
                self.y[key] = self.y[key][:self.slice, ...]


class QuantileLogger:
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

    def __init__(self, chunk_length=1000, quantiles=[0.0455, 0.3173, 0.5, 0.6827, 0.9545]):
        self.quantiles = quantiles.copy()
        self.chunk_length = chunk_length
        self.t = np.zeros(shape=(self.chunk_length,))
        self.slice = 0

    def initialize_with_state(self, state):
        self.y_samples = {}
        self.t[0] = state.t
        self.compartments = state.compartments.copy()
        for key in self.compartments:
            val = state.y[key]
            ary = np.zeros(shape=(self.chunk_length,)+val.shape)
            ary[0] = val
            self.y_samples[key] = ary

    def __call__(self, state):
        self.slice += 1
        if self.slice == self.t.shape[0]:
            self.add_chunk()

        self.t[self.slice] = state.t
        for key in self.compartments:
            self.y_samples[key][self.slice] = state.__getattr__(key)

    def cleanup(self, flatten_first_axis_if_unit=True):
        self.trim(flatten_first_axis_if_unit=flatten_first_axis_if_unit)
        self.quantile_data = {}
        for key in self.y_samples:
            # FIXME: this will not work for Gillespie direct
            self.quantile_data[key] = np.array([ np.quantile(self.y_samples[key], quantile, axis=1)
                            for quantile in self.quantiles ])

    def add_chunk(self):
        self.t = np.concatenate([self.t, np.zeros(shape=(self.chunk_length,))])
        for key, val in self.y_samples.items():
            shape = (self.chunk_length,)+val.shape[1:]
            self.y_samples[key] = np.concatenate([val, np.zeros(shape=shape)])

    def trim(self, flatten_first_axis_if_unit=True):
        self.t = self.t[:self.slice]
        for key in self.y_samples.keys():
            if flatten_first_axis_if_unit and self.y_samples[key].shape[1] == 1:
                self.y_samples[key] = self.y_samples[key][:self.slice, 0, ...]
            else:
                self.y_samples[key] = self.y_samples[key][:self.slice, ...]


class Simulation:
    """
    Main driver for compartmental model simulations.

    .. automethod:: __init__
    .. automethod:: __call__

    .. attribute:: compartments

        The compartment names comprising the simulation state, inferred as the set of
        all :attr:`Reaction.lhs`'s and :attr:`Reaction.rhs`'s from the input list
        of :class:`Reaction`'s.
    """

    def __init__(self, reactions):
        """
        :arg reactions: A :class:`list` of :class:`Reaction`'s
            (or subclasses thereof) used to specify the dynamics of the
            compartmental model.
        """

        lhs_keys = set(x.lhs for x in reactions)
        rhs_keys = set(x.rhs for x in reactions)
        self.compartments = list(lhs_keys | rhs_keys)

        self._network = tuple(react for reaction in reactions
                              for react in reaction.get_reactions())

        all_lhs = set(x.lhs for x in self._network)
        all_rhs = set(x.rhs for x in self._network)
        self.hidden_compartments = list((all_lhs | all_rhs) - set(self.compartments))

    def print_network(self):
        for reaction in self._network:
            print(reaction)

    def step_gillespie_direct(self, time, state, dt):
        increments = {}

        for reaction in self._network:
            reaction_rate = reaction.evaluator(time, state)
            dY = reaction_rate

            if (reaction.lhs, reaction.rhs) in increments:
                increments[reaction.lhs, reaction.rhs] += dY
            else:
                increments[reaction.lhs, reaction.rhs] = dY

        # WARNING: need to be sure we're pulling from the right
        # reaction here! I might have solved an XY problem ...
        reactions = list(increments.keys())
        r1, r2 = np.random.rand(2)
        flattened_array = np.hstack([increments[k].reshape(-1) for k in reactions])
        cumulative_rates = np.cumsum(flattened_array)

        if cumulative_rates[-1] == 0.:
            return dt

        dt = - np.log(r1) / cumulative_rates[-1]
        r2 *= cumulative_rates[-1]
        reaction_index = np.searchsorted(cumulative_rates, r2)
        full_shape = (len(reactions),) + (increments[reactions[0]].shape)
        full_index = np.unravel_index(reaction_index, full_shape)

        # WARNING: It's also not entirely clear that this produces
        # the right rate distributions for processes that have two
        # different lhs <--> rhs reactions ...

        lhs, rhs = reactions[full_index[0]]
        state.y[lhs][full_index[1:]] -= 1.
        state.y[rhs][full_index[1:]] += 1.

        if state.y[lhs][full_index[1:]] < 0:
            state.y[lhs][full_index[1:]] += 1.
            state.y[rhs][full_index[1:]] -= 1.

        return dt

    def step(self, time, state, dt, stochastic_method=None):
        increments = {}

        for reaction in self._network:
            dY = np.empty_like(state.y[reaction.lhs])
            dY[...] = dt * reaction.evaluator(time, state)

            if stochastic_method == "tau_leap":
                dY[...] = poisson.rvs(dY)

            dY_max = state.y[reaction.lhs].copy()
            for (_lhs, _rhs), incr in increments.items():
                if reaction.lhs == _lhs:
                    dY_max -= incr
            dY = np.minimum(dY_max, dY)

            if (reaction.lhs, reaction.rhs) in increments:
                increments[reaction.lhs, reaction.rhs] += dY
            else:
                increments[reaction.lhs, reaction.rhs] = dY

        for (lhs, rhs), dY in increments.items():
            state.y[lhs] -= dY
            state.y[rhs] += dY

        return dt

    def initialize_full_state(self, time, y0, samples):
        compartment_vals = {}
        for key, ary in y0.items():
            ary = np.array(ary)
            shape = (samples,) + ary.shape
            compartment_vals[key] = np.empty(shape, dtype='float64')
            compartment_vals[key][...] = ary[None, ...]

        hidden_compartment_vals = {}
        template = compartment_vals[self.compartments[0]]
        for key in self.hidden_compartments:
            hidden_compartment_vals[key] = np.zeros_like(template)

        state = SimulationState(time, compartment_vals, hidden_compartment_vals)
        return state

    def __call__(self, tspan, y0, dt, stochastic_method=None, samples=1, seed=None, logger=None):
        """
        :arg tspan: A :class:`tuple` specifying the initiala and final times.

        :arg y0: A :class:`dict` with the initial values
            (as :class:`numpy.ndarray`'s) for each of :attr:`compartments`.

        :arg dt: The (initial) timestep to use.

        :arg stochastic_method: A :class:`string` specifying whether to use
            direct Gillespie stepping (`'direct'`) or :math:`\\tau`-leaing
            (`'tau_leap'`).
            Defaults to *None*, i.e., a deterministic evolution.

        :arg samples: The number of stochastic samples to simulate simultaneously.
            Defaults to ``1``.

        :arg seed: The value with which to seed :mod:`numpy`'s random number.
            Defaults to *None*, in which case no seed is passed.

        :returns: A :class:`~pydemic.simulation.StateLogger`.
        """

        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()

        start_time, end_time = tspan
        state = self.initialize_full_state(start_time, y0, samples)

        if logger is None:
            result = StateLogger()
        else:
            result = logger

        result.initialize_with_state(state)

        time = start_time
        while time < end_time:
            if stochastic_method in [None, "tau_leap"]:
                dt = self.step(time, state, dt, stochastic_method=stochastic_method)
            elif stochastic_method in ["direct"]:
                dt = self.step_gillespie_direct(time, state, dt)
            time += dt
            state.t = time
            result(state)

        result.cleanup()

        return result
