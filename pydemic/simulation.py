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

    :arg t: The current time.

    :arg compartments: A :class:`dict` of current values
        (as :class:`numpy.ndarray`'s) of all canonical compartments (the keys).

    :arg hidden_compartments: A :class:`dict` of current values
        (as :class:`numpy.ndarray`'s) of all compartments (the keys) not present
        in ``compartments`` (i.e., those used to implement
        and :class:`ErlangProcess`.)

    :arg passive_compartments: A :class:`tuple` of compartment keys which are
        computed for :class:`PassiveReaction`'s
        (i.e., those which do not count toward the total population).

    .. automethod:: sum
    """

    def __init__(self, t, compartments, hidden_compartments, passive_compartments):
        self.t = t
        self.y = {**compartments, **hidden_compartments}
        self.compartments = list(compartments.keys())
        self.hidden_compartments = list(hidden_compartments.keys())
        self.passive_compartments = passive_compartments

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

        return sum(val.sum() for key, val in self.y.items()
                   if key not in self.passive_compartments)


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
        self.slice = 1

    def __call__(self, state):
        if self.slice == self.t.shape[0]:
            self.add_chunk()

        self.t[self.slice] = state.t
        for key in state.compartments:
            self.y[key][self.slice] = state.__getattr__(key)
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
        for key in self.y.keys():
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
        for compartment in self.compartments:
            text += "  - {0:s} {1:s}\n".format(compartment,
                                               str(self.y[compartment].shape))
        return text[:-1]


_default_quantiles = (0.0455, 0.3173, 0.5, 0.6827, 0.9545)


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

    def __init__(self, chunk_length=1000, quantiles=_default_quantiles):
        self.quantiles = quantiles
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
            self.quantile_data[key] = np.array(
                [np.quantile(self.y_samples[key], quantile, axis=1)
                 for quantile in self.quantiles]
            )

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

    :arg reactions: A :class:`list` of :class:`Reaction`'s
        (or subclasses thereof) used to specify the dynamics of the
        compartmental model.

    .. automethod:: __call__

    .. attribute:: compartments

        The compartment names comprising the simulation state, inferred as the set of
        all :attr:`Reaction.lhs`'s and :attr:`Reaction.rhs`'s from the input list
        of :class:`Reaction`'s.
    """

    def __init__(self, reactions):
        def flatten(items):
            for i in items:
                if isinstance(i, (list, tuple)):
                    for j in flatten(i):
                        yield j
                else:
                    yield i

        rhs_keys = []
        lhs_keys = []
        passive_compartments = []
        for reaction in reactions:
            from pydemic.reactions import PassiveReaction
            if not isinstance(reaction, PassiveReaction):
                lhs_keys.append(reaction.lhs)
                rhs_keys.append(reaction.rhs)
            else:
                passive_compartments.extend([reaction.lhs, reaction.rhs])

        lhs_keys = set(flatten(lhs_keys))
        rhs_keys = set(flatten(rhs_keys))
        self.compartments = list((lhs_keys | rhs_keys) - set([None]))
        self.passive_compartments = list(set(passive_compartments) - set([None]))
        self.evolved_compartments = self.compartments + self.passive_compartments

        self._network = tuple(react for reaction in reactions
                              for react in reaction.get_reactions())

        all_lhs = set(x.lhs for x in self._network) - set([None])
        all_rhs = set(x.rhs for x in self._network) - set([None])
        self.hidden_compartments = list((all_lhs | all_rhs) - set(self.compartments))

    def print_network(self):
        for reaction in self._network:
            print(reaction)

    def step_gillespie_direct(self, time, state, dt):
        increments = {}

        # FIXME: fix this for split reactions
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

        lhs, rhs = reactions[full_index[0]]  # pylint: disable=E1126
        state.y[lhs][full_index[1:]] -= 1.
        state.y[rhs][full_index[1:]] += 1.

        # FIXME: fix this for split reactions
        if state.y[lhs][full_index[1:]] < 0:
            state.y[lhs][full_index[1:]] += 1.
            state.y[rhs][full_index[1:]] -= 1.

        return dt

    def step(self, time, state, dt, stochastic_method=None):
        increments = {}

        for reaction in self._network:
            from pydemic.reactions import PassiveReaction
            if not isinstance(reaction, PassiveReaction):  # FIXME
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

        state = SimulationState(time, compartment_vals, hidden_compartment_vals,
                                self.passive_compartments)
        return state

    def __call__(self, tspan, y0, dt, stochastic_method=None, samples=1, seed=None,
                 logger=None):
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

    def step_deterministic(self, t, y):
        dy = np.zeros_like(y)
        state = self.array_to_state(t, y)
        dy_state = self.array_to_state(t, dy)

        for reaction in self._network:
            rate = reaction.evaluator(t, state)
            if type(reaction.rhs) == tuple:
                for rhs in reaction.rhs:
                    dy_state.y[rhs] += rate
            else:
                dy_state.y[reaction.rhs] += rate
            if reaction.lhs is not None:
                dy_state.y[reaction.lhs] -= rate

        return self.state_to_array(dy_state)

    def array_to_state(self, time, array):
        n_evolved = len(self.evolved_compartments)
        array = array.reshape(n_evolved, *self.compartment_shape)
        y = {comp: array[i] for i, comp in enumerate(self.evolved_compartments)}
        return SimulationState(time, y, {}, self.passive_compartments)

    def state_to_array(self, state):
        array = np.empty((len(self.evolved_compartments),)+self.compartment_shape)
        for i, comp in enumerate(self.evolved_compartments):
            array[i] = state.y[comp]

        return array.reshape(-1)

    def solve_deterministic(self, t_span, y0, rtol=1e-6):
        """
        :arg tspan: A :class:`tuple` specifying the initiala and final times.

        :arg y0: A :class:`dict` with the initial values
            (as :class:`numpy.ndarray`'s) for each of :attr:`compartments`.
        """

        template = y0[self.compartments[0]]
        self.compartment_shape = template.shape

        state = SimulationState(t_span[0], y0, {}, self.passive_compartments)
        y0_array = self.state_to_array(state)

        from scipy.integrate import solve_ivp
        result = solve_ivp(self.step_deterministic, t_span, y0_array,
                           first_step=.1, dense_output=True, rtol=rtol, atol=1e-20,
                           method='DOP853')

        return result

    def dense_to_logger(self, solve_ivp_result, times):
        lower = (1 - 1e-10) * solve_ivp_result.t[0]
        upper = (1 + 1e-10) * solve_ivp_result.t[-1]
        all_within_t = all(lower <= t <= upper for t in times)
        if not all_within_t:
            raise ValueError(
                'Extrapolation outside of simulation timespan not allowed.'
            )

        logger = StateLogger()
        shape = (len(self.evolved_compartments),)+self.compartment_shape

        def get_state_at_t(t):
            array = solve_ivp_result.sol(t).reshape(*shape)
            comps = {comp: array[i]
                     for i, comp in enumerate(self.evolved_compartments)}
            return SimulationState(t, comps, {}, self.passive_compartments)

        logger.initialize_with_state(get_state_at_t(times[0]))

        for t in times[1:]:
            logger(get_state_at_t(t))

        logger.cleanup()
        return logger
