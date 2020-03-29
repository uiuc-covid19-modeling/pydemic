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
.. currentmodule:: pydemic.simulation
"""


class SimulationState:
    def __init__(self, t, compartments, hidden_compartments):
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
        return sum(val.sum() for val in self.y.values())


class StateLogger:
    def __init__(self, state, chunk_length=1000):
        self.chunk_length = chunk_length
        self.t = np.zeros(shape=(chunk_length,))
        self.t[0] = state.t
        self.slice = 0

        self.y = {}
        for key in state.compartments:
            val = state.y[key]
            ary = np.zeros(shape=(chunk_length,)+val.shape)
            ary[0] = val
            self.y[key] = ary

    def extend(self, state):
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

    def trim(self):
        self.t = self.t[:self.slice]
        for key in self.y.keys():
            self.y[key] = self.y[key][:self.slice, ...]


class CompartmentalModelSimulation:
    def __init__(self, reactions):
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

    def step(self, time, state, dt, stochastic_method=None):
        increments = {}

        for reaction in self._network:
            reaction_rate = reaction.evaluator(time, state)
            dY = dt * reaction_rate
            if stochastic_method == "tau_leap":
                dY = poisson.rvs(dY)

            dY_min = state.y[reaction.lhs].copy()
            for (_lhs, _rhs), incr in increments.items():
                if reaction.lhs == _lhs:
                    dY_min -= incr
            dY = np.minimum(dY_min, dY)

            if (reaction.lhs, reaction.rhs) in increments:
                increments[reaction.lhs, reaction.rhs] += dY
            else:
                increments[reaction.lhs, reaction.rhs] = dY

        if stochastic_method == "direct":
            # WARNING: need to be sure we're pulling from the right
            # reaction here! I might have solved an XY problem ...
            reactions = list(increments.keys())
            r1, r2 = np.random.random(2)
            cumulative_rates = np.cumsum([increments[k] for k in reactions])
            dt = - np.log(r1) / cumulative_rates[-1]
            r2 *= cumulative_rates[-1]
            reaction_index = np.searchsorted(cumulative_rates, r2)
            # WARNING: It's also not entirely clear that this produces
            # the right rate distributions for processes that have two
            # different lhs <--> rhs reactions ...
            lhs,rhs = reactions[reaction_index]
            state.y[lhs] -= 1.
            state.y[rhs] += 1.
        else:
            for (lhs, rhs), dY in increments.items():
                state.y[lhs] -= dY
                state.y[rhs] += dY

        return dt

    def initialize_full_state(self, time, y0):
        compartment_vals = {}
        for key, ary in y0.items():
            compartment_vals[key] = np.array(ary, dtype='float64')

        hidden_compartment_vals = {}
        template = compartment_vals[self.compartments[0]]
        for key in self.hidden_compartments:
            hidden_compartment_vals[key] = np.zeros_like(template)

        state = SimulationState(time, compartment_vals, hidden_compartment_vals)
        return state

    def __call__(self, tspan, y0, sampler, dt=.01, stochastic_method=None):
        """
        :arg tspan: A :class:`tuple` specifying the initiala and final times
            in miliseconds from January 1st, 1970.

        :arg y0: A :class:`dict` with the initial values
            (as :class:`numpy.ndarray`'s) for each of :attr:`compartments`.

        :returns: The :class:`SimulationResult`.
        """

        start_time, end_time = tspan
        state = self.initialize_full_state(start_time, y0)

        result = StateLogger(state)

        time = start_time
        while time < end_time:
            # dt = 1. # FIXME: get dt in a reasonable way
            dt = self.step(time, state, dt, stochastic_method=stochastic_method)
            time += dt
            state.t = time
            result.extend(state)

        result.trim()
        return result
