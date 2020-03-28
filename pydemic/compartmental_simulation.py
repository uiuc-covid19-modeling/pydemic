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
from pydemic import AttrDict, ErlangProcess, Reaction

__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: Simulation
.. currentmodule:: pydemic.simulation
"""


class SimulationState(AttrDict):


    expected_kwargs = {
    }

    def get_total_population(self):
        # FIXME: does this work?
        return sum(self[key] for key in self.expected_kwargs)

    def copy(self):
        input_vals = {}
        for key in self.expected_kwargs:
            if isinstance(self[key], np.ndarray):
                input_vals[key] = self[key].copy()
            else:
                input_vals[key] = self[key]
        return SimulationState(**input_vals)

    def __repr__(self):
        string = ""
        for key in self.expected_kwargs:
            string += key + '\t' + str(self[key]) + '\n'
        return string


class SimulationResult(AttrDict):
    def __init__(self, initial_state, n_time_steps):
        input_vals = {}
        for key in initial_state.expected_kwargs:
            val = initial_state[key]
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            ary = np.zeros(shape=(n_time_steps,)+val.shape)
            ary[0] = val
            input_vals[key] = ary

        super().__init__(**input_vals)
        self.slice = 0

    def extend(self, population):
        self.slice += 1
        for key in population.expected_kwargs:
            self[key][self.slice] = population[key]



class CompartmentalModelSimulation:
    def __init__(self, reactions):
        lhs_keys = set(x.lhs for x in reactions)
        rhs_keys = set(x.rhs for x in reactions)
        self.compartments = lhs_keys & rhs_keys

        self._network = tuple(react for reaction in reactions
                              for react in reaction.get_reactions())

    def print_network(self):
        for reaction in self._network:
            print(reaction)

    def step(self, time, state, dt):
        increments = {}

        for reaction in self._network:
            reaction_rate = reaction.evaluator(time, state)
            dY = (dt * reaction_rate)

            dY_min = state[reaction.lhs].copy()
            for (_lhs, _rhs), incr in increments.items():
                if reaction.lhs == _lhs:
                    dY_min -= incr
            dY = np.minimum(dY_min, dY)

            increments[(reaction.lhs, reaction.rhs)] = dY

        for (lhs, rhs), dY in increments.items():
            state[lhs] -= dY
            state[rhs] += dY

    def __call__(self, tspan, y0, sampler):
        """
        :arg start_time: The initial time, in miliseconds from January 1st, 1970.

        :arg end_time: The final time, in miliseconds from January 1st, 1970.

        :arg sample: The sampling function.

        :returns: The :class:`SimulationResult`.
        """

        start_time, end_time = tspan
        state = y0

        time = start_time
        while time < end_time:
            dt = 1. # FIXME: get dt in a reasonable way
            self.step(time, state, dt)
            time += dt


        """
        n_time_steps = int(np.ceil((end_time - start_time) / self.dt)) + 1
        state = self.initialize_population(start_time)
        result = SimulationResult(state, n_time_steps)

        time = start_time
        while time < end_time:
            state = self.step(time, state, sample)
            result.extend(state)
            time += self.dt

        return result
        """




