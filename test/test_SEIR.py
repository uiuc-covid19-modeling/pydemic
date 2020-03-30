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
import pytest
from pydemic.models import SEIRModelSimulation


@pytest.mark.parametrize("total_pop", [1e4, 1e6])
@pytest.mark.parametrize("avg_infection_rate", [12, 8])
def test_SEIR(total_pop, avg_infection_rate, infectious_rate=1,
              removal_rate=1, plot=False):
    simulation = SEIRModelSimulation(avg_infection_rate, infectious_rate,
                                     removal_rate)

    compartments = ('susceptible', 'exposed', 'infectious', 'removed')
    y0 = {
        'susceptible': np.array(total_pop-1),
        'exposed': np.array(1),
        'infectious': np.array(0),
        'removed': np.array(0),
    }

    tspan = (0, 10)
    dt = 1e-3
    result = simulation(tspan, y0, dt)
    t = result.t

    def f(t, y):
        S, E, I, R = y
        S_to_E = avg_infection_rate * S * I / total_pop
        E_to_I = infectious_rate * E
        I_to_R = removal_rate * I
        dydt = [-S_to_E, S_to_E - E_to_I, E_to_I - I_to_R, I_to_R]
        return np.array(dydt)

    initial_position = [total_pop-1, 1, 0, 0]
    from scipy.integrate import solve_ivp
    res = solve_ivp(f, tspan, initial_position, rtol=1.e-13,
                    dense_output=True)

    scipy_sol = {comp: res.sol(t)[i] for i, comp in enumerate(compartments)}
    for i, name in enumerate(compartments):
        non_zero = np.logical_and(scipy_sol[name] > 0, result.y[name] > 0)
        test = np.logical_and(non_zero, t > 1)
        relerr = np.abs(1 - scipy_sol[name][test] / result.y[name][test])
        print('max err for', name, 'is', np.max(relerr))
        assert np.max(relerr) < .05

    total_people = sum(result.y[name] for name in compartments)
    total_err = np.max(np.abs(1 - total_people / total_pop))
    print('total error is', np.max(total_err))
    assert np.max(total_err) < 1.e-13

    if plot:
        import matplotlib as mpl
        mpl.use('agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, name in enumerate(compartments):
            ax.semilogy(t, scipy_sol[name], linewidth=.5, label=name+', scipy')
            ax.semilogy(t, result.y[name], '--', label=name + ', pydemic')

        ax.legend(loc='center left', bbox_to_anchor=(1, .5))
        fig.savefig('SEIR_example.png', bbox_inches='tight')


@pytest.mark.parametrize("total_pop", [1e4, 1e6])
@pytest.mark.parametrize("avg_infection_rate", [12, 8])
def test_deterministic(total_pop, avg_infection_rate, infectious_rate=1,
                       removal_rate=1, plot=False):
    from pydemic import Reaction
    from pydemic.simulation import DeterministicSimulation

    class SEIRModelSimulation(DeterministicSimulation):
        def __init__(self, avg_infection_rate=10, infectious_rate=5, removal_rate=1):
            self.avg_infection_rate = avg_infection_rate

            reactions = (
                Reaction("susceptible", "exposed",
                        lambda t, y: (self.beta(t, y) * y.susceptible
                                    * y.infectious.sum() / y.sum())),
                Reaction("exposed", "infectious",
                        lambda t, y: infectious_rate * y.exposed),
                Reaction("infectious", "removed",
                        lambda t, y: removal_rate * y.infectious),
            )
            super().__init__(reactions)

        def beta(self, t, y):
            return self.avg_infection_rate

    sim = SEIRModelSimulation(avg_infection_rate, infectious_rate, removal_rate)

    compartments = ('susceptible', 'exposed', 'infectious', 'removed')
    y0 = {
        'susceptible': np.array(total_pop-1),
        'exposed': np.array(1),
        'infectious': np.array(0),
        'removed': np.array(0),
    }

    tspan = (0, 10)
    result = sim(tspan, y0)
    t = result.t
    determ_sol = {comp: result.sol(t)[i] for i, comp in enumerate(sim.compartments)}

    def f(t, y):
        S, E, I, R = y
        S_to_E = avg_infection_rate * S * I / total_pop
        E_to_I = infectious_rate * E
        I_to_R = removal_rate * I
        dydt = [-S_to_E, S_to_E - E_to_I, E_to_I - I_to_R, I_to_R]
        return np.array(dydt)

    initial_position = [total_pop-1, 1, 0, 0]
    from scipy.integrate import solve_ivp
    res = solve_ivp(f, tspan, initial_position, rtol=1.e-13, method='DOP853',
                    dense_output=True)
    scipy_sol = {comp: res.sol(t)[i] for i, comp in enumerate(compartments)}

    for i, name in enumerate(compartments):
        non_zero = np.logical_and(scipy_sol[name] > 0, determ_sol[name] > 0)
        test = np.logical_and(non_zero, t > 1)
        relerr = np.abs(1 - scipy_sol[name][test] / determ_sol[name][test])
        print('max err for', name, 'is', np.max(relerr))
        assert np.max(relerr) < 1.e-6

    total_people = sum(determ_sol[name] for name in compartments)
    total_err = np.max(np.abs(1 - total_people / total_pop))
    print('total error is', np.max(total_err))
    assert np.max(total_err) < 1.e-13


if __name__ == "__main__":
    test_deterministic(1e6, 12, plot=True)
    test_SEIR(1e6, 12, plot=True)
