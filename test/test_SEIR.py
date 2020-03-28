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
from pydemic import Reaction, GammaProcess, CompartmentalModelSimulation


def test_SEIR(total_pop=1e6, beta=12, a=1, gamma=1, plot=False):
    reactions = (
        Reaction('susceptible', 'exposed',
                    lambda t, y: y['susceptible']*y['infectious']*beta/total_pop),
        Reaction('exposed', 'infectious',
                    lambda t, y: a * y['exposed']),
        Reaction('infectious', 'recovered',
                    lambda t, y: gamma * y['infectious']),
    )

    simulation = CompartmentalModelSimulation(reactions)

    y0 = {
        'susceptible': np.array(total_pop-1),
        'exposed': np.array(1),
        'infectious': np.array(0),
        'recovered': np.array(0),
    }

    compartments = ('susceptible', 'exposed', 'infectious', 'recovered')

    tspan = (0, 10)
    dt = 1e-3

    result = simulation(tspan, y0, lambda x: x, dt=dt)

    def f(t, y):
        S, E, I, R = y
        dydt = [-beta*S*I/total_pop, beta*S*I/total_pop - a*E, a*E-gamma*I, gamma*I]
        return np.array(dydt)

    initial_position = [total_pop-1, 1, 0, 0]
    from scipy.integrate import solve_ivp
    res = solve_ivp(f, tspan, initial_position, rtol=1.e-13,
                    dense_output=True)

    t = result['time']

    scipy_sol = {comp: res.sol(t)[i] for i, comp in enumerate(compartments)}
    for i, name in enumerate(compartments):
        non_zero = np.logical_and(scipy_sol[name] > 0, result[name] > 0)
        test = np.logical_and(non_zero, t > 1)
        relerr = np.abs(1 - scipy_sol[name][test] / result[name][test])
        print('max err for', name, 'is', np.max(relerr))
        assert np.max(relerr) < .05

    total_people = sum(result[name] for name in compartments)
    total_err = np.max(np.abs(1 - total_people / total_pop))
    print('total error is', np.max(total_err))
    assert np.max(total_err) < 1.e-13

    if plot:
        import matplotlib as mpl
        mpl.use('agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, name in enumerate(compartments):
            ax.semilogy(t, scipy_sol[name], linewidth=.5, label=name)
            ax.semilogy(t, result[name], '--', label=name + ', pydemic')

        ax.legend(loc='center left', bbox_to_anchor=(1, .5))
        fig.savefig('SEIR_example.png', bbox_inches='tight')

if __name__ == "__main__":
    test_SEIR(plot=True)
