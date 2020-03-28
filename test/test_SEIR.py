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


def test_SEIR(N=1e6, beta=12, a=1, gamma=1, plot=False):
    reactions = (
        Reaction('susceptible', 'exposed',
                    lambda t, y: y['susceptible']*y['infectious']*beta/N),
        Reaction('exposed', 'infectious',
                    lambda t, y: a * y['exposed']),
        Reaction('infectious', 'recovered',
                    lambda t, y: gamma * y['infectious']),
    )

    simulation = CompartmentalModelSimulation(reactions)

    y0 = {
        'susceptible': np.array(N-1),
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
        dydt = [-beta*S*I/N, beta*S*I/N - a*E, a*E-gamma*I, gamma*I]
        return np.array(dydt)

    initial_position = [N-1, 1, 0, 0]
    from scipy.integrate import solve_ivp
    res = solve_ivp(f, tspan, initial_position, rtol=1.e-13,
                    dense_output=True)

    t = np.linspace(0, 10, int(10/dt + 2))

    scipy_sol = {comp: res.sol(t)[i] for i, comp in enumerate(compartments)}
    for i, name in enumerate(compartments):
        non_zero = np.logical_and(scipy_sol[name] > 0, result[name] > 0)
        relerr = np.abs(1 - scipy_sol[name][non_zero] / result[name][non_zero])
        print(name, relerr, np.max(relerr))

    print(result['recovered'])
    print(scipy_sol['recovered'])

    if plot:
        import matplotlib as mpl ; mpl.use('agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i, name in enumerate(compartments):
            ax.semilogy(t, scipy_sol[name], linewidth=.5, label=name)
            ax.semilogy(t, result[name], '--', label=name + ', pydemic')

        ax.legend(loc='center left', bbox_to_anchor=(1, .5))
        fig.savefig('SEIR_example.png', bbox_inches='tight')

if __name__ == "__main__":
    test_SEIR(plot=True)