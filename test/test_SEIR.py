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
              removal_rate=1):
    sim = SEIRModelSimulation(avg_infection_rate, infectious_rate, removal_rate)

    compartments = ('susceptible', 'exposed', 'infectious', 'removed')
    y0 = {
        'susceptible': np.array(total_pop-1),
        'exposed': np.array(1),
        'infectious': np.array(0),
        'removed': np.array(0),
    }

    tspan = (0, 10)
    dt = 1e-3
    result = sim(tspan, y0, dt)
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
    res = solve_ivp(f, tspan, initial_position, rtol=1.e-13, method='DOP853',
                    dense_output=True)

    true_sol = {comp: res.sol(t)[i] for i, comp in enumerate(compartments)}
    for i, name in enumerate(compartments):
        non_zero = np.logical_and(true_sol[name] > 0, result.y[name] > 0)
        test = np.logical_and(non_zero, t > 1)
        relerr = np.abs(1 - true_sol[name][test] / result.y[name][test])
        print('max err for', name, 'is', np.max(relerr))
        assert np.max(relerr) < .05

    total_people = sum(result.y[name] for name in compartments)
    total_err = np.max(np.abs(1 - total_people / total_pop))
    print('total error is', np.max(total_err))
    assert np.max(total_err) < 1.e-13

    tspan = (0, 10.1)
    scipy_res = sim.solve_deterministic(tspan, y0)
    scipy_res = sim.dense_to_logger(scipy_res, t)

    for i, name in enumerate(compartments):
        non_zero = np.logical_and(true_sol[name] > 0, scipy_res.y[name] > 0)
        test = np.logical_and(non_zero, t > 1)
        relerr = np.abs(1 - true_sol[name][test] / scipy_res.y[name][test])
        print('max err for', name, 'is', np.max(relerr))
        assert np.max(relerr) < .05

    total_people = sum(scipy_res.y[name] for name in compartments)
    total_err = np.max(np.abs(1 - total_people / total_pop))
    print('total error is', np.max(total_err))
    assert np.max(total_err) < 1.e-13


if __name__ == "__main__":
    test_SEIR(1e6, 12)
