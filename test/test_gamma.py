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
from pydemic import GammaProcess, Simulation


@pytest.mark.parametrize("shape", [1, 2, 8])
@pytest.mark.parametrize("scale", [1, 1.4])
def test_gamma(shape, scale):
    reactions = (
        GammaProcess('a', 'b', shape=shape, scale=lambda t, y: scale),
    )

    simulation = Simulation(reactions)

    y0 = {'a': np.array(1), 'b': np.array(0)}
    tspan = (0, 10)
    dt = 1e-3

    result = simulation(tspan, y0, dt=dt)
    t = result.t

    def f(t, y):
        dy = np.zeros_like(y)
        for i in range(1, len(y)-1):
            dy[i] = - y[i] / scale + y[i-1] / scale
        dy[0] = - y[0] / scale
        dy[-1] = y[-2] / scale
        return dy

    compartments = ('a', 'b')

    y0 = np.zeros(2 + shape-1)
    y0[0] = 1
    from scipy.integrate import solve_ivp
    res = solve_ivp(f, tspan, y0, rtol=1.e-13,
                    dense_output=True)

    scipy_sol = {'a': np.sum(res.sol(t)[:-1], axis=0), 'b': res.sol(t)[-1]}
    for i, name in enumerate(compartments):
        non_zero = np.logical_and(scipy_sol[name] > 0, result.y[name] > 0)
        test = np.logical_and(non_zero, t > 1)
        relerr = np.abs(1 - scipy_sol[name][test] / result.y[name][test])
        print('max err for', name, 'is', np.max(relerr))
        assert np.max(relerr) < .05

    all_keys = simulation.compartments
    total_people = sum(result.y[key] for key in all_keys)
    total_pop = total_people[0]
    total_err = np.max(np.abs(1 - total_people / total_pop))
    print('total error is', np.max(total_err))
    assert np.max(total_err) < 1.e-13


if __name__ == "__main__":
    test_gamma(3, 2.3)
