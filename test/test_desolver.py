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

from pathlib import Path
from pydemic.desolver import differential_evolution


def test_desolver():
    path = Path("__test_desolver.h5")
    if path.exists():
        path.unlink()

    from pydemic.hdf import HDFOptimizationBackend
    backend = HDFOptimizationBackend(path)

    from scipy.optimize import rosen
    bounds = [(0, 2), (0, 2), (0, 2)]
    maxiter = 100

    for i in range(20):
        sol = differential_evolution(
            rosen, bounds=bounds, maxiter=maxiter, progress=False, backend=backend,
        )

        print(sol)
        if sol.success:
            break

        correct_nit = (sol.nit == (i+1) * maxiter)

        if not correct_nit:
            path.unlink()

        assert correct_nit, \
            "number of iterations (%s) should be %s" % (sol.nit, (i+1) * maxiter)

    if not sol.success:
        path.unlink()

    assert sol.success, "solver should have succeeded"
    path.unlink()


if __name__ == "__main__":
    test_desolver()
