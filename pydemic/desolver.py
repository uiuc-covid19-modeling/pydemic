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

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from emcee.pbar import get_progress_bar

__doc__ = """
.. currentmodule:: pydemic.desolver
.. autofunction:: differential_evolution
.. currentmodule:: pydemic
"""


class PicklingDifferentialEvolutionSolver(DifferentialEvolutionSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_nit = 0
        self.backend = None
        self.pbar = None
        self.workers = kwargs.get("workers", 1)

    def __next__(self):
        result = super().__next__()
        self.cumulative_nit += 1
        if self.pbar is not None:
            self.pbar.update(1)

        if self.backend is not None:
            self.backend.save_optimizer(self)

        return result

    def solve(self, backend=None, progress=None):
        self.backend = backend
        with get_progress_bar(progress, self.maxiter) as self.pbar:
            result = super().solve()
        result.nit = self.cumulative_nit
        self.backend = None
        return result

    next = __next__

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["pbar"]
        del state["backend"]
        del state["_mapwrapper"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from scipy._lib._util import MapWrapper
        self._mapwrapper = MapWrapper(self.workers)
        self.pbar = None
        self.backend = None


def differential_evolution(*args, progress=True, backend=None,
                           maxiter=1000, **kwargs):
    """
    A wrapper to :func:`scipy.optimize.differential_evolution` which optionally
    implements checkpointing and restarting (by pickling) as well as a progress bar.

    :arg filename: The filename (including extension) for pickling.
        If the file already exists, optimization is continued from the pickled object
        for ``maxiter`` steps (and checkpointing will continue overwriting the same
        file).
        If the file does not exist, a new solver instance is created and checkpointed
        to the specified file.
        Defaults to *None*, in which case checkpointing is not implemented.

    :arg progress: Whether to display a progress bar (if a :class:`bool`) or the type
        of progress bar to display (if a :class:`str`).

    All positional and (other) keyword arguments are as specified to
    :class:`scipy.optimize.differential_evolution.
    """

    if backend is not None:
        if backend.initialized:
            with backend.load_optimizer() as solver:
                solver.maxiter = maxiter
                if 'tol' in kwargs:
                    solver.tol = kwargs['tol']
                ret = solver.solve(backend=backend, progress=progress)
                return ret

    with PicklingDifferentialEvolutionSolver(*args, maxiter=maxiter,
                                             **kwargs) as solver:
        ret = solver.solve(backend=backend, progress=progress)

    return ret
