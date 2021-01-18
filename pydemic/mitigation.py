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
from scipy.interpolate import PchipInterpolator

__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: MitigationModel
"""


class MitigationModel(PchipInterpolator):
    """
    An interface for creating (smooth, monotonic) piecewise linear functions.
    Subclasses :class:`scipy.interpolate.PchipInterpolator`.

    Constructs the interpolating function which takes the constant values
    ``factors[0]`` between ``t0`` and ``t[0]`` and ``factors[-1]`` between
    ``t[-1]`` and ``tf``.

    :arg t0: A :class:`float` representing the first input value for
        interpolation.

    :arg tf: A :class:`float` representing the last input value for
        interpolation.

    :arg t: A :class:`numpy.ndarray` of interpolating nodes
        (between ``t0`` and ``tf``).

    :arg factors: A :class:`numpy.ndarray` of function values to interpolate to
        at the nodes ``t``.

    .. automethod:: init_from_kwargs
    """

    def __init__(self, t0, tf, t, factors, multiplier=None):
        self.init_args = (t0, tf, t, factors)

        self.times = t
        self.factors = factors
        if len(t) > 0:
            t = np.insert(t, 0, min(t0, t[0]) - 10)
            t = np.append(t, max(tf, t[-1]) + 10)
        if len(factors) > 0:
            factors = np.insert(factors, 0, factors[0])
            factors = np.append(factors, factors[-1])
        else:
            t = np.array([t0 - 10, tf + 10])
            factors = np.array([1, 1])

        self.multiplier = multiplier
        super().__init__(t, factors)

    @classmethod
    def init_from_kwargs(cls, t0, tf, prefix="mitigation", **kwargs):
        """
        A convenience constructor which collects values for ``t`` based on (sorted)
        keyword arguments beginning with ``mitigation_t`` with ``factors``
        from those beginning with ``mitigation_factor``.
        """

        factor_keys = sorted(
            (name for name in kwargs.keys() if name.startswith(f"{prefix}_factor")),
            key=lambda name: int(name.strip(f"{prefix}_factor"))
        )
        factors = np.array([kwargs.pop(key) for key in factor_keys])

        time_keys = sorted(
            (name for name in kwargs.keys() if name.startswith(f"{prefix}_t")),
            key=lambda name: int(name.strip(f"{prefix}_t"))
        )
        times = np.array([kwargs.pop(key) for key in time_keys])

        return cls(t0, tf, times, factors)

    def __mul__(self, other):
        if self.multiplier is not None:
            multiplier = other * self.multiplier
        else:
            multiplier = other
        return MitigationModel(*self.init_args, multiplier=multiplier)

    __rmul__ = __mul__

    def __call__(self, x, **kwargs):
        res = super().__call__(x, **kwargs)
        if self.multiplier is not None:
            res *= self.multiplier(x, **kwargs)
        return res
