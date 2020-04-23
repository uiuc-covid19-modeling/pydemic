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


class MitigationModel(PchipInterpolator):
    def __init__(self, t0, tf, t, factors):
        self.times = t
        self.factors = factors
        t = np.insert(t, 0, min(t0, t[0]) - 10)
        t = np.append(t, max(tf, t[-1]) + 10)

        factors = np.insert(factors, 0, factors[0])
        factors = np.append(factors, factors[-1])
        super().__init__(t, factors)

    @classmethod
    def init_from_kwargs(cls, t0, tf, **kwargs):
        factor_keys = sorted([key for key in kwargs.keys()
                              if key.startswith('mitigation_factor')])
        factors = np.array([kwargs.pop(key) for key in factor_keys])

        time_keys = sorted([key for key in kwargs.keys()
                            if key.startswith('mitigation_t')])
        times = np.array([kwargs.pop(key) for key in time_keys])

        return cls(t0, tf, times, factors)
