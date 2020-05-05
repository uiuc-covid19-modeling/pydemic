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

__doc__ = """
.. currentmodule:: pydemic.distributions
.. autoclass:: DistributionBase
.. currentmodule:: pydemic
.. autoclass:: GammaDistribution
"""


class DistributionBase:
    def pdf(self, t):
        raise NotImplementedError

    def cdf(self, t):
        raise NotImplementedError

    def convolve_pdf(self, t, influx, prefactor=1, method='fft'):
        pdf = self.pdf(t[:] - t[0])
        prefactor = prefactor * np.ones_like(influx[0, ...])

        end = t.shape[0]
        if method == 'fft':
            kernel = np.outer(pdf, prefactor)
            from scipy.signal import fftconvolve
            result = fftconvolve(kernel, influx, mode='full', axes=0)[:end]
        elif method == 'direct':
            result = np.zeros_like(influx)
            for i in range(1, end):
                result[i, ...] = prefactor * np.dot(influx[i-1::-1].T, pdf[:i])

        return result

    def convolve_survival(self, t, influx, prefactor=1, method='fft'):
        survival = 1 - self.cdf(t - t[0])

        prefactor = prefactor * np.ones_like(influx[0, ...])
        kernel = np.outer(survival, prefactor)

        end = t.shape[0]
        from scipy.signal import fftconvolve
        result = fftconvolve(kernel, influx, mode='full', axes=0)[:end]

        return result


class GammaDistribution(DistributionBase):
    def __init__(self, mean=None, std=None, shape=None, scale=None):
        if shape is None:
            self.shape = mean**2 / std**2
        else:
            self.shape = shape

        if scale is None:
            self.scale = std**2 / mean
        else:
            self.scale = scale

    def pdf(self, t, method='diff'):
        if method == 'diff':
            cdf = self.cdf(t)
            # FIXME: prepend or append?
            return np.diff(cdf, prepend=0)
        else:
            from scipy.stats import gamma
            return gamma.pdf(t, self.shape, scale=self.scale)

    def cdf(self, t):
        from scipy.stats import gamma
        return gamma.cdf(t, self.shape, scale=self.scale)
