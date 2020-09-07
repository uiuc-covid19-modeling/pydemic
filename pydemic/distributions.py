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
    """
    Base class for distributions.

    .. automethod:: convolve_pdf

    .. automethod:: convolve_survival
    """

    def pdf(self, t):
        raise NotImplementedError

    def cdf(self, t):
        raise NotImplementedError

    def convolve_pdf(self, t, influx, prefactor=1, profile=None, method='fft',
                     complement=False):
        """
        Convolves an array ``influx`` with the PDF of the distribution.

        :arg t: The times of evaluation.

        :arg influx: The array to be convolved, :math:`y`.

        :arg prefactor: A multiplicative prefactor, :math:`P`.

        :arg method: A :class:`str` specifying whether to convolve by
            Fast Fourier Transform (``'fft'``) or via direct covolution
            (``'direct'``).
        """

        ones = np.ones_like(influx)
        pdf = (self.pdf(t[:] - t[0]) * ones.T).T
        prefactor = prefactor * ones

        if profile is not None:
            prof = profile(t)
        else:
            prof = np.ones_like(t)

        prefactor = prefactor.T * prof

        if complement:
            prefactor = 1 - prefactor

        influx = influx * prefactor.T

        end = t.shape[0]

        if method == 'fft':
            from scipy.signal import fftconvolve
            result = fftconvolve(pdf, influx, mode='full', axes=0)[:end]
        elif method == 'direct':
            result = np.zeros_like(influx)
            for i in range(end):
                result[i, ...] = np.einsum('i...,i...', influx[i::-1], pdf[:i+1])

        return result

    def convolve_survival(self, t, influx, prefactor=1, method='fft'):
        """
        Convolves an array ``influx`` with the survival function of the distribution,

        .. math::

            S(x) = 1 - \\int \\mathrm{d} x' \\, f(x').

        :arg t: The times of evaluation.

        :arg influx: The array to be convolved.

        :arg prefactor: A multiplicative prefactor.

        :arg method: A :class:`str` specifying whether to convolve by
            Fast Fourier Transform (``'fft'``) or via direct covolution
            (``'direct'``).
        """

        survival = 1 - self.cdf(t - t[0])

        prefactor = prefactor * np.ones_like(influx[0, ...])
        kernel = np.outer(survival, prefactor)

        end = t.shape[0]
        from scipy.signal import fftconvolve
        result = fftconvolve(kernel, influx, mode='full', axes=0)[:end]

        return result


class GammaDistribution(DistributionBase):
    """
    Implements functionality for the gamma distribution, with PDF

    .. math::

        f(x)
        = \\frac{1}{\\Gamma(k) \\theta^{k}} x^{k-1} e^{- x / \\theta }

    One can specify the distribution by its mean :math:`\\mu`
    and standard deviation :math:`\\sigma`
    or by the standard shape and scale parameters :math:`k` and :math:`\\theta`,
    which are related by

    .. math::

        k = \\frac{\\mu^2}{\\sigma^2}

        \\theta = \\frac{\\sigma^2}{\\mu}.

    :arg mean: The mean, :math:`\\mu`.

    :arg std: The standard derviation, :math:`\\sigma`.

    :arg shape: The shape, :math:`k`.

    :arg scale: The scale, :math:`\\theta`.

    Passed values for ``mean`` and ``std`` take precendece over those for
    ``shape`` and ``scale``.
    """

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
            return np.diff(cdf, prepend=0)
        else:
            from scipy.stats import gamma
            return gamma.pdf(t, self.shape, scale=self.scale)

    def cdf(self, t):
        from scipy.stats import gamma
        return gamma.cdf(t, self.shape, scale=self.scale)

    def __repr__(self):
        return f"<GammaDistribution(shape={self.shape:.3g}, scale={self.scale:.3g})>"
