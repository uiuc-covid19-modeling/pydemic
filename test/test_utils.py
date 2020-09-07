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
from pydemic import MitigationModel, GammaDistribution


def test_gamma():
    t = np.linspace(0, 20, 100)
    dist = GammaDistribution(4, 2)
    influx = np.exp(-(t-10)**2 / 4**2) * np.ones((3, t.size))
    influx = influx.T

    prefactor = .23
    res1 = dist.convolve_pdf(t, influx, prefactor=prefactor, method='fft')[20:]
    res2 = dist.convolve_pdf(t, influx, prefactor=prefactor, method='direct')[20:]

    err = np.max(np.abs(res1 - res2)) / np.max(np.abs(res1))
    assert err < 1e-12, err

    prefactor = np.random.rand(3)
    res1 = dist.convolve_pdf(t, influx, prefactor=prefactor, method='fft')[20:]
    res2 = dist.convolve_pdf(t, influx, prefactor=prefactor, method='direct')[20:]

    err = np.max(np.abs(res1 - res2)) / np.max(np.abs(res1))
    assert err < 1e-12, err

    profile = MitigationModel(0, 20, [5, 15], [1, .5])
    res1 = dist.convolve_pdf(t, influx, prefactor=prefactor, profile=profile,
                             method='fft')[20:]
    res2 = dist.convolve_pdf(t, influx, prefactor=prefactor, profile=profile,
                             method='direct')[20:]

    err = np.max(np.abs(res1 - res2)) / np.max(np.abs(res1))
    assert err < 1e-12, err


def test_mitigation():
    t = np.linspace(0, 20, 100)
    mm1 = MitigationModel(0, 1, [1/3, 2/3], [1, .5])
    mm2 = MitigationModel(0, 1, [1/4, 3/4], [1, .2])

    err = mm1(t) * mm2(t) - (mm1 * mm2)(t)
    assert err.all() < 1e-14, err

    err = mm2(t) * mm1(t) * mm2(t) - (mm2 * mm1 * mm2)(t)
    assert err.all() < 1e-14, err


if __name__ == "__main__":
    test_gamma()
    test_mitigation()
