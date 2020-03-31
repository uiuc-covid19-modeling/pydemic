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


def l2_log_norm(a, b):
    return np.sum(np.power(np.log(a)-np.log(b), 2.))


class LikelihoodEstimatorBase:
    def __init__(self, fit_priors, fixed_parameters, data, norm=None):
        self.fit_priors = fit_priors
        self.fixed_parameters = fixed_parameters
        self.data = data

        if norm is None:
            self.norm = l2_log_norm
        else:
            self.norm = norm

    def __call__(self, theta):
        raise NotImplementedError

    def get_model_result(self, **kwargs):
        raise NotImplementedError


from pydemic.models.seir import SEIRModelSimulation
from pydemic.models.neher import NeherModelSimulation


__all__ = [
    "SEIRModelSimulation",
    "NeherModelSimulation",
]
