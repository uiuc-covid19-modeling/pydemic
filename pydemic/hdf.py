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
import emcee


class HDFBackend(emcee.backends.HDFBackend):
    def __init__(self, filename, fit_parameters=None, fixed_values=None, data=None,
                 **kwargs):
        super().__init__(filename, **kwargs)

        with self.open('a') as f:
            if fixed_values is not None:
                f.create_group('fixed_values')
                for key, value in fixed_values.items():
                    f['fixed_values'][key] = value

            if fit_parameters is not None:
                f.create_group('fit_parameters')
                f['fit_parameters/names'] = np.array(
                    [par.name for par in fit_parameters], dtype="S10"
                )
                f['fit_parameters/guess'] = [par.guess for par in fit_parameters]
                f['fit_parameters/bounds'] = [par.bounds for par in fit_parameters]
                f['fit_parameters/uncertainty'] = [
                    par.uncertainty for par in fit_parameters
                ]

            if data is not None:
                f.create_group('data')
                f['data/t'] = data.t
                f.create_group('data/y')
                for key, val in data.y.items():
                    if np.array(val).dtype.char in ('S', 'U'):
                        f['data/y'][key] = np.array(val, dtype="S10")
                    else:
                        f['data/y'][key] = val

    @property
    def fixed_values(self):
        with self.open() as f:
            return {key: val[()] for key, val in f['fixed_values'].items()}

    @property
    def fit_parameters(self):
        from pydemic.sampling import SampleParameter
        with self.open() as f:
            names = f['fit_parameters/names'][:]
            pars = []
            for i, name in enumerate(names):
                pars.append(
                    SampleParameter(name,
                                    f['fit_parameters/bounds'][i],
                                    f['fit_parameters/guess'][i],
                                    f['fit_parameters/uncertainty'][i])
                )
            return pars

    @property
    def data(self):
        from pydemic.data import CaseData
        with self.open() as f:
            t = f['data/t'][()]
            y = {key: val[()] for key, val in f['data/y'].items()}
            return CaseData(t=t, y=y)
