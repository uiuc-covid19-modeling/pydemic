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
import pandas as pd
import emcee


class HDFBackend(emcee.backends.HDFBackend):
    def __init__(self, filename, fit_parameters=None, fixed_values=None, data=None,
                 **kwargs):
        super().__init__(filename, **kwargs)
        import h5py
        string_dt = h5py.string_dtype(encoding='ascii')

        with self.open('a') as f:
            if fixed_values is not None:
                f.create_group('fixed_values')
                for key, value in fixed_values.items():
                    f['fixed_values'][key] = value

            if fit_parameters is not None:
                def nanify(x):
                    return x if x is not None else np.nan

                f.create_group('fit_parameters')
                f['fit_parameters/names'] = np.array(
                    [par.name for par in fit_parameters], dtype=string_dt
                )
                f['fit_parameters/bounds'] = [par.bounds for par in fit_parameters]
                f['fit_parameters/guess'] = [nanify(par.guess)
                                             for par in fit_parameters]
                f['fit_parameters/uncertainty'] = [nanify(par.uncertainty)
                                                   for par in fit_parameters]
                f['fit_parameters/sigma'] = [nanify(par.sigma)
                                             for par in fit_parameters]

            from pydemic.data import CaseData
            if isinstance(data, CaseData):
                f.create_group('data')
                f['data/t'] = data.t
                f.create_group('data/y')
                for key, val in data.y.items():
                    if np.array(val).dtype.char in ('S', 'U'):
                        f['data/y'][key] = np.array(val, dtype=string_dt)
                    elif np.array(val).dtype.char != 'O':
                        f['data/y'][key] = val
            elif isinstance(data, pd.DataFrame):
                data.to_hdf(f.filename, 'df_data')

    @property
    def fixed_values(self):
        with self.open() as f:
            return {key: val[()] for key, val in f['fixed_values'].items()}

    @property
    def fit_parameters(self):
        from pydemic.sampling import SampleParameter
        with self.open() as f:
            def denanify(x):
                return x if np.isfinite(x) else None

            names = [name.decode('utf-8') for name in f['fit_parameters/names'][:]]
            pars = []
            for i, name in enumerate(names):
                args = [f['fit_parameters/bounds'][i]]
                args += [denanify(f['fit_parameters'][key][i])
                         for key in ('guess', 'uncertainty', 'sigma')]
                pars.append(SampleParameter(name, *args))
            return pars

    @property
    def data(self):
        with self.open() as f:
            if 'data' in f.keys():
                from pydemic.data import CaseData
                t = f['data/t'][()]
                y = {key: val[()] for key, val in f['data/y'].items()}
                return CaseData(t=t, y=y)
            elif 'df_data' in f.keys():
                return pd.read_hdf(f.filename, key='df_data')
            else:
                return None
