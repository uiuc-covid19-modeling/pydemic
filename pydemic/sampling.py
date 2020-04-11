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


class SampleParameter:
    def __init__(self, name, bounds, guess, uncertainty):
        self.name = name
        self.bounds = bounds
        self.guess = guess
        self.uncertainty = uncertainty


def l2_log_norm(a, b):
    return -1/2 * np.sum(np.power(np.log(a)-np.log(b), 2.))


def clipped_l2_log_norm(model, data, model_uncert):
    model = np.maximum(model, .1)
    sig = np.log(model_uncert / model)
    sig += 0.05

    top = np.power(np.log(data)-np.log(model), 2.)
    bot = np.power(sig, 2.)

    return - 1/2 * np.sum(top / bot)


def poisson_norm(model, data):
    from scipy.special import gammaln  # pylint: disable=E0611
    return np.sum(- model - gammaln(data) + data * np.log(model))


class LikelihoodEstimatorBase:
    def __init__(self, fit_parameters, fixed_values, data, weights, norm=None,
                 fit_cumulative=False):
        self.fit_parameters = fit_parameters
        self.fit_names = tuple(par.name for par in fit_parameters)
        self.fixed_values = fixed_values
        self.data = data.copy()
        self._original_data = data
        self.fit_cumulative = fit_cumulative
        self.weights = weights

        if self.fit_cumulative and norm is None:
            self.norm = clipped_l2_log_norm
        elif norm is None:
            self.norm = poisson_norm

        if not self.fit_cumulative:
            self.data.y = {
                key: np.diff(self.data.y[key], prepend=0)
                for key in self.weights
            }

    def check_within_bounds(self, theta):
        for par, value in zip(self.fit_parameters, theta):
            bounds = par.bounds
            if not bounds[0] <= value <= bounds[1]:
                return False
        return True

    def __call__(self, theta):
        if not self.check_within_bounds(theta):
            return -np.inf
        else:
            parameters = dict(zip(self.fit_names, theta))
            return self.get_log_likelihood(parameters)

    def get_log_likelihood(self, parameters):
        if not self.check_within_bounds(list(parameters.values())):
            return -np.inf

        # get model data at daily values
        # when computing diffs, datasets were prepended with 0, so there is no need
        # to evaluate at an extra data point on day earlier
        t_eval = np.arange(self.data.t[0], self.data.t[-1]+2)
        model_data = self.get_model_data(
            t_eval, **parameters, **self.fixed_values
        )
        if model_data == -np.inf:
            return -np.inf
        data_t_indices = np.isin(t_eval, self.data.t)

        def get_one_likelihood(_model, data):
            if not self.fit_cumulative:
                model = np.diff(_model, prepend=0)
            else:
                model = _model

            # slice to match data time coordinates
            model = model[data_t_indices]
            # ensure no model data is smaller than .1
            model = np.maximum(.1, model)
            # only compare data points whose values are >= 1
            data_nonzero = data > .9

            if self.fit_cumulative:
                sigma = np.power(model, .5)
                return self.norm(model[data_nonzero],
                                 data[data_nonzero],
                                 sigma[data_nonzero])
            else:
                return self.norm(model[data_nonzero],
                                 data[data_nonzero])

        likelihood = 0
        for compartment, weight in self.weights.items():
            if weight > 0:
                L = get_one_likelihood(
                    model_data.y[compartment].sum(axis=-1),
                    self.data.y[compartment]
                )
                likelihood += weight * L

        return likelihood

    @classmethod
    def get_model_data(cls, t, **kwargs):
        raise NotImplementedError

    def get_initial_positions(self, walkers, method='normal'):
        if method == 'uniform':
            init = np.array([np.random.uniform(par.bounds[0], par.bounds[1], walkers)
                             for par in self.fit_parameters])
        else:
            init = np.array([par.guess + np.random.randn(walkers) * par.uncertainty
                             for par in self.fit_parameters])
        return init.T

    def sample_uniform(self, num_points, pool=None):
        if not isinstance(num_points, dict):
            num_points = {par.name: num_points for par in self.fit_parameters}

        samples = {
            par.name: list(np.linspace(*par.bounds, num_points[par.name]))
            for par in self.fit_parameters
        }

        from itertools import product
        all_value_sets = product(*[sample for sample in samples.values()])
        values = [values for values in all_value_sets]

        if pool is not None:
            likelihoods = pool.map(self.__call__, values)
        else:
            likelihoods = [self.__call__(value) for value in values]

        return np.array(values), np.array(likelihoods)

    def sample_emcee(self, steps, walkers=None, pool=None, moves=None, progress=True,
                     init_method='normal', backend=None, backend_filename=None):
        if pool is None:
            from multiprocessing import Pool
            pool = Pool()
        if walkers is None:
            walkers = pool._processes

        import emcee

        if backend is not None:
            is_initialized = backend.initialized
        elif backend_filename is not None:
            from pydemic.hdf import HDFBackend
            backend = HDFBackend(backend_filename, self.fit_parameters,
                                 self.fixed_values, self._original_data)
            is_initialized = False
        else:
            is_initialized = False

        if not is_initialized:
            initial_positions = self.get_initial_positions(
                walkers, method=init_method
            )
            ndim = initial_positions.shape[-1]
        else:
            initial_positions = None
            walkers, ndim = backend.shape

        sampler = emcee.EnsembleSampler(walkers, ndim, self, moves=moves,
                                        backend=backend, pool=pool)

        sampler.run_mcmc(initial_positions, steps, progress=progress)

        return sampler
