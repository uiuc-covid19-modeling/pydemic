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
from scipy.special import gammaln  # pylint: disable=E0611
from warnings import warn
from itertools import product


class InvalidParametersError(Exception):
    pass


class SampleParameter:
    def __init__(self, name, bounds, guess=None, uncertainty=None, sigma=None):
        self.name = name
        self.bounds = bounds
        self.guess = guess
        self.uncertainty = uncertainty
        self.sigma = sigma

    def __repr__(self):
        text = "SampleParameter<"
        text += "{0:s}, {1:s}, ".format(str(self.name), str(self.bounds))
        text += "{0:s}, {1:s}>".format(str(self.guess), str(self.uncertainty))
        return text


def l2_log_norm(a, b):
    return -1/2 * np.sum(np.power(np.log(a)-np.log(b), 2.))


def clipped_l2_log_norm(model, data, model_uncert=None):
    if model_uncert is None:
        model_uncert = model**.5
    model = np.maximum(model, .1)
    sig = np.log(model_uncert / model)
    sig += 0.05

    top = np.power(np.log(data)-np.log(model), 2.)
    bot = np.power(sig, 2.)

    return - 1/2 * np.sum(top / bot)


def poisson_norm(model, data):
    # ensure no model data is smaller than .1
    model = np.maximum(.1, model)
    # only compare data points whose values are >= 1
    data_nonzero = data > .9
    model = model[data_nonzero]
    data = data[data_nonzero]
    return np.sum(- model - gammaln(data) + data * np.log(model))


def poisson_norm_diff(model, data):
    model = np.diff(model, prepend=0)
    data = np.diff(data, prepend=0)
    return poisson_norm(model, data)


class LikelihoodEstimator:
    def __init__(self, fit_parameters, fixed_values, data, simulator,
                 norms={}, weights=None, norm=None, fit_cumulative=None):
        self.fit_parameters = fit_parameters
        self.fit_names = tuple(par.name for par in fit_parameters)
        self.fixed_values = fixed_values
        self.data = data.copy().fillna(0)
        self._original_data = self.data
        self.simulator = simulator

        if norm is not None:
            warn("Passing norm is deprecated. "
                 "Pass custom norm functions to norms instead.",
                 DeprecationWarning, stacklevel=2)

        if weights is not None:
            warn("Passing weights is deprecated. "
                 "Pass custom norm functions to norms instead.",
                 DeprecationWarning, stacklevel=2)
            if len(norms) == 0:
                norms = weights

        if fit_cumulative is not None:
            warn("Passing fit_cumulative is deprecated. "
                 "Pass 'L2' or a custom norm function to norms instead.",
                 DeprecationWarning, stacklevel=2)

        if len(norms) == 0:
            raise ValueError('Must fit over at least one dataset.')

        self.norms = {}
        for key, norm in norms.items():
            if norm == 'poisson_diff':
                self.norms[key] = poisson_norm_diff
            elif norm == 'poisson':
                self.norms[key] = poisson_norm
            elif norm == 'L2':
                self.norms[key] = clipped_l2_log_norm
            elif not callable(norm):
                warn("Passing weights is deprecated. "
                     "Pass norm functions (or 'poisson'/'l2') to norms instead. "
                     "This will raise an exception in future versions.",
                     DeprecationWarning, stacklevel=2)
                if norm != 1.:
                    raise ValueError(
                        'weights not equal to one must be implemented manualy'
                    )
                if fit_cumulative:
                    self.norms[key] = clipped_l2_log_norm
                else:
                    self.norms[key] = poisson_norm_diff
            else:
                self.norms[key] = norm

    def get_log_prior(self, theta):
        log_prior = 0
        for par, value in zip(self.fit_parameters, theta):
            bounds = par.bounds
            if not bounds[0] <= value <= bounds[1]:
                log_prior += - np.inf
            elif par.sigma is not None and par.guess is not None:
                guess = par.guess
                sigma = par.sigma
                log_prior += (- np.log(2 * np.pi * sigma**2)
                              - (value - guess)**2 / 2 / sigma**2)

        return log_prior

    def __call__(self, theta):
        log_prior = self.get_log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf
        else:
            parameters = dict(zip(self.fit_names, theta))
            return log_prior + self.get_log_likelihood(parameters)

    def get_log_likelihood(self, parameters):
        try:
            model_data = self.simulator.get_model_data(
                self.data.index, **parameters, **self.fixed_values
            )
        except InvalidParametersError:
            return -np.inf

        likelihood = 0
        for key, norm in self.norms.items():
            likelihood += norm(model_data[key].to_numpy(),
                               self.data[key].to_numpy())

        return likelihood

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

        import emcee
        sampler = emcee.EnsembleSampler(walkers, ndim, self, moves=moves,
                                        backend=backend, pool=pool)

        sampler.run_mcmc(initial_positions, steps, progress=progress)

        return sampler
