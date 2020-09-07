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
from itertools import product

__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: SampleParameter
.. autoclass:: LikelihoodEstimator

Likelihood norms
----------------

.. currentmodule:: pydemic.sampling
.. autofunction:: poisson_norm
.. currentmodule:: pydemic
"""


class InvalidParametersError(Exception):
    pass


class SampleParameter:
    """
    Representation of sample parameters as interpreted by
    :class:`LikelihoodEstimator`.

    .. attribute:: name

        The name of the parameter (as to be passed by keyword to likelihood
        estimators).

    .. attribute:: bounds

        A :class:`tuple` ``(lower, upper)`` specifying the range of permitted values
        for the parameter.

    .. attribute:: mean

        The mean of the prior for the parameter.
        If *None*, no prior is assummed for the parameter.

    .. attribute:: sigma

        The uncertainty of the prior for the parameter.
        If *None*, no prior is assummed for the parameter.
    """

    def __init__(self, name, bounds, mean=None, uncertainty=None, sigma=None,
                 guess=None):
        self.name = name
        self.bounds = bounds
        self.mean = mean
        if guess is not None:
            self.mean = guess
        self.guess = guess
        self.uncertainty = uncertainty
        self.sigma = sigma

    def __repr__(self):
        text = "SampleParameter<"
        text += "{0:s}, {1:s}, ".format(str(self.name), str(self.bounds))
        text += "{0:s}, {1:s}>".format(str(self.mean), str(self.uncertainty))
        return text


def l2_log_norm(model, data, **kwargs):
    """
    :arg model: A :class:`numpy.ndarray` of model predictions.

    :arg data: A :class:`numpy.ndarray` of real data.

    :returns: The (negative of the) :math:`L_2` norm of the
        difference of ``log(model)`` and ``log(data)``.
    """

    return -1/2 * np.sum(np.power(np.log(model)-np.log(data), 2.))


def clipped_l2_log_norm(model, data, model_uncert=None, **kwargs):
    """
    :arg model: A :class:`numpy.ndarray` of model predictions, which will be clipped
        from below at ``.1``.

    :arg data: A :class:`numpy.ndarray` of real data.

    :arg model_uncert: A :class:`numpy.ndarray` of uncertainty in the model
        prediction.
        Defaults to ``np.sqrt(model)``.

    :returns: The (negative of the) :math:`L_2` norm of the
        difference of ``log(a)`` and ``log(b)``, weighted elementwise by
        ``model_uncert``.
    """

    if model_uncert is None:
        model_uncert = model**.5
    model = np.maximum(model, .1)
    sig = np.log(model_uncert / model)
    sig += 0.05

    top = np.power(np.log(data)-np.log(model), 2.)
    bot = np.power(sig, 2.)

    return - 1/2 * np.sum(top / bot)


def poisson_norm(model, data, **kwargs):
    """
    :arg model: A :class:`numpy.ndarray` of model predictions.

    :arg data: A :class:`numpy.ndarray` of real data.

    :returns: The log-Poisson likelihood estimator.
    """

    data_finite = np.isfinite(data)
    model = np.maximum(1e-15, model[data_finite])
    data = data[data_finite]
    return np.sum(- model - gammaln(data + 1) + data * np.log(model))


class PoissonPowerNorm:
    fields = {'number'}

    def __init__(self, number):
        self.number = number

    def __call__(self, model, data, **kwargs):
        return poisson_norm(model, data, **kwargs) * self.number


class NegativeBinomialNorm:
    """
    The log-likelihood using a Negative Binomial estimator,

    .. math::

        \\ln P(x)
        = \\ln \\Gamma(x + k) - \\ln (x!)  - \\ln \\Gamma(k)
        + k \\ln \\frac{k}{k + m}
        + x \\ln \\frac{m}{k + m}

    with mean :math:`m`, data :math:`x`, and dispersion :math:`1/k`
    (i.e., where :math:`k \\to \\infty` recovers the Poisson distribution).

    :arg dispersion: The value of :math:`1/k`.

    .. automethod:: __call__
    """

    fields = {'dispersion'}

    def __init__(self, dispersion):
        self.dispersion = dispersion

    def __call__(self, model, data, **kwargs):
        k = 1 / self.dispersion
        x = data
        m = model
        return np.sum(
            gammaln(x + k)
            - gammaln(x + 1)
            - gammaln(k)
            + k * np.log(k / (k + m))
            + x * np.log(m / (k + m))
        )


class LikelihoodEstimator:
    """
    Driver for likelihood estimation.

    The following keyword-only arguments are required:

    :arg simulator: A :class:`class` with a ``get_model_data`` method to be
        used for sampling.
        ``get_model_data`` must have signature ``(t, **kwargs)`` where
        ``t`` is a :class:`pandas.DatetimeIndex` and paramter values
        (from ``fixed_values`` and the particular sample of ``sample_parameters``)
        are passed through ``**kwargs``.

    :arg fixed_values: A :class:`dict` of values fixed for non-sample parameters.

    :arg sample_parameters: A :class:`list` of :class:`SampleParameter`'s
        for sampling.

    :arg data: A :class:`pandas.DataFrame` of the real data to fit against.

    :arg norms: A :class:`dict` specifying the columns of ``data`` (and of the
        result of ``simulator.get_model_data``) by key and the likelihood
        estimator to use for that dataset.
        The values may be ``'poisson'`` (specifying usage of
        :func:`~pydemic.sampling.poisson_norm`)
        or a function with signature ``(model, data, **kwargs)``.
        Values for sample parameters and :attr:`fixed_values`
        are propagated to norm functions by keyword.

    .. automethod:: __call__
    .. automethod:: get_log_likelihood
    .. automethod:: get_initial_positions
    .. automethod:: sample_uniform
    .. automethod:: sample_emcee
    .. automethod:: differential_evolution
    """

    def __init__(self, *, simulator, fixed_values, sample_parameters, data, norms):
        self.sample_parameters = sample_parameters
        self.fit_names = tuple(par.name for par in self.sample_parameters)
        self.fixed_values = fixed_values
        self.data = data.copy()
        self._original_data = self.data
        self.simulator = simulator

        self.norms = {}
        for key, norm in norms.items():
            if norm == 'poisson':
                self.norms[key] = poisson_norm
            elif norm == 'L2':
                self.norms[key] = clipped_l2_log_norm
            elif not callable(norm):
                raise ValueError("values of norms dict must be either "
                                 "'poisson' or a callable.")
            else:
                self.norms[key] = norm

    def get_log_prior(self, theta):
        log_prior = 0
        for par, value in zip(self.sample_parameters, theta):
            bounds = par.bounds
            if not bounds[0] <= value <= bounds[1]:
                log_prior += - np.inf
            elif par.sigma is not None and par.mean is not None:
                guess = par.mean
                sigma = par.sigma
                log_prior += (- np.log(2 * np.pi * sigma**2)
                              - (value - guess)**2 / 2 / sigma**2)

        return log_prior

    def __call__(self, theta):
        """
        Method used internally to compute likelihoods for a set of parameters
        ``theta`` (e.g., by :mod:`emcee`).

        :arg theta: A :class:`numpy.ndarray` of parameter values (with order
            specified by :attr:`sample_parameters`).

        :returns: The likelihood.
        """

        log_prior = self.get_log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf
        else:
            parameters = dict(zip(self.fit_names, theta))
            return log_prior + self.get_log_likelihood(parameters)

    def get_model_result(self, theta):
        parameters = dict(zip(self.fit_names, theta))
        model_data = self.simulator.get_model_data(
            self.data.index, **parameters, **self.fixed_values
        )
        return model_data

    def get_log_likelihood(self, parameters):
        """
        :arg parameters: A :class:`dict` of parameter values for those specified
            specified by :attr:`sample_parameters`, to be passed to
            :attr:`simulator.get_model_data` (along with :attr:`fixed_values`).

        :returns: The likelihood.
        """

        try:
            model_data = self.simulator.get_model_data(
                self.data.index, **parameters, **self.fixed_values
            )
        except InvalidParametersError:
            return -np.inf

        likelihood = 0
        for key, norm in self.norms.items():
            likelihood += norm(model_data[key].values, self.data[key].values,
                               **parameters)

        return likelihood

    def get_initial_positions(self, walkers, method='normal'):
        """
        Generates initial samples for MCMC sampling.

        :arg walkers: The number of walkers used in sampling.

        :returns: A :class:`numpy.ndarray` of initial walker positions with shape
            ``(walkers, len(sample_parameters))``.
        """

        if method == 'uniform':
            init = np.array([np.random.uniform(par.bounds[0], par.bounds[1], walkers)
                             for par in self.sample_parameters])
        else:
            init = np.array([par.mean + np.random.randn(walkers) * par.uncertainty
                             for par in self.sample_parameters])
        return init.T

    def minimizer(self, theta):
        return - self.__call__(theta)

    def basinhopping(self, x0=None, bounds=None, **kwargs):
        if x0 is None:
            x0 = [par.mean for par in self.sample_parameters]
        if bounds is None:
            bounds = [par.bounds for par in self.sample_parameters]

        xmin = np.array(bounds)[:, 0]
        xmax = np.array(bounds)[:, 1]

        def bounds_enforcer(**kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= xmax))
            tmin = bool(np.all(x >= xmin))
            return tmax and tmin

        from scipy.optimize import basinhopping
        sol = basinhopping(
            self.minimizer, x0, accept_test=bounds_enforcer, **kwargs
        )
        sol.x = dict(zip([par.name for par in self.sample_parameters], sol.x))
        return sol

    def differential_evolution(self, workers=-1, progress=True,
                               backend=None, backend_filename=None, **kwargs):
        """
        Performs global optimization using
        :meth:`scipy.optimize.differential_evolution`.

        In addition to the arguments recognized by
        :meth:`scipy.optimize.differential_evolution`
        (which must be passed by keyword), the following arguments are recognized:

        :arg progress: Whether to display a progress bar.

        :arg backend: The :class:`pydemic.hdf.HDFBackend` to use for sampling.
            Defaults to *None*, i.e., no backend.

        :arg backend_filename: The filename to use to create a
            :class:`pydemic.hdf.HDFBackend`.
            Defaults to *None*, in which case no backend file is created.
        """

        bounds = [par.bounds for par in self.sample_parameters]

        if backend is None and backend_filename is not None:
            from pydemic.hdf import HDFOptimizationBackend
            backend = HDFOptimizationBackend(
                backend_filename,
                sample_parameters=self.sample_parameters,
                fixed_values=self.fixed_values,
                data=self._original_data,
                simulator=self.simulator
            )

        from pydemic.desolver import differential_evolution
        sol = differential_evolution(
            self.minimizer, bounds=bounds, workers=workers,
            backend=backend, progress=progress,
            updating=('immediate' if workers == 1 else 'deferred'),
            **kwargs
        )

        if backend is not None:
            backend.set_result(sol,
                               tol=kwargs.get('tol', 1e-2),
                               popsize=kwargs.get('popsize', 15))
            sol = backend.result

        return sol

    def dual_annealing(self, bounds=None, **kwargs):
        if bounds is None:
            bounds = [par.bounds for par in self.sample_parameters]
        bounds = [par.bounds for par in self.sample_parameters]

        from scipy.optimize import dual_annealing
        sol = dual_annealing(
            self.minimizer, bounds=bounds,
            **kwargs
        )
        sol.x = dict(zip([par.name for par in self.sample_parameters], sol.x))
        return sol

    def sample_uniform(self, num_points, pool=None):
        """
        Driver for uniform sampling of the parameter space.

        :arg num_points: The number of points to sample across each dimension
            (with bounds specified by :attr:`SampleParameter.bounds`).

        :arg pool: An :class:`multiprocessing.Pool` to use for parallelization.
            Defaults to *None*, in which case sampling is not parallelized.

        :returns: A :class:`tuple` of two :class:`numpy.ndarray`'s containing the
            sample parameter values and the likelihoods.
        """

        if not isinstance(num_points, dict):
            num_points = {par.name: num_points for par in self.sample_parameters}

        samples = {
            par.name: list(np.linspace(*par.bounds, num_points[par.name]))
            for par in self.sample_parameters
        }

        all_value_sets = product(*[sample for sample in samples.values()])
        values = [values for values in all_value_sets]

        if pool is not None:
            likelihoods = pool.map(self.__call__, values)
        else:
            likelihoods = [self.__call__(value) for value in values]

        return np.array(values), np.array(likelihoods)

    def sample_emcee(self, steps, walkers=None, pool=None, moves=None, progress=True,
                     init_method='uniform', backend=None, backend_filename=None):
        """
        Driver for MCMC sampling using :mod:`emcee`.

        :arg steps: The number of MCMC steps to take.

        :arg walkers: The number of MCMC walkers to use.

        :arg pool: An :class:`multiprocessing.Pool` to use for parallelization.
            Defaults to *None*, in which case sampling is not parallelized.

        :arg init_method:

        :arg backend: The :class:`pydemic.hdf.HDFBackend` to use for sampling.
            Defaults to *None*, i.e., no backend.

        :arg backend_filename: The filename to use to create a
            :class:`pydemic.hdf.HDFBackend`.
            Defaults to *None*, in which case no backend file is created.

        Any remaining keyword arguments are used as specified by
        :class:`emcee.EnsembleSampler`.

        :returns: An :class:`emcee.EnsembleSampler`.
        """

        if pool is None:
            from multiprocessing import Pool
            pool = Pool()
        if walkers is None:
            walkers = pool._processes

        if backend is not None:
            is_initialized = backend.initialized
        elif backend_filename is not None:
            from pydemic.hdf import HDFBackend
            backend = HDFBackend(backend_filename,
                                 sample_parameters=self.sample_parameters,
                                 fixed_values=self.fixed_values,
                                 data=self._original_data,
                                 simulator=self.simulator)
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
