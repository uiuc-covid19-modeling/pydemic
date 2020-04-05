import emcee
import numpy as np
from multiprocessing import Pool, cpu_count


class EmceeSampler:
   
    def __init__(self, estimator):
        self._estimator = estimator

    def __call__(self, **kwargs):
        n_steps = kwargs.pop('n_steps', 1000)
        n_walkers = kwargs.pop('n_walkers', cpu_count())
        n_workers = kwargs.pop('n_workers', max(cpu_count(), n_walkers))
        checkpoint_steps = kwargs.pop('checkpoint_steps', 100)
        backend_fname = kwargs.pop('backend_fname', "samples.h5")
        pool = Pool(n_workers)

        initial_positions = self._estimator.get_initial_positions(n_walkers)
        n_dims = initial_positions.shape[-1]

        backend = emcee.backends.HDFBackend(backend_fname)
        backend.reset(n_walkers, n_dims)

        sampler = emcee.EnsembleSampler(n_walkers, n_dims, self._estimator, backend=backend, pool=pool)

        index = 0
        autocorrelations = np.zeros((int(n_steps//checkpoint_steps)+1, n_dims))
        for sample in sampler.sample(initial_positions, iterations=n_steps, progress=True):

            if sampler.iteration % checkpoint_steps:
                continue

            tau = sampler.get_autocorr_time(tol=0)
            autocorrelations[index,:] = np.mean(tau)
            index += 1

            # FIXME: automatic convergence doesn't work .. odd array shapes?

            print(autocorrelations)

        self.autocorrelations = autocorrelations

        return sampler


