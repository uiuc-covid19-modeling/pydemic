import os
os.environ["OMP_NUM_THREADS"] = '1'
# import numpy as np

population = "USA-New York"
age_dist_pop = "United States of America"
from pydemic.data import united_states
data = united_states.get_case_data('NY')

fit_cumulative = False

from pydemic.sampling import SampleParameter

fit_parameters = [
    SampleParameter('r0', (1, 5), 3, .2),
    SampleParameter('start_day', (40, 60), 50, 2),
    SampleParameter('mitigation_factor', (.01, 1), .9, .1),
    SampleParameter('mitigation_day', (70, 95), 80, 2),
    SampleParameter('mitigation_width', (.05, 20), 10, 2),
    SampleParameter('fraction_hospitalized', (.05, 10), 5, 3),
]

fixed_values = dict(
    end_day=data.t[-1] + 2,
    population=population,
    age_dist_pop=age_dist_pop,
    initial_cases=10.,
    imports_per_day=1.1,
    length_ICU_stay=14,
)

from pydemic.models.neher import NeherModelEstimator
estimator = NeherModelEstimator(
    fit_parameters, fixed_values, data, fit_cumulative=fit_cumulative,
    weights={'dead': 1, 'positive': 1, 'critical': 0}
)

from multiprocessing import Pool
pool = Pool(32)
sampler = estimator.sample_emcee(
    80000, walkers=64, pool=pool, backend_filename="2020-04-05-new-york.h5"
)
pool.close()
pool.join()
