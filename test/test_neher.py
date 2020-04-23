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
import pytest
from pydemic.models.neher import (PopulationModel, AgeDistribution, SeverityModel,
                                  EpidemiologyModel)
from pydemic import date_to_ms
from neher_port import NeherPortSimulation
from pydemic.models import NeherModelSimulation

n_age_groups = 9
start_date = (2020, 3, 1)
end_date = (2020, 9, 1)
containment_date = (2020, 3, 20)
containment_factor = 1

population = PopulationModel(
    country='United States of America',
    cases='USA-Illinois',
    population_served=12659682,
    initial_cases=215,
    ICU_beds=1e10,  # originally 1055
    hospital_beds=1e10,  # originally 31649
    imports_per_day=5.0,
)
age_distribution = AgeDistribution(
    bin_edges=np.arange(0, 90, 10),
    counts=[39721484, 42332393, 46094077, 44668271, 40348398, 42120077,
            38488173, 24082598, 13147180]
)
severity = SeverityModel(
    id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
    age_group=np.arange(0., 90., 10),
    isolated=np.array([0., 0., 20., 10., 0., 0., 50., 90., 0.]),
    confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
    severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
    critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
    fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
)
epidemiology = EpidemiologyModel(
    r0=3.7,
    incubation_time=5,
    infectious_period=3,
    length_hospital_stay=4,
    length_ICU_stay=14,
    seasonal_forcing=0.2,
    peak_month=0,
    overflow_severity=2
)


@pytest.mark.parametrize("fraction_hospitalized", [.5, .005])
def test_neher(fraction_hospitalized):
    from pydemic.models.neher import ContainmentModel
    containment = ContainmentModel(start_date, (2021, 1, 1))
    containment.add_sharp_event(containment_date, containment_factor)

    sim = NeherModelSimulation(
        epidemiology, severity, population.imports_per_day,
        n_age_groups, containment, fraction_hospitalized=fraction_hospitalized
    )
    compartments = sim.compartments

    from neher_port import NeherContainmentModel
    containment = NeherContainmentModel(start_date, end_date)
    containment.add_sharp_event(containment_date, containment_factor)
    port = NeherPortSimulation(population, epidemiology, severity, age_distribution,
                               containment)
    start_time = date_to_ms(start_date)
    end_time = date_to_ms(end_date)
    port_result = port(start_time, end_time, lambda x: x)

    y0 = sim.get_initial_population(population, age_distribution)

    new_result = sim((start_date, end_date), y0, dt=.25)
    for key, val in new_result.y.items():
        new_result.y[key] = val[:-1, ...]
    new_result.t = new_result.t[:-1]

    for name in compartments:
        test = np.logical_and(port_result.y[name] > 0, new_result.y[name] > 0)
        relerr = np.abs(1 - port_result.y[name][test] / new_result.y[name][test])
        print('max err for', name, 'is', np.max(relerr))
        assert np.max(relerr) < .05

    # compare to scipy with a (much) smaller timestep
    new_result = sim((start_date, end_date), y0, dt=.025)

    scipy_res = sim.solve_deterministic((start_date, (2020, 9, 2)), y0)
    scipy_res = sim.dense_to_logger(scipy_res, new_result.t)

    for name in compartments:
        test = np.logical_and(new_result.y[name] > 0, scipy_res.y[name] > 0)
        relerr = np.abs(1 - new_result.y[name][test] / scipy_res.y[name][test])
        avg_digits = np.average(np.log(relerr[relerr > 0]))
        print('avg number of digits of agreement for', name, 'is', avg_digits)
        assert avg_digits < -2

    total_people = sum(np.sum(scipy_res.y[name], axis=-1) for name in compartments)
    total_pop = total_people[0]
    total_err = np.max(np.abs(1 - total_people / total_pop))
    print('total error is', np.average(total_err))
    assert np.max(total_err) < 1.e-13


def test_neher_estimator():
    from pydemic.sampling import LikelihoodEstimator
    t = np.array([78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90])
    y = {'dead': np.array([4,  5,  6,  9, 12, 16, 19, 26, 34, 47, 65, 73, 99])}
    dates = pd.to_datetime(t, origin='2020-01-01', unit='D')
    data = pd.DataFrame(index=dates, data=y)
    # the values from here are all overwritten
    population = "USA-Illinois"
    age_dist_pop = "United States of America"

    from pydemic.sampling import SampleParameter

    fit_parameters = [
        SampleParameter('r0', (2, 4), 3, .2),
        SampleParameter('start_day', (50, 60), 55, 2),
    ]

    fixed_values = dict(
        end_day=t[-1] + 2,
        population=population,
        age_dist_pop=age_dist_pop,
        population_served=12659682,
        initial_cases=10.,
        imports_per_day=1.1,
        age_distribution=age_distribution,
        mitigation_factor_0=1,
        mitigation_factor_1=1,
        mitigation_t_0=69,
        mitigation_t_1=79,
        fraction_hospitalized=1.,
        min_mitigation_spacing=0,
    )
    estimator = LikelihoodEstimator(
        fit_parameters, fixed_values, data, NeherModelSimulation,
        {'dead': 'L2'}
    )

    test_L = estimator([2.7, 53.8])

    rtol = 1e-3
    assert np.abs(1 - test_L / (-0.019927175841621653)) < rtol, test_L

    # run uniform sampling
    nsamples = 10
    uniform_values, uniform_likelihoods = estimator.sample_uniform(nsamples)
    uniform_likelihoods = uniform_likelihoods.reshape(nsamples, nsamples)
    r0_vals = uniform_values[:, 0].reshape(nsamples, nsamples)
    start_day_vals = uniform_values[:, 1].reshape(nsamples, nsamples)

    max_loc = np.where(uniform_likelihoods == uniform_likelihoods.max())
    r0_best = r0_vals[max_loc][0]
    start_day_best = start_day_vals[max_loc][0]
    max_L = uniform_likelihoods.max()

    assert np.abs(1 - r0_best / 2.6666666666666665) < rtol, r0_best
    assert np.abs(1 - start_day_best / 53.333333333333336) < rtol, start_day_best
    assert np.abs(1 - max_L / (-0.0550594031551889)) < rtol, max_L

    import emcee
    np.random.seed(42)

    n_walkers = 4
    n_steps = 10

    initial_positions = estimator.get_initial_positions(n_walkers)
    n_dims = initial_positions.shape[-1]

    sampler = emcee.EnsembleSampler(n_walkers, n_dims, estimator)
    sampler.run_mcmc(initial_positions, n_steps)

    flat_samples = sampler.get_chain(discard=0, flat=True)
    emcee_likelihoods = sampler.get_log_prob(discard=0, flat=True)
    r0_best = flat_samples[np.argmax(emcee_likelihoods)][0]
    start_day_best = flat_samples[np.argmax(emcee_likelihoods)][1]
    max_L = emcee_likelihoods.max()

    assert np.abs(1 - r0_best / 2.677571956653811) < rtol, r0_best
    assert np.abs(1 - start_day_best / 53.5794313238115) < rtol, start_day_best
    assert np.abs(1 - max_L / (-0.025986918976544388)) < rtol, max_L


if __name__ == "__main__":
    test_neher(.5)
    test_neher_estimator()
