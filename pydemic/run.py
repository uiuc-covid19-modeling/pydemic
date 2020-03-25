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
from pydemic import AttrDict


class Parameters(AttrDict):
    expected_kwargs = {
        'importsPerDay',
        'time_delta_days',
        'isolatedFrac',
        'infectionRate',
        'incubationTime',
        'recoveryRate',
        'hospitalizedRate',
        'dischargeRate',
        'criticalRate',
        'stabilizationRate',
        'deathRate',
        'overflowDeathRate',
        'ICUBeds',
    }


class Population(AttrDict):
    expected_kwargs = {
        'infections',
        'time',
        'susceptible',
        'exposed',
        'hospitalized',
        'critical',
        'overflow',
    }


def evolve(population, pars, sample):
    frac_infected = sum(population.infectious) / pars.population_served
    new_time = population.time + pars.time_delta
    age_groups = population.infectious.keys().sort()

    new_population = population.copy()

    new_cases = (
        sample(pars.importsPerDay * pars.time_delta_days)
        + sample((1 - pars.isolatedFrac) * pars.infectionRate(new_time)
                 * population.susceptible * frac_infected * pars.time_delta_days)
    )
    new_infectious = min(
        population.exposed,
        sample(population.exposed * pars.time_delta_days / pars.incubationTime)
    )
    new_recovered = min(
        population.infectious,
        sample(population.infectious * pars.time_delta_days * pars.recoveryRate)
    )
    new_hospitalized = min(
        population.infectious - new_recovered,
        sample(population.infectious * pars.time_delta_days * pars.hospitalizedRate)
    )
    new_discharged = min(
        population.hospitalized,
        sample(population.hospitalized * pars.time_delta_days * pars.dischargeRate)
    )
    new_critical = min(
        population.hospitalized - new_discharged,
        sample(population.hospitalized * pars.time_delta_days * pars.criticalRate)
    )
    new_stabilized = min(
        population.critical,
        sample(population.critical * pars.time_delta_days * pars.stabilizationRate)
    )
    new_ICU_dead = min(
        population.critical - new_stabilized,
        sample(population.critical * pars.time_delta_days * pars.deathRate)
    )
    new_overflow_stabilized = min(
        population.overflow,
        sample(population.overflow * pars.time_delta_days * pars.stabilizationRate)
    )
    new_overflow_dead = min(
        population.overflow - new_overflow_stabilized,
        sample(population.overflow * pars.time_delta_days * pars.overflowDeathRate)
    )

    new_population.susceptible = population.susceptible - new_cases
    new_population.exposed = new_cases - new_infectious
    new_population.infectious = new_infectious - new_recovered - new_hospitalized
    new_population.hospitalized = (new_hospitalized + new_stabilized
                                   + new_overflow_stabilized - new_discharged
                                   - new_critical)

    # Cumulative categories
    new_population.recovered = new_recovered + new_discharged
    new_population.intensive = new_critical
    new_population.discharged = new_discharged
    new_population.dead = new_ICU_dead + new_overflow_dead

    free_ICU_beds = (
        pars.ICUBeds
        - (sum(population.critical) - sum(new_stabilized) - sum(new_ICU_dead))
    )

    for age in age_groups:
        if free_ICU_beds > new_critical[age]:
            free_ICU_beds -= new_critical[age]
            new_population.critical[age] = (new_critical[age] - new_stabilized[age]
                                    - new_ICU_dead[age])
            new_population.overflow[age] = (- new_overflow_dead[age]
                                    - new_overflow_stabilized[age])
        elif free_ICU_beds > 0:
            newOverflow = new_critical[age] - free_ICU_beds
            new_population.critical[age] = (free_ICU_beds - new_stabilized[age]
                                            - new_ICU_dead[age])
            new_population.overflow[age] = (newOverflow - new_overflow_dead[age]
                                            - new_overflow_stabilized[age])
            free_ICU_beds = 0
        else:
            new_population.critical[age] = - new_stabilized[age] - new_ICU_dead[age]
            new_population.overflow[age] = (
                new_critical[age] - new_overflow_dead[age]
                - new_overflow_stabilized[age]
            )

    # If any overflow patients are left AND there are free beds, move them back.
    # Again, move w/ lower age as priority.
    i = 0
    while free_ICU_beds > 0 and i < len(age_groups):
        age = age_groups[i]
        if new_population.overflow[age] < free_ICU_beds:
            new_population.critical[age] += new_population.overflow[age]
            free_ICU_beds -= new_population.overflow[age]
            new_population.overflow[age] = 0
        else:
            new_population.critical[age] += free_ICU_beds
            new_population.overflow[age] -= free_ICU_beds
            free_ICU_beds = 0
        i += 1

    # NOTE: For debug purposes only.
    # const popSum = sum(new_population.susceptible) + sum(new_population.exposed) + sum(new_population.infectious) + sum(new_population.recovered) + sum(new_population.hospitalized) + sum(new_population.critical) + sum(new_population.overflow) + sum(new_population.dead);
    # console.log(math.abs(popSum - pars.population_served));

    return new_population



def simulate(initial_state, func):
    dynamics = [initialState]
    while (dynamics[dynamics.length - 1].time < tMax):
        pop = dynamics[dynamics.length - 1]
        dynamics.push(evolve(pop, modelParams, func))

    return collectTotals(dynamics)


def run(population, simulation, epidemiology,
        severity, age_distribution, containments):
    # modelParams = getPopulationParams(params, severity, age_distribution,
    #                                   interpolateTimeSeries(containment))
    # tMin = params.simulationTimeRange.tMin.getTime()  # int ms
    # tMax = params.simulationTimeRange.tMax.getTime()  # int ms
    # initialCases = params.suspectedCasesToday
    # initialState = initializePopulation(modelParams.populationServed,
    #                                     initialCases, tMin, age_distribution)

    # sim: AlgorithmResult = {
    # deterministicTrajectory: simulate(initialState, identity),
    # stochasticTrajectories: [],
    # params: modelParams,
    # }

    init = {key: np.zeros_like(age_distribution)
            for key in Population.expected_kwargs}
    init['susceptible'] = age_distribution.copy()
    initial_state = Population(**init)
    print(initial_state.items())

    for i in range(modelParams.numberStochasticRuns):
        initialState = initializePopulation(
            modelParams.populationServed, initialCases, tMin, age_distribution
        )
        sim.stochasticTrajectories.push(simulate(initialState, poisson))

    return sim


if __name__ == "__main__":
    simulation = {
        "start": [ 2020, 3, 1, 0, 0, 0 ],
        "end": [ 2020, 9, 1, 0, 0, 0 ]
    }

    epidemiology = {
        "r0": 3.7,
        "incubationTime": 5,
        "infectiousPeriod": 3,
        "lengthHospitalStay": 4,
        "lengthICUStay": 14,
        "seasonalForcing": 0.2,
        "peakMonth": 0,
        "overflowSeverity": 2
    }

    mitigation_factor = 0.8
    containments = {
        "times": [
        [ 2020, 3, 1, 0, 0, 0 ],
        [ 2020, 3, 14, 0, 0, 0 ],
        [ 2020, 3, 15, 0, 0, 0 ],
        [ 2020, 9, 1, 0, 0, 0 ]
        ],
        "factors": [
        1.0,
        1.0,
        mitigation_factor,
        mitigation_factor
        ]
    }

    severity = [
        {
        "id": 0,
        "ageGroup": "0-9",
        "isolated": 0.0,
        "confirmed": 5.0,
        "severe": 1.0,
        "critical": 5,
        "fatal": 30
        },
        {
        "id": 2,
        "ageGroup": "10-19",
        "isolated": 0.0,
        "confirmed": 5.0,
        "severe": 3.0,
        "critical": 10,
        "fatal": 30
        },
        {
        "id": 4,
        "ageGroup": "20-29",
        "isolated": 0.0,
        "confirmed": 10.0,
        "severe": 3.0,
        "critical": 10,
        "fatal": 30
        },
        {
        "id": 6,
        "ageGroup": "30-39",
        "isolated": 0.0,
        "confirmed": 15.0,
        "severe": 3.0,
        "critical": 15,
        "fatal": 30
        },
        {
        "id": 8,
        "ageGroup": "40-49",
        "isolated": 0.0,
        "confirmed": 20.0,
        "severe": 6.0,
        "critical": 20,
        "fatal": 30
        },
        {
        "id": 10,
        "ageGroup": "50-59",
        "isolated": 0.0,
        "confirmed": 25.0,
        "severe": 10.0,
        "critical": 25,
        "fatal": 40
        },
        {
        "id": 12,
        "ageGroup": "60-69",
        "isolated": 0.0,
        "confirmed": 30.0,
        "severe": 25.0,
        "critical": 35,
        "fatal": 40
        },
        {
        "id": 14,
        "ageGroup": "70-79",
        "isolated": 0.0,
        "confirmed": 40.0,
        "severe": 35.0,
        "critical": 45,
        "fatal": 50
        },
        {
        "id": 16,
        "ageGroup": "80+",
        "isolated": 0.0,
        "confirmed": 50.0,
        "severe": 50.0,
        "critical": 55,
        "fatal": 50
        }
    ]

    severity = {}

    POPULATION_NAME = "USA-Illinois"
    AGE_DATA_NAME = "United States of America"

    from pydemic.load import get_country_population_model
    population = get_country_population_model(POPULATION_NAME, AGE_DATA_NAME)
    age_distribution = population['populationsByDecade']

    run(population, simulation, epidemiology,
        severity, age_distribution, containments)
