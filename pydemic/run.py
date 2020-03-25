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


class Population:
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


def run(params, severity, age_distribution, contaiments):
    pass
