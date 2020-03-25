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


from pydemic import AttrDict


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


class SimulationResult:
    def __init__(self):
        pass

    def extend(self, population):
        pass


class Simulation:
    def __init__(self, population, epidemiology, severity, age_distribution,
                 containments):
        self.population = population
        self.epidemiology = epidemiology
        self.severity = severity
        self.age_distribution = age_distribution
        self.containments = containments

    def step(self, time, population):
        frac_infected = sum(population.infectious) / self.population_served
        new_time = time + self.dt
        age_groups = population.infectious.keys().sort()

        new_pop = population.copy()

        new_cases = (
            sample(self.importsPerDay * self.dt_days)
            + sample((1 - self.isolatedFrac) * self.infection_rate(new_time)
                     * population.susceptible * frac_infected * self.dt_days)
        )
        new_infectious = min(
            population.exposed,
            sample(population.exposed * self.dt_days / self.incubation_time)
        )
        new_recovered = min(
            population.infectious,
            sample(population.infectious * self.dt_days * self.recovery_rate)
        )
        new_hospitalized = min(
            population.infectious - new_recovered,
            sample(population.infectious * self.dt_days * self.hospitalized_rate)
        )
        new_discharged = min(
            population.hospitalized,
            sample(population.hospitalized * self.dt_days * self.discharge_rate)
        )
        new_critical = min(
            population.hospitalized - new_discharged,
            sample(population.hospitalized * self.dt_days * self.critical_rate)
        )
        new_stabilized = min(
            population.critical,
            sample(population.critical * self.dt_days * self.stabilization_rate)
        )
        new_ICU_dead = min(
            population.critical - new_stabilized,
            sample(population.critical * self.dt_days * self.death_rate)
        )
        new_overflow_stabilized = min(
            population.overflow,
            sample(population.overflow * self.dt_days * self.stabilization_rate)
        )
        new_overflow_dead = min(
            population.overflow - new_overflow_stabilized,
            sample(population.overflow * self.dt_days * self.overflowDeath_rate)
        )

        new_pop.susceptible = population.susceptible - new_cases
        new_pop.exposed = new_cases - new_infectious
        new_pop.infectious = new_infectious - new_recovered - new_hospitalized
        new_pop.hospitalized = (new_hospitalized + new_stabilized
                                + new_overflow_stabilized - new_discharged
                                - new_critical)

        # Cumulative categories
        new_pop.recovered = new_recovered + new_discharged
        new_pop.intensive = new_critical
        new_pop.discharged = new_discharged
        new_pop.dead = new_ICU_dead + new_overflow_dead

        free_ICU_beds = (
            self.total_ICU_beds
            - (sum(population.critical) - sum(new_stabilized) - sum(new_ICU_dead))
        )

        for age in age_groups:
            if free_ICU_beds > new_critical[age]:
                free_ICU_beds -= new_critical[age]
                new_pop.critical[age] = (new_critical[age] - new_stabilized[age]
                                        - new_ICU_dead[age])
                new_pop.overflow[age] = (- new_overflow_dead[age]
                                        - new_overflow_stabilized[age])
            elif free_ICU_beds > 0:
                newOverflow = new_critical[age] - free_ICU_beds
                new_pop.critical[age] = (free_ICU_beds - new_stabilized[age]
                                                - new_ICU_dead[age])
                new_pop.overflow[age] = (newOverflow - new_overflow_dead[age]
                                                - new_overflow_stabilized[age])
                free_ICU_beds = 0
            else:
                new_pop.critical[age] = - new_stabilized[age] - new_ICU_dead[age]
                new_pop.overflow[age] = (
                    new_critical[age] - new_overflow_dead[age]
                    - new_overflow_stabilized[age]
                )

        # If any overflow patients are left AND there are free beds, move them back.
        # Again, move w/ lower age as priority.
        i = 0
        while free_ICU_beds > 0 and i < len(age_groups):
            age = age_groups[i]
            if new_pop.overflow[age] < free_ICU_beds:
                new_pop.critical[age] += new_pop.overflow[age]
                free_ICU_beds -= new_pop.overflow[age]
                new_pop.overflow[age] = 0
            else:
                new_pop.critical[age] += free_ICU_beds
                new_pop.overflow[age] -= free_ICU_beds
                free_ICU_beds = 0
            i += 1

        # NOTE: For debug purposes only.
        # const popSum = sum(new_pop.susceptible) + sum(new_pop.exposed) + sum(new_pop.infectious) + sum(new_pop.recovered) + sum(new_pop.hospitalized) + sum(new_pop.critical) + sum(new_pop.overflow) + sum(new_pop.dead);
        # console.log(math.abs(popSum - self.population_served));

        return new_pop


    def initialize_population(self):
        init = {key: np.zeros_like(age_distribution)
                for key in Population.expected_kwargs}
        init['susceptible'] = age_distribution.copy()
        initial_state = Population(**init)

        return initial_state

    def __call__(self, start_time, end_time):
        population = self.initialize_population()
        result = SimulationResult(population)

        time = start_time
        while time < end_time:
            population = self.step(time, population)
            result.extend(population)

        return result
