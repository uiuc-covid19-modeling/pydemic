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

    def get_total_population(self):
        # FIXME: does this work?
        return sum(self[key] for key in self.expected_kwargs)


class SimulationResult:
    def __init__(self):
        pass

    def extend(self, population):
        pass


class Simulation:
    def __init__(self, population, epidemiology, severity, age_distribution,
                 containment):
        self.population = population
        self.epidemiology = epidemiology
        self.severity = severity
        self.age_distribution = age_distribution
        self.containment = containment

    def step(self, time, state, sample):
        frac_infected = sum(state.infectious) / self.population_served
        new_time = time + self.dt
        age_groups = state.infectious.keys().sort()

        new_state = state.copy()

        new_cases = (
            sample(self.population.imports_per_day * self.dt_days)
            + sample((1 - self.isolatedFrac) * self.infection_rate(new_time)
                     * state.susceptible * frac_infected * self.dt_days)
        )
        new_infectious = min(
            state.exposed,
            sample(state.exposed * self.dt_days / self.incubation_time)
        )
        new_recovered = min(
            state.infectious,
            sample(state.infectious * self.dt_days * self.recovery_rate)
        )
        new_hospitalized = min(
            state.infectious - new_recovered,
            sample(state.infectious * self.dt_days * self.hospitalized_rate)
        )
        new_discharged = min(
            state.hospitalized,
            sample(state.hospitalized * self.dt_days * self.discharge_rate)
        )
        new_critical = min(
            state.hospitalized - new_discharged,
            sample(state.hospitalized * self.dt_days * self.critical_rate)
        )
        new_stabilized = min(
            state.critical,
            sample(state.critical * self.dt_days * self.stabilization_rate)
        )
        new_ICU_dead = min(
            state.critical - new_stabilized,
            sample(state.critical * self.dt_days * self.death_rate)
        )
        new_overflow_stabilized = min(
            state.overflow,
            sample(state.overflow * self.dt_days * self.stabilization_rate)
        )
        new_overflow_dead = min(
            state.overflow - new_overflow_stabilized,
            sample(state.overflow * self.dt_days * self.overflowDeath_rate)
        )

        new_state.susceptible = state.susceptible - new_cases
        new_state.exposed = new_cases - new_infectious
        new_state.infectious = new_infectious - new_recovered - new_hospitalized
        new_state.hospitalized = (new_hospitalized + new_stabilized
                                + new_overflow_stabilized - new_discharged
                                - new_critical)

        # Cumulative categories
        new_state.recovered = new_recovered + new_discharged
        new_state.intensive = new_critical
        new_state.discharged = new_discharged
        new_state.dead = new_ICU_dead + new_overflow_dead

        free_ICU_beds = (
            self.total_ICU_beds
            - (sum(state.critical) - sum(new_stabilized) - sum(new_ICU_dead))
        )

        for age in age_groups:
            if free_ICU_beds > new_critical[age]:
                free_ICU_beds -= new_critical[age]
                new_state.critical[age] = (new_critical[age] - new_stabilized[age]
                                        - new_ICU_dead[age])
                new_state.overflow[age] = (- new_overflow_dead[age]
                                        - new_overflow_stabilized[age])
            elif free_ICU_beds > 0:
                newOverflow = new_critical[age] - free_ICU_beds
                new_state.critical[age] = (free_ICU_beds - new_stabilized[age]
                                                - new_ICU_dead[age])
                new_state.overflow[age] = (newOverflow - new_overflow_dead[age]
                                                - new_overflow_stabilized[age])
                free_ICU_beds = 0
            else:
                new_state.critical[age] = - new_stabilized[age] - new_ICU_dead[age]
                new_state.overflow[age] = (
                    new_critical[age] - new_overflow_dead[age]
                    - new_overflow_stabilized[age]
                )

        # If any overflow patients are left AND there are free beds, move them back.
        # Again, move w/ lower age as priority.
        i = 0
        while free_ICU_beds > 0 and i < len(age_groups):
            age = age_groups[i]
            if new_state.overflow[age] < free_ICU_beds:
                new_state.critical[age] += new_state.overflow[age]
                free_ICU_beds -= new_state.overflow[age]
                new_state.overflow[age] = 0
            else:
                new_state.critical[age] += free_ICU_beds
                new_state.overflow[age] -= free_ICU_beds
                free_ICU_beds = 0
            i += 1

        # NOTE: For debug purposes only.
        # const popSum = new_state.get_total_population()
        # console.log(math.abs(popSum - self.population_served));

        return new_state


    def initialize_population(self):
        init = {key: np.zeros_like(age_distribution)
                for key in Population.expected_kwargs}
        init['susceptible'] = age_distribution.copy()
        initial_state = Population(**init)

        return initial_state

    def __call__(self, start_time, end_time, sample):
        population = self.initialize_population()
        result = SimulationResult(population)

        time = start_time
        while time < end_time:
            population = self.step(time, population, sample)
            result.extend(population)

        return result
