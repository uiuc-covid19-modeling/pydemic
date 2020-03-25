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


class SimulationState(AttrDict):
    expected_kwargs = {
        'infectious',
        'time',
        'susceptible',
        'exposed',
        'hospitalized',
        'intensive',
        'discharged',
        'recovered',
        'critical',
        'dead',
        'overflow',
    }

    def get_total_population(self):
        # FIXME: does this work?
        return sum(self[key] for key in self.expected_kwargs)

    def copy(self):
        input_vals = {key: self[key] for key in self.expected_kwargs}
        return SimulationState(**input_vals)

    def __repr__(self):
        string = ""
        for key in self.expected_kwargs:
            string += key + '\t' + str(self[key]) + '\n'
        return string


class SimulationResult(AttrDict):
    def __init__(self, initial_state):
        input_vals = {key: initial_state[key]
                      for key in initial_state.expected_kwargs}
        # input_vals['times'] = np.array(start_time)
        super().__init__(**input_vals)

    def extend(self, population):
        # self.times = np.append(self.times, time)
        for key in population.expected_kwargs:
            self[key] = np.vstack((self[key], population[key]))


ms_per_day = 24 * 60 * 60 * 1000


class Simulation:
    def __init__(self, population, epidemiology, severity, age_distribution,
                 containment):
        self.population = population
        self.epidemiology = epidemiology
        self.severity = severity
        self.age_distribution = age_distribution
        self.containment = containment

        # infer parameters
        self.dt_days = .25
        self.dt = .25 * ms_per_day

        self.num_age_groups = len(age_distribution.counts)
        total = np.sum(age_distribution)

        freqs = age_distribution.counts / np.sum(age_distribution.counts)
        self.infection_severity_ratio = (
            severity.severe / 100 * severity.confirmed / 100
        )

        self.infection_critical = (
            self.infection_severity_ratio * (severity.critical / 100)
        )
        self.infection_fatality = self.infection_critical * (severity.fatal / 100)

        dHospital = self.infection_severity_ratio
        dCritical = severity.critical / 100
        dFatal = severity.fatal / 100

        # Age specific rates
        self.isolated_frac = severity.isolated / 100
        self.recovery_rate = (1 - dHospital) / epidemiology.infectious_period
        self.hospitalized_rate = dHospital / epidemiology.infectious_period
        self.discharge_rate = (1 - dCritical) / epidemiology.length_hospital_stay
        self.critical_rate = dCritical / epidemiology.length_hospital_stay
        self.stabilization_rate = (1 - dFatal) / epidemiology.length_ICU_stay
        self.death_rate = dFatal / epidemiology.length_ICU_stay
        self.overflow_death_rate = epidemiology.overflow_severity * self.death_rate

        hospitalized_frac = np.sum(freqs * dHospital)
        critical_frac_hospitalized = np.sum(freqs * dCritical)
        fatal_frac_critical = np.sum(freqs * dFatal)
        avg_isolated_frac = np.sum(freqs * severity.isolated) / 100

        # assume flat distribution of imports among age groups
        fractional_imports = population.imports_per_day / self.num_age_groups
        self.imports_per_day = fractional_imports * np.ones(self.num_age_groups)

        self.totals = {
            'recovery_rate': (
                (1 - hospitalized_frac) / epidemiology.infectious_period
            ),
            'hospitalized_rate': hospitalized_frac / epidemiology.infectious_period,
            'discharge_rate': (
                (1 - critical_frac_hospitalized) / epidemiology.length_hospital_stay
            ),
            'critical_rate': (
                critical_frac_hospitalized / epidemiology.length_hospital_stay
            ),
            'death_rate': fatal_frac_critical / epidemiology.length_ICU_stay,
            'stabilization_rate': (
                (1 - fatal_frac_critical) / epidemiology.length_ICU_stay
            ),
            'overflowDeath_rate': (
                epidemiology.overflow_severity
                * fatal_frac_critical
                / epidemiology.length_ICU_stay
            ),
            'isolatedFrac': avg_isolated_frac,
        }

        # nfectivity dynamics
        self.avg_infection_rate = epidemiology.r0 / epidemiology.infectious_period

    def infection_rate(self, time):
        from pydemic import date_to_ms
        jan_2020 = date_to_ms((2020, 1, 1))
        peak_day = 30 * self.epidemiology.peak_month + 15
        time_offset = (time - jan_2020) / ms_per_day - peak_day
        phase = 2 * np.pi * time_offset / 365
        cont = self.containment(time)
        return (
            self.avg_infection_rate *
            (1 + self.epidemiology.seasonal_forcing  * np.cos(phase))
        )

    def step(self, time, state, sample):
        frac_infected = sum(state.infectious) / self.population.population_served
        new_time = time + self.dt
        new_state = state.copy()
        new_state.time = new_time

        new_cases = (
            sample(self.imports_per_day * self.dt_days)
            + sample((1 - self.isolated_frac)
                     * self.containment(time)
                     * self.infection_rate(new_time)
                     * state.susceptible * frac_infected * self.dt_days)
        )
        new_infectious = np.minimum(
            state.exposed,
            sample(state.exposed * self.dt_days / self.epidemiology.incubation_time)
        )
        new_recovered = np.minimum(
            state.infectious,
            sample(state.infectious * self.dt_days * self.recovery_rate)
        )
        new_hospitalized = np.minimum(
            state.infectious - new_recovered,
            sample(state.infectious * self.dt_days * self.hospitalized_rate)
        )
        new_discharged = np.minimum(
            state.hospitalized,
            sample(state.hospitalized * self.dt_days * self.discharge_rate)
        )
        new_critical = np.minimum(
            state.hospitalized - new_discharged,
            sample(state.hospitalized * self.dt_days * self.critical_rate)
        )
        new_stabilized = np.minimum(
            state.critical,
            sample(state.critical * self.dt_days * self.stabilization_rate)
        )
        new_ICU_dead = np.minimum(
            state.critical - new_stabilized,
            sample(state.critical * self.dt_days * self.death_rate)
        )
        new_overflow_stabilized = np.minimum(
            state.overflow,
            sample(state.overflow * self.dt_days * self.stabilization_rate)
        )
        new_overflow_dead = np.minimum(
            state.overflow - new_overflow_stabilized,
            sample(state.overflow * self.dt_days * self.overflow_death_rate)
        )

        new_state.susceptible += - new_cases
        new_state.exposed += new_cases - new_infectious
        new_state.infectious += new_infectious - new_recovered - new_hospitalized
        new_state.hospitalized += (new_hospitalized + new_stabilized
                                   + new_overflow_stabilized - new_discharged
                                   - new_critical)

        # Cumulative categories
        new_state.recovered += new_recovered + new_discharged
        new_state.intensive += new_critical
        new_state.discharged += new_discharged
        new_state.dead += new_ICU_dead + new_overflow_dead

        free_ICU_beds = (
            self.population.ICU_beds
            - (sum(state.critical) - sum(new_stabilized) - sum(new_ICU_dead))
        )

        for age in range(self.num_age_groups):
            if free_ICU_beds > new_critical[age]:
                free_ICU_beds -= new_critical[age]
                new_state.critical[age] = (new_critical[age] - new_stabilized[age]
                                           - new_ICU_dead[age])
                new_state.overflow[age] = (- new_overflow_dead[age]
                                           - new_overflow_stabilized[age])
            elif free_ICU_beds > 0:
                new_overflow = new_critical[age] - free_ICU_beds
                new_state.critical[age] = (free_ICU_beds - new_stabilized[age]
                                           - new_ICU_dead[age])
                new_state.overflow[age] = (new_overflow - new_overflow_dead[age]
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
        while free_ICU_beds > 0 and i < self.num_age_groups:
            if new_state.overflow[i] < free_ICU_beds:
                new_state.critical[i] += new_state.overflow[i]
                free_ICU_beds -= new_state.overflow[i]
                new_state.overflow[i] = 0
            else:
                new_state.critical[i] += free_ICU_beds
                new_state.overflow[i] -= free_ICU_beds
                free_ICU_beds = 0
            i += 1

        # NOTE: For debug purposes only.
        # popSum = new_state.get_total_population()

        return new_state

    def initialize_population(self, start_time):
        ages = np.array(self.age_distribution.counts, dtype='float64')

        init = {key: np.zeros(len(ages))
                for key in SimulationState.expected_kwargs}
        init['time'] = start_time

        fracs = ages / np.sum(ages)
        init['susceptible'] = np.round(fracs * self.population.population_served)

        i_middle = round(ages.shape[0] / 2) + 1
        initial_cases = self.population.suspected_cases_today
        init['susceptible'][i_middle] -= initial_cases
        init['infectious'][i_middle] = 0.3 * initial_cases
        init['exposed'][i_middle] = 0.7 * initial_cases

        initial_state = SimulationState(**init)

        return initial_state

    def __call__(self, start_time, end_time, sample):
        state = self.initialize_population(start_time)
        result = SimulationResult(state)

        time = start_time
        while time < end_time:
            state = self.step(time, state, sample)
            result.extend(state)
            time += self.dt

        return result
