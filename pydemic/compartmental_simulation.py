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
from pydemic import AttrDict, ErlangProcess, Reaction

__doc__ = """
.. currentmodule:: pydemic
.. autoclass:: Simulation
.. currentmodule:: pydemic.simulation
"""


class SimulationState(AttrDict):


    expected_kwargs = {
    }

    def get_total_population(self):
        # FIXME: does this work?
        return sum(self[key] for key in self.expected_kwargs)

    def copy(self):
        input_vals = {}
        for key in self.expected_kwargs:
            if isinstance(self[key], np.ndarray):
                input_vals[key] = self[key].copy()
            else:
                input_vals[key] = self[key]
        return SimulationState(**input_vals)

    def __repr__(self):
        string = ""
        for key in self.expected_kwargs:
            string += key + '\t' + str(self[key]) + '\n'
        return string


class SimulationResult(AttrDict):
    def __init__(self, initial_state, n_time_steps):
        input_vals = {}
        for key in initial_state.expected_kwargs:
            val = initial_state[key]
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            ary = np.zeros(shape=(n_time_steps,)+val.shape)
            ary[0] = val
            input_vals[key] = ary

        super().__init__(**input_vals)
        self.slice = 0

    def extend(self, population):
        self.slice += 1
        for key in population.expected_kwargs:
            self[key][self.slice] = population[key]



class CompartmentalModelSimulation:

    _compartments = {}
    _network = []
    _compartment_shape = 1

    def _get_uniq_compartment_name(self,name):
        return name.replace(":","::")

    def __init__(self, reactions, demographics):

        print(" - creating new CompartmentalModelSimulation instance")

        #self._compartments = compartments

        ## generate list of internal graph nodes. this has to 
        ## happen after we have processed the compartments
        self.generate_network(reactions)

        # FIXME: load demographic information and update
        # self._compartment_shape appropriately
        # initialize compartments 
        # FIXME: maybe also/alternatively get compartment shape from initial condition?
        for compartment in set([x.lhs for x in self._network]+[x.rhs for x in self._network]):
            self._compartments[compartment] = np.zeros(self._compartment_shape)

        self.__dict__ == self._compartments




    def generate_network(self, reactions):
        self._network = []
        for reaction in reactions:
            if type(reaction) == ErlangProcess:
                # TODO, doesn't work if reaction.shape == 1
                count = 2
                rx = Reaction(
                    lhs=self._get_uniq_compartment_name(reaction.lhs),
                    rhs=reaction.lhs+":{0:s}:{1:d}".format(reaction.rhs, count),
                    evaluation=reaction.scale
                )
                self._network.append(rx)
                for k in range(1,reaction.shape-1):
                    rx = Reaction(
                        lhs=reaction.lhs+":{0:s}:{1:d}".format(reaction.rhs, count),
                        rhs=reaction.lhs+":{0:s}:{1:d}".format(reaction.rhs, count+1),
                        evaluation=reaction.scale
                    )
                    count += 1
                    self._network.append(rx)
                if reaction.shape > 1:
                    rx = Reaction(
                    lhs=reaction.lhs+":{0:s}:{1:d}".format(reaction.rhs, count),
                    rhs=self._get_uniq_compartment_name(reaction.rhs),
                        evaluation=reaction.scale
                    )
                    self._network.append(rx)
            else:
                self._network.append(reaction)
            #    print(reaction)
            """for compartment in self._compartments:
            reactions_from_compartment = [ x for x in reactions if x.lhs==compartment ]
            for reaction in reactions_from_compartment:
                print(reaction)"""


    def print_network(self):

        print(" - printing internal node connection network")

        for reaction in self._network:
            print(reaction)



        """
        :arg population: A :class:`PopulationModel.`

        :arg population: A :class:`EpidemiologyModel.`

        :arg population: A :class:`SeverityModelModel.`

        :arg population: A :class:`AgeDistribution.`

        :arg population: A :class:`ContainmentModel.`
        """
        """

        self.population = population
        self.epidemiology = epidemiology
        self.severity = severity
        self.age_distribution = age_distribution
        self.containment = containment

        # infer parameters
        self.dt_days = .25
        self.dt = .25 * ms_per_day

        self.avg_infection_rate = epidemiology.r0 / epidemiology.infectious_period

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
        """




    def infection_rate(self, time):
        """
        :arg time: The time in units of miliseconds.

        :returns: The current infection rate, given by the average rate
            :attr:`avg_infection_rate` modulated by the relative importance
            (:attr:`EpidemiologyModel.seasonal_forcing`)
            of the time of year (as a phase), relative to the month of peak
            infection rate (:attr:`EpidemiologyModel.peak_month`).
        """

        from pydemic import date_to_ms
        jan_2020 = date_to_ms((2020, 1, 1))
        peak_day = 30 * self.epidemiology.peak_month + 15
        time_offset = (time - jan_2020) / ms_per_day - peak_day
        phase = 2 * np.pi * time_offset / 365
        return (
            self.avg_infection_rate *
            (1 + self.epidemiology.seasonal_forcing  * np.cos(phase))
        )

    #def step(self, time, state, sample):

    def step(self, time, dt):

        print(" - we are stepping now")

        increments = {}


        for reaction in self._network:

            reaction_rate = reaction.evaluation(time, self)

            print(reaction)

        """

increments = {}
for reaction in reactions:
    dY = poisson(dt * reaction.function(time, state))
    dY_min = state[reaction.lhs].copy()
    for pairs, incr in increment.items():
        if reaction.lhs == pairs[0]:
            dY_min -= incr
    dY = np.minimum(dY_min, dY)
    increment[(reaction.lhs, reaction.rhs)] = dY
    
        """



        pass

        """
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
                new_state.critical[age] += (new_critical[age] - new_stabilized[age]
                                            - new_ICU_dead[age])
                new_state.overflow[age] += (- new_overflow_dead[age]
                                            - new_overflow_stabilized[age])
            elif free_ICU_beds > 0:
                new_overflow = new_critical[age] - free_ICU_beds
                new_state.critical[age] += (free_ICU_beds - new_stabilized[age]
                                            - new_ICU_dead[age])
                new_state.overflow[age] += (new_overflow - new_overflow_dead[age]
                                            - new_overflow_stabilized[age])
                free_ICU_beds = 0
            else:
                new_state.critical[age] += - new_stabilized[age] - new_ICU_dead[age]
                new_state.overflow[age] += (
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
        """

    def initialize_population(self, start_time):
        pass
        """
        ages = np.array(self.age_distribution.counts, dtype='float64')

        init = {key: np.zeros(len(ages))
                for key in SimulationState.expected_kwargs}
        init['time'] = start_time

        fracs = ages / np.sum(ages)
        init['susceptible'] = np.round(fracs * self.population.population_served)

        i_middle = round(ages.shape[0] / 2) + 1  # pylint: disable=E1136
        initial_cases = self.population.suspected_cases_today
        init['susceptible'][i_middle] -= initial_cases
        init['infectious'][i_middle] = 0.3 * initial_cases
        init['exposed'][i_middle] = 0.7 * initial_cases

        initial_state = SimulationState(**init)

        return initial_state
        """

    def __call__(self, start_time, end_time, sample):
        """
        :arg start_time: The initial time, in miliseconds from January 1st, 1970.

        :arg end_time: The final time, in miliseconds from January 1st, 1970.

        :arg sample: The sampling function.

        :returns: The :class:`SimulationResult`.
        """

        time = start_time
        while time < end_time:

            dt = 1. # FIXME: get dt in a reasonable way

            self.step(dt)
            time += dt

        pass

        """
        n_time_steps = int(np.ceil((end_time - start_time) / self.dt)) + 1
        state = self.initialize_population(start_time)
        result = SimulationResult(state, n_time_steps)

        time = start_time
        while time < end_time:
            state = self.step(time, state, sample)
            result.extend(state)
            time += self.dt

        return result
        """




