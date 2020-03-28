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

from pydemic import Reaction, GammaProcess, CompartmentalModelSimulation


class NeherModelSimulation(CompartmentalModelSimulation):
    """
    Each compartment has n=9 age bins (demographics)
    ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

    Interactions between compartments are according to equations
    [TODO FIXME src/ref] in the pdf
    and are encapsulated in the reactions definition below.

    FIXME TODO Currently does not implement hospital overflow.
    TODO FIXME Currently does not implement seasonal forcing.
    TODO FIXME Currently does not implement containment.

    FIXME TODO Currently does not implement hospital overflow.
    TODO FIXME Currently does not implement seasonal forcing.
    """

    population = 1.e6
    avg_infection_rate = 1.

    def beta(self, t, y):
        return self.avg_infection_rate

    def containment(self, t, y):
        return 1.

    def __init__(self, epidemiology, severity, imports_per_day,
                 population, n_age_groups):
        # TODO FIXME make sure we set population when we pass
        # a new population initial condition
        self.population = population

        # translate from epidemiology/severity models into rates
        dHospital = severity.severe/100. * severity.confirmed/100.
        dCritical = severity.critical/100.
        dFatal = severity.fatal/100.

        isolated_frac = severity.isolated / 100
        exposed_infectious_rate = 1. / epidemiology.incubation_time
        infectious_hospitalized_rate = dHospital / epidemiology.infectious_period
        infectious_recovered_rate = (1.-dHospital) / epidemiology.infectious_period
        hospitalized_discharged_rate = (
            (1 - dCritical) / epidemiology.length_hospital_stay
        )
        hospitalized_critical_rate = dCritical / epidemiology.length_hospital_stay
        critical_hospitalized_rate = (1 - dFatal) / epidemiology.length_ICU_stay
        critical_dead_rate = dFatal / epidemiology.length_ICU_stay

        self.avg_infection_rate = epidemiology.r0 / epidemiology.infectious_period

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
        return 1.
        """

        reactions = (
            Reaction("susceptible", "exposed",
                     lambda t, y: ((1. - isolated_frac) * self.beta(t, y) * self.containment(t, y) * y.susceptible
                                   * y.infectious.sum() / self.population)),
            Reaction("susceptible", "exposed",
                     lambda t, y: imports_per_day / n_age_groups),
            Reaction("exposed", "infectious",
                     lambda t, y: y.exposed * exposed_infectious_rate),
            Reaction("infectious", "hospitalized",
                     lambda t, y: y.infectious * infectious_hospitalized_rate),
            Reaction("infectious", "recovered",
                     lambda t, y: y.infectious * infectious_recovered_rate),
            Reaction("hospitalized", "recovered",
                     lambda t, y: y.hospitalized * hospitalized_discharged_rate),
            Reaction("hospitalized", "critical",
                     lambda t, y: y.hospitalized * hospitalized_critical_rate),
            Reaction("critical", "hospitalized",
                     lambda t, y: y.critical * critical_hospitalized_rate),
            Reaction("critical", "dead",
                     lambda t, y: y.critical * critical_dead_rate)
        )
        super().__init__(reactions)



class ExtendedSimulation(CompartmentalModelSimulation):
    def __init__(self, population, avg_infection_rate, *args):
        reactions = (
            Reaction('susceptible', 'exposed',
                     lambda t, y: (avg_infection_rate * y.susceptible
                                   * y.infectious / population)),
            GammaProcess('exposed', 'infectious', shape=3, scale=5),
            Reaction('infectious', 'critical', lambda t, y: 1/5),
            GammaProcess('infectious', 'recovered', shape=4, scale=lambda t, y: 5),
            GammaProcess('infectious', 'dead', shape=3, scale=lambda t, y: 10),
            Reaction('critical', 'dead',
                     lambda t, y: y.critical/y.susceptible/population),
            Reaction('critical', 'recovered', lambda t, y: 1/7),
        )
        super().__init__(reactions)


class SEIRModelSimulation(CompartmentalModelSimulation):
    def __init__(self, avg_infection_rate=12, infectious_rate=1, removal_rate=1):
        self.avg_infection_rate = avg_infection_rate
        # FIXME: need a reference to y.population
        self.population = 1.e6

        reactions = (
            Reaction("susceptible", "exposed",
                     lambda t, y: (self.beta(t, y) * y.susceptible
                                   * y.infectious.sum() / self.population)),
            Reaction("exposed", "infectious",
                     lambda t, y: y.exposed*infectious_rate),
            Reaction("infectious", "removed",
                     lambda t, y: y.removed*removal_rate),
        )
        super().__init__(reactions)

    def beta(self, t, y):
        return self.avg_infection_rate
