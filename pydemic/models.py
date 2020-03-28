import numpy as np

from pydemic import Reaction
from pydemic import CompartmentalModelSimulation




class NeherModelSimulation(CompartmentalModelSimulation):
    """
        Each compartment has n=9 age bins (demographics)
            [ "0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+" ]

        Interactions between compartments are according to equations [TODO FIXME src/ref] in the pdf
        and are encapsulated in the reactions definition below.

        FIXME TODO Currently does not implement hospital overflow.
        TODO FIXME Currently does not implement seasonal forcing.

    """

    population = 1.e6
    avg_infection_rate = 1.

    def beta(self, t, y):
        return self.avg_infection_rate

    def __init__(self, epidemiology, severity):

        ## TODO FIXME make sure we set population when we pass
        ##            a new population initial condition
        self.population = 1.e6

        ## translate from epidemiology/severity models into rates
        dHospital = severity.severe/100. * severity.critical/100.
        dCritical = severity.critical/100.
        dFatal = severity.fatal/100.

        exposed_infectious_rate = 1. / epidemiology.incubation_time
        infectious_hospitalized_rate = dHospital / epidemiology.infectious_period
        infectious_recovered_rate = (1.-dHospital) / epidemiology.infectious_period
        hospitalized_discharged_rate = (1 - dCritical) / epidemiology.length_hospital_stay
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

        ## define reactions given rates above
        reactions = (
            Reaction(
                "susceptible", 
                "exposed",
                lambda t,y: self.beta(t,y)*y["susceptible"]*y["infectious"].sum()/self.population
            ),
            Reaction(
                lhs="exposed", 
                rhs="infectious",
                evaluator=lambda t,y: y["exposed"] * exposed_infectious_rate
            ),
            Reaction(
                lhs="infectious",
                rhs="hospitalized",
                evaluator=lambda t,y: y["infectious"] * infectious_hospitalized_rate
            ),
            Reaction(
                lhs="infectious",
                rhs="recovered",
                evaluator=lambda t,y: y["infectious"] * infectious_recovered_rate
            ),
            Reaction(
                lhs="hospitalized",
                rhs="recovered",
                evaluator=lambda t,y: y["hospitalized"] * hospitalized_discharged_rate
            ),
            Reaction(
                lhs="hospitalized",
                rhs="critical",
                evaluator=lambda t,y: y["hospitalized"] * hospitalized_critical_rate
            ),
            Reaction(
                lhs="critical",
                rhs="hospitalized",
                evaluator=lambda t,y: y["critical"] * critical_hospitalized_rate
            ),
            Reaction(
                lhs="critical",
                rhs="dead",
                evaluator=lambda t,y: y["critical"] * critical_dead_rate
            )
        )

        super().__init__(reactions)








class ExtendedSimulation(CompartmentalModelSimulation):

    def __init__(self, *args):

        reactions = (
          Reaction(
              lhs='susceptible',
              rhs='exposed',
              evaluation=lambda t,y: y.susceptible*y.infectious*beta/N
          ),
          GammaProcess(
              lhs='exposed',
              rhs='infectious',
              shape=3,
              scale=5.
          ),
          Reaction(
              lhs='infectious',
              rhs='critical',
              evaluation=lambda t,y: 1./5.
          ),
          GammaProcess(
              lhs='infectious',
              rhs='recovered',
              shape=4,
              scale=lambda t,y: 5.
          ),
          GammaProcess(
              lhs='infectious',
              rhs='dead',
              shape=3,
              scale=lambda t,y: 10.
          ),
          Reaction(
              lhs='critical',
              rhs='dead',
              evaluation=lambda t,y: y.critical/y.susceptible/N
          ),
          Reaction(
              lhs='critical',
              rhs='recovered',
              evaluation=lambda t,y: 1./7.
          )
      )


class SEIRModelSimulation(CompartmentalModelSimulation):

    def __init__(self, beta=12., a=1., gamma=1.):

        ## FIXME: need a reference to y.population
        population = 1.e6

        reactions = (
            Reaction(
                lhs="susceptible", 
                rhs="exposed",
                evaluator=lambda t,y: beta*y["susceptible"]*y["infectious"]/population
            ),
            Reaction(
                lhs="exposed", 
                rhs="infectious",
                evaluator=lambda t,y: y["exposed"]*a
            ),
            Reaction(
                lhs="infectious", 
                rhs="removed",
                evaluator=lambda t,y: y["infectious"]*gamma
            )
        )

        super().__init__(reactions)




