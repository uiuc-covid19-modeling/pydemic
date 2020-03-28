import numpy as np

from pydemic import DemographicClass, Reaction
from pydemic import CompartmentalModelSimulation




class NeherModelSimulation(CompartmentalModelSimulation):
    """
        Each compartment has n=9 age bins (demographics)
            [ "0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+" ]


        Interactions between compartments are according to equations [src/ref] in the pdf
        and are encapsulated in the reactions definition below

    """

    population = 1.e6

    # parameters of the model (hardcoded for now)
    t_l = 1.
    t_i = 1.
    t_c = 1.
    t_h = 1.

    m = np.ones(9)  # going from infectious to recovered
    c = np.ones(9)  # going from hospitalized to go to critical
    f = np.ones(9)  # going from critical to dead


    def __init__(self):

        ## FIXME: TODO?

        reactions = (
            Reaction(
                "susceptible", 
                "exposed",
                lambda t,y: y.beta()*y["susceptible"]*y["infectious"].sum()/population
            ),
            Reaction(
                lhs="exposed", 
                rhs="infectious",
                evaluator=lambda t,y: y["exposed"]/t_l
            ),
            Reaction(
                lhs="infectious",
                rhs="hospitalized",
                evaluator=lambda t,y: m*y["infectious"]/t_i
            ),
            Reaction(
                lhs="infectious",
                rhs="recovered",
                evaluator=lambda t,y: (1.-m)*y["infectious"]/t_i
            ),
            Reaction(
                lhs="hospitalized",
                rhs="recovered",
                evaluator=lambda t,y: (1.-c)*y["hospitalized"]/t_h
            ),
            Reaction(
                lhs="hospitalized",
                rhs="critical",
                evaluator=lambda t,y: c*y["hospitalized"]/t_h
            ),
            Reaction(
                lhs="critical",
                rhs="hospitalized",
                evaluator=lambda t,y: (1.-f)*y["critical"]/t_c
            ),
            Reaction(
                lhs="critical",
                rhs="dead",
                evaluator=lambda t,y: f*y["critical"]/t_c
            )
        )

        super().__init__(reactions)


    def beta(self):

        return 1.



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




