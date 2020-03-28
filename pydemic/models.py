

from pydemic import DemographicClass, Reaction
from pydemic import CompartmentalModelSimulation


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

        demographics = ()

        super().__init__(reactions, demographics)




