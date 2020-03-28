from scipy.integrate import solve_ivp
import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from pydemic import DemographicClass, Reaction, GammaProcess
from pydemic import CompartmentalModelSimulation

from pydemic.models import SEIRModelSimulation, NeherModelSimulation


def solve_seir(N, beta, a, gamma):
    ## implements simple SEIR model
    def f(_,y):
        S,E,I,R = y
        dydt = [ -beta*S*I/N, beta*S*I/N-a*E, a*E-gamma*I, gamma*I ]
        return np.array(dydt)
    initial_position = [ N-1, 1, 0, 0 ]
    res = solve_ivp(f, [0,10], initial_position)
    return res.t, res.y[0,:], res.y[1,:], res.y[2,:], res.y[3,:]

if __name__ == "__main__":

    ## 
    N = 1.e6
    beta = 12.
    a = 1.
    gamma = 1.

    ## solve simplist SEIR model using scipy.integrate solve_ivp
    t,S,E,I,R = solve_seir(N, beta, a, gamma)


    ## do the same thing with pydemic
    sim = SEIRModelSimulation()


    ## run a Neherlab-like simulation with pydemic
    sim = NeherModelSimulation()


    """
    ## now run the same SEIR model but using pydemic## here we provide an example of a slightly more complicated
    ## SEIR model extension that is meant to demonstrate several
    ## of the more complicated features pydemic supports. 
    reactions = (
        Reaction(
            lhs='susceptible',
            rhs='exposed',
            evaluation=lambda t,y: y['susceptible']*y['infectious']*beta/N
        ),
        GammaProcess(
            lhs='exposed',
            rhs='infectious',
            shape=3,
            scale=lambda t,y: 5.
        ),
        GammaProcess(
            lhs='infectious',
            rhs='recovered',
            shape=2,
            scale=lambda t,y: 8.
        )
    )
    demographics = (
    )

        
    simulation = CompartmentalModelSimulation(reactions, demographics)

    simulation.print_network()
    simulation.step(1.0, 0.1)


    print("\n\n\n")


    # TODO


    ## here we provide an example of a slightly more complicated
    ## SEIR model extension that is meant to demonstrate several
    ## of the more complicated features pydemic supports. 
    compartments = (
        'susceptible',
        'exposed',
        'infectious',
        'critical',
        'recovered',
        'dead'
    )
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
    demographics = (
        DemographicClass(
            name="age", 
            options=[
                ("<10"), 
                ("10-19"), 
                ("20-29"), 
                ("30-39"), 
                ("40-49"), 
                ("50-59"), 
                ("60-69"), 
                ("70+")
            ]
        ),
        DemographicClass(
            name="state",
            options=[
                ("IL", DemographicClass(
                    name="county",
                    options=["Cook", "Champaign"]
                )),
                "NY"
            ]
        )
    )

        

    simulation = CompartmentalModelSimulation(reactions, demographics)

    #simulation.print_network()

    """




    ## plot
    plt.close('all')
    plt.figure(figsize=(8,5))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t,S,'-',label='S')
    ax1.plot(t,E,'-',label='E')
    ax1.plot(t,I,'-',label='I')
    ax1.plot(t,R,'-',label='R')
    ax1.legend()
    ax1.set_xlabel('time')
    ax1.set_ylabel('count')
    ax1.set_yscale('log')
    ax1.set_xlim(-1,11)
    ax1.set_ylim(0.8, 2.*N)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('SEIR_example.png')



    ##

    """
    simulation = CompartmentalModelSimulation(compartments, reactions)

Example of SEIR implementation:

class SEIRModel(CompartmentalModelSimulation):
    def __init__(self, beta(t), t_EI, t_IR):
        reactions = (
            Reaction('susceptible', 'exposed', lambda:... ),
            GammaProcess('exposed', 'infectious', mean(t_EI), std(t_EI)),
            GammaProcess('infectious', 'recovered', mean(t_IR), std(t_IR)),
        )
        
        compartments = {
            'susceptible',
            'exposed',
            'infectious',
            'recovered',
        }
        
        super().__init__(compartments, reactions)
      """




