from scipy.integrate import solve_ivp
import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from pydemic import DemographicClass, Reaction, GammaProcess
from pydemic import CompartmentalModelSimulation


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

    ## now run the same SEIR model but using pydemic## here we provide an example of a slightly more complicated
    ## SEIR model extension that is meant to demonstrate several
    ## of the more complicated features pydemic supports.
    reactions = (
        Reaction('susceptible', 'exposed',
                 lambda t,y: y['susceptible']*y['infectious']*beta/N),
        GammaProcess('exposed', 'infectious', shape=1,
                     scale=lambda t, y: 5.),
        GammaProcess(
            lhs='infectious',
            rhs='recovered',
            shape=1,
            scale=lambda t, y: 8.
        )
    )


    simulation = CompartmentalModelSimulation(reactions)

    y0 = {
        'susceptible': np.array(N-1),
        'exposed': np.array(1),
        'infectious': np.array(0),
        'recovered': np.array(0),
    }

    simulation.print_network()
    result = simulation((0, 10), y0, lambda x: x)
    print(result)

    fig, ax = plt.subplots()
    ax.plot(t,S,'-',label='S')
    ax.plot(t,E,'-',label='E')
    ax.plot(t,I,'-',label='I')
    ax.plot(t,R,'-',label='R')

    t = np.linspace(0, 10, int(10/.01 + 2))
    ax.plot(t, result['susceptible'], '--', label='S2')
    ax.plot(t, result['exposed'], '--', label='E2')
    ax.plot(t, result['infectious'], '--', label='I2')
    ax.plot(t, result['recovered'], '--', label='R2')

    fig.savefig('SEIR_example.png')


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




