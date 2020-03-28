from scipy.integrate import solve_ivp
import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from pydemic import DemographicClass, Reaction, GammaProcess
from pydemic import CompartmentalModelSimulation, SeverityModel, EpidemiologyModel

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
    n_age_groups = 9

    # set severity model
    severity = SeverityModel(
        id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
        age_group=np.arange(0., 90., 10),
        isolated=np.zeros(n_age_groups),
        confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
        severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
        critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
        fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
    )

    # set epidemiology model
    epidemiology = EpidemiologyModel(
        r0=2.7,
        incubation_time=5,
        infectious_period=3,
        length_hospital_stay=4,
        length_ICU_stay=14,
        seasonal_forcing=0.,
        peak_month=0,
        overflow_severity=2
    )

    simulation = NeherModelSimulation(epidemiology, severity)

    y0 = {
        'susceptible': np.ones(n_age_groups) * (N-1),
        'exposed': np.ones(n_age_groups),
        'infectious': np.zeros(n_age_groups),
        'recovered': np.zeros(n_age_groups),
        'hospitalized': np.zeros(n_age_groups),
        'critical': np.zeros(n_age_groups),
        'dead': np.zeros(n_age_groups)
    }

    compartments = ('susceptible', 'exposed', 'infectious', 'recovered', 'hospitalized', 'critical', 'dead')

    tspan = (0, 100)
    dt = 1e-3

    result = simulation(tspan, y0, lambda x: x, dt=dt)


    ## plot
    plt.close('all')
    plt.figure(figsize=(8,5))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(result["susceptible"].sum(axis=1),'-',label='S')
    ax1.plot(result["exposed"].sum(axis=1),'-',label='E')
    ax1.plot(result["infectious"].sum(axis=1),'-',label='I')
    ax1.plot(result["recovered"].sum(axis=1),'-',label='R')
    ax1.legend()
    ax1.set_xlabel('time')
    ax1.set_ylabel('count')
    ax1.set_yscale('log')
    #ax1.set_xlim(-1,11)
    ax1.set_ylim(ymin=0.8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Neher_example.png')




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

    """
