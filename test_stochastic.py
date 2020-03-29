import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from pydemic.models import SEIRModelSimulation


if __name__ == "__main__":

    # run deterministic simulation
    t_span = [0., 10.]
    y0 = {
        'susceptible': 1.e4,
        'exposed': 6.,
        'infectious': 14.,
        'removed': 0 
    }
    simulation = SEIRModelSimulation()
    result = simulation(t_span, y0, lambda x: x)

    # run several stochastic simulations
    stochastic_results = []
    n_sims = 100
    for i in range(n_sims):
        print(" - running stochastic sample (with tau leap) {0:d} of {1:d}".format(i+1,n_sims))
        sresult = simulation(t_span, y0, lambda x:x, stochastic_method='tau_leap', dt=0.001) # 'tau_leap' versus 'direct'
        stochastic_results.append(sresult)

    
    colors = 'r', 'g', 'b', 'm'
    dkeys = ['susceptible', 'exposed', 'infectious', 'removed']

    # make figure
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)

    # plot deterministic solutions
    for i in range(len(dkeys)):
        key = dkeys[i]
        c = colors[i]
        for j in range(len(stochastic_results)):
            s_result = stochastic_results[j]
            ax1.plot(s_result.t, s_result.y[key], 'o', c=c, ms=2, alpha=0.1)
        ax1.plot([], [], '-', lw=2, c=c, label=key)

    # plot deterministic trjectory
    for key in dkeys:
        ax1.plot(result.t, result.y[key], '-', c="#888888", lw=2)

    # plot on y log scale
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=0.8, ymax=2.e4)

    # formatting hints
    ax1.legend(loc='upper right')
    ax1.set_xlabel('time')
    ax1.set_ylabel('count (persons)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("stochastic_examples.png")


    import h5py
    ohfp = h5py.File("stochastic_data.h5", 'w')
    for j in range(len(stochastic_results)):
        ohfp['{0:d}_t'.format(j)] = stochastic_results[j].t
        for key in dkeys:
            idname = '{0:d}_{1:s}'.format(j,key)
            ohfp[idname] = stochastic_results[j].y[key]
    ohfp.close()




    exit()

    ## more complicated SEIR+ model

    # initial conditions
    n_age_groups = 9
    start_date = (2020, 3, 1, 0, 0, 0)
    end_date = (2020, 9, 1, 0, 0, 0)

    population = PopulationModel(
        country='United States of America',
        cases='USA-Illinois',
        population_served=12659682,
        suspected_cases_today=215,
        ICU_beds=1e10,  # originally 1055
        hospital_beds=1e10,  # originally 31649
        imports_per_day=5.0,
    )
    age_distribution = AgeDistribution(
        bin_edges=np.arange(0, 90, 10),
        counts=[39721484, 42332393, 46094077, 44668271, 40348398, 42120077,
                38488173, 24082598, 13147180]
    )
    severity = SeverityModel(
        id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
        age_group=np.arange(0., 90., 10),
        isolated=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
        severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
        critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
        fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
    )
    epidemiology = EpidemiologyModel(
        r0=3.7,
        incubation_time=5,
        infectious_period=3,
        length_hospital_stay=4,
        length_ICU_stay=14,
        seasonal_forcing=0.2,
        peak_month=0,
        overflow_severity=2
    )
    containment = ContainmentModel(start_date, end_date, is_in_days=True)
    containment.add_sharp_event((2020, 3, 20), 0.5)

    ## run Neher-like model simulation
    simulation = NeherModelSimulation(
        epidemiology, severity, population.imports_per_day,
        n_age_groups, containment
    )
    y0 = simulation.get_initial_population(population, age_distribution)
    t0_new = time.time()
    new_result = simulation([start_date, end_date], y0, lambda x: x, dt=0.001)
    t1_new = time.time()
    new_dates = [datetime(2020, 1, 1)+timedelta(x) for x in new_result.t]

    
    dkeys = ['time', 'susceptible', 'exposed', 'infectious', 'hospitalized',
             'critical', 'recovered', 'dead']

    print("new method elapsed:", t1_new-t0_new, "s")

    # make figure
    fig = plt.figure(figsize=(10,8))
    ax1 = plt.subplot(1,1,1)


    for key in dkeys[1:]:
        ax1.plot(new_dates, new_result.y[key].sum(axis=1), label=key)

    # plot on y log scale
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=1)

    # plot x axis as dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.autofmt_xdate()

    # formatting hints
    ax1.legend()
    ax1.set_xlabel('time')
    ax1.set_ylabel('count (persons)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('stochastic_examples.png')
