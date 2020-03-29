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
        sresult = simulation(t_span, y0, lambda x:x, stochastic_method='tau_leap', dt=0.01) # 'tau_leap' versus 'direct'
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


