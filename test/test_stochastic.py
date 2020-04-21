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

from pydemic.models import SEIRModelSimulation


def test_direct_vs_tau_leap(plot=False):
    # 'direct' versus 'tau_leap'
    stochastic_method = 'direct'
    timestep = 0.01

    # run deterministic simulation
    t_span = [0., 10.]
    y0 = {
        'susceptible': 1.e3,
        'exposed': 6.,
        'infectious': 14.,
        'removed': 0
    }
    simulation = SEIRModelSimulation()
    result = simulation(t_span, y0, dt=timestep)

    # run several stochastic simulations
    stochastic_results = []
    n_sims = 100
    for i in range(n_sims):
        print(" - running tau leap sample {0:d} of {1:d}".format(i+1, n_sims))
        sresult = simulation(t_span, y0, dt=timestep,
                             stochastic_method=stochastic_method)
        stochastic_results.append(sresult)

    if plot:
        import matplotlib.pyplot as plt

        colors = 'r', 'g', 'b', 'm'
        dkeys = ['susceptible', 'exposed', 'infectious', 'removed']

        # make figure
        fig = plt.figure(figsize=(10, 8))  # noqa
        ax1 = plt.subplot(1, 1, 1)

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
        ax1.set_xlim(xmin=-1, xmax=11)
        ax1.set_ylim(ymin=0.8, ymax=2.e3)

        # formatting hints
        ax1.legend(loc='upper right')
        ax1.set_xlabel('time')
        ax1.set_ylabel('count (persons)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("stochastic_examples_{0:s}_{1:f}.png".format(
            stochastic_method, 1000*timestep
        ))


if __name__ == "__main__":
    test_direct_vs_tau_leap(plot=True)
