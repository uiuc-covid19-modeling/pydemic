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

from pydemic import Reaction, GammaProcess, Simulation


class SIRModelSimulation(Simulation):
    def __init__(self, avg_infection_rate=10, infectious_rate=5, removal_rate=1):
        self.avg_infection_rate = avg_infection_rate

        reactions = (
            Reaction("susceptible", "infectious",
                     lambda t, y: (self.beta(t, y) * y.susceptible
                                   * y.infectious.sum() / y.sum())),
            Reaction("infectious", "removed",
                     lambda t, y: removal_rate * y.infectious),
        )
        super().__init__(reactions)

    def beta(self, t, y):
        return self.avg_infection_rate


class SEIRModelSimulation(Simulation):
    def __init__(self, avg_infection_rate=10, infectious_rate=5, removal_rate=1):
        self.avg_infection_rate = avg_infection_rate

        reactions = (
            Reaction("susceptible", "exposed",
                     lambda t, y: (self.beta(t, y) * y.susceptible
                                   * y.infectious.sum() / y.sum())),
            Reaction("exposed", "infectious",
                     lambda t, y: infectious_rate * y.exposed),
            Reaction("infectious", "removed",
                     lambda t, y: removal_rate * y.infectious),
        )
        super().__init__(reactions)

    def beta(self, t, y):
        return self.avg_infection_rate


class SEIRHospitalCriticalDeathSimulation(Simulation):
    def __init__(self, avg_infection_rate, *args):
        reactions = (
            Reaction('susceptible', 'exposed',
                     lambda t, y: (avg_infection_rate * y.susceptible
                                   * y.infectious / y.sum())),
            GammaProcess('exposed', 'infectious', shape=3, scale=5),
            Reaction('infectious', 'critical', lambda t, y: 1/5),
            GammaProcess('infectious', 'recovered', shape=4, scale=lambda t, y: 5),
            GammaProcess('infectious', 'dead', shape=3, scale=lambda t, y: 10),
            Reaction('critical', 'dead',
                     lambda t, y: y.critical/y.susceptible/y.sum()),
            Reaction('critical', 'recovered', lambda t, y: 1/7),
        )
        super().__init__(reactions)
