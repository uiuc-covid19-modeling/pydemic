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

import numpy as np
from datetime import datetime, timedelta
from pydemic import (PopulationModel, AgeDistribution, SeverityModel,
                     EpidemiologyModel, ContainmentModel, date_to_ms)
from neher_port import NeherPortSimulation
from pydemic.models import NeherModelSimulation


def test_neher(plot=False):
    n_age_groups = 9
    start_date = (2020, 3, 1, 0, 0, 0)
    end_date = (2020, 9, 1, 0, 0, 0)
    containment_date = (2020, 3, 20)
    containment_factor = 0.6

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
        isolated=np.array([0., 0., 20., 10., 0., 0., 50., 90., 0.]),
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

    # generate, run, and aggregate results for new pydemic model version
    containment = ContainmentModel(start_date, end_date, is_in_days=True)
    containment.add_sharp_event(containment_date, containment_factor)
    simulation = NeherModelSimulation(
        epidemiology, severity, population.imports_per_day,
        n_age_groups, containment
    )
    compartments = simulation.compartments

    containment = ContainmentModel(start_date, end_date)
    containment.add_sharp_event(containment_date, containment_factor)

    # generate, run, and aggregate results for old pydemic model version
    sim = NeherPortSimulation(population, epidemiology, severity, age_distribution,
                              containment)
    start_time = date_to_ms(start_date)
    end_time = date_to_ms(end_date)
    result = sim(start_time, end_time, lambda x: x)

    og_data = {key: np.sum(result.y[key], axis=-1) for key in compartments}
    dates = [datetime.utcfromtimestamp(x//1000) for x in result.t]

    y0 = simulation.get_initial_population(population, age_distribution)
    new_result = simulation([start_date, end_date], y0, dt=0.25)
    new_dates = [datetime(2020, 1, 1)+timedelta(x) for x in new_result.t]

    for name in compartments:
        test = np.logical_and(result.y[name] > 0, new_result.y[name] > 0)
        relerr = np.abs(1 - result.y[name][test] / new_result.y[name][test])
        print('max err for', name, 'is', np.max(relerr))
        assert np.max(relerr) < .05

    total_people = sum(np.sum(new_result.y[name], axis=-1) for name in compartments)
    total_pop = total_people[0]
    total_err = np.max(np.abs(1 - total_people / total_pop))
    print('total error is', np.max(total_err))
    assert np.max(total_err) < 1.e-13

    if plot:
        import matplotlib as mpl
        mpl.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        # make figure
        fig = plt.figure(figsize=(10, 8))
        ax1 = plt.subplot(1, 1, 1)

        for key in compartments:
            ax1.plot(dates, og_data[key], label=key)
            ax1.plot(new_dates, new_result.y[key].sum(axis=1), '--')

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
        plt.savefig('compare_rewrite.png')


if __name__ == "__main__":
    test_neher(plot=True)
