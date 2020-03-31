import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from pydemic import (PopulationModel, AgeDistribution, SeverityModel,
                     EpidemiologyModel, ContainmentModel, QuantileLogger)
from pydemic.models import NeherModelSimulation
from pydemic.plot import plot_quantiles, plot_deterministic


def set_numpy_threads(nthreads=1):
    # see also https://codereview.stackexchange.com/questions/206736/better-way-to-set-number-of-threads-used-by-numpy  # noqa
    import os
    try:
        import mkl
        mkl.set_num_threads(nthreads)
        return 0
    except:  # noqa=E722
        pass
    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            import ctypes
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
            return 0
        except:  # noqa=E722
            pass
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)


if __name__ == "__main__":

    # define containment event
    containment_date = (2020, 3, 20)
    containment_factor = 1.0

    # set some base model stats for the neher model
    compartments = ["susceptible", "exposed", "infectious",
                    "recovered", "hospitalized", "critical", "dead"]
    n_age_groups = 9
    start_date = (2020, 3, 1, 0, 0, 0)
    end_date = (2020, 5, 1, 0, 0, 0)
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
    containment = ContainmentModel(start_date, end_date)
    containment.add_sharp_event(containment_date, containment_factor)

    # generate neher model simulation
    simulation = NeherModelSimulation(
        epidemiology, severity, population.imports_per_day,
        n_age_groups, containment
    )
    y0 = simulation.get_initial_population(population, age_distribution)

    # run simulation
    logger = QuantileLogger()
    result = simulation([start_date, end_date], y0, 0.05,
                        samples=1000, stochastic_method='tau_leap', logger=logger)
    deterministic = simulation([start_date, end_date], y0, 0.05)

    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(1, 1, 1)

    plot_quantiles(ax1, result, legend=True)
    plot_deterministic(ax1, deterministic, force_color='k')

    ax1.legend()
    ax1.set_xlabel('time')
    ax1.set_ylabel('count (persons)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/stochastic_neher_model.png')
