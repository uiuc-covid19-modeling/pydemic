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

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pydemic.models import SEIRPlusPlusSimulation
from pydemic.distributions import GammaDistribution
from pydemic import MitigationModel

# WARNING: don't set to True unless you want to change the regression test data!
overwrite = False


def test_overwrite_isnt_true(ctx_factory, grid_shape, proc_shape):
    # only runs in pytest
    assert not overwrite


tspan = (50, 125)
t_eval = np.linspace(70, 120, 100)

cases_call = {
    "defaults": dict(
        age_distribution=np.array([1.]),
        total_population=1e6,
        initial_cases=10,
        p_critical=.9,
        p_dead=.9,
        p_positive=.4,
    ),
    "no_ifr": dict(
        age_distribution=np.array([.2, .3, .4, .1]),
        total_population=1e6,
        initial_cases=10,
        ifr=None,
        p_symptomatic=np.array([.1, .3, .5, .9]),
        p_critical=.9,
        p_dead=.9,
        p_positive=np.array([.4, .5, .6, .7]),
    ),
    "log_ifr": dict(
        age_distribution=np.array([.2, .3, .4, .1]),
        total_population=1e6,
        initial_cases=10,
        ifr=.007,
        p_symptomatic=np.array([.1, .3, .5, .9]),
        p_critical=.9,
        p_dead=.9,
        p_positive=np.array([.4, .5, .6, .7]),
    ),
    "change_all_params": dict(
        mitigation=MitigationModel(*tspan, [70, 80], [1., .4]),
        age_distribution=np.array([.2, .3, .4, .1]),
        total_population=1e6,
        initial_cases=9,
        ifr=.008,
        r0=2.5,
        serial_dist=GammaDistribution(4, 3.3),
        seasonal_forcing_amp=.1,
        peak_day=7,
        incubation_dist=GammaDistribution(5.3, 4),
        p_symptomatic=np.array([.2, .4, .5, .8]),
        p_positive=.9 * np.array([.2, .4, .5, .8]),
        hospitalized_dist=GammaDistribution(8, 4),
        p_hospitalized=np.array([.4, .6, .7, .8]),
        discharged_dist=GammaDistribution(7, 3),
        critical_dist=GammaDistribution(4, 1),
        p_critical=np.array([.3, .3, .7, .9]),
        dead_dist=GammaDistribution(4, 3),
        p_dead=np.array([.4, .4, .7, .9]),
        recovered_dist=GammaDistribution(8, 2.5),
        all_dead_dist=GammaDistribution(2., 1.5),
        all_dead_multiplier=1.3,
    )
}

cases_get_model_data = {
    "defaults": dict(
        start_day=tspan[0],
        age_distribution=np.array([1.]),
        total_population=1e6,
        initial_cases=10,
        p_critical=.9,
        p_dead=.9,
        p_positive=.4,
    ),
    "no_ifr": dict(
        start_day=tspan[0],
        age_distribution=np.array([.2, .3, .4, .1]),
        total_population=1e6,
        initial_cases=10,
        ifr=None,
        p_symptomatic=np.array([.1, .3, .5, .9]),
        p_critical=.9,
        p_dead=.9,
        p_positive=np.array([.4, .5, .6, .7]),
    ),
    "log_ifr": dict(
        start_day=tspan[0],
        age_distribution=np.array([.2, .3, .4, .1]),
        total_population=1e6,
        initial_cases=10,
        log_ifr=np.log(.007),
        p_symptomatic=np.array([.1, .3, .5, .9]),
        p_critical=.9,
        p_dead=.9,
        p_positive=np.array([.4, .5, .6, .7]),
    ),
    "change_all_params": dict(
        start_day=tspan[0],
        mitigation_t_0=70,
        mitigation_t_1=80,
        mitigation_factor_0=1.,
        mitigation_factor_1=.4,
        age_distribution=np.array([.2, .3, .4, .1]),
        total_population=1e6,
        initial_cases=9,
        ifr=.008,
        r0=2.5,
        serial_mean=4,
        serial_std=3.3,
        seasonal_forcing_amp=.1,
        peak_day=7,
        incubation_mean=5.3,
        incubation_std=4,
        p_symptomatic=np.array([.2, .4, .5, .8]),
        p_positive=.9 * np.array([.2, .4, .5, .8]),
        hospitalized_mean=8,
        hospitalized_std=4,
        p_hospitalized=np.array([.4, .6, .7, .8]),
        discharged_mean=7,
        discharged_std=3,
        critical_mean=4,
        critical_std=1,
        p_critical=np.array([.3, .3, .7, .9]),
        dead_mean=4,
        dead_std=3,
        p_dead=np.array([.4, .4, .7, .9]),
        recovered_mean=8,
        recovered_std=2.5,
        all_dead_mean=2.0,
        all_dead_std=1.5,
        all_dead_multiplier=1.3,
    )
}

change_prefactors = {
    # "p_symptomatic": .04,
    "p_positive": .234,
    "p_hospitalized": .2523,
    "p_critical": .34,
    "p_dead": .12,
}


def compare_results(a, b):
    diffs = {}
    for col in a.columns:
        err = np.abs(1 - a[col].to_numpy() / b[col].to_numpy())
        max_err = np.nanmax(err)
        avg_err = np.nanmean(err)
        if np.isfinite([max_err, avg_err]).all():
            diffs[col] = (max_err, avg_err)
        else:
            print(col, a[col])

    return diffs


regression_path = Path(__file__).parent / "regression.h5"


@pytest.mark.parametrize("case, params", cases_call.items())
def test_seirpp_call(case, params):
    def get_df(**params):
        total_population = params.get("total_population")
        initial_cases = params.pop("initial_cases")
        age_distribution = params.get("age_distribution")

        sim = SEIRPlusPlusSimulation(**params)

        y0 = {}
        for key in ("susceptible", "infected"):
            y0[key] = np.zeros_like(age_distribution)

        y0["infected"][...] = initial_cases * np.array(age_distribution)
        y0["susceptible"][...] = (
            total_population * np.array(age_distribution) - y0["infected"]
        )

        result = sim(tspan, y0)

        from scipy.interpolate import interp1d
        y = {}
        for key, val in result.y.items():
            y[key] = interp1d(result.t, val.sum(axis=-1), axis=0)(t_eval)

        for key in sim.increment_keys:
            if key in result.y.keys():
                spline = interp1d(result.t, result.y[key].sum(axis=-1), axis=0)
                y[key+"_incr"] = spline(t_eval) - spline(t_eval - 1)

        _t = pd.to_datetime(t_eval, origin="2020-01-01", unit="D")
        return pd.DataFrame(y, index=_t)

    df = get_df(**params)

    max_rtol = 1.e-8
    avg_rtol = 1.e-10

    if overwrite:
        df.to_hdf(regression_path, "seirpp_call/"+case)
    else:
        for group in ("seirpp_call/", "seirpp_get_model_data/"):
            true = pd.read_hdf(regression_path, group+case)
            for key, (max_err, avg_err) in compare_results(true, df).items():
                assert (max_err < max_rtol and avg_err < avg_rtol), \
                    "case %s: %s failed against %s, %s, %s" % \
                    (case, key, group, max_err, avg_err)

    case2 = case+"_changed_prefactors"
    if "ifr" in params:
        params["ifr"] = None
    if "log_ifr" in params:
        params.pop("log_ifr")
    for key, val in change_prefactors.items():
        if key in params:
            params[key] *= val
        else:
            params[key] = val

    df = get_df(**params)
    if overwrite:
        df.to_hdf(regression_path, "seirpp_call/"+case2)
    else:
        for group in ("seirpp_call/", "seirpp_get_model_data/"):
            true = pd.read_hdf(regression_path, group+case2)
            for key, (max_err, avg_err) in compare_results(true, df).items():
                assert (max_err < max_rtol and avg_err < avg_rtol), \
                    "case %s: %s failed against %s, %s, %s" % \
                    (case2, key, group, max_err, avg_err)


@pytest.mark.parametrize("case, params", cases_get_model_data.items())
def test_seirpp_get_model_data(case, params):
    df = SEIRPlusPlusSimulation.get_model_data(t_eval, **params)

    max_rtol = 1.e-8
    avg_rtol = 1.e-10

    if overwrite:
        df.to_hdf(regression_path, "seirpp_get_model_data/"+case)
    else:
        for group in ("seirpp_call/", "seirpp_get_model_data/"):
            true = pd.read_hdf(regression_path, group+case)
            for key, (max_err, avg_err) in compare_results(true, df).items():
                assert (max_err < max_rtol and avg_err < avg_rtol), \
                    "case %s: %s failed against %s, %s, %s" % \
                    (case, key, group, max_err, avg_err)

    case2 = case+"_changed_prefactors"
    if "ifr" in params:
        params["ifr"] = None
    if "log_ifr" in params:
        params.pop("log_ifr")
    check_ps = {}
    for key, val in change_prefactors.items():
        check_ps[key] = np.copy(params.get(key, 1))
        params[key+"_prefactor"] = val

    df = SEIRPlusPlusSimulation.get_model_data(t_eval, **params)

    # check that p_* didn't change
    for key, val in change_prefactors.items():
        assert np.allclose(check_ps[key], params.get(key, 1), rtol=1.e-13, atol=0)

    if overwrite:
        df.to_hdf(regression_path, "seirpp_get_model_data/"+case2)
    else:
        for group in ("seirpp_call/", "seirpp_get_model_data/"):
            true = pd.read_hdf(regression_path, group+case2)
            for key, (max_err, avg_err) in compare_results(true, df).items():
                assert (max_err < max_rtol and avg_err < avg_rtol), \
                    "case %s: %s failed against %s, %s, %s" % \
                    (case2, key, group, max_err, avg_err)


if __name__ == "__main__":
    for case, params in cases_call.items():
        test_seirpp_call(case, params)

    for case, params in cases_get_model_data.items():
        test_seirpp_get_model_data(case, params)
