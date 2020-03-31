import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from datetime import date

import models.neher as neher
from pydemic.load import get_case_data
from pydemic.plot import plot_quantiles
from plutil import plot_data, format_axis


if __name__ == "__main__":
    # load reported data
    cases = get_case_data("USA-Illinois")
    target_date = date(*cases.last_date)

    # define parameter space
    # TODO

    # choose some model parameters
    model_params = {
        'r0': 3.7,
        'start_day': (date(2020, 2, 28)-date(2020, 1, 1)).days,
        'end_day': (date(*cases.last_date)-date(2020, 1, 1)).days,
        # mitigation date
        # mitigaion factor
        # depth of testing?
    }

    model_params['r0'] = 3.1
    model_params['start_day'] = 56

    likelihood = neher.calculate_likelihood_for_model(model_params, cases.deaths)

    # start_date = datetime(2020,1,1) + timedelta(model_params['start_day'])
    # model_dates = [ start_date+timedelta(x) for x in range(len(test)) ]
    # print(model_dates)

    params_string = " ".join(["{0:s}={1:g}".format(
        x, model_params[x]) for x in model_params])

    # plot best-fit model
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(1, 1, 1)

    quantiles_result = neher.get_model_result(model_params, 0.05, n_samples=1000)
    plot_quantiles(ax1, quantiles_result)
    plot_data(ax1, cases.dates, cases.deaths, target_date)
    format_axis(fig, ax1)

    plt.suptitle("likelihood={0:.2g} {1:s}".format(likelihood, params_string))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/neher_emcee.png')
