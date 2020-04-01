import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from models.pydemic import PydemicModel
from plutil import plot_model, plot_data, format_axis
from pydemic.load import get_case_data
# from pydemic import CaseData

LOG_ZERO_VALUE = -2.


def get_log_likelihood(data, model_mean, model_std):
    ncut = len(model_mean) - len(data)
    nonzero = np.argmax(data > 0)
    return -0.5 * np.sum(
        np.power(data[nonzero:] - model_mean[nonzero+ncut:], 2.)
        / np.power(model_std[nonzero+ncut:], 2.)
    )


def calculate_likelihood_for_model(model_params, data_y, zero_value=LOG_ZERO_VALUE):
    model_y, model_std, dead_mean, dead_abv, dead_bel = get_model_data(model_params)
    logdata_y = np.log(data_y)
    logdata_y[logdata_y < zero_value] = zero_value  # also see in get_model_data
    return get_log_likelihood(logdata_y, model_y, model_std)


def get_model_data(model_params, zero_value=LOG_ZERO_VALUE):
    # get model
    pmd = PydemicModel()
    infect_mean, infect_std, dead_mean, dead_std = pmd.get_mean_std(model_params)
    # restructure data to do likelihood in log space
    data_y = np.log(cases.deaths)
    # also see in calculate_likelihood_for_model
    data_y[data_y < zero_value] = zero_value
    model_y = np.log(dead_mean)
    model_y[model_y < zero_value] = zero_value
    model_std = np.ones(model_y.shape)*0.2
    correction = (np.log(np.exp(model_y)+1)-np.log(np.exp(model_y)))
    model_std += correction
    # synthesize fake uncertainties
    dead_abv = np.exp(model_y + model_std)
    dead_bel = np.exp(model_y - model_std)
    return model_y, model_std, dead_mean, dead_abv, dead_bel


if __name__ == "__main__":

    # choose some model parameters
    model_params = {
        'R0': 3.7,
        'start_day': 58
    }

    # load reported data
    cases = get_case_data("USA-Illinois")
    target_date = date(*cases.last_date)

    # get model and likelihood
    _, _, dead_mean, dead_abv, dead_bel = get_model_data(model_params)
    likelihood = calculate_likelihood_for_model(model_params, cases.deaths)

    # plot example model
    fig = plt.figure(figsize=(6, 4))
    ax1 = plt.subplot(1, 1, 1)
    plot_model(ax1, dead_mean, dead_abv, dead_bel, target_date, c='b')
    plot_data(ax1, cases.dates, cases.deaths, target_date)
    format_axis(fig, ax1)
    plt.suptitle("likelihood = {0:.2g}".format(likelihood))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('imgs/manual_example.png')
