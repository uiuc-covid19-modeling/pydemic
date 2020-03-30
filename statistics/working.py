import sys
sys.path.insert(0, '../parameter_estimation')
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta

import models.neher as neher
from pydemic.load import get_case_data
from pydemic.plot import plot_quantiles, plot_deterministic
from plutil import plot_model, plot_data, format_axis


if __name__ == "__main__":

    # load reported data
    cases = get_case_data("USA-Illinois")
    target_date = date(*cases.last_date)

    # define parameter space
    ## TODO

    # choose some model parameters
    model_params = {
      'r0': 3.7,
      'start_day': (date(2020,2,28)-date(2020,1,1)).days,
      'end_day': (date(*cases.last_date)-date(2020,1,1)).days,
      # mitigation date
      # mitigaion factor
      # depth of testing?
    }

    model_params['r0'] = 4.9
    model_params['start_day'] = 61

    import numpy as np
    icases_zero = np.argmax(cases.deaths>0) + 2
    y_data = cases.deaths[icases_zero:]

    likelihood, (y_data, y_model, y_below, y_above) = neher.calculate_likelihood_for_model(model_params, y_data, n_samples=200)
    print(model_params, likelihood)
    print("====")


    ax1 = plt.subplot(2,1,1)
    ax1.plot(y_data, 'xk')
    ax1.plot(y_model)
    ax1.plot(y_above)
    ax1.plot(y_below)

    ax2 = plt.subplot(2,1,2)
    ax2.plot(y_data, 'xk')
    ax2.plot(y_model)
    ax2.plot(y_above)
    ax2.plot(y_below)
    ax2.set_yscale('log')

    plt.savefig('working.png')


