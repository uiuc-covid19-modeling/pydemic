import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import date, timedelta
from models.pydemic import PydemicModel
from plutil import plot_model, plot_data, format_axis
import itertools

### example search over parameters
parameters = [ 
  # [ "parameter name", x0, x1, n_samples ] parses to [x0,x1] inclusive with n_samples points
  [ "R0", 2., 9., 15 ],
  [ "start_day", 50, 70, 21 ]  # days since january first
]


## aux functions
def load_deaths(filename):
  # get dates of recorded deaths
  dates = []
  for line in open(filename,'r').readlines():
    year,mon,day = [ int(x) for x in line.split() ]
    dates.append( date(year, mon, day) )
  return dates
def get_date_before(target_date, x):
  # useful for plotting purposes
  return target_date + timedelta(x)

## naive likelihood evaluation and functions
def get_log_likelihood(data, model_mean, model_std):
  ncut = len(model_mean) - len(data)
  return -0.5 * np.sum( np.power(data - model_mean[ncut:], 2.) / np.power(model_std[ncut:], 2.) )
def calculate_likelihood_for_model(model_params, data_y):
  model_y, model_std, dead_mean, dead_abv, dead_bel = get_model_data(model_params)
  logdata_y = np.log(data_y)
  logdata_y[logdata_y < -1.] = -1.  ## also see in get_model_data
  return get_log_likelihood(logdata_y, model_y, model_std)

## for death statistics, generate pseudo-distribution of variance
def get_model_data(model_params):
  ## get model
  pmd = PydemicModel()
  infect_mean, infect_std, dead_mean, dead_std = pmd.get_mean_std(model_params)
  ## restructure data to do likelihood in log space
  data_y = np.log(deaths_points)
  model_y = np.log(dead_mean)
  model_std = np.ones(model_y.shape) * 0.5
  data_y[data_y < -1.] = -1.   ## also see in calculate_likelihood_for_model
  model_y[model_y < -1.] = -1.
  ## synthesize fake uncertainties
  dead_abv = np.exp(model_y + model_std)
  dead_bel = np.exp(model_y - model_std)
  return model_y, model_std, dead_mean, dead_abv, dead_bel

if __name__ == "__main__":

  ## load "true" data
  deaths_data = load_deaths("data/ILdeaths.txt")
  target_date = max(deaths_data)
  data_span = (max(deaths_data)-min(deaths_data)).days + 2
  dates = [ get_date_before(target_date, -x) for x in [ x for x in range(data_span) ][::-1] ]
  deaths_points = np.zeros( data_span )
  for death in deaths_data: deaths_points[np.argmax([death==x for x in dates])] += 1
  deaths_points = np.cumsum(deaths_points)
  dates_points = [ (x-target_date).days for x in dates ]
  # at this point, have "xdata" and "ydata" as 
  #  dates_points   ==  xdata :  number of days before final data point
  #  deaths_points  ==  ydata :  number of cumulative deaths as of end of this day

  ## generate data point
  data = np.zeros([ x[-1] for x in parameters ])

  ## generate 1d sampling points
  full_parameters = []
  for param in parameters:
    full_parameters.append([ param[0] ] + list(np.linspace(param[1],param[2],param[3])))

  ## list all combinations
  params_product = itertools.product(*[ x[1:] for x in full_parameters ])
  params_indices = itertools.product(*[ range(len(x)-1) for x in full_parameters ])
  for (params, index) in zip(params_product, params_indices):
    model_params = {}
    for i in range(len(parameters)):
      model_params[parameters[i][0]] = params[i]
    likelihood = calculate_likelihood_for_model(model_params, deaths_points)
    print(" - likelihood {0:g} for".format(likelihood))
    print("    ",model_params)
    data[index] = likelihood

  ## plot likelihoods over grid for 2d example
  plt.figure(figsize=(6,6))
  ax1 = plt.subplot(1,1,1)
  ax1.pcolormesh(full_parameters[0][1:], full_parameters[1][1:], np.exp(data.T))
  ax1.set_xlabel(full_parameters[0][0])
  ax1.set_ylabel(full_parameters[1][0])
  plt.savefig('imgs/naive_grid.png')

  ## example likelihood calculation
  index = np.unravel_index(np.argmax(data, axis=None), data.shape)
  model_params = { 
    'R0': full_parameters[0][index[0]+1],
    'start_day': full_parameters[1][index[1]+1],
  }
  print(" - parameter input:")
  print(model_params) 
  print(" - likelihood: {0:g}".format(calculate_likelihood_for_model(model_params, deaths_points)))

  ## get some data to plot
  _, _, dead_mean, dead_abv, dead_bel = get_model_data(model_params)

  ## plot example model
  fig = plt.figure(figsize=(10,6))
  ax1 = plt.subplot(1,1,1)
  plot_data(ax1, dates_points, deaths_points, target_date)
  plot_model(ax1, dead_mean, dead_abv, dead_bel, target_date, c='b')
  format_axis(fig, ax1)
  plt.savefig('imgs/naive_example.png')


