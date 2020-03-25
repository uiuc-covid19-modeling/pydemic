import requests
import json
import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from pydemic import PopulationModel, AgeDistribution

import pydemic.population_models as pm

URL = "http://localhost:8081"

if __name__ == "__main__":

  ## example parameter sets here
  population = {
    # total demographic information and labels
    "country": "Switzerland",
    "cases": "Switzerland",
    "populationServed": 8600000,
    # reported medical facilities
    "hospitalBeds": 30799,
    "ICUBeds": 1400,
    # estimated infectivity model parameters
    "suspectedCasesToday": 1148,
    "importsPerDay": 4.0,
    # granular population statistics
    "populationsByDecade": [
      4994996,
      5733447,
      6103437,
      6998434,
      9022004,
      9567192,
      7484860,
      6028907,
      4528548
    ]
  }

  simulation = {
    "start": [ 2020, 3, 1, 0, 0, 0 ],
    "end": [ 2020, 9, 1, 0, 0, 0 ]
  }

  epidemiology = {
      "r0": 3.7,
      "incubationTime": 5,
      "infectiousPeriod": 3,
      "lengthHospitalStay": 4,
      "lengthICUStay": 14,
      "seasonalForcing": 0.2,
      "peakMonth": 0,
      "overflowSeverity": 2
  }

  mitigation_factor = 0.8
  containment = {
    "times": [
      [ 2020, 3, 1, 0, 0, 0 ],
      [ 2020, 3, 14, 0, 0, 0 ],
      [ 2020, 3, 15, 0, 0, 0 ],
      [ 2020, 9, 1, 0, 0, 0 ]
    ],
    "factors": [
      1.0,
      1.0,
      mitigation_factor,
      mitigation_factor
    ]
  }

  ## automatically load population from json data
  POPULATION_NAME = "USA-Illinois"
  AGE_DATA_NAME = "United States of America"
  age_data = pm._age_data[AGE_DATA_NAME]
  population = [ x for x in pm._populations if x['name']==POPULATION_NAME ][0]['data']
  population['populationsByDecade'] = [ age_data[key] for key in age_data.keys() ]

  # example of kludge on how to load population data
  from pydemic.load import get_country_population_model

  population = get_country_population_model(POPULATION_NAME, AGE_DATA_NAME)


  ## generate and POST request to javascript api
  body = { "simulation":simulation, "population":population, "containment":containment, "epidemiology":epidemiology }
  r = requests.post(url=URL, data=json.dumps(body))
  data = r.json()
  dkeys = [ 'times', 'suspectible', 'exposed', 'infectious', 'recovered', 'hospitalized', 'critical', 'overflow', 'discharged', 'intensive', 'dead' ]
  dates = [ datetime.utcfromtimestamp(x//1000) for x in data['times'] ]

  ## make figure
  fig = plt.figure(figsize=(10,6))
  ax1 = plt.subplot(1,1,1)

  for key in dkeys[1:]:
    ax1.plot(dates, data[key], label=key)

  # plot on y log scale
  ax1.set_yscale('log')
  ax1.set_ylim(ymin=1)

  # plot x axis as dates
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
  #ax1.set_xlim(dates[0],dates[-1])
  fig.autofmt_xdate()


  # formatting hints
  ax1.legend()
  ax1.set_xlabel('time')
  ax1.set_ylabel('count (persons)')

  plt.savefig('example.png')





