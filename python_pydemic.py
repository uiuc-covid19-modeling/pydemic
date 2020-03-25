import requests
import json
import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from pydemic import PopulationModel, AgeDistribution

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
#   import pydemic.population_models as pm
#   age_data = pm._age_data[AGE_DATA_NAME]
#   population = [ x for x in pm._populations if x['name']==POPULATION_NAME ][0]['data']
#   population['populationsByDecade'] = [ age_data[key] for key in age_data.keys() ]

  # example of kludge on how to load population data
  from pydemic.load import get_country_population_model, get_age_distribution_model

  population = get_country_population_model(POPULATION_NAME)
  agedistribution = get_age_distribution_model(AGE_DATA_NAME)



  ## generate and POST request to javascript api
  body = { "simulation":simulation, "population":population, "containment":containment, "epidemiology":epidemiology, "agedistribution":agedistribution }
  #data = pydemic.run(body)

  dkeys = [ 'times', 'suspectible', 'exposed', 'infectious', 'recovered', 'hospitalized', 'critical', 'overflow', 'discharged', 'intensive', 'dead' ]
  dates = [ datetime.utcfromtimestamp(x//1000) for x in data['times'] ]


  """
  r = requests.post(url=URL, data=json.dumps(body))
  data = r.json()
  dkeys = [ 'times', 'suspectible', 'exposed', 'infectious', 'recovered', 'hospitalized', 'critical', 'overflow', 'discharged', 'intensive', 'dead' ]
  dates = [ datetime.utcfromtimestamp(x//1000) for x in data['times'] ]
  """



  ## make figure
  fig = plt.figure(figsize=(10,8))
  gs = fig.add_gridspec(3, 1)
  ax1 = fig.add_subplot(gs[:2,0])
  ax2 = fig.add_subplot(gs[2,0], sharex=ax1)

  for key in dkeys[1:]:
    ax1.plot(dates, data[key], label=key)
  
  # plot nice hint data
  ax1.axhline(y=population['hospitalBeds'],ls=':',c='#999999')
  ax1.axhline(y=population['ICUBeds'],ls=':',c='#999999')

  # plot containment
  mitigation_dates = [ datetime(*x[:-2]) for x in containment["times"] ]
  ax2.plot(mitigation_dates, containment["factors"], 'ok-', lw=2)
  ax2.set_ylim(0,1.2)

  # plot on y log scale
  ax1.set_yscale('log')
  ax1.set_ylim(ymin=1)

  # plot x axis as dates
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
  #ax1.set_xlim(dates[0],dates[-1])
  fig.autofmt_xdate()


  # formatting hints
  ax1.legend()
  ax2.set_xlabel('time')
  ax2.set_ylabel('mitigation factor')
  ax1.set_ylabel('count (persons)')

  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.savefig('example.png')





