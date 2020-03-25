import requests
import json
import matplotlib as mpl ; mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

URL = "http://localhost:8080"

if __name__ == "__main__":

  mitigation_factor = 0.4

  simulation = {
    "start": [ 2020, 3, 1, 0, 0, 0 ],
    "end": [ 2020, 9, 1, 0, 0, 0 ]
  }
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
    "importedPerDay": 4.0,
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
  
  body = { "simulation":simulation, "population":population, "containment":containment }

  r = requests.post(url=URL, data=json.dumps(body))
  data = r.json()
  dkeys = [ 'times', 'suspectible', 'exposed', 'infectious', 'recovered', 'hospitalized', 'critical', 'overflow', 'discharged', 'intensive', 'dead' ]
  dates = [ datetime.utcfromtimestamp(x//1000) for x in data['times'] ]

  # make figure
  fig = plt.figure(figsize=(10,6))
  ax1 = plt.subplot(1,1,1)

  for key in dkeys[1:]:
    ax1.plot(dates, data[key])
  
  # plot on y log scale
  ax1.set_yscale('log')
  ax1.set_ylim(ymin=1)

  # plot x axis as dates
  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
  #ax1.set_xlim(dates[0],dates[-1])
  fig.autofmt_xdate()

  plt.savefig('example.png')





