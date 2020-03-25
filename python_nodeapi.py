import requests

URL = "http://localhost:8080"
PARAMS = { "idek": "really" }

if __name__ == "__main__":

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

  containemnt = {
    "factors": [ 
      1.0, 
      0.9, 
      0.8, 
      0.8, 
      0.8, 
      0.8, 
      0.8, 
      0.8, 
      0.8, 
      0.8
    ]
  }

  r = requests.post(url=URL, params=PARAMS)
  print(r.json())
