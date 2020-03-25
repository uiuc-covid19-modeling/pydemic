import requests

URL = "http://localhost:8080"
PARAMS = { "idek": "really" }

if __name__ == "__main__":

  r = requests.post(url=URL, params=PARAMS)
  print(r.json())
