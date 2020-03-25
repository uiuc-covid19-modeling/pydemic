import json
import os

_popdata_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src/assets/data/population.json")
_agedata_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src/assets/data/country_age_distribution.json")

with open(_popdata_filename, 'r') as f:
    _populations = json.load(f)

with open(_agedata_filename, 'r') as f:
    _age_data = json.load(f)


def get_valid_pops():
    return [ x['name'] for x in _populations ]

def get_valid_ages():
    return _age_data.keys()

def get(country, subregion):
    # FIXME: get subregiondata
    # FIXME: assert subregion belongs to country?
    country_data = [ x for x in _populations if x['name'] == country ][0]
    age_data = _age_data[subregion]

    from common import AttrDict
    return AttrDict(**country_data, **age_data)


if __name__ == "__main__":

  print("valid populations are:")
  print( get_valid_pops() )
  print("valid age distributions are:")
  print( get_valid_ages() )

  popdata = get("USA-Illinois", "Ukraine")

  print( popdata )
