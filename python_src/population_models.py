import json

with open('../src/assets/data/population.json', 'w') as f:
    pops = json.load(f)

with open('../src/assets/data/country_age_distribution.json', 'w') as f:
    age_data = json.load(f)


def get(country, subregion):
    # FIXME: get subregiondata
    # FIXME: assert subregion belongs to country?
    country_data = pops[country]
    age_data = age_data[country]

    from common import AttrDict
    return AttrDict(**country_data, **age_data)
