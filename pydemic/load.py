__copyright__ = """
Copyright (C) 2020 George N Wong
Copyright (C) 2020 Zachary J Weiner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import json
import os

_popdata_filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../src/assets/data/population.json"
)
_agedata_filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../src/assets/data/country_age_distribution.json"
)

with open(_popdata_filename, 'r') as f:
    _populations = json.load(f)
    _population_dict = {pop['name']: pop['data'] for pop in _populations}

with open(_agedata_filename, 'r') as f:
    _age_data = json.load(f)


def get_valid_pops():
    return _population_dict.keys()


def get_valid_ages():
    return _age_data.keys()


def get_country_population_model(country, subregion):
    country_data = _population_dict[country]
    age_data = list(_age_data[subregion].values())

    from pydemic import PopulationModel
    return PopulationModel(**country_data, populationsByDecade=age_data)


if __name__ == "__main__":
  print("valid populations are:")
  print(get_valid_pops())
  print("valid age distributions are:")
  print(get_valid_ages())

  popdata = get_country_population_model("USA-Illinois", "Ukraine")

  for key, val in popdata.__dict__.items():
      print(key, val)
