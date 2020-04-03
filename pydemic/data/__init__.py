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
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))

_popdata_filename = os.path.join(
    cwd, "../../assets/population.json"
)
_agedata_filename = os.path.join(
    cwd, "../../assets/country_age_distribution.json"
)

with open(_popdata_filename, 'r') as f:
    _populations = json.load(f)
    for el in _populations:
        el["data"]["population_served"] = el["data"].pop("populationServed")
        el["data"]["initial_cases"] = el["data"].pop("suspectedCasesToday")
        el["data"]["ICU_beds"] = el["data"].pop("ICUBeds")
        el["data"]["hospital_beds"] = el["data"].pop("hospitalBeds")
        el["data"]["imports_per_day"] = el["data"].pop("importsPerDay")
    _population_dict = {pop['name']: pop['data'] for pop in _populations}

with open(_agedata_filename, 'r') as f:
    _age_data = json.load(f)


def get_population_model(name):
    data = _population_dict[name]
    from pydemic import PopulationModel
    return PopulationModel(**data)


def get_age_distribution_model(subregion):
    age_data = list(_age_data[subregion].values())
    from pydemic import AgeDistribution
    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    return AgeDistribution(bin_edges=bin_edges, counts=age_data)


def camel_to_snake(name):
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class CaseData:
    def __init__(self, t, y):
        self.t = t
        self.y = y


def dict_to_case_data(data_dict):
    from pydemic import days_from
    t = np.array([days_from(x) for x in data_dict['date']])
    y = {}
    for key, val in data_dict.items():
        y[key] = np.array([0 if x is None else x for x in val])
    return CaseData(t, y)


__all__ = [
    "camel_to_snake",
    "CaseData",
    "dict_to_case_data",
]
