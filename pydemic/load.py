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
from pydemic import days_from

cwd = os.path.dirname(os.path.abspath(__file__))

_popdata_filename = os.path.join(
    cwd, "../assets/population.json"
)
_agedata_filename = os.path.join(
    cwd, "../assets/country_age_distribution.json"
)
_casedata_filename = os.path.join(
    cwd, "../assets/case_counts.json"
)

with open(_popdata_filename, 'r') as f:
    _populations = json.load(f)
    for el in _populations:
        el["data"]["population_served"] = el["data"].pop("populationServed")
        el["data"]["suspected_cases_today"] = el["data"].pop("suspectedCasesToday")
        el["data"]["ICU_beds"] = el["data"].pop("ICUBeds")
        el["data"]["hospital_beds"] = el["data"].pop("hospitalBeds")
        el["data"]["imports_per_day"] = el["data"].pop("importsPerDay")
    _population_dict = {pop['name']: pop['data'] for pop in _populations}

with open(_agedata_filename, 'r') as f:
    _age_data = json.load(f)

with open(_casedata_filename, 'r') as f:
    _case_data = json.load(f)


def get_valid_pops():
    return _population_dict.keys()


def get_valid_ages():
    return _age_data.keys()


def get_valid_cases():
    return _case_data.keys()


def get_country_population_model(country, initial_cases=10., imports_per_day=1.1,
                                 ICU_beds=1.e10, hospital_beds=1.e10):
    country_data = _population_dict[country]
    from pydemic import PopulationModel
    population = PopulationModel(**country_data)
    population['suspected_cases_today'] = initial_cases
    population['imports_per_day'] = imports_per_day
    population['ICU_beds'] = ICU_beds
    population['hospital_beds'] = hospital_beds
    return PopulationModel(**country_data)


def get_age_distribution_model(subregion):
    age_data = list(_age_data[subregion].values())
    from pydemic import AgeDistribution
    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    return AgeDistribution(bin_edges=bin_edges, counts=age_data)


def get_case_data(subregion):
    data_series = _case_data[subregion].copy()
    from pydemic import CaseData

    data_dict = {}
    for key in CaseData.expected_kwargs:
        if key not in ('dates', 'last_date'):
            # replace None with 0
            dat = [d[key] or 0 for d in data_series]
            data_dict[key] = np.array(dat)

    def to_tuple(date):
        return tuple(map(int, date.split('-')))

    date_tuples = [to_tuple(d['time']) for d in data_series]

    dates = [days_from(x) for x in date_tuples]
    data_dict['dates'] = dates
    data_dict['last_date'] = date_tuples[-1]

    return CaseData(**data_dict)


if __name__ == "__main__":
    print("valid populations are:")
    print(get_valid_pops())
    print("valid age distributions are:")
    print(get_valid_ages())
    print("valid case records are:")
    print(get_valid_cases())

    print("\n\n")
    popdata = get_country_population_model("USA-Illinois")
    for key, val in popdata.__dict__.items():
        print(key, val)
    print(get_case_data('USA-Illinois'))
