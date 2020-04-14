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

import os
import json
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))


def camel_to_snake(name):
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class CaseData:
    def __init__(self, t, y):
        self.t = t
        self.y = y

    def copy(self):
        y_copy = {key: val.copy() for key, val in self.y.items()}
        return CaseData(self.t.copy(), y_copy)

    def __repr__(self):
        text = "{0:s} with\n".format(str(type(self)))
        text += "  - t from {0:g} to {1:g}\n".format(self.t[0], self.t[-1])
        for key in self.y:
            text += "  - {0:s} {1:s}\n".format(key,
                                               str(self.y[key].shape))
        return text[:-1]


def dict_to_case_data(data_dict):
    from pydemic import days_from
    t = np.array([days_from(x) for x in data_dict['date']])
    y = {}
    for key, val in data_dict.items():
        y[key] = np.array([0 if x is None else x for x in val])
    return CaseData(t, y)


class DataParser:
    _casedata_filename = None
    data_url = None
    region_specifier = 'state'
    _popdata_filename = os.path.join(
        cwd, "../../assets/population.json"
    )
    _agedata_filename = os.path.join(
        cwd, "../../assets/country_age_distribution.json"
    )
    translation = {}

    def __init__(self):
        import os.path
        if not os.path.isfile(self._casedata_filename):
            self.scrape_case_data()

    def translate(self, key):
        return self.translation.get(key, key)

    def convert_to_date(self, num):
        year = num // 10000
        month = (num - 10000 * year) // 100
        day = (num - 10000 * year - 100 * month)
        return (year, month, day)

    def parse_case_data(self):
        import requests
        r = requests.get(self.data_url)
        all_data = json.loads(r.text)
        r.close()

        all_data = [
            {self.translate(camel_to_snake(key)): val for key, val in x.items()}
            for x in all_data
        ]

        regions = set([x[self.region_specifier] for x in all_data])
        data_fields = all_data[0].keys()

        region_data = {}
        for region in regions:
            region_data[region] = []

        for data_point in all_data:
            region = data_point.pop(self.region_specifier)
            region_data[region].append(data_point)

        region_data_series = {}
        for region, data in region_data.items():
            sorted_data = sorted(data, key=lambda x: x['date'])
            for dp in sorted_data:
                dp['date'] = self.convert_to_date(dp['date'])

            data_series = {}
            for key in data_fields:
                data_series[key] = [x.get(key) for x in sorted_data]

            region_data_series[region] = data_series

        return region_data_series

    def scrape_case_data(self):
        region_data_series = self.parse_case_data()

        with open(self._casedata_filename, 'w') as f:
            json.dump(region_data_series, f)

    def get_case_data(self, region):
        with open(self._casedata_filename, 'r') as f:
            case_data = json.load(f)

        return dict_to_case_data(case_data[region])

    def get_population(self, name):
        with open(self._popdata_filename, 'r') as f:
            populations = json.load(f)
        return populations[name]['populationServed']

    def get_age_distribution_model(self, name):
        with open(self._agedata_filename, 'r') as f:
            age_data = json.load(f)

        age_data = list(age_data[name].values())
        from pydemic import AgeDistribution
        bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        return AgeDistribution(bin_edges=bin_edges, counts=age_data)

    def get_age_distribution(self, name):
        model = self.get_age_distribution_model(name)
        counts = np.array(model.counts)
        return counts / np.sum(counts)


from pydemic.data.us import UnitedStatesDataParser
united_states = UnitedStatesDataParser()
from pydemic.data.italy import ItalyDataParser
italy = ItalyDataParser()

all_parsers = [united_states, italy]


def scrape_all_data():
    for parser in all_parsers:
        parser.scrape_case_data()


def get_population_model(name):
    with open(DataParser._popdata_filename, 'r') as f:
        _populations = json.load(f)

    data = _populations[name]
    data_translated = {}
    for key, val in data.items():
        if key == 'ICUBeds':
            key2 = 'ICU_beds'
        elif key == 'suspectedCasesToday':
            key2 = 'initial_cases'
        else:
            key2 = camel_to_snake(key)
        data_translated[key2] = val

    from pydemic import PopulationModel
    return PopulationModel(**data_translated)


def get_age_distribution_model(name):
    with open(DataParser._agedata_filename, 'r') as f:
        age_data = json.load(f)

    age_data = list(age_data[name].values())
    from pydemic import AgeDistribution
    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    return AgeDistribution(bin_edges=bin_edges, counts=age_data)


__all__ = [
    "camel_to_snake",
    "CaseData",
    "dict_to_case_data",
    "scrape_all_data"
]
