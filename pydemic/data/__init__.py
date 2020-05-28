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
import pandas as pd

cwd = os.path.dirname(os.path.abspath(__file__))


def camel_to_snake(name):
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class DataParser:
    _filename = None
    data_url = None
    region_specifier = 'state'
    _popdata_filename = os.path.join(
        cwd, "../../assets/population.json"
    )
    _agedata_filename = os.path.join(
        cwd, "../../assets/country_age_distribution.json"
    )
    translation = {}

    def translate(self, key):
        _key = camel_to_snake(key)
        return self.translation.get(_key, _key)

    def parse_case_data(self):
        df = pd.read_json(self.data_url)
        df = df.rename(mapper=self.translate, axis='columns')
        return df

    def scrape_case_data(self):
        df = self.parse_case_data()
        df.to_hdf(self._filename, 'covid_tracking_data')

    def get_case_data(self, region):
        import os.path
        if not os.path.isfile(self._filename):
            self.scrape_case_data()
        df = pd.read_hdf(self._filename, 'covid_tracking_data')

        df = df[df[self.region_specifier] == region]
        df.date = pd.to_datetime(df.date.astype(str)).dt.normalize()
        df = df.set_index('date')
        df = df.sort_index()
        return df

    def get_population(self, name):
        with open(self._popdata_filename, 'r') as f:
            populations = json.load(f)
        return populations[name]['populationServed']

    def get_age_distribution_model(self, name):
        with open(self._agedata_filename, 'r') as f:
            age_data = json.load(f)

        age_data = list(age_data[name].values())
        from pydemic.models.neher import AgeDistribution
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

    from pydemic.models.neher import PopulationModel
    return PopulationModel(**data_translated)


def get_age_distribution_model(name):
    with open(DataParser._agedata_filename, 'r') as f:
        age_data = json.load(f)

    age_data = list(age_data[name].values())
    from pydemic.models.neher import AgeDistribution
    bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    return AgeDistribution(bin_edges=bin_edges, counts=age_data)


__all__ = [
    "camel_to_snake",
    "DataParser",
    "scrape_all_data",
    "get_population_model",
    "get_age_distribution_model",
]
