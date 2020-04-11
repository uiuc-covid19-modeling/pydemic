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
from pydemic.data import DataParser

cwd = os.path.dirname(os.path.abspath(__file__))


class UnitedStatesDataParser(DataParser):
    _casedata_filename = os.path.join(
        cwd, "../../assets/us_case_counts.json"
    )
    _info_filename = os.path.join(
        cwd, "../../assets/us_info.json"
    )
    data_url = "https://covidtracking.com/api/states/daily"
    info_url = "https://covidtracking.com/api/states/info"
    region_specifier = 'state'

    translation = {
        'death': 'dead',
        'in_icu_currently': 'critical',
    }

    def __init__(self):
        super().__init__()

        import os.path
        if not os.path.isfile(self._info_filename):
            self.update_info()

        with open(self._info_filename, 'r') as f:
            info = json.load(f)

        self.abbreviations = {
            val['name']: key for key, val in info.items()
        }
        self.inverse_abbreviations = {
            val: key for key, val in self.abbreviations.items()
        }

    def update_info(self):
        import requests
        r = requests.get(self.info_url)
        info = json.loads(r.text)
        r.close()

        info = {item.pop('state'): item for item in info}

        with open(self._info_filename, 'w') as f:
            json.dump(info, f)

    def get_case_data(self, region):
        if region in self.abbreviations:
            region = self.abbreviations[region]

        return super().get_case_data(region)

    def get_population(self, name='United States of America'):
        if name != 'United States of America':
            if name in self.inverse_abbreviations:
                name = self.inverse_abbreviations[name]
            name = 'USA-' + name
        return super().get_population(name)

    def get_age_distribution_model(self):
        return super().get_age_distribution_model('United States of America')

    def get_age_distribution(self):
        model = self.get_age_distribution_model()
        counts = np.array(model.counts)
        return counts / np.sum(counts)
