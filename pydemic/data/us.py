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
import pandas as pd
from pydemic.data import DataParser

cwd = os.path.dirname(os.path.abspath(__file__))


class UnitedStatesDataParser(DataParser):
    _filename = os.path.join(
        cwd, "../../assets/us.h5"
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

        try:
            df = pd.read_hdf(self._filename, 'covid_tracking_info')
        except:  # noqa
            df = pd.read_json(self.info_url)
            try:
                df.to_hdf(self._filename, 'covid_tracking_info')
            except:  # noqa
                pass  # pytables isn't installed

        self.abbreviations = dict(zip(df.name, df.state))
        self.inverse_abbreviations = dict(zip(df.state, df.name))

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

    def get_age_distribution(self):
        return super().get_age_distribution('United States of America')
