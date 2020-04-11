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
import numpy as np
from pydemic.data import DataParser

cwd = os.path.dirname(os.path.abspath(__file__))


class ItalyDataParser(DataParser):
    _casedata_filename = os.path.join(
        cwd, "../../assets/italy_case_counts.json"
    )
    # info: https://github.com/pcm-dpc/COVID-19
    data_url = "https://raw.github.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-regioni.json"  # noqa
    region_specifier = 'region'

    translation = {
        'data': 'date',
        'stato': 'state',
        'codice_regione': 'region_code',
        'denominazione_regione': 'region',
        'lat': 'lat',
        'long': 'long',
        'ricoverati_con_sintomi': 'hospitalized_with_symptoms',
        'terapia_intensiva': 'critical',
        'totale_ospedalizzati': 'total_hospitalized',
        'isolamento_domiciliare': 'isolated',
        'totale_casi': 'positive',
        'nuovi_positivi': 'new_positive',
        'totale_positivi': 'currently_positive',
        'variazione_totale_positivi': 'change_in_currently_positive',
        'dimessi_guariti': 'discharged',
        'deceduti': 'dead',
        'tamponi': 'total_tests',
        'note_it': 'note_it',
        'note_en': 'note_en',
    }

    def translate(self, key):
        return self.translation[key]

    def parse_case_data(self):
        region_data_series = super().parse_case_data()
        for region, data in region_data_series.items():
            data['positive'] = [int(x) for x in np.cumsum(data['new_positive'])]

        return region_data_series

    def convert_to_date(self, string):
        return tuple(map(int, string[:10].split('-')))

    def get_population(self, name='Italy'):
        if name != 'Italy':
            name = 'ITA-' + name
        return super().get_population(name)

    def get_age_distribution_model(self):
        return super().get_age_distribution_model('Italy')

    def get_age_distribution(self):
        model = self.get_age_distribution_model()
        counts = np.array(model.counts)
        return counts / np.sum(counts)
