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
from pydemic.data import DataParser

cwd = os.path.dirname(os.path.abspath(__file__))


class ItalyDataParser(DataParser):
    _filename = os.path.join(
        cwd, "../../assets/italy.h5"
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
        'casi_testati': 'total_tests',
        'tamponi': 'tests_performed',
        'note_it': 'note_it',
        'note_en': 'note_en',
    }

    def translate(self, key):
        return self.translation[key]

    def parse_case_data(self):
        df = super().parse_case_data()
        df['positive'] = df['new_positive'].cumsum()
        return df

    def get_population(self, name='Italy'):
        if name != 'Italy':
            name = 'ITA-' + name
        return super().get_population(name)

    def get_age_distribution(self):
        return super().get_age_distribution('Italy')
