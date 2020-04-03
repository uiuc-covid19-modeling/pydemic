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
import requests
from pydemic.data import camel_to_snake, dict_to_case_data


cwd = os.path.dirname(os.path.abspath(__file__))

_casedata_filename = os.path.join(
    cwd, "../../assets/italy_case_counts.json"
)

translation = {
    'data': 'date',
    'stato': 'state',
    'codice_regione': 'region_code',
    'denominazione_regione': 'region',
    'lat': 'lat',
    'long': 'long',
    'ricoverati_con_sintomi': 'hospitalized_with_symptoms',
    'terapia_intensiva': 'ICU',
    'totale_ospedalizzati': 'total_hospitalized',
    'isolamento_domiciliare': 'isolated',
    'totale_positivi': 'positive',
    'variazione_totale_positivi': 'total_change_positive',
    'nuovi_positivi': 'positive_increase',
    'dimessi_guariti': 'discharged',
    'deceduti': 'deaths',
    'totale_casi': 'total_cases',
    'tamponi': 'total_tests',
    'note_it': 'note_it',
    'note_en': 'note_en',
}

data_url  = "https://raw.github.com/pcm-dpc/COVID-19/master/dati-json/dpc-covid19-ita-regioni.json"  # noqa


def string_to_date(string):
    return tuple(map(int, string[:10].split('-')))


def scrape_case_data():
    r  = requests.get(data_url)
    all_data = json.loads(r.text)
    r.close()
    all_data = [
        {translation[key]: val for key, val in x.items()}
        for x in all_data
    ]

    regions = set([x['region'] for x in all_data])
    data_fields = all_data[0].keys()

    region_data = {}
    for region in regions:
        region_data[region] = []

    for data_point in all_data:
        region = data_point.pop('region')
        region_data[region].append(data_point)

    region_data_series = {}
    for region, data in region_data.items():
        sorted_data = sorted(data, key=lambda x: x['date'])
        for dp in sorted_data:
            dp['date'] = string_to_date(dp['date'])

        data_series = {}
        for key in data_fields:
            snake_key = camel_to_snake(key)
            data_series[snake_key] = [x.get(key) for x in sorted_data]

        region_data_series[region] = data_series

    with open(_casedata_filename, 'w') as f:
        json.dump(region_data_series, f)


def get_case_data(state):
    with open(_casedata_filename, 'r') as f:
        case_data = json.load(f)

    return dict_to_case_data(case_data[state])
