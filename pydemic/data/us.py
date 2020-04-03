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
    cwd, "../../assets/case_counts.json"
)
data_url = "https://covidtracking.com/api/states/daily"
info_url = "https://covidtracking.com/api/states/info"


def num_to_date(num):
    year = num // 10000
    month = (num - 10000 * year) // 100
    day = (num - 10000 * year - 100 * month)
    return (year, month, day)


def scrape_case_data():
    r = requests.get(info_url)
    info = json.loads(r.text)
    r.close()

    info = {item.pop('state'): item for item in info}

    r = requests.get(data_url)
    all_data = json.loads(r.text)
    r.close()

    data_fields = all_data[0].keys()

    state_data = {}
    for state in info.keys():
        state_data[state] = []

    for data_point in all_data:
        state = data_point.pop('state')
        state_data[state].append(data_point)

    state_data_series = {}
    for state, data in state_data.items():
        sorted_data = sorted(data, key=lambda x: x['date'])
        for dp in sorted_data:
            dp['date'] = num_to_date(dp['date'])

        data_series = {}
        for key in data_fields:
            snake_key = camel_to_snake(key)
            data_series[snake_key] = [x.get(key) for x in sorted_data]

        state_data_series[state] = data_series

    with open(_casedata_filename, 'w') as f:
        json.dump(state_data_series, f)


def get_case_data(state):
    with open(_casedata_filename, 'r') as f:
        case_data = json.load(f)

    return dict_to_case_data(case_data[state])
