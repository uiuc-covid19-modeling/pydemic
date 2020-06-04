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

import numpy as np
from pydemic.data import DataParser


__doc__ = """
.. currentmodule:: pydemic.data.united_states
.. autofunction:: covid_tracking
.. autofunction:: nyt
.. autofunction:: get_population
.. autofunction:: get_age_distribution
.. currentmodule:: pydemic
"""


state_populations = {
    'Wyoming': 567025,
    'Vermont': 628061,
    'District of Columbia': 720687,
    'Alaska': 734002,
    'North Dakota': 761723,
    'South Dakota': 903027,
    'Delaware': 982895,
    'Rhode Island': 1056161,
    'Montana': 1086759,
    'Maine': 1345790,
    'New Hampshire': 1371246,
    'Hawaii': 1412687,
    'West Virginia': 1778070,
    'Idaho': 1826156,
    'Nebraska': 1952570,
    'New Mexico': 2096640,
    'Kansas': 2910357,
    'Mississippi': 2989260,
    'Puerto Rico': 3032160,
    'Arkansas': 3038999,
    'Nevada': 3139658,
    'Iowa': 3179840,
    'Utah': 3282115,
    'Connecticut': 3563070,
    'Oklahoma': 3954821,
    'Oregon': 4301089,
    'Kentucky': 4499692,
    'Louisiana': 4645184,
    'Alabama': 4908621,
    'South Carolina': 5210095,
    'Minnesota': 5700671,
    'Colorado': 5845526,
    'Wisconsin': 5851754,
    'Maryland': 6083116,
    'Missouri': 6169270,
    'Indiana': 6745354,
    'Tennessee': 6897576,
    'Massachusetts': 6976597,
    'Arizona': 7378494,
    'Washington': 7797095,
    'Virginia': 8626207,
    'New Jersey': 8936574,
    'Michigan': 10045029,
    'North Carolina': 10611862,
    'Georgia': 10736059,
    'Ohio': 11747694,
    'Illinois': 12659682,
    'Pennsylvania': 12820878,
    'New York': 19440469,
    'Florida': 21992985,
    'Texas': 29472295,
    'California': 39937489
}

abbreviations = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AS': 'American Samoa',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District Of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'GU': 'Guam',
    'HI': 'Hawaii',
    'IA': 'Iowa',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'MA': 'Massachusetts',
    'MD': 'Maryland',
    'ME': 'Maine',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MO': 'Missouri',
    'MP': 'Northern Mariana Islands',
    'MS': 'Mississippi',
    'MT': 'Montana',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'NE': 'Nebraska',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NV': 'Nevada',
    'NY': 'New York',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VA': 'Virginia',
    'VI': 'US Virgin Islands',
    'VT': 'Vermont',
    'WA': 'Washington',
    'WI': 'Wisconsin',
    'WV': 'West Virginia',
    'WY': 'Wyoming'
}

inverse_abbreviations = dict(zip(abbreviations.values(), abbreviations.keys()))

usa_age = np.array([0.12000352, 0.1278914, 0.13925591, 0.13494838,
                    0.12189751, 0.12724997, 0.11627754, 0.07275651,
                    0.03971926])


def get_population(region=None):
    """
    :arg region: The region whose  shall be returned.
        May be specified by abbreviation or by full name.
        Defaults to *None*, in which case the entire population
        of the United States is returned.

    :returns: The population of ``region``.
    """

    if region is None:
        return sum(x for x in state_populations.values())
    else:
        region = abbreviations.get(region, region)
        return state_populations[region]


def get_age_distribution(region=None):
    """
    Returns the age distribution, stratified into the bins
    :math:`[0, 10)`, :math:`[10, 20)`, ..., :math:`[70, 80)`, and
    :math:`[80, \\inf)`.

    :arg region: The region whose  shall be returned.
        May be specified by abbreviation or by full name.
        Currently, this argument is ignored and the aggregated
        age distribution of the United States is returned.

    :returns: The United States age distribution (a :class:`numpy.ndarray`).
    """

    return usa_age


class COVIDTrackingDataParser(DataParser):
    data_url = "https://covidtracking.com/api/v1/states/daily.csv"
    date_column = 'date'
    region_column = 'state'

    translation = {
        'death': 'all_dead',
        'hospitalized_currently': 'hospitalized',
        'in_icu_currently': 'critical',
    }

    def __call__(self, region=None):
        region = inverse_abbreviations.get(region, region)
        df = super().__call__(region)
        for key in ('positive', 'all_dead'):
            df[key+'_incr'] = df[key].diff()
        return df


class NewYorkTimesDataParser(DataParser):
    data_url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"  # noqa
    date_column = 'date'
    region_column = 'state'

    translation = {
        'cases': 'positive',
        'deaths': 'all_dead',
    }

    def __call__(self, region=None):
        region = abbreviations.get(region, region)
        df = super().__call__(region)
        for key in ('positive', 'all_dead'):
            df[key+'_incr'] = df[key].diff()
        return df


def covid_tracking(region=None):
    """
    Returns state-level data as collected by the
    `COVID Tracking Project <https://covidtracking.com/>`_.

    :arg region: The state whose data shall be returned.
        May be specified by abbreviation or by full name.
        Defaults to *None*, in which all data (grouped by state) is returned.

    :returns: A :class:`pandas.DataFrame`.
    """

    return COVIDTrackingDataParser()(region)


def nyt(region=None):
    """
    Returns state-level COVID-19 data as collected by the
    `New York Times <https://github.com/nytimes/covid-19-data>`_.

    :arg region: The state whose data shall be returned.
        May be specified by abbreviation or by full name.
        Defaults to *None*, in which all data (grouped by state) is returned.

    :returns: A :class:`pandas.DataFrame`.
    """

    return NewYorkTimesDataParser()(region)
