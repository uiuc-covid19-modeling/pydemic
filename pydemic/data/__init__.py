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

import pandas as pd

__doc__ = """
Currently, two simple parsers are implemented to collect United States data.
More parsers can be added straightforwardly by subclassing
:class:`pydemic.data.DataParser`.

.. automodule:: pydemic.data.united_states
"""


def camel_to_snake(name):
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class DataParser:
    data_url = None
    date_column = 'date'
    region_column = 'region'
    translation = {}

    def translate_columns(self, key):
        _key = camel_to_snake(key)
        return self.translation.get(_key, _key)

    def __call__(self, region=None):
        df = pd.read_csv(self.data_url, parse_dates=[self.date_column],
                         index_col=[self.region_column, self.date_column])
        df = df.drop(columns=set(self.translation.values()) & set(df.columns))
        df = df.rename(columns=self.translate_columns)

        if region is not None:
            df = df.sort_index().loc[region]

        return df


__all__ = [
    "camel_to_snake",
    "DataParser",
]
