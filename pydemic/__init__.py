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

class AttrDict(dict):
    expected_kwargs = set()

    def __init__(self, *args, **kwargs):
        if not self.expected_kwargs.issubset(set(kwargs.keys())):
            raise ValueError

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


from pydemic.containment import ContainmentModel
from pydemic.simulation import Simulation


class AgeDistribution(AttrDict):
    """
    .. attribute:: bin_edges

        A :class:`numpy.ndarray` specifying the lower end of each of the age ranges
        represented in :attr:`counts`.
        I.e., the age group ``[bin_edges[i], bin_edges[i+1])``
        has count ``counts[i]``.
        Has the same length as :attr:`counts`, i.e., the final bin is
        ``[bin_edges[-1], inf]``.

    .. attribute:: counts

        A :class:`numpy.ndarray` of the total population of the age groups specified
        by :attr:`bin_edges`.
    """

    expected_kwargs = {
        'bin_edges',
        'counts'
    }


class PopulationModel(AttrDict):
    """
    .. attribute:: country

        A :class:`string` with the name of the country.

    .. attribute:: cases

    .. attribute:: population_served

        The total population.

    .. attribute:: population_served

    .. attribute:: hospital_beds

    .. attribute:: ICU_beds

    .. attribute:: suspected_cases_today

    .. attribute:: imports_per_day
    """

    expected_kwargs = {
        'country',
        'cases',
        'population_served',
        'hospital_beds',
        'ICU_beds',
        'suspected_cases_today',
        'imports_per_day'
    }


class EpidemiologyModel(AttrDict):
    """
    .. attribute:: r0

        The average number of infections caused by an individual who is infected
        themselves.

    .. attribute:: incubation_time

        The number of days of incubation.

    .. attribute:: infectious_period

        The number of days an individual remains infections.

    .. attribute:: length_hospital_stay

        The average amount of time a patient remains in the hospital.

    .. attribute:: length_ICU_stay

        The average amount of time a critical patient remains in the ICU.

    .. attribute:: seasonal_forcing

        The amplitude of the seasonal modulation to the
        :meth:`Simulation.infection_rate`.

    .. attribute:: peak_month

        The month (as an integer in ``[0, 12)``) of peak
        :meth:`Simulation.infection_rate`.

    .. attribute:: overflow_severity

        The factor by which the :attr:`Simulation.overflow_death_rate`
        exceeds the ICU :attr:`Simulation.death_rate`
    """

    expected_kwargs = {
        'r0',
        'incubation_time',
        'infectious_period',
        'length_hospital_stay',
        'length_ICU_stay',
        'seasonal_forcing',
        'peak_month',
        'overflow_severity'
    }


class SeverityModel(AttrDict):
    """
    .. attribute:: id

    .. attribute:: age_group

    .. attribute:: isolated

    .. attribute:: confirmed

    .. attribute:: severe

    .. attribute:: critical

    .. attribute:: fatal
    """

    expected_kwargs = {
        'id',
        'age_group',
        'isolated',
        'confirmed',
        'severe',
        'critical',
        'fatal'
    }


class CaseData(AttrDict):
    """
    .. attribute:: dates

    .. attribute:: last_date

    .. attribute:: cases

    .. attribute:: deaths

    .. attribute:: hospitalized

    .. attribute:: ICU

    .. attribute:: recovered
    """

    expected_kwargs = {
        'dates',
        'last_date',
        'cases',
        'deaths',
        'hospitalized',
        'ICU',
        'recovered'
    }


def date_to_ms(date):
    from datetime import datetime, timezone
    return int(datetime(*date, tzinfo=timezone.utc).timestamp() * 1000)


__all__ = [
    "AttrDict",
    "AgeDistribution",
    "PopulationModel",
    "EpidemiologyModel",
    "SeverityModel",
    "ContainmentModel",
    "CaseData",
    "Simulation",
    "date_to_ms",
]
