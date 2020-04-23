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

from pydemic.simulation import (Simulation, SimulationState, StateLogger,
                                QuantileLogger)
from pydemic.reactions import Reaction, PassiveReaction, ErlangProcess, GammaProcess
from pydemic.mitigation import MitigationModel
from pydemic.sampling import SampleParameter, LikelihoodEstimator


def date_to_ms(date):
    from datetime import datetime, timezone
    return int(datetime(*date, tzinfo=timezone.utc).timestamp() * 1000)


_ms_per_day = 86400000


def days_from(date, relative_to=(2020, 1, 1)):
    """
    Converts a date into a number of days.

    :arg date: A :class:`tuple` specifying a date as ``(year, month, day)``.

    :arg relative_to: A :class:`tuple` specifying the date to which ``date``
        will be compared when computing the number of days.
        Defaults to ``(2020, 1, 1)``.

    :returns: A :class:`float`.
    """

    from warnings import warn
    warn("days_from is deprecated. Use pandas instead.",
         DeprecationWarning, stacklevel=2)

    return int(date_to_ms(date) - date_to_ms(relative_to)) // _ms_per_day


def date_from(days, relative_to=(2020, 1, 1)):
    """
    Converts a date into a number of days.

    :arg days: A :class:`float` the number of days since ``relative_to``.

    :arg relative_to: A :class:`tuple` specifying the date to which ``date``
        will be compared when computing the number of days.
        Defaults to ``(2020, 1, 1)``.

    :returns: A :class:`tuple` ``(year, month, day)``
    """

    from warnings import warn
    warn("date_from is deprecated. Use pandas instead.",
         DeprecationWarning, stacklevel=2)

    from datetime import datetime, timedelta
    full_date = datetime(*relative_to) + timedelta(days)
    return tuple([full_date.year, full_date.month, full_date.day])


def days_to_dates(days, relative_to=(2020, 1, 1)):
    """
    Converts a list of days (represented in float format) to datetime
    objects.

    :arg days: A :class:`float` number of days since 2020-01-01.

    :arg relative_to: A :class:`tuple` specifying the date to which ``date``
        will be compared when computing the number of days.
        Defaults to ``(2020, 1, 1)``.

    :returns: A :class:`tuple` of datetime objects for each day in the
        passed array.
    """

    from warnings import warn
    warn("days_to_dates is deprecated. Use pandas instead.",
         DeprecationWarning, stacklevel=2)

    from datetime import datetime, timedelta
    return [datetime(*relative_to) + timedelta(float(x)) for x in days]


__all__ = [
    "Simulation",
    "Reaction",
    "PassiveReaction",
    "ErlangProcess",
    "GammaProcess",
    "Simulation",
    "SimulationState",
    "StateLogger",
    "QuantileLogger",
    "MitigationModel",
    "SampleParameter",
    "LikelihoodEstimator",
    "date_to_ms",
]
