from datetime import datetime, timezone
from scipy.interpolate import interp1d


_2020_01_01 = 1577836800000
_ms_per_day = 86400000


def days_from_jan_1_2020(date):
    ms_val = 1000 * datetime(*date, tzinfo=timezone.utc).timestamp()
    return int(ms_val - _2020_01_01) // _ms_per_day


class ContainmentModel:
    """
        If is_in_days is True, then all times are measured with
        respect to 2020-01-01.
    """

    _is_in_days = False

    _interp = None

    def __init__(self, start_time, end_time, is_in_days=False):
        self._is_in_days = is_in_days
        if self._is_in_days:
            self.times = [days_from_jan_1_2020(start_time),
                          days_from_jan_1_2020(end_time)]
        else:
            self.times = [
                int(datetime(*start_time, tzinfo=timezone.utc).timestamp()*1000),
                int(datetime(*end_time, tzinfo=timezone.utc).timestamp()*1000)
            ]
        self.factors = [1., 1.]

    def add_sharp_event(self, time, factor):
        if self._is_in_days:
            time_ts = days_from_jan_1_2020(time)
        else:
            time_ts = int(datetime(*time, tzinfo=timezone.utc).timestamp()*1000)
        index_before = [i for i, v in enumerate(self.times) if v < time_ts][-1]
        index_after = [i for i, v in enumerate(self.times) if v > time_ts][0]
        if self._is_in_days:
            self.times.append(time_ts-0.000347)  # 30 second delta
            self.times.append(time_ts+0.000347)
        else:
            self.times.append(time_ts-30000)
            self.times.append(time_ts+30000)
        self.factors.append(self.factors[index_before])
        self.factors.append(factor)
        self.factors[index_after] = factor
        self.sort_times()
        self._regenerate()

    def sort_times(self):
        self.times, self.factors = (
            list(l) for l in zip(*sorted(zip(self.times, self.factors)))
        )

    def _regenerate(self):
        self._interp = interp1d(self.times, self.factors)

    def get_dictionary(self):
        obj = {}
        dts = [datetime.utcfromtimestamp(x//1000) for x in self.times]
        obj['times'] = [[x.year, x.month, x.day, x.hour, x.minute, x.second]
                        for x in dts]
        obj['factors'] = self.factors
        return obj

    def __call__(self, time):
        return self._interp(time)
