from pydemic import date_to_ms
from scipy.interpolate import interp1d


_2020_01_01 = 1577836800000
_ms_per_day = 86400000


def days_from_jan_1_2020(date):
    return int(date_to_ms(date) - _2020_01_01) // _ms_per_day


class ContainmentModel:
    """
        If is_in_days is True, then all times are measured with
        respect to 2020-01-01.
    """

    times = []
    _events = []
    _interp = None

    def __init__(self, start_time, end_time, is_in_days=False):
        if is_in_days is False:
            print("! please rewrite your containment to be in date tuples")
        self._events.append(['start', days_from_jan_1_2020(start_time), 1]) 
        self._events.append(['end', days_from_jan_1_2020(end_time)]) 
        self.sort_times()
        self._regenerate()

    def add_sharp_event(self, time, factor, dt_days=0.05):
        self._events.append(['sharp', days_from_jan_1_2020(time), factor, dt_days])
        # regenerate list
        self.sort_times()
        self._regenerate()

    def sort_times(self):
        self._events = sorted(self._events, key=lambda x: x[1])
        c_factor = 1.
        times = []
        factors = []
        for event in self._events:
            if event[0] == "start":
                times.append(event[1])
                factors.append(c_factor)
            elif event[0] == "end":
                times.append(event[1])
                factors.append(factors[-1])
            elif event[0] == "sharp":
                times.append(event[1]-event[3])
                factors.append(factors[-1])
                times.append(event[1])
                factors.append(event[2])
        self.times, self.factors = (
            list(l) for l in zip(*sorted(zip(times, factors)))
        )

    def _regenerate(self):
        self._interp = interp1d(self.times, self.factors)

    def get_dictionary(self):
        obj = {}
        from datetime import datetime
        dts = [datetime.utcfromtimestamp(x//1000) for x in self.times]
        obj['times'] = [[x.year, x.month, x.day, x.hour, x.minute, x.second]
                        for x in dts]
        obj['factors'] = self.factors
        return obj

    def __call__(self, time):
        return self._interp(time)

