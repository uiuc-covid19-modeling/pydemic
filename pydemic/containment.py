from datetime import datetime, timedelta
from scipy.interpolate import interp1d


class ContainmentModel:
    _interp = None

    def __init__(self, start_time, end_time):
        self.times = [ int(datetime(*start_time).timestamp()*1000), int(datetime(*end_time).timestamp()*1000) ]
        self.factors = [1., 1.]

    def add_sharp_event(self, time, factor):
        time_ts = int(datetime(*time).timestamp()*1000)
        index_before = [ i for i,v in enumerate(self.times) if v<time_ts ][-1]
        index_after = [ i for i,v in enumerate(self.times) if v>time_ts ][0]
        self.times.append(time_ts-30000)
        self.factors.append(self.factors[index_before])
        self.times.append(time_ts+30000)
        self.factors.append(factor)
        self.factors[index_after] = factor
        self.sort_times()
        self._regenerate()

    def sort_times(self):
        self.times, self.factors = (list(l) for l in zip(*sorted(zip(self.times, self.factors))))

    def _regenerate(self):
        self._interp = interp1d(self.times, self.factors)

    def __call__(self, time):
        return self._interp(time)
