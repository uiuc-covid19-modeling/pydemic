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
from pydemic import map_to_days_if_needed
from scipy.interpolate import interp1d, CubicSpline


class SmoothPiecewiseCurve(CubicSpline):
    def __init__(self, x, y, refinement=20, window_length=5, polyorder=2, **kwargs):
        from scipy.signal import savgol_filter
        # from scipy.interpolate import interp1d

        piecewise = interp1d(x, y)
        _x = np.linspace(x[0], x[-1], refinement * len(x))
        _y = piecewise(_x)
        _y = savgol_filter(_y, window_length, polyorder)
        super().__init__(_x, _y, **kwargs)


class MitigationModel(SmoothPiecewiseCurve):
    def __init__(self, t0, tf, t, factors, **kwargs):
        t = np.insert(t, 0, min(t0, t[0]) - 10)
        t = np.append(t, max(tf, t[-1]) + 10)

        factors = np.insert(factors, 0, factors[0])
        factors = np.append(factors, factors[-1])
        super().__init__(t, factors, **kwargs)


class ContainmentModel:
    def __init__(self, start_time, end_time):
        self._events = [
            ['start', map_to_days_if_needed(start_time), 1],
            ['end', map_to_days_if_needed(end_time)]
        ]
        self.sort_times()
        self._regenerate()

    def add_sharp_event(self, time, factor, dt_days=0.05):
        self._events.append(['sharp', map_to_days_if_needed(time), factor, dt_days])
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
