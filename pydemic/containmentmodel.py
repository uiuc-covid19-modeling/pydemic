from datetime import datetime, timedelta

class ContainmentModel():

    def __init__(self, start_time, end_time):
        self.times = [ datetime(*start_time), datetime(*end_time) ]
        self.factors = [1., 1.]

    def add_sharp_event(self, time, factor):
        index_before = [ i for i,v in enumerate(self.times) if v<datetime(*time) ][-1]
        index_after = [ i for i,v in enumerate(self.times) if v>datetime(*time) ][0]
        self.times.append(datetime(*time)+timedelta(-1/1440))
        self.factors.append(self.factors[index_before])
        self.times.append(datetime(*time)+timedelta(1/1440))
        self.factors.append(factor)
        self.factors[index_after] = factor
        self.sort_times()

    def sort_times(self):
        self.times, self.factors = (list(l) for l in zip(*sorted(zip(self.times, self.factors))))

