import numpy as np
from pydemic import PopulationModel, AgeDistribution, SeverityModel, EpidemiologyModel, ContainmentModel, date_to_ms
from pydemic.load import get_country_population_model, get_age_distribution_model
from pydemic import Simulation
from datetime import date, timedelta


class PydemicModel:
    def __init__(self):
        # load population from remote data
        self.POPULATION_NAME = "USA-Illinois"
        self.AGE_DATA_NAME = "United States of America"
        self.population = get_country_population_model(self.POPULATION_NAME)
        self.age_distribution = get_age_distribution_model(self.AGE_DATA_NAME)
        self.population.suspected_cases_today = 65
        self.population.imports_per_day = 5
        self.population.population_served = 12659682
        # define simulation parameters
        self.start_date = [2020, 2, 24, 0, 0, 0]
        self.end_date = [2020, 3, 26, 0, 0, 0]
        # set severity model
        n_age_groups = 9
        self.severity = SeverityModel(
            id=np.array([0, 2, 4, 6, 8, 10, 12, 14, 16]),
            age_group=np.arange(0., 90., 10),
            isolated=np.zeros(n_age_groups),
            confirmed=np.array([5., 5., 10., 15., 20., 25., 30., 40., 50.]),
            severe=np.array([1., 3., 3., 3., 6., 10., 25., 35., 50.]),
            critical=np.array([5., 10., 10., 15., 20., 25., 35., 45., 55.]),
            fatal=np.array([30., 30., 30., 30., 30., 40., 40., 50., 50.]),
        )
        # set epidemiology model
        self.epidemiology = EpidemiologyModel(
            r0=2.7,
            incubation_time=5,
            infectious_period=3,
            length_hospital_stay=4,
            length_ICU_stay=14,
            seasonal_forcing=0.2,
            peak_month=0,
            overflow_severity=2
        )

    def get_date_tuple(self, date):
        return [date.year, date.month, date.day]

    def update_model_params(self, model_params):
        # update times
        if 'start_date' in model_params:
            for i in range(len(model_params['start_date'])):
                self.start_date[i] = model_params['start_date'][i]
        if 'end_date' in model_params:
            for i in range(len(model_params['end_date'])):
                self.end_date[i] = model_params['end_date'][i]
        if 'start_day' in model_params:
            self.start_date = self.get_date_tuple(
                date(2020, 1, 1) + timedelta(model_params['start_day']))
        if 'end_day' in model_params:
            self.end_day[2] = self.get_date_tuple(
                date(2020, 1, 1) + timedelta(model_params['end_day']))
        # update epidemiology
        if 'R0' in model_params:
            self.epidemiology.r0 = model_params['R0']

    def get_mean_std(self, model_params, skip=4):
        # update model parameters
        self.update_model_params(model_params)
        # set containment model
        self.containment = ContainmentModel(self.start_date, self.end_date)
        mitigation = 1.0
        # if "mitigation" in model_params:
        #   mitigation = model_params["mitigation"]
        self.containment.add_sharp_event((2020, 3, 15), mitigation)
        # create simulation object
        self.start_time = date_to_ms(self.start_date)
        self.end_time = date_to_ms(self.end_date)
        sim = Simulation(self.population, self.epidemiology, self.severity, self.age_distribution,
                         self.containment)
        result = sim(self.start_time, self.end_time, lambda x: x)
        # get simpler keys
        time = result.time
        exposed = result.exposed
        recovered = result.recovered
        infectious = result.infectious
        susceptible = result.susceptible
        dead = result.dead
        # get mean
        infectious_mean = np.sum(infectious, axis=1)
        dead_mean = np.sum(dead, axis=1)
        # return
        return infectious_mean[::skip], None, dead_mean[::skip], None
