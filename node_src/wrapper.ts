import { OneCountryAgeDistribution } from './types/CountryAgeDistribution.types'
import { DateRange, AllParamsFlat, PopulationData, EpidemiologicalData, SimulationData } from './types/Param.types'
import { AlgorithmResult } from './types/Result.types'

import { runfunc } from './run'

function getDate(datelist) {
  return new Date(datelist[0], datelist[1]-1, datelist[2], datelist[3], datelist[4], datelist[5]);
} 

export function wrapper(argdata) {
  
  // get data from argument
  var passedEpidemiology = argdata.epidemiology;
  var passedSim = argdata.simulation;
  var passedPop = argdata.population;
  var passedContainment = argdata.containment;

  // javascript run function expects data in this format
  var popData : PopulationData = {
    "populationServed": passedPop.population_served, 
    "country": passedPop.country, 
    "hospitalBeds": passedPop.hospital_beds, 
    "ICUBeds": passedPop.ICU_beds, 
    "suspectedCasesToday": passedPop.suspected_cases_today, 
    "importsPerDay": passedPop.imports_per_day, 
    "cases": passedPop.cases
  };

  var epiData : EpidemiologicalData = { 
    "r0": passedEpidemiology.r0,
    "incubationTime": passedEpidemiology.incubation_time,
    "infectiousPeriod": passedEpidemiology.infectious_period,
    "lengthHospitalStay": passedEpidemiology.length_hospital_stay,
    "lengthICUStay": passedEpidemiology.length_ICU_stay,
    "seasonalForcing": passedEpidemiology.seasonal_forcing,
    "peakMonth": passedEpidemiology.peak_month,
    "overflowSeverity": passedEpidemiology.overflow_severity
  };

  var simData : SimulationData = {
    simulationTimeRange: {
      tMin: getDate(passedSim.start),
      tMax: getDate(passedSim.end),
    },
    numberStochasticRuns: 0,
  };

  var params : AllParamsFlat = {
    ...popData,
    ...epiData,
    ...simData
  };

  var severity = argdata.severities;

  var ageDistribution : OneCountryAgeDistribution = {
    "0-9": passedPop.populations_by_decade[0],
    "10-19": passedPop.populations_by_decade[1],
    "20-29": passedPop.populations_by_decade[2],
    "30-39": passedPop.populations_by_decade[3],
    "40-49": passedPop.populations_by_decade[4],
    "50-59": passedPop.populations_by_decade[5],
    "60-69": passedPop.populations_by_decade[6],
    "70-79": passedPop.populations_by_decade[7],
    "80+": passedPop.populations_by_decade[8]
  };

  var containment_ts = [];
  for (var i in passedContainment.factors) {
    containment_ts.push({"t": getDate(passedContainment.times[i]), "y": passedContainment.factors[i]});
  }

  return runfunc(params, severity, ageDistribution, containment_ts);

}


