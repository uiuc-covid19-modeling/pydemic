import { collectTotals, evolve, getPopulationParams, initializePopulation } from './model'

import { TimeSeries } from './types/TimeSeries.types'
import { DateRange, AllParamsFlat, PopulationData, EpidemiologicalData, SimulationData } from './types/Param.types'
import { AlgorithmResult, SimulationTimePoint } from './types/Result.types'
import { SeverityTableRow } from './types/SeverityTableRow.types'

import { OneCountryAgeDistribution } from './types/CountryAgeDistribution.types'

const identity = (x: number) => x
const poisson = (x: number) => {
  throw new Error('We removed dependency on `random` package. Currently `poisson` is not implemented')
}



// NOTE: Assumes containment is sorted ascending in time.
export function interpolateTimeSeries(containment: TimeSeries): (t: Date) => number {
  // If user hasn't touched containment, this vector is empty
  if (containment.length === 0) {
    return (t: Date) => {
      return 1.0
    }
  }

  return (t: Date) => {
    if (t <= containment[0].t){
      return containment[0].y
    } else if (t >= containment[containment.length-1].t) {
      return containment[containment.length-1].y
    } else {
      const index = containment.findIndex(d => Number(t) < Number(d.t))

      // Deal with extrapolation
      // i.e. the time given exceeds the containment series.
      // should no longer be needed!
      if (index <= 0) {
        return 1.0
      }

      const deltaY = containment[index].y - containment[index - 1].y
      const deltaT = Number(containment[index].t) - Number(containment[index - 1].t)

      const dS = deltaY / deltaT
      const dT = Number(t) - Number(containment[index - 1].t)
      return containment[index - 1].y + dS * dT
    }
  }
}



import * as d3 from 'd3'
export function uniformDatesBetween(min: number, max: number, n: number): Date[] {
  const d = (max - min) / (n - 1)
  const dates = d3.range(min, max + d, d).filter((_, i) => i < n)
  return dates.map(d => new Date(d))
}
export function makeTimeSeries(simulationTimeRange: DateRange, values: number[]): TimeSeries {
  const { tMin, tMax } = simulationTimeRange
  const n = values.length

  const dates = uniformDatesBetween(tMin.getTime(), tMax.getTime(), n)

  const tSeries = []
  for (let i = 0; i < n; i++) {
    tSeries.push({ t: dates[i], y: values[i] })
  }

  return tSeries
}


function getDate(datelist) {
  return new Date(datelist[0], datelist[1]-1, datelist[2], datelist[3], datelist[4], datelist[5]);
} 

export function wrapper(argdata) {
  
  // get data from argument
  var passedSim = argdata.simulation;
  var passedPop = argdata.population;
  var passedContainment = argdata.containment;

  // javascript run function expects data in this format
  var popData : PopulationData = {
    "populationServed": passedPop.populationServed, 
    "country": passedPop.country, 
    "hospitalBeds": passedPop.hospitalBeds, 
    "ICUBeds": passedPop.ICUBeds, 
    "suspectedCasesToday": passedPop.suspectedCasesToday, 
    "importsPerDay": passedPop.importedPerDay, 
    "cases": passedPop.cases
  };

  var epiData : EpidemiologicalData = { 
    "r0": 2.2,
    "incubationTime": 5,
    "infectiousPeriod": 3,
    "lengthHospitalStay": 4,
    "lengthICUStay": 14,
    "seasonalForcing": 0.2,
    "peakMonth": 0,
    "overflowSeverity": 2
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

  var severity = [
    {
      "id": 0,
      "ageGroup": "0-9",
      "isolated": 0.0,
      "confirmed": 5.0,
      "severe": 1.0,
      "critical": 5,
      "fatal": 30
    },
    {
      "id": 2,
      "ageGroup": "10-19",
      "isolated": 0.0,
      "confirmed": 5.0,
      "severe": 3.0,
      "critical": 10,
      "fatal": 30
    },
    {
      "id": 4,
      "ageGroup": "20-29",
      "isolated": 0.0,
      "confirmed": 10.0,
      "severe": 3.0,
      "critical": 10,
      "fatal": 30
    },
    {
      "id": 6,
      "ageGroup": "30-39",
      "isolated": 0.0,
      "confirmed": 15.0,
      "severe": 3.0,
      "critical": 15,
      "fatal": 30
    },
    {
      "id": 8,
      "ageGroup": "40-49",
      "isolated": 0.0,
      "confirmed": 20.0,
      "severe": 6.0,
      "critical": 20,
      "fatal": 30
    },
    {
      "id": 10,
      "ageGroup": "50-59",
      "isolated": 0.0,
      "confirmed": 25.0,
      "severe": 10.0,
      "critical": 25,
      "fatal": 40
    },
    {
      "id": 12,
      "ageGroup": "60-69",
      "isolated": 0.0,
      "confirmed": 30.0,
      "severe": 25.0,
      "critical": 35,
      "fatal": 40
    },
    {
      "id": 14,
      "ageGroup": "70-79",
      "isolated": 0.0,
      "confirmed": 40.0,
      "severe": 35.0,
      "critical": 45,
      "fatal": 50
    },
    {
      "id": 16,
      "ageGroup": "80+",
      "isolated": 0.0,
      "confirmed": 50.0,
      "severe": 50.0,
      "critical": 55,
      "fatal": 50
    }
  ];

  var ageDistribution : OneCountryAgeDistribution = {
    "0-9": passedPop.populationsByDecade[0],
    "10-19": passedPop.populationsByDecade[1],
    "20-29": passedPop.populationsByDecade[2],
    "30-39": passedPop.populationsByDecade[3],
    "40-49": passedPop.populationsByDecade[4],
    "50-59": passedPop.populationsByDecade[5],
    "60-69": passedPop.populationsByDecade[6],
    "70-79": passedPop.populationsByDecade[7],
    "80+": passedPop.populationsByDecade[8]
  };

  var containment_ts = [];
  for (var i in passedContainment.factors) {
    containment_ts.push({"t": getDate(passedContainment.times[i]), "y": passedContainment.factors[i]});
  }

  return run(params, severity, ageDistribution, containment_ts);

}





/**
 *
 * Entry point for the algorithm
 *
 */
export default async function run(
  params: AllParamsFlat,
  severity: SeverityTableRow[],
  ageDistribution: OneCountryAgeDistribution,
  containment: TimeSeries,
): Promise<AlgorithmResult> {
  const modelParams = getPopulationParams(params, severity, ageDistribution, interpolateTimeSeries(containment))
  const tMin: number = params.simulationTimeRange.tMin.getTime()
  const tMax: number = params.simulationTimeRange.tMax.getTime()
  const initialCases = params.suspectedCasesToday
  let initialState = initializePopulation(modelParams.populationServed, initialCases, tMin, ageDistribution)

  function simulate(initialState: SimulationTimePoint, func: (x: number) => number) {
    const dynamics = [initialState]
    while (dynamics[dynamics.length - 1].time < tMax) {
      const pop = dynamics[dynamics.length - 1]
      dynamics.push(evolve(pop, modelParams, func))
    }

    return collectTotals(dynamics)
  }

  const sim: AlgorithmResult = {
    deterministicTrajectory: simulate(initialState, identity),
    stochasticTrajectories: [],
    params: modelParams,
  }

  for (let i = 0; i < modelParams.numberStochasticRuns; i++) {
    initialState = initializePopulation(modelParams.populationServed, initialCases, tMin, ageDistribution)
    sim.stochasticTrajectories.push(simulate(initialState, poisson))
  }

  return sim
}