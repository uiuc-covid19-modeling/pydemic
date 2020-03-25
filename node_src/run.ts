import { collectTotals, evolve, getPopulationParams, initializePopulation } from './model'

import { TimeSeries } from './types/TimeSeries.types'
import { AllParamsFlat, PopulationData, EpidemiologicalData, SimulationData } from './types/Param.types'
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


export function wrapper() {
  
  var popData : PopulationData = {
    "populationServed": 8600000, 
    "country": "Switzerland", 
    "hospitalBeds": 30799, 
    "ICUBeds": 1400, 
    "suspectedCasesToday": 1148, 
    "importsPerDay": 4.0, 
    "cases": "Switzerland"
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
      tMin: new Date(2020, 3, 3),
      tMax: new Date(2020, 9, 3)
    },
    numberStochasticRuns: 0,
  };

  var params : AllParamsFlat = {
    ...popData,
    ...epiData,
    ...simData
  };


  var ageDistribution : OneCountryAgeDistribution = {
    "0-9": 4994996,
    "10-19": 5733447,
    "20-29": 6103437,
    "30-39": 6998434,
    "40-49": 9022004,
    "50-59": 9567192,
    "60-69": 7484860,
    "70-79": 6028907,
    "80+": 4528548
  };




  //return run(params);


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