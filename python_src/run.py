import numpy as np


# this should be named better
def evolve(pop, P, sample):
    fracInfected = sum(pop.infectious) / P.populationServed
    newTime = pop.time + P.timeDelta
    Keys = pop.infectious.keys().sort()

    newPop = dict()

    def push(sub, age, delta):
        newPop[age] = pop[sub][age] + delta

    newCases = dict()
    newInfectious = dict()
    newRecovered = dict()
    newHospitalized = dict()
    newDischarged = dict()
    newCritical = dict()
    newStabilized = dict()
    newICUDead = dict()
    newOverflowStabilized = dict()
    newOverflowDead = dict()

    for age in Keys:
        newCases[age] = (
            sample(P.importsPerDay[age] * P.timeDeltaDays) +
            sample(
                (1 - P.isolatedFrac[age]) * P.infectionRate(newTime) * pop.susceptible[age] * fracInfected * P.timeDeltaDays,
            )
        )
        newInfectious[age] = (
            min(pop.exposed[age],
                sample((pop.exposed[age] * P.timeDeltaDays) / P.incubationTime))
        )
        newRecovered[age] = (
            min(pop.infectious[age],
                sample(pop.infectious[age] * P.timeDeltaDays * P.recoveryRate[age]))
        )
        newHospitalized[age] = (
            min(pop.infectious[age] - newRecovered[age],
                sample(pop.infectious[age] * P.timeDeltaDays * P.hospitalizedRate[age]))
        )
        newDischarged[age] = (
            min(pop.hospitalized[age],
                sample(pop.hospitalized[age] * P.timeDeltaDays * P.dischargeRate[age]))
        )
        newCritical[age] = (
            min(pop.hospitalized[age] - newDischarged[age],
                sample(pop.hospitalized[age] * P.timeDeltaDays * P.criticalRate[age]))
        )
        newStabilized[age] = (
            min(pop.critical[age],
                sample(pop.critical[age] * P.timeDeltaDays * P.stabilizationRate[age]))
        )
        newICUDead[age] = (
            min(pop.critical[age] - newStabilized[age],
                sample(pop.critical[age] * P.timeDeltaDays * P.deathRate[age]))
        )
        newOverflowStabilized[age] = (
            min(pop.overflow[age],
                sample(pop.overflow[age] * P.timeDeltaDays * P.stabilizationRate[age]))
        )
        newOverflowDead[age] = (
            min(pop.overflow[age] - newOverflowStabilized[age],
                sample(pop.overflow[age] * P.timeDeltaDays * P.overflowDeathRate[age]))
        )

        push('susceptible', age, -newCases[age])
        push('exposed', age, newCases[age] - newInfectious[age])
        push('infectious', age,
                newInfectious[age] - newRecovered[age] - newHospitalized[age])
        push('hospitalized', age,
                newHospitalized[age] + newStabilized[age]
                + newOverflowStabilized[age] - newDischarged[age]
                - newCritical[age]
        )
        # Cumulative categories
        push('recovered', age, newRecovered[age] + newDischarged[age])
        push('intensive', age, newCritical[age])
        push('discharged', age, newDischarged[age])
        push('dead', age, newICUDead[age] + newOverflowDead[age])

    freeICUBeds = (
        P.ICUBeds - (sum(pop.critical) - sum(newStabilized) - sum(newICUDead))
    )

    for ag in Keys:
        if freeICUBeds > newCritical[age]:
            freeICUBeds -= newCritical[age]
            push('critical', age,
                 newCritical[age] - newStabilized[age] - newICUDead[age])
            push('overflow', age,
                 -newOverflowDead[age] - newOverflowStabilized[age])
        elif freeICUBeds > 0:
            newOverflow = newCritical[age] - freeICUBeds
            push('critical', age,
                 freeICUBeds - newStabilized[age] - newICUDead[age])
            push('overflow', age,
                 newOverflow - newOverflowDead[age] - newOverflowStabilized[age])
            freeICUBeds = 0
        else:
            push('critical', age, -newStabilized[age] - newICUDead[age])
            push('overflow', age,
                 newCritical[age] - newOverflowDead[age] - newOverflowStabilized[age])

    # If any overflow patients are left AND there are free beds, move them back.
    # Again, move w/ lower age as priority.
    i = 0
    while freeICUBeds > 0 and i < len(Keys):
        age = Keys[i]
        if newPop.overflow[age] < freeICUBeds:
            newPop.critical[age] += newPop.overflow[age]
            freeICUBeds -= newPop.overflow[age]
            newPop.overflow[age] = 0
        else:
            newPop.critical[age] += freeICUBeds
            newPop.overflow[age] -= freeICUBeds
            freeICUBeds = 0
        i += 1

    # NOTE: For debug purposes only.
    # const popSum = sum(newPop.susceptible) + sum(newPop.exposed) + sum(newPop.infectious) + sum(newPop.recovered) + sum(newPop.hospitalized) + sum(newPop.critical) + sum(newPop.overflow) + sum(newPop.dead);
    # console.log(math.abs(popSum - P.populationServed));

    return newPop

def run(params, severity, age_distribution, contaiments):
    pass
