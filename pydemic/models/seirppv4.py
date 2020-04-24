import numpy as np
from pydemic.models.seirpp import NonMarkovianSIRSimulationBase

class SEIRPlusPlusSimulationHospitalCriticalAndDeath(NonMarkovianSIRSimulationBase):
    """
    SEIR++ model with unconnected infectivity loop. Readout topology is:
        
        -> symptomatic
            -> hospitalized -> recovered
                            -> critical -> dead
                                        -> hospitalized -> recovered

    Arguments are:
        seasonal_forcing:   [FIXME: sources?]
            seasonal_forcing_amp=.2
            peak_day=15
        infectivity track:  [FIXME: sources?]
            r0=3.2
            serial_mean=4
            serial_std=3.25
        symptomaticity track:
            incubation_mean=5.5
            incubation_std=2
            p_symptomatic=[1] * n_demographics
            p_symptomatic_prefactor=None  WARNING: setting this will override ifr
        hospitalization track:
            hospitalized_mean=6.5
            hospitalized_std=4
            p_hospitalized=[1] * n_demographics
            p_hospitalized_prefactor=1.  

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, mitigation, *,
                 r0=3.2, serial_mean=4, serial_std=3.25,
                 seasonal_forcing_amp=.2, peak_day=15,

                 ifr=0.009, 
                 incubation_mean=5.5, incubation_std=2,
                 # WARNING: setting p_symptomatic_prefactor will override ifr
                 p_symptomatic=1., p_symptomatic_prefactor=None,
                 
                 p_hospitalized=1., p_hospitalized_prefactor=1.,
                 # symptomatic -> hospitalized below
                 hospitalized_mean=6.5, hospitalized_std=4.,
                 # hospitalized -> discharged below
                 discharged_mean=6., discharged_std=4.,

                 p_critical=1., p_critical_prefactor=1.,
                 # hospitalized -> critical below
                 critical_mean=2., critical_std=2.,

                 p_dead=1.,
                 p_dead_prefactor=1.,
                 # critical -> dead below
                 dead_mean=7.5, dead_std=7.5,
                 # critical -> hospitalized below
                 recovered_mean=7.5, recovered_std=7.5,

                 age_distribution=None, **kwargs):

        super().__init__(
            mitigation, r0=r0, serial_mean=serial_mean, serial_std=serial_std,
            seasonal_forcing_amp=seasonal_forcing_amp, peak_day=peak_day
        )

        if age_distribution is None:
            # default to usa_population
            age_distribution = np.array([0.12000352, 0.12789140, 0.13925591, 0.13494838, 0.12189751, 
                                         0.12724997, 0.11627754, 0.07275651, 0.03971926])

        # if p_symptomatic_prefactor is None, set according to ifr
        if p_symptomatic_prefactor is None:
            p_dead_product = p_symptomatic * p_hospitalized * p_critical * p_dead
            synthetic_ifr = (p_dead_product * age_distribution).sum()
            p_symptomatic_prefactor = ifr / synthetic_ifr

        # make numpy arrays first in case p_* passed as lists
        p_symptomatic = np.array(p_symptomatic) * p_symptomatic_prefactor
        p_hospitalized = np.array(p_hospitalized) * p_hospitalized_prefactor
        p_critical = np.array(p_critical) * p_critical_prefactor
        p_dead = np.array(p_dead) * p_dead_prefactor

        # now check that none of the prefactors are too large
        from pydemic.sampling import InvalidParametersError

        # first check p_symptomatic_prefactor
        top = age_distribution
        bottom = age_distribution * p_symptomatic
        if top.sum() < bottom.sum():
            raise InvalidParametersError(
                "p_symptomatic_prefactor must not be too large"
            )

        # then check p_hospitalized_prefactor
        top = bottom.copy()
        bottom *= p_hospitalized
        if top.sum() < bottom.sum():
            raise InvalidParametersError(
                "p_hospitalized_prefactor must not be too large"
            )

        # then check p_critical_prefactor
        top = bottom.copy()
        bottom *= p_critical
        if top.sum() < bottom.sum():
            raise InvalidParametersError(
                "p_critical_prefactor must not be too large"
            )

        # and finally check p_dead_prefactor
        top = bottom.copy()
        bottom *= p_dead
        if top.sum() < bottom.sum():
            raise InvalidParametersError(
                "p_dead_prefactor must not be too large"
            )

        self.readouts = {
            "symptomatic": ('infected', p_symptomatic, incubation_mean, incubation_std),
            "admitted_to_hospital": ('symptomatic', p_hospitalized, hospitalized_mean, hospitalized_std),
            "icu": ('admitted_to_hospital', p_critical, critical_mean, critical_std),
            "dead": ('icu', p_dead, dead_mean, dead_std),
            "general_ward": ('icu', 1.-p_dead, recovered_mean, recovered_std),
            "hospital_recovered": ('admitted_to_hospital', 1.-p_critical, discharged_mean, discharged_std),
            "general_ward_recovered": ('general_ward', 1., discharged_mean, discharged_std)
        }

    def __call__(self, tspan, y0, dt=.05):
        """
        :arg tspan: A :class:`tuple` specifying the initial and final times.

        :arg y0: A :class:`dict` with the initial values
            (as :class:`numpy.ndarray`'s) for each of :attr:`compartments`.

        :returns: A :class:`~pydemic.simulation.StateLogger`. FIXME: maybe not?
        """

        influxes = super().__call__(tspan, y0, dt=dt)
        t = influxes.t

        # FIXME: remove when we move this back into the original file
        from pydemic.models.seirpp import convolve_pdf, convolve_survival
        from pydemic.models.seirpp import SimulationResult

        for key, (src, prob, mean, std) in self.readouts.items():
            influxes.y[key] = convolve_pdf(t, influxes.y[src], prob, mean, std)

        sol = SimulationResult(t, {})

        for key, val in influxes.y.items():
            if key not in ["susceptible", "population"]:
                sol.y[key] = np.cumsum(val, axis=0)
            else:
                sol.y[key] = val

        # FIXME: something wrong with this -- infectious > infected at small time
        sol.y['infectious'] = convolve_survival(t, influxes.y['infected'], 2, 5, 2)

        sol.y['critical'] = sol.y['icu'] - sol.y['general_ward'] - sol.y['dead']
        sol.y['ventilators'] = .73 * sol.y['critical']
        sol.y['hospitalized'] = sol.y['admitted_to_hospital'] - sol.y['hospital_recovered'] - sol.y['icu']
        sol.y['hospitalized'] += sol.y['general_ward'] - sol.y['general_ward_recovered']

        return sol
