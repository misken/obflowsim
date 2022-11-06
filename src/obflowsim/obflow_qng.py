import logging

import numpy as np
import networkx as nx

from obflow_sim import Config
from obflow_sim import PatientType

logger = logging.getLogger(__name__)

def unit_loads(config: Config):

    # Determine arrival rate to each unit by patient type

    # Start with spont_labor arrival stream
    spont_labor_rates = spont_labor_subrates(config)
    scheduled_rates = scheduled_subrates(config)
    non_delivery_rates = non_delivery_subrates(config)

    # Combine arrival rate dicts
    arrival_rates = spont_labor_rates | scheduled_rates | non_delivery_rates
    los_means = config.los_means

    # Compute overall load and traffic intensity at each unit
    load = {}
    for unit in config.locations:
        load[unit] = 0

    for pat_type, rate in arrival_rates.items():
        for unit in config.locations:
            if unit in los_means[pat_type]:
                load[unit] += rate * los_means[pat_type][unit]

    traffic_intensity = {}
    for unit_name, unit in config.locations.items():
        traffic_intensity[unit_name] = round(load[unit_name] / unit['capacity'], 3)

        if traffic_intensity[unit_name] >= 1.0:
            logger.warning(
                f"Traffic intensity = {traffic_intensity[unit_name]:.2f} for {unit_name} (load={load[unit_name]:.1f}, cap={unit['capacity']})")

    return load, traffic_intensity

def spont_labor_subrates(config: Config):

    # Determine arrival rate to each unit by patient type

    # Start with spont_labor arrival stream

    if config.rand_arrival_rates['spont_labor'] > 0.0 and config.rand_arrival_toggles['spont_labor'] > 0:
        spont_labor_rate = config.rand_arrival_rates['spont_labor']
    else:
        spont_labor_rate = 0.0

    spont_labor_subrate = {}

    pct_spont_labor_aug = config.branching_probabilities['pct_spont_labor_aug']
    pct_spont_labor_to_c = config.branching_probabilities['pct_spont_labor_to_c']
    pct_aug_labor_to_c = config.branching_probabilities['pct_aug_labor_to_c']

    # Type 1: random arrival spont labor, regular delivery, route = 1-2-4
    spont_labor_subrate[PatientType.RAND_SPONT_REG.value] = \
        (1 - pct_spont_labor_aug) * (1 - pct_spont_labor_to_c) * spont_labor_rate

    # Type 2: random arrival spont labor, C-section delivery, route = 1-3-2-4
    spont_labor_subrate[PatientType.RAND_SPONT_CSECT.value] = \
        (1 - pct_spont_labor_aug) * pct_spont_labor_to_c * spont_labor_rate

    # Type 3: random arrival augmented labor, regular delivery, route = 1-2-4
    spont_labor_subrate[PatientType.RAND_AUG_REG.value] = \
        pct_spont_labor_aug * (1 - pct_aug_labor_to_c) * spont_labor_rate

    # Type 4: random arrival augmented labor, C-section delivery, route = 1-3-2-4
    spont_labor_subrate[PatientType.RAND_AUG_CSECT.value] = \
        pct_spont_labor_aug * pct_aug_labor_to_c * spont_labor_rate

    return spont_labor_subrate

    # Determine mean LOS at each unit by patient type


    # Compute overall load and traffic intensity at each unit

def scheduled_subrates(config: Config):

    scheduled_subrate = {}
    pct_sched_ind_to_c = config.branching_probabilities['pct_sched_ind_to_c']

    if 'sched_csect' in config.schedules and config.sched_arrival_toggles['sched_csect'] > 0:
        tot_weekly_patients = np.sum(config.schedules['sched_csect'])
        scheduled_subrate[PatientType.SCHED_CSECT.value] = tot_weekly_patients / 168.0
    else:
        scheduled_subrate[PatientType.SCHED_CSECT.value] = 0.0

    if 'sched_induced_labor' in config.schedules and config.sched_arrival_toggles['sched_induced_labor'] > 0:
        tot_scheduled_induction_rate = \
            np.sum(config.schedules['sched_induced_labor']) / 168.0
    else:
        tot_scheduled_induction_rate = 0.0

        scheduled_subrate[PatientType.SCHED_IND_REG.value] = \
            (1 - pct_sched_ind_to_c) * tot_scheduled_induction_rate

        scheduled_subrate[PatientType.SCHED_IND_CSECT.value] = \
            pct_sched_ind_to_c * tot_scheduled_induction_rate

    return scheduled_subrate


def non_delivery_subrates(config: Config):

    non_delivery_subrate = {}

    if config.rand_arrival_toggles['non_delivery_ldr'] > 0:
        non_delivery_subrate[PatientType.RAND_NONDELIV_LDR.value] = config.rand_arrival_rates['non_delivery_ldr']
    else:
        non_delivery_subrate[PatientType.RAND_NONDELIV_LDR.value] = 0.0

    if config.rand_arrival_toggles['non_delivery_pp'] > 0:
        non_delivery_subrate[PatientType.RAND_NONDELIV_PP.value] = config.rand_arrival_rates['non_delivery_pp']
    non_delivery_subrate[PatientType.RAND_NONDELIV_PP.value] = 0.0

    return non_delivery_subrate


