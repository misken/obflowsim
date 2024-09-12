from __future__ import annotations

import logging

import numpy as np
import networkx as nx

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from obflowsim.config import Config
    from obflowsim.simulate import PatientFlowSystem

from obflowsim.obconstants import PatientType
from obflowsim.obconstants import BASE_TIME_UNITS_PER_YEAR


logger = logging.getLogger(__name__)


def arrival_rates(config: Config):
    """
    Compute arrival rates by patient type

    Parameters
    ----------
    config : Config

    Returns
    -------
    Dict of patient type specific arrival rates

    """

    spont_labor_rates = spont_labor_subrates(config)
    scheduled_rates = scheduled_subrates(config)
    urgent_induction_rates = urgent_induction_subrates(config)
    non_delivery_rates = non_delivery_subrates(config)

    rates = spont_labor_rates | scheduled_rates | urgent_induction_rates | non_delivery_rates

    return rates


def spont_labor_subrates(config: Config):
    """
    Compute arrival rates by patient type for spontaneous labor arrival stream

    Parameters
    ----------
    config : Config

    Returns
    -------
    Dict of patient type specific arrival rates

    """

    # Determine arrival rate to each unit by patient type for the spont_labor arrival stream

    if config.rand_arrival_rates['spont_labor'] > 0.0 and config.rand_arrival_toggles['spont_labor'] > 0:
        spont_labor_rate = config.rand_arrival_rates['spont_labor']
    else:
        spont_labor_rate = 0.0

    spont_labor_subrate = {}

    pct_spont_labor_aug = config.branching_probabilities['pct_spont_labor_aug']
    pct_spont_labor_to_c = config.branching_probabilities['pct_spont_labor_to_c']
    pct_aug_labor_to_c = config.branching_probabilities['pct_aug_labor_to_c']

    # Type 1: random arrival spont labor, regular delivery, route = 1-2-4
    spont_labor_subrate[PatientType.RAND_SPONT_NAT.value] = \
        (1 - pct_spont_labor_aug) * (1 - pct_spont_labor_to_c) * spont_labor_rate

    # Type 2: random arrival spont labor, C-section delivery, route = 1-3-2-4
    spont_labor_subrate[PatientType.RAND_SPONT_CSECT.value] = \
        (1 - pct_spont_labor_aug) * pct_spont_labor_to_c * spont_labor_rate

    # Type 3: random arrival augmented labor, regular delivery, route = 1-2-4
    spont_labor_subrate[PatientType.RAND_AUG_NAT.value] = \
        pct_spont_labor_aug * (1 - pct_aug_labor_to_c) * spont_labor_rate

    # Type 4: random arrival augmented labor, C-section delivery, route = 1-3-2-4
    spont_labor_subrate[PatientType.RAND_AUG_CSECT.value] = \
        pct_spont_labor_aug * pct_aug_labor_to_c * spont_labor_rate

    return spont_labor_subrate


def urgent_induction_subrates(config: Config):
    """
    Compute arrival rates by patient type for urgent induction arrival stream

    Parameters
    ----------
    config : Config

    Returns
    -------
    Dict of patient type specific arrival rates

    """

    # Determine arrival rate to each unit by patient type

    # Start with spont_labor arrival stream

    if config.rand_arrival_rates['urgent_induced_labor'] > 0.0 and config.rand_arrival_toggles[
        'urgent_induced_labor'] > 0:
        urgent_induction_rate = config.rand_arrival_rates['urgent_induced_labor']
    else:
        urgent_induction_rate = 0.0

    urgent_induction_subrate = {}

    pct_urgent_induction_to_c = config.branching_probabilities['pct_urg_ind_to_c']

    # Type 8: urgent induction, regular delivery, route = 1-2-4
    urgent_induction_subrate[PatientType.URGENT_IND_NAT.value] = \
        (1 - pct_urgent_induction_to_c) * urgent_induction_rate

    # Type 9: urgent induction, C-section delivery, route = 1-3-2-4
    urgent_induction_subrate[PatientType.URGENT_IND_CSECT.value] = \
        pct_urgent_induction_to_c * urgent_induction_rate

    return urgent_induction_subrate


def scheduled_subrates(config: Config):
    """
    Compute arrival rates by patient type for scheduled arrival stream

    Parameters
    ----------
    config : Config

    Returns
    -------
    Dict of patient type specific arrival rates

    """

    scheduled_subrate = {}
    pct_sched_ind_to_c = config.branching_probabilities['pct_sched_ind_to_c']

    if 'sched_csect' in config.schedules and config.sched_arrival_toggles['sched_csect'] > 0:
        tot_weekly_schedc_patients = 0
        for sched_tuple in config.schedules['sched_csect']:
            tot_weekly_schedc_patients += sched_tuple[1]
        scheduled_subrate[PatientType.SCHED_CSECT.value] = tot_weekly_schedc_patients / 168.0
    else:
        scheduled_subrate[PatientType.SCHED_CSECT.value] = 0.0

    if 'sched_induced_labor' in config.schedules and config.sched_arrival_toggles['sched_induced_labor'] > 0:
        tot_weekly_sched_induction_patients = 0
        for sched_tuple in config.schedules['sched_induced_labor']:
            tot_weekly_sched_induction_patients += sched_tuple[1]
        tot_scheduled_induction_rate = tot_weekly_sched_induction_patients / 168.0
    else:
        tot_scheduled_induction_rate = 0.0

    scheduled_subrate[PatientType.SCHED_IND_NAT.value] = \
        (1 - pct_sched_ind_to_c) * tot_scheduled_induction_rate

    scheduled_subrate[PatientType.SCHED_IND_CSECT.value] = \
        pct_sched_ind_to_c * tot_scheduled_induction_rate

    return scheduled_subrate


def non_delivery_subrates(config: Config):
    """
    Compute arrival rates by patient type for non-delivered arrival stream

    Parameters
    ----------
    config : Config

    Returns
    -------
    Dict of patient type specific arrival rates

    """

    non_delivery_subrate = {}

    if config.rand_arrival_toggles['non_delivery_ldr'] > 0:
        non_delivery_subrate[PatientType.RAND_NONDELIV_LDR.value] = config.rand_arrival_rates['non_delivery_ldr']
    else:
        non_delivery_subrate[PatientType.RAND_NONDELIV_LDR.value] = 0.0

    if config.rand_arrival_toggles['non_delivery_pp'] > 0:
        non_delivery_subrate[PatientType.RAND_NONDELIV_PP.value] = config.rand_arrival_rates['non_delivery_pp']
    else:
        non_delivery_subrate[PatientType.RAND_NONDELIV_PP.value] = 0.0

    return non_delivery_subrate


def static_load_analysis(obsystem: PatientFlowSystem):
    """
    Compute offered loads and intensities to identify under capacitated systems.

    These calculations are based entirely on the simulation input values and NOT on any simulation results.

    Parameters
    ----------
    obsystem

    Returns
    -------
    Five `Dict` objects: load_unit, load_unit_ptype, traffic_intensity, annual_volume_ptype, annual_volume.

    - `load_unit` - offered load by patient care unit
    - `load_unit_ptype` - offered load by patient type at each patient care unit
    - `traffic_intensity` - traffic intensity by patient care unit
    - `annual_volume_ptype` - annualized volume by patient type
    - `annual_volume` - annualized volume of total, regular and c-section deliveries

    """

    # Determine arrival rate to each unit by patient type and mean los values
    config = obsystem.config
    arrival_rates_pattype = arrival_rates(config)
    route_graphs_pattype = obsystem.router.route_graphs

    # Compute overall volume
    annual_volume = {}
    annual_volume_ptype = {}

    for pat_type, arr_rate in arrival_rates_pattype.items():
        annual_volume_ptype[pat_type] = arr_rate * BASE_TIME_UNITS_PER_YEAR[config.base_time_unit]

    annual_volume['reg_births'] = annual_volume_ptype['RAND_SPONT_NAT'] + \
                                  annual_volume_ptype['RAND_AUG_NAT'] + \
                                  annual_volume_ptype['SCHED_IND_NAT'] + \
                                  annual_volume_ptype['URGENT_IND_NAT']

    annual_volume['csect_births'] = annual_volume_ptype['RAND_SPONT_CSECT'] + \
                                    annual_volume_ptype['RAND_AUG_CSECT'] + \
                                    annual_volume_ptype['SCHED_IND_CSECT'] + \
                                    annual_volume_ptype['URGENT_IND_CSECT'] + \
                                    annual_volume_ptype['SCHED_CSECT']

    annual_volume['total_births'] = annual_volume['reg_births'] + annual_volume['csect_births']

    annual_volume['non_delivered'] = annual_volume_ptype['RAND_NONDELIV_LDR'] + annual_volume_ptype['RAND_NONDELIV_PP']

    # Compute overall load and traffic intensity at each unit
    load_unit = {}
    load_unit_ptype = {}

    # Initialize loads to zero
    for unit in config.locations:
        load_unit[unit] = 0.0
        for pat_type in arrival_rates_pattype:
            ptype_key = f'{unit}_{pat_type}'
            load_unit_ptype[ptype_key] = 0.0

    # Compute loads by unit and by unit, patient type combo
    for pat_type, rate in arrival_rates_pattype.items():
        if pat_type in route_graphs_pattype:
            route_graph = route_graphs_pattype[pat_type]
            for u, v, data in route_graph.edges(data=True):
                edge = route_graph.edges[u, v]
                unit = v  # Destination node is second component of edge tuple
                if 'los_mean' in edge:
                    los_mean = edge['los_mean']
                    ptype_key = f'{unit}_{pat_type}'
                    load_unit[unit] += rate * los_mean
                    load_unit_ptype[ptype_key] = rate * los_mean

    # Compute traffic intensity based on load and capacity
    traffic_intensity = {}
    for unit_name, unit in config.locations.items():
        traffic_intensity[unit_name] = round(load_unit[unit_name] / unit['capacity'], 3)
        if traffic_intensity[unit_name] >= 1.0:
            logger.warning(
                f"Traffic intensity = {traffic_intensity[unit_name]:.2f} for {unit_name} (load={load_unit[unit_name]:.1f}, cap={unit['capacity']})")

    return {'load_unit': load_unit,
            'load_unit_type': load_unit_ptype,
            'intensity_unit': traffic_intensity,
            'annual_volume_type': annual_volume_ptype,
            'annual_volume': annual_volume}
