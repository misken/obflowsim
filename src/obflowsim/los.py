import copy
import json
from functools import partial
from typing import Tuple, Dict

import numpy.random
import pandas as pd

from obflowsim import obconstants as obconstants

from obflowsim.simulate import Patient, PatientCareUnit, PatientFlowSystem
from obflowsim.obconstants import UnitName
from obflowsim.config import Config
from networkx import DiGraph


# def los_blocking_adjustment(unit: PatientCareUnit, patient: Patient, planned_los: float, previous_unit_name: str):
#     if previous_unit_name != UnitName.ENTRY and patient.current_stop_num > 1:
#         G = patient.route_graph
#         los_adjustment_type = G[previous_unit_name][unit.name]['blocking_adjustment']
#         if los_adjustment_type == 'delay':
#             blocking_adj_los = max(0, planned_los - patient.wait_to_exit[patient.current_stop_num - 1])
#         else:
#             blocking_adj_los = planned_los
#     else:
#         blocking_adj_los = planned_los
#
#     return blocking_adj_los
#
#
# def los_discharge_adjustment(config: Config, pfs: PatientFlowSystem, patient: Patient,
#                              current_unit_name: str, planned_los: float,
#                              previous_unit_name: str):
#     if patient.current_stop_num > 1:
#         G = patient.route_graph
#         discharge_pdf = G[previous_unit_name][current_unit_name]['discharge_adjustment']
#         if discharge_pdf is not None:
#
#             sim_calendar = pfs.sim_calendar
#             now_datetime = sim_calendar.datetime(pfs.env.now)
#
#             # Get period of day of discharge
#             rg = config.rg['los']
#             discharge_period = rg.choice(discharge_pdf.index, discharge_pdf)
#             period_fraction = rg.random()
#
#             # Get datetime of initial discharge
#             initial_discharge_datetime = now_datetime + pd.Timedelta(planned_los, sim_calendar.base_time_unit)
#             initial_discharge_date = initial_discharge_datetime.date
#
#             new_discharge_datetime = initial_discharge_date + pd.Timedelta(discharge_period + period_fraction,
#                                                                            sim_calendar.base_time_unit)
#
#             discharge_adj_los = (new_discharge_datetime - now_datetime) / pd.Timedelta(1, sim_calendar.base_time_unit)
#         else:
#             discharge_adj_los = planned_los
#     else:
#         discharge_adj_los = planned_los
def mean_from_dist_params(dist_name: str, params: Tuple, kwparams):
    """
    Compute mean from distribution name and parameters - numpy based

    Parameters
    ----------
    dist_name
    params

    Returns
    -------

    """

    if dist_name == 'gamma':
        _shape = params[0]
        _scale = params[1]
        _mean = _shape * _scale
    elif dist_name == 'triangular':
        _left = params[0]
        _mode = params[1]
        _right = params[2]
        _mean = (_left + _mode + _right) / 3
    elif dist_name == 'normal':
        _mean = params[0]
    elif dist_name == 'exponential':
        _mean = params[0]
    elif dist_name == 'uniform':
        _left = params[0]
        _right = params[1]
        _mean = (_left + _right) / 2
    elif dist_name == 'choice':
        _a = params[0]
        _p = kwparams['p']
        _mean = sum(x * y for x, y in zip(_a, _p))
    else:
        raise ValueError(f'The {dist_name} distribution is not implemented yet for LOS modeling')

    return _mean


def create_los_partial(raw_los_dist_input: str, los_params: Dict, rg: numpy.random.Generator):
    """

    Parameters
    ----------
    raw_los_dist_input: str
    los_params: Dict
    rg

    Returns
    -------
    Updated DiGraph with planned_los attribute set to partial functions for LOS generation by pat type and edge.

    The planned_los attribute represents the planned LOS in the unit represented by the destination node of the edge.

    """

    # Replace all los_param use with literal values
    los_params_sorted = [key for key in los_params]
    los_params_sorted.sort(key=len, reverse=True)

    for param in los_params_sorted:
        los_dist_str = raw_los_dist_input.replace(param, str(los_params[param]))

    func_name = _convert_str_to_func_name(los_dist_str)

    if func_name in obconstants.ALLOWED_LOS_DIST_LIST:
        args, kwargs = _convert_str_to_args_and_kwargs(los_dist_str)
        partial_dist_func = partial(eval(f'rg.{func_name}'), *args, **kwargs)
        los_mean = mean_from_dist_params(func_name, args, kwargs)
    else:
        raise NameError(f"The use of '{func_name}' is not allowed")

    return partial_dist_func, los_mean

# def create_los_partials(raw_los_dists: Dict, los_params: Dict, rg: numpy.random.Generator):
#     """
#
#     Parameters
#     ----------
#     rg
#     raw_los_dists: Dict
#     los_params: Dict
#
#     Returns
#     -------
#     Dict of partial functions for LOS generation by pat type and unit
#
#     """
#
#     # Replace all los_param use with literal values
#     los_dists_str_json = json.dumps(raw_los_dists)
#
#     los_params_sorted = [key for key in los_params]
#     los_params_sorted.sort(key=len, reverse=True)
#
#     for param in los_params_sorted:
#         los_dists_str_json = los_dists_str_json.replace(param, str(los_params[param]))
#
#     los_dists_instantiated = json.loads(los_dists_str_json)
#
#     los_dists_partials = copy.deepcopy(los_dists_instantiated)
#     los_means = copy.deepcopy(los_dists_instantiated)
#     for key_pat_type in los_dists_partials:
#         for key_unit, raw_dist_str in los_dists_partials[key_pat_type].items():
#             func_name = _convert_str_to_func_name(raw_dist_str)
#             # Check for valid func name
#             if func_name in obconstants.ALLOWED_LOS_DIST_LIST:
#                 args, kwargs = _convert_str_to_args_and_kwargs(raw_dist_str)
#                 partial_dist_func = partial(eval(f'rg.{func_name}'), *args, **kwargs)
#                 los_dists_partials[key_pat_type][key_unit] = partial_dist_func
#                 los_means[key_pat_type][key_unit] = mean_from_dist_params(func_name, args, kwargs)
#             else:
#                 raise NameError(f"The use of '{func_name}' is not allowed")
#
#     return los_dists_partials, los_means
def _get_args_and_kwargs(*args, **kwargs):
    return args, kwargs


def _convert_str_to_args_and_kwargs(s: str):
    return eval(s.replace(s[:s.find('(')], '_get_args_and_kwargs'))


def _convert_str_to_func_name(s: str):
    return s[:s.find('(')]
