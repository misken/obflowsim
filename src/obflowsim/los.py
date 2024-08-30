import pandas as pd

from obflowsim.simulate import Patient, PatientCareUnit, PatientFlowSystem
from obflowsim.obconstants import UnitName
from obflowsim.config import Config


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
