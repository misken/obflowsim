from __future__ import annotations

import sys
import logging
from logging import Logger
from pathlib import Path
import argparse

from numpy.typing import (
    NDArray,
)

# if TYPE_CHECKING and TYPE_CHECKING != 'SPHINX':  # Avoid circular import
from simpy.core import Environment

import pandas as pd
# import numpy as np
import simpy
# from numpy.random import default_rng
# from numpy.random import Generator
# import json

import obflowsim.io as obio
import obflowsim.stats as obstat
import obflowsim.obqueueing as obq
import obflowsim.config as obconfig
from obflowsim.obconstants import ArrivalType, PatientType, UnitName, ATT_GET_BED, ATT_RELEASE_BED, MARKED_PATIENT
from obflowsim.clock_tools import SimCalendar
from obflowsim.routing import StaticRouter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obflowsim.config import Config


# TODO - make sure all docstrings are complete


class PatientFlowSystem:
    """
    Acts as a container for inputs such as the config and the `SimCalendar` as well as for
    system objects created from these inputs and (maybe) timestamp dicts

    Instead of passing around the above individually, just pass this system object around

    """

    def __init__(self, env: Environment, config: Config, sim_calendar: SimCalendar):
        self.env = env
        self.config = config
        self.sim_calendar = sim_calendar
        self.router = None  # Currently only a single router can be registered

        # Create entry and exit nodes
        self.entry = EntryNode(self.env)
        self.exit = ExitNode(self.env)

        # Create units container and individual patient care units
        self.patient_care_units = {}
        for location, data in config.locations.items():
            self.patient_care_units[location] = PatientCareUnit(env, name=location, capacity=data['capacity'])

        # Create list to hold timestamps dictionaries (one per patient stop)
        self.stops_timestamps_list = []


class Patient:
    """
    These are the *entities* who flow through a *patient flow system* (``PatientFlowSystem``)
    consisting of a network of *patient care units* (``PatientCareUnit``).

    """

    def __init__(self, patient_id: str, arrival_type: ArrivalType,
                 arr_time: float, patient_flow_system: PatientFlowSystem, entry_delay: float = 0):
        """

        Parameters
        ----------
        patient_id : str
            Unique identifier for each patient instance
        arrival_type : ArrivalType
            Unique identifier for arrival streams
        arr_time : float
            Simulation time at which patient arrives to system
        patient_flow_system : PatientFlowSystem
            System being modeled and to which new patient arrives
        entry_delay : float
            Length of time to hold patient in ENTRY node before routing to first location
        """
        self.patient_id = patient_id
        self.arrival_type = arrival_type
        self.system_arrival_ts = arr_time + entry_delay
        self.entry_delay = entry_delay
        self.pfs = patient_flow_system

        # Determine patient type
        self.patient_type = self.assign_patient_type()

        # Initialize unit stop attributes
        self.current_stop_num = -1

        # Get route
        self.route_graph = self.pfs.router.create_route(self)

        self.bed_requests = {}  # simpy request() events with unit name as keys
        self.unit_stops = []  # unit name
        self.planned_los = []
        self.adjusted_los = []
        self.request_entry_ts = []
        self.entry_ts = []
        self.wait_to_enter = []
        self.request_exit_ts = []
        self.exit_ts = []
        self.blocked = []
        self.wait_to_exit = []
        self.skipped_edge = []
        self.system_exit_ts = None  # redundant, get from exit_ts last element

        # Initiate process of patient entering system
        self.pfs.env.process(
            self.pfs.entry.put(self, self.pfs))

    def get_route_length_nodes(self):
        """
        Get the number of nodes in the current route including entry and exit nodes.

        Returns
        -------
        int: number of nodes
        """
        return len(self.route_graph.edges) + 1

    def get_previous_unit_name(self):

        # If this is first stop, previous unit was Entry node
        if self.current_stop_num == 1:
            previous_unit_name = UnitName.ENTRY.value
        else:  # Not the first stop
            if self.skipped_edge[self.current_stop_num - 1] is None:
                previous_unit_name = self.unit_stops[self.current_stop_num - 1]
            else:
                previous_unit_name = self.skipped_edge[self.current_stop_num - 1][1]

        return previous_unit_name


    def get_previous_unit(self):

        # If this is first stop, previous unit was Entry node
        if self.current_stop_num == 1:
            previous_unit = self.pfs.entry
        else:  # Not the first stop
            previous_unit_name = self.get_previous_unit_name()
            previous_unit = self.pfs.patient_care_units[previous_unit_name]

        return previous_unit

    def get_current_unit_name(self):

        if self.current_stop_num == 0:
            current_unit_name = UnitName.ENTRY.value
        else:  # Not the first stop
            current_unit_name = self.unit_stops[self.current_stop_num]

        return current_unit_name

    def get_current_unit(self):

        if self.current_stop_num == 0:
            current_unit = self.pfs.entry
        else:  # Not the first stop
            current_unit_name = self.get_current_unit_name()
            current_unit = self.pfs.patient_care_units[current_unit_name]

        return current_unit

    def get_current_route_edge(self):
        """
        Get edge containing current unit as destination node.

        Returns
        -------
        `edge`

        """

        previous_unit_name = self.get_previous_unit_name()
        current_unit_name = self.get_current_unit_name()
        current_route_edge = self.route_graph.edges[previous_unit_name, current_unit_name]

        return current_route_edge

    def assign_patient_type(self):
        arr_stream_rg = self.pfs.config.rg['arrivals']
        if self.arrival_type == ArrivalType.SPONT_LABOR.value:
            # Determine if labor augmented or not
            if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_spont_labor_aug']:
                # Augmented labor
                if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_aug_labor_to_c']:
                    return PatientType.RAND_AUG_CSECT.value
                else:
                    return PatientType.RAND_AUG_NAT.value
            else:
                # Labor not augmented
                if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_spont_labor_to_c']:
                    return PatientType.RAND_SPONT_CSECT.value
                else:
                    return PatientType.RAND_SPONT_NAT.value
        elif self.arrival_type == ArrivalType.NON_DELIVERY_LDR.value:
            return PatientType.RAND_NONDELIV_LDR.value
        elif self.arrival_type == ArrivalType.NON_DELIVERY_PP.value:
            return PatientType.RAND_NONDELIV_PP.value
        elif self.arrival_type == ArrivalType.URGENT_INDUCED_LABOR.value:
            if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_urg_ind_to_c']:
                return PatientType.URGENT_IND_CSECT.value
            else:
                return PatientType.URGENT_IND_NAT.value
        elif self.arrival_type == ArrivalType.SCHED_CSECT.value:
            # Scheduled c-section
            return PatientType.SCHED_CSECT.value
        else:
            # Determine if scheduled induction ends up with c-section
            if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_sched_ind_to_c']:
                return PatientType.SCHED_IND_CSECT.value
            else:
                return PatientType.SCHED_IND_NAT.value

    def append_empty_unit_stop(self):
        """
        Append None placeholders to patient flow lists

        Returns
        -------

        """
        self.unit_stops.append(None)
        self.planned_los.append(None)
        self.adjusted_los.append(None)
        self.request_entry_ts.append(None)
        self.entry_ts.append(None)
        self.wait_to_enter.append(None)
        self.request_exit_ts.append(None)
        self.exit_ts.append(None)
        self.blocked.append(None)
        self.wait_to_exit.append(None)
        self.skipped_edge.append(None)

    def __repr__(self):
        return "patient id: {}, patient_type: {}, time: {}". \
            format(self.patient_id, self.patient_type, self.system_arrival_ts)


class EntryNode:

    def __init__(self, env: Environment, name: str = UnitName.ENTRY):
        self.env = env
        self.name = name

        # Statistical accumulators
        self.num_entries = 0
        self.num_exits = 0
        self.tot_occ_time = 0.0
        self.last_entry_ts = None
        self.last_exit_ts = None

        # Create list to hold occupancy tuples (time, occ)
        self.occupancy_list = [(0.0, 0.0)]

    def put(self, patient: Patient, obsystem: PatientFlowSystem):
        """
        A process method called when entry to the PatientFlowSystem is requested.

        Parameters
        ----------
        patient : Patient object
            the patient requesting the bed
        obsystem : PatientFlowSystem object

        """

        if patient.patient_id == MARKED_PATIENT:
            pass

        logging.debug(
            f"{self.env.now:.4f}: {patient.patient_id} enters {self.name} node.")

        # Update unit attributes
        self.num_entries += 1
        self.last_entry_ts = self.env.now
        # Increment occupancy
        self.inc_occ()

        # Update patient attributes
        patient.current_stop_num = 0
        csn = patient.current_stop_num
        patient.append_empty_unit_stop()
        patient.unit_stops[csn] = UnitName.ENTRY.value
        patient.request_entry_ts[csn] = self.env.now
        patient.entry_ts[csn] = self.env.now
        patient.planned_los[csn] = patient.entry_delay
        patient.adjusted_los[csn] = patient.entry_delay

        # Wait for any entry_delay needed
        yield self.env.timeout(patient.entry_delay)

        logging.debug(
            f"{self.env.now:.4f}: {patient.patient_id} ready to leave {self.name} node.")

        # Determine first stop in route and try to get a bed in that unit
        next_step = patient.pfs.router.get_next_step(patient)
        next_unit_name = next_step[1]
        self.env.process(obsystem.patient_care_units[next_unit_name].put(patient, obsystem))

    def inc_occ(self, increment=1):
        """Update occupancy - increment by 1"""
        prev_occ = self.occupancy_list[-1][1]
        new_ts_occ = (self.env.now, prev_occ + increment)
        self.occupancy_list.append(new_ts_occ)

    # Can't decrement occ in ENTRY until first patient care unit obtained
    def dec_occ(self, decrement=1):
        """Update occupancy - decrement by 1"""
        prev_occ = self.occupancy_list[-1][1]
        new_ts_occ = (self.env.now, prev_occ - decrement)
        self.occupancy_list.append(new_ts_occ)


class ExitNode:
    """

    """

    def __init__(self, env: Environment, name: str = UnitName.EXIT):

        self.env = env
        self.name = name

        # Statistical accumulators
        self.num_entries = 0
        self.num_exits = 0
        self.last_entry_ts = None
        self.last_exit_ts = None

    def put(self, patient: Patient, obsystem: PatientFlowSystem):
        """
        A process method called when exit from the PatientFlowSystem is requested.

        Parameters
        ----------
        patient : Patient object
            the patient requesting the bed
        obsystem : PatientFlowSystem object

        """

        # Update unit attributes
        self.num_entries += 1
        self.last_entry_ts = self.env.now
        self.last_exit_ts = self.env.now

        # Increment stop number for this patient
        patient.current_stop_num += 1
        csn = patient.current_stop_num
        patient.append_empty_unit_stop()  # Appends None to all patient flow related lists
        patient.unit_stops[csn] = UnitName.EXIT
        patient.planned_los[csn] = 0.0
        patient.adjusted_los[csn] = 0.0
        patient.request_entry_ts[csn] = self.env.now
        patient.entry_ts[csn] = self.env.now
        patient.wait_to_enter[csn] = 0.0
        patient.request_exit_ts[csn] = self.env.now
        patient.exit_ts[csn] = self.env.now
        patient.wait_to_exit[csn] = 0.0

        # Create dictionaries of timestamps for patient_stop log
        for stop in range(len(patient.unit_stops)):
            if patient.unit_stops[stop] is not None:
                # noinspection PyUnresolvedReferences
                try:
                    timestamps = {'patient_id': patient.patient_id,
                                  'patient_type': patient.patient_type,
                                  'arrival_type': patient.arrival_type,
                                  'unit': patient.unit_stops[stop],
                                  'request_entry_ts': patient.request_entry_ts[stop],
                                  'entry_ts': patient.entry_ts[stop],
                                  'request_exit_ts': patient.request_exit_ts[stop],
                                  'exit_ts': patient.exit_ts[stop],
                                  'planned_los': patient.planned_los[stop],
                                  'adjusted_los': patient.adjusted_los[stop],
                                  'entry_tryentry': patient.entry_ts[stop] - patient.request_entry_ts[stop],
                                  'tryexit_entry': patient.request_exit_ts[stop] - patient.entry_ts[stop],
                                  'exit_tryexit': patient.exit_ts[stop] - patient.request_exit_ts[stop],
                                  'exit_enter': patient.exit_ts[stop] - patient.entry_ts[stop],
                                  'exit_tryenter': patient.exit_ts[stop] - patient.request_entry_ts[stop],
                                  'wait_to_enter': patient.wait_to_enter[stop],
                                  'wait_to_exit': patient.wait_to_exit[stop],
                                  'bwaited_to_enter': patient.entry_ts[stop] > patient.request_entry_ts[stop],
                                  'bwaited_to_exit': patient.exit_ts[stop] > patient.request_exit_ts[stop]}
                except TypeError:
                    raise TypeError(f'Unable to create timestamps dict for stop {stop} for patient {patient}.')

                obsystem.stops_timestamps_list.append(timestamps)

        self.num_exits += 1
        logging.debug(
            f"{self.env.now:.4f}: {patient.patient_id} exited system at {self.env.now:.2f}.")


class PatientCareUnit:
    """ Models an OB unit with fixed capacity.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        name : str
            unit name
        capacity : integer (or None)
            Number of beds. Use None for infinite capacity.

    """

    def __init__(self, env: Environment, name: str, capacity: int = simpy.core.Infinity):

        self.env = env
        self.name = name
        self.capacity = capacity

        # Use a simpy Resource as one of the class instance members
        self.unit = simpy.Resource(env, capacity)

        # Statistical accumulators
        self.num_entries = 0
        self.num_exits = 0
        self.tot_occ_time = 0.0
        self.last_entry_ts = None
        self.last_exit_ts = None

        # Create list to hold occupancy tuples (time, occ)
        self.occupancy_list = [(0.0, 0.0)]

    def put(self, patient: Patient, pfs: PatientFlowSystem):
        """
        A process method called when a patient wants to enter this patient care unit.

        Parameters
        ----------
        patient : OBPatient object
            the patient requesting the bed
        pfs : OBSystem object

        """

        if patient.patient_id == MARKED_PATIENT:
            pass

        # We are trying to leave the unit patient currently in to visit another unit (this unit)
        logging.debug(
            f"{self.env.now:.4f}: {patient.patient_id} trying to get {self.name} for stop_num {patient.current_stop_num + 1}")

        # TODO: data structure for tracking patient progress through system

        patient.request_exit_ts[patient.current_stop_num] = self.env.now
        # TODO: Should I append to the patient flow lists now or wait until I get into next unit?
        # patient.append_empty_unit_stop()  # Appends None to all patient flow related lists

        request_entry_ts = self.env.now   # Local variable version
        # patient.request_entry_ts[patient.current_stop_num + 1] = self.env.now  # Patient list version

        current_unit_name = patient.get_current_unit_name()
        next_unit_name = self.name
        got_bed = False

        # This is the arc terminating at ths unit
        try:
            assert current_unit_name != next_unit_name
        except AssertionError:
            print(f'Patient {patient.patient_id} has invalid incoming route edge.')

        if patient.skipped_edge[patient.current_stop_num] is None:
            incoming_route_edge = (current_unit_name, next_unit_name, patient.route_graph.edges[current_unit_name, next_unit_name])
        else:
            skipped_unit_name = patient.skipped_edge[patient.current_stop_num][1]
            incoming_route_edge = (skipped_unit_name, next_unit_name,
                                   patient.route_graph.edges[skipped_unit_name, next_unit_name])


        # Sample from LOS distribution for this arc and patient type
        planned_los = incoming_route_edge[2]['planned_los']()

        # Request bed if indicated
        if incoming_route_edge[2][ATT_GET_BED]:
            # Request a bed
            bed_request = self.unit.request()
            # Store bed request and timestamp in patient's request lists
            patient.bed_requests[self.name] = bed_request

            # Yield until we get a bed or our planned los has elapsed due to being blocked
            get_bed = yield bed_request | self.env.timeout(planned_los)

            # Check if we got a bed before our los has elapsed
            if bed_request in get_bed:
                got_bed = True  # Good to continue processing at this patient care unit
            else:
                # Our LOS has elapsed while we were blocked trying to enter ths unit.
                # Need to get rid of last bed request
                patient.bed_requests.pop(self.name)
                # Determine next stop in route
                current_edge_num = incoming_route_edge[2]['edge_num']
                current_unit = patient.get_current_unit()
                next_route_edge = patient.pfs.router.get_next_step(patient, after=current_edge_num,
                                                                   unit=next_unit_name)
                next_unit_name = next_route_edge[1]
                patient.skipped_edge[patient.current_stop_num] = incoming_route_edge

                if next_unit_name != UnitName.EXIT:
                    # Try to get bed in next unit
                    self.env.process(pfs.patient_care_units[next_unit_name].put(patient, pfs))
                else:
                    # Patient is ready to exit system

                    # Release the bed
                    if current_unit_name in patient.bed_requests and incoming_route_edge[2][ATT_RELEASE_BED]:
                        # Release the previous bed
                        current_unit.unit.release(patient.bed_requests[current_unit_name])
                        unit_released = patient.bed_requests.pop(current_unit_name)

                    try:
                        assert not patient.bed_requests
                    except AssertionError:
                        print(f'Patient {patient.patient_id} trying to exit with bed requests.')

                    # Accumulate total time this unit occupied and other unit attributes
                    current_unit.tot_occ_time += \
                        self.env.now - patient.entry_ts[patient.current_stop_num]
                    current_unit.num_exits += 1
                    current_unit.last_exit_ts = self.env.now

                    # Decrement occupancy in this unit since bed now released
                    current_unit.dec_occ()

                    patient.request_exit_ts[patient.current_stop_num] = self.env.now
                    patient.exit_ts[patient.current_stop_num] = self.env.now

                    # Send patient to Exit node
                    pfs.exit.put(patient, pfs)

        if got_bed or not incoming_route_edge[2][ATT_GET_BED]:
            # Seized a bed if needed. Update patient flow attributes for this stop
            # Increment stop number for this patient now, only after we are sure we are actually entering this unit.
            if patient.patient_id == MARKED_PATIENT:
                pass

            patient.current_stop_num += 1
            patient.append_empty_unit_stop()  # Appends None to all patient flow related lists
            patient.request_entry_ts[patient.current_stop_num] = request_entry_ts

            csn = patient.current_stop_num
            patient.unit_stops[csn] = self.name
            previous_unit_name = patient.get_previous_unit_name()
            previous_unit = patient.get_previous_unit()

            patient.entry_ts[csn] = self.env.now
            patient.wait_to_enter[csn] = self.env.now - patient.request_exit_ts[patient.current_stop_num - 1]
            if patient.wait_to_enter[csn] > 0:
                patient.blocked[csn] = 1
            else:
                patient.blocked[csn] = 0


            # Update unit attributes
            self.num_entries += 1
            self.last_entry_ts = self.env.now

            # Increment occupancy in this unit
            self.inc_occ()

            # Update stats for previous unit.
            patient.exit_ts[csn - 1] = self.env.now
            patient.wait_to_exit[csn - 1] = \
                self.env.now - patient.request_exit_ts[csn - 1]
            # Accumulate total time previous unit occupied and other unit attributes
            previous_unit.tot_occ_time += \
                self.env.now - patient.entry_ts[csn - 1]
            previous_unit.num_exits += 1
            previous_unit.last_exit_ts = self.env.now
            # Decrement occupancy in previous unit (ignores patient who keeps bed in previous unit since the occupancy
            # tally is not used for checking if space available in unit - resource requests are used)
            previous_unit.dec_occ()

            # Check if we have a bed from a previous stay and release it if we do and want to release it.
            if previous_unit_name in patient.bed_requests and incoming_route_edge[2][ATT_RELEASE_BED]:
                # Release the previous bed
                previous_unit.unit.release(patient.bed_requests[previous_unit_name])
                # What happens to the reference in patient.bed_requests[]?
                unit_released = patient.bed_requests.pop(previous_unit_name)

            logging.debug(f"{self.env.now:.4f}: {patient.patient_id} entering {self.name} at {self.env.now}")

            # Do any blocking related los adjustments.
            blocking_adj_los = self.los_blocking_adjustment(patient, planned_los, previous_unit_name)

            # Do discharge timing related los adjustments
            adjusted_los = self.los_discharge_adjustment(pfs.config, pfs, patient,
                                                         blocking_adj_los, previous_unit_name)

            # Update los related patient attributes
            patient.planned_los[patient.current_stop_num] = planned_los
            patient.adjusted_los[patient.current_stop_num] = adjusted_los

            # Wait for LOS to elapse
            yield self.env.timeout(adjusted_los)

            # Determine next stop in route
            outgoing_route_edge = patient.pfs.router.get_next_step(patient)
            next_unit_name = outgoing_route_edge[1]

            if next_unit_name != UnitName.EXIT:
                # Try to get bed in next unit
                self.env.process(pfs.patient_care_units[next_unit_name].put(patient, pfs))
            else:
                # Patient is ready to exit system

                # Release the bed
                if self.name in patient.bed_requests and incoming_route_edge[2][ATT_RELEASE_BED]:
                    # Release the previous bed
                    self.unit.release(patient.bed_requests[self.name])
                    unit_released = patient.bed_requests.pop(self.name)

                try:
                    assert not patient.bed_requests
                except AssertionError:
                    print(f'Patient {patient.patient_id} trying to exit with bed requests.')

                # Accumulate total time this unit occupied and other unit attributes
                self.tot_occ_time += \
                    self.env.now - patient.entry_ts[patient.current_stop_num]
                self.num_exits += 1
                self.last_exit_ts = self.env.now

                # Decrement occupancy in this unit since bed now released
                self.dec_occ()

                patient.request_exit_ts[patient.current_stop_num] = self.env.now
                patient.exit_ts[patient.current_stop_num] = self.env.now

                # Send patient to Exit node
                pfs.exit.put(patient, pfs)

    def inc_occ(self, increment=1):
        """Update occupancy - increment by 1"""
        prev_occ = self.occupancy_list[-1][1]
        new_ts_occ = (self.env.now, prev_occ + increment)
        self.occupancy_list.append(new_ts_occ)

    def dec_occ(self, decrement=1):
        """Update occupancy - decrement by 1"""
        prev_occ = self.occupancy_list[-1][1]
        new_ts_occ = (self.env.now, prev_occ - decrement)
        self.occupancy_list.append(new_ts_occ)

    def basic_flow_stats_msg(self):
        """ Compute entries, exits, avg los and create summary message.

        Returns
        -------
        str
            Message with basic stats
        """

        if self.num_exits > 0:
            alos = self.tot_occ_time / self.num_exits
        else:
            alos = 0

        msg = "{:6}:\t Entries ={:7}, Exits ={:7}, Occ ={:4}, ALOS={:4.2f}". \
            format(self.name, self.num_entries, self.num_exits,
                   self.unit.count, alos)
        return msg

    def los_blocking_adjustment(self, patient: Patient, planned_los: float, previous_unit_name: str):
        if previous_unit_name != UnitName.ENTRY and patient.current_stop_num > 1:
            G = patient.route_graph

            try:
                assert (previous_unit_name, self.name) in G.edges
            except AssertionError:
                print(f'{(previous_unit_name, self.name)} not in G for {patient.patient_type}')

            los_adjustment_type = G[previous_unit_name][self.name]['blocking_adjustment']
            if los_adjustment_type == 'delay':
                blocking_adj_los = max(0, planned_los - patient.wait_to_exit[patient.current_stop_num - 1])
            else:
                blocking_adj_los = planned_los
        else:
            blocking_adj_los = planned_los

        return blocking_adj_los

    def los_discharge_adjustment(self, config: Config, pfs: PatientFlowSystem, patient: Patient,
                                 planned_los: float,
                                 previous_unit_name: str):
        if patient.current_stop_num > 1:
            G = patient.route_graph
            discharge_pdf = G[previous_unit_name][self.name]['discharge_adjustment']
            if discharge_pdf is not None:

                sim_calendar = pfs.sim_calendar
                now_datetime = sim_calendar.datetime(pfs.env.now)

                # Get period of day of discharge
                rg = config.rg['los']
                discharge_period = rg.choice(discharge_pdf.index, p=discharge_pdf['p'].values)
                period_fraction = rg.random()

                # Get datetime of initial discharge
                initial_discharge_datetime = now_datetime + pd.Timedelta(planned_los, sim_calendar.base_time_unit)
                initial_discharge_date = pd.Timestamp(initial_discharge_datetime.date())

                new_discharge_datetime = initial_discharge_date + pd.Timedelta(discharge_period + period_fraction,
                                                                               sim_calendar.base_time_unit)

                if new_discharge_datetime < now_datetime:
                    # Time travel to past not allowed
                    discharge_adj_los = planned_los
                else:
                    discharge_adj_los = (new_discharge_datetime - now_datetime) / pd.Timedelta(1,
                                                                                        sim_calendar.base_time_unit)
            else:
                discharge_adj_los = planned_los
        else:
            discharge_adj_los = planned_los

        return discharge_adj_los


class PatientPoissonArrivals:
    """ Generates patients according to a stationary Poisson process with specified rate.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        arrival_stream_uid : ArrivalType
            unique name of random arrival stream
        arrival_rate : float
            Poisson arrival rate (expected number of arrivals per unit time)
        arrival_stream_rg : numpy.random.Generator
            used for interarrival time generation
        stop_time : float
            Stops generation at the stoptime. (default Infinity)
        max_arrivals : int
            Stops generation after max_arrivals. (default Infinity)
        patient_flow_system : PatientFlowSystem into which the arrival is inserted
            This allows us to kick off a patient flowing through the system

        # TODO: Decouple the PP generator from the actual creation of model specific entities
    """

    def __init__(self, env, arrival_stream_uid: ArrivalType, arrival_rate: float, arrival_stream_rg,
                 stop_time=simpy.core.Infinity, max_arrivals=simpy.core.Infinity,
                 patient_flow_system=None):

        # Parameter attributes
        self.env = env
        self.arrival_stream_uid = arrival_stream_uid
        self.arr_rate = arrival_rate
        self.arr_stream_rg = arrival_stream_rg
        self.stop_time = stop_time
        self.max_arrivals = max_arrivals
        self.patient_flow_system = patient_flow_system

        # State attributes
        self.num_patients_created = 0

        # Trigger the run() method and register it as a SimPy process
        env.process(self.run())

    def run(self):
        """
        Generate entities until stopping condition met
        """

        # Main entity creation loop that terminates when stoptime reached
        while self.env.now < self.stop_time and \
                self.num_patients_created < self.max_arrivals:
            # Compute next interarrival time
            iat = self.arr_stream_rg.exponential(1.0 / self.arr_rate)
            # Delay until time for next arrival
            yield self.env.timeout(iat)
            self.num_patients_created += 1

            new_entity_id = f'{self.arrival_stream_uid}_{self.num_patients_created}'

            if self.patient_flow_system is not None:
                new_patient = Patient(new_entity_id, self.arrival_stream_uid,
                                      self.env.now, self.patient_flow_system)

                logging.debug(
                    f"{self.env.now:.4f}: {new_patient.patient_id} created at {self.env.now:.4f} ({self.patient_flow_system.sim_calendar.now()}).")


class PatientGeneratorWeeklyStaticSchedule:
    """ Generates patients according to a repeating one week schedule

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment


    """

    def __init__(self, env, arrival_stream_uid: ArrivalType,
                 schedule: NDArray,
                 stop_time: float = simpy.core.Infinity, max_arrivals: int = simpy.core.Infinity,
                 patient_flow_system: PatientFlowSystem = None):

        # Parameter attributes
        self.env = env
        self.arrival_stream_uid = arrival_stream_uid
        self.schedule = schedule
        # self.arr_stream_rg = arrival_stream_rg
        self.stop_time = stop_time
        self.max_arrivals = max_arrivals
        self.patient_flow_system = patient_flow_system

        # State attributes
        self.num_patients_created = 0

        # Trigger the run() method and register it as a SimPy process
        env.process(self.run())

    def run(self):
        """
        Generate patients.
        """

        base_time_unit = self.patient_flow_system.sim_calendar.base_time_unit
        weekly_cycle_length = \
            pd.to_timedelta(1, unit='w') / pd.to_timedelta(1, unit=base_time_unit)

        # Main generator loop that terminates when stopping condition reached
        while self.env.now < self.stop_time and self.num_patients_created < self.max_arrivals:

            # Create schedule patients and send them to ENTRY to wait until their scheduled procedure time
            for (time_of_week, num_sched) in self.schedule:
                for patient in range(num_sched):
                    self.num_patients_created += 1
                    new_entity_id = f'{self.arrival_stream_uid}_{self.num_patients_created}'

                    # Generate new patient
                    if self.patient_flow_system is not None:
                        new_patient = Patient(new_entity_id, self.arrival_stream_uid,
                                              self.env.now, self.patient_flow_system, entry_delay=time_of_week)

                        logging.debug(
                            f"{self.env.now:.4f}: {new_patient.patient_id} created at {self.env.now:.4f} ({self.patient_flow_system.sim_calendar.now()}).")

            # Yield until beginning of next weekly cycle
            yield self.env.timeout(weekly_cycle_length)


def process_command_line(argv=None):
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='obflowsim',
                                     description='Run inpatient OB simulation')

    # Add arguments
    parser.add_argument(
        "config", type=str,
        help="Configuration file containing input parameter arguments and values"
    )

    parser.add_argument("--loglevel", default='WARNING',
                        help="Use valid values for logging package")

    # do the parsing
    args = parser.parse_args(argv)

    return args


def simulate(config: Config, rep_num: int):
    """
    Run one replication of simulation model.

    Parameters
    ----------
    config : Config
    rep_num : int, simulation replication number

    Returns
    -------
    Dict: Summary statistics for the replication

    """

    # Initialize a simulation environment and calendar
    env = simpy.Environment()
    sim_calendar = SimCalendar(env, config)

    # Create an OB System
    obsystem = PatientFlowSystem(env, config, sim_calendar)

    # Create router and register it with the PatientFlowSystem
    router = StaticRouter(env, obsystem)
    obsystem.router = router

    # Create patient generators for random arrivals
    patient_generators_poisson = {}
    for arrival_stream_uid, arr_rate in config.rand_arrival_rates.items():
        # Check if this arrival stream is enabled
        if arr_rate > 0.0 and config.rand_arrival_toggles[arrival_stream_uid] > 0:
            # TODO - create patient generator for urgent inductions. For now, we'll ignore these patient types.
            if arrival_stream_uid == ArrivalType.URGENT_INDUCED_LABOR:
                logging.warning(
                    'Urgent inductions are not yet implemented. No arrivals will be generated for this stream.')
            else:
                patient_generator = PatientPoissonArrivals(env, arrival_stream_uid, arr_rate, config.rg['arrivals'],
                                                           stop_time=config.run_time, max_arrivals=config.max_arrivals,
                                                           patient_flow_system=obsystem)

                patient_generators_poisson[arrival_stream_uid] = patient_generator

    # Create patient generators for scheduled c-sections and scheduled inductions
    patient_generators_scheduled = {}
    for sched_id, schedule in config.schedules.items():
        if len(schedule) > 0 and config.sched_arrival_toggles[sched_id] > 0:
            patient_generators_scheduled[sched_id] = \
                PatientGeneratorWeeklyStaticSchedule(
                    env, sched_id, config.schedules[sched_id],
                    stop_time=config.run_time, max_arrivals=config.max_arrivals, patient_flow_system=obsystem)

    # Check for undercapacitated system and compute basic load stats
    static_load_summary = obq.static_load_analysis(obsystem)
    logging.info(
        f"{0.0:.4f}: annual_volume\n{static_load_summary['annual_volume']}).")
    logging.info(
        f"{0.0:.4f}: annual_volume_type\n{static_load_summary['annual_volume_type']}).")
    logging.info(
        f"{0.0:.4f}: unit_load\n{static_load_summary['load_unit']}).")
    logging.info(
        f"{0.0:.4f}: unit_load_type\n{static_load_summary['load_unit_type']}).")
    logging.info(
        f"{0.0:.4f}: intensity_unit\n{static_load_summary['intensity_unit']}).")

    # Run the simulation replication
    env.run(until=config.run_time)

    # Create the stops and visits dataframes
    stops_df, visits_df = obstat.create_stops_visits_dfs(obsystem)

    # Compute occupancy stats
    occ_stats_df, occ_log_df = obstat.compute_occ_stats(obsystem, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    # Compute and gather summary stats for this scenario replication
    scenario_rep_summary_dict = obstat.create_rep_summary(
        config.scenario, rep_num, obsystem, stops_df, visits_df, occ_stats_df)

    # TODO - design output processing scheme
    # Patient generator stats
    arrival_summary = obstat.ReportArrivalGeneratorSummary(config.scenario, rep_num,
                                                           patient_generators_poisson,
                                                           patient_generators_scheduled)
    print(arrival_summary)

    # Patient type summary
    pat_type_summary = obstat.ReportPatientTypeSummary(config.scenario,
                                                       rep_num,
                                                       visits_df,
                                                       config)
    print(pat_type_summary)

    # Delivery summary
    delivery_summary = obstat.ReportDeliverySummary(config.scenario,
                                                    rep_num,
                                                    visits_df,
                                                    config)
    print(delivery_summary)

    # Occupancy summary
    occ_summary = obstat.ReportOccupancySummary(config.scenario, rep_num, occ_stats_df)
    print(occ_summary)

    # Unit stats
    flow_stat_summary = obstat.ReportUnitFlowStats(config.scenario, rep_num, obsystem)
    print(flow_stat_summary)

    # System exit stats
    exit_summary = obstat.ReportExitSummary(config.scenario, rep_num, obsystem)
    print(exit_summary)

    # Write stats and log files
    if config.paths['stop_logs'] is not None:
        obio.write_log('stop_log', config.paths['stop_logs'],
                       stops_df, rep_num, config)

    if config.paths['occ_stats'] is not None:
        obio.write_stats('occ_stats', config.paths['occ_stats'], occ_stats_df, config.scenario, rep_num)

    if config.paths['occ_logs'] is not None:
        obio.write_log('occ_log', config.paths['occ_logs'], occ_log_df, rep_num, config)

    if config.paths['visit_logs'] is not None:
        obio.write_log('visit_log', config.paths['visit_logs'],
                       visits_df, rep_num, config)

    return scenario_rep_summary_dict


def logger_setup(loglevel: int) -> Logger:
    """Create and setup logger"""
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # Create the Handler for logging data to console.
    logger_handler = logging.StreamHandler()
    logger_handler.setLevel(loglevel)

    # Create a Formatter for formatting the log messages
    logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the Formatter to the Handler
    logger_handler.setFormatter(logger_formatter)

    # Add the Handler to the Logger
    logger.addHandler(logger_handler)

    return logger


def main(argv=None):
    """

    Parameters
    ----------
    argv

    Returns
    -------

    """

    # Parse command line arguments
    args = process_command_line(argv)

    # Create root logger
    logger = logger_setup(args.loglevel)

    # Load scenario configuration file and create OBConfig object
    config_dict = obio.load_config(args.config)
    config = obconfig.Config(config_dict)

    # Initialize scenario specific variables
    scenario = config.scenario
    summary_stat_path = \
        config_dict['outputs']['summary_stats']['path'] / Path(f'summary_stats_scenario_{scenario}.csv')

    results = []
    for i in range(1, config.num_replications + 1):
        print(f'\nRunning scenario {scenario}, replication {i}')
        scenario_rep_summary_dict = simulate(config, i)
        results.append(scenario_rep_summary_dict)

    # Convert results dict to DataFrame
    scenario_rep_summary_df = pd.DataFrame(results)
    obio.write_summary_stats(summary_stat_path, scenario_rep_summary_df)

    return 0


if __name__ == '__main__':
    sys.exit(main())
