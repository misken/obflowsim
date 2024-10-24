from __future__ import annotations

import logging
from typing import (
    Tuple, )

import pandas as pd
import simpy
from simpy import Environment


from obflowsim.clock_tools import SimCalendar
from obflowsim.config import Config
from obflowsim.obconstants import UnitName, MARKED_PATIENT, ATT_GET_BED, ATT_RELEASE_BED
from obflowsim.patient import Patient


class PatientFlowSystem:
    """
    Acts as a container for inputs such as the config and the `SimCalendar` as well as for
    system objects created from these inputs and (maybe) timestamp dicts.

    Instead of passing around the above individually, just pass this system object around.

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


class EntryNode:
    """
    All patients start at this node. It is the first stop in all routes.

    Patients with scheduled arrival times get held here until their arrival time occurs.
    Patients may wait here for a bed to become available in their first patient care unit.
    """

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

        logging.debug(
            f"{self.env.now:.4f}: {patient.patient_id} enters {self.name} node.")

        # Update unit attributes
        self.num_entries += 1
        self.last_entry_ts = self.env.now
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
        patient.request_exit_ts[csn] = self.env.now
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
     All patients end at this node. It is the last stop in all routes.


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
                                  'waited_to_enter': patient.entry_ts[stop] > patient.request_entry_ts[stop],
                                  'waited_to_exit': patient.exit_ts[stop] > patient.request_exit_ts[stop]}
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

    def __init__(self, env: simpy.Environment, name: str, capacity: int = simpy.core.Infinity):

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

        # We are trying to leave the unit patient currently in to visit another unit (this unit)
        logging.debug(
            f"{self.env.now:.4f}: {patient.patient_id} trying to get {self.name} for stop_num {patient.current_stop_num + 1}")

        if patient.patient_id == MARKED_PATIENT:
            pass

        request_entry_ts = self.env.now   # Note the current time we tried to enter this unit
        exiting_unit_name = patient.get_current_unit_name()  # Unit we are in right now while trying to enter this unit
        exiting_unit = patient.get_current_unit()
        entering_unit_name = self.name
        got_bed = False

        # This is the arc terminating at ths unit
        try:
            assert exiting_unit_name != entering_unit_name
        except AssertionError:
            print(f'Patient {patient.patient_id} has invalid incoming route edge.')

        # Get route edge terminating at this unit. Account for any edges skipped due to extended blocking.
        if patient.skipped_edge[patient.current_stop_num] is None:
            skipped_unit_name = None
            blocked_unit_name = None
            incoming_route_edge = (exiting_unit_name, entering_unit_name,
                                   patient.route_graph.edges[exiting_unit_name, entering_unit_name])
        else:
            skipped_unit_name = patient.skipped_edge[patient.current_stop_num][1]
            blocked_unit_name = exiting_unit_name
            # indexes: 0=source node, 1=dest node, 2=edge data
            incoming_route_edge = (skipped_unit_name, entering_unit_name,
                                   patient.route_graph.edges[skipped_unit_name, entering_unit_name])

        # Sample from LOS distribution for this arc and patient type
        planned_los = incoming_route_edge[2]['planned_los']()

        # Request bed if indicated
        if incoming_route_edge[2][ATT_GET_BED]:
            # Request a bed - triggers (or at least creates) SimPy event object
            bed_request = self.unit.request()
            # Store bed request and timestamp in patient's request dictionary
            patient.bed_requests[self.name] = bed_request

            # Yield until we get a bed or our planned los has elapsed due to being blocked
            get_bed = yield bed_request | self.env.timeout(planned_los)

            # Check if we got a bed before our los has elapsed
            if bed_request in get_bed:
                if patient.patient_id == MARKED_PATIENT:
                    pass

                got_bed = True  # Good to continue processing at this patient care unit
            else:  # Our LOS has elapsed while we were blocked trying to enter this unit.
                # Need to get rid of last bed request as we'll never enter this unit.
                # Also need to cancel the reqeust
                if patient.patient_id == MARKED_PATIENT:
                    pass

                patient.bed_requests.pop(self.name)
                bed_request.cancel()
                # Determine next stop in route
                current_edge_num = incoming_route_edge[2]['edge_num']
                next_route_edge = patient.pfs.router.get_next_step(patient, after=current_edge_num,
                                                                   unit=entering_unit_name)
                entering_unit_name = next_route_edge[1]
                patient.skipped_edge[patient.current_stop_num] = incoming_route_edge

                #current_unit = pfs.patient_care_units[exiting_unit_name]

                if entering_unit_name != UnitName.EXIT:
                    # Try to get bed in next unit
                    self.env.process(pfs.patient_care_units[entering_unit_name].put(patient, pfs))
                else:  # Patient is ready to exit system
                    # Release the bed
                    if exiting_unit_name in patient.bed_requests and incoming_route_edge[2][ATT_RELEASE_BED]:
                        # Release the previous bed
                        exiting_unit.unit.release(patient.bed_requests[exiting_unit_name])
                        unit_released = patient.bed_requests.pop(exiting_unit_name)

                    try:
                        assert not patient.bed_requests
                    except AssertionError:
                        print(f'Patient {patient.patient_id} trying to exit with active bed requests.')

                    # Accumulate total time this unit occupied and other unit attributes
                    exiting_unit.tot_occ_time += \
                        self.env.now - patient.entry_ts[patient.current_stop_num]
                    exiting_unit.num_exits += 1
                    exiting_unit.last_exit_ts = self.env.now
                    exiting_unit.dec_occ()

                    # Update patient attributes
                    patient.request_exit_ts[patient.current_stop_num] = self.env.now
                    patient.exit_ts[patient.current_stop_num] = self.env.now

                    # Send patient to Exit node
                    pfs.exit.put(patient, pfs)

        if got_bed or not incoming_route_edge[2][ATT_GET_BED]:  # Seized a bed if needed.

            if patient.patient_id == MARKED_PATIENT:
                pass

            # Update patient flow attributes for this stop
            patient.current_stop_num += 1
            csn = patient.current_stop_num

            patient.append_empty_unit_stop()  # Appends None to all patient flow related lists
            patient.request_entry_ts[csn] = request_entry_ts
            patient.unit_stops[csn] = self.name
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

            # Update timestamps for stop at previous unit.
            patient.exit_ts[csn - 1] = self.env.now
            patient.wait_to_exit[csn - 1] = \
                self.env.now - patient.request_exit_ts[csn - 1]

            # Accumulate total time previous unit occupied and other unit attributes
            previous_unit_name = patient.get_previous_unit_name()
            previous_unit = patient.get_previous_unit()
            previous_unit.tot_occ_time += \
                self.env.now - patient.entry_ts[csn - 1]
            previous_unit.num_exits += 1
            previous_unit.last_exit_ts = self.env.now

            # Check if we have a bed from a previous stay and release it if we do and want to release it.
            if previous_unit_name in patient.bed_requests and incoming_route_edge[2][ATT_RELEASE_BED]:
                # Release the previous bed
                previous_unit.unit.release(patient.bed_requests[previous_unit_name])
                # What happens to the reference in patient.bed_requests[]?
                unit_released = patient.bed_requests.pop(previous_unit_name)
                previous_unit.dec_occ()
            elif blocked_unit_name is not None and blocked_unit_name != UnitName.ENTRY.value:
                pfs.patient_care_units[blocked_unit_name].unit.release(patient.bed_requests[blocked_unit_name])
                pfs.patient_care_units[blocked_unit_name].dec_occ()
                unit_released = patient.bed_requests.pop(blocked_unit_name)

            logging.debug(f"{self.env.now:.4f}: {patient.patient_id} entering {self.name} at {self.env.now}")

            # Do any blocking related los adjustments.
            blocking_adj_los = self.los_blocking_adjustment(patient, planned_los, incoming_route_edge)

            # Do discharge timing related los adjustments
            adjusted_los = self.los_discharge_adjustment(pfs.config, pfs, patient,
                                                         blocking_adj_los, incoming_route_edge)

            # Update los related patient attributes
            patient.planned_los[patient.current_stop_num] = planned_los
            patient.adjusted_los[patient.current_stop_num] = adjusted_los

            # Wait for LOS to elapse
            yield self.env.timeout(adjusted_los)
            if patient.patient_id == MARKED_PATIENT:
                pass

            # Determine next stop in route
            outgoing_route_edge = patient.pfs.router.get_next_step(patient)
            entering_unit_name = outgoing_route_edge[1]

            if entering_unit_name != UnitName.EXIT:
                if patient.patient_id == MARKED_PATIENT:
                    pass
                # Try to get bed in next unit
                patient.request_exit_ts[patient.current_stop_num] = self.env.now
                self.env.process(pfs.patient_care_units[entering_unit_name].put(patient, pfs))
            else:
                # Patient is ready to exit system
                if patient.patient_id == MARKED_PATIENT:
                    pass
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
                patient.wait_to_exit[patient.current_stop_num] = \
                    patient.exit_ts[patient.current_stop_num] = patient.request_exit_ts[patient.current_stop_num]

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

    def los_blocking_adjustment(self, patient: Patient, planned_los: float, incoming_route_edge: Tuple):

        previous_unit_name = incoming_route_edge[0]
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

    def los_discharge_adjustment(self, config: Config,
                                 pfs: PatientFlowSystem,
                                 patient: Patient,
                                 planned_los: float,
                                 incoming_route_edge: Tuple):

        G = patient.route_graph
        previous_unit_name = incoming_route_edge[0]
        try:
            discharge_pdf = G[previous_unit_name][self.name]['discharge_adjustment']
        except KeyError:
            print(f'key error for {patient.patient_id}')

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

        return discharge_adj_los
