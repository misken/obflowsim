from __future__ import annotations

import logging
from typing import (
    Tuple, )

import pandas as pd
import simpy
from simpy import Environment
from simpy.events import AnyOf
import networkx as nx


from obflowsim.clock_tools import SimCalendar
from obflowsim.config import Config
from obflowsim.obconstants import UnitName, MARKED_PATIENT
from obflowsim.patient import Patient
from obflowsim.obconstants import UnitName, DEFAULT_GET_BED, DEFAULT_RELEASE_BED, ATT_RELEASE_BED, ATT_GET_BED
from obflowsim.obconstants import SRC, DEST, DATA
from obflowsim.los import los_mean
from obflowsim.routing import find_next_unit_stop


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

        self.network = self.create_pfs_graph(config)

        # Create list to hold timestamps dictionaries (one per patient stop)
        self.stops_timestamps_list = []

    def create_pfs_graph(self, config):

        pfs_graph = nx.DiGraph()
        for edge in self.config.network['edges']:

            # Add edges connecting patient care units
            pfs_graph.add_edge(edge['from'], edge['to'])

            # Set edge attributes
            if 'id' in edge:
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {'id': edge['id']}})
            else:
                # Default edge name
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {'id': f"{edge['from']}_{edge['to']}"}})

            if 'los' in edge:
                edge['los_mean'] = los_mean(edge['los'], config.los_params)

                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {'los': edge['los']}})

                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {'los_mean': edge['los_mean']}})

            # Add get and keep bed attributes
            if ATT_GET_BED in edge:
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {ATT_GET_BED: edge[ATT_GET_BED]}})
            else:
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {ATT_GET_BED: DEFAULT_GET_BED}})

            if ATT_RELEASE_BED in edge:
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {ATT_RELEASE_BED: edge[ATT_RELEASE_BED]}})
            else:
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {ATT_RELEASE_BED: DEFAULT_RELEASE_BED}})

            # Add blocking adjustment attribute
            if 'blocking_adjustment' in edge:
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {'blocking_adjustment': edge['blocking_adjustment']}})
            else:
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {'blocking_adjustment': None}})

            # Add discharge timing adjustment attribute
            if 'discharge_adjustment' in edge:
                discharge_pmf_file = edge['discharge_adjustment']
                discharge_pmf = pd.read_csv(discharge_pmf_file, sep=r'\s+', header=None, names=['x', 'p'])
                discharge_pmf.set_index('x', inplace=True, drop=True)
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {'discharge_adjustment': discharge_pmf}})
            else:
                nx.set_edge_attributes(pfs_graph, {
                    (edge['from'], edge['to']): {'discharge_adjustment': None}})

        return pfs_graph


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

    def put(self, patient: Patient, pfs: PatientFlowSystem):
        """
        A process method called when entry to the PatientFlowSystem is requested.

        Parameters
        ----------
        patient : Patient object
            the patient requesting the bed
        pfs : PatientFlowSystem object

        """

        logging.debug(
            f"{self.env.now:.4f}: {patient.patient_id} enters {self.name} node.")

        # Update unit attributes
        self.num_entries += 1
        self.last_entry_ts = self.env.now
        self.inc_occ()

        # Update patient attributes
        csn = 0
        patient.current_stop_num = csn
        patient.current_unit_name = self.name
        patient.append_empty_unit_stop()
        patient.unit_stops[csn] = UnitName.ENTRY.value
        patient.request_entry_ts[csn] = self.env.now
        patient.entry_ts[csn] = self.env.now
        patient.planned_los[csn] = patient.entry_delay
        patient.adjusted_los[csn] = patient.entry_delay

        # Wait for any entry_delay needed
        yield self.env.timeout(patient.entry_delay)

        patient.request_exit_ts[csn] = self.env.now

        logging.debug(
            f"{self.env.now:.4f}: {patient.patient_id} ready to leave {self.name} node.")

        # Get bed in next destination
        # self.env.process(find_next_unit_stop(self.env, patient, pfs))

        # ---------------------------------------------------
        got_new_bed = False
        skip = False
        while not got_new_bed:

            # TODO: This next line needs to take into account that we may have just skipped a stop
            patient.next_step = pfs.router.get_next_step(patient, skip=skip)

            # We know where we are going, get ready to try to grab a new bed
            patient.next_unit_name = patient.next_step[0][DEST]
            patient.request_exit_ts[csn] = self.env.now

            request_entry_ts = self.env.now  # Note the current time we tried to enter next unit
            exiting_unit = patient.get_current_unit()
            exiting_unit_name = exiting_unit.name  # Unit we are in right now while trying to enter this unit

            outgoing_route_edge = patient.next_step[0]

            # We are trying to leave the unit patient currently in to visit another unit - patient.next_unit_name
            logging.debug(
                f"{self.env.now:.4f}: {patient.patient_id} trying to get {patient.next_unit_name} for stop_num {csn + 1}")

            # Sample from LOS distribution for this arc and patient type
            sampled_los = outgoing_route_edge[DATA]['planned_los']()

            # Need request objects for each destination in next_step edges
            dest_unit_names = [v for (u, v, d) in patient.next_step]
            dest_units = [pfs.patient_care_units[name] for name in dest_unit_names]
            # Request bed(s) - Creates SimPy event objects
            bed_request_events = {pfs.patient_care_units[v].unit.request(): {'dest_unit_name': v,
                                                                             'dest_unit': pfs.patient_care_units[v],
                                                                             'edge': (u, v, d)} for (u, v, d) in
                                  patient.next_step}

            # Yield until we get a bed or our planned los has elapsed due to being blocked
            bed_req_los_events = [key for key in bed_request_events.keys()]
            # bed_req_los_events[env.timeout(planned_los, value='los_elapsed')] = 'los_elapsed'
            los_timeout = self.env.timeout(sampled_los)
            bed_req_los_events.append(los_timeout)

            # Try to get a bed
            get_bed = yield AnyOf(self.env, bed_req_los_events)

            # Check if we got a bed before our los has elapsed
            if los_timeout not in get_bed:
                successful_req = None
                for req in bed_request_events:
                    if req in get_bed:
                        successful_req = req
                        next_edge = bed_request_events[req]['edge']

                entering_unit_name = bed_request_events[successful_req]['dest_unit_name']
                patient.bed_requests[entering_unit_name] = successful_req
                patient.next_unit_name = entering_unit_name
                patient.sampled_los = sampled_los
                got_new_bed = True
                skip = False
                # Good to send patient to next patient care unit
            else:
                # Our LOS elapsed before we got a bed in next unit

                # 1) Record the fact that we are skipping an entire stop
                # 2) Cancel and remove bed request events
                # 3) Figure out where we are going next

                # Record the skip event
                skipped_edge_record = {'current_stop_num': csn,
                                       'id': outgoing_route_edge['id'],
                                       'skipped_edge': outgoing_route_edge,
                                       'planned_los': sampled_los}

                patient.skipped_edges.append(skipped_edge_record)
                patient.skipped_edges_cache.append(skipped_edge_record)
                skip = True

                # Cancel the bed requests as they will never be fulfilled
                for bed_request in bed_request_events:
                    bed_request.cancel()


        # ----------------------------------------------------



        # Update timestamps for stop at ENTRY
        patient.exit_ts[csn] = self.env.now
        patient.wait_to_exit[csn] = \
            self.env.now - patient.request_exit_ts[csn]

        # Accumulate total time previous unit occupied and other unit attributes
        self.tot_occ_time += \
            self.env.now - patient.entry_ts[csn]
        self.num_exits += 1
        self.last_exit_ts = self.env.now
        self.dec_occ()


        # Put patient in next unit
        self.env.process(pfs.patient_care_units[patient.next_unit_name].put(patient, pfs,
                                                                            next_edge,
                                                                            request_entry_ts))



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

    def put(self, patient: Patient,
            pfs: PatientFlowSystem,
            incoming_route_edge: Tuple,
            request_entry_ts: float):
        """
        A process method called when a patient wants to enter this patient care unit.

        Parameters
        ----------
        
        

        patient : OBPatient object
            the patient requesting the bed
        pfs : OBSystem object
        incoming_route_edge
        exiting_unit_name
        request_entry_ts

        """

        # Things I might need to pass in
        # request_entry_ts
        # exiting_unit_name

        # Update patient flow attributes for this stop
        patient.current_stop_num += 1
        csn = patient.current_stop_num
        patient.previous_unit_name = patient.current_unit_name
        patient.current_unit_name = self.name
        patient.next_unit_name = None  # Temporarily

        patient.append_empty_unit_stop()
        patient.request_entry_ts[csn] = request_entry_ts  # Was set earlier when request created
        patient.unit_stops[csn] = self.name
        patient.entry_ts[csn] = self.env.now
        patient.wait_to_enter[csn] = self.env.now - patient.request_exit_ts[csn - 1]
        if patient.wait_to_enter[csn] > 0:
            patient.blocked[csn] = 1
        else:
            patient.blocked[csn] = 0

        # Update unit attributes
        self.num_entries += 1
        self.last_entry_ts = self.env.now

        # Increment occupancy in this unit
        self.inc_occ()

        # The following shouldn't be necessary as we should now be able to release bed right after current stay
        # and once new bed is found via find_next_unit_stop()

        # Check if we have a bed from a previous stay and release it if we do and want to release it.
        # if exiting_unit_name in patient.bed_requests and incoming_route_edge[DATA][ATT_RELEASE_BED]:
        #     # Release the previous bed
        #     exiting_unit.unit.release(patient.bed_requests[exiting_unit_name])
        #     # What happens to the reference in patient.bed_requests[]?
        #     unit_released = patient.bed_requests.pop(exiting_unit_name)
        #     exiting_unit.dec_occ()
        # elif blocked_unit_name is not None and blocked_unit_name != UnitName.ENTRY.value:
        #     pfs.patient_care_units[blocked_unit_name].unit.release(patient.bed_requests[blocked_unit_name])
        #     pfs.patient_care_units[blocked_unit_name].dec_occ()
        #     unit_released = patient.bed_requests.pop(blocked_unit_name)

        logging.debug(f"{self.env.now:.4f}: {patient.patient_id} entering {self.name} at {self.env.now}")

        # Do any blocking related los adjustments.
        blocking_adj_los = self.los_blocking_adjustment(patient, patient.sampled_los, incoming_route_edge)

        # Do discharge timing related los adjustments
        adjusted_los = self.los_discharge_adjustment(pfs.config, pfs, patient,
                                                     blocking_adj_los, incoming_route_edge)

        # Update los related patient attributes
        patient.planned_los[csn] = patient.sampled_los
        patient.adjusted_los[csn] = adjusted_los

        # Wait for LOS to elapse
        yield self.env.timeout(adjusted_los)

        # Determine next stop in route
        patient.next_step = patient.pfs.router.get_next_step(patient)

        if patient.next_step[DEST] != UnitName.EXIT:
            # ---------------------------------------------------
            got_new_bed = False
            skip = False
            while not got_new_bed:

                # TODO: This next line needs to take into account that we may have just skipped a stop
                patient.next_step = pfs.router.get_next_step(patient, skip=skip)

                # We know where we are going, get ready to try to grab a new bed
                patient.next_unit_name = patient.next_step[0][DEST]
                patient.request_exit_ts[csn] = self.env.now

                request_entry_ts = self.env.now  # Note the current time we tried to enter next unit
                exiting_unit = patient.get_current_unit()
                exiting_unit_name = exiting_unit.name  # Unit we are in right now while trying to enter this unit

                outgoing_route_edge = patient.next_step[0]

                # We are trying to leave the unit patient currently in to visit another unit - patient.next_unit_name
                logging.debug(
                    f"{self.env.now:.4f}: {patient.patient_id} trying to get {patient.next_unit_name} for stop_num {csn + 1}")

                # Sample from LOS distribution for this arc and patient type
                sampled_los = outgoing_route_edge[DATA]['planned_los']()

                # Need request objects for each destination in next_step edges
                dest_unit_names = [v for (u, v, d) in patient.next_step]
                dest_units = [pfs.patient_care_units[name] for name in dest_unit_names]
                # Request bed(s) - Creates SimPy event objects
                bed_request_events = {pfs.patient_care_units[v].unit.request(): {'dest_unit_name': v,
                                                                                 'dest_unit': pfs.patient_care_units[v],
                                                                                 'edge': (u, v, d)} for (u, v, d) in
                                      patient.next_step}

                # Yield until we get a bed or our planned los has elapsed due to being blocked
                bed_req_los_events = [key for key in bed_request_events.keys()]
                # bed_req_los_events[env.timeout(planned_los, value='los_elapsed')] = 'los_elapsed'
                los_timeout = self.env.timeout(sampled_los)
                bed_req_los_events.append(los_timeout)

                # Try to get a bed
                get_bed = yield AnyOf(self.env, bed_req_los_events)

                # Check if we got a bed before our los has elapsed
                if los_timeout not in get_bed:
                    successful_req = None
                    for req in bed_request_events:
                        if req in get_bed:
                            successful_req = req
                            next_edge = bed_request_events[req]['edge']

                    entering_unit_name = bed_request_events[successful_req]['dest_unit_name']
                    patient.bed_requests[entering_unit_name] = successful_req
                    patient.next_unit_name = entering_unit_name
                    patient.sampled_los = sampled_los
                    got_new_bed = True
                    skip = False
                    # Good to send patient to next patient care unit
                else:
                    # Our LOS elapsed before we got a bed in next unit

                    # 1) Record the fact that we are skipping an entire stop
                    # 2) Cancel and remove bed request events
                    # 3) Figure out where we are going next

                    # Record the skip event
                    skipped_edge_record = {'current_stop_num': csn,
                                           'id': outgoing_route_edge['id'],
                                           'skipped_edge': outgoing_route_edge,
                                           'planned_los': sampled_los}

                    patient.skipped_edges.append(skipped_edge_record)
                    patient.skipped_edges_cache.append(skipped_edge_record)
                    skip = True

                    # Cancel the bed requests as they will never be fulfilled
                    for bed_request in bed_request_events:
                        bed_request.cancel()

            # ----------------------------------------------------
        else:
            # Patient is ready to exit system
            # Release the bed
            if self.name in patient.bed_requests and incoming_route_edge[DEST][ATT_RELEASE_BED]:
                # Release the previous bed
                self.unit.release(patient.bed_requests[self.name])
                unit_released = patient.bed_requests.pop(self.name)

            try:
                assert not patient.bed_requests
            except AssertionError:
                print(f'Patient {patient.patient_id} trying to exit with bed requests.')

            # Accumulate total time this unit occupied and other unit attributes
            self.tot_occ_time += \
                self.env.now - patient.entry_ts[csn]
            self.num_exits += 1
            self.last_exit_ts = self.env.now

            # Decrement occupancy in this unit since bed now released
            self.dec_occ()

            patient.request_exit_ts[csn] = self.env.now
            patient.exit_ts[csn] = self.env.now
            patient.wait_to_exit[csn] = \
                patient.exit_ts[csn] = patient.request_exit_ts[csn]

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

    def prepare_to_exit_system(self, exiting_unit, patient):
        """
        Update patient and unit stats and attributes before patient exits system.

        Parameters
        ----------
        exiting_unit - the patient care unit about to be exited
        patient - the patient exiting the system

        Returns
        -------

        """

        csn = patient.current_stop_num
        # Accumulate total time this unit occupied and other unit attributes
        exiting_unit.tot_occ_time += \
            self.env.now - patient.entry_ts[csn]
        exiting_unit.num_exits += 1
        exiting_unit.last_exit_ts = self.env.now
        exiting_unit.dec_occ()

        # Update patient attributes
        patient.request_exit_ts[csn] = self.env.now
        patient.exit_ts[csn] = self.env.now
        patient.wait_to_exit[csn - 1] = 0.0

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

        previous_unit_name = incoming_route_edge[SRC]
        if previous_unit_name != UnitName.ENTRY and patient.current_stop_num > 1:
            G = patient.planned_route

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

        G = patient.planned_route
        previous_unit_name = incoming_route_edge[SRC]
        try:
            discharge_pdf = G[previous_unit_name][self.name]['discharge_adjustment']
        except KeyError:
            discharge_pdf = None
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
        patient.previous_unit_name = patient.current_unit_name
        patient.current_unit_name = self.name
        patient.next_unit_name = None
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
        patient.previous_step = patient.next_step
        patient.next_step = None

        # Create dictionaries of timestamps for patient_stop log
        for stop_num in range(len(patient.unit_stops)):
            if patient.unit_stops[stop_num] is not None:
                try:
                    timestamps = {'patient_id': patient.patient_id,
                                  'patient_type': patient.patient_type,
                                  'arrival_type': patient.arrival_type,
                                  'unit': patient.unit_stops[stop_num],
                                  'request_entry_ts': patient.request_entry_ts[stop_num],
                                  'entry_ts': patient.entry_ts[stop_num],
                                  'request_exit_ts': patient.request_exit_ts[stop_num],
                                  'exit_ts': patient.exit_ts[stop_num],
                                  'planned_los': patient.planned_los[stop_num],
                                  'adjusted_los': patient.adjusted_los[stop_num],
                                  'entry_tryentry': patient.entry_ts[stop_num] - patient.request_entry_ts[stop_num],
                                  'tryexit_entry': patient.request_exit_ts[stop_num] - patient.entry_ts[stop_num],
                                  'exit_tryexit': patient.exit_ts[stop_num] - patient.request_exit_ts[stop_num],
                                  'exit_enter': patient.exit_ts[stop_num] - patient.entry_ts[stop_num],
                                  'exit_tryenter': patient.exit_ts[stop_num] - patient.request_entry_ts[stop_num],
                                  'wait_to_enter': patient.wait_to_enter[stop_num],
                                  'wait_to_exit': patient.wait_to_exit[stop_num],
                                  'waited_to_enter': patient.entry_ts[stop_num] > patient.request_entry_ts[stop_num],
                                  'waited_to_exit': patient.exit_ts[stop_num] > patient.request_exit_ts[stop_num]}
                except TypeError:
                    raise TypeError(f'Unable to create timestamps dict for stop {stop_num} for patient {patient}.')

                obsystem.stops_timestamps_list.append(timestamps)

        self.num_exits += 1
        logging.debug(
            f"{self.env.now:.4f}: {patient.patient_id} exited system at {self.env.now:.2f}.")