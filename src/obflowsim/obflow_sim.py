import sys
import logging
from logging import Logger
from enum import Enum
from copy import deepcopy
from pathlib import Path
import argparse
from pprint import pprint
from abc import ABC, abstractmethod

from typing import (
    TYPE_CHECKING,
    Dict,
    Union,
)

from numpy.typing import (
    NDArray,
    DTypeLike,
)

#if TYPE_CHECKING and TYPE_CHECKING != 'SPHINX':  # Avoid circular import
from simpy.core import Environment

import pandas as pd
import numpy as np
import simpy
from numpy.random import default_rng
from numpy.random import Generator
import networkx as nx
from networkx import DiGraph
import json

import obflowsim.obflow_io as obio
import obflowsim.obflow_stat as obstat
import obflowsim.obflow_qng as obq

ENTRY = 'ENTRY'
EXIT = 'EXIT'

MAX_ARRIVALS = 1e+6

# TODO - make sure all docstrings are complete
# TODO - use type annotations (?)

"""
Basic OB patient flow model

Details:

- Generates arrivals via Poisson process patient generators as well as weekly schedules
- Defines an ``OBUnit`` class that contains a ``simpy.Resource`` object as a member.
  Not subclassing Resource, just using a ``Resource`` as a member.
- Routing is done via setting ``out`` member of an ``OBUnit`` instance to
 another ``OBUnit`` instance to which the ``OBPatient`` instance should be
 routed. The routing logic is in ``OBStaticRouter`` with routes defined
 in the config file.

"""


class OBConfig:
    """
    OBConfig is the class used to store simulation scenario instance configuration information.

    Attributes
    ----------
    scenario : str
        scenario identifier used in output filenames
    run_time : float
        length of simulation run in base time units
    warmup_time : float
        time after which steady state simulation output statistics begin to be computed
    num_replications : int
        the number of independent replications of the simulation model
    rg : dict
        key is 'arrivals' or 'los' and value is a `numpy.random.default_rng` object. The seed
        value is based on `config_dict['random_number_streams']`, which has same key values
    rand_arrival_rates : dict of floats
        keys are valid arrival stream identifiers (see documentation for the config file). Units are the
        mean number of arrivals per base time unit. These means are used with the `OBPatientGeneratorPoisson`
        class for generating unscheduled (random) arrivals.
    rand_arrival_toggles : dict of ints
        same keys as `rand_arrival_rates` with a value of 0 shutting off the arrival stream and 1 enabling
        the arrival stream.



    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(self, config_dict: Dict):
        self.scenario = config_dict['scenario']

        self.run_time = config_dict['run_settings']['run_time']
        self.warmup_time = config_dict['run_settings']['warmup_time']
        self.num_replications = config_dict['run_settings']['num_replications']

        # Create random number generators
        self.rg = {}
        for stream, seed in config_dict['random_number_streams'].items():
            self.rg[stream] = default_rng(seed)

        # Arrival rates
        self.rand_arrival_rates = config_dict['rand_arrival_rates']
        self.rand_arrival_toggles = config_dict['rand_arrival_toggles']

        # Schedules
        self.schedules = {}
        if 'sched_csect' in config_dict['schedule_files']:
            sched_file = config_dict['schedule_files']['sched_csect']
            self.schedules['sched_csect'] = np.loadtxt(sched_file, dtype=int)

        if 'sched_induced_labor' in config_dict['schedule_files']:
            sched_file = config_dict['schedule_files']['sched_induced_labor']
            self.schedules['sched_induced_labor'] = np.loadtxt(sched_file, dtype=int)

        self.sched_arrival_toggles = config_dict['sched_arrival_toggles']

        # Branching probabilities
        self.branching_probabilities = config_dict['branching_probabilities']

        # Length of stay
        self.los_params = config_dict['los_params']
        self.los_distributions, self.los_means = obio.create_los_partials(config_dict['los_distributions'],
                                                          self.los_params, self.rg['los'])

        self.locations = config_dict['locations']
        self.routes = config_dict['routes']
        self.outputs = config_dict['outputs']

        # Calendar related
        self.use_calendar_time = config_dict['run_settings']['use_calendar_time']
        if 'start_date' in config_dict['run_settings']:
            self.start_date = pd.Timestamp(config_dict['run_settings']['start_date'])
        else:
            # Default start_date is first Monday after Unix epoch
            self.start_date = pd.Timestamp('1970-01-05')

        self.base_time_unit = config_dict['run_settings']['base_time_unit']

        # Output paths
        outputs = self.outputs.keys()
        self.paths = {output: None for output in outputs}
        for output in outputs:
            if self.outputs[output]['write']:
                Path(self.outputs[output]['path']).mkdir(parents=True, exist_ok=True)
                self.paths[output] = Path(self.outputs[output]['path'])


class SimCalendar:
    """

    """

    def __init__(self, env: 'Environment', config: OBConfig):
        self.env = env
        self.start_date = config.start_date
        self.base_time_unit = config.base_time_unit
        self.use_calendar_time = config.use_calendar_time

    def now(self):
        return self.to_sim_calendar_time(self.env.now)

    def to_sim_calendar_time(self, sim_time):
        elapsed_timedelta = pd.to_timedelta(sim_time, unit=self.base_time_unit)
        return self.start_date + elapsed_timedelta

    def to_sim_time(self, sim_calendar_time):
        elapsed_timedelta = sim_calendar_time - self.start_date
        return elapsed_timedelta / pd.to_timedelta(1, unit=self.base_time_unit)


class PatientType(Enum):
    """
    # Patient Type and Patient Flow Definitions

    # Type 1: random arrival spont labor, regular delivery, route = 1-2-4
    # Type 2: random arrival spont labor, C-section delivery, route = 1-3-2-4
    # Type 3: random arrival augmented labor, regular delivery, route = 1-2-4
    # Type 4: random arrival augmented labor, C-section delivery, route = 1-3-2-4
    # Type 5: sched arrival induced labor, regular delivery, route = 1-2-4
    # Type 6: sched arrival induced labor, C-section delivery, route = 1-3-2-4
    # Type 7: sched arrival, C-section delivery, route = 1-3-2-4

    # Type 8: urgent induced arrival, regular delivery, route = 1-2-4
    # Type 9: urgent induced arrival, C-section delivery, route = 1-3-2-4

    # Type 10: random arrival, non-delivered LD, route = 1
    # Type 11: random arrival, non-delivered PP route = 4
    """
    RAND_SPONT_REG = 'RAND_SPONT_REG'
    RAND_SPONT_CSECT = 'RAND_SPONT_CSECT'
    RAND_AUG_REG = 'RAND_AUG_REG'
    RAND_AUG_CSECT = 'RAND_AUG_CSECT'
    SCHED_IND_REG = 'SCHED_IND_REG'
    SCHED_IND_CSECT = 'SCHED_IND_CSECT'
    SCHED_CSECT = 'SCHED_CSECT'
    URGENT_IND_REG = 'URGENT_IND_REG'
    URGENT_IND_CSECT = 'URGENT_IND_CSECT'
    RAND_NONDELIV_LDR = 'RAND_NONDELIV_LDR'
    RAND_NONDELIV_PP = 'RAND_NONDELIV_PP'


class ArrivalType(Enum):
    """

    """
    SPONT_LABOR = 'spont_labor'
    URGENT_INDUCED_LABOR = 'urgent_induced_labor'
    NON_DELIVERY_LDR = 'non_delivery_ldr'
    NON_DELIVERY_PP = 'non_delivery_pp'
    SCHED_CSECT = 'sched_csect'
    SCHED_INDUCED_LABOR = 'sched_induced_labor'


class PatientTypeArrivalType:

    pat_type_to_arrival_type = {
        PatientType.RAND_SPONT_REG.value: ArrivalType.SPONT_LABOR.value,
        PatientType.RAND_SPONT_CSECT.value: ArrivalType.SPONT_LABOR.value,
        PatientType.RAND_AUG_REG.value: ArrivalType.SPONT_LABOR.value,
        PatientType.RAND_SPONT_CSECT.value: ArrivalType.SPONT_LABOR.value,
        PatientType.SCHED_IND_REG.value: ArrivalType.SCHED_INDUCED_LABOR.value,
        PatientType.SCHED_IND_CSECT.value: ArrivalType.SCHED_INDUCED_LABOR.value,
        PatientType.SCHED_CSECT.value: ArrivalType.SCHED_CSECT.value,
        PatientType.URGENT_IND_REG.value: ArrivalType.URGENT_INDUCED_LABOR.value,
        PatientType.URGENT_IND_CSECT.value: ArrivalType.URGENT_INDUCED_LABOR.value,
        PatientType.RAND_NONDELIV_LDR.value: ArrivalType.NON_DELIVERY_LDR.value,
        PatientType.RAND_NONDELIV_PP.value: ArrivalType.NON_DELIVERY_PP.value,
    }

# class OBunitId(IntEnum):
#     ENTRY = 0
#     OBS = 1
#     LDR = 2
#     CSECT = 3
#     PP = 4
#     LDRP = 5
#     LD = 6
#     RECOVERY = 8
#     EXIT = 8


class Patient:
    """

    """

    def __init__(self, patient_id: str, patient_type: PatientType, arrival_type: ArrivalType,
                 arr_time: float, los_distributions: Dict, entry_delay: float = 0):
        """

        Parameters
        ----------
        patient_id
        patient_type
        arrival_type
        arr_time
        los_distributions
        entry_delay : float
            Length of time to hold patient in ENTRY node before routing to first location
        """
        self.patient_id = patient_id
        self.patient_type = patient_type
        self.arrival_type = arrival_type
        self.system_arrival_ts = arr_time
        self.entry_delay = entry_delay

        # Initialize unit stop attributes
        self.current_stop_num = -1
        self.previous_unit_id = None
        self.current_unit_id = None
        self.next_unit_id = None

        # Initialize route related attributes
        self.route_graph = None
        self.bed_requests = []
        self.unit_stops = []
        self.planned_los = []
        self.adjusted_los = []
        self.request_entry_ts = []
        self.entry_ts = []
        self.wait_to_enter = []
        self.request_exit_ts = []
        self.exit_ts = []
        self.wait_to_exit = []
        self.system_exit_ts = None


        # Determine route
        # self.route_graph = router.create_route(self.patient_type,
        #                                        los_distributions, self.entry_delay)
        # self.route_graph = len(self.route_graph.edges) + 1  # Includes ENTRY and EXIT

        # Since we have fixed route, just initialize full list to hold bed requests
        # The index numbers are stop numbers and so slot 0 is for ENTRY location
        # self.bed_requests = [None for _ in range(self.route_length)]
        # self.unit_stops = [None for _ in range(self.route_length)]
        # self.planned_los = [None for _ in range(self.route_length)]
        # self.adjusted_los = [None for _ in range(self.route_length)]
        # self.request_entry_ts = [None for _ in range(self.route_length)]
        # self.entry_ts = [None for _ in range(self.route_length)]
        # self.wait_to_enter = [None for _ in range(self.route_length)]
        # self.request_exit_ts = [None for _ in range(self.route_length)]
        # self.exit_ts = [None for _ in range(self.route_length)]
        # self.wait_to_exit = [None for _ in range(self.route_length)]
        self.system_exit_ts = None

    def exit_system(self, env, obsystem):

        logging.debug(
            f"{env.now:.4f}: {self.patient_id} exited system at {env.now:.2f}.")

        # Create dictionaries of timestamps for patient_stop log
        for stop in range(len(self.unit_stops)):
            if self.unit_stops[stop] is not None:
                # noinspection PyUnresolvedReferences
                timestamps = {'patient_id': self.patient_id,
                              'patient_type': self.patient_type,
                              'arrival_type': self.arrival_type,
                              'unit': self.unit_stops[stop],
                              'request_entry_ts': self.request_entry_ts[stop],
                              'entry_ts': self.entry_ts[stop],
                              'request_exit_ts': self.request_exit_ts[stop],
                              'exit_ts': self.exit_ts[stop],
                              'planned_los': self.planned_los[stop],
                              'adjusted_los': self.adjusted_los[stop],
                              'entry_tryentry': self.entry_ts[stop] - self.request_entry_ts[stop],
                              'tryexit_entry': self.request_exit_ts[stop] - self.entry_ts[stop],
                              'exit_tryexit': self.exit_ts[stop] - self.request_exit_ts[stop],
                              'exit_enter': self.exit_ts[stop] - self.entry_ts[stop],
                              'exit_tryenter': self.exit_ts[stop] - self.request_entry_ts[stop],
                              'wait_to_enter': self.wait_to_enter[stop],
                              'wait_to_exit': self.wait_to_exit[stop],
                              'bwaited_to_enter': self.entry_ts[stop] > self.request_entry_ts[stop],
                              'bwaited_to_exit': self.exit_ts[stop] > self.request_exit_ts[stop]}

                obsystem.stops_timestamps_list.append(timestamps)

    def __repr__(self):
        return "patientuid: {}, patient_type: {}, time: {}". \
            format(self.patient_id, self.patient_type, self.system_arrival_ts)


class OBsystem:
    """
    Purpose:

    - acts as a container for the collection of units
    - acts as a container for the timestamps dictionaries

    Instead of passing around the above individually, just pass this system object around

    """

    def __init__(self, env: 'Environment', locations: Dict, los_distributions: Dict,
                 sim_calendar: SimCalendar):

        self.env = env
        self.sim_calendar = sim_calendar
        self.los_distributions = los_distributions

        # Create units container and individual patient care units
        self.obunits = {}
        for location, data in locations.items():
            self.obunits[location] = OBunit(env, name=location, capacity=data['capacity'])

        # Create list to hold timestamps dictionaries (one per patient stop)
        self.patient_timestamps_list = []

        # Create list to hold timestamps dictionaries (one per patient)
        self.stops_timestamps_list = []


class Router(ABC):

    @abstractmethod
    def validate_route_graph(self, route_graph: DiGraph) -> bool:
        """
        Determine if a route is considered valid

        Parameters
        ----------
        route_graph

        Returns
        -------
        True if route is valid
        """
        pass

    @abstractmethod
    def get_next_stop(self, obpatient: type(Patient)):
        pass


class OBStaticRouter(object):
    def __init__(self, env, obsystem, routes):
        """
        Routes patients having a fixed, serial route

        Parameters
        ----------
        env: Environment
        obsystem: OBsystem
        routes: Dict
        """

        self.env = env
        self.obsystem = obsystem

        # Dict of networkx DiGraph objects
        self.route_graphs = {}

        # Create route templates from routes list (of unit numbers)
        for route_name, route in routes.items():
            route_graph = nx.DiGraph()

            # Add edges - simple serial route in this case
            for edge in route['edges']:
                route_graph.add_edge(edge['from'], edge['to'])

            for node in route_graph.nodes():
                nx.set_node_attributes(route_graph,
                                       {node: {'planned_los': 0.0, 'actual_los': 0.0, 'blocked_duration': 0.0}})

            # Each patient will eventually end up with their own copy of the route since
            # it will contain LOS values
            self.route_graphs[route_name] = route_graph.copy()
            logging.debug(f"{self.env.now:.4f}: route graph {route_name} - {route_graph.edges}")

    def validate_route_graph(self, route_graph: DiGraph) -> bool:
        """
        Make sure route is of appropriate structure for router.

        Example: Static routes should have exactly one arc entering and one arc emanating from each non-egress node.

        Parameters
        ----------
        route_graph: DiGraph

        Returns
        -------
        bool
            True if route is valid

        """
        # TODO: Implement route validation rules
        return True

    def create_route(self, patient_type: Enum, los_distributions: Dict, entry_delay: float = 0) -> DiGraph:
        """

        Parameters
        ----------
        patient_type: PatientType
        los_distributions: Dict of Generator
        entry_delay: float (default is 0 implying patient uses ENTRY only as a queueing location)
            Used with scheduled arrivals by holding patient for ``entry_delay`` time units before allowed to enter

        Returns
        -------
        DiGraph
            Nodes are units with LOS information stored as node attributes

        Notes
        -----

        """

        # Copy the route template to create new graph object
        route_graph = deepcopy(self.route_graphs[patient_type])

        # Sample from los distributions for planned_los
        for unit, data in route_graph.nodes(data=True):
            if unit == ENTRY:
                # Entry delays are used to model scheduled procedures. Delay
                # time is number of time units from start of current week
                route_graph.nodes[unit]['planned_los'] = entry_delay
            elif unit == EXIT:
                route_graph.nodes[unit]['planned_los'] = 0.0
            else:
                route_graph.nodes[unit]['planned_los'] = los_distributions[patient_type][unit]()

        return route_graph

    def get_next_stop(self, obpatient: type(Patient)):
        """
        Get next unit in route

        Parameters
        ----------
        obpatient: Patient

        Returns
        -------
        str
            Unit names are used as node id's

        """

        # Get this patient's route graph
        G = obpatient.route_graph

        # Find all possible next units
        successors = [n for n in G.successors(obpatient.current_unit_id)]
        next_unit_id = successors[0]

        if next_unit_id is None:
            logging.error(
                f"{self.env.now:.4f}: {obpatient.patient_id} has no next unit at {obpatient.current_unit_id}.")
            exit(1)

        logging.debug(
            f"{self.env.now:.4f}: {obpatient.patient_id} current_unit_id {obpatient.current_unit_id}, next_unit_id {next_unit_id}")
        return next_unit_id


class OBunit:
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

    def __init__(self, env: 'Environment', name: str, capacity: int = simpy.core.Infinity):

        self.env = env
        self.name = name
        self.id = name
        self.capacity = capacity

        # Use a simpy Resource as one of the class members
        self.unit = simpy.Resource(env, capacity)

        # Statistical accumulators
        self.num_entries = 0
        self.num_exits = 0
        self.tot_occ_time = 0.0
        self.last_entry_ts = None
        self.last_exit_ts = None

        # Create list to hold occupancy tuples (time, occ)
        self.occupancy_list = [(0.0, 0.0)]

    def put(self, obpatient: Patient, obsystem: OBsystem):
        """
        A process method called when a bed is requested in the unit.

        Parameters
        ----------
        obpatient : OBPatient object
            the patient requesting the bed
        obsystem : OBSystem object

        """
        obpatient.current_stop_num += 1
        logging.debug(
            f"{self.env.now:.4f}: {obpatient.patient_id} trying to get {self.name} for stop_num {obpatient.current_stop_num}")

        # Timestamp of request time
        bed_request_ts = self.env.now
        # Request a bed
        bed_request = self.unit.request()
        # Store bed request and timestamp in patient's request lists
        obpatient.bed_requests[obpatient.current_stop_num] = bed_request
        obpatient.unit_stops[obpatient.current_stop_num] = self.name
        obpatient.request_entry_ts[obpatient.current_stop_num] = self.env.now

        # If we are coming from upstream unit, we are trying to exit that unit now
        if obpatient.bed_requests[obpatient.current_stop_num - 1] is not None:
            obpatient.request_exit_ts[obpatient.current_stop_num - 1] = self.env.now

        # Yield until we get a bed
        yield bed_request

        # Seized a bed.
        # Update patient flow attributes for this stop
        obpatient.entry_ts[obpatient.current_stop_num] = self.env.now
        obpatient.wait_to_enter[obpatient.current_stop_num] = \
            self.env.now - obpatient.request_entry_ts[obpatient.current_stop_num]
        obpatient.current_unit_id = self.name

        # Update unit attributes
        self.num_entries += 1
        self.last_entry_ts = self.env.now

        # Increment occupancy
        self.inc_occ()

        # Check if we have a bed from a previous stay and release it if we do.
        # Update stats for previous unit.
        if obpatient.bed_requests[obpatient.current_stop_num - 1] is not None:
            obpatient.exit_ts[obpatient.current_stop_num - 1] = self.env.now
            obpatient.wait_to_exit[obpatient.current_stop_num - 1] = \
                self.env.now - obpatient.request_exit_ts[obpatient.current_stop_num - 1]
            obpatient.previous_unit_id = obpatient.unit_stops[obpatient.current_stop_num - 1]
            previous_unit = obsystem.obunits[obpatient.previous_unit_id]
            previous_request = obpatient.bed_requests[obpatient.current_stop_num - 1]
            # Release the previous bed
            previous_unit.unit.release(previous_request)
            # Accumulate total time this unit occupied and other unit attributes
            previous_unit.tot_occ_time += \
                self.env.now - obpatient.entry_ts[obpatient.current_stop_num - 1]
            previous_unit.num_exits += 1
            previous_unit.last_exit_ts = self.env.now
            # Decrement occupancy in previous unit since bed now released
            previous_unit.dec_occ()

        logging.debug(f"{self.env.now:.4f}: {obpatient.patient_id} entering {self.name}")
        logging.debug(
            f"{self.env.now:.4f}: {obpatient.patient_id} waited {self.env.now - bed_request_ts:.4f} time units for {self.name} bed")

        # Generate los
        # TODO: Modeling discharge timing
        los = obpatient.route_graph.nodes(data=True)[obpatient.current_unit_id]['planned_los']
        obpatient.planned_los[obpatient.current_stop_num] = los

        # Do any blocking related los adjustments.
        # TODO: This is hard coded logic. Need general scheme for blocking adjustments.
        if self.name == 'LDR':
            adj_los = max(0, los - obpatient.wait_to_exit[obpatient.current_stop_num - 1])
        else:
            adj_los = los

        obpatient.adjusted_los[obpatient.current_stop_num] = adj_los

        # Wait for LOS to elapse
        yield self.env.timeout(adj_los)

        # Go to next destination (which could be EXIT)
        if obpatient.current_unit_id != EXIT:
            # Determine next stop in route and try to get a bed in that unit
            obpatient.next_unit_id = obpatient.router.get_next_stop(obpatient)
            self.env.process(obsystem.obunits[obpatient.next_unit_id].put(obpatient, obsystem))
        else:
            # Patient is ready to exit system
            obpatient.previous_unit_id = obpatient.unit_stops[obpatient.current_stop_num]
            previous_unit = obsystem.obunits[obpatient.previous_unit_id]
            previous_request = obpatient.bed_requests[obpatient.current_stop_num]
            # Release the bed
            previous_unit.unit.release(previous_request)
            # Accumulate total time this unit occupied and other unit attributes
            previous_unit.tot_occ_time += \
                self.env.now - obpatient.entry_ts[obpatient.current_stop_num]
            previous_unit.num_exits += 1
            previous_unit.last_exit_ts = self.env.now

            # # Decrement occupancy in previous unit since bed now released
            previous_unit.dec_occ()

            obpatient.request_exit_ts[obpatient.current_stop_num] = self.env.now
            obpatient.exit_ts[obpatient.current_stop_num] = self.env.now
            obpatient.exit_system(self.env, obsystem)

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

    def basic_stats_msg(self):
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

        msg = "{:6}:\t Entries={}, Exits={}, Occ={}, ALOS={:4.2f}". \
            format(self.name, self.num_entries, self.num_exits,
                   self.unit.count, alos)
        return msg


class PoissonArrivals:
    """ Generates patients according to a stationary Poisson process with specified rate.

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        uid : str
            unique name of random arrival stream
        arrival_rate : float
            Poisson arrival rate (expected number of arrivals per unit time)
        arrival_stream_rg : numpy.random.Generator
            used for interarrival time generation
        stop_time : float
            Stops generation at the stoptime. (default Infinity)
        max_arrivals : int
            Stops generation after max_arrivals. (default Infinity)
        entity_cls : Class (Default is None)
            Each arrival creates an instance of this class
        kwargs : dict
            additional arguments to pass to entity_cls.__init__
    """

    def __init__(self, env, uid, arrival_rate, arrival_stream_rg,
                 stop_time=simpy.core.Infinity, max_arrivals=simpy.core.Infinity,
                 entity_cls=None, **kwargs):

        # Parameter attributes
        self.env = env
        self.uid = uid
        self.arr_rate = arrival_rate
        self.arr_stream_rg = arrival_stream_rg
        self.stop_time = stop_time
        self.max_arrivals = max_arrivals
        self.entity_cls = entity_cls

        # State attributes
        self.num_entities_created = 0

        # Trigger the run() method and register it as a SimPy process
        env.process(self.run())

    def run(self):
        """
        Generate entities until stopping condition met
        """

        # Main entity creation loop that terminates when stoptime reached
        while self.env.now < self.stop_time and \
                self.num_entities_created < self.max_arrivals:
            # Compute next interarrival time
            iat = self.arr_stream_rg.exponential(1.0 / self.arr_rate)
            # Delay until time for next arrival
            yield self.env.timeout(iat)
            self.num_entities_created += 1
            patient_type = self.assign_patient_type()
            new_patient_id = f'{patient_type}_{self.num_entities_created}'
            # Create new patient
            obpatient = Patient(
                new_patient_id, patient_type, self.uid, self.env.now, self.config.los_distributions)

            logging.debug(
                f"{self.env.now:.4f}: {obpatient.patient_id} created at {self.env.now:.4f} ({self.obsystem.sim_calendar.now()}).")

            # Initiate process of patient entering system
            self.env.process(self.obsystem.obunits[ENTRY].put(obpatient, self.obsystem))





class OBPatientGeneratorPoisson:
    """ Generates patients according to a stationary Poisson process.

        Parameters
        ----------
        uid : str
            name of random arrival stream
        env : simpy.Environment
            the simulation environment
        obsystem : OBSystem
            the OB system containing the obunits list
        arr_rate : float
            Poisson arrival rate (expected number of arrivals per unit time)
        arr_stream_rg : numpy.random.Generator
            used for interarrival time generation
        stop_time : float
            Stops generation at the stoptime. (default Infinity)
        max_arrivals : int
            Stops generation after max_arrivals. (default Infinity)

    """

    def __init__(self, uid, env, config, obsystem, arr_rate, arr_stream_rg,
                 stop_time=simpy.core.Infinity, max_arrivals=simpy.core.Infinity):

        self.uid = uid
        self.env = env
        self.config = config
        self.obsystem = obsystem
        self.arr_rate = arr_rate
        self.arr_stream_rg = arr_stream_rg
        self.stop_time = stop_time
        self.max_arrivals = max_arrivals
        self.out = None
        self.num_patients_created = 0

        # Trigger the run() method and register it as a SimPy process
        env.process(self.run())

    def run(self):
        """
        Generate patients.
        """

        # Main generator loop that terminates when stoptime reached
        while self.env.now < self.stop_time and \
                self.num_patients_created < self.max_arrivals:
            # Compute next interarrival time
            iat = self.arr_stream_rg.exponential(1.0 / self.arr_rate)
            # Delay until time for next arrival
            yield self.env.timeout(iat)
            self.num_patients_created += 1
            patient_type = self.assign_patient_type()
            new_patient_id = f'{patient_type}_{self.num_patients_created}'
            # Create new patient
            obpatient = Patient(
                new_patient_id, patient_type, self.uid, self.env.now, self.config.los_distributions)

            logging.debug(
                f"{self.env.now:.4f}: {obpatient.patient_id} created at {self.env.now:.4f} ({self.obsystem.sim_calendar.now()}).")

            # Initiate process of patient entering system
            self.env.process(self.obsystem.obunits[ENTRY].put(obpatient, self.obsystem))

    def assign_patient_type(self):
        if self.uid == ArrivalType.SPONT_LABOR.value:
            # Determine if labor augmented or not
            if self.arr_stream_rg.random() < self.config.branching_probabilities['pct_spont_labor_aug']:
                # Augmented labor
                if self.arr_stream_rg.random() < self.config.branching_probabilities['pct_aug_labor_to_c']:
                    return PatientType.RAND_AUG_CSECT.value
                else:
                    return PatientType.RAND_AUG_REG.value
            else:
                # Labor not augmented
                if self.arr_stream_rg.random() < self.config.branching_probabilities['pct_spont_labor_to_c']:
                    return PatientType.RAND_SPONT_CSECT.value
                else:
                    return PatientType.RAND_SPONT_REG.value
        elif self.uid == ArrivalType.NON_DELIVERY_LDR.value:
            return PatientType.RAND_NONDELIV_LDR.value
        elif self.uid == ArrivalType.NON_DELIVERY_PP.value:
            return PatientType.RAND_NONDELIV_PP.value
        elif self.uid == ArrivalType.URGENT_INDUCED_LABOR.value:
            if self.arr_stream_rg.random() < self.config.branching_probabilities['pct_urg_ind_to_c']:
                return PatientType.URGENT_IND_CSECT.value
            else:
                return PatientType.URGENT_IND_REG.value


class OBPatientGeneratorWeeklyStaticSchedule:
    """ Generates patients according to a repeating one week schedule

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        obsystem : OBSystem
            the OB system containing the obunits list
        router : OBStaticRouter like
            used to route new arrival to first location
        schedule : ndarray of int of shape (7, 24)
            Number of scheduled arrivals by day of week and hour of day
        stop_time : float
            Stops generation at the stoptime. (default Infinity)
        max_arrivals : int
            Stops generation after max_arrivals. (default Infinity)

    """

    def __init__(self, uid: Union[str, int], env: 'Environment',
                 config: OBConfig, obsystem: OBsystem,
                 router: OBStaticRouter, schedule: NDArray,
                 arr_stream_rg: DTypeLike,
                 stop_time=simpy.core.Infinity, max_arrivals=simpy.core.Infinity):

        self.uid = uid
        self.env = env
        self.config = config
        self.obsystem = obsystem
        self.router = router
        self.schedule = schedule
        self.arr_stream_rg = arr_stream_rg
        self.stop_time = stop_time
        self.max_arrivals = max_arrivals
        self.out = None
        self.num_patients_created = 0

        # Trigger the run() method and register it as a SimPy process
        env.process(self.run())

    def run(self):
        """
        Generate patients.
        """

        weekly_cycle_length = pd.to_timedelta(1, unit='w') / pd.to_timedelta(1, unit=self.config.base_time_unit)
        # Main generator loop that terminates when stoptime reached
        while self.env.now < self.stop_time and \
                self.num_patients_created < self.max_arrivals:

            # Create tuples (day, hour, num scheduled)
            arrival_epochs = [(d, h, self.schedule[d, h]) for d in range(6)
                              for h in range(23) if self.schedule[d, h] > 0]

            # Create schedule patients and send them to ENTRY to wait until their scheduled procedure time
            for (day, hour, num_sched) in arrival_epochs:
                # Compute number of time units from the start of current week until scheduled procedure time
                time_of_week = \
                    pd.to_timedelta(day * 24 + hour, unit='h') / pd.to_timedelta(1, unit=self.config.base_time_unit)
                for patient in range(num_sched):
                    self.num_patients_created += 1
                    patient_type = self.assign_patient_type()
                    new_patient_id = f'{patient_type}_{self.num_patients_created}'
                    # Create new patient
                    obpatient = Patient(
                        new_patient_id, patient_type, self.uid, self.env.now, self.router,
                        self.config.los_distributions, entry_delay=time_of_week)

                    logging.debug(
                        f"{self.env.now:.4f}: {obpatient.patient_id} created at {self.env.now:.4f} ({self.obsystem.sim_calendar.now()}).")

                    # Initiate process of patient entering system
                    self.env.process(self.obsystem.obunits[ENTRY].put(obpatient, self.obsystem))

            # Yield until beginning of next weekly cycle
            yield self.env.timeout(weekly_cycle_length)

    def assign_patient_type(self):
        # Determine if scheduled c-section or scheduled induction
        if self.uid == ArrivalType.SCHED_CSECT.value:
            # Scheduled c-section
            return PatientType.SCHED_CSECT.value
        else:
            # Determine if scheduled induction ends up with c-section
            if self.arr_stream_rg.random() < self.config.branching_probabilities['pct_sched_ind_to_c']:
                return PatientType.SCHED_IND_CSECT.value
            else:
                return PatientType.SCHED_IND_REG.value


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


def simulate(config: OBConfig, rep_num: int):
    """
    Run one replication of simulation model.

    Parameters
    ----------
    config : OBConfig
    rep_num : int, simulation replication number

    Returns
    -------
    Dict: Summary statistics for the replication

    """

    # Determine stopping conditions specified in config file
    if hasattr(config, 'max_arrivals'):
        max_arrivals = config.max_arrivals
    else:
        max_arrivals = simpy.core.Infinity

    if hasattr(config, 'run_time'):
        run_time = config.run_time
    else:
        run_time = simpy.core.Infinity

    # Initialize a simulation environment and calendar
    env = simpy.Environment()
    sim_calendar = SimCalendar(env, config)

    # Create an OB System
    obsystem = OBsystem(env, config.locations, config.los_distributions, sim_calendar)

    # Create router
    router = OBStaticRouter(env, obsystem, config.routes, config.rg['los'])

    # Create patient generators for random arrivals
    patient_generators_poisson = {}
    for poisson_id, arr_rate in config.rand_arrival_rates.items():
        if arr_rate > 0.0 and config.rand_arrival_toggles[poisson_id] > 0:
            patient_generators_poisson[poisson_id] = \
                OBPatientGeneratorPoisson(
                    poisson_id, env, config, obsystem, router,
                    config.rand_arrival_rates[poisson_id], config.rg['arrivals'],
                    stop_time=run_time, max_arrivals=max_arrivals)

    # Create patient generators for scheduled c-sections and scheduled inductions
    patient_generators_scheduled = {}
    for sched_id, schedule in config.schedules.items():
        if len(schedule) > 0 and config.sched_arrival_toggles[sched_id] > 0:
            patient_generators_scheduled[sched_id] = \
                OBPatientGeneratorWeeklyStaticSchedule(
                    sched_id, env, config, obsystem, router,
                    config.schedules[sched_id], config.rg['arrivals'],
                    stop_time=run_time, max_arrivals=max_arrivals)

    # TODO - create patient generator for urgent inductions. For now, we'll ignore these patient types.

    # Run the simulation replication
    env.run(until=config.run_time)

    # TODO - design output processing scheme
    # Patient generator stats
    print(obio.output_header("Patient generator stats", 70, config.scenario, rep_num))

    for poisson_id in patient_generators_poisson:
        print(f"Num patients generated by {poisson_id}: {patient_generators_poisson[poisson_id].num_patients_created}")

    for sched_id in patient_generators_scheduled:
        print(
            f"Num patients generated by {sched_id}: {patient_generators_poisson[sched_id].num_patients_created}")

    # Unit stats
    print(obio.output_header("Unit entry/exit stats", 70, config.scenario, rep_num))
    for unit_name, unit in obsystem.obunits.items():
        print(unit.basic_stats_msg())

    # System exit stats
    print(obio.output_header("Patient exit stats", 70, config.scenario, rep_num))
    print("Num patients exiting system: {}".format(obsystem.obunits[EXIT].num_exits))
    print("Last exit at: {:.2f}\n".format(obsystem.obunits[EXIT].last_exit_ts))

    # Compute occupancy stats
    occ_stats_df, occ_log_df = obstat.compute_occ_stats(obsystem, config.run_time,
                                                        warmup=config.warmup_time,
                                                        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    if config.paths['occ_stats'] is not None:
        obio.write_stats('occ_stats', config.paths['occ_stats'], occ_stats_df, config.scenario, rep_num)

    # Print occupancy summary
    print(obio.output_header("Occupancy stats", 80, config.scenario, rep_num))
    print(occ_stats_df)

    # Compute summary stats for this scenario replication
    scenario_rep_summary_dict = obstat.process_stop_log(
        config.scenario, rep_num, obsystem, config.paths['occ_stats'], config.run_time, config.warmup_time)

    # Write log files
    if config.paths['occ_logs'] is not None:
        obio.write_log('occ_log', config.paths['occ_logs'], occ_log_df, config.scenario, rep_num)

    if config.paths['stop_logs'] is not None:
        stop_log_df = pd.DataFrame(obsystem.stops_timestamps_list)
        # Convert timestamps to calendar time if needed
        if obsystem.sim_calendar.use_calendar_time:
            stop_log_df['request_entry_ts'] = stop_log_df['request_entry_ts'].map(
                lambda x: obsystem.sim_calendar.to_sim_calendar_time(x))
            stop_log_df['entry_ts'] = stop_log_df['entry_ts'].map(lambda x: obsystem.sim_calendar.to_sim_calendar_time(x))
            stop_log_df['request_exit_ts'] = stop_log_df['request_exit_ts'].map(
                lambda x: obsystem.sim_calendar.to_sim_calendar_time(x))
            stop_log_df['exit_ts'] = stop_log_df['exit_ts'].map(lambda x: obsystem.sim_calendar.to_sim_calendar_time(x))
        obio.write_log('stop_log', config.paths['stop_logs'], stop_log_df, config.scenario, rep_num)

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
    # logger = logging.getLogger()
    # logger.setLevel(args.loglevel)
    #
    # # Create the Handler for logging data to console.
    # logger_handler = logging.StreamHandler()
    # logger_handler.setLevel(args.loglevel)
    #
    # # Create a Formatter for formatting the log messages
    # logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #
    # # Add the Formatter to the Handler
    # logger_handler.setFormatter(logger_formatter)
    #
    # # Add the Handler to the Logger
    # logger.addHandler(logger_handler)

    # Quick setup of root logger
    # logging.basicConfig(
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     stream=sys.stdout,
    # )
    # # Retrieve root logger (no logger name passed to ``getLogger()``) and update its level
    # logger = logging.getLogger()
    # logger.setLevel(args.loglevel)

    # Load scenario configuration file and create OBConfig object
    config_dict = obio.load_config(args.config)
    config = OBConfig(config_dict)

    # Initialize scenario specific variables
    scenario = config.scenario
    summary_stat_path = \
        config_dict['outputs']['summary_stats']['path'] / Path(f'summary_stats_scenario_{scenario}.csv')

    # Check for undercapacitated system and compute basic load stats
    unit_load, unit_intensity = obq.unit_loads(config)
    # pprint(unit_load)
    # pprint(unit_intensity)

    results = []
    for i in range(1, config.num_replications + 1):
        print(f'Running scenario {scenario}, replication {i}')
        scenario_rep_summary_dict = simulate(config, i)
        results.append(scenario_rep_summary_dict)

    # Convert results dict to DataFrame
    scenario_rep_summary_df = pd.DataFrame(results)
    obio.write_summary_stats(summary_stat_path, scenario_rep_summary_df)

    return 0


if __name__ == '__main__':
    sys.exit(main())
