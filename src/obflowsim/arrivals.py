from __future__ import annotations

import logging

import pandas as pd
import simpy
from numpy._typing import NDArray

from obflowsim.config import Config
from obflowsim.obconstants import ArrivalType
from obflowsim.patient import Patient

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obflowsim.patient_flow_system import PatientFlowSystem


def create_poisson_generators(env, config: Config, obsystem: PatientFlowSystem):
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

    return patient_generators_poisson


def create_scheduled_generators(env, config: Config, obsystem: PatientFlowSystem):
    patient_generators_scheduled = {}
    for sched_id, schedule in config.schedules.items():
        if len(schedule) > 0 and config.sched_arrival_toggles[sched_id] > 0:
            patient_generators_scheduled[sched_id] = \
                PatientGeneratorWeeklyStaticSchedule(
                    env, sched_id, config.schedules[sched_id],
                    stop_time=config.run_time, max_arrivals=config.max_arrivals, patient_flow_system=obsystem)

    return patient_generators_scheduled

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
                 patient_flow_system=None):

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
