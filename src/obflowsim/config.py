from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from numpy.random import default_rng

from typing import (
    Dict, Tuple,
)

import simpy.core

import obflowsim.obconstants as obconstants
import obflowsim.io as obio


class ConfigError(Exception):
    """Exception raised for a variety of configuration errors."""


class Config:
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
    MORE



    Methods
    -------

    """

    def __init__(self, config_dict: Dict):

        # Scenario
        try:
            self.scenario = config_dict['scenario']
        except KeyError:
            # Default scenario name using current datetime
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
            self.scenario = f'scenario_{ts}'

        # Calendar related
        try:
            self.use_calendar_time = config_dict['run_settings']['use_calendar_time']
        except KeyError:
            self.use_calendar_time = obconstants.DEFAULT_USE_CALENDAR_TIME

        try:
            self.start_date = pd.Timestamp(config_dict['run_settings']['start_date'])
        except KeyError:
            # Default start_date is first Monday in 2024
            self.start_date = pd.Timestamp(obconstants.DEFAULT_START_DATE)

        try:
            self.base_time_unit = config_dict['run_settings']['base_time_unit']
            if self.base_time_unit not in ['h', 'm']:
                raise ValueError('Base time unit must be h or m.')
        except KeyError:
            # Default start_date is first Monday in 2024
            self.base_time_unit = obconstants.DEFAULT_BASE_TIME_UNIT

        # Stopping conditions and other run time settings
        try:
            self.run_time = config_dict['run_settings']['run_time']
        except KeyError:
            self.run_time = obconstants.DEFAULT_RUN_TIME

        try:
            self.max_arrivals = config_dict['run_settings']['max_arrivals']
        except KeyError:
            self.max_arrivals = obconstants.DEFAULT_MAX_ARRIVALS

        if self.run_time == simpy.core.Infinity and self.max_arrivals == simpy.core.Infinity:
            raise ConfigError(f'Either run_time or max_arrivals must be finite.')

        try:
            self.warmup_time = config_dict['run_settings']['warmup_time']
        except KeyError:
            self.warmup_time = obconstants.DEFAULT_WARMUP_TIME

        try:
            self.num_replications = config_dict['run_settings']['num_replications']
        except KeyError:
            self.num_replications = obconstants.DEFAULT_NUM_REPLICATIONS

        # Create random number generators
        self.rg = {}
        for stream, seed in config_dict['random_number_streams'].items():
            self.rg[stream] = default_rng(seed)

        # Random arrival rates and toggles
        self.rand_arrival_rates = config_dict['rand_arrival_rates']
        self.rand_arrival_toggles = config_dict['rand_arrival_toggles']

        # Scheduled arrivals and toggles
        self.schedules = {}
        if 'sched_csect' in config_dict['schedule_files']:
            sched_file_c = config_dict['schedule_files']['sched_csect']
            self.schedules['sched_csect'] = obio.process_schedule_file(sched_file_c,
                                                                       self.start_date, self.base_time_unit)

        if 'sched_induced_labor' in config_dict['schedule_files']:
            sched_file_ind = config_dict['schedule_files']['sched_induced_labor']
            self.schedules['sched_induced_labor'] = obio.process_schedule_file(sched_file_ind,
                                                                               self.start_date, self.base_time_unit)

        self.sched_arrival_toggles = config_dict['sched_arrival_toggles']

        # Discharge patterns
        try:
            self.discharge_pattern_file = config_dict['discharge_pattern_file']
            self.discharge_pattern = obio.process_discharge_pattern_file(self.discharge_pattern_file)
        except KeyError:
            self.discharge_pattern_file = None
            self.discharge_pattern = None

        # Branching probabilities
        self.branching_probabilities = config_dict['branching_probabilities']

        # Length of stay
        self.los_params = config_dict['los_params']
        self.los_distributions, self.los_means = obio.create_los_partials(config_dict['los_distributions'],
                                                                          self.los_params, self.rg['los'])

        self.locations = config_dict['locations']
        self.routes = config_dict['routes']
        self.outputs = config_dict['outputs']

        # Output paths
        outputs = self.outputs.keys()
        self.paths = {output: None for output in outputs}
        for output in outputs:
            if self.outputs[output]['write']:
                Path(self.outputs[output]['path']).mkdir(parents=True, exist_ok=True)
                self.paths[output] = Path(self.outputs[output]['path'])


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
