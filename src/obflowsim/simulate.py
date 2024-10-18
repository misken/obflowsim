from __future__ import annotations

import sys
import logging
from logging import Logger
from pathlib import Path
import argparse

# if TYPE_CHECKING and TYPE_CHECKING != 'SPHINX':  # Avoid circular import

import pandas as pd
import simpy


import obflowsim.io as obio
import obflowsim.stats as obstat
import obflowsim.obqueueing as obq
import obflowsim.config as obconfig
from obflowsim.arrivals import create_poisson_generators, create_scheduled_generators
from obflowsim.clock_tools import SimCalendar
from obflowsim.patient_flow_system import PatientFlowSystem
from obflowsim.routing import StaticRouter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obflowsim.config import Config


# TODO - make sure all docstrings are complete


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

    # Create patient generators
    patient_generators_poisson = create_poisson_generators(env, config, obsystem)
    patient_generators_scheduled = create_scheduled_generators(env, config, obsystem)

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
