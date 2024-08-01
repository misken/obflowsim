import sys
import argparse
from pathlib import Path
from enum import IntEnum

import pandas as pd
from pandas import Timestamp
import yaml
import json

import numpy as np
from numpy.random import default_rng
import numpy.random
import json
from functools import partial
import copy

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    NewType,
    Optional,
    Tuple,
)

import obflowsim.obflow_stat as obstat
import obflowsim.obconstants as obconstants


def load_config(cfg):
    """
    Load YAML configuration file

    Parameters
    ----------
    cfg : str, YAML configuration filename

    Returns
    -------
    dict
    """

    # Read inputs from config file
    with open(cfg, 'rt') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    return yaml_config


def _get_args_and_kwargs(*args, **kwargs):
    return args, kwargs


def _convert_str_to_args_and_kwargs(s):
    return eval(s.replace(s[:s.find('(')], '_get_args_and_kwargs'))


def _convert_str_to_func_name(s):
    return s[:s.find('(')]


def create_los_partials(raw_los_dists: Dict, los_params: Dict, rg):
    """

    Parameters
    ----------
    rg
    raw_los_dists: Dict
    los_params: Dict

    Returns
    -------
    Dict of partial functions for LOS generation by pat type and unit

    """

    # Replace all los_param use with literal values
    los_dists_str_json = json.dumps(raw_los_dists)

    los_params_sorted = [key for key in los_params]
    los_params_sorted.sort(key=len, reverse=True)

    for param in los_params_sorted:
        los_dists_str_json = los_dists_str_json.replace(param, str(los_params[param]))

    los_dists_instantiated = json.loads(los_dists_str_json)

    los_dists_partials = copy.deepcopy(los_dists_instantiated)
    los_means = copy.deepcopy(los_dists_instantiated)
    for key_pat_type in los_dists_partials:
        for key_unit, raw_dist_str in los_dists_partials[key_pat_type].items():
            func_name = _convert_str_to_func_name(raw_dist_str)
            # Check for valid func name
            if func_name in obconstants.ALLOWED_LOS_DIST_LIST:
                args, kwargs = _convert_str_to_args_and_kwargs(raw_dist_str)
                partial_dist_func = partial(eval(f'rg.{func_name}'), *args, **kwargs)
                los_dists_partials[key_pat_type][key_unit] = partial_dist_func
                los_means[key_pat_type][key_unit] = obstat.mean_from_dist_params(func_name, args, kwargs)
            else:
                raise NameError(f"The use of '{func_name}' is not allowed")

    return los_dists_partials, los_means


def process_schedule_file(sched_file: str | Path, start_date: Timestamp, base_time_unit: str):
    """

    Parameters
    ----------
    sched_file
    start_date
    base_time_unit

    Returns
    -------

    """
    Weekdays = IntEnum('Weekdays', 'mon tue wed thu fri sat sun', start=0)
    start_dow = start_date.weekday()

    with open(sched_file, 'r') as f:
        sched_lines = [line.strip().split(',') for line in f.readlines()]

    new_sched_lines = []
    for line in sched_lines:
        line[0] = line[0].lower()
        dow = Weekdays[line[0]].value
        line[1] = float(line[1])
        line[2] = int(line[2])

        # Compute entry_delay as offset from start_date
        if dow >= start_dow:
            num_days = dow - start_dow
        else:
            num_days = 7 - (start_dow - dow)

        if base_time_unit == 'h':
            entry_delay_time = 24 * num_days + line[1]
        elif base_time_unit == 'm':
            entry_delay_time = 1440 * num_days + line[1]
        else:
            raise ValueError('Base time unit must be h or m.')

        new_sched_lines.append([entry_delay_time, line[2]])

    return new_sched_lines


def process_discharge_pattern_file(discharge_file: str | Path):
    """

    Parameters
    ----------
    discharge_file

    Returns
    -------

    """

    with open(discharge_file, 'r') as f:
        discharge_lines = [line.strip().split() for line in f.readlines() if len(line.strip()) > 0]

    bins = np.array([int(x) for x in list(zip(*discharge_lines))[0]])
    cdf = np.array([float(x) for x in list(zip(*discharge_lines))[1]])

    return bins, cdf


def setup_output_paths(config, rep_num):
    stats = config.output.keys()
    config.paths = {stat: None for stat in stats}
    for stat in stats:
        if config.output[stat]['write']:
            Path(config.output[stat]['path']).mkdir(parents=True, exist_ok=True)
            config.paths[stat] = Path(
                config.output[stat]['path']) / f"{stat}_scenario_{config.scenario}_rep_{rep_num}.csv"

    return config


def write_log(which_log, log_path, df, scenario, rep_num, egress=True):
    """

    Parameters
    ----------
    csv_path
    obsystem
    egress

    Returns
    -------

    """
    csv_path = Path(log_path / f"{which_log}_scenario_{scenario}_rep_{rep_num}.csv")

    if egress:
        df.to_csv(csv_path, index=False)
    else:
        df[(df['unit'] != 'ENTRY') & (df['unit'] != 'EXIT')].to_csv(csv_path, index=False)


def concat_stop_summaries(stop_summaries_path, output_path,
                          summary_stats_file_stem='summary_stats_scenario',
                          output_file_stem=f'scenario_rep_simout'):
    """
    Creates and writes out summary by scenario and replication to csv

    Parameters
    ----------
    stop_summaries_path
    output_path
    summary_stats_file_stem
    output_file_stem

    Returns
    -------

    """

    summary_files = [fn for fn in Path(stop_summaries_path).glob(f'{summary_stats_file_stem}*.csv')]
    scenario_rep_summary_df = pd.concat([pd.read_csv(fn) for fn in summary_files])

    output_csv_file = Path(output_path) / f'{output_file_stem}.csv'
    scenario_rep_summary_df = scenario_rep_summary_df.sort_values(by=['scenario', 'rep'])

    scenario_rep_summary_df.to_csv(output_csv_file, index=False)

    print(f'Scenario replication csv file written to {output_csv_file}')


def write_stats(which_stats, stats_path, stats_df, scenario, rep_num):
    """
    Export occupancy stats to csv

    Parameters
    ----------
    occ_stats_path
    occ_stats_df

    Returns
    -------
    """
    csv_path = Path(stats_path / f"{which_stats}_scenario_{scenario}_rep_{rep_num}.csv")
    stats_df.to_csv(csv_path, index=False)


def write_occ_stats(occ_stats_path, occ_stats_df):
    """
    Export occupancy stats to csv

    Parameters
    ----------
    occ_stats_path
    occ_stats_df

    Returns
    -------
    """

    occ_stats_df.to_csv(occ_stats_path, index=False)


def occ_stats_to_string(occ_stats_df, scenario, rep_num):
    """
    Export occupancy stats to csv

    Parameters
    ----------
    occ_stats_df

    Returns
    -------
    """

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(output_header("Occupancy stats", 130, scenario, rep_num))
    occ_stats_string = occ_stats_df.reset_index(drop=True).to_string(index=False)
    pd.reset_option('display.width')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.float_format')

    return occ_stats_string


def write_summary_stats(summary_stats_path, summary_stats_df):
    """
    Export occupancy stats to csv

    Parameters
    ----------
    summary_stats_path
    summary_stats_df

    Returns
    -------
    """

    summary_stats_df.to_csv(summary_stats_path, index=False)


def output_header(msg, linelen, scenario, rep_num):
    header = f"\n{msg} (scenario={scenario} rep={rep_num})\n{'-' * linelen}"
    return header


def process_command_line(argv=None):
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='obflow_io',
                                     description='create the main output summary file with one row per (scenario, '
                                                 'replication) pair')

    # Add arguments
    parser.add_argument(
        "stop_summaries_path", type=str,
        help="Folder containing the scenario rep summaries created by simulation runs"
    )

    parser.add_argument(
        "output_path", type=str,
        help="Destination folder for combined scenario rep summary csv"
    )

    parser.add_argument(
        "summary_stats_file_stem", type=str,
        help="Summary stat file name without extension"
    )

    parser.add_argument(
        "output_file_stem", type=str,
        help="Combined summary stat file name without extension to be output"
    )

    # do the parsing
    args = parser.parse_args(argv)

    return args


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

    concat_stop_summaries(args.stop_summaries_path,
                          args.output_path,
                          args.summary_stats_file_stem,
                          args.output_file_stem)


if __name__ == '__main__':
    sys.exit(main())
