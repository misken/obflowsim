import sys
import argparse
from pathlib import Path

import pandas as pd
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

ALLOWED_LOS_DIST_LIST = ['beta', 'binomial', 'chisquare', 'exponential', 'gamma',
                         'geometric', 'hypergeometric', 'laplace', 'logistic', 'lognormal',
                         'multinomial', 'negative_binomial', 'normal', 'pareto',
                         'poisson', 'triangular', 'uniform', 'weibull', 'zipf']


def load_config(cfg):
    """

    Parameters
    ----------
    cfg : str, configuration filename

    Returns
    -------
    dict
    """

    # Read inputs from config file
    with open(cfg, 'rt') as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    return yaml_config


def get_args_and_kwargs(*args, **kwargs):
    return args, kwargs


def convert_str_to_args_and_kwargs(s):
    return eval(s.replace(s[:s.find('(')], 'get_args_and_kwargs'))


def convert_str_to_func_name(s):
    return s[:s.find('(')]


def create_los_partials(raw_los_dists: Dict, los_params: Dict, rg):
    """

    Parameters
    ----------
    rg
    raw_los_dist: Dict
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
    for key_pat_type in los_dists_partials:
        for key_unit, raw_dist_str in los_dists_partials[key_pat_type].items():
            func_name = convert_str_to_func_name(raw_dist_str)
            # Check for valid func name
            if func_name in ALLOWED_LOS_DIST_LIST:
                args, kwargs = convert_str_to_args_and_kwargs(raw_dist_str)
                partial_dist_func = partial(eval(f'rg.{func_name}'), *args)
                los_dists_partials[key_pat_type][key_unit] = partial_dist_func
            else:
                raise NameError(f"The use of '{func_name}' is not allowed")

    return los_dists_partials


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
    csv_path =  Path(log_path / f"{which_log}_scenario_{scenario}_rep_{rep_num}.csv")

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
                                     description='create the main output summary file with one row per (scenario, replication) pair')

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
