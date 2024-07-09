=====
Usage
=====

.. highlight:: bash

Setting up and running a multiscenario simulation experiment
-------------------------------------------------------------

The main steps are:

* Create the scenario input file (``scenario_tools``)
* Create the run settings file (manually for now)
* Generate simulation config file and for each scenario and shell scripts to run the scenarios (``create_configs``)
* Run the shell scripts to run the simulation scenarios (``obflow_sim``)
* Concatenate the scenario rep files into one big scenario rep file(``obflow_io``)
* Create the simulation summary files (``obflow_stat``)

The input output summary file created after this final step
is ready to use in metamodel fitting and evaluation. The summary file
contains simulation inputs, outputs, and queueing approximations for
each scenario run. 


Create the scenario input file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The scenario input file is generated from a *scenario recipe* - a YAML
formatted file that specifies one or more values for each of the simulation input
parameters. Here is an example:

.. code::
    arrival_rate:
    - 0.2
    - 0.6
    - 1.0
    mean_los_obs:
    - 2.0
    mean_los_ldr:
    - 12.0
    mean_los_csect:
    - 2.0
    mean_los_pp_noc:
    - 48.0
    mean_los_pp_c:
    - 72.0
    c_sect_prob:
    - 0.25
    num_erlang_stages_obs:
    - 1
    num_erlang_stages_ldr:
    - 2
    num_erlang_stages_csect:
    - 1
    num_erlang_stages_pp:
    - 8
    acc_obs:
    - 0.95
    - 0.99
    acc_ldr:
    - 0.85
    - 0.95
    acc_pp:
    - 0.85
    - 0.95

A few important things to note:

* The recipe file can be created manually or via the code in ``scenario_scratchpad.py``.
* The ``acc_obs``, ``acc_ldr``, and ``acc_pp`` accommodation probabilities lead to capacity lower bounds
based on an inverse Poisson approach. You can also directly specify `cap_obs`, `cap_ldr`,
and ``cap_pp`` capacity levels.

Assume you've create a scenario recipe file named ``exp14_scenario_recipe.yaml``. Calling

.. code::
    scenario_tools exp14 input/exp14/exp14_scenario_recipe.yaml -i input/exp14/
    
will generate the simulation scenario input file named ``exp14_obflowsim_scenario_inputs.csv`` in
the ``.inputs/exp14/`` directory. Now we are ready to generate the configuration files for
each simulation scenario.

Create run settings file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is another YAML file for specifying simulation settings such as:

- run time, warmup time, and number of replications
- paths for output files along with indicators of whether or not to write out certain log files
- random number stream seeds
- locations and routes through the system (modeled as nodes and arcs consistent with NetworkX package)

Each simulation experiment has one settings file and for now we've just been manually creating them.
As more work gets done on extending the simulation architecture we may create tools to generate these files in a more automated way.

Generate simulation config file for each scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``create_configs.py`` module does two main things:

* creates a config file for each simulation scenario
* generates shell scripts for running the simulation scenarios

.. code::

    usage: create_configs [-h] [--chunk_size CHUNK_SIZE] [--update_rho_checks]
                      exp scenario_inputs_file_path sim_settings_file_path
                      configs_path run_script_path

For example,

.. code::

    create_configs exp14 \
        input/exp14/exp14_obflowsim_scenario_inputs.csv \
        input/exp14/exp14_obflowsim_settings.yaml \
        input/exp14/config run/exp14 --chunk_size 500 --update_rho_checks

Set ``--update_rho_checks`` if you manually set capacity levels in the scenario inputs file. This
will help you detect scenarios with insufficient capacity (i.e. $\rho > 1$).
                      
Generate shell scripts to run the simulation scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned in the previous step, ``create_configs.py`` creates the
shell scripts containing the commands to run the simulation scenarios. 
In order to take advantage of multiple CPUs, we can specify a 
``--chunk_size`` parameter to break up the runs into multiple
scripts - each of which can be launched separately. It's a crude form
of parallel processing.

Run the shell scripts to run the simulation scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single scenario can be run by using ``obflow_sim``.

.. code::
    usage: obflow_io [-h] stop_summaries_path output_path summary_stats_file_stem output_file_stem

    Run inpatient OB simulation

    positional arguments:
      stop_summaries_path   Folder containing the scenario rep summaries created by simulation runs
      output_path           Destination folder for combined scenario rep summary csv
      summary_stats_file_stem
                            Summary stat file name without extension
      output_file_stem      Combined summary stat file name without extension to be output

    optional arguments:
      -h, --help            show this help message and exit
    (obflowsim) mark@quercus:~/Documents/research/OBsim/mm_interpet/rerun25$ obflow_sim -h
    usage: obflow_6 [-h] [--loglevel LOGLEVEL] config

    Run inpatient OB simulation

    positional arguments:
      config               Configuration file containing input parameter arguments and values

    optional arguments:
      -h, --help           show this help message and exit
      --loglevel LOGLEVEL  Use valid values for logging package



.. code::
    obflow_sim input/exp14/config/exp14_scenario_1.yaml

The shell scripts generated in the previous step are just a sequence of such
single scenario command lines.

.. code::

    sh ./run/exp14/exp14_run.sh

 
Run ``obflow_io`` to concatenate the scenario replication files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This will create the main output summary file with one row per (scenario, replication) pair.

.. code::

    usage: obflow_io [-h] stop_summaries_path output_path summary_stats_file_stem output_file_stem

    create the main output summary file with one row per (scenario, replication) pair

    positional arguments:
      stop_summaries_path   Folder containing the scenario rep summaries created by simulation runs
      output_path           Destination folder for combined scenario rep summary csv
      summary_stats_file_stem
                            Summary stat file name without extension
      output_file_stem      Combined summary stat file name without extension to be output

    optional arguments:
      -h, --help            show this help message and exit

    
.. code::

    obflow_io output/exp14/summary_stats/ output/exp14/ summary_stats_scenario exp14_scenario_rep_simout


Run ``obflow_stat`` to create the simulation summary files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point we have statistics for each (scenario, rep) pair and need to aggregate
over the replications to get stats by scenario.

.. code::
    obflow_stat [-h] [--process_logs] [--stop_log_path STOP_LOG_PATH]
                   [--occ_stats_path OCC_STATS_PATH] [--run_time RUN_TIME]
                   [--warmup_time WARMUP_TIME] [--include_inputs]
                   [--scenario_inputs_path SCENARIO_INPUTS_PATH]
                   scenario_rep_simout_path output_path suffix

.. code::

    obflow_stat output/exp14/exp14_scenario_rep_simout.csv output/exp14 exp14 --include_inputs --scenario_inputs_path input/exp14/exp14_obflowsim_scenario_inputs.csv

Aggregates by scenario (over the replications).
Merges scenario inputs (which include the queueing approximations) with scenario simulation summary stats.

The input output summary file is ready to use in metamodeling experiments. It will
be named ``scenario_siminout_{experiment id}.csv``. Continuing our example, the output
file is ``scenario_siminout_exp14.csv``


