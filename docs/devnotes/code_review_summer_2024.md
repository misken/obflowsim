# Initial code review

I'm picking this project back up after putting it down for almost two years. I need to assess
the current state of the code and figure out how to move forward.

## Build in constants

Defined constants for:

- patient types
- arrival streams
- units

## The config file

The `Config` class takes a config file as input to create an object instance.

From the `desmod` documentation:

> It is common for models to have configurable paramaters. Desmod provides an opinionated mechanism for simulation 
> configuration. 
> A single, comprehensive configuration dictionary captures all configuration for the simulation. 
> The configuration dictionary is 
> propogated to all Components via the SimEnvironment. 
> 
> The various components (or component hierarchies) may maintain separate configuration namespaces within the 
> configuration dictionary by use of keys conforming to the dot-separated naming convention. 
> For example, “mymodel.compA.cfgitem”.
>
> The desmod.config module provides various functionality useful for managing configuration dictionaries.
> 

Here's the config file from the gas_station.py example:

```commandline
# Desmod uses a plain dictionary to represent the simulation configuration.
# The various 'sim.xxx' keys are reserved for desmod while the remainder are
# application-specific.
config = {
    'car.capacity': 50,
    'car.level': [5, 25],
    'gas_station.capacity': 200,
    'gas_station.count': 3,
    'gas_station.pump_rate': 2,
    'gas_station.pumps': 2,
    'gas_station.arrival_interval': 60,
    'sim.dot.enable': True,
    'sim.dot.colorscheme': 'blues5',
    'sim.duration': '500 s',
    'sim.log.enable': True,
    'sim.log.file': 'sim.log',
    'sim.log.format': '{level:7} {ts:.3f} {ts_unit}: {scope:<16}:',
    'sim.log.level': 'INFO',
    'sim.result.file': 'results.yaml',
    'sim.seed': 42,
    'sim.timescale': 's',
    'sim.workspace': 'workspace',
    'tanker.capacity': 200,
    'tanker.count': 2,
    'tanker.pump_rate': 10,
    'tanker.travel_time': 100,
}
```

You can see that desmod uses a flat dictionary with dotted naming convention to act as a way to add structure to the
config file. The desmod package also has a Config class with various functions for organizing config file sections.

I think I'm going to stick with my yaml based config files with hierarchy explicitly used in the yaml structure.

## Main simulation module - obflow_sim.py

This module contains most of the objects making up the simulation model.

### ENTRY and EXIT nodes

Currently, ENTRY and EXIT are just instances of `PatientCareUnit`, even though they really
aren't patient care units. This means they show up in routes and locations section of config.
When I started implementing scheduled arrivals and needing to have patients delay in ENTRY until
their scheduled procedure time, I started to think it might make more sense to have separate
dedicated ENTRY and EXIT object types.

- avoids conditional logic based on checking if in ENTRY or EXIT
- only actual patient care units will get modeled with `PatientCareUnit`.
- user won't have to specify fake large capacity values for ENTRY and EXIT
- user won't have to include ENTRY and EXIT in routes.

Created new branch called 'egress_nodes' to test this idea. The commit right before
this branch created contains non-working code for scheduled arrivals using old
approach of ENTRY and EXIT as `PatientCareUnit` objects.

#### Commit 2d78c85 implements egress nodes

Added `EntryNode` and `ExitNode` as part of `PatientFlowSystem`. These nodes handle entry and exit logic.

- removed ENTRY and EXIT from Locations section of config
- kept both ENTRY and EXIT in routing section and these will be required. Otherwise, things like 
routes with a single stop get awkward to enter. Now every route has at least three nodes and two edges.

### Scheduled arrivals

Currently, the input file for scheduled arrivals has one column per hour and one row per day of week. The values
represent the number of cases scheduled at that time. We need to make this more flexible.

#### Option 1: Each row is explicit time

This approach only stores the non-zero times that cases are scheduled as is assumed to repeat each week.
Time units are in same as `base_time_unit` specified in config - usually hours.
```
# simtime, num scheduled
6.0, 1
7.5, 1
10.0, 2
```

This is a little tricky in terms of readability because we don't know the day of week for "day 1" and values
after one day are tough to mentally translate. 

#### Option 2: Each row is explicit DOW and time

This approach only stores both the DOW and the non-zero times that cases are scheduled as is assumed to repeat each week.
Time units are in same as `base_time_unit` specified in config - usually hours.

```
# DOW, TOD, num scheduled
Mon, 6.0, 1
Mon, 7.5, 1
Tue, 10.0, 2
...
```