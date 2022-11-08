Architecture and detailed design of obflowsim
==============================================

Overview
---------

Patients
--------------

These are the *entities* who flow through a *patient flow system*
consisting of a network of *patient care units*.

Patient care units
-------------------

Patient flow system
--------------------

Simulation calendar
--------------------


Generating patient arrivals
----------------------------



Random arrivals


The ``PatientPoissonArrivals`` class generates ``Patient`` objects
according to a stationary poisson process with a specified
rate. In addition to the mean arrival rate, the arrival generator
is initialized with a unique arrival stream identifier (``str``), and
a numpy random number generator (``numpy.random.default_rng``) whose
seed is specified in the simulation scenario config file. There
are two ways to control the stopping of patient generation.

- by time via setting ``stop_time`` (default is ``simpy.core.Infinity``)
- by number of arrivals via setting ``max_arrivals`` (default is ``simpy.core.Infinity``)


Scheduled arrivals
^^^^^^^^^^^^^^^^^^^

Process flow and routing
-------------------------

Length of stay
^^^^^^^^^^^^^^^


Routing
^^^^^^^^

System object
--------------

Simulation driver
------------------

Interfaces
-----------

CLI
^^^


API
^^^
