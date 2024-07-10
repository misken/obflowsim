#############################################
Architecture and detailed design of obflowsim
#############################################

************************************
High level architecture of obflowsim
************************************

************************************
Detailed design of obflowsim
************************************

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

Arrival streams
-----------------

Random arrivals
^^^^^^^^^^^^^^^^

Poisson arrivals - stationary and non-stationary versions

Urgent arrivals
^^^^^^^^^^^^^^^^

These are for the urgent inductions

Poisson arrivals - stationary and non-stationary versions

For stationary Poisson arrivals, implemented a ``OBPatientGeneratorPoisson``
class that samples from an exponential distribution at the specified
rate to generate interarrival times. Upon each "arrival" a new ``OBPatient``
object gets created.

Scheduled arrivals
^^^^^^^^^^^^^^^^^^^

Scheduled inductions and scheduled c-sections

For this, we need the notion of a calendar. 

    - include a ``start_date`` parameter in config file
    - can do datetime math to convert ``simpy.env.now`` to calendar datetime.
    
For the schedule itself, need way of filling weekly scheduling template at
some user specified density level (or other approach) and then generating patients each week
that wait in ENTRY until their scheduled procedure time to show up to the first
unit location after ENTRY.

For now, I've implemented a static one week scheduling template for
C-sections and another for inductions. 

.. topic:: Scheduled procedures

   In general:
   
   - scheduled C-sections happen no further than ~6 weeks out. Once scheduled,
    it's a high likelihood that it will occur on scheduled date.
    - scheduled inductions happen no further than 2-3 weeks out. More 
    uncertainty as to whether or not the induction will happen since
    mom could go into labor prior to scheduled induction date.
   - when modeling schedule filling dynamics, want the ability to model
    different scheduling practices. For example, a simple approach would be
    to open up entire template and let people schedule procedures anywhere
    in the template with capacity. In order to encourage occupancy smoothing,
    a better approach may be to open additional capacity in phases as you
    schedule becomes fuller.
    
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

Different entity types with different processing times and graphics
-------------------------------------------------------------------
Since multiple patient types 
will visit the same Server objects (e.g. post-partum unit) and will have different LOS distributions, we need to 
have a general approach to managing different parameters for different patient types. In Simio, the easiest way to 
do this is through a Data Table (Chapter 7). Tables can contain any number of columns and the allowable data types includes a 
wide variety of Standard Properties, Element References or Object References. Once the table is created, it can be 
referenced in a variety of ways (p219) in the model. Row selection from tables can be done randomly based 
on user specified probabilities or some rule. Often each entity will simply be referencing a specific row every time. 
Simio provides an easy way to implement this by setting a Table Reference Assignment in the Source object.


Routing
^^^^^^^^

System object
--------------

Simulation driver
------------------

Interfaces
-----------





