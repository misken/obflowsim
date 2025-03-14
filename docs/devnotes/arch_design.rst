Design overview
==================

Model should be able to handle:

* static routing
* Standard 11 patient types
* Standard 3 configurations: LDR, LDRP, traditional
* blocking
* stationary and time-dependent occupancy stats
* random arrivals, scheduled arrivals, urgent arrivals

Location resources
-------------------

Primarily designed for:

* observation
* labor and delivery 
* recovery
* CSection procedure room
* LDR
* LDRP
* PP

Typical system configurations are:

* obs ---> LDR --> PP
* obs ---> LDRP 
* obs ---> LD --> R --> PP

If there is an unscheduled C-section, there would be a visit to a
C-section procedure room before recovery or PP.

But, we want to be able to handle more complicated paths. These often
occur when an unscheduled C-section takes place in an LDRP setting. If the
facility has some designated PP beds to be used by patients after an
unscheduled C-section then the route might look like this:

* obs ---> LDRP --> C-section --> PP

But, if there are no PP beds available, patient might return to LDRP. Or, they
might overflow to another non-LDRP, but not ideal unit for their PP stay. Or, they
might stay in C-section recovery until a bed is available.

Patient types and routing
--------------------------

Patient Type and Patient Flow Definitions

* Type 1: random arrival spont labor, regular delivery, 
* Type 2: random arrival spont labor, C-section delivery, 
* Type 3: random arrival augmented labor, regular delivery, 
* Type 4: random arrival augmented labor, C-section delivery, 

Starting to think that labor augmentation shouldn't define a separate patient type.
Augmentation is indicated when labor is prolonged. Yes, we could model this as
a separate patient type and adjust the LOS distribution accordingly. Or, we
could model this is as an intervention that happens at a specific time and
with a specific probability. Coming up with a way to model it this way opens
the door for a pattern or method for modeling general interventions that
may or may not occur. I guess even C-sections could be treated this away.
Maybe it's not so much as redefining patient types as it is in how
these types are modeled. Predetermining them via the various branching probabilities
and LOS distributions makes the modeling easier but we should also think
about the data gathering and distribution fitting implications.

Upon further thought, I think our original approach is easiest from a modeling point of view
and really no difference from a data gathering point of view. The modeling is much easier since
we don't have to do things like figure out when to decide to augment or not. It's all just
baked into the LOS. But, to be able to do things like model a stay in an LDRP with a combination
of a stay for the labor and delivery part and then a stay for PP (using the "days" approach) we
need to extend the ``los`` parameter so that we can do summations of allowed distributions. To
do this, we need to NOT use ``eval()`` and do it "correctly". Looking into ASTs. One way or
another, it's doable and we can come back to this after making other modeling changes.



* Type 5: sched arrival induced labor, regular delivery, 
* Type 6: sched arrival induced labor, C-section delivery, 
* Type 7: sched arrival, C-section delivery, 

* Type 8: urgent induced arrival, regular delivery, 
* Type 9: urgent induced arrival, C-section delivery, 

* Type 10: random arrival, non-delivered LD, 
* Type 11: random arrival, non-delivered PP 

The route that patients take will depend on the configuration of units used
by the facility. In general, routes are deterministic except for perhaps
an LDRP configuration that has some PP beds for post C-section patients. In that
case lack of availability of a PP bed might necessitate a return to an LDRP (?) or
placement on some other "overflow" unit. Currently we have no way to model the
placement on an alternate unit when insufficient capacity exists on primary 
unit - **we need to have this general capability to handle more general patient
flow modeling**.

Router design
--------------

How to handle alternate destination units? Probabalistic branching?

- while neither of the above are super common in inpatient OB flow,
they are very common in other more general patient flow settings. 

Where to do LOS assignment?
    - happening in create_route
    - LOS distribution is specified on the arcs within the route

Should we assign entire route at time of patient creation?

Is the edge_num attribute on arcs needed? How are we using it?

Should we have a network that describes the possible moves between
patient care units and a separate thing that actually constructs
routes, either static or dynamic? This is how I modeled the ptube system.

We could specify a physical flow network that defines arcs between
nodes of patient care units. The attributes of those arcs could be
the default behaviors with respect to getting and releasing beds, los,
or blocking adjustments. Then, for each patient type, we create a 
data structure for specifying a route which could include patient
type specific attribute values that override the defaults. This
data structure would have to be general enough to handle the 
different types of routing we might want to do: static deterministic,
alternate paths, probabalistic, other.



Blocking
---------

Need way to specify if and how any blocking LOS adjustments should be done.

TJW - LOS in LDR should be adjusted by time blocked in triage. However, once baby is born, time blocked in LDR waiting for PP is largely irrelevant.


Arrival streams
-----------------

Random arrivals
^^^^^^^^^^^^^^^^

Poisson arrivals - stationary and non-stationary versions

For stationary Poisson arrivals, implemented a ``OBPatientGeneratorPoisson``
class that samples from an exponential distribution at the specified
rate to generate interarrival times. Upon each "arrival" a new ``OBPatient``
object gets created.


Urgent arrivals
^^^^^^^^^^^^^^^^

These are for the urgent inductions

Poisson arrivals - stationary and non-stationary versions


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

Per conversation with TJW (2022-06-21):
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





Input config file design
---------------------------

Use YAML.

Can create separate YAML files for different parts of the config file
and then just concatenate them all together.

How to specify LOS distributions?
How to specify routes by patient type?

Network representation
------------------------

@InProceedings{SciPyProceedings_11,
  author =       {Aric A. Hagberg and Daniel A. Schult and Pieter J. Swart},
  title =        {Exploring Network Structure, Dynamics, and Function using NetworkX},
  booktitle =   {Proceedings of the 7th Python in Science Conference},
  pages =     {11 - 15},
  address = {Pasadena, CA USA},
  year =      {2008},
  editor =    {Ga\"el Varoquaux and Travis Vaught and Jarrod Millman},
}


Length of stay modeling
-------------------------

Currently just doing standard real number los generation. If we are
interested in TOD stats, need to implement TOD adjustment to model
discharge timing.

TJW - The best way to model PP LOS, if
 you ask me, is this:  -  patient arrives on PP
 whenever they get there on the first PP day  -
  LOS in days is best modeled by a distribution of 1, 2 or 3
 days for vaginal or 2, 3, or 4 days for csec.
    -  on the discharged day, the discharge
 time is selected from the appropriate LOS distribution for
 time of day.  It's the same no matter how many days the
 patient was on the PP unit.  So, interestingly
 enough, arriving early in the AM at PP actually increases
 LOS.  Arriving late, say 1800 or so, decreases overall PP
 LOS.  If you want to shorten PP LOS, the best thing to do
 is schedule procedures later in the afternoon and move the
 patient to PP between 1800 and 2100 in the evening.
 
TJW - Labor LOS
is very dependent on labor type  
-  spontaneous labor, vaginal birth/csec  
-  augmented labor, vaginal birth/csec  
-  induced labor, vaginal birth/csec    
-  -  should be a variable 10, 20, 30 and 40% of total birth vol    
-  - the non-induced patient volume should be split evenly between spontaneous and augmented labor    -
-  can have different probabilities for vag birth vs csec delivery for the three labor types above.

Of course, scheduled csec patients do not spend any time in
labor.  Rather, these patients go straight to the pre-op
area.
 
TJW - 
have a few very descriptive LOS distributions for each of
the patient types noted in c aboveFor PP, there
are only two patient types, vag birth and csec delivery.
PP LOS is independent on anything that happened in labor
except how did the baby come out.
 
TJW - Mark, the big problem these days is the
exploding induction rates.  Induced patients have more than
double the LDR LOS in labor.  This is primarily one-on-one
nursing and LDR room consuming for an additional 11 or 12
hours, on average.  So, induced patients consume LDR rooms
and csec patients consume PP rooms.  Spontaneous labor,
un-augmented labor, vaginal birth patients - the natural way
- is far more efficient and frugal regarding resource
consumption.  Induced labor that results in a csec is the
most expensive patient type on the planet.

Dist fitting in Python - https://fitter.readthedocs.io/en/latest/





Occupancy tracking
-------------------

.. admonition:: And, by the way...

   You can make up your own admonition too.

Should we track occ history or just post-process a stop log with hillmaker?


Logging and tracing
--------------------

How best to do trace messages? Is this same use case as "logging"?

In ns-3:

No, tracing is for simulation output and logging for debugging, warnings and errors.

https://www.nsnam.org/docs/release/3.29/manual/html/tracing.html
https://www.nsnam.org/docs/release/3.29/manual/html/data-collection.html

Developing a good tracing system is very important for subsequent
analysis of output and potential animation.

SimPy docs have some tracing examples that require monkey patching


https://docs.python.org/3/library/logging.html

https://bitbucket.org/snippets/benhowes/MKLXy/simpy30-fridge

https://guicommits.com/how-to-log-in-python-like-a-pro/

Strong opinions on how to do logging - https://www.palkeo.com/en/blog/python-logging.html
The ``extra=<dict>`` param lets you add contextual info to log message.

Loguru - builds on top of standing logging module - https://github.com/Delgan/loguru
    - uses notion of sinks which seem to be used in simulation tracing
    - adds a TRACE level
    
structlog is another option for structured logging (dicts instead of just string messages)
    
https://opentelemetry.io/docs/instrumentation/python/

Statistical reporting
-----------------------


Staffing resources
-------------------

No staffing within model. Post-process occupancy log or stop log.

This post by jprayson describes a grocery store staffing approach:
https://groups.google.com/g/python-simpy/c/m6ogUwIWtMU
Hmm, this might be the maintainer of SimPy and desmod and he has the grocery store model in the desmod examples section of docs.

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

CLI
^^^


API
^^^

Useful links
============

Network models
https://www.grotto-networking.com/DiscreteEventPython.html#Intro

One approach to custom Resource
http://simpy.readthedocs.io/en/latest/examples/latency.html


DesMod = New DES package that builds on SimPy
http://desmod.readthedocs.io/en/latest/

Not sure how active. I think I should start with just SimPy to
decide for myself on the metalevel needs in terms of model building,
logging, config files, CLI, etc.

Tidygraph - maybe for representing flow networks visually?
http://www.data-imaginist.com/2017/Introducing-tidygraph/

Vehicle traffic simulation with SUMO
http://www.sumo.dlr.de/userdoc/Sumo_at_a_Glance.html
http://sumo.dlr.de/wiki/Tutorials
