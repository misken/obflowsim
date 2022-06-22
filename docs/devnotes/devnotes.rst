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

Patient types
----------------

Patient Type and Patient Flow Definitions

* Type 1: random arrival spont labor, regular delivery, route = 1-2-4
* Type 2: random arrival spont labor, C-section delivery, route = 1-3-2-4
* Type 3: random arrival augmented labor, regular delivery, route = 1-2-4
* Type 4: random arrival augmented labor, C-section delivery, route = 1-3-2-4
* Type 5: sched arrival induced labor, regular delivery, route = 1-2-4
* Type 6: sched arrival induced labor, C-section delivery, route = 1-3-2-4
* Type 7: sched arrival, C-section delivery, route = 1-3-2-4

* Type 8: urgent induced arrival, regular delivery, route = 1-2-4
* Type 9: urgent induced arrival, C-section delivery, route = 1-3-2-4

* Type 10: random arrival, non-delivered LD, route = 1
* Type 11: random arrival, non-delivered PP route = 4


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


Router design
--------------

Where to do LOS assignment?
    - happening in create_route

Should we assign entire route at time of patient creation?



Blocking
---------

Need way to specify if and how any blocking LOS adjustments should be done


Occupancy tracking
-------------------

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




Staffing resources
-------------------

No staffing within model. Post-process occupancy log or stop log.

This post by jprayson describes a grocery store staffing approach:
https://groups.google.com/g/python-simpy/c/m6ogUwIWtMU

Software Project Mgt
====================

Semantic versioning seems like a good idea - https://semver.org/

Useful links
============

Docs
https://simpy.readthedocs.io/en/latest/index.html

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
