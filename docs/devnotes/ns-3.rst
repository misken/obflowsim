ns-3 notes
==========

Learn how ns-3 is architected and use ideas as appropriate

https://www.nsnam.org/about/

https://www.nsnam.org/docs/release/3.29/manual/html/index.html

ns3 is written in C++ but also exports its API to Python.

A high level view of the architecture is:

.. image:: images/ns3-software-organization.png

At the bottom is the core and then the network layer. On top of that
are things that depend on these two foundational layers such as
a module for simulating the internet or for 

Random variables
------------------

The first section of the Manual is about generating random variables.

Default behavior, using the built in rv classes, how to control
randomness across replications, how to set seeds.

Events and Simulator
--------------------

SimPy will handle this part of core functionality.

Callbacks
---------

Tricky pointer to function stuff used for decoupled modules that
need to talk to each other. Docs say this is used in the ns3 tracing
subsystem.

Object model
-------------

ns3 is written in C++ and is an OO based program. OO makes a lot of sense
for simulation.

Configuration and attributes
-----------------------------

Examples of use cases for custom attributes - https://www.nsnam.org/docs/release/3.29/manual/html/attributes.html#id1.

Logging
--------

Intended for debugging

Tracing
--------

Intending for custom detailed output for further analysis.

The modeler can scatter custom trace sources and trace sinks throughout
a model. Each source can communicate to zero or more sinks. Callbacks are
used to make this all happen. Traces also make use of Objects and Attributes.

Trace sources live inside objects. Complex pointers and callback stuff.

A trace helper facility is available to make it easier to configure tracers.

Seems complex but maybe that's because my C++ is rusty.

The Tutorial has a more friendly intro to tracing and how it differs
from logging: https://www.nsnam.org/docs/release/3.29/tutorial/html/tweaking.html#using-the-tracing-system

Data Collection Facility
-------------------------

The DCF appears to leverage the tracing facilities.

.. image:: images/dcf-overview.png

.. image:: images/dcf-overview-with-aggregation.png

There are some DCF Helper functions to make it easy to do some
simple and common DCF tasks such as making gnuplots or writing to
a file. These allow user to avoid writing a bunch of detailed trace
config code. Cost is less flexibility.

Probes
^^^^^^^

> The Probe can be thought of as kind of a filter on trace sources.

The docs give more info on why you might want to use a Probe instead
of directly connecting to a trace source.

Collectors
^^^^^^^^^^^

Some sort of intermediate data collection abstraction that sits between
probes and aggregators. Not implemented yet.

Aggregators
^^^^^^^^^^^^

End point for data collected with probes for aggregating data.

Adaptors
^^^^^^^^^

Helps you connect the other DCF objects


Statistical Framework
----------------------

See https://www.nsnam.org/docs/release/3.29/manual/html/statistics.html

ns-3 Tutorial
==============

https://www.nsnam.org/docs/release/3.29/tutorial/html/introduction.html






