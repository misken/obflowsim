Which simulation package?
==========================

Before going any further with the current SimPy based model, I really need to decide if we are sticking with SimPy or or going with something else:

- DESpy - port of SimKit and uses event graphs
- desmod - layer on top of SimPy
- salabim - see kalabim for Kotlin version for arch ideas
- nxsim - not sure if still active
- other - has anything new come along in the last two years?

SimPy
-----

According to https://gitlab.com/team-simpy/simpy/-/issues/152, it appears that the maintainer will try to make sure that SimPy remains compatible with future Python releases (including type annotations) but that new features will have to be added by others and that there would be a high bar for merging them - (1) not complicating core SimPy; and (2) not regressing simulation performance.

So, if the core engine provided by SimPy is sufficient for obflowsim, then I could use it knowing I'd have to build supporting tools around it (e.g. output analysis, animation, logging, ...). 

Obviously, I have a decent amount of experience with SimPy and there is a decent amount of activity in the Google Group and on Stack Overflow.

The Grotto Networking page - https://www.grotto-networking.com/DiscreteEventPython.html, has some nice examples of creating higher level abstractions on top of SimPy to make modeling easier.

- https://gitlab.com/team-simpy/simpy
- https://simpy.readthedocs.io/en/latest/
= https://www.grotto-networking.com/DiscreteEventPython.html
- https://bitsofanalytics.org/posts/simpy-getting-started-patflow-model/simpy-getting-started
- https://bitsofanalytics.org/posts/simpy-oo-patflow-model/simpy-oo-patflow-model
- https://bitsofanalytics.org/posts/simpy-vaccine-clinic-part1/simpy_getting_started_vaccine_clinic
- https://bitsofanalytics.org/posts/simpy-vaccine-clinic-part2/simpy_vaccine_clinic_improvements

desmod
-------

This package relies on SimPy under the hood for the simulation kernel. It's goal is to make it easier to create large scale simulation models by introducing a component architecture with connection objects between components. It also adds some simulation niceties such as configuration management, logging, monitoring and probes. 

The main developer is also the current maintainer of SimPy.

DESpy
------

Arnie Buss's port of SimKit from Java to Python. See my email exchange with him.

- https://pypi.org/project/DESpy/
- https://github.com/ahbuss/DESpy

Py-DES
------

This is a new Python based DES package.

- https://github.com/vitostamatti/pydes
- https://pydes.readthedocs.io/en/latest/
- https://medium.com/@vitostamatti1995/discrete-event-simulation-in-python-8baf9694948f

DE-Sim
------

This is another new (2020) based package. There is a paper in JOSS. However, there have been no commits since 2020. The developers seem to be from the medical community and have used this for "whole cell simulation". They position this package as being better integrated with the Python data science world (?) and as being the only OO DES package in Python (?).

- https://github.com/KarrLab/de_sim
- https://joss.theoj.org/papers/10.21105/joss.02685
- https://docs.karrlab.org/de_sim/master/1.0.5/source/de_sim.html

Simulus
--------

This is another new (2019) based package. No commits in 5 years.

- https://github.com/liuxfiu/simulus
- https://simulus.readthedocs.io/en/latest/index.html



What to do?
-----------

I think best approach is to start with a pure SimPy model and then consider whether using some of the things offered by desmod makes sense. For example, I've seen Issues in which people discuss using just the ``Queue`` and ``Pool`` objects from desmod (i.e. not using the components. This will also force me to become intimately familiar with SimPy, which is necessary for understanding and using desmod (per the developer).

While I really like the event-graph approach, I've got a lot invested in the current SimPy model and SimPy does appear to be the most widely used Python based DES package.



Packaging
=========

Software Project Mgt
====================

Semantic versioning seems like a good idea - https://semver.org/

Documentation
==============

I think I'll use Sphinx for documentation even though I used Jupyter Book for hillmaker. Back in the day I used Sphinx for the ptube model docs and that worked great. 

There is a Sphinx extension for testing code snippets in documentation:

- https://sphinx-rtd-trial.readthedocs.io/en/1.0.8/ext/doctest.html
- https://www.b-list.org/weblog/2023/dec/10/python-doctest/


Useful links
============


