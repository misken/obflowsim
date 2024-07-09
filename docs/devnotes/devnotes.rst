Which simulation package?
==========================

Before going any further with the current SimPy based model, I really need to decide if we are sticking with SimPy or or going with something else:

- DESpy - port of SimKit and uses event graphs
- desmod - layer on top of SimPy
- salabim - see kalabim for Kotlin version for arch ideas
- nxsim - not sure if still active
- other - has anything new come along in the last two years?



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


