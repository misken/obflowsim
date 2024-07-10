# obflowsim - a flexible obstetric patient flow simulation model in Python

**IMPORTANT** The OB patient flow simulation metamodeling research project formerly residing in this repo has
moved to its new home at [obflowsim-mm](https://github.com/misken/obflowsim-mm).

## Overview

This project is intended to provide a way to simulate OB patient flow
systems in Python. The model is based on the [SimPy](https://simpy.readthedocs.io/en/latest/) 
package and is designed to be flexible and data driven. The first version
of this model was developed for a simulation metamodeling research project 
(CITATION) and we have moved that repo to 
[obflowsim-mm](https://github.com/misken/obflowsim-mm). This obflowsim
repo is the home for continued development of the core discrete event
simulation (DES) model. Active development is happening on the `develop`
branch (and sub-branches off of `develop`) whereas the `main` branch contains the original simplified 
model used in the research project.

The next generation of the simulation model will include a number of
features needed for simulation of real OB systems for capacity planning,
scheduling, and process analysis projects. Some of these features
include:

* multiple patient arrival streams including spontaneous labor, scheduled c-sections and scheduled inductions,
* multiple patient types based on arrival stream, whether or not labor is augmented, and whether or not a c-section is performed,
* ability to model standard unit configurations such as LDR+PP and LDRP,
* detailed statistics regarding patient flow and unit occupancy (including time of day and day of week dependencies),
* ability to model impacts of patients blocked due to insufficient downstream capacity,
* YAML configuration files for defining simulation scenarios and experimental settings (e.g. run length, warmup time, number of replications).

Our goal is to provide a high quality, open source, fully transparent DES model that both practitioners and other researchers can use and build upon. We
are motivated by our decades of experience in using DES in actual healthcare operational and capacity planning projects using a 
variety of commercial DES products. While such products have many advantages, they make it difficult to share models
and for the healthcare modeling community to assess and improve on modeling approaches for important healthcare systems. This project
is a very modest first step and there is much work to do to achieve this vision.

This pre-release version has the following capabilities:

- random patient arrivals via stationary Poisson arrival process
- scheduled patient arrivals via fixed, repeating, one week schedules
- YAML config file for specifying all simulation inputs
- LOS distributions from a select number of numpy random functions
- serial routes
- summary statistics by patient and by unit
- detailed log files for patient stops and for unit occupancy changes

A few key features that have **NOT** yet been implemented include:

- blocking related LOS adjustments
- realistic discharge timing for non-stationary analysis

