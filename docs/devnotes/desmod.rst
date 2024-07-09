DesMod
=======

This is a framework built on top of SimPy. On the surface, this looks
promising:

> Full-featured, high-level modeling using SimPy.
>
> The desmod package provides a variety of tools for composing, configuring, running, monitoring, and analyzing discrete event simulation (DES) models. It builds on top of the simpy simulation kernel, providing features useful for building large-scale models which are out-of-scope for simpy itself.
>
> An understanding of SimPy is required to use desmod effectively.

It has a nice test suite and two sample models that I need to 
figure out.

While reading the Issues (only a few), found reference to SimSharp,
a .NET port of SimPy. https://github.com/heal-research/SimSharp

Components
------------

These are the primary building blocks. Contain method to output to DOT file.

Configurtion is done through a flat dict with dotted keys to allow
hierarchical namespaces.

The configuration is available everywhere in the simulation via the
simulation environment instance (DesMod's, not SimPy's).

Probes and tracing
-------------------

Appears built in
