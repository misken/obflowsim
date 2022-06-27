SimPy
======

https://simpy.readthedocs.io/en/latest/

Not clear if still under active development. Unanswered Issue.
https://gitlab.com/team-simpy/simpy/-/issues/152

https://groups.google.com/g/python-simpy
https://stackoverflow.com/questions/tagged/simpy

Subscribe-broadcast pattern
-----------------------------

This elevator simulation model implements a subscribe-broadcast pattern
that is reminiscent of how I modeled ptube systems with Java and Simkit.

    - https://stackoverflow.com/questions/71738047/simpy-how-to-implement-a-resource-that-handles-multiple-events-at-once
    
The SimPy docs also contain a "Processor Communication" example as well as a topical guide
on "Process Interaction" which touches on various ways of communicating between
processes using events as signals that are broadcast to listeners.

    - https://simpy.readthedocs.io/en/latest/topical_guides/process_interaction.html
    - https://simpy.readthedocs.io/en/latest/examples/process_communication.html
