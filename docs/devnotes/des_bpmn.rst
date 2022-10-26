

Nice series of articles on DES and BPMN
----------------------------------------

https://www.linkedin.com/pulse/7-key-insights-bp-modeling-simulation-gerd-wagner/

https://www.linkedin.com/pulse/des-tools-lack-scientific-foundation-gerd-wagner/

https://www.linkedin.com/pulse/business-processes-petri-nets-event-graphs-gerd-wagner/

https://www.linkedin.com/pulse/event-graphs-fundamental-des-bpm-gerd-wagner/

https://sim4edu.com/reading/bpms-dpmn/bpms-dpmn

https://sim4edu.com/reading/des-engineering/ - open access book

Wagner's Winter Sim paper on generalization to event graphs

https://www.informs-sim.org/wsc17papers/includes/files/056.pdf

Wagner's OEM&S approach
-----------------------

Combine event graphs with object models. The rationale is that objects
are natural constructs for simulation modeling and event graphs are
very general and intimately linked with event scheduling that underlies
DES.


Entities and resources are "roles", not fundamentally different constructs:
https://sim4edu.com/reading/des-engineering/resourceconstrainedactivities-intro

It's a well-known fact that in the real world people may switch roles and may play several roles at the same time, but many modeling approaches/platforms fail to admit this. For instance, the simulation language (SIMAN) of the well-known DES modeling tool Arena does not treat resources and processing objects ("entities") as roles, but as strictly separate categories. This language design decision was a meta-modeling mistake, as admitted by Denis Pegden, the main creator of SIMAN/Arena, in (Drogoul et al 2018) where he says "it was a conceptualization mistake to view Entities and Resources as different constructs". 
