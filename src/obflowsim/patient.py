from __future__ import annotations

from obflowsim.obconstants import ArrivalType, UnitName, MARKED_PATIENT, PatientType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obflowsim.patient_flow_system import PatientFlowSystem


class Patient:
    """
    These are the *entities* who flow through a *patient flow system* (``PatientFlowSystem``)
    consisting of a network of *patient care units* (``PatientCareUnit``).

    """

    def __init__(self,
                 patient_id: str,
                 arrival_type: ArrivalType,
                 arr_time: float,
                 patient_flow_system: PatientFlowSystem,
                 entry_delay: float = 0):
        """

        Parameters
        ----------
        patient_id : str
            Unique identifier for each patient instance
        arrival_type : ArrivalType
            Unique identifier for arrival streams
        arr_time : float
            Simulation time at which patient arrives to system
        patient_flow_system : PatientFlowSystem
            System being modeled and to which new patient arrives
        entry_delay : float
            Length of time to hold patient in ENTRY node before routing to first location
        """
        self.patient_id = patient_id
        self.arrival_type = arrival_type
        self.system_arrival_ts = arr_time + entry_delay
        self.entry_delay = entry_delay
        self.pfs = patient_flow_system

        # Determine patient type
        self.patient_type = self.assign_patient_type()

        # Initialize unit stop attributes
        self.current_stop_num = -1

        # Get route
        self.route_graph = self.pfs.router.create_route(self)

        # Initialize data structures for holding flow related quantities
        self.bed_requests = {}  # simpy request() events with unit name as keys
        self.unit_stops = []  # unit name
        self.planned_los = []
        self.adjusted_los = []
        self.request_entry_ts = []
        self.entry_ts = []
        self.wait_to_enter = []
        self.request_exit_ts = []
        self.exit_ts = []
        self.blocked = []
        self.wait_to_exit = []
        self.skipped_edge = []  # Destination node is the skipped unit

        # Initiate process of patient entering system
        self.pfs.env.process(
            self.pfs.entry.put(self, self.pfs))

    def get_route_length_nodes(self):
        """
        Get the number of nodes in the current route including entry and exit nodes.

        Returns
        -------
        int: number of nodes
        """
        return len(self.route_graph.edges) + 1

    def get_previous_unit_name(self):

        # If this is first stop, previous unit was Entry node
        if self.current_stop_num == 1:
            previous_unit_name = UnitName.ENTRY.value
        else:  # Not the first stop
            previous_unit_name = self.unit_stops[self.current_stop_num - 1]

        return previous_unit_name

    def get_previous_unit(self):

        # If this is first stop, previous unit was Entry node
        if self.current_stop_num == 1:
            previous_unit = self.pfs.entry
        else:  # Not the first stop
            previous_unit_name = self.get_previous_unit_name()
            previous_unit = self.pfs.patient_care_units[previous_unit_name]

        return previous_unit

    def get_current_unit_name(self):

        if self.current_stop_num == 0:
            current_unit_name = UnitName.ENTRY.value
        else:  # Not the first stop
            current_unit_name = self.unit_stops[self.current_stop_num]

        return current_unit_name

    def get_current_unit(self):

        if self.current_stop_num == 0:
            current_unit = self.pfs.entry
        else:  # Not the first stop
            current_unit_name = self.get_current_unit_name()
            current_unit = self.pfs.patient_care_units[current_unit_name]

        return current_unit

    def get_current_route_edge(self):
        """
        Get edge containing current unit as destination node.

        Returns
        -------
        `edge`

        """

        if self.skipped_edge[self.current_stop_num - 1] is None:
            previous_unit_name = self.get_previous_unit_name()
        else:
            previous_unit_name = self.skipped_edge[self.current_stop_num - 1][1]

        current_unit_name = self.get_current_unit_name()
        try:
            current_route_edge = self.route_graph.edges[previous_unit_name, current_unit_name]
        except KeyError:
            print(f'patient {self.patient_id} has no arc from {previous_unit_name} to {current_unit_name}')
            current_route_edge = None

        return current_route_edge

    def assign_patient_type(self):
        arr_stream_rg = self.pfs.config.rg['arrivals']
        if self.arrival_type == ArrivalType.SPONT_LABOR.value:
            # Determine if labor augmented or not
            if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_spont_labor_aug']:
                # Augmented labor
                if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_aug_labor_to_c']:
                    return PatientType.RAND_AUG_CSECT.value
                else:
                    return PatientType.RAND_AUG_NAT.value
            else:
                # Labor not augmented
                if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_spont_labor_to_c']:
                    return PatientType.RAND_SPONT_CSECT.value
                else:
                    return PatientType.RAND_SPONT_NAT.value
        elif self.arrival_type == ArrivalType.NON_DELIVERY_LDR.value:
            return PatientType.RAND_NONDELIV_LDR.value
        elif self.arrival_type == ArrivalType.NON_DELIVERY_PP.value:
            return PatientType.RAND_NONDELIV_PP.value
        elif self.arrival_type == ArrivalType.URGENT_INDUCED_LABOR.value:
            if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_urg_ind_to_c']:
                return PatientType.URGENT_IND_CSECT.value
            else:
                return PatientType.URGENT_IND_NAT.value
        elif self.arrival_type == ArrivalType.SCHED_CSECT.value:
            # Scheduled c-section
            return PatientType.SCHED_CSECT.value
        else:
            # Determine if scheduled induction ends up with c-section
            if arr_stream_rg.random() < self.pfs.config.branching_probabilities['pct_sched_ind_to_c']:
                return PatientType.SCHED_IND_CSECT.value
            else:
                return PatientType.SCHED_IND_NAT.value

    def append_empty_unit_stop(self):
        """
        Append None placeholders to patient flow lists when current_stop_num is incremented

        Returns
        -------

        """
        self.unit_stops.append(None)
        self.planned_los.append(None)
        self.adjusted_los.append(None)
        self.request_entry_ts.append(None)
        self.entry_ts.append(None)
        self.wait_to_enter.append(None)
        self.request_exit_ts.append(None)
        self.exit_ts.append(None)
        self.blocked.append(None)
        self.wait_to_exit.append(None)
        self.skipped_edge.append(None)

    def __repr__(self):
        return "patient id: {}, patient_type: {}, time: {}". \
            format(self.patient_id, self.patient_type, self.system_arrival_ts)
