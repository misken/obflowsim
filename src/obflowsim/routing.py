import logging
from abc import ABC, abstractmethod
from copy import deepcopy

from typing import TYPE_CHECKING

import simpy
from simpy.events import AnyOf
import networkx as nx
from networkx import DiGraph

from obflowsim.obconstants import DEFAULT_GET_BED, DEFAULT_RELEASE_BED, ATT_RELEASE_BED, ATT_GET_BED
from obflowsim.obconstants import SRC, DEST, DATA
from obflowsim.los import create_los_partial, los_mean
from obflowsim.patient import Patient




class Router(ABC):

    @abstractmethod
    def get_next_step(self, entity):
        pass





class OBRouter(Router):
    def __init__(self, env, pfs):
        """
        Routes patients having mostly fixed serial routes though might have some capacity modulated alternate
        paths.

        Parameters
        ----------
        env: Environment
        pfs: PatientFlowSystem - need access to patient flow network
        """

        self.env = env
        self.patient_flow_system = pfs

        # Dict of networkx DiGraph objects
        self.route_graphs: dict[str, DiGraph] = {}

        los_params = pfs.config.los_params

        # Create route templates from routes list
        for route_name, route in self.patient_flow_system.config.routes.items():
            route_graph = nx.DiGraph()

            # Add edges - simple serial route in this case
            for edge in route['edges']:
                # Find the edge in the pfs network
                network_edge = [(u, v, d) for u, v, d in pfs.network.edges(data=True) if d['id'] == edge['id']].pop()
                source = network_edge[SRC]
                dest = network_edge[DEST]

                route_graph.add_edge(source, dest)

                nx.set_edge_attributes(route_graph, {
                    (source, dest): {'id': edge['id']}})

                if 'next' in edge:
                    nx.set_edge_attributes(route_graph, {
                        (source, dest): {'next': edge['next']}})
                else:
                    nx.set_edge_attributes(route_graph, {
                        (source, dest): {'next': None}})

                if 'los' in edge:
                    edge_los_mean = los_mean(edge['los'], los_params)
                    los = edge['los']
                else:
                    edge_los_mean = 0.0
                    los = '0.0'

                if 'discharge_adjustment' in edge:
                    nx.set_edge_attributes(route_graph, {
                        (source, dest): {'discharge_adjustment': edge['discharge_adjustment']}})
                else:
                    nx.set_edge_attributes(route_graph, {
                        (source, dest): {'discharge_adjustment': None}})

                nx.set_edge_attributes(route_graph, {
                    (source, dest): {'los': los}})
                nx.set_edge_attributes(route_graph, {
                    (source, dest): {'los_mean': edge_los_mean}})

                # Add get and keep bed attributes
                if ATT_GET_BED in edge:
                    att_get_bed = edge[ATT_GET_BED]
                else:
                    att_get_bed = DEFAULT_GET_BED

                nx.set_edge_attributes(route_graph, {
                    (source, dest): {ATT_GET_BED: att_get_bed}})

                if ATT_RELEASE_BED in edge:
                    att_release_bed = edge[ATT_RELEASE_BED]
                else:
                    att_release_bed = DEFAULT_RELEASE_BED

                nx.set_edge_attributes(route_graph, {
                    (source, dest): {ATT_RELEASE_BED: att_release_bed}})

                if 'blocking_adjustment' in edge:
                    blocking_adj = edge['blocking_adjustment']
                else:
                    blocking_adj = None

                nx.set_edge_attributes(route_graph, {
                    (source, dest): {'blocking_adjustment': blocking_adj}})

                if 'discharge_adjustment' in edge:
                    discharge_adj = edge['discharge_adjustment']
                else:
                    discharge_adj = None

                nx.set_edge_attributes(route_graph, {
                    (source, dest): {'discharge_adjustment': discharge_adj}})

            # Each patient will eventually end up with their own copy of the route since
            # it will contain LOS values
            self.route_graphs[route_name] = route_graph.copy()
            logging.debug(f"{self.env.now:.4f}: route graph {route_name} - {route_graph.edges}")

    def validate_route_graph(self, route_graph: DiGraph) -> bool:
        """
        Make sure route is of appropriate structure for router.

        Example: Static routes should have exactly one arc entering and one arc emanating from each non-egress node.

        Parameters
        ----------
        route_graph: DiGraph

        Returns
        -------
        bool
            True if route is valid

        """
        # TODO: Implement route validation rules

        # For example, all beds must eventually be released and can't keep bed if dest is EXIT
        if route_graph:
            return True
        else:
            return False

    def create_route(self, patient) -> DiGraph:
        """

        Parameters
        ----------
        patient

        Returns
        -------
        DiGraph
            Nodes are units with LOS information stored as node attributes

        Notes
        -----

        """

        # Copy the route template to create new graph object
        route_graph = deepcopy(self.route_graphs[patient.patient_type])

        # Sample from los distributions for planned_los
        for u, v, data in route_graph.edges(data=True):
            edge = route_graph.edges[u, v]
            if 'los' in data:
                try:
                    planned_los = float(edge['los'])
                    edge['planned_los'] = planned_los
                except ValueError:
                    los_params = self.patient_flow_system.config.los_params
                    rg = self.patient_flow_system.config.rg['arrivals']
                    edge['planned_los'] = \
                        create_los_partial(edge['los'], los_params, rg)

        return route_graph

    def get_next_step(self, patient, skip=False):
        """
        Get next step (edge) in route

        Parameters
        ----------
        patient: Patient
        skip: bool

        Returns
        -------
        List[Edges]


        """

        # Get this patient's route graph
        planned_route = patient.planned_route

        if patient.current_stop_num == 0:
            # We are at the ENTRY node
            next_edges = [(u, v, d) for (u, v, d) in
                          planned_route.out_edges(patient.current_unit_name, data=True)]
        elif not skip:
            # Not at ENTRY and not skipping the next edge (i.e., not blocked so long that LOS elapsed)
            next_edge_names = planned_route.edges[patient.current_step[SRC], patient.current_step[DEST]]['next']
            if not isinstance(next_edge_names, list):
                next_edge_names = [next_edge_names]
            next_edges = [(u, v, d) for (u, v, d) in
                          patient.planned_route.edges(data=True) for name in next_edge_names if d['id'] == name]
        else:
            num_stops_skipped = len(patient.skipped_edges_cache)
            last_skipped_edge_record = patient.skipped_edges_cache.pop()
            last_skipped_edge = last_skipped_edge_record['skipped_edge']

            next_edge_names = planned_route.edges[last_skipped_edge[SRC], last_skipped_edge[DEST]]['next']
            next_edges = [(u, v, d) for (u, v, d) in
                          planned_route.edges(data=True) for name in next_edge_names if d['id'] == name]

            # skipped_edge_record = {'current_stop_num': csn,
            #                        'skipped_edge': outgoing_route_edge,
            #                        'planned_los': planned_los}


        # Get all the edges out of current node whose edge_num is one more than current edge_num
        # For static routes, this should be a single edge.
        # next_edges = [(u, v, d) for (u, v, d) in
        #               G.out_edges(current_unit_name, data=True) if d['edge_num'] == next_edge_num]



        return next_edges
