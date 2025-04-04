import logging
from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd
import networkx as nx
from networkx import DiGraph

from obflowsim.obconstants import UnitName, DEFAULT_GET_BED, DEFAULT_RELEASE_BED, ATT_RELEASE_BED, ATT_GET_BED
from obflowsim.obconstants import SRC, DEST, DATA
from obflowsim.los import create_los_partial, los_mean


class Router(ABC):

    @abstractmethod
    def get_next_step(self, entity):
        pass


class StaticRouter(Router):
    def __init__(self, env, pfs):
        """
        Routes patients having a fixed, serial route

        Parameters
        ----------
        env: Environment
        pfs: PatientFlowSystem - need access to patient flow network
        """

        self.env = env
        self.patient_flow_system = pfs

        # Dict of networkx DiGraph objects
        self.route_graphs = {}

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

                if 'next' in edge:
                    nx.set_edge_attributes(route_graph, {
                        (source, dest): {'next': edge['next']}})
                else:
                    nx.set_edge_attributes(route_graph, {
                        (source, dest): {'next': None}})

                if 'los' in network_edge and 'los' not in edge:
                    edge_los_mean = los_mean(network_edge['los'], los_params)
                    los = network_edge['los']
                elif 'los' in edge:
                    edge_los_mean = los_mean(edge['los'], los_params)
                    los = edge['los']
                else:
                    edge_los_mean = 0.0
                    los = '0.0'

                nx.set_edge_attributes(route_graph, {
                    (network_edge[SRC], network_edge[DEST]): {'los': los}})
                nx.set_edge_attributes(route_graph, {
                    (network_edge[SRC], network_edge[DEST]): {'los_mean': edge_los_mean}})

                # Add get and keep bed attributes
                if ATT_GET_BED in network_edge and ATT_GET_BED not in edge:
                    att_get_bed = network_edge[ATT_GET_BED]
                elif ATT_GET_BED in edge:
                    att_get_bed = edge[ATT_GET_BED]
                else:
                    att_get_bed = DEFAULT_GET_BED
                nx.set_edge_attributes(route_graph, {
                    (network_edge[SRC], network_edge[DEST]): {ATT_GET_BED: att_get_bed}})

                if ATT_RELEASE_BED in network_edge and ATT_RELEASE_BED not in edge:
                    att_release_bed = network_edge[ATT_RELEASE_BED]
                elif ATT_RELEASE_BED in edge:
                    att_release_bed = edge[ATT_RELEASE_BED]
                else:
                    att_release_bed = DEFAULT_RELEASE_BED
                nx.set_edge_attributes(route_graph, {
                    (network_edge[SRC], network_edge[DEST]): {ATT_RELEASE_BED: att_release_bed}})

                if 'blocking_adjustment' in network_edge and 'blocking_adjustment' not in edge:
                    blocking_adj = network_edge['blocking_adjustment']
                elif 'blocking_adjustment' in edge:
                    blocking_adj = edge['blocking_adjustment']
                else:
                    blocking_adj = None
                nx.set_edge_attributes(route_graph, {
                    (network_edge[SRC], network_edge[DEST]): {'blocking_adjustment': blocking_adj}})

                if 'discharge_adjustment' in network_edge and 'discharge_adjustment' not in edge:
                    discharge_adj = network_edge['discharge_adjustment']
                elif 'discharge_adjustment' in edge:
                    discharge_adj = edge['discharge_adjustment']
                else:
                    discharge_adj = None
                nx.set_edge_attributes(route_graph, {
                    (network_edge[SRC], network_edge[DEST]): {'discharge_adjustment': discharge_adj}})

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
        return True

    def create_route(self, patient) -> DiGraph:
        """

        Parameters
        ----------
        patient

        entry_delay: float (default is 0 implying patient uses ENTRY only as a queueing location)
            Used with scheduled arrivals by holding patient for ``entry_delay`` time units before allowed to enter

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
                except:
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
            next_edge_names = planned_route.edges[patient.next_step[SRC], patient.next_step[DEST]]['next']
            next_edges = [(u, v, d) for (u, v, d) in
                          planned_route.edges(data=True) for name in next_edge_names if d['id'] == name]
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
