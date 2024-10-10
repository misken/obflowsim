import logging
from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd
import networkx as nx
from networkx import DiGraph

from obflowsim.obconstants import UnitName, DEFAULT_GET_BED, DEFAULT_RELEASE_BED, ATT_RELEASE_BED, ATT_GET_BED
from obflowsim.los import create_los_partial, los_mean


class Router(ABC):

    @abstractmethod
    def validate_route_graph(self, route_graph: DiGraph) -> bool:
        """
        Determine if a route is considered valid

        Parameters
        ----------
        route_graph

        Returns
        -------
        True if route is valid
        """
        pass

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
        pfs: PatientFlowSystem
        """

        self.env = env
        self.patient_flow_system = pfs


        # Dict of networkx DiGraph objects
        self.route_graphs = {}

        los_params = pfs.config.los_params

        # Create route templates from routes list (of unit numbers)
        for route_name, route in self.patient_flow_system.config.routes.items():
            route_graph = nx.DiGraph()

            # Add edges - simple serial route in this case
            for edge in route['edges']:
                route_graph.add_edge(edge['from'], edge['to'])

                if 'edge_num' in edge:
                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {'edge_num': edge['edge_num']}})

                if 'los' in edge:
                    edge['los_mean'] = los_mean(edge['los'], los_params)

                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {'los': edge['los']}})

                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {'los_mean': edge['los_mean']}})

                # Add get and keep bed attributes
                if ATT_GET_BED in edge:
                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {ATT_GET_BED: edge[ATT_GET_BED]}})
                else:
                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {ATT_GET_BED: DEFAULT_GET_BED}})

                if ATT_RELEASE_BED in edge:
                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {ATT_RELEASE_BED: edge[ATT_RELEASE_BED]}})
                else:
                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {ATT_RELEASE_BED: DEFAULT_RELEASE_BED}})

                # Add blocking adjustment attribute
                if 'blocking_adjustment' in edge:
                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {'blocking_adjustment': edge['blocking_adjustment']}})
                else:
                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {'blocking_adjustment': None}})

                # Add discharge timing adjustment attribute
                if 'discharge_adjustment' in edge:
                    discharge_pmf_file = edge['discharge_adjustment']
                    discharge_pmf = pd.read_csv(discharge_pmf_file, sep='\s+', header=None, names=['x', 'p'])
                    discharge_pmf.set_index('x', inplace=True, drop=True)
                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {'discharge_adjustment': discharge_pmf}})
                else:
                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {'discharge_adjustment': None}})

            for node in route_graph.nodes():
                nx.set_node_attributes(route_graph,
                                       {node: {'planned_los': 0.0, 'actual_los': 0.0, 'blocked_duration': 0.0}})

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
                los_params = self.patient_flow_system.config.los_params
                rg = self.patient_flow_system.config.rg['arrivals']
                edge['planned_los'] = \
                    create_los_partial(edge['los'], los_params, rg)

        return route_graph

    def get_next_step(self, patient, after=None, unit=None):
        """
        Get next step (edge) in route

        Parameters
        ----------
        patient: Patient
        after: Edge

        Returns
        -------
        Edge


        """

        try:
            (after is None and unit is None) or (after is not None and unit is not None)
        except ValueError:
            raise ValueError('Both after and unit must be None or both must not be None')

        # Get this patient's route graph
        G = patient.route_graph

        # Find all possible next units
        if unit is None:
            current_unit_name = patient.get_current_unit_name()
        else:
            current_unit_name = unit

        if after is None:
            if current_unit_name == UnitName.ENTRY:
                next_edge_num = 1
            else:
                current_route_edge = patient.get_current_route_edge()
                next_edge_num = current_route_edge['edge_num'] + 1
        else:
            next_edge_num = after + 1

        # Get all the edges out of current node whose edge_num is one more than current edge_num
        # For static routes, this should be a single edge.
        next_edges = [(u, v, d) for (u, v, d) in
                      G.out_edges(current_unit_name, data=True) if d['edge_num'] == next_edge_num]

        return next_edges[0]
