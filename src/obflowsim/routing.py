import logging
from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd
import networkx as nx
from networkx import DiGraph

from obflowsim.obconstants import UnitName
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
    def get_next_stop(self, entity):
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

                if 'los' in edge:
                    edge['los_mean'] = los_mean(edge['los'], los_params)

                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {'los': edge['los']}})

                    nx.set_edge_attributes(route_graph, {
                        (edge['from'], edge['to']): {'los_mean': edge['los_mean']}})

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

    def get_next_stop(self, patient):
        """
        Get next unit in route

        Parameters
        ----------
        patient: Patient

        Returns
        -------
        str
            Unit names are used as node id's

        """

        # Get this patient's route graph
        G = patient.route_graph

        # Find all possible next units
        if patient.current_stop_num > 0:
            current_unit_name = patient.unit_stops[patient.current_stop_num]
        else:
            current_unit_name = UnitName.ENTRY

        successors = [n for n in G.successors(current_unit_name)]
        next_unit_name = successors[0]

        # if next_unit_name is None:
        #     if patient.current_stop_num == patient.route_length:
        #         # Patient is at last stop
        #         pass
        #     else:
        #         logging.error(
        #             f"{self.env.now:.4f}: {patient.patient_id} has no next unit at {current_unit_name}.")
        #     exit(1)
        #
        # else:
        #     if next_unit_name == UnitName.EXIT:
        #         next_unit_name = None
        #
        # logging.debug(
        #     f"{self.env.now:.4f}: {patient.patient_id} current_unit_name {current_unit_name}, next_unit_name {next_unit_name}")

        return next_unit_name
