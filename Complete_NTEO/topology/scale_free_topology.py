#!/usr/bin/env python3
"""
🏗️ SCALE-FREE NETWORK TOPOLOGY GENERATOR

Implementation of Barabási-Albert preferential attachment model with geographic constraints
for realistic Sydney transport network generation. This module creates hub-dominated networks
where major stations (Central, Parramatta, Liverpool) act as preferential attachment points.

Key Features:
- Preferential attachment with geographic constraints
- Transport hierarchy preservation (train hubs vs bus stations)
- Sydney-specific hub identification and weighting
- Realistic connection patterns respecting urban geography
- Integration with existing NetworkX-based routing system

Research Question: How do hub-dominated networks affect transport equity
compared to more distributed topologies?
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from topology.network_topology import SydneyNetworkTopology, TransportMode, NetworkNode, NodeType
from collections import defaultdict, Counter
import math

class ScaleFreeTopologyGenerator:
    """
    Implements Barabási-Albert preferential attachment model with Sydney-specific
    geographic and transport hierarchy constraints
    """
    
    def __init__(self, base_network: SydneyNetworkTopology):
        self.base_network = base_network
        self.route_counter = {'train': 300, 'bus': 3000}  # Different range for scale-free
        self.base_nodes = list(base_network.nodes.keys())
        
        # Geographic constraints for realistic attachment
        self.max_attachment_distance = 50  # Maximum distance for new connections
        self.min_attachment_distance = 3   # Minimum distance to avoid over-clustering
        
        # Transport hierarchy weights for preferential attachment
        self.hierarchy_weights = {
            NodeType.MAJOR_HUB: 3.0,        # Central, Parramatta, Liverpool
            NodeType.TRANSPORT_HUB: 2.0,     # Town Hall, Strathfield, etc.
            NodeType.LOCAL_STATION: 1.0,     # Regular stations
            NodeType.RESIDENTIAL_ZONE: 0.5,  # Residential areas
            NodeType.EMPLOYMENT_ZONE: 1.5    # Employment centers
        }
        
        # Sydney-specific hub identification
        self.natural_hubs = {
            'CENTRAL', 'PARRAMATTA', 'LIVERPOOL', 'TOWN_HALL', 
            'STRATHFIELD', 'HORNSBY', 'CHATSWOOD', 'BANKSTOWN'
        }
        
        # Mode-specific attachment preferences
        self.mode_compatibility = {
            TransportMode.TRAIN: [TransportMode.TRAIN, TransportMode.BUS],
            TransportMode.BUS: [TransportMode.BUS, TransportMode.TRAIN],
            TransportMode.METRO: [TransportMode.METRO, TransportMode.TRAIN, TransportMode.BUS]
        }
        
    def generate_scale_free_network(self, 
                                m_edges: int = 2,
                                seed_hubs: Optional[List[str]] = None,
                                preserve_geography: bool = True,
                                alpha: float = 1.0) -> nx.Graph:
        """
        Generate scale-free network using Barabási-Albert model with Sydney constraints
        """
        
        print(f"Generating scale-free network (m={m_edges}, α={alpha:.1f})...")
        
        # Step 1: Initialize with seed hubs
        scale_free_graph = self._initialize_seed_network(seed_hubs or self._identify_seed_hubs())
        
        # Step 2: Apply preferential attachment for remaining nodes
        remaining_nodes = [node for node in self.base_nodes 
                        if node not in scale_free_graph.nodes()]
        
        scale_free_graph = self._apply_preferential_attachment(
            scale_free_graph, remaining_nodes, m_edges, alpha, preserve_geography
        )
        
        # Step 3: COMMENT OUT OR REMOVE THIS LINE - it's likely calling small-world code
        # scale_free_graph = self._add_transport_hierarchy_connections(scale_free_graph)
        
        # Step 4: Validate and optimize network
        scale_free_graph = self._ensure_network_connectivity(scale_free_graph)
        scale_free_graph = self._optimize_hub_connectivity(scale_free_graph)
        
        # DO NOT call any rewiring functions here!
        # DO NOT call any small-world methods!
        
        print(f"Scale-free network created: {scale_free_graph.number_of_nodes()} nodes, "
            f"{scale_free_graph.number_of_edges()} edges")
        
        return scale_free_graph
    def _create_edge_attributes(self, node1_data: NetworkNode, 
                              node2_data: NetworkNode, 
                              distance: float) -> Dict:
        """Create edge attributes with Sydney-style route IDs - FIXED VERSION"""
        
        # 🎯 CRITICAL FIX: Check if this edge exists in original Sydney network first
        if self.base_network.graph.has_edge(node1_data.node_id, node2_data.node_id):
            # Return original Sydney edge data completely
            original_data = self.base_network.graph[node1_data.node_id][node2_data.node_id]
            return original_data
        
        # For new edges, determine importance and create appropriate route
        node1_importance = self._calculate_node_importance(node1_data)
        node2_importance = self._calculate_node_importance(node2_data)
        combined_importance = node1_importance + node2_importance
        
        # High-importance connections get premium routes
        if combined_importance > 15:  # Both major hubs
            transport_mode = TransportMode.TRAIN
            travel_time = distance * 0.7  # Premium routes are fastest
            capacity = 2000
            frequency = 15
            
            # 🎯 Create Sydney-style premium train route ID
            self.route_counter['train'] += 1
            route_id = f"T{self.route_counter['train']}_EXPRESS"  # T301_EXPRESS, etc.
            
        elif combined_importance > 8:  # At least one important hub
            transport_mode = TransportMode.BUS
            travel_time = distance * 0.9  # Express buses
            capacity = 800
            frequency = 12
            
            # 🎯 Create Sydney-style express bus route ID
            self.route_counter['bus'] += 1
            route_id = f"BUS_X{self.route_counter['bus']}"  # BUS_X3001, BUS_X3002, etc.
            
        else:  # Regular connections
            transport_mode = TransportMode.BUS
            travel_time = distance * 1.2
            capacity = 600
            frequency = 8
            
            # 🎯 Create Sydney-style regular bus route ID
            self.route_counter['bus'] += 1
            route_id = f"BUS_{self.route_counter['bus']}"  # BUS_3001, BUS_3002, etc.
        
        return {
            'transport_mode': transport_mode,
            'travel_time': travel_time,
            'capacity': capacity,
            'frequency': frequency,
            'distance': distance,
            'route_id': route_id,  # 🎯 KEY FIX: Add route_id!
            'segment_order': 1,
            'edge_type': 'scale_free_generated',
            'weight': travel_time
        }
    
    def _calculate_node_importance(self, node_data: NetworkNode) -> float:
        """Calculate node importance for route assignment"""
        importance = 0
        
        # Base importance from node type
        if node_data.node_type == NodeType.MAJOR_HUB:
            importance += 10
        elif node_data.node_type == NodeType.TRANSPORT_HUB:
            importance += 5
        elif node_data.node_type == NodeType.LOCAL_STATION:
            importance += 2
        
        # Employment/population weight
        importance += node_data.employment_weight * 3
        importance += node_data.population_weight * 2
        
        return importance
    def _identify_seed_hubs(self) -> List[str]:
        """Identify natural seed hubs based on Sydney transport hierarchy"""
        seed_hubs = []
        
        # Priority 1: Major transport hubs
        for node_id, node_data in self.base_network.nodes.items():
            if (node_data.node_type == NodeType.MAJOR_HUB or 
                node_id in self.natural_hubs):
                seed_hubs.append(node_id)
        
        # Priority 2: Key transport interchanges
        for node_id, node_data in self.base_network.nodes.items():
            if (node_data.node_type == NodeType.TRANSPORT_HUB and 
                len(node_data.transport_modes) >= 2):
                seed_hubs.append(node_id)
        
        # Ensure minimum seed size
        if len(seed_hubs) < 3:
            additional_nodes = [node for node in self.base_nodes 
                              if node not in seed_hubs][:3-len(seed_hubs)]
            seed_hubs.extend(additional_nodes)
        
        print(f"Identified {len(seed_hubs)} seed hubs: {seed_hubs[:5]}...")
        return seed_hubs
    
    def _initialize_seed_network(self, seed_hubs: List[str]) -> nx.Graph:
        """Create initial connected network from seed hubs"""
        seed_graph = nx.Graph()
        
        # Add seed nodes
        for hub_id in seed_hubs:
            if hub_id in self.base_network.nodes:
                node_data = self.base_network.nodes[hub_id]
                seed_graph.add_node(hub_id, **self._create_node_attributes(node_data))
        
        # Connect seed hubs to form initial network
        seed_graph = self._create_seed_connections(seed_graph, seed_hubs)
        
        print(f"Seed network initialized: {len(seed_hubs)} hubs, "
              f"{seed_graph.number_of_edges()} initial connections")
        
        return seed_graph
    
    def _create_seed_connections(self, seed_graph: nx.Graph, seed_hubs: List[str]) -> nx.Graph:
        """Create realistic connections between seed hubs"""
        
        # Connect based on geographic proximity and transport compatibility
        for i, hub1 in enumerate(seed_hubs):
            for j, hub2 in enumerate(seed_hubs[i+1:], i+1):
                if hub1 in self.base_network.nodes and hub2 in self.base_network.nodes:
                    
                    node1_data = self.base_network.nodes[hub1]
                    node2_data = self.base_network.nodes[hub2]
                    
                    distance = self._calculate_euclidean_distance(
                        node1_data.coordinates, node2_data.coordinates
                    )
                    
                    # Connect if reasonable distance and transport compatibility
                    if (distance <= 25 and  # Reasonable hub-to-hub distance
                        self._check_transport_compatibility(node1_data, node2_data)):
                        
                        edge_attrs = self._create_edge_attributes(node1_data, node2_data, distance)
                        seed_graph.add_edge(hub1, hub2, **edge_attrs)
        
        # Ensure connectivity of seed network
        if not nx.is_connected(seed_graph):
            seed_graph = self._force_seed_connectivity(seed_graph, seed_hubs)
        
        return seed_graph
    
    def _apply_preferential_attachment(self, 
                                     graph: nx.Graph, 
                                     remaining_nodes: List[str],
                                     m_edges: int,
                                     alpha: float,
                                     preserve_geography: bool) -> nx.Graph:
        """Apply Barabási-Albert preferential attachment with constraints"""
        
        # Randomize node addition order for variation
        remaining_nodes = remaining_nodes.copy()
        random.shuffle(remaining_nodes)
        
        for node_id in remaining_nodes:
            if node_id not in self.base_network.nodes:
                continue
                
            new_node_data = self.base_network.nodes[node_id]
            
            # Add new node
            graph.add_node(node_id, **self._create_node_attributes(new_node_data))
            
            # Find attachment targets using preferential attachment
            attachment_targets = self._find_attachment_targets(
                graph, node_id, new_node_data, m_edges, alpha, preserve_geography
            )
            
            # Create connections to targets
            for target_id in attachment_targets:
                target_data = self.base_network.nodes[target_id]
                distance = self._calculate_euclidean_distance(
                    new_node_data.coordinates, target_data.coordinates
                )
                
                edge_attrs = self._create_edge_attributes(new_node_data, target_data, distance)
                graph.add_edge(node_id, target_id, **edge_attrs)
        
        print(f"Preferential attachment completed for {len(remaining_nodes)} nodes")
        return graph
    
    def _find_attachment_targets(self,
                               graph: nx.Graph,
                               new_node_id: str,
                               new_node_data: NetworkNode,
                               m_edges: int,
                               alpha: float,
                               preserve_geography: bool) -> List[str]:
        """Find nodes to attach to using preferential attachment with constraints"""
        
        # Calculate attachment probabilities
        attachment_probs = {}
        existing_nodes = list(graph.nodes())
        
        for node_id in existing_nodes:
            if node_id in self.base_network.nodes:
                node_data = self.base_network.nodes[node_id]
                
                # Base probability from degree (preferential attachment)
                degree = graph.degree(node_id)
                base_prob = (degree + 1) ** alpha
                
                # Geographic constraint
                distance = self._calculate_euclidean_distance(
                    new_node_data.coordinates, node_data.coordinates
                )
                
                if preserve_geography:
                    if distance > self.max_attachment_distance:
                        geographic_weight = 0.0
                    elif distance < self.min_attachment_distance:
                        geographic_weight = 0.3
                    else:
                        # Inverse distance weighting with minimum
                        geographic_weight = max(0.1, 1.0 / (1.0 + distance / 10.0))
                else:
                    geographic_weight = 1.0
                
                # Transport hierarchy weight
                hierarchy_weight = self.hierarchy_weights.get(node_data.node_type, 1.0)
                
                # Transport mode compatibility
                compatibility_weight = self._calculate_transport_compatibility(
                    new_node_data, node_data
                )
                
                # Final probability
                attachment_probs[node_id] = (base_prob * geographic_weight * 
                                           hierarchy_weight * compatibility_weight)
        
        # Select m_edges targets based on probabilities
        return self._select_attachment_targets(attachment_probs, m_edges)
    
    def _select_attachment_targets(self, probs: Dict[str, float], m_edges: int) -> List[str]:
        """Select attachment targets based on calculated probabilities"""
        
        # Remove zero-probability nodes
        valid_probs = {k: v for k, v in probs.items() if v > 0}
        
        if len(valid_probs) <= m_edges:
            return list(valid_probs.keys())
        
        # Normalize probabilities
        total_prob = sum(valid_probs.values())
        if total_prob == 0:
            # Fallback: random selection
            return random.sample(list(valid_probs.keys()), min(m_edges, len(valid_probs)))
        
        normalized_probs = {k: v/total_prob for k, v in valid_probs.items()}
        
        # Sample without replacement
        targets = []
        remaining_nodes = list(normalized_probs.keys())
        remaining_probs = list(normalized_probs.values())
        
        for _ in range(min(m_edges, len(remaining_nodes))):
            if not remaining_nodes:
                break
                
            # Weighted random selection
            selected_idx = np.random.choice(len(remaining_nodes), p=remaining_probs)
            selected_node = remaining_nodes.pop(selected_idx)
            targets.append(selected_node)
            
            # Renormalize remaining probabilities
            remaining_probs.pop(selected_idx)
            if remaining_probs:
                prob_sum = sum(remaining_probs)
                if prob_sum > 0:
                    remaining_probs = [p/prob_sum for p in remaining_probs]
        
        return targets
    
    def _calculate_transport_compatibility(self, node1: NetworkNode, node2: NetworkNode) -> float:
        """Calculate transport mode compatibility weight"""
        
        # Find common transport modes
        common_modes = set(node1.transport_modes) & set(node2.transport_modes)
        if common_modes:
            return 2.0  # Strong compatibility
        
        # Check for compatible mode combinations
        for mode1 in node1.transport_modes:
            compatible_modes = self.mode_compatibility.get(mode1, [])
            if any(mode2 in compatible_modes for mode2 in node2.transport_modes):
                return 1.5  # Moderate compatibility
        
        return 0.8  # Low but possible compatibility
    
    def _add_transport_hierarchy_connections(self, graph: nx.Graph) -> nx.Graph:
        """Add transport-specific hierarchy connections"""
        
        # This method should NOT do any rewiring!
        # It should only ensure major hubs are well-connected
        
        major_hubs = [node_id for node_id in graph.nodes() 
                    if (node_id in self.base_network.nodes and 
                        self.base_network.nodes[node_id].node_type == NodeType.MAJOR_HUB)]
        
        # Only add direct connections between disconnected major hubs
        for i, hub1 in enumerate(major_hubs):
            for hub2 in major_hubs[i+1:]:
                if not graph.has_edge(hub1, hub2):
                    # Check distance constraint
                    coord1 = self.base_network.nodes[hub1].coordinates
                    coord2 = self.base_network.nodes[hub2].coordinates
                    distance = self._calculate_euclidean_distance(coord1, coord2)
                    
                    if distance <= self.max_attachment_distance:
                        # Add edge WITHOUT calling any rewiring functions
                        edge_attrs = self._create_edge_attributes(
                            self.base_network.nodes[hub1],
                            self.base_network.nodes[hub2],
                            distance
                        )
                        graph.add_edge(hub1, hub2, **edge_attrs)
        
        # DO NOT CALL ANY REWIRING HERE
        # DO NOT CREATE SHORTCUTS
        # DO NOT APPLY SMALL-WORLD MODIFICATIONS
        
        return graph
    
    def _enhance_hub_connections(self, graph: nx.Graph, hub_id: str, target_degree: int):
        """Enhance connectivity of a specific hub"""
        
        current_degree = graph.degree(hub_id)
        additional_needed = max(0, target_degree - current_degree)
        
        if additional_needed == 0 or hub_id not in self.base_network.nodes:
            return
        
        hub_data = self.base_network.nodes[hub_id]
        
        # Find nearby unconnected nodes
        candidates = []
        for node_id in graph.nodes():
            if (node_id != hub_id and 
                not graph.has_edge(hub_id, node_id) and
                node_id in self.base_network.nodes):
                
                node_data = self.base_network.nodes[node_id]
                distance = self._calculate_euclidean_distance(
                    hub_data.coordinates, node_data.coordinates
                )
                
                if distance <= 20:  # Reasonable connection distance
                    candidates.append((node_id, distance))
        
        # Sort by distance and connect to closest
        candidates.sort(key=lambda x: x[1])
        for node_id, distance in candidates[:additional_needed]:
            node_data = self.base_network.nodes[node_id]
            edge_attrs = self._create_edge_attributes(hub_data, node_data, distance)
            graph.add_edge(hub_id, node_id, **edge_attrs)
    
    def _add_bus_coverage_connections(self, graph: nx.Graph) -> nx.Graph:
        """Add bus connections to ensure network coverage"""
        
        # Identify areas with low connectivity
        low_connectivity_nodes = [node_id for node_id in graph.nodes() 
                                if graph.degree(node_id) <= 1]
        
        for node_id in low_connectivity_nodes:
            if node_id not in self.base_network.nodes:
                continue
                
            node_data = self.base_network.nodes[node_id]
            
            # Find nearby bus-compatible nodes
            bus_candidates = []
            for other_id in graph.nodes():
                if (other_id != node_id and 
                    not graph.has_edge(node_id, other_id) and
                    other_id in self.base_network.nodes):
                    
                    other_data = self.base_network.nodes[other_id]
                    
                    # Check if bus connection makes sense
                    if (TransportMode.BUS in node_data.transport_modes or
                        TransportMode.BUS in other_data.transport_modes):
                        
                        distance = self._calculate_euclidean_distance(
                            node_data.coordinates, other_data.coordinates
                        )
                        
                        if 5 <= distance <= 15:  # Good bus route distance
                            bus_candidates.append((other_id, distance))
            
            # Add one bus connection to closest suitable node
            if bus_candidates:
                bus_candidates.sort(key=lambda x: x[1])
                target_id, distance = bus_candidates[0]
                target_data = self.base_network.nodes[target_id]
                
                edge_attrs = self._create_edge_attributes(node_data, target_data, distance)
                edge_attrs['transport_mode'] = TransportMode.BUS
                graph.add_edge(node_id, target_id, **edge_attrs)
        
        return graph
    
    def _optimize_hub_connectivity(self, graph: nx.Graph) -> nx.Graph:
        """Optimize connectivity of major hubs for scale-free properties"""
        
        # Calculate degree distribution
        degrees = dict(graph.degree())
        
        # Identify top hubs by degree
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        top_hubs = [node_id for node_id, degree in sorted_nodes[:5]]
        
        # Ensure top hubs are geographically sensible major stations
        validated_hubs = []
        for hub_id in top_hubs:
            if (hub_id in self.base_network.nodes and
                self.base_network.nodes[hub_id].node_type in 
                [NodeType.MAJOR_HUB, NodeType.TRANSPORT_HUB]):
                validated_hubs.append(hub_id)
        
        print(f"Top network hubs: {validated_hubs}")
        return graph
    
    # Helper methods (same pattern as small_world_topology.py)
    def _calculate_euclidean_distance(self, coord1: Tuple[float, float], 
                                    coord2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between coordinates"""
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    def _check_transport_compatibility(self, node1: NetworkNode, node2: NetworkNode) -> bool:
        """Check if two nodes have compatible transport modes"""
        return bool(set(node1.transport_modes) & set(node2.transport_modes))
    
    def _create_node_attributes(self, node_data: NetworkNode) -> Dict:
        """Create node attributes for NetworkX graph"""
        return {
            'node_type': node_data.node_type,
            'coordinates': node_data.coordinates,
            'population_weight': node_data.population_weight,
            'employment_weight': node_data.employment_weight,
            'transport_modes': node_data.transport_modes,
            'zone_name': node_data.zone_name
        }
    
    def _create_edge_attributes(self, node1: NetworkNode, node2: NetworkNode, 
                              distance: float) -> Dict:
        """Create edge attributes for NetworkX graph"""
        
        # Determine transport mode
        common_modes = set(node1.transport_modes) & set(node2.transport_modes)
        if TransportMode.TRAIN in common_modes:
            transport_mode = TransportMode.TRAIN
            travel_time = distance * 1.2  # Train speed factor
            capacity = 1800
            frequency = 8
        elif TransportMode.BUS in common_modes:
            transport_mode = TransportMode.BUS
            travel_time = distance * 2.0  # Bus speed factor
            capacity = 120
            frequency = 15
        else:
            transport_mode = TransportMode.BUS  # Default fallback
            travel_time = distance * 2.2
            capacity = 120
            frequency = 20
        
        return {
            'transport_mode': transport_mode,
            'travel_time': max(1.0, travel_time),
            'capacity': capacity,
            'frequency': frequency,
            'distance': distance,
            'route_id': f"SF_{transport_mode.value}_{random.randint(100, 999)}",
            'segment_order': 1
        }
    
    def _ensure_network_connectivity(self, graph: nx.Graph) -> nx.Graph:
        """Ensure the network is fully connected"""
        
        if nx.is_connected(graph):
            return graph
        
        print("Network not connected, adding bridge connections...")
        
        # Find connected components
        components = list(nx.connected_components(graph))
        main_component = max(components, key=len)
        
        # Connect smaller components to main component
        for component in components:
            if component == main_component:
                continue
            
            # Find closest nodes between components
            min_distance = float('inf')
            best_connection = None
            
            for node1 in component:
                for node2 in main_component:
                    if (node1 in self.base_network.nodes and 
                        node2 in self.base_network.nodes):
                        
                        coord1 = self.base_network.nodes[node1].coordinates
                        coord2 = self.base_network.nodes[node2].coordinates
                        distance = self._calculate_euclidean_distance(coord1, coord2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_connection = (node1, node2, distance)
            
            # Add bridge connection
            if best_connection:
                node1, node2, distance = best_connection
                node1_data = self.base_network.nodes[node1]
                node2_data = self.base_network.nodes[node2]
                
                edge_attrs = self._create_edge_attributes(node1_data, node2_data, distance)
                graph.add_edge(node1, node2, **edge_attrs)
        
        return graph
    
    def _force_seed_connectivity(self, graph: nx.Graph, seed_hubs: List[str]) -> nx.Graph:
        """Force connectivity in seed network if needed"""
        
        components = list(nx.connected_components(graph))
        if len(components) <= 1:
            return graph
        
        main_component = max(components, key=len)
        
        for component in components:
            if component == main_component:
                continue
            
            # Connect to main component
            component_node = list(component)[0]
            main_node = list(main_component)[0]
            
            if (component_node in self.base_network.nodes and 
                main_node in self.base_network.nodes):
                
                node1_data = self.base_network.nodes[component_node]
                node2_data = self.base_network.nodes[main_node]
                distance = self._calculate_euclidean_distance(
                    node1_data.coordinates, node2_data.coordinates
                )
                
                edge_attrs = self._create_edge_attributes(node1_data, node2_data, distance)
                graph.add_edge(component_node, main_node, **edge_attrs)
        
        return graph

class ScaleFreeParameterOptimizer:
    """
    Optimizer for scale-free network parameters to achieve desired network properties
    """
    
    def __init__(self, base_network: SydneyNetworkTopology):
        self.base_network = base_network
        self.generator = ScaleFreeTopologyGenerator(base_network)
    
    def find_optimal_parameters(self, 
                               target_properties: Dict[str, float],
                               m_range: Tuple[int, int] = (1, 5),
                               alpha_range: Tuple[float, float] = (0.5, 2.0)) -> Dict:
        """
        Find optimal parameters for desired network properties
        
        Args:
            target_properties: Desired network characteristics
            m_range: Range for m_edges parameter
            alpha_range: Range for alpha parameter
        
        Returns:
            Dictionary with optimal parameters and achieved properties
        """
        
        best_score = float('inf')
        best_params = None
        best_properties = None
        
        # Grid search over parameter space
        m_values = range(m_range[0], m_range[1] + 1)
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], 5)
        
        for m in m_values:
            for alpha in alpha_values:
                try:
                    # Generate network
                    network = self.generator.generate_scale_free_network(
                        m_edges=m, alpha=alpha
                    )
                    
                    # Analyze properties
                    properties = self.analyze_scale_free_properties(network)
                    
                    # Calculate score based on target properties
                    score = self._calculate_property_score(properties, target_properties)
                    
                    if score < best_score:
                        best_score = score
                        best_params = {'m_edges': m, 'alpha': alpha}
                        best_properties = properties
                        
                except Exception as e:
                    print(f"Failed for m={m}, α={alpha}: {e}")
                    continue
        
        return {
            'optimal_parameters': best_params,
            'achieved_properties': best_properties,
            'optimization_score': best_score
        }
    
    def analyze_scale_free_properties(self, graph: nx.Graph) -> Dict[str, float]:
        """Analyze scale-free network properties"""
        
        properties = {}
        
        # Basic network metrics
        properties['num_nodes'] = graph.number_of_nodes()
        properties['num_edges'] = graph.number_of_edges()
        properties['density'] = nx.density(graph)
        
        # Scale-free specific metrics
        degrees = dict(graph.degree())
        degree_values = list(degrees.values())
        
        properties['avg_degree'] = np.mean(degree_values)
        properties['max_degree'] = max(degree_values)
        properties['degree_variance'] = np.var(degree_values)
        
        # Power-law fitting (simplified)
        degree_counts = Counter(degree_values)
        properties['num_hubs'] = sum(1 for d in degree_values if d >= 5)
        properties['hub_ratio'] = properties['num_hubs'] / properties['num_nodes']
        
        # Connectivity metrics
        properties['avg_clustering'] = nx.average_clustering(graph)
        properties['avg_path_length'] = nx.average_shortest_path_length(graph)
        
        # Transport-specific metrics
        major_hubs = [node for node in graph.nodes() 
                     if node in self.base_network.nodes and
                     self.base_network.nodes[node].node_type == NodeType.MAJOR_HUB]
        
        if major_hubs:
            hub_degrees = [degrees[hub] for hub in major_hubs]
            properties['avg_hub_degree'] = np.mean(hub_degrees)
        else:
            properties['avg_hub_degree'] = 0
        
        return properties
    
    def _calculate_property_score(self, properties: Dict[str, float], 
                                targets: Dict[str, float]) -> float:
        """Calculate how well properties match targets"""
        
        score = 0.0
        for prop_name, target_value in targets.items():
            if prop_name in properties:
                actual_value = properties[prop_name]
                # Normalized squared difference
                diff = abs(actual_value - target_value) / max(target_value, 0.001)
                score += diff ** 2
        
        return score

# Utility functions for integration with existing system
def create_scale_free_network_manager(m_edges: int = 2, alpha: float = 1.0):
    """Create network manager with scale-free topology"""
    from topology.network_topology import SydneyNetworkTopology
    from unified_network_integration import TwoLayerNetworkManager, NetworkSpatialMapper, NetworkEdgeCongestionModel, NetworkRouter
    
    # Create base Sydney network
    base_network = SydneyNetworkTopology()
    base_network.initialize_base_sydney_network()
    
    # Generate scale-free network
    generator = ScaleFreeTopologyGenerator(base_network)
    scale_free_graph = generator.generate_scale_free_network(
        m_edges=m_edges,
        alpha=alpha,
        preserve_geography=True
    )
    
    # Create custom network manager
    class ScaleFreeNetworkManager(TwoLayerNetworkManager):
        def __init__(self, sf_graph, base_net):
            self.base_network = base_net
            self.active_network = sf_graph
            
            # Initialize components
            self.spatial_mapper = NetworkSpatialMapper(base_net, 100, 80)
            self.congestion_model = NetworkEdgeCongestionModel(sf_graph)
            self.router = NetworkRouter(sf_graph, self.congestion_model)
            self.route_calculation_count = 0
            
            # Ensure connectivity
            self._ensure_network_connectivity()
    
    return ScaleFreeNetworkManager(scale_free_graph, base_network)

def run_scale_free_analysis(m_values: List[int] = [1, 2, 3], 
                           alpha_values: List[float] = [0.5, 1.0, 1.5]) -> Dict:
    """Run scale-free analysis for multiple parameter combinations"""
    
    from topology.network_topology import SydneyNetworkTopology
    
    base_network = SydneyNetworkTopology()
    base_network.initialize_base_sydney_network()
    
    generator = ScaleFreeTopologyGenerator(base_network)
    optimizer = ScaleFreeParameterOptimizer(base_network)
    
    results = {}
    
    for m in m_values:
        for alpha in alpha_values:
            param_key = f"m{m}_a{alpha:.1f}"
            
            try:
                # Generate network
                network = generator.generate_scale_free_network(
                    m_edges=m, alpha=alpha
                )
                
                # Analyze properties
                properties = optimizer.analyze_scale_free_properties(network)
                
                results[param_key] = {
                    'parameters': {'m_edges': m, 'alpha': alpha},
                    'network_properties': properties,
                    'network': network
                }
                
                print(f"✅ {param_key}: {properties['num_nodes']} nodes, "
                      f"avg_degree={properties['avg_degree']:.2f}, "
                      f"hubs={properties['num_hubs']}")
                
            except Exception as e:
                print(f"❌ {param_key} failed: {e}")
                results[param_key] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    # Quick test
    print("🏗️ SCALE-FREE NETWORK TOPOLOGY GENERATOR - TEST")
    print("=" * 60)
    
    # Run basic test
    test_results = run_scale_free_analysis()
    
    print(f"\n📊 Generated {len(test_results)} scale-free networks")
    print("Ready for integration with ABM system!")