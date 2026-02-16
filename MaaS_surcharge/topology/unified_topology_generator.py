# # unified_topology_generator.py
# """
# UNIFIED TOPOLOGY GENERATOR - ALL NETWORK TYPES IN ONE FILE
# Contains all topology generation methods: degree-constrained, small-world, scale-free, base Sydney
# Easy switching between topology types with standardized interface.
# """

# import networkx as nx
# import numpy as np
# import random
# from typing import Dict, List, Tuple, Optional, Union
# from enum import Enum
# from collections import Counter
# import math

# # Import base network definitions
# from topology.network_topology import SydneyNetworkTopology, NetworkNode, NetworkEdge, NodeType, TransportMode

# class TopologyType(Enum):
#     """Enumeration of all supported topology types"""
#     DEGREE_CONSTRAINED = "degree_constrained"
#     SMALL_WORLD = "small_world"
#     SCALE_FREE = "scale_free"
#     BASE_SYDNEY = "base_sydney"
#     GRID = "grid"

# class UnifiedTopologyGenerator:
#     """
#     Unified generator for all network topology types.
#     Single class handles all topology generation with consistent interface.
#     """
#     def __init__(self, base_network: SydneyNetworkTopology = None):
#         """Initialize with base Sydney network and route counters"""
#         if base_network is None:
#             base_network = SydneyNetworkTopology()
#             base_network.initialize_base_sydney_network()
        
#         self.base_network = base_network
#         # CRITICAL: Validate base network has required routes
#         self.nodes = self.base_network.nodes  # Fix for self.nodes references
    
#         self._validate_base_network()
    
#         # Route counters for generating unique IDs
#         self.route_counters = {
#             'train': 100,
#             'premium_train': 200, 
#             'express_bus': 300,
#             'hub_bus': 400,
#             'bus': 1000
#         }
        
#         # CONSISTENT TIME FACTORS (in steps per km)
#         self.time_factor_map = {
#             'PREMIUM_TRAIN': 0.017,  # ~0.017 steps per km
#             'TRAIN': 0.020,          # ~0.020 steps per km  
#             'EXPRESS_BUS': 0.024,    # ~0.024 steps per km
#             'BUS': 0.030             # ~0.030 steps per km
#         }
        
#         # CONSISTENT FREQUENCY RANGES (0.5-3 steps)
#         self.frequency_map = {
#             'PREMIUM_TRAIN': 0.5,    # Every 5 minutes
#             'TRAIN': 1.0,            # Every 10 minutes  
#             'EXPRESS_BUS': 1.5,      # Every 15 minutes
#             'BUS': 2.5               # Every 25 minutes
#         }
#         print(f"✅ UnifiedTopologyGenerator initialized with {base_network.graph.number_of_edges()} base edges")

#     def _validate_base_network(self):
#         """Validate base network has essential Sydney routes"""
#         required_routes = ["T1_WESTERN", "T4_ILLAWARRA", "T8_AIRPORT", "BUS_380"]
        
#         found_routes = set()
#         for u, v, edge_data in self.base_network.graph.edges(data=True):
#             if 'route_id' in edge_data:
#                 for route in required_routes:
#                     if route in edge_data['route_id']:
#                         found_routes.add(route)
        
#         missing = set(required_routes) - found_routes
#         if missing:
#             raise ValueError(f"Base network missing required routes: {missing}")
        
#         print(f"✅ Base network validation passed - all required routes present")

#     def _get_next_route_counter(self, route_type: str) -> int:
#         """Get next route counter - SIMPLE"""
#         # KISS: Map any variant to base type
#         if 'train' in route_type.lower():
#             key = 'train'
#         elif 'express' in route_type.lower():
#             key = 'express_bus'
#         else:
#             key = 'bus'
        
#         # Ensure key exists
#         if key not in self.route_counters:
#             self.route_counters[key] = 1000
        
#         self.route_counters[key] += 1
#         return self.route_counters[key]

#     def _calculate_distance(self, node1: str, node2: str) -> float:
#         """Calculate distance between nodes"""
#         if node1 not in self.base_network.nodes or node2 not in self.base_network.nodes:
#             return 50.0  # Default distance
        
#         coord1 = self.base_network.nodes[node1].coordinates
#         coord2 = self.base_network.nodes[node2].coordinates
#         return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

#     def _is_geographically_reasonable(self, node1: str, node2: str, max_distance: float = 40) -> bool:
#         """Check if connection is geographically reasonable"""
#         return self._calculate_distance(node1, node2) <= max_distance

#     def _identify_sydney_backbone(self, graph: nx.Graph) -> set:
#         """Identify critical Sydney routes that must be preserved"""
#         backbone = set()
        
#         for u, v, data in graph.edges(data=True):
#             route_id = data.get('route_id', '')
#             if any(critical in route_id for critical in ['T1_WESTERN', 'T4_ILLAWARRA', 'BUS_380']):
#                 backbone.add((u, v))
#                 backbone.add((v, u))  # Undirected
        
#         return backbone

#     def _identify_major_hubs(self, graph: nx.Graph) -> list:
#         """Identify major hubs for scale-free preferential attachment"""
#         major_hubs = []
        
#         for node_id, node_data in self.base_network.nodes.items():
#             if node_data.node_type == NodeType.MAJOR_HUB:
#                 major_hubs.append(node_id)
        
#         return major_hubs

#     def _select_node_by_degree(self, graph: nx.Graph, all_nodes: list) -> str:
#         """Select node using preferential attachment (degree-based probability)"""
#         degrees = [graph.degree(node) + 1 for node in all_nodes]  # +1 to avoid zero probability
#         total_degree = sum(degrees)
        
#         if total_degree == 0:
#             return random.choice(all_nodes)
        
#         probabilities = [d / total_degree for d in degrees]
#         return np.random.choice(all_nodes, p=probabilities)

#     def _is_shortcut_viable(self, node1: str, node2: str, graph: nx.Graph) -> bool:
#         """Check if shortcut creates meaningful network improvement"""
#         distance = self._calculate_distance(node1, node2)
        
#         # Must be geographically reasonable
#         if distance > 50:
#             return False
        
#         # Should connect different network regions (simplified check)
#         try:
#             current_path_length = nx.shortest_path_length(graph, node1, node2)
#             return current_path_length > 2  # Creates shortcut if current path > 2 hops
#         except nx.NetworkXNoPath:
#             return True  # Connect disconnected components
    
    

#     def _assign_shortcut_attributes(self, node1, node2, distance, topology_type, parameter):
#         """Assign attributes for topology-specific shortcuts"""
        
#         if topology_type == 'small_world':
#             # Express services for small-world shortcuts
#             if distance > 30:
#                 self.route_counters['premium_train'] += 1
#                 return ('PREMIUM_TRAIN', f"PT_{self.route_counters['premium_train']}_SW{int(parameter*100)}", 
#                     2000, 6)
#             elif distance > 15:
#                 self.route_counters['express_bus'] += 1
#                 return ('EXPRESS_BUS', f"EB_{self.route_counters['express_bus']}_SW{int(parameter*100)}", 
#                     1000, 10)
        
#         elif topology_type == 'scale_free':
#             # Hub-based services for scale-free
#             if self._is_hub_connection(node1, node2):
#                 self.route_counters['train'] += 1
#                 return ('TRAIN', f"T{self.route_counters['train']}_SF{parameter}", 
#                     1800, 8)
        
#         # Default bus service
#         self.route_counters['bus'] += 1
#         return ('BUS', f"BUS_{self.route_counters['bus']}", 600, 15)
    
#     def _is_hub_connection(self, node1: str, node2: str) -> bool:
#         """Check if connection involves at least one major or transport hub"""
#         if node1 not in self.nodes or node2 not in self.nodes:
#             return False
        
#         node1_type = self.nodes[node1].node_type
#         node2_type = self.nodes[node2].node_type
        
#         hub_types = [NodeType.MAJOR_HUB, NodeType.TRANSPORT_HUB]
#         return node1_type in hub_types or node2_type in hub_types
    
#     def generate_topology(self, 
#                          topology_type: Union[str, TopologyType],
#                          variation_parameter: Union[int, float],
#                          **kwargs) -> nx.Graph:
#         """
#         Universal topology generation method - handles ALL topology types
        
#         Args:
#             topology_type: Type of topology to generate
#             variation_parameter: Main parameter controlling topology variation
#             **kwargs: Additional topology-specific parameters
        
#         Returns:
#             NetworkX graph with specified topology
#         """
        
#         # Convert string to enum if needed
#         if isinstance(topology_type, str):
#             topology_type = TopologyType(topology_type)
        
#         print(f"🌐 Generating {topology_type.value} topology (param: {variation_parameter})")
        
#         # Route to appropriate generation method
#         if topology_type == TopologyType.DEGREE_CONSTRAINED:
#             return self._generate_degree_constrained(int(variation_parameter), **kwargs)
        
#         elif topology_type == TopologyType.SMALL_WORLD:
#             return self._generate_small_world(float(variation_parameter), **kwargs)
        
#         elif topology_type == TopologyType.SCALE_FREE:
#             return self._generate_scale_free(int(variation_parameter), **kwargs)
        
#         elif topology_type == TopologyType.BASE_SYDNEY:
#             return self._generate_base_sydney(int(variation_parameter), **kwargs)
        
#         elif topology_type == TopologyType.GRID:
#             return self._generate_grid(int(variation_parameter), **kwargs)
        
#         else:
#             raise ValueError(f"Unsupported topology type: {topology_type}")
    
#     # ===== DEGREE-CONSTRAINED NETWORKS =====
#     def _generate_degree_constrained(self, target_degree: int, **kwargs) -> nx.Graph:
#         """Generate degree-constrained network - FIXED with two-layer architecture"""
        
#         print(f"🔧 Generating degree-constrained network (target_degree={target_degree})")
        
#         # CRITICAL: Start with copy of base network (Layer 1)
#         graph = self.base_network.graph.copy()
#         print(f"  Starting with base network: {graph.number_of_nodes()} nodes")
        
#         # Calculate current degrees
#         node_degrees = dict(graph.degree())
        
#         # Add shortcuts to achieve target degree (Layer 2)
#         shortcuts_added = 0
#         for node in graph.nodes():
#             current_degree = node_degrees[node]
#             deficit = target_degree - current_degree
            
#             if deficit > 0:
#                 # Find suitable targets for new edges
#                 candidates = [n for n in graph.nodes() 
#                             if n != node and not graph.has_edge(node, n)]
                
#                 # Sort by distance for geographic realism
#                 candidates.sort(key=lambda n: self._calculate_distance(node, n))
                
#                 # Add edges up to target degree
#                 for candidate in candidates[:deficit]:
#                     self._add_transport_edge(graph, node, candidate, 
#                                             edge_type='degree_shortcut',
#                                             topology_type='degree_constrained',
#                                             parameter_value=target_degree)
#                     shortcuts_added += 1
        
 
#         print(f"✅ Degree-constrained network: {graph.number_of_nodes()} nodes, "
#             f"{graph.number_of_edges()} edges, {shortcuts_added} shortcuts added")
        
#         return graph

#     def _add_degree_constrained_shortcuts(self, graph: nx.Graph, target_degree: int) -> int:
#         """Add degree-constrained shortcuts while preserving base network"""
#         shortcuts_added = 0
#         all_nodes = list(graph.nodes())
        
#         for node in all_nodes:
#             current_degree = graph.degree(node)
#             shortage = target_degree - current_degree
            
#             if shortage <= 0:
#                 continue
            
#             # Find candidates for new connections
#             candidates = []
#             for other_node in all_nodes:
#                 if (other_node != node and 
#                     not graph.has_edge(node, other_node) and
#                     self._is_geographically_reasonable(node, other_node)):
#                     candidates.append(other_node)
            
#             # Add connections up to target degree
#             random.shuffle(candidates)
#             for i in range(min(shortage, len(candidates))):
#                 target_node = candidates[i]
                
#                 # Create shortcut edge with proper attributes
#                 edge_attrs = self._create_degree_constrained_edge_attributes(
#                     node, target_node, target_degree
#                 )
                
#                 graph.add_edge(node, target_node, **edge_attrs)
#                 shortcuts_added += 1
#                 print(f"   Added shortcut: {node} -> {target_node} ({edge_attrs['route_id']})")
        
#         return shortcuts_added

#     def _create_degree_constrained_edge_attributes(self, node1: str, node2: str, 
#                                                 degree_param: int) -> dict:
#         """Create edge attributes for degree-constrained shortcuts"""
#         distance = self._calculate_distance(node1, node2)
        
#         # Parameter-dependent thresholds (no hard-coding)
#         network_diameter = 100  # Based on Sydney grid
#         train_threshold = network_diameter * (0.2 + degree_param * 0.05)
#         express_threshold = train_threshold * 0.6
        
#         if distance > train_threshold:
#             return {
#                 'transport_mode': TransportMode.TRAIN,
#                 'route_id': f"T{self._get_next_route_counter('train')}_DC{degree_param}",
#                 'capacity': 1500 + (degree_param - 3) * 100,
#                 'frequency': max(6, 15 - degree_param),
#                 'travel_time': distance / 35,  # Train speed
#                 'distance': distance,
#                 'segment_order': 1
#             }
#         elif distance > express_threshold:
#             return {
#                 'transport_mode': TransportMode.BUS,
#                 'route_id': f"EB_{self._get_next_route_counter('express_bus')}_DC{degree_param}",
#                 'capacity': 800 + (degree_param - 3) * 50,
#                 'frequency': max(8, 12 - degree_param * 0.5),
#                 'travel_time': distance / 25,  # Express bus speed
#                 'distance': distance,
#                 'segment_order': 1
#             }
#         else:
#             return {
#                 'transport_mode': TransportMode.BUS,
#                 'route_id': f"BUS_{self._get_next_route_counter('bus')}_DC{degree_param}",
#                 'capacity': 600,
#                 'frequency': 15,
#                 'travel_time': distance / 20,  # Regular bus speed
#                 'distance': distance,
#                 'segment_order': 1
#             }
    
#     # ===== SMALL-WORLD NETWORKS =====
#     def _generate_small_world(self, rewiring_probability: float, **kwargs) -> nx.Graph:
#         """Generate small-world network preserving Sydney base"""
#         print(f"🏗️ Generating small-world topology (p={rewiring_probability})")
        
#         # LAYER 1: Start with Sydney base network
#         graph = self.base_network.graph.copy()
#         base_edges_count = graph.number_of_edges()
#         print(f"   Layer 1 (Base): {base_edges_count} Sydney edges preserved")
        
#         # LAYER 2: Add small-world shortcuts through strategic rewiring
#         shortcuts_added = self._add_small_world_shortcuts(graph, rewiring_probability)
        
#         total_edges = graph.number_of_edges()
#         print(f"   Layer 2 (Shortcuts): {shortcuts_added} shortcuts added")
#         print(f"   ✅ Total network: {total_edges} edges")
        
#         return graph

#     def _add_small_world_shortcuts(self, graph: nx.Graph, rewiring_prob: float) -> int:
#         """Add small-world shortcuts without removing base Sydney routes"""
#         shortcuts_added = 0
        
#         # Identify Sydney backbone routes (NEVER remove these)
#         sydney_backbone = self._identify_sydney_backbone(graph)
        
#         # Calculate how many shortcuts to add based on rewiring probability
#         total_possible_shortcuts = len(list(graph.nodes())) * 2  # Approximate
#         target_shortcuts = int(total_possible_shortcuts * rewiring_prob)
        
#         all_nodes = list(graph.nodes())
        
#         for _ in range(target_shortcuts):
#             # Select random nodes for shortcut
#             node1 = random.choice(all_nodes)
#             node2 = random.choice(all_nodes)
            
#             if (node1 != node2 and 
#                 not graph.has_edge(node1, node2) and
#                 self._is_shortcut_viable(node1, node2, graph)):
                
#                 # Create shortcut edge
#                 edge_attrs = self._create_small_world_edge_attributes(
#                     node1, node2, rewiring_prob
#                 )
                
#                 graph.add_edge(node1, node2, **edge_attrs)
#                 shortcuts_added += 1
#                 print(f"   Added shortcut: {node1} -> {node2} ({edge_attrs['route_id']})")
        
#         return shortcuts_added

#     def _create_small_world_edge_attributes(self, node1: str, node2: str, 
#                                         rewiring_param: float) -> dict:
#         """Create edge attributes for small-world shortcuts"""
#         distance = self._calculate_distance(node1, node2)
        
#         # Parameter-dependent service quality
#         premium_threshold = 100 * (0.3 + rewiring_param * 0.4)  # Network diameter based
#         express_threshold = premium_threshold * 0.7
        
#         if distance > premium_threshold:
#             return {
#                 'transport_mode': TransportMode.TRAIN,
#                 'route_id': f"PT_{self._get_next_route_counter('premium_train')}_SW{int(rewiring_param*100)}",
#                 'capacity': 2000,
#                 'frequency': 6,  # Very frequent premium service
#                 'travel_time': distance / 40,  # High speed
#                 'distance': distance,
#                 'segment_order': 1
#             }
#         elif distance > express_threshold:
#             return {
#                 'transport_mode': TransportMode.BUS,
#                 'route_id': f"EX_{self._get_next_route_counter('express_bus')}_SW{int(rewiring_param*100)}",
#                 'capacity': 1000,
#                 'frequency': 10,
#                 'travel_time': distance / 25,
#                 'distance': distance,
#                 'segment_order': 1
#             }
#         else:
#             return {
#                 'transport_mode': TransportMode.BUS,
#                 'route_id': f"BUS_{self._get_next_route_counter('bus')}_SW{int(rewiring_param*100)}",
#                 'capacity': 600,
#                 'frequency': 15,
#                 'travel_time': distance / 20,
#                 'distance': distance,
#                 'segment_order': 1
#             }

#     # ===== SCALE-FREE NETWORKS =====
#     def _generate_scale_free(self, attachment_parameter: int, **kwargs) -> nx.Graph:
#         """Generate scale-free network preserving Sydney base"""
#         print(f"🏗️ Generating scale-free topology (m={attachment_parameter})")
        
#         # LAYER 1: Start with Sydney base network
#         graph = self.base_network.graph.copy()
#         base_edges_count = graph.number_of_edges()
#         print(f"   Layer 1 (Base): {base_edges_count} Sydney edges preserved")
        
#         # LAYER 2: Add scale-free connections via preferential attachment
#         shortcuts_added = self._add_scale_free_shortcuts(graph, attachment_parameter)
        
#         total_edges = graph.number_of_edges()
#         print(f"   Layer 2 (Shortcuts): {shortcuts_added} shortcuts added")
#         print(f"   ✅ Total network: {total_edges} edges")
        
#         return graph

#     def _add_scale_free_shortcuts(self, graph: nx.Graph, attachment_param: int) -> int:
#         """Add scale-free shortcuts using preferential attachment"""
#         shortcuts_added = 0
        
#         # Identify major hubs in Sydney network
#         major_hubs = self._identify_major_hubs(graph)
#         all_nodes = list(graph.nodes())
        
#         # Calculate target number of shortcuts based on parameter
#         target_shortcuts = attachment_param * 10  # Scales with parameter
        
#         for _ in range(target_shortcuts):
#             # Select nodes using preferential attachment
#             node1 = self._select_node_by_degree(graph, all_nodes)
#             node2 = self._select_node_by_degree(graph, all_nodes)
            
#             if (node1 != node2 and 
#                 not graph.has_edge(node1, node2)):
                
#                 # Create hub-focused edge
#                 edge_attrs = self._create_scale_free_edge_attributes(
#                     node1, node2, attachment_param, major_hubs
#                 )
                
#                 graph.add_edge(node1, node2, **edge_attrs)
#                 shortcuts_added += 1
      
        
#         return shortcuts_added

#     def _create_scale_free_edge_attributes(self, node1: str, node2: str, 
#                                         attachment_param: int, major_hubs: list) -> dict:
#         """Create edge attributes for scale-free shortcuts"""
#         distance = self._calculate_distance(node1, node2)
        
#         # Hub importance calculation
#         hub1_importance = 1.0 if node1 in major_hubs else 0.0
#         hub2_importance = 1.0 if node2 in major_hubs else 0.0
#         combined_importance = hub1_importance + hub2_importance
        
#         # Mode assignment based on hub connections and parameter
#         if combined_importance >= 1.5 and attachment_param >= 3:
#             return {
#                 'transport_mode': TransportMode.TRAIN,
#                 'route_id': f"PT_{self._get_next_route_counter('premium_train')}_SF{attachment_param}",
#                 'capacity': 2000,
#                 'frequency': 6,
#                 'travel_time': distance / 40,  # Premium speed
#                 'distance': distance,
#                 'segment_order': 1
#             }
#         elif combined_importance >= 1.0 or attachment_param >= 2:
#             return {
#                 'transport_mode': TransportMode.TRAIN,
#                 'route_id': f"T_{self._get_next_route_counter('train')}_SF{attachment_param}",
#                 'capacity': 1500 + attachment_param * 100,
#                 'frequency': max(6, 12 - attachment_param * 2),
#                 'travel_time': distance / 30,
#                 'distance': distance,
#                 'segment_order': 1
#             }
#         elif combined_importance > 0:
#             return {
#                 'transport_mode': TransportMode.BUS,
#                 'route_id': f"HB_{self._get_next_route_counter('hub_bus')}_SF{attachment_param}",
#                 'capacity': 800 + attachment_param * 50,
#                 'frequency': max(8, 15 - attachment_param),
#                 'travel_time': distance / 25,
#                 'distance': distance,
#                 'segment_order': 1
#             }
#         else:
#             return {
#                 'transport_mode': TransportMode.BUS,
#                 'route_id': f"BUS_{self._get_next_route_counter('bus')}_SF{attachment_param}",
#                 'capacity': 600,
#                 'frequency': 15,
#                 'travel_time': distance / 20,
#                 'distance': distance,
#                 'segment_order': 1
#             }
        
#     # ===== BASE SYDNEY NETWORKS =====
#     def _generate_base_sydney(self, connectivity_level: int,
#                             enhance_connectivity: bool = True) -> nx.Graph:
#         """
#         Generate enhanced Sydney network with regular spacing.
        
#         Args:
#             connectivity_level: Level of connectivity enhancement (4-8)
#             enhance_connectivity: Whether to add extra connections
        
#         Returns:
#             NetworkX graph with enhanced Sydney topology
#         """
        
#         print(f"   Generating base Sydney network (connectivity={connectivity_level})")
        
#         # Start with base network
#         graph = self.base_network.graph.copy()

        
#         if enhance_connectivity:
#             # Add extra connections based on connectivity level
#             node_positions = {node_id: node_data.coordinates 
#                             for node_id, node_data in self.nodes.items()}
#             distances = self._calculate_distance_matrix(node_positions)
            
#             # For each node, ensure it has at least connectivity_level connections
#             for node in graph.nodes():
#                 current_degree = graph.degree(node)
#                 if current_degree < connectivity_level:
#                     # Find nearest unconnected nodes
#                     candidates = []
#                     for other_node in graph.nodes():
#                         if (node != other_node and 
#                             not graph.has_edge(node, other_node)):
#                             distance = distances[node][other_node]
#                             candidates.append((other_node, distance))
                    
#                     # Sort by distance and connect to nearest
#                     candidates.sort(key=lambda x: x[1])
#                     connections_needed = connectivity_level - current_degree
                    
#                     for other_node, _ in candidates[:connections_needed]:
#                         self._add_transport_edge(graph, node, other_node)
        
#         # Add topology metadata
#         graph.graph['topology_type'] = 'base_sydney'
#         graph.graph['connectivity_level'] = connectivity_level
#         graph.graph['enhanced'] = enhance_connectivity
        
#         print(f"   ✅ Base Sydney network: {graph.number_of_nodes()} nodes, "
#               f"{graph.number_of_edges()} edges")
        
#         return graph
    

#     def _is_major_hub(self, node: str) -> bool:
#         """Check if node is a major hub"""
#         return (node in self.nodes and 
#                 self.nodes[node].node_type == NodeType.MAJOR_HUB)

#     def _is_transport_hub(self, node: str) -> bool:
#         """Check if node is a transport hub"""
#         return (node in self.nodes and 
#                 self.nodes[node].node_type == NodeType.TRANSPORT_HUB)

#     def _get_node_type(self, node: str) -> NodeType:
#         """Get node type safely"""
#         if node in self.nodes:
#             return self.nodes[node].node_type
#         return NodeType.LOCAL_STATION  # Default type
    
#     # ===== GRID NETWORKS =====
#     def _generate_grid(self, connectivity: int,
#                   grid_width: int = 100, 
#                   grid_height: int = 80) -> nx.Graph:
#         """
#         Generate regular grid network with Sydney station name mapping.
        
#         Args:
#             connectivity: Grid connectivity (4, 6, or 8)
#             grid_width: Width of spatial grid
#             grid_height: Height of spatial grid
        
#         Returns:
#             NetworkX graph with grid topology using Sydney station names where possible
#         """
        
#         print(f"[DEBUG] Starting grid generation with connectivity={connectivity}")
    
#         # Create regular grid
#         if connectivity == 4:
#             # 4-connected grid (von Neumann neighborhood)
#             grid = nx.grid_2d_graph(85, 85)  # Reduced size for manageability
#         elif connectivity == 6:
#             # 6-connected hexagonal grid
#             grid = nx.hexagonal_lattice_graph(10, 8)
#         elif connectivity == 8:
#             # 8-connected grid (Moore neighborhood)
#             grid = nx.grid_2d_graph(85, 85)
#             # Add diagonal connections
#             for node in grid.nodes():
#                 x, y = node
#                 for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
#                     neighbor = (x + dx, y + dy)
#                     if neighbor in grid.nodes():
#                         grid.add_edge(node, neighbor)
#         else:
#             raise ValueError(f"Unsupported grid connectivity: {connectivity}")
        
#         # ===== NEW: CREATE SYDNEY STATION COORDINATE MAPPING =====
#         sydney_station_coords = {}
#         used_station_names = set()
        
#         # Extract Sydney station coordinates from base network
#         if hasattr(self, 'base_network') and self.base_network:
#             for node_id, node_data in self.base_network.nodes.items():
#                 # Convert coordinates to grid positions
#                 grid_x = int((node_data.coordinates[0] / 100) * grid_width)
#                 grid_y = int((node_data.coordinates[1] / 80) * grid_height)
                
#                 # Clamp to grid bounds
#                 grid_x = max(0, min(grid_x, grid_width - 1))
#                 grid_y = max(0, min(grid_y, grid_height - 1))
                
#                 grid_pos = (grid_x, grid_y)
#                 sydney_station_coords[grid_pos] = node_id
  
#         def find_nearest_sydney_station(grid_coords, tolerance=2):
#             """Find Sydney station name near grid coordinates"""
#             x, y = grid_coords
            
#             # Check exact match first
#             if grid_coords in sydney_station_coords:
#                 station_name = sydney_station_coords[grid_coords]
#                 if station_name not in used_station_names:
#                     used_station_names.add(station_name)
#                     return station_name
            
#             # Check nearby positions within tolerance
#             for dx in range(-tolerance, tolerance + 1):
#                 for dy in range(-tolerance, tolerance + 1):
#                     if dx == 0 and dy == 0:
#                         continue
#                     nearby_pos = (x + dx, y + dy)
#                     if nearby_pos in sydney_station_coords:
#                         station_name = sydney_station_coords[nearby_pos]
#                         if station_name not in used_station_names:
#                             used_station_names.add(station_name)
#                             return station_name
            
#             return None
        
#         # Convert to standard node IDs and add transport attributes
#         graph = nx.Graph()
#         node_mapping = {}
#         sydney_stations_mapped = 0
#         grid_nodes_created = 0
#         print(f"[DEBUG] Starting node addition loop, grid has {len(list(grid.nodes()))} nodes")
    
#         for i, grid_node in enumerate(grid.nodes()):
#             # Calculate coordinates within the grid bounds
#             if isinstance(grid_node, tuple) and len(grid_node) == 2:
#                 x_ratio = grid_node[0] / 84
#                 y_ratio = grid_node[1] / 84
#             else:
#                 x_ratio = i % 20 / 20
#                 y_ratio = i // 20 / 16
            
#             x_coord = x_ratio * grid_width
#             y_coord = y_ratio * grid_height
            
#             # Convert to integer grid position
#             grid_x = int(x_coord)
#             grid_y = int(y_coord)
#             grid_position = (grid_x, grid_y)
            
#             # ===== NEW: TRY TO MAP TO SYDNEY STATION NAME =====
#             sydney_station_name = find_nearest_sydney_station(grid_position, tolerance=3)
            
#             if sydney_station_name:
#                 # Use Sydney station name
#                 new_node_id = sydney_station_name
#                 sydney_stations_mapped += 1
#                 # print(f"   ✅ Grid node {grid_node} → Sydney station '{sydney_station_name}' at {grid_position}")
#             else:
#                 # Use generic grid ID
#                 new_node_id = f"grid_{i}"
#                 grid_nodes_created += 1
            
#             node_mapping[grid_node] = new_node_id
            
#             # Create network node with appropriate data
#             if sydney_station_name:
#                 # Use Sydney station data if available
#                 if hasattr(self, 'base_network') and sydney_station_name in self.base_network.nodes:
#                     sydney_node_data = self.base_network.nodes[sydney_station_name]
#                     node_data = NetworkNode(
#                         node_id=new_node_id,
#                         node_type=sydney_node_data.node_type,
#                         coordinates=(x_coord, y_coord),
#                         population_weight=sydney_node_data.population_weight,
#                         employment_weight=sydney_node_data.employment_weight,
#                         transport_modes=sydney_node_data.transport_modes,
#                         zone_name=sydney_node_data.zone_name
#                     )
#                 else:
#                     # Fallback node data
#                     node_data = NetworkNode(
#                         node_id=new_node_id,
#                         node_type=NodeType.TRANSPORT_HUB,
#                         coordinates=(x_coord, y_coord),
#                         population_weight=0.5,
#                         employment_weight=0.4,
#                         transport_modes=[TransportMode.TRAIN, TransportMode.BUS],
#                         zone_name=sydney_station_name
#                     )
#             else:
#                 # Generic grid node data
#                 node_data = NetworkNode(
#                     node_id=new_node_id,
#                     node_type=NodeType.LOCAL_STATION,
#                     coordinates=(x_coord, y_coord),
#                     population_weight=0.3,
#                     employment_weight=0.2,
#                     transport_modes=[TransportMode.BUS],
#                     zone_name=f"Grid_Zone_{i}"
#                 )

#             graph.add_node(new_node_id, **node_data.__dict__)
        
#         # Add edges with transport attributes
#         for edge in grid.edges():
#             node1 = node_mapping[edge[0]]
#             node2 = node_mapping[edge[1]]
#             self._add_transport_edge(graph, node1, node2)
        
#         # Add topology metadata
#         graph.graph['topology_type'] = 'grid'
#         graph.graph['connectivity'] = connectivity
#         graph.graph['grid_width'] = grid_width
#         graph.graph['grid_height'] = grid_height
        
#         print(f"   ✅ Grid network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
#         print(f"   📊 Station mapping: {sydney_stations_mapped} Sydney stations, {grid_nodes_created} generic grid nodes")
        
#         return graph
    
#     # ===== UTILITY METHODS =====
#     def _calculate_distance_matrix(self, node_positions: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
#         """Calculate pairwise distances between all nodes"""
#         distances = {}
#         for node1, pos1 in node_positions.items():
#             distances[node1] = {}
#             for node2, pos2 in node_positions.items():
#                 if node1 == node2:
#                     distances[node1][node2] = 0
#                 else:
#                     dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
#                     distances[node1][node2] = dist
#         return distances
    

#     def _add_transport_edge(self, graph: nx.Graph, node1: str, node2: str, 
#                        edge_type: str = 'regular', topology_type: str = None,
#                        parameter_value: float = None):
#         """Add edge with COMPLETE transport attributes - FIXED VERSION"""
    
#         # Check if edge already exists (preserve base network)
#         if graph.has_edge(node1, node2):
#             print(f"  Preserving existing edge {node1}-{node2}")
#             return
        
#         distance = self._calculate_distance(node1, node2)
        
#         # Topology-specific mode and route assignment
#         if edge_type in ['shortcut', 'degree_shortcut']:
#             mode, route_id, capacity, frequency = self._assign_shortcut_attributes(
#                 node1, node2, distance, topology_type, parameter_value
#             )
#         else:
#             mode, route_id, capacity, frequency = self._assign_regular_attributes(
#                 node1, node2, distance
#             )
        
#         # Calculate travel time based on mode
#         speed_map = {
#             'PREMIUM_TRAIN': 35,
#             'TRAIN': 30,
#             'EXPRESS_BUS': 25,
#             'BUS': 20
#         }
#         speed = speed_map.get(str(mode), 20)
#         travel_time = distance / speed * 60  # Convert to minutes
        
#         # CRITICAL FIX: Add edge with ALL required attributes
#         graph.add_edge(node1, node2,
#                     transport_mode=mode,
#                     route_id=route_id,
#                     capacity=capacity,
#                     frequency=frequency,
#                     travel_time=travel_time,
#                     distance=distance,
#                     segment_order=1,
#                     edge_type=edge_type)  # CRITICAL: Preserve edge_type
        
#         print(f"  ✅ Added {edge_type} edge: {node1}-{node2}, mode={mode}, route={route_id}")

#     def _assign_shortcut_attributes(self, node1, node2, distance, topology_type, parameter):
#         """Assign attributes for topology-specific shortcuts"""
        
#         if topology_type == 'small_world':
#             # Express services for small-world shortcuts
#             if distance > 30:
#                 self.route_counters['premium_train'] += 1
#                 return ('PREMIUM_TRAIN', 
#                     f"PT_{self.route_counters['premium_train']}_SW{int(parameter*100)}", 
#                     2000, 6)
#             else:
#                 self.route_counters['express_bus'] += 1
#                 return ('EXPRESS_BUS', 
#                     f"EX_{self.route_counters['express_bus']}_SW{int(parameter*100)}", 
#                     1000, 10)
        
#         elif topology_type == 'scale_free':
#             # Hub-based services for scale-free
#             self.route_counters['train'] += 1
#             return ('TRAIN', f"T{self.route_counters['train']}_SF{parameter}", 1800, 8)
        
#         elif topology_type == 'degree_constrained':
#             # Balanced services for degree-constrained
#             self.route_counters['bus'] += 1
#             return ('BUS', f"BUS_NEW{self.route_counters['bus']}_DC{parameter}", 600, 15)
        
#         # Default shortcut service
#         self.route_counters['bus'] += 1
#         return ('BUS', f"BUS_NEW{self.route_counters['bus']}", 600, 15)

#     # def _add_transport_edge(self, graph: nx.Graph, node1: str, node2: str, 
#     #                     edge_type: str = 'default', 
#     #                     topology_type: str = None,
#     #                     parameter_value: float = None):
#     #     """Add transport edge with proper mode assignment"""
        
#     #     # KISS: Check if edge already exists
#     #     if graph.has_edge(node1, node2):
#     #         return
        
#     #     # Calculate distance
#     #     distance = self._calculate_distance(node1, node2)
        
#     #     # KISS: Simple distance-based mode assignment
#     #     if distance > 25:
#     #         transport_mode = 'TRAIN'
#     #         route_prefix = 'T'
#     #         capacity = 1500
#     #         frequency = 10
#     #         speed = 30
#     #     elif distance > 15:
#     #         transport_mode = 'EXPRESS_BUS'
#     #         route_prefix = 'EB'
#     #         capacity = 1000
#     #         frequency = 12
#     #         speed = 25
#     #     else:
#     #         transport_mode = 'BUS'
#     #         route_prefix = 'BUS'
#     #         capacity = 600
#     #         frequency = 15
#     #         speed = 20
        
#     #     # Generate route ID based on topology
#     #     if topology_type:
#     #         route_id = f"{route_prefix}_{self._get_next_route_counter(transport_mode.lower())}_{topology_type.upper()[:2]}{int(parameter_value) if parameter_value else 0}"
#     #     else:
#     #         route_id = f"{route_prefix}_{self._get_next_route_counter(transport_mode.lower())}"
        
#     #     # Calculate travel time
#     #     travel_time = distance / speed * 60  # Convert to minutes
        
#     #     # Add edge
#     #     graph.add_edge(node1, node2,
#     #                 transport_mode=transport_mode,
#     #                 route_id=route_id,
#     #                 travel_time=travel_time,
#     #                 distance=distance,
#     #                 capacity=capacity,
#     #                 frequency=frequency,
#     #                 segment_order=1,
#     #                 edge_type=edge_type,
#     #                 weight=travel_time)

    
#     def _ensure_connectivity(self, graph: nx.Graph):
#         """Ensure network is fully connected"""
        
#         if not nx.is_connected(graph):
#             # Find connected components
#             components = list(nx.connected_components(graph))
            
#             # Connect components by adding edges between their closest nodes
#             main_component = max(components, key=len)
            
#             for component in components:
#                 if component == main_component:
#                     continue
                
#                 # Find closest pair of nodes between main component and this component
#                 min_distance = float('inf')
#                 best_connection = None
                
#                 for node1 in main_component:
#                     for node2 in component:
#                         if node1 in self.nodes and node2 in self.nodes:
#                             pos1 = self.nodes[node1].coordinates
#                             pos2 = self.nodes[node2].coordinates
#                             distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                            
#                             if distance < min_distance:
#                                 min_distance = distance
#                                 best_connection = (node1, node2)
                
#                 if best_connection:

#                     self._add_transport_edge(graph, best_connection[0], best_connection[1])
                    
#                     main_component.update(component)
    
#     def get_topology_properties(self, graph: nx.Graph) -> Dict[str, float]:
#         """Calculate standard topology properties for any network"""
        
#         properties = {
#             'num_nodes': graph.number_of_nodes(),
#             'num_edges': graph.number_of_edges(),
#             'is_connected': nx.is_connected(graph),
#             'density': nx.density(graph)
#         }
        
#         if nx.is_connected(graph):
#             properties['avg_shortest_path_length'] = nx.average_shortest_path_length(graph)
#             properties['diameter'] = nx.diameter(graph)
        
#         properties['avg_clustering'] = nx.average_clustering(graph)
        
#         # Degree statistics
#         degrees = dict(graph.degree())
#         degree_values = list(degrees.values())
#         properties['avg_degree'] = np.mean(degree_values)
#         properties['max_degree'] = max(degree_values)
#         properties['degree_variance'] = np.var(degree_values)
        
#         # Centrality measures
#         betweenness = nx.betweenness_centrality(graph)
#         properties['max_betweenness'] = max(betweenness.values())
#         properties['avg_betweenness'] = np.mean(list(betweenness.values()))
        
#         return properties

# # ===== CONVENIENCE FUNCTIONS =====
# def create_topology(topology_type: str, variation_parameter: Union[int, float], 
#                    base_network: SydneyNetworkTopology = None, **kwargs) -> nx.Graph:
#     """
#     Convenience function to create any topology type
    
#     Args:
#         topology_type: Type of topology ('degree_constrained', 'small_world', 'scale_free', etc.)
#         variation_parameter: Main parameter controlling the topology
#         base_network: Base network (optional)
#         **kwargs: Additional parameters
    
#     Returns:
#         Generated NetworkX graph
#     """
#     generator = UnifiedTopologyGenerator(base_network)
#     return generator.generate_topology(topology_type, variation_parameter, **kwargs)

# def get_available_topologies() -> List[str]:
#     """Get list of all available topology types"""
#     return [topology.value for topology in TopologyType]

# def get_topology_info(topology_type: str) -> Dict[str, str]:
#     """Get information about a specific topology type"""
    
#     info = {
#         'degree_constrained': {
#             'description': 'Networks where all nodes have approximately the same degree',
#             'parameter': 'Target degree (3-7)',
#             'use_case': 'Studying impact of uniform connectivity'
#         },
#         'small_world': {
#             'description': 'Watts-Strogatz networks with local clustering and shortcuts',
#             'parameter': 'Rewiring probability (0.0-1.0)',
#             'use_case': 'Studying balance between local and global connectivity'
#         },
#         'scale_free': {
#             'description': 'Barabási-Albert networks with hub-dominated structure',
#             'parameter': 'Number of edges per new node (1-5)',
#             'use_case': 'Studying impact of transport hubs and preferential attachment'
#         },
#         'base_sydney': {
#             'description': 'Enhanced Sydney network with regular spacing',
#             'parameter': 'Connectivity level (4-8)',
#             'use_case': 'Realistic Sydney transport network baseline'
#         },
#         'grid': {
#             'description': 'Regular grid networks with uniform spatial structure',
#             'parameter': 'Grid connectivity (4, 6, or 8)',
#             'use_case': 'Mathematical baseline for comparison'
#         }
#     }
    
#     return info.get(topology_type, {'description': 'Unknown topology type'})

# # ===== EXAMPLE USAGE =====
# if __name__ == "__main__":
#     print("UNIFIED TOPOLOGY GENERATOR - TESTING ALL TYPES")
#     print("=" * 60)
    
#     # Initialize generator
#     generator = UnifiedTopologyGenerator()
    
#     # Test all topology types
#     test_configs = [
#         ('degree_constrained', 4),
#         ('small_world', 0.1),
#         ('scale_free', 2),
#         ('base_sydney', 6),
#         ('grid', 6)
#     ]
    
#     for topology_type, param in test_configs:
#         print(f"\n🧪 Testing {topology_type} (param: {param})")
        
#         try:
#             graph = generator.generate_topology(topology_type, param)
#             properties = generator.get_topology_properties(graph)
            
#             print(f"   📊 Properties:")
#             print(f"   - Nodes: {properties['num_nodes']}")
#             print(f"   - Edges: {properties['num_edges']}")
#             print(f"   - Connected: {properties['is_connected']}")
#             print(f"   - Avg degree: {properties['avg_degree']:.2f}")
#             print(f"   - Clustering: {properties['avg_clustering']:.3f}")
            
#         except Exception as e:
#             print(f"❌ Failed in test_configs: {e}")
    
#     print(f"\n✅ Unified topology generator testing completed!")
#     print(f"📋 Available topologies: {get_available_topologies()}")

# unified_topology_generator.py
"""
UNIFIED TOPOLOGY GENERATOR - ALL NETWORK TYPES IN ONE FILE (FIXED VERSION)
Contains all topology generation methods: degree-constrained, small-world, scale-free, base Sydney
Easy switching between topology types with standardized interface.
Fixed: Time units (steps not minutes), realistic frequencies (0.5-3 steps), consistent attribute maps
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from collections import Counter
import math

# Import base network definitions
from topology.network_topology import SydneyNetworkTopology, NetworkNode, NetworkEdge, NodeType, TransportMode

class TopologyType(Enum):
    """Enumeration of all supported topology types"""
    DEGREE_CONSTRAINED = "degree_constrained"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    BASE_SYDNEY = "base_sydney"
    GRID = "grid"

class UnifiedTopologyGenerator:
    """
    Unified generator for all network topology types.
    Single class handles all topology generation with consistent interface.
    FIXED VERSION: Correct time units, realistic frequencies, missing methods added.
    """
    
    def __init__(self, base_network: SydneyNetworkTopology = None):
        """Initialize with base Sydney network and consistent attribute maps"""
        if base_network is None:
            base_network = SydneyNetworkTopology()
            base_network.initialize_base_sydney_network()
        
        self.base_network = base_network
        self.nodes = self.base_network.nodes  # Fix for self.nodes references
        self._validate_base_network()
    
        # Route counters for generating unique IDs
        self.route_counters = {
            'train': 100,
            'premium_train': 200, 
            'express_bus': 300,
            'hub_bus': 400,
            'bus': 1000
        }
        
        # CONSISTENT TIME FACTORS (in steps per km) - FIXED!
        self.time_factor_map = {
            'PREMIUM_TRAIN': 0.025,  # 40 km per step = 240 km/h (bullet train)
            'TRAIN': 0.050,          # 20 km per step = 120 km/h (fast metro)
            'EXPRESS_BUS': 0.067,    # 15 km per step = 90 km/h (BRT)  
            'BUS': 0.100             # 10 km per step = 60 km/h (express bus)
        }
        #
        # self.time_factor_map = {
        #     'PREMIUM_TRAIN': 0.000001,  # 40 km per step = 240 km/h (bullet train)
        #     'TRAIN': 0.0000001,          # 20 km per step = 120 km/h (fast metro)
        #     'EXPRESS_BUS': 0.000001,    # 15 km per step = 90 km/h (BRT)  
        #     'BUS': 0.000001           # 10 km per step = 60 km/h (express bus)
        # }
        #  CONSISTENT FREQUENCY RANGES (0.5-3 steps) - FIXED!
        self.frequency_map = {
            'PREMIUM_TRAIN': 0.5,    # Every 5 minutes (premium)
            'TRAIN': 1.0,            # Every 10 minutes  
            'EXPRESS_BUS': 1.5,      # Every 15 minutes
            'BUS': 2.5               # Every 25 minutes
        }
        
        print(f"✅ UnifiedTopologyGenerator initialized with {base_network.graph.number_of_edges()} base edges")

    def _validate_base_network(self):
        """Validate base network has essential Sydney routes"""
        required_routes = ["T1_WESTERN", "T4_ILLAWARRA", "T8_AIRPORT", "BUS_380"]
        
        found_routes = set()
        for u, v, edge_data in self.base_network.graph.edges(data=True):
            if 'route_id' in edge_data:
                for route in required_routes:
                    if route in edge_data['route_id']:
                        found_routes.add(route)
        
        missing = set(required_routes) - found_routes
        if missing:
            raise ValueError(f"Base network missing required routes: {missing}")
        
        print(f"✅ Base network validation passed - all required routes present")

    def _get_next_route_counter(self, route_type: str) -> int:
        """Get next route counter - SIMPLE"""
        # KISS: Map any variant to base type
        if 'train' in route_type.lower():
            key = 'train'
        elif 'express' in route_type.lower():
            key = 'express_bus'
        else:
            key = 'bus'
        
        # Ensure key exists
        if key not in self.route_counters:
            self.route_counters[key] = 1000
        
        self.route_counters[key] += 1
        return self.route_counters[key]

    def _calculate_distance(self, node1: str, node2: str) -> float:
        """Calculate distance between nodes"""
        if node1 not in self.base_network.nodes or node2 not in self.base_network.nodes:
            print(f"  ❌Warning: Node {node1} or {node2} not found in base network. Using default distance.")
            return 50.0  # Default distance
        
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

    def _is_hub_connection(self, node1: str, node2: str) -> bool:
        """Check if connection involves at least one major or transport hub"""
        if node1 not in self.nodes or node2 not in self.nodes:
            return False
        
        node1_type = self.nodes[node1].node_type
        node2_type = self.nodes[node2].node_type
        
        hub_types = [NodeType.MAJOR_HUB, NodeType.TRANSPORT_HUB]
        return node1_type in hub_types or node2_type in hub_types

    def _assign_regular_attributes(self, node1: str, node2: str, distance: float) -> tuple:
        """Assign mode, route_id, capacity, frequency for regular (non-shortcut) edges - FIXED!"""
        
        # Check if nodes are major hubs
        is_hub_connection = self._is_hub_connection(node1, node2)
        
        # Distance-based mode assignment for regular edges
        if distance > 25 and is_hub_connection:
            # Long distance between hubs = TRAIN
            mode = 'TRAIN'
            self.route_counters['train'] += 1
            route_id = f"T{self.route_counters['train']}_REG"
            capacity = 1500
            frequency = self.frequency_map['TRAIN']
            
        elif distance > 15:
            # Medium distance = EXPRESS_BUS
            mode = 'EXPRESS_BUS'
            self.route_counters['express_bus'] += 1
            route_id = f"EB_{self.route_counters['express_bus']}_REG"
            capacity = 800
            frequency = self.frequency_map['EXPRESS_BUS']
            
        else:
            # Short distance = regular BUS
            mode = 'BUS'
            self.route_counters['bus'] += 1
            route_id = f"BUS_{self.route_counters['bus']}_REG"
            capacity = 600
            frequency = self.frequency_map['BUS']
        
        return mode, route_id, capacity, frequency

    def _assign_shortcut_attributes(self, node1, node2, distance, topology_type, parameter):
        """Assign attributes for topology-specific shortcuts - FIXED!"""
        
        if topology_type == 'small_world':
            # Express services for small-world shortcuts
            if distance > 30:
                mode = 'PREMIUM_TRAIN'
                self.route_counters['premium_train'] += 1
                return (mode, f"PT_{self.route_counters['premium_train']}_SW{int(parameter*100)}", 
                       2000, self.frequency_map[mode])  # USE THE MAP!
            else:
                mode = 'EXPRESS_BUS' 
                self.route_counters['express_bus'] += 1
                return (mode, f"EX_{self.route_counters['express_bus']}_SW{int(parameter*100)}", 
                       1000, self.frequency_map[mode])  # USE THE MAP!
        
        elif topology_type == 'scale_free':
            # Hub-based services for scale-free - frequency scales with parameter
            mode = 'TRAIN'
            self.route_counters['train'] += 1
            hub_frequency = max(0.5, self.frequency_map[mode] - parameter * 0.3)  # Parameter scaling
            return (mode, f"T{self.route_counters['train']}_SF{parameter}", 1800, hub_frequency)
        
        elif topology_type == 'degree_constrained':
            # Balanced services for degree-constrained - frequency scales with parameter
            if distance > 20:
                mode = 'TRAIN'
                self.route_counters['train'] += 1
                equity_frequency = max(0.5, 3.0 - parameter * 0.4)  # 0.5-3 steps for equity
                return (mode, f"T{self.route_counters['train']}_DC{parameter}", 1500, equity_frequency)
            else:
                mode = 'BUS'
                self.route_counters['bus'] += 1
                equity_frequency = max(1.0, 3.0 - parameter * 0.3)  # 1-3 steps
                return (mode, f"BUS_NEW{self.route_counters['bus']}_DC{parameter}", 600, equity_frequency)
        
        # Default shortcut service
        mode = 'BUS'
        self.route_counters['bus'] += 1
        return (mode, f"BUS_NEW{self.route_counters['bus']}", 600, self.frequency_map[mode])

    def _add_transport_edge(self, graph: nx.Graph, node1: str, node2: str, 
                       edge_type: str = 'regular', topology_type: str = None,
                       parameter_value: float = None):
        """Add edge with COMPLETE transport attributes - FIXED VERSION"""
        print(f"  _add_transport_edge Adding {edge_type} edge: {node1}-{node2} (topology={topology_type}, param={parameter_value})")
        # Check if edge already exists (preserve base network)
        if graph.has_edge(node1, node2):
            print(f"  Preserving existing edge {node1}-{node2}")
            return
        
        distance = self._calculate_distance(node1, node2)
        
        # Topology-specific mode and route assignment
        if edge_type in ['shortcut', 'degree_shortcut']:
            mode, route_id, capacity, frequency = self._assign_shortcut_attributes(
                node1, node2, distance, topology_type, parameter_value
            )
        else:
            mode, route_id, capacity, frequency = self._assign_regular_attributes(
                node1, node2, distance
            )
        
        # Calculate travel time using CLASS ATTRIBUTE (FIXED!)
        time_factor = self.time_factor_map.get(str(mode), self.time_factor_map['BUS'])
        print(f"  [DEBUG] Using time factor {time_factor} for mode {mode}")
        travel_time = distance * time_factor  # In steps, not minutes!
        
        # Add edge with ALL required attributes
        graph.add_edge(node1, node2,
                    transport_mode=mode,
                    route_id=route_id,
                    capacity=capacity,
                    frequency=frequency,
                    travel_time=travel_time,
                    distance=distance,
                    segment_order=1,
                    edge_type=edge_type)
        
        print(f"  ✅ Added {edge_type} edge: {node1}-{node2}, mode={mode}, route={route_id}")

    # ===== TOPOLOGY-SPECIFIC SHORTCUT CREATION METHODS - ALL FIXED =====

    def _create_degree_constrained_edge_attributes(self, node1: str, node2: str, 
                                               degree_param: int) -> dict:
        """Create edge attributes for degree-constrained shortcuts - FIXED!"""
        distance = self._calculate_distance(node1, node2)
        print(f"  [DEBUG] Creating degree-constrained edge {node1}-{node2} with distance {distance} and degree_param {degree_param}")
        # Parameter-dependent thresholds
        train_threshold = 100 * (0.2 + degree_param * 0.05)
        express_threshold = train_threshold * 0.6
        
        if distance > train_threshold:
            mode = 'TRAIN'
            print(f"time_factor: {self.time_factor_map[mode]}")
            print(f"travel_time: {distance * self.time_factor_map[mode]}")
            return {
                'transport_mode': TransportMode.TRAIN,
                'route_id': f"T_{self._get_next_route_counter('train')}_EQ{degree_param}",
                'capacity': 1500 + (degree_param - 3) * 100,
                'frequency': max(0.5, self.frequency_map[mode] - degree_param * 0.1),  # Parameter scaling
                'travel_time': distance * self.time_factor_map[mode],  # USE THE MAP!
                'distance': distance,
                'segment_order': 1
            }
        elif distance > express_threshold:
            mode = 'EXPRESS_BUS'
            print(f"time_factor: {self.time_factor_map[mode]}")
            print(f"travel_time: {distance * self.time_factor_map[mode]}")
            return {
                'transport_mode': TransportMode.BUS,
                'route_id': f"EB_{self._get_next_route_counter('express_bus')}_DC{degree_param}",
                'capacity': 800 + (degree_param - 3) * 50,
                'frequency': max(1.0, self.frequency_map[mode] - degree_param * 0.1),  # Parameter scaling
                'travel_time': distance * self.time_factor_map[mode],  # USE THE MAP!
                'distance': distance,
                'segment_order': 1
            }
        else:
            mode = 'BUS'
            print(f"time_factor: {self.time_factor_map[mode]}")
            print(f"travel_time: {distance * self.time_factor_map[mode]}")
            return {
                'transport_mode': TransportMode.BUS,
                'route_id': f"BUS_{self._get_next_route_counter('bus')}_DC{degree_param}",
                'capacity': 600,
                'frequency': self.frequency_map[mode],      # USE THE MAP!
                'travel_time': distance * self.time_factor_map[mode],  # USE THE MAP!
                'distance': distance,
                'segment_order': 1
            }

    def _create_small_world_edge_attributes(self, node1: str, node2: str, 
                                        rewiring_param: float) -> dict:
        """Create edge attributes for small-world shortcuts - FIXED!"""
        distance = self._calculate_distance(node1, node2)
        
        # Parameter-dependent service quality
        premium_threshold = 100 * (0.3 + rewiring_param * 0.4)
        express_threshold = premium_threshold * 0.7
        
        if distance > premium_threshold:
            mode = 'PREMIUM_TRAIN'
            print(f"time_factor: {self.time_factor_map[mode]}")
            print(f"travel_time: {distance * self.time_factor_map[mode]}")
            return {
                'transport_mode': TransportMode.TRAIN,
                'route_id': f"PT_{self._get_next_route_counter('premium_train')}_SW{int(rewiring_param*100)}",
                'capacity': 2000,
                'frequency': self.frequency_map[mode],      # USE THE MAP!
                'travel_time': distance * self.time_factor_map[mode],  # USE THE MAP!
                'distance': distance,
                'segment_order': 1
            }
        elif distance > express_threshold:
            mode = 'EXPRESS_BUS'
            print(f"time_factor: {self.time_factor_map[mode]}")
            print(f"travel_time: {distance * self.time_factor_map[mode]}")
            return {
                'transport_mode': TransportMode.BUS,
                'route_id': f"EX_{self._get_next_route_counter('express_bus')}_SW{int(rewiring_param*100)}",
                'capacity': 1000,
                'frequency': self.frequency_map[mode],      # USE THE MAP!
                'travel_time': distance * self.time_factor_map[mode],  # USE THE MAP!
                'distance': distance,
                'segment_order': 1
            }
        else:
            mode = 'BUS'
            print(f"time_factor: {self.time_factor_map[mode]}")
            print(f"travel_time: {distance * self.time_factor_map[mode]}")
            return {
                'transport_mode': TransportMode.BUS,
                'route_id': f"BUS_{self._get_next_route_counter('bus')}_SW{int(rewiring_param*100)}",
                'capacity': 600,
                'frequency': self.frequency_map[mode],      # USE THE MAP!
                'travel_time': distance * self.time_factor_map[mode],  # USE THE MAP!
                'distance': distance,
                'segment_order': 1
            }

    def _create_scale_free_edge_attributes(self, node1: str, node2: str, 
                                       attachment_param: int) -> dict:
        """Create edge attributes for scale-free shortcuts - FIXED!"""
        distance = self._calculate_distance(node1, node2)
        
        # Hub-based service assignment
        is_hub_conn = self._is_hub_connection(node1, node2)
        
        if is_hub_conn and distance > 20:
            # Hub-to-hub connections get premium service
            mode = 'PREMIUM_TRAIN'
            frequency = max(0.5, self.frequency_map[mode] - attachment_param * 0.1)
            return {
                'transport_mode': TransportMode.TRAIN,
                'route_id': f"PT_{self._get_next_route_counter('premium_train')}_SF{attachment_param}",
                'capacity': 2000,
                'frequency': frequency,  # Parameter scaling
                'travel_time': distance * self.time_factor_map[mode],  # USE THE MAP!
                'distance': distance,
                'segment_order': 1
            }
        elif is_hub_conn:
            # Hub connections get train service
            mode = 'TRAIN'
            frequency = max(0.5, self.frequency_map[mode] - attachment_param * 0.2)
            return {
                'transport_mode': TransportMode.TRAIN,
                'route_id': f"T_{self._get_next_route_counter('train')}_SF{attachment_param}",
                'capacity': 1800,
                'frequency': frequency,  # Parameter scaling
                'travel_time': distance * self.time_factor_map[mode],  # USE THE MAP!
                'distance': distance,
                'segment_order': 1
            }
        else:
            # Regular connections get bus service
            mode = 'BUS'
            print(f"time_factor: {self.time_factor_map[mode]}")
            print(f"travel_time: {distance * self.time_factor_map[mode]}")
            return {
                'transport_mode': TransportMode.BUS,
                'route_id': f"BUS_{self._get_next_route_counter('bus')}_SF{attachment_param}",
                'capacity': 600,
                'frequency': self.frequency_map[mode],      # USE THE MAP!
                'travel_time': distance * self.time_factor_map[mode],  # USE THE MAP!
                'distance': distance,
                'segment_order': 1
            }

    # ===== MAIN TOPOLOGY GENERATION METHODS =====

    def generate_topology(self, 
                         topology_type: Union[str, TopologyType],
                         variation_parameter: Union[int, float],
                         **kwargs) -> nx.Graph:
        """
        Universal topology generation method - handles ALL topology types
        
        Args:
            topology_type: Type of topology to generate
            variation_parameter: Main parameter controlling topology variation
            **kwargs: Additional topology-specific parameters
        
        Returns:
            NetworkX graph with specified topology
        """
        
        # Convert string to enum if needed
        if isinstance(topology_type, str):
            topology_type = TopologyType(topology_type)
        
        print(f"🌐 Generating {topology_type.value} topology (param: {variation_parameter})")
        
        # Route to appropriate generation method
        if topology_type == TopologyType.DEGREE_CONSTRAINED:
            return self._generate_degree_constrained(int(variation_parameter), **kwargs)
        
        elif topology_type == TopologyType.SMALL_WORLD:
            return self._generate_small_world(float(variation_parameter), **kwargs)
        
        elif topology_type == TopologyType.SCALE_FREE:
            return self._generate_scale_free(int(variation_parameter), **kwargs)
        
        elif topology_type == TopologyType.BASE_SYDNEY:
            return self._generate_base_sydney(int(variation_parameter), **kwargs)
        
        elif topology_type == TopologyType.GRID:
            return self._generate_grid(int(variation_parameter), **kwargs)
        
        else:
            raise ValueError(f"Unsupported topology type: {topology_type}")

    def _generate_degree_constrained(self, target_degree: int, **kwargs) -> nx.Graph:
        """Generate degree-constrained network - FIXED with two-layer architecture"""
        
        print(f"🔧 Generating degree-constrained network (target_degree={target_degree})")
        
        # LAYER 1: Start with Sydney base network (always preserved)
        graph = self.base_network.graph.copy()
        base_edges_count = graph.number_of_edges()
        print(f"   Layer 1 (Base): {base_edges_count} Sydney edges preserved")
        
        # LAYER 2: Add shortcuts to achieve target degree
        shortcuts_added = 0
        
        # Calculate current degrees
        node_degrees = dict(graph.degree())
        
        for node in graph.nodes():
            current_degree = node_degrees[node]
            deficit = target_degree - current_degree
            
            if deficit > 0:
                # Find suitable targets for new edges
                candidates = [n for n in graph.nodes() 
                            if n != node and not graph.has_edge(node, n)]
                
                # Sort by distance for geographic realism
                candidates.sort(key=lambda n: self._calculate_distance(node, n))
                
                # Add edges up to target degree
                added_for_node = 0
                for candidate in candidates:
                    if added_for_node >= deficit:
                        break
                    
                    # Check if candidate also needs connections
                    candidate_deficit = target_degree - graph.degree(candidate)
                    if candidate_deficit > 0:
                        # Create degree-constrained shortcut
                        edge_attrs = self._create_degree_constrained_edge_attributes(
                            node, candidate, target_degree
                        )
                        
                        graph.add_edge(node, candidate, **edge_attrs)
                        shortcuts_added += 1
                        added_for_node += 1
                        
                        print(f"   Added degree shortcut: {node} -> {candidate} ({edge_attrs['route_id']})")
        
        total_edges = graph.number_of_edges()
        print(f"   Layer 2 (Shortcuts): {shortcuts_added} degree shortcuts added")
        print(f"   ✅ Total network: {total_edges} edges")
        
        return graph

    def _generate_small_world(self, rewiring_probability: float, **kwargs) -> nx.Graph:
        """Generate small-world network preserving Sydney base - FIXED!"""
        print(f"🗺️ Generating small-world topology (p={rewiring_probability})")
        
        # LAYER 1: Start with Sydney base network
        graph = self.base_network.graph.copy()
        base_edges_count = graph.number_of_edges()
        print(f"   Layer 1 (Base): {base_edges_count} Sydney edges preserved")
        
        # LAYER 2: Add small-world shortcuts through strategic rewiring
        shortcuts_added = self._add_small_world_shortcuts(graph, rewiring_probability)
        
        total_edges = graph.number_of_edges()
        print(f"   Layer 2 (Shortcuts): {shortcuts_added} shortcuts added")
        print(f"   ✅ Total network: {total_edges} edges")
        
        return graph

    def _add_small_world_shortcuts(self, graph: nx.Graph, rewiring_prob: float) -> int:
        """Add small-world shortcuts without removing base Sydney routes - FIXED!"""
        shortcuts_added = 0
        
        # Calculate how many shortcuts to add based on rewiring probability
        total_possible_shortcuts = len(list(graph.nodes())) * 2  # Approximate
        target_shortcuts = int(total_possible_shortcuts * rewiring_prob)
        
        all_nodes = list(graph.nodes())
        
        for _ in range(target_shortcuts):
            # Select random nodes for shortcut
            node1 = random.choice(all_nodes)
            node2 = random.choice(all_nodes)
            
            if (node1 != node2 and 
                not graph.has_edge(node1, node2) and
                self._is_shortcut_viable(node1, node2, graph)):
                
                # Create shortcut edge
                edge_attrs = self._create_small_world_edge_attributes(
                    node1, node2, rewiring_prob
                )
                
                edge_attrs = self._create_small_world_edge_attributes(node1, node2, rewiring_prob)
                graph.add_edge(node1, node2, **edge_attrs)
                shortcuts_added += 1
                print(f"   Added shortcut: {node1} -> {node2} ({edge_attrs['route_id']})")
        
        return shortcuts_added

    def _is_shortcut_viable(self, node1: str, node2: str, graph: nx.Graph) -> bool:
        """Check if shortcut is geographically viable"""
        distance = self._calculate_distance(node1, node2)
        return 5 < distance < 80  # Reasonable shortcut distance range

    def _generate_scale_free(self, attachment_parameter: int, **kwargs) -> nx.Graph:
        """Generate scale-free network with preferential attachment - FIXED!"""
        print(f"📈 Generating scale-free topology (m={attachment_parameter})")
        
        # LAYER 1: Start with Sydney base network
        graph = self.base_network.graph.copy()
        base_edges_count = graph.number_of_edges()
        print(f"   Layer 1 (Base): {base_edges_count} Sydney edges preserved")
        
        # LAYER 2: Add scale-free shortcuts using preferential attachment
        shortcuts_added = self._add_scale_free_shortcuts(graph, attachment_parameter)
        
        total_edges = graph.number_of_edges()
        print(f"   Layer 2 (Shortcuts): {shortcuts_added} scale-free shortcuts added")
        print(f"   ✅ Total network: {total_edges} edges")
        
        return graph

    def _add_scale_free_shortcuts(self, graph: nx.Graph, attachment_param: int) -> int:
        """Add scale-free shortcuts using preferential attachment - FIXED!"""
        shortcuts_added = 0
        nodes = list(graph.nodes())
        
        # Calculate node degrees for preferential attachment
        degrees = dict(graph.degree())
        
        # Add multiple rounds of preferential attachment
        for round_num in range(attachment_param):
            # Select high-degree nodes with higher probability
            total_degree = sum(degrees.values())
            
            for node in nodes:
                if shortcuts_added >= attachment_param * 5:  # Limit total shortcuts
                    break
                
                # Preferential attachment probability
                prob = degrees[node] / total_degree if total_degree > 0 else 1.0 / len(nodes)
                
                if random.random() < prob:
                    # Find candidate for new connection
                    candidates = [n for n in nodes 
                                if n != node and not graph.has_edge(node, n)]
                    
                    if candidates:
                        # Prefer connecting to other high-degree nodes
                        candidates.sort(key=lambda n: degrees[n], reverse=True)
                        target = candidates[0]
                        
                        # Create scale-free shortcut
                        edge_attrs = self._create_scale_free_edge_attributes(
                            node, target, attachment_param
                        )
                        
                        edge_attrs = self._create_scale_free_edge_attributes(node, target, attachment_param)
                        graph.add_edge(node, target, **edge_attrs)
                        
                        # Update degrees
                        degrees[node] += 1
                        degrees[target] += 1
                        shortcuts_added += 1
                        
                        print(f"   Added scale-free shortcut: {node} -> {target} ({edge_attrs['route_id']})")
        
        return shortcuts_added

    def _generate_base_sydney(self, connectivity_level: int, **kwargs) -> nx.Graph:
        """Return base Sydney network (no shortcuts)"""
        print(f"🏙️ Generating base Sydney network (connectivity={connectivity_level})")
        return self.base_network.graph.copy()

    def _generate_grid(self, connectivity_level: int, **kwargs) -> nx.Graph:
        """Generate grid network for comparison"""
        print(f"🔲 Generating grid network (connectivity={connectivity_level})")
        # For now, return base network - can be extended later
        return self.base_network.graph.copy()

    # ===== UTILITY METHODS =====
    def _calculate_distance_matrix(self, node_positions: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate pairwise distances between all nodes"""
        distances = {}
        for node1, pos1 in node_positions.items():
            distances[node1] = {}
            for node2, pos2 in node_positions.items():
                if node1 == node2:
                    distances[node1][node2] = 0
                else:
                    dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    distances[node1][node2] = dist
        return distances

    def get_topology_properties(self, graph: nx.Graph) -> Dict:
        """Calculate network topology properties"""
        return {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
            'is_connected': nx.is_connected(graph),
            'avg_clustering': nx.average_clustering(graph) if graph.number_of_nodes() > 0 else 0,
            'diameter': nx.diameter(graph) if nx.is_connected(graph) and graph.number_of_nodes() > 1 else None
        }

# ===== TESTING FUNCTION =====
def test_unified_topology_generator():
    """Test all topology generation methods"""
    print("\n🧪 Testing UnifiedTopologyGenerator with FIXED frequencies and time units")
    
    generator = UnifiedTopologyGenerator()
    
    # Test all topology types
    test_configs = [
        ('degree_constrained', 4),
        ('small_world', 0.1),
        ('scale_free', 2),
        ('base_sydney', 1),
    ]
    
    for topology_type, param in test_configs:
        print(f"\n🧪 Testing {topology_type} (param: {param})")
        
        try:
            graph = generator.generate_topology(topology_type, param)
            properties = generator.get_topology_properties(graph)
            
            print(f"   📊 Properties:")
            print(f"   - Nodes: {properties['num_nodes']}")
            print(f"   - Edges: {properties['num_edges']}")
            print(f"   - Connected: {properties['is_connected']}")
            print(f"   - Avg degree: {properties['avg_degree']:.2f}")
            print(f"   - Clustering: {properties['avg_clustering']:.3f}")
            
            # Check a few edges for correct attributes
            sample_edges = list(graph.edges(data=True))[:3]
            for u, v, data in sample_edges:
                print(f"   - Edge {u}-{v}: freq={data.get('frequency', 'missing')}, time={data.get('travel_time', 'missing'):.3f}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Unified topology generator testing completed!")

if __name__ == "__main__":
    test_unified_topology_generator()