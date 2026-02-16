# unified_topology_generator.py
"""
UNIFIED TOPOLOGY GENERATOR - ALL NETWORK TYPES IN ONE FILE
Contains all topology generation methods: degree-constrained, small-world, scale-free, base Sydney
Easy switching between topology types with standardized interface.
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
    """
    def __init__(self, base_network: SydneyNetworkTopology = None):
        """
        Initialize with base Sydney network
        
        Args:
            base_network: Base Sydney network topology. If None, creates default.
        """
        if base_network is None:
            self.base_network = SydneyNetworkTopology()
            self.base_network.initialize_base_sydney_network()
        else:
            self.base_network = base_network
        
        self.base_graph = self.base_network.graph
        self.nodes = self.base_network.nodes
        self.edges = self.base_network.edges
        self.route_counter = {'train': 200, 'bus': 2000}  # Different range for small-world
    
        print(f"🏗️ Unified topology generator initialized")
        print(f"   Base network: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def _add_transport_edge(self, graph: nx.Graph, node1: str, node2: str, 
                          edge_type: str = 'regular'):
        """Add transport edge preserving Sydney route structure - FIXED VERSION"""
        print(f"   Adding transport edge: {node1} <-> {node2} (type={edge_type})")
        # 🎯 CRITICAL FIX: Check for existing Sydney edge first
        if hasattr(self, 'base_network') and self.base_network.graph.has_edge(node1, node2):
            # Use original Sydney edge data completely
            original_data = self.base_network.graph[node1][node2]
            graph.add_edge(node1, node2, **original_data)
            return
        
        # Calculate distance
        pos1 = self.nodes[node1].coordinates if node1 in self.nodes else (0, 0)
        pos2 = self.nodes[node2].coordinates if node2 in self.nodes else (0, 0)
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # Determine transport mode and create Sydney-style route ID
        if edge_type == 'shortcut':
            # Small-world shortcuts are typically fast buses
            transport_mode = TransportMode.BUS
            travel_time = distance / 30  # Faster for shortcuts
            capacity = 800
            frequency = 15
            
            # 🎯 Create Sydney-style express route ID for shortcuts
            self.route_counter['bus'] += 1
            route_id = f"BUS_E{self.route_counter['bus']}"  # BUS_E2001, BUS_E2002, etc.
            
        elif distance > 25:
            # Long-distance regular connections = train
            transport_mode = TransportMode.TRAIN
            travel_time = distance / 25
            capacity = 1500
            frequency = 12
            
            # 🎯 Create Sydney-style train route ID
            self.route_counter['train'] += 1
            route_id = f"T{self.route_counter['train']}_SW"  # T201_SW, T202_SW, etc.
            
        else:
            # Regular bus connections
            transport_mode = TransportMode.BUS
            travel_time = distance / 20
            capacity = 600
            frequency = 10
            
            # 🎯 Create Sydney-style bus route ID
            self.route_counter['bus'] += 1
            route_id = f"BUS_{self.route_counter['bus']}"  # BUS_2001, BUS_2002, etc.
        
        # Add edge with Sydney-compatible attributes
        graph.add_edge(node1, node2,
                      transport_mode=transport_mode,
                      travel_time=max(0.5, travel_time),
                      distance=distance,
                      capacity=capacity,
                      frequency=frequency,
                      route_id=route_id,  # 🎯 KEY FIX!
                      segment_order=1,
                      edge_type=edge_type,
                      weight=travel_time)
        print(f"[DEBUG success] Edge added: {node1} <-> {node2} (route_id={route_id}, edge_type={edge_type})")

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
    
    # ===== DEGREE-CONSTRAINED NETWORKS =====
    def _generate_degree_constrained(self, target_degree: int, 
                                   preserve_geography: bool = False,
                                   preserve_major_hubs: bool = True) -> nx.Graph:
        """
        Generate degree-constrained network where all nodes have approximately the same degree.
        
        Args:
            target_degree: Target degree for all nodes
            preserve_geography: Whether to respect geographic distances
            preserve_major_hubs: Whether to preserve major transport hubs
        
        Returns:
            NetworkX graph with degree-constrained topology
        """
        
        print(f"   Generating degree-{target_degree} constrained network")
        
        # Start with base network structure
        graph = nx.Graph()
        
        # Add all nodes from base network
        for node_id, node_data in self.nodes.items():
            graph.add_node(node_id, **node_data.__dict__)
        
        # Calculate geographic distances if preserving geography
        if preserve_geography:
            node_positions = {node_id: node_data.coordinates 
                            for node_id, node_data in self.nodes.items()}
            distances = self._calculate_distance_matrix(node_positions)
        
        # Generate degree-constrained connections
        nodes_list = list(graph.nodes())
        connections_count = {node: 0 for node in nodes_list}
        
        # If preserving major hubs, give them slightly higher degree
        hub_bonus = 2 if preserve_major_hubs else 0
        
        for node in nodes_list:
            node_data = self.nodes[node]
            current_degree = connections_count[node]
            
            # Determine target degree for this node
            if preserve_major_hubs and node_data.node_type == NodeType.MAJOR_HUB:
                node_target_degree = target_degree + hub_bonus
            else:
                node_target_degree = target_degree
            
            # Connect to nearest unconnected nodes
            candidates = []
            for other_node in nodes_list:
                if (node != other_node and 
                    not graph.has_edge(node, other_node) and
                    connections_count[other_node] < target_degree + hub_bonus):
                    
                    if preserve_geography:
                        distance = distances[node][other_node]
                        candidates.append((other_node, distance))
                    else:
                        candidates.append((other_node, random.random()))
            
            # Sort by distance/preference and connect to closest
            candidates.sort(key=lambda x: x[1])
            connections_needed = node_target_degree - current_degree
            
            for other_node, _ in candidates[:connections_needed]:
                if connections_count[other_node] < target_degree + hub_bonus:
                    self._add_transport_edge(graph, node, other_node)
                    connections_count[node] += 1
                    connections_count[other_node] += 1
                    
                    if connections_count[node] >= node_target_degree:
                        break
        
        # Ensure network connectivity
        self._ensure_connectivity(graph)
        
        # Add topology metadata
        graph.graph['topology_type'] = 'degree_constrained'
        graph.graph['target_degree'] = target_degree
        graph.graph['preserve_geography'] = preserve_geography
        
        actual_avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        print(f"   ✅ Degree-constrained network: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges, avg_degree={actual_avg_degree:.2f}")
        
        return graph
    
    # ===== SMALL-WORLD NETWORKS =====
    def _generate_small_world(self, rewiring_probability: float,
                            initial_neighbors: int = 4,
                            preserve_geography: bool = False) -> nx.Graph:
        """
        Generate small-world network using Watts-Strogatz algorithm.
        
        Args:
            rewiring_probability: Probability of rewiring each edge (0.0 = regular, 1.0 = random)
            initial_neighbors: Number of initial neighbors in regular network
            preserve_geography: Whether to respect geographic constraints
        
        Returns:
            NetworkX graph with small-world topology
        """
        
        print(f"   Generating small-world network (p={rewiring_probability}, k={initial_neighbors})")
        
        # Start with regular network (degree-constrained)
        graph = self._generate_degree_constrained(initial_neighbors, preserve_geography=False)
        
        # Get node positions for geographic constraints
        if preserve_geography:
            node_positions = {node_id: node_data.coordinates 
                            for node_id, node_data in self.nodes.items()}
            distances = self._calculate_distance_matrix(node_positions)
        
        # Rewiring phase
        edges_to_rewire = list(graph.edges())
        shortcut_count = 0
        
        for u, v in edges_to_rewire:
            if random.random() < rewiring_probability:
                # Remove existing edge
                graph.remove_edge(u, v)
                
                # Find new target for rewiring
                possible_targets = [n for n in graph.nodes() 
                                 if n != u and not graph.has_edge(u, n)]
                
                if possible_targets:
                    if preserve_geography:
                        # Weight by inverse distance (closer nodes more likely)
                        weights = [1 / (distances[u][target] + 0.1) 
                                 for target in possible_targets]
                        total_weight = sum(weights)
                        weights = [w / total_weight for w in weights]
                        new_target = np.random.choice(possible_targets, p=weights)
                    else:
                        new_target = random.choice(possible_targets)
                    
                    # Add rewired edge as "shortcut"
                    self._add_transport_edge(graph, u, new_target, edge_type='shortcut')
                    shortcut_count += 1
                else:
                    # If no valid targets, restore original edge
                    self._add_transport_edge(graph, u, v)
        
        # Ensure connectivity
        self._ensure_connectivity(graph)
        
        # Add topology metadata
        graph.graph['topology_type'] = 'small_world'
        graph.graph['rewiring_probability'] = rewiring_probability
        graph.graph['initial_neighbors'] = initial_neighbors
        graph.graph['shortcuts_created'] = shortcut_count
        
        print(f"   ✅ Small-world network: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges, {shortcut_count} shortcuts")
        
        return graph
    
    # ===== SCALE-FREE NETWORKS =====
    def _generate_scale_free(self, m_edges: int,
                           alpha: float = 1.0,
                           preserve_geography: bool = False,
                           hub_preference: float = 0.8) -> nx.Graph:
        """
        Generate scale-free network using preferential attachment.
        
        Args:
            m_edges: Number of edges to attach from each new node
            alpha: Preferential attachment parameter (higher = more hub-dominated)
            preserve_geography: Whether to respect geographic constraints
            hub_preference: Preference for connecting to existing major hubs
        
        Returns:
            NetworkX graph with scale-free topology
        """
        
        print(f"   Generating scale-free network (m={m_edges}, α={alpha})")
        
        # Start with empty graph
        graph = nx.Graph()
        
        # Add all nodes
        for node_id, node_data in self.nodes.items():
            graph.add_node(node_id, **node_data.__dict__)
        
        nodes_list = list(graph.nodes())
        
        # Identify major hubs for preferential attachment
        major_hubs = [node_id for node_id, node_data in self.nodes.items()
                     if node_data.node_type == NodeType.MAJOR_HUB]
        
        # Get geographic distances if preserving geography
        if preserve_geography:
            node_positions = {node_id: node_data.coordinates 
                            for node_id, node_data in self.nodes.items()}
            distances = self._calculate_distance_matrix(node_positions)
        
        # Initial connections - connect major hubs to each other
        if len(major_hubs) >= 2:
            for i, hub1 in enumerate(major_hubs):
                for hub2 in major_hubs[i+1:i+3]:  # Connect to 1-2 other hubs
                    self._add_transport_edge(graph, hub1, hub2)
        
        # Preferential attachment process
        for node in nodes_list:
            if graph.degree(node) > 0:  # Skip if already connected
                continue
            
            # Calculate attachment probabilities
            attachment_probs = {}
            total_weight = 0
            
            for target_node in graph.nodes():
                if target_node == node or graph.has_edge(node, target_node):
                    continue
                
                # Base probability from degree (preferential attachment)
                degree_weight = (graph.degree(target_node) + 1) ** alpha
                
                # Hub bonus
                hub_weight = hub_preference if target_node in major_hubs else 1.0
                
                # Geographic weight (if preserving geography)
                if preserve_geography:
                    distance = distances[node][target_node]
                    geo_weight = 1 / (distance + 1)  # Closer nodes more likely
                else:
                    geo_weight = 1.0
                
                # Combined weight
                combined_weight = degree_weight * hub_weight * geo_weight
                attachment_probs[target_node] = combined_weight
                total_weight += combined_weight
            
            # Select targets for attachment
            if attachment_probs and total_weight > 0:
                # Normalize probabilities
                for target_node in attachment_probs:
                    attachment_probs[target_node] /= total_weight
                
                # Select m_edges targets (or fewer if not enough candidates)
                candidates = list(attachment_probs.keys())
                probabilities = list(attachment_probs.values())
                
                num_connections = min(m_edges, len(candidates))
                
                if num_connections > 0:
                    selected_targets = np.random.choice(
                        candidates, 
                        size=num_connections, 
                        replace=False, 
                        p=probabilities
                    )
                    
                    for target in selected_targets:
                        self._add_transport_edge(graph, node, target)
        
        # Ensure all nodes are connected
        self._ensure_connectivity(graph)
        
        # Calculate scale-free properties
        degrees = dict(graph.degree())
        max_degree = max(degrees.values())
        hubs = [node for node, degree in degrees.items() if degree >= 5]
        
        # Add topology metadata
        graph.graph['topology_type'] = 'scale_free'
        graph.graph['m_edges'] = m_edges
        graph.graph['alpha'] = alpha
        graph.graph['max_degree'] = max_degree
        graph.graph['num_hubs'] = len(hubs)
        
        print(f"   ✅ Scale-free network: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges, {len(hubs)} hubs (max_degree={max_degree})")
        
        return graph
    
    # ===== BASE SYDNEY NETWORKS =====
    def _generate_base_sydney(self, connectivity_level: int,
                            enhance_connectivity: bool = True) -> nx.Graph:
        """
        Generate enhanced Sydney network with regular spacing.
        
        Args:
            connectivity_level: Level of connectivity enhancement (4-8)
            enhance_connectivity: Whether to add extra connections
        
        Returns:
            NetworkX graph with enhanced Sydney topology
        """
        
        print(f"   Generating base Sydney network (connectivity={connectivity_level})")
        
        # Start with base network
        graph = self.base_graph.copy()
        
        if enhance_connectivity:
            # Add extra connections based on connectivity level
            node_positions = {node_id: node_data.coordinates 
                            for node_id, node_data in self.nodes.items()}
            distances = self._calculate_distance_matrix(node_positions)
            
            # For each node, ensure it has at least connectivity_level connections
            for node in graph.nodes():
                current_degree = graph.degree(node)
                if current_degree < connectivity_level:
                    # Find nearest unconnected nodes
                    candidates = []
                    for other_node in graph.nodes():
                        if (node != other_node and 
                            not graph.has_edge(node, other_node)):
                            distance = distances[node][other_node]
                            candidates.append((other_node, distance))
                    
                    # Sort by distance and connect to nearest
                    candidates.sort(key=lambda x: x[1])
                    connections_needed = connectivity_level - current_degree
                    
                    for other_node, _ in candidates[:connections_needed]:
                        self._add_transport_edge(graph, node, other_node)
        
        # Add topology metadata
        graph.graph['topology_type'] = 'base_sydney'
        graph.graph['connectivity_level'] = connectivity_level
        graph.graph['enhanced'] = enhance_connectivity
        
        print(f"   ✅ Base Sydney network: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")
        
        return graph
    
    # ===== GRID NETWORKS =====
    def _generate_grid(self, connectivity: int,
                  grid_width: int = 100, 
                  grid_height: int = 80) -> nx.Graph:
        """
        Generate regular grid network with Sydney station name mapping.
        
        Args:
            connectivity: Grid connectivity (4, 6, or 8)
            grid_width: Width of spatial grid
            grid_height: Height of spatial grid
        
        Returns:
            NetworkX graph with grid topology using Sydney station names where possible
        """
        
        print(f"[DEBUG] Starting grid generation with connectivity={connectivity}")
    
        # Create regular grid
        if connectivity == 4:
            # 4-connected grid (von Neumann neighborhood)
            grid = nx.grid_2d_graph(85, 85)  # Reduced size for manageability
        elif connectivity == 6:
            # 6-connected hexagonal grid
            grid = nx.hexagonal_lattice_graph(10, 8)
        elif connectivity == 8:
            # 8-connected grid (Moore neighborhood)
            grid = nx.grid_2d_graph(85, 85)
            # Add diagonal connections
            for node in grid.nodes():
                x, y = node
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    neighbor = (x + dx, y + dy)
                    if neighbor in grid.nodes():
                        grid.add_edge(node, neighbor)
        else:
            raise ValueError(f"Unsupported grid connectivity: {connectivity}")
        
        # ===== NEW: CREATE SYDNEY STATION COORDINATE MAPPING =====
        sydney_station_coords = {}
        used_station_names = set()
        
        # Extract Sydney station coordinates from base network
        if hasattr(self, 'base_network') and self.base_network:
            for node_id, node_data in self.base_network.nodes.items():
                # Convert coordinates to grid positions
                grid_x = int((node_data.coordinates[0] / 100) * grid_width)
                grid_y = int((node_data.coordinates[1] / 80) * grid_height)
                
                # Clamp to grid bounds
                grid_x = max(0, min(grid_x, grid_width - 1))
                grid_y = max(0, min(grid_y, grid_height - 1))
                
                grid_pos = (grid_x, grid_y)
                sydney_station_coords[grid_pos] = node_id
  
        def find_nearest_sydney_station(grid_coords, tolerance=2):
            """Find Sydney station name near grid coordinates"""
            x, y = grid_coords
            
            # Check exact match first
            if grid_coords in sydney_station_coords:
                station_name = sydney_station_coords[grid_coords]
                if station_name not in used_station_names:
                    used_station_names.add(station_name)
                    return station_name
            
            # Check nearby positions within tolerance
            for dx in range(-tolerance, tolerance + 1):
                for dy in range(-tolerance, tolerance + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nearby_pos = (x + dx, y + dy)
                    if nearby_pos in sydney_station_coords:
                        station_name = sydney_station_coords[nearby_pos]
                        if station_name not in used_station_names:
                            used_station_names.add(station_name)
                            return station_name
            
            return None
        
        # Convert to standard node IDs and add transport attributes
        graph = nx.Graph()
        node_mapping = {}
        sydney_stations_mapped = 0
        grid_nodes_created = 0
        print(f"[DEBUG] Starting node addition loop, grid has {len(list(grid.nodes()))} nodes")
    
        for i, grid_node in enumerate(grid.nodes()):
            # Calculate coordinates within the grid bounds
            if isinstance(grid_node, tuple) and len(grid_node) == 2:
                x_ratio = grid_node[0] / 84
                y_ratio = grid_node[1] / 84
            else:
                x_ratio = i % 20 / 20
                y_ratio = i // 20 / 16
            
            x_coord = x_ratio * grid_width
            y_coord = y_ratio * grid_height
            
            # Convert to integer grid position
            grid_x = int(x_coord)
            grid_y = int(y_coord)
            grid_position = (grid_x, grid_y)
            
            # ===== NEW: TRY TO MAP TO SYDNEY STATION NAME =====
            sydney_station_name = find_nearest_sydney_station(grid_position, tolerance=3)
            
            if sydney_station_name:
                # Use Sydney station name
                new_node_id = sydney_station_name
                sydney_stations_mapped += 1
                # print(f"   ✅ Grid node {grid_node} → Sydney station '{sydney_station_name}' at {grid_position}")
            else:
                # Use generic grid ID
                new_node_id = f"grid_{i}"
                grid_nodes_created += 1
            
            node_mapping[grid_node] = new_node_id
            
            # Create network node with appropriate data
            if sydney_station_name:
                # Use Sydney station data if available
                if hasattr(self, 'base_network') and sydney_station_name in self.base_network.nodes:
                    sydney_node_data = self.base_network.nodes[sydney_station_name]
                    node_data = NetworkNode(
                        node_id=new_node_id,
                        node_type=sydney_node_data.node_type,
                        coordinates=(x_coord, y_coord),
                        population_weight=sydney_node_data.population_weight,
                        employment_weight=sydney_node_data.employment_weight,
                        transport_modes=sydney_node_data.transport_modes,
                        zone_name=sydney_node_data.zone_name
                    )
                else:
                    # Fallback node data
                    node_data = NetworkNode(
                        node_id=new_node_id,
                        node_type=NodeType.TRANSPORT_HUB,
                        coordinates=(x_coord, y_coord),
                        population_weight=0.5,
                        employment_weight=0.4,
                        transport_modes=[TransportMode.TRAIN, TransportMode.BUS],
                        zone_name=sydney_station_name
                    )
            else:
                # Generic grid node data
                node_data = NetworkNode(
                    node_id=new_node_id,
                    node_type=NodeType.LOCAL_STATION,
                    coordinates=(x_coord, y_coord),
                    population_weight=0.3,
                    employment_weight=0.2,
                    transport_modes=[TransportMode.BUS],
                    zone_name=f"Grid_Zone_{i}"
                )

            graph.add_node(new_node_id, **node_data.__dict__)
        
        # Add edges with transport attributes
        for edge in grid.edges():
            node1 = node_mapping[edge[0]]
            node2 = node_mapping[edge[1]]
            self._add_transport_edge(graph, node1, node2)
        
        # Add topology metadata
        graph.graph['topology_type'] = 'grid'
        graph.graph['connectivity'] = connectivity
        graph.graph['grid_width'] = grid_width
        graph.graph['grid_height'] = grid_height
        
        print(f"   ✅ Grid network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"   📊 Station mapping: {sydney_stations_mapped} Sydney stations, {grid_nodes_created} generic grid nodes")
        
        return graph
    
    # ===== UTILITY METHODS =====
    def _calculate_distance_matrix(self, node_positions: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate distance matrix between all node pairs"""
        distances = {}
        for node1, pos1 in node_positions.items():
            distances[node1] = {}
            for node2, pos2 in node_positions.items():
                if node1 == node2:
                    distances[node1][node2] = 0
                else:
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    distances[node1][node2] = math.sqrt(dx*dx + dy*dy)
        return distances
    
    def _add_transport_edge(self, graph: nx.Graph, node1: str, node2: str, 
                          edge_type: str = 'regular'):
        """Add transport edge preserving Sydney route structure"""
        
        # 🎯 CRITICAL FIX: Check for existing Sydney edge first
        if hasattr(self, 'base_network') and self.base_network.graph.has_edge(node1, node2):
            # Use original Sydney edge data
            original_data = self.base_network.graph[node1][node2]
            graph.add_edge(node1, node2, **original_data)
            return
        
        # Calculate distance
        pos1 = self.nodes[node1].coordinates if node1 in self.nodes else (0, 0)
        pos2 = self.nodes[node2].coordinates if node2 in self.nodes else (0, 0)
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # Determine transport mode and create Sydney-style route ID
        if edge_type == 'shortcut' or distance > 25:
            transport_mode = TransportMode.TRAIN if distance > 25 else TransportMode.BUS
            route_id = f"T{400 + len(graph.edges())}_UNIFIED" if distance > 25 else f"BUS_{4000 + len(graph.edges())}"
            travel_time = distance / 30 if edge_type == 'shortcut' else distance / 20
            capacity = 1500 if distance > 25 else 800
        else:
            transport_mode = TransportMode.BUS
            route_id = f"BUS_{4000 + len(graph.edges())}"
            travel_time = distance / 20
            capacity = 600
        
        # Add edge with Sydney-compatible attributes
        graph.add_edge(node1, node2,
                      transport_mode=transport_mode,
                      travel_time=max(0.5, travel_time),
                      distance=distance,
                      capacity=capacity,
                      frequency=12,
                      route_id=route_id,  # 🎯 KEY FIX!
                      segment_order=1,
                      edge_type=edge_type,
                      weight=travel_time)

    
    def _ensure_connectivity(self, graph: nx.Graph):
        """Ensure network is fully connected"""
        
        if not nx.is_connected(graph):
            # Find connected components
            components = list(nx.connected_components(graph))
            
            # Connect components by adding edges between their closest nodes
            main_component = max(components, key=len)
            
            for component in components:
                if component == main_component:
                    continue
                
                # Find closest pair of nodes between main component and this component
                min_distance = float('inf')
                best_connection = None
                
                for node1 in main_component:
                    for node2 in component:
                        if node1 in self.nodes and node2 in self.nodes:
                            pos1 = self.nodes[node1].coordinates
                            pos2 = self.nodes[node2].coordinates
                            distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                            
                            if distance < min_distance:
                                min_distance = distance
                                best_connection = (node1, node2)
                
                if best_connection:

                    self._add_transport_edge(graph, best_connection[0], best_connection[1])
                    
                    main_component.update(component)
    
    def get_topology_properties(self, graph: nx.Graph) -> Dict[str, float]:
        """Calculate standard topology properties for any network"""
        
        properties = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'is_connected': nx.is_connected(graph),
            'density': nx.density(graph)
        }
        
        if nx.is_connected(graph):
            properties['avg_shortest_path_length'] = nx.average_shortest_path_length(graph)
            properties['diameter'] = nx.diameter(graph)
        
        properties['avg_clustering'] = nx.average_clustering(graph)
        
        # Degree statistics
        degrees = dict(graph.degree())
        degree_values = list(degrees.values())
        properties['avg_degree'] = np.mean(degree_values)
        properties['max_degree'] = max(degree_values)
        properties['degree_variance'] = np.var(degree_values)
        
        # Centrality measures
        betweenness = nx.betweenness_centrality(graph)
        properties['max_betweenness'] = max(betweenness.values())
        properties['avg_betweenness'] = np.mean(list(betweenness.values()))
        
        return properties

# ===== CONVENIENCE FUNCTIONS =====
def create_topology(topology_type: str, variation_parameter: Union[int, float], 
                   base_network: SydneyNetworkTopology = None, **kwargs) -> nx.Graph:
    """
    Convenience function to create any topology type
    
    Args:
        topology_type: Type of topology ('degree_constrained', 'small_world', 'scale_free', etc.)
        variation_parameter: Main parameter controlling the topology
        base_network: Base network (optional)
        **kwargs: Additional parameters
    
    Returns:
        Generated NetworkX graph
    """
    generator = UnifiedTopologyGenerator(base_network)
    return generator.generate_topology(topology_type, variation_parameter, **kwargs)

def get_available_topologies() -> List[str]:
    """Get list of all available topology types"""
    return [topology.value for topology in TopologyType]

def get_topology_info(topology_type: str) -> Dict[str, str]:
    """Get information about a specific topology type"""
    
    info = {
        'degree_constrained': {
            'description': 'Networks where all nodes have approximately the same degree',
            'parameter': 'Target degree (3-7)',
            'use_case': 'Studying impact of uniform connectivity'
        },
        'small_world': {
            'description': 'Watts-Strogatz networks with local clustering and shortcuts',
            'parameter': 'Rewiring probability (0.0-1.0)',
            'use_case': 'Studying balance between local and global connectivity'
        },
        'scale_free': {
            'description': 'Barabási-Albert networks with hub-dominated structure',
            'parameter': 'Number of edges per new node (1-5)',
            'use_case': 'Studying impact of transport hubs and preferential attachment'
        },
        'base_sydney': {
            'description': 'Enhanced Sydney network with regular spacing',
            'parameter': 'Connectivity level (4-8)',
            'use_case': 'Realistic Sydney transport network baseline'
        },
        'grid': {
            'description': 'Regular grid networks with uniform spatial structure',
            'parameter': 'Grid connectivity (4, 6, or 8)',
            'use_case': 'Mathematical baseline for comparison'
        }
    }
    
    return info.get(topology_type, {'description': 'Unknown topology type'})

# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    print("UNIFIED TOPOLOGY GENERATOR - TESTING ALL TYPES")
    print("=" * 60)
    
    # Initialize generator
    generator = UnifiedTopologyGenerator()
    
    # Test all topology types
    test_configs = [
        ('degree_constrained', 4),
        ('small_world', 0.1),
        ('scale_free', 2),
        ('base_sydney', 6),
        ('grid', 6)
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
            
        except Exception as e:
            print(f"❌ Failed in test_configs: {e}")
    
    print(f"\n✅ Unified topology generator testing completed!")
    print(f"📋 Available topologies: {get_available_topologies()}")