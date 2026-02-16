import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from mesa.space import MultiGrid
from Complete_NTEO.topology.network_topology import SydneyNetworkTopology, DegreeConstrainedTopologyGenerator, NetworkNode, TransportMode
import random
from collections import defaultdict
# At the top of network_integration.py, add:
from Complete_NTEO.topology.network_topology import TransportMode

class NetworkSpatialMapper:
    """Maps abstract network topology to Mesa spatial grid"""
    
    def __init__(self, network_topology: SydneyNetworkTopology, grid_width: int = 100, grid_height: int = 80):
        self.network = network_topology
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        self.node_to_grid: Dict[str, Tuple[int, int]] = {}
        self.grid_to_node: Dict[Tuple[int, int], str] = {}
        self.access_points: Dict[Tuple[int, int], List[str]] = {}
        self.population_zones: Dict[str, List[Tuple[int, int]]] = {}
        
        self._initialize_spatial_mapping()
    
    def _initialize_spatial_mapping(self):
        """Create mapping between network nodes and grid coordinates"""
        
        for node_id, node_data in self.network.nodes.items():
            grid_x = int((node_data.coordinates[0] / 100) * self.grid_width)
            grid_y = int((node_data.coordinates[1] / 80) * self.grid_height)
            
            grid_x = max(0, min(grid_x, self.grid_width - 1))
            grid_y = max(0, min(grid_y, self.grid_height - 1))
            
            self.node_to_grid[node_id] = (grid_x, grid_y)
            self.grid_to_node[(grid_x, grid_y)] = node_id
        
        self._create_access_zones()
        self._create_population_zones()
    
    def _create_access_zones(self):
        """Create access zones around network nodes"""
        
        for node_id, grid_coord in self.node_to_grid.items():
            node_data = self.network.nodes[node_id]
            
            if node_data.node_type.value in ['major_hub', 'transport_hub']:
                access_radius = 3
            else:
                access_radius = 2
            
            x_center, y_center = grid_coord
            for dx in range(-access_radius, access_radius + 1):
                for dy in range(-access_radius, access_radius + 1):
                    x, y = x_center + dx, y_center + dy
                    
                    if (0 <= x < self.grid_width and 0 <= y < self.grid_height and
                        dx*dx + dy*dy <= access_radius*access_radius):
                        
                        if (x, y) not in self.access_points:
                            self.access_points[(x, y)] = []
                        
                        self.access_points[(x, y)].append(node_id)
    
    def _create_population_zones(self):
        """Create population zones around residential nodes"""
        
        for node_id, node_data in self.network.nodes.items():
            if node_data.population_weight > 0.3:
                self.population_zones[node_id] = []
                
                x_center, y_center = self.node_to_grid[node_id]
                zone_radius = int(5 + node_data.population_weight * 10)
                
                for dx in range(-zone_radius, zone_radius + 1):
                    for dy in range(-zone_radius, zone_radius + 1):
                        x, y = x_center + dx, y_center + dy
                        
                        if (0 <= x < self.grid_width and 0 <= y < self.grid_height):
                            distance = np.sqrt(dx*dx + dy*dy)
                            probability = max(0, 1 - distance / zone_radius)
                            
                            if random.random() < probability:
                                self.population_zones[node_id].append((x, y))
    
    
    
    def get_nearest_node(self, grid_position: Tuple[int, int]) -> Optional[str]:
        """Find nearest network node to a grid position"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node_grid_pos in self.node_to_grid.items():
            distance = np.sqrt((grid_position[0] - node_grid_pos[0])**2 + 
                             (grid_position[1] - node_grid_pos[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
    
    def get_random_residential_location(self) -> Tuple[int, int]:
        """Get random location in a residential zone"""
        nodes_with_population = [(node_id, data.population_weight) 
                               for node_id, data in self.network.nodes.items() 
                               if data.population_weight > 0.2]
        
        if not nodes_with_population:
            return (random.randint(0, self.grid_width-1), random.randint(0, self.grid_height-1))
        
        total_weight = sum(weight for _, weight in nodes_with_population)
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for node_id, weight in nodes_with_population:
            cumulative += weight
            if r <= cumulative:
                if node_id in self.population_zones and self.population_zones[node_id]:
                    return random.choice(self.population_zones[node_id])
                else:
                    return self.node_to_grid[node_id]
        
        return self.node_to_grid[nodes_with_population[0][0]]

class NetworkEdgeCongestionModel:
    """Congestion modeling on network edges instead of grid cells"""
    
    def __init__(self, network_graph: nx.Graph, alpha: float = 0.03, beta: float = 1.5):
        self.network = network_graph
        self.alpha = alpha
        self.beta = beta
        
        # Track traffic on network edges
        self.edge_traffic: Dict[Tuple[str, str, int], int] = defaultdict(int)  # (from_node, to_node, time_step) -> count
        self.edge_capacities: Dict[Tuple[str, str], int] = {}
        self.base_travel_times: Dict[Tuple[str, str], float] = {}
        
        self._initialize_edge_properties()
    
    def _initialize_edge_properties(self):
        """Initialize edge capacities and base travel times from network"""
        for u, v, edge_data in self.network.edges(data=True):
            edge_id = (u, v)
            self.edge_capacities[edge_id] = edge_data.get('capacity', 1000)
            self.base_travel_times[edge_id] = edge_data.get('travel_time', 5.0)
            
            # Add reverse direction
            reverse_edge_id = (v, u)
            self.edge_capacities[reverse_edge_id] = edge_data.get('capacity', 1000)
            self.base_travel_times[reverse_edge_id] = edge_data.get('travel_time', 5.0)
    
    def record_trip_on_network(self, network_route: List[str], start_time: int, duration: int):
        """Record traffic on network edges"""
        affected_steps = list(range(start_time, start_time + duration))
        
        for i in range(len(network_route) - 1):
            from_node = network_route[i]
            to_node = network_route[i + 1]
            
            for step in affected_steps:
                self.edge_traffic[(from_node, to_node, step)] += 1
                # Also record reverse direction for bidirectional traffic
                self.edge_traffic[(to_node, from_node, step)] += 1
    
    def get_edge_congestion_factor(self, from_node: str, to_node: str, time_step: int) -> float:
        """Calculate congestion factor for a specific edge at a specific time"""
        edge_id = (from_node, to_node)
        
        if edge_id not in self.edge_capacities:
            return 1.0  # No congestion if edge not found
        
        current_traffic = self.edge_traffic.get((from_node, to_node, time_step), 0)
        capacity = self.edge_capacities[edge_id]
        
        if capacity <= 0:
            return 1.0
        
        utilization_ratio = current_traffic / capacity
        congestion_factor = 1 + self.alpha * (utilization_ratio ** self.beta)
        
        return congestion_factor
    
    def get_congested_travel_time(self, from_node: str, to_node: str, time_step: int) -> float:
        """Get travel time including congestion effects"""
        edge_id = (from_node, to_node)
        
        if edge_id not in self.base_travel_times:
            return 5.0  # Default travel time
        
        base_time = self.base_travel_times[edge_id]
        congestion_factor = self.get_edge_congestion_factor(from_node, to_node, time_step)
        
        return base_time * congestion_factor

class NetworkRouter:
    """Enhanced routing with network-edge congestion"""
    
    def __init__(self, network_graph: nx.Graph, congestion_model: NetworkEdgeCongestionModel):
        self.graph = network_graph
        self.congestion_model = congestion_model
        self.shortest_paths_cache = {}
    
    def find_shortest_path(self, origin_node: str, destination_node: str, mode_preference=None):
        """Alias for backward compatibility"""
        return self.find_route(origin_node, destination_node, mode_preference)
    def find_route_with_congestion(self, origin_node: str, destination_node: str, 
                                  current_time: int, mode_preference: List[TransportMode] = None) -> Optional[Dict]:
        """Find optimal route considering network edge congestion"""
        
        cache_key = (origin_node, destination_node, current_time, tuple(mode_preference) if mode_preference else None)
        if cache_key in self.shortest_paths_cache:
            return self.shortest_paths_cache[cache_key]
        
        try:
            # Create temporary graph with congestion-adjusted weights
            temp_graph = self.graph.copy()
            
            for u, v, edge_data in temp_graph.edges(data=True):
                # Apply congestion to edge weights
                congested_time = self.congestion_model.get_congested_travel_time(u, v, current_time)
                temp_graph[u][v]['congested_weight'] = congested_time
            
            # Use congestion-adjusted weights for routing
            path = nx.shortest_path(temp_graph, origin_node, destination_node, weight='congested_weight')
            
            if len(path) < 2:
                return None
            
            # Build detailed route information
            route_details = {
                'path': path,
                'segments': [],
                'total_time': 0,
                'total_distance': 0,
                'modes_used': set()
            }
            
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                edge_data = self.graph[from_node][to_node]
                
                # Use congested travel time
                congested_time = self.congestion_model.get_congested_travel_time(from_node, to_node, current_time)
                
                segment = {
                    'from': from_node,
                    'to': to_node,
                    'mode': edge_data.get('transport_mode', TransportMode.BUS), 
                    'time': congested_time,
                    'distance': edge_data['distance'],
                    'route_id': edge_data.get('route_id', 'unknown')
                }
                
                route_details['segments'].append(segment)
                route_details['total_time'] += congested_time
                route_details['total_distance'] += edge_data['distance']
                if 'transport_mode' not in edge_data:
                    print(f"⚠️  Missing transport_mode in edge {from_node}->{to_node}")
                    print(f"   Edge data keys: {list(edge_data.keys())}")
                    
                    # Try to infer transport mode from other attributes
                    if 'route_id' in edge_data:
                        route_id = edge_data['route_id']
                        if 'train' in route_id.lower() or 'T1' in route_id or 'T4' in route_id or 'T8' in route_id:
                            transport_mode = TransportMode.TRAIN
                        elif 'bus' in route_id.lower() or 'BUS' in route_id:
                            transport_mode = TransportMode.BUS
                        else:
                            transport_mode = TransportMode.BUS  # Default fallback
                    else:
                        transport_mode = TransportMode.BUS  # Default fallback
                    
                    print(f"   Using inferred mode: {transport_mode}")
                    edge_data['transport_mode'] = transport_mode  # Fix the edge for future use

                transport_mode = edge_data['transport_mode']
                route_details['modes_used'].add(transport_mode)
            
            # Cache result
            self.shortest_paths_cache[cache_key] = route_details
            
            # Limit cache size
            if len(self.shortest_paths_cache) > 1000:
                keys_to_remove = list(self.shortest_paths_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.shortest_paths_cache[key]
            
            return route_details
            
        except nx.NetworkXNoPath:
            return None
    
    def find_route(self, origin_node: str, destination_node: str, 
                   mode_preference: List[TransportMode] = None) -> Optional[Dict]:
        """Find route without congestion (for public transport)"""
        
        cache_key = (origin_node, destination_node, tuple(mode_preference) if mode_preference else None)
        if cache_key in self.shortest_paths_cache:
            return self.shortest_paths_cache[cache_key]
        
        try:
            path = nx.shortest_path(self.graph, origin_node, destination_node, weight='travel_time')
            
            if len(path) < 2:
                return None
            
            route_details = {
                'path': path,
                'segments': [],
                'total_time': 0,
                'total_distance': 0,
                'modes_used': set()
            }
            
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                edge_data = self.graph[from_node][to_node]
                
                segment = {
                    'from': from_node,
                    'to': to_node,
                    'mode': edge_data['transport_mode'],
                    'time': edge_data['travel_time'],
                    'distance': edge_data['distance'],
                    'route_id': edge_data.get('route_id', 'unknown')
                }
                
                route_details['segments'].append(segment)
                route_details['total_time'] += edge_data['travel_time']
                route_details['total_distance'] += edge_data['distance']
                route_details['modes_used'].add(edge_data['transport_mode'])
            
            self.shortest_paths_cache[cache_key] = route_details
            
            if len(self.shortest_paths_cache) > 1000:
                keys_to_remove = list(self.shortest_paths_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.shortest_paths_cache[key]
            
            return route_details
            
        except nx.NetworkXNoPath:
            return None

class TwoLayerNetworkManager:
    """Enhanced network manager with congestion modeling"""
    
    def __init__(self, topology_type: str = "degree_constrained", 
                 degree: int = 3, grid_width: int = 100, grid_height: int = 80,
                 rewiring_probability: float = 0.1, initial_neighbors: int = 4):
        
        # Layer 1: Abstract network topology
        self.base_network = SydneyNetworkTopology()
        self.base_network.initialize_base_sydney_network()
        
        # Generate specific topology
        if topology_type == "degree_constrained":
            generator = DegreeConstrainedTopologyGenerator(self.base_network)
            self.active_network = generator.generate_degree_constrained_network(degree)
            
        elif topology_type == "small_world":
            # ADD THIS SMALL-WORLD CASE
            from small_world_topology import SmallWorldTopologyGenerator
            generator = SmallWorldTopologyGenerator(self.base_network)
            self.active_network = generator.generate_small_world_network(
                rewiring_probability=rewiring_probability,
                initial_neighbors=initial_neighbors,
                preserve_geography=True
            )
            print(f"🌐 Small-world network generated with {rewiring_probability} rewiring probability")
        elif topology_type == "scale_free":
            from Complete_NTEO.topology.scale_free_topology import ScaleFreeTopologyGenerator
            generator = ScaleFreeTopologyGenerator(self.base_network)
            self.active_network = generator.generate_scale_free_network(
                m_edges=degree,  # Use degree parameter as m_edges
                alpha=1.0,
                preserve_geography=True
            )
            print(f"🌐 Scale-free network generated with m_edges={degree}, alpha=1.0")
            
        else:
            # Default fallback
            print(f"⚠️ Unknown topology type: {topology_type}, using degree_constrained")
            generator = DegreeConstrainedTopologyGenerator(self.base_network)
            self.active_network = generator.generate_degree_constrained_network(degree)
        # Layer 2: Spatial mapping
        self.spatial_mapper = NetworkSpatialMapper(self.base_network, grid_width, grid_height)
        
        # Congestion and routing
        self.congestion_model = NetworkEdgeCongestionModel(self.active_network)
        self.router = NetworkRouter(self.active_network, self.congestion_model)
        
        # Performance tracking
        self.route_calculation_count = 0
        
        # Ensure connectivity
        self._ensure_network_connectivity()
    
    def find_network_route(self, origin_grid: Tuple[int, int], 
                          destination_grid: Tuple[int, int], 
                          mode: str = 'car', current_time: int = 0) -> Optional[Dict]:
        """Find route using network topology with congestion"""
        self.route_calculation_count += 1
        
        # Find nearest network nodes
        origin_node = self._find_accessible_node(origin_grid)
        destination_node = self._find_accessible_node(destination_grid)
        
        if not origin_node or not destination_node:
            print(f"No accessible network nodes found for {origin_grid} -> {destination_grid}")
            return None
        # If same node, create direct route
        if origin_node == destination_node:
            origin_coord = self.spatial_mapper.node_to_grid.get(origin_node, origin_grid)
            dest_coord = self.spatial_mapper.node_to_grid.get(destination_node, destination_grid)
            return {
                'network_route': {'path': [origin_node]},
                'spatial_route': [origin_grid, origin_coord, dest_coord, destination_grid],
                'access_node': origin_node,
                'egress_node': destination_node
            }
        # Route on network with or without congestion
        if mode in ['car', 'bike']:  # Apply congestion for road-based modes
            network_route = self.router.find_route_with_congestion(
                origin_node, destination_node, current_time
            )
        else:  # No congestion for public transport
            network_route = self.router.find_route(origin_node, destination_node)
        
        if not network_route:
            print(f"No network route found from {origin_node} to {destination_node}")
            return None
        # Convert to spatial route
        spatial_route = self._convert_to_spatial_route(network_route, origin_grid, destination_grid)
        
        result = {
            'network_route': network_route,
            'spatial_route': spatial_route,
            'access_node': origin_node,
            'egress_node': destination_node
        }
        # At the very end, before returning result:
        if hasattr(self, 'log_route_analysis'):
            self.log_route_analysis(result, f"route_{current_time}")
        
        return result
    
    def _find_accessible_node(self, grid_position: Tuple[int, int], max_search_radius: int = 10) -> Optional[str]:
        """Find accessible network node with expanding search"""
        
        # First try: direct lookup
        nearest_node = self.spatial_mapper.get_nearest_node(grid_position)
        if nearest_node:
            return nearest_node
        
        # Second try: expanding radius search
        for radius in range(1, max_search_radius + 1):
            x, y = grid_position
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:  # Within circular radius
                        search_x = max(0, min(self.spatial_mapper.grid_width - 1, x + dx))
                        search_y = max(0, min(self.spatial_mapper.grid_height - 1, y + dy))
                        
                        search_pos = (search_x, search_y)
                        node = self.spatial_mapper.get_nearest_node(search_pos)
                        if node:
                            return node
        
        # Third try: any available node (fallback)
        available_nodes = list(self.spatial_mapper.node_to_grid.keys())
        if available_nodes:
            print(f"Using fallback node for {grid_position}")
            return available_nodes[0]
        
        return None
    
    def record_trip_on_network(self, origin_grid: Tuple[int, int], destination_grid: Tuple[int, int],
                              start_time: int, duration: int) -> bool:
        """Record a trip for congestion modeling"""
        
        origin_node = self.spatial_mapper.get_nearest_node(origin_grid)
        destination_node = self.spatial_mapper.get_nearest_node(destination_grid)
        
        if not origin_node or not destination_node:
            return False
        
        try:
            # Find network path
            path = nx.shortest_path(self.active_network, origin_node, destination_node, weight='travel_time')
            
            # Record on congestion model
            self.congestion_model.record_trip_on_network(path, start_time, duration)
            return True
            
        except nx.NetworkXNoPath:
            return False
    
    def _convert_to_spatial_route(self, network_route: Dict, 
                                origin_grid: Tuple[int, int], 
                                destination_grid: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Convert network route to spatial coordinates"""
        spatial_route = [origin_grid]
        
        for node_id in network_route['path']:
            if node_id in self.spatial_mapper.node_to_grid:
                spatial_route.append(self.spatial_mapper.node_to_grid[node_id])
        
        if destination_grid != spatial_route[-1]:
            spatial_route.append(destination_grid)
        
        return spatial_route
    
    def get_network_statistics(self) -> Dict:
        """Get network topology statistics"""
        graph = self.active_network
        
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
            'clustering_coefficient': nx.average_clustering(graph),
            'is_connected': nx.is_connected(graph),
            'diameter': nx.diameter(graph) if nx.is_connected(graph) else float('inf'),
            'avg_path_length': nx.average_shortest_path_length(graph) if nx.is_connected(graph) else float('inf'),
            'route_calculations': self.route_calculation_count,
            'cache_size': len(self.router.shortest_paths_cache)
        }
        
        degrees = [d for n, d in graph.degree()]
        stats['degree_distribution'] = {
            'min': min(degrees),
            'max': max(degrees),
            'std': np.std(degrees),
            'variance': np.var(degrees)
        }
        
        return stats
    
    def _ensure_network_connectivity(self):
        """Ensure the network is fully connected"""
        if not nx.is_connected(self.active_network):
            print("Network has disconnected components. Adding connectivity edges...")
            
            # Find all connected components
            components = list(nx.connected_components(self.active_network))
            print(f"Found {len(components)} disconnected components")
            
            # Connect components by finding closest nodes between them
            main_component = max(components, key=len)
            
            for component in components:
                if component == main_component:
                    continue
                    
                # Find closest pair of nodes between main component and this component
                min_distance = float('inf')
                best_pair = None
                
                for node1 in main_component:
                    coord1 = self.base_network.nodes[node1].coordinates
                    for node2 in component:
                        coord2 = self.base_network.nodes[node2].coordinates
                        distance = ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_pair = (node1, node2)
                
                # Add connecting edge
                if best_pair:
                    node1, node2 = best_pair
                    self._add_connectivity_edge(node1, node2)
                    print(f"Added connectivity edge between {node1} and {node2}")
        
        # Verify connectivity
        if not nx.is_connected(self.active_network):
            print("WARNING: Network still not fully connected after repair attempts")
    
    def _add_connectivity_edge(self, node1: str, node2: str):
        """Add edge to connect disconnected components"""
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        distance = ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
        
        # Determine appropriate transport mode based on distance
        if distance > 20:
            mode = TransportMode.TRAIN
            travel_time = distance * 0.8
            capacity = 1500
        else:
            mode = TransportMode.BUS
            travel_time = distance * 1.0
            capacity = 600
        
        edge_data = {
            'transport_mode': mode,
            'travel_time': travel_time,
            'capacity': capacity,
            'frequency': 8,
            'distance': distance,
            'route_id': f"CONNECTIVITY_{node1}_{node2}",
            'segment_order': 1
        }
        
        self.active_network.add_edge(node1, node2, **edge_data)