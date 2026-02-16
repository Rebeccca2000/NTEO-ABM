import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import random
from collections import defaultdict

# Add the required imports for TwoLayerNetworkManager compatibility
try:
    import warnings
except ImportError:
    pass

class UnifiedSpatialMapper:
    """Fixed spatial mapper with exact method names from working system"""
    
    def __init__(self, network_topology, grid_width: int = 100, grid_height: int = 80):
        # CRITICAL FIX: Handle both NetworkX graphs and SydneyNetworkTopology objects
        if hasattr(network_topology, 'graph') and hasattr(network_topology.graph, 'nodes'):
            self.network = network_topology.graph  # Extract NetworkX graph from SydneyNetworkTopology
            print(f"🔧 UnifiedSpatialMapper: Using .graph from SydneyNetworkTopology")
        elif hasattr(network_topology, 'nodes') and callable(getattr(network_topology, 'nodes', None)):
            self.network = network_topology  # Already a NetworkX graph
            print(f"🔧 UnifiedSpatialMapper: Using NetworkX graph directly")
        else:
            raise TypeError(f"Expected NetworkX graph or SydneyNetworkTopology, got {type(network_topology)}")

        self.grid_width = grid_width
        self.grid_height = grid_height
        
        self.node_to_grid: Dict[str, Tuple[int, int]] = {}
        self.grid_to_node: Dict[Tuple[int, int], str] = {}
        
        self._initialize_spatial_mapping()
    
    def _initialize_spatial_mapping(self):
        """Create mapping between network nodes (both Sydney stations and grid nodes) and grid coordinates"""
        
        for node_id, node_data in self.network.nodes(data=True):
            # Get coordinates from node data
            if 'coordinates' in node_data:
                # Direct coordinates available
                raw_coords = node_data['coordinates']
            else:
                # Fallback: extract from NetworkNode object if available
                if hasattr(node_data, 'coordinates'):
                    raw_coords = node_data.coordinates
                else:
                    # Last fallback: estimate from node ID
                    raw_coords = self._estimate_coordinates_from_node_id(node_id)
            
            # Convert to grid coordinates
            if len(raw_coords) >= 2:
                # Handle both (x, y) and (x, y, z) coordinate formats
                grid_x = int((raw_coords[0] / 100) * self.grid_width)
                grid_y = int((raw_coords[1] / 80) * self.grid_height)
            else:
                # Fallback for malformed coordinates
                grid_x, grid_y = 50, 40
            
            # Clamp to grid bounds
            grid_x = max(0, min(grid_x, self.grid_width - 1))
            grid_y = max(0, min(grid_y, self.grid_height - 1))
            
            grid_pos = (grid_x, grid_y)
            
            self.node_to_grid[node_id] = grid_pos
            self.grid_to_node[grid_pos] = node_id
            
           
    
    def get_nearest_node(self, grid_position: Tuple[int, int]) -> Optional[str]:
        """EXACT METHOD NAME from working system"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node_grid_pos in self.node_to_grid.items():
            distance = np.sqrt((grid_position[0] - node_grid_pos[0])**2 + 
                             (grid_position[1] - node_grid_pos[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
    
    def find_nearest_network_node(self, grid_position: Tuple[int, int]) -> Optional[str]:
        """Compatibility method for MaaS agent calls"""
        return self.get_nearest_node(grid_position)
    def _estimate_coordinates_from_node_id(self, node_id: str) -> Tuple[float, float]:
        """Estimate coordinates for nodes without explicit coordinate data"""
        
        # Handle grid nodes
        if node_id.startswith('grid_'):
            try:
                grid_num = int(node_id.split('_')[1])
                # Distribute grid nodes evenly across space
                x = (grid_num % 20) * 5  # 20 columns
                y = (grid_num // 20) * 5  # Variable rows
                return (float(x), float(y))
            except (ValueError, IndexError):
                return (50.0, 40.0)  # Center fallback
        
        # Handle Sydney station names - use hardcoded positions for known stations
        sydney_positions = {
            'CENTRAL': (50, 40),
            'CHATSWOOD': (55, 65),
            'BLACKTOWN': (15, 25),
            'CROWS_NEST': (54, 55),
            'ST_LEONARDS': (55, 58),
            'BONDI_JUNCTION': (65, 45),
            'PARRAMATTA': (25, 30),
            'LIVERPOOL': (25, 15),
            'KOGARAH': (47, 23),
            'BURWOOD': (35, 35),
            'MAROUBRA': (62, 30),
            'NEWTOWN': (45, 35),
            'REDFERN': (48, 38),
            'MANLY': (65, 55),
            'DEE_WHY': (68, 60),
            'MONA_VALE': (70, 65),
        }
        
        if node_id in sydney_positions:
            return sydney_positions[node_id]
        
        # Final fallback
        return (50.0, 40.0)
    
    def get_nearest_node(self, grid_position: Tuple[int, int]) -> Optional[str]:
        """Find nearest network node to given grid position - works with both Sydney stations and grid nodes"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node_grid_pos in self.node_to_grid.items():
            distance = np.sqrt((grid_position[0] - node_grid_pos[0])**2 + 
                             (grid_position[1] - node_grid_pos[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
    
    def find_nearest_network_node(self, grid_position: Tuple[int, int]) -> Optional[str]:
        """Compatibility method for MaaS agent calls"""
        return self.get_nearest_node(grid_position)
    
    def get_station_name_by_coordinates(self, grid_position: Tuple[int, int], tolerance: int = 2) -> Optional[str]:
        """Get Sydney station name if one exists at or near the given coordinates"""
        
        # Check exact position first
        if grid_position in self.grid_to_node:
            node_id = self.grid_to_node[grid_position]
            # Return only if it's a Sydney station (not grid_X)
            if not node_id.startswith('grid_'):
                return node_id
        
        # Check nearby positions
        x, y = grid_position
        for dx in range(-tolerance, tolerance + 1):
            for dy in range(-tolerance, tolerance + 1):
                check_pos = (x + dx, y + dy)
                if check_pos in self.grid_to_node:
                    node_id = self.grid_to_node[check_pos]
                    if not node_id.startswith('grid_'):
                        return node_id
        
        return None
    
    def is_sydney_station(self, node_id: str) -> bool:
        """Check if a node ID represents a Sydney station (not a generic grid node)"""
        return not node_id.startswith('grid_')
    
    def get_grid_coordinates(self, node_id: str) -> Tuple[int, int]:
        """Get grid coordinates for a given node ID"""
        return self.node_to_grid.get(node_id, (50, 40))

class NetworkCongestionModel:
    """Modified congestion model with grid-based BPR function for cars only"""
    
    def __init__(self, network_graph: nx.Graph, grid_width: int = 100, grid_height: int = 80):
        import config.database_updated as db

        self.network = network_graph
        self.alpha = db.CONGESTION_ALPHA  # Import from database
        self.beta = db.CONGESTION_BETA    # Import from database
        self.grid_capacity = db.CONGESTION_CAPACITY  # Capacity per grid cell
        self.grid_free_flow_time = db.CONGESTION_T_IJ_FREE_FLOW  # Free flow time per grid cell
        
        # Existing network traffic tracking (for public transport)
        self.edge_traffic: Dict[Tuple[str, str, int], int] = defaultdict(int)
        self.edge_capacities: Dict[Tuple[str, str], int] = {}
        self.base_travel_times: Dict[Tuple[str, str], float] = {}
        
        # NEW: Grid traffic tracking (for cars)
        self.grid_traffic: Dict[Tuple[int, int, int], int] = defaultdict(int)  # (x, y, time_step) -> traffic_count
        self.grid_width = grid_width
        self.grid_height = grid_height
        
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
    
    def get_edge_travel_time(self, from_node: str, to_node: str, mode: str = 'car') -> float:
        """Get travel time with BPR congestion calculation - mode-specific"""
        
        if mode == 'car':
            # For cars, use grid-based congestion
            return self._get_car_travel_time_from_grid(from_node, to_node)
        else:
            # For public transport, use network-based congestion
            return self._get_network_travel_time(from_node, to_node)
    
    def _get_car_travel_time_from_grid(self, from_node: str, to_node: str) -> float:
        """Calculate car travel time using grid-based BPR congestion"""
        
        # Get spatial coordinates for nodes (you'll need spatial mapper)
        try:
            # This assumes you have a spatial mapper - adjust path as needed
            from_coord = self._get_node_coordinates(from_node)
            to_coord = self._get_node_coordinates(to_node)
        except:
            # Fallback to network-based if coordinates not available
            return self._get_network_travel_time(from_node, to_node)
        
        if not from_coord or not to_coord:
            return self._get_network_travel_time(from_node, to_node)
        
        # Calculate path through grid
        path_cells = self._get_grid_path(from_coord, to_coord)
        total_congested_time = 0.0
        
        current_step = getattr(self, '_current_step', 0)
        
        for cell_x, cell_y in path_cells:
            # Get traffic in this cell (average across recent time steps)
            cell_traffic = 0
            for step in range(max(0, current_step - 5), current_step + 1):
                cell_traffic += self.grid_traffic.get((cell_x, cell_y, step), 0)
            cell_traffic = cell_traffic / 6  # Average
            
            # Apply BPR function to this grid cell
            base_time = self.grid_free_flow_time  # ✅ FIXED: was self.base_travel_speed
            
            if self.grid_capacity <= 0:
                congested_time = base_time
            else:
                # BPR function: t = t0 * (1 + alpha * (v/c)^beta)
                volume_capacity_ratio = cell_traffic / self.grid_capacity
                congestion_factor = 1 + self.alpha * (volume_capacity_ratio ** self.beta)
                congested_time = base_time * congestion_factor
            
            total_congested_time += congested_time
        
        return max(total_congested_time, len(path_cells) * self.grid_free_flow_time)  # ✅ FIXED: was self.base_travel_speed
    
    def _get_network_travel_time(self, from_node: str, to_node: str) -> float:
        """Get travel time using network-based congestion (for public transport)"""
        edge_id = (from_node, to_node)
        
        if edge_id not in self.base_travel_times:
            return 5.0  # Default travel time
        
        base_time = self.base_travel_times[edge_id]
        capacity = self.edge_capacities.get(edge_id, 1000)
        
        # Get current traffic (average across recent time steps)
        current_traffic = 0
        current_step = getattr(self, '_current_step', 0)
        for step in range(max(0, current_step - 5), current_step + 1):
            current_traffic += self.edge_traffic.get((from_node, to_node, step), 0)
        current_traffic = current_traffic / 6  # Average
        
        if capacity <= 0:
            return base_time
        
        # BPR function: t = t0 * (1 + alpha * (v/c)^beta)
        volume_capacity_ratio = current_traffic / capacity
        congestion_factor = 1 + self.alpha * (volume_capacity_ratio ** self.beta)
        
        return base_time * congestion_factor
    
    def _get_grid_path(self, from_coord: Tuple[int, int], to_coord: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get grid path between two coordinates using simple line algorithm"""
        x1, y1 = from_coord
        x2, y2 = to_coord
        
        path = []
        
        # Use Bresenham's line algorithm or simple interpolation
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        
        if dx > dy:
            error = dx / 2.0
            while x != x2:
                path.append((x, y))
                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
        else:
            error = dy / 2.0
            while y != y2:
                path.append((x, y))
                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc
        
        path.append((x2, y2))  # Add destination
        
        # Ensure coordinates are within grid bounds
        bounded_path = []
        for x, y in path:
            x = max(0, min(x, self.grid_width - 1))
            y = max(0, min(y, self.grid_height - 1))
            bounded_path.append((x, y))
        
        return bounded_path
    
    def _get_node_coordinates(self, node_id: str) -> Optional[Tuple[int, int]]:
        """Get grid coordinates for a network node"""
        # This needs to interface with your spatial mapper
        # You'll need to adapt this to your specific spatial mapping system
        try:
            # Try to get from network manager if available
            if hasattr(self, '_spatial_mapper'):
                return self._spatial_mapper.node_to_grid.get(node_id)
            # Fallback - you may need to adapt this
            if hasattr(self.network.nodes[node_id], 'coordinates'):
                return self.network.nodes[node_id].coordinates
        except:
            pass
        return None
    
    def record_trip_on_network(self, network_route: List[str], start_time: int, duration: int, mode: str = 'car'):
        """Record traffic on network or grid based on mode"""
        self._current_step = start_time
        
        if mode == 'car':
            # Record car traffic on grid
            self._record_car_trip_on_grid(network_route, start_time, duration)
        else:
            # Record public transport on network edges
            self._record_network_trip(network_route, start_time, duration)
    
    def _record_car_trip_on_grid(self, network_route: List[str], start_time: int, duration: int):
        """Record car traffic on grid cells"""
        affected_steps = list(range(start_time, start_time + duration))
        
        # Convert network route to grid path
        for i in range(len(network_route) - 1):
            from_node = network_route[i]
            to_node = network_route[i + 1]
            
            from_coord = self._get_node_coordinates(from_node)
            to_coord = self._get_node_coordinates(to_node)
            
            if from_coord and to_coord:
                path_cells = self._get_grid_path(from_coord, to_coord)
                
                # Add traffic to each grid cell for each affected time step
                for cell_x, cell_y in path_cells:
                    for step in affected_steps:
                        self.grid_traffic[(cell_x, cell_y, step)] += 1
    
    def _record_network_trip(self, network_route: List[str], start_time: int, duration: int):
        """Record public transport traffic on network edges"""
        affected_steps = list(range(start_time, start_time + duration))
        
        for i in range(len(network_route) - 1):
            from_node = network_route[i]
            to_node = network_route[i + 1]
            
            for step in affected_steps:
                self.edge_traffic[(from_node, to_node, step)] += 1
                # Also record reverse direction for bidirectional traffic
                self.edge_traffic[(to_node, from_node, step)] += 1
    
    def set_spatial_mapper(self, spatial_mapper):
        """Set spatial mapper for coordinate conversion"""
        self._spatial_mapper = spatial_mapper
    
    def get_grid_congestion_heatmap(self) -> np.ndarray:
        """Get current grid congestion as heatmap for visualization"""
        heatmap = np.zeros((self.grid_height, self.grid_width))
        current_step = getattr(self, '_current_step', 0)
        
        for (x, y, step), traffic in self.grid_traffic.items():
            if step == current_step and 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                heatmap[y, x] = traffic  # Note: y first for proper array indexing
        
        return heatmap
    
    def record_trip_on_grid(self, spatial_route: List[Tuple[int, int]], start_time: int, duration: int, mode: str = 'car'):
        """Record car traffic on grid cells"""
        if mode not in ['car', 'uber_like1', 'uber_like2'] or not spatial_route:
            return
        
        self._current_step = start_time
        affected_steps = list(range(start_time, start_time + duration))
        
        # Record traffic on each grid cell in the route
        for cell in spatial_route:
            x, y = cell
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                for step in affected_steps:
                    self.grid_traffic[(x, y, step)] += 1


class UnifiedNetworkRouter:
    """Fixed router with exact interface from working system"""
    
    def __init__(self, network_graph: nx.Graph, congestion_model: NetworkCongestionModel):
        self.network = network_graph
        self.congestion_model = congestion_model
        self.route_cache = {}

        print(f"🛣️ Unified router initialized for {network_graph.number_of_nodes()} nodes")
    
    def find_shortest_path(self, origin_node: str, destination_node: str, 
                          mode_preference=None, use_congestion: bool = False) -> Optional[Dict]:
        """EXACT METHOD SIGNATURE that MaaS agent expects - now mode-aware"""
        
        # Determine mode for congestion calculation
        if mode_preference and isinstance(mode_preference, list) and len(mode_preference) > 0:
            primary_mode = mode_preference[0] if mode_preference[0] in ['car', 'bike', 'train', 'bus', 'walk'] else 'car'
        else:
            primary_mode = 'car'  # Default to car
        
        cache_key = (origin_node, destination_node, primary_mode, use_congestion)
        
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        try:
            if use_congestion:
                # Use Dijkstra with congestion-aware weights
                def weight_function(u, v, edge_data):
                    return self.congestion_model.get_edge_travel_time(u, v, primary_mode)
                
                path = nx.dijkstra_path(self.network, origin_node, destination_node, weight=weight_function)
            else:
                # Use simple shortest path
                path = nx.shortest_path(self.network, origin_node, destination_node, weight='travel_time')
            
            # Build route details
            route_details = {
                'path': path,
                'segments': [],
                'total_time': 0.0,
                'total_distance': 0.0,
                'modes_used': set(),
                'primary_mode': primary_mode
            }
            
            # Calculate segment details
            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                
                edge_data = self.network[from_node][to_node]
                
                if use_congestion:
                    travel_time = self.congestion_model.get_edge_travel_time(from_node, to_node, primary_mode)
                else:
                    travel_time = edge_data.get('travel_time', 1.0)
                
                segment = {
                    'from_node': from_node,
                    'to_node': to_node,
                    'travel_time': travel_time,
                    'distance': edge_data.get('distance', 1.0),
                    'route_id': edge_data.get('route_id', 'unknown'),
                    'mode': primary_mode
                }
                
                route_details['segments'].append(segment)
                route_details['total_time'] += travel_time
                route_details['total_distance'] += edge_data.get('distance', 1.0)
                
                if 'transport_mode' in edge_data:
                    route_details['modes_used'].add(edge_data['transport_mode'])
            
            # Cache result
            self.route_cache[cache_key] = route_details
            
            # Limit cache size
            if len(self.route_cache) > 1000:
                keys_to_remove = list(self.route_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.route_cache[key]
            
            return route_details
            
        except nx.NetworkXNoPath:
            return None
    
    def find_route_with_congestion(self, origin_node: str, destination_node: str, 
                              current_time: int, mode: str = 'car') -> Optional[Dict]:
        """Congestion-aware routing - now mode-aware for grid vs network congestion"""
        cache_key = (origin_node, destination_node, True, mode, current_time)
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        try:
            # Create temporary graph with congestion-adjusted weights
            temp_graph = self.network.copy()
            
            for u, v, edge_data in temp_graph.edges(data=True):
                # Use mode-aware congestion calculation
                congested_time = self.congestion_model.get_edge_travel_time(u, v, mode)
                temp_graph[u][v]['congested_weight'] = congested_time
            
            path = nx.shortest_path(temp_graph, origin_node, destination_node, weight='congested_weight')
            
            if len(path) < 2:
                return None
            
            # Build route details in the EXACT format your MaaS agent expects
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
                edge_data = self.network[from_node][to_node]
                
                # Use congested travel time
                travel_time = self.congestion_model.get_edge_travel_time(from_node, to_node, mode)
                
                segment = {
                    'from': from_node,
                    'to': to_node,
                    'mode': edge_data.get('transport_mode'),
                    'time': travel_time,
                    'distance': edge_data.get('distance', 1.0),
                    'route_id': edge_data.get('route_id', 'unknown')
                }
                
                route_details['segments'].append(segment)
                route_details['total_time'] += travel_time
                route_details['total_distance'] += edge_data.get('distance', 1.0)
                
                if 'transport_mode' in edge_data:
                    route_details['modes_used'].add(edge_data['transport_mode'])
            
            # Cache result
            self.route_cache[cache_key] = route_details
            
            # Limit cache size
            if len(self.route_cache) > 1000:
                keys_to_remove = list(self.route_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.route_cache[key]
            
            return route_details
            
        except nx.NetworkXNoPath:
            return None
    
    def find_route(self, origin_node: str, destination_node: str, 
                   mode_preference: List = None) -> Optional[Dict]:
        """Compatibility method for standard routing"""
        return self.find_shortest_path(origin_node, destination_node, mode_preference, use_congestion=False)



class UnifiedNetworkManager:
    """Fixed network manager with exact interface from working system"""
    
    def __init__(self, topology_generator, topology_type: str, parameter: int, 
             grid_width: int = 100, grid_height: int = 80):
    
        # Generate the network topology
        self.base_network = topology_generator.base_network
        self.active_network = topology_generator.active_network
        
        # ===== UPDATED: Pass the actual NetworkX graph to spatial mapper =====
        self.spatial_mapper = UnifiedSpatialMapper(self.active_network, grid_width, grid_height)
        self.congestion_model = NetworkCongestionModel(self.active_network)
        self.router = UnifiedNetworkRouter(self.active_network, self.congestion_model)
        self._update_router_network()  # Ensure synchronization
        
        # Performance tracking
        self.route_calculation_count = 0
        
        # Ensure connectivity
        self._ensure_network_connectivity()
        
        # ===== DEBUG: Print node mapping results =====
        sydney_stations = []
        grid_nodes = []
        
        for node_id in self.active_network.nodes():
            if self.spatial_mapper.is_sydney_station(node_id):
                sydney_stations.append(node_id)
            else:
                grid_nodes.append(node_id)
        
        print(f"✅ Unified network manager initialized")
        print(f"   Network: {self.active_network.number_of_nodes()} nodes, {self.active_network.number_of_edges()} edges")
        print(f"   📍 Station mapping: {len(sydney_stations)} Sydney stations, {len(grid_nodes)} grid nodes")
        print(f"   🏢 Sydney stations: {sydney_stations[:10]}...")  # Show first 10
        
        # ===== VERIFICATION: Test coordinate lookup =====
        print(f"   🧪 Testing coordinate lookups:")
        test_coords = [(55, 65), (15, 25), (54, 55)]  # Should be CHATSWOOD, BLACKTOWN, CROWS_NEST
        for coord in test_coords:
            node = self.spatial_mapper.get_nearest_node(coord)
            station_name = self.spatial_mapper.get_station_name_by_coordinates(coord)
            print(f"      {coord} → nearest_node: {node}, station_name: {station_name}")
    def find_route(self, start_pos, end_pos):
        """Find route between grid positions"""
        print(f"[DEBUG] find_route called: {start_pos} → {end_pos}")
        try:
            result = self.find_network_route(start_pos, end_pos)
            print(f"[DEBUG] find_route result: {'Found' if result else 'None'}")
            return result
        except Exception as e:
            print(f"[ERROR] find_route failed: {e}")
            return None
    
    # def find_network_route(self, origin_grid: Tuple[int, int], 
    #                       destination_grid: Tuple[int, int], 
    #                       mode: str = 'car', current_time: int = 0) -> Optional[Dict]:
    #     """EXACT METHOD from your working system"""
    #     self.route_calculation_count += 1
        
    #     # Find nearest network nodes
    #     origin_node = self._find_accessible_node(origin_grid)
    #     destination_node = self._find_accessible_node(destination_grid)
        
    #     if not origin_node or not destination_node:
    #         print(f"No accessible network nodes found for {origin_grid} -> {destination_grid}")
    #         return None
        
    #     # If same node, create direct route
    #     if origin_node == destination_node:
    #         origin_coord = self.spatial_mapper.node_to_grid.get(origin_node, origin_grid)
    #         dest_coord = self.spatial_mapper.node_to_grid.get(destination_node, destination_grid)
    #         return {
    #             'network_route': {'path': [origin_node]},
    #             'spatial_route': [origin_grid, origin_coord, dest_coord, destination_grid],
    #             'access_node': origin_node,
    #             'egress_node': destination_node
    #         }
        
    #     # Route on network with or without congestion
    #     if mode in ['car', 'bike']:  # Apply congestion for road-based modes
    #         network_route = self.router.find_route_with_congestion(
    #             origin_node, destination_node, current_time
    #         )
    #     else:  # No congestion for public transport
    #         network_route = self.router.find_shortest_path(origin_node, destination_node)
        
    #     if not network_route:
    #         print(f"No network route found from {origin_node} to {destination_node}")
    #         return None
        
    #     # Convert to spatial route
    #     spatial_route = self._convert_to_spatial_route(network_route, origin_grid, destination_grid)
        
    #     result = {
    #         'network_route': network_route,
    #         'spatial_route': spatial_route,
    #         'access_node': origin_node,
    #         'egress_node': destination_node
    #     }
        
    #     return result
    
    def _update_router_network(self):
        """Update router's network reference after topology changes"""
        if hasattr(self, 'router'):
            self.router.network = self.active_network
            self.router.congestion_model.network = self.active_network
            # Clear router's cache since network changed
            if hasattr(self.router, 'route_cache'):
                self.router.route_cache.clear()
            if hasattr(self.router, 'shortest_paths_cache'):
                self.router.shortest_paths_cache.clear()

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
    
    def _create_direct_route(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create direct route between two points"""
        return [start_pos, end_pos]
    
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
        from topology.network_topology import TransportMode
        
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
    
    def get_network_stats(self) -> Dict:
        """Get network statistics"""
        return {
            'num_nodes': self.active_network.number_of_nodes(),
            'num_edges': self.active_network.number_of_edges(),
            'is_connected': nx.is_connected(self.active_network),
            'topology_type': getattr(self, 'topology_type', 'unified')
        }


# ===== BACKWARD COMPATIBILITY =====
class TwoLayerNetworkManager:
    """
    Backward compatibility class that replicates the exact interface from your working system.
    Uses the fixed components but maintains the original API.
    """
    def __init__(self, topology_type: str = "degree_constrained", 
             degree: int = 3, grid_width: int = 100, grid_height: int = 80,
             rewiring_probability: float = 0.1, initial_neighbors: int = 4,
             attachment_parameter: int = 2, connectivity_level: int = 4,
             **kwargs):
        """Initialize with guaranteed two-layer architecture"""
        
        # CRITICAL: Always create base network first
        from topology.network_topology import SydneyNetworkTopology
        self.base_network = SydneyNetworkTopology()
        self.base_network.initialize_base_sydney_network()
        
        # Validate base network creation
        if not self._validate_base_network_exists():
            raise RuntimeError("Failed to create valid base Sydney network")
        
        print(f"   ✅ Base network validated: {self.base_network.graph.number_of_edges()} edges")
        
        # Store configuration
        self.topology_type = topology_type
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Generate topology using UNIFIED GENERATOR with two-layer architecture
        self.active_network = self._create_topology_network(topology_type, locals())
        
        # Final validation - ensure base routes still exist
        self._validate_active_network_integrity()
        
        # Initialize components with validated network
        self._initialize_network_components()

    def _create_topology_network(self, topology_type: str, params: dict) -> nx.Graph:
        """Create topology network using unified generator with two-layer architecture"""
        print(f"🔧 _create_topology_network called with topology_type='{topology_type}'")
        print(f"🔧 params = {params}")
        from topology.unified_topology_generator import UnifiedTopologyGenerator
        
        # Initialize unified generator with base network
        generator = UnifiedTopologyGenerator(self.base_network)
        print(f"🔧 UnifiedTopologyGenerator initialized")
        try:
            if topology_type == "degree_constrained":
                degree = params.get('degree', 3)
                print(f"🔧 About to call generator.generate_topology('degree_constrained', {degree})")
                active_network = generator.generate_topology('degree_constrained', degree)
                
            elif topology_type == "small_world":
                rewiring_prob = params.get('rewiring_probability', 0.1)
                initial_neighbors = params.get('initial_neighbors', 4)
                active_network = generator.generate_topology(
                    'small_world', 
                    rewiring_prob,
                    initial_neighbors=initial_neighbors
                )
                
            elif topology_type == "scale_free":
                attachment_param = params.get('attachment_parameter', 2)
                active_network = generator.generate_topology('scale_free', attachment_param)
                
            else:
                # Fallback to base network only
                print(f"⚠️ Unknown topology type {topology_type}, using base network only")
                active_network = self.base_network.graph.copy()
            
            print(f"   ✅ Topology network created: {active_network.number_of_edges()} total edges")
            return active_network
            
        except Exception as e:
            print(f"❌ Failed to create topology network: {e}")
            print(f"   Falling back to base network only")
            return self.base_network.graph.copy()


    def _validate_base_network_exists(self) -> bool:
        """Validate that base Sydney network has required routes"""
        required_routes = ['T1_WESTERN', 'T4_ILLAWARRA', 'BUS_380']
        found_routes = []
        
        for u, v, data in self.base_network.graph.edges(data=True):
            route_id = data.get('route_id', '')
            for required in required_routes:
                if required in route_id and required not in found_routes:
                    found_routes.append(required)
        
        success = len(found_routes) >= len(required_routes)
        print(f"   Base network validation: {found_routes} (success: {success})")
        return success

    def _validate_active_network_integrity(self):
        """Validate that active network preserves base Sydney routes"""
        
        base_routes = []
        active_routes = []
        
        # Count base routes
        for u, v, data in self.base_network.graph.edges(data=True):
            route_id = data.get('route_id', '')
            if any(critical in route_id for critical in ['T1_WESTERN', 'T4_ILLAWARRA', 'BUS_380']):
                base_routes.append(route_id)
        
        # Count preserved routes in active network
        for u, v, data in self.active_network.edges(data=True):
            route_id = data.get('route_id', '')
            if any(critical in route_id for critical in ['T1_WESTERN', 'T4_ILLAWARRA', 'BUS_380']):
                active_routes.append(route_id)
        
        preserved_count = len(active_routes)
        base_count = len(base_routes)
        
        print(f"   Network integrity: {preserved_count}/{base_count} base routes preserved")
        
        if preserved_count == 0:
            print("❌ CRITICAL: No base routes preserved! Forcing base route addition...")
            self._force_add_missing_base_routes()

    def _force_add_missing_base_routes(self):
        """Force addition of missing base routes to active network"""
        
        for u, v, data in self.base_network.graph.edges(data=True):
            route_id = data.get('route_id', '')
            if any(critical in route_id for critical in ['T1_WESTERN', 'T4_ILLAWARRA', 'BUS_380']):
                if not self.active_network.has_edge(u, v):
                    print(f"   Adding missing base route: {route_id} ({u} -> {v})")
                    self.active_network.add_edge(u, v, **data)

    def _initialize_network_components(self):
        """Initialize network components with validated active network"""
        
        # Initialize spatial mapper - FIX: Use correct class name
        self.spatial_mapper = UnifiedSpatialMapper(self.active_network, self.grid_width, self.grid_height)
        self.congestion_model = NetworkCongestionModel(self.active_network)
        
        print(f"✅ Network components initialized")
        # Initialize congestion model - FIX: Use correct class name  
        try:
            self.congestion_model = NetworkCongestionModel(self.active_network)
            print(f"   ✅ Congestion model initialized")
        except Exception as e:
            print(f"⚠️ Congestion model initialization failed: {e}")
        

    
    # def find_network_route(self, origin_grid: Tuple[int, int], 
    #                   destination_grid: Tuple[int, int], 
    #                   mode: str = 'car', current_time: int = 0) -> Optional[Dict]:
    #     """EXACT METHOD from your working system - now with grid congestion for cars"""
    #     self.route_calculation_count += 1
        
    #     # Find nearest network nodes
    #     origin_node = self._find_accessible_node(origin_grid)
    #     destination_node = self._find_accessible_node(destination_grid)
        
    #     if not origin_node or not destination_node:
    #         print(f"No accessible network nodes found for {origin_grid} -> {destination_grid}")
    #         return None
        
    #     # If same node, create direct route
    #     if origin_node == destination_node:
    #         origin_coord = self.spatial_mapper.node_to_grid.get(origin_node, origin_grid)
    #         dest_coord = self.spatial_mapper.node_to_grid.get(destination_node, destination_grid)
    #         return {
    #             'network_route': {'path': [origin_node]},
    #             'spatial_route': [origin_grid, origin_coord, dest_coord, destination_grid],
    #             'access_node': origin_node,
    #             'egress_node': destination_node
    #         }
        
    #     # Route on network with or without congestion
    #     if mode in ['car', 'bike', 'uber_like1', 'uber_like2']:  # Apply congestion for road-based modes
    #         network_route = self.router.find_route_with_congestion(
    #             origin_node, destination_node, current_time, mode  # Pass mode parameter
    #         )
    #     else:  # No congestion for public transport
    #         network_route = self.router.find_shortest_path(origin_node, destination_node)
        
    #     if not network_route:
    #         print(f"No network route found from {origin_node} to {destination_node}")
    #         return None
        
    #     # Convert to spatial route
    #     spatial_route = self._convert_to_spatial_route(network_route, origin_grid, destination_grid)
        
    #     result = {
    #         'network_route': network_route,
    #         'spatial_route': spatial_route,
    #         'access_node': origin_node,
    #         'egress_node': destination_node
    #     }
        
    #     return result
    
    # def _find_accessible_node(self, grid_position: Tuple[int, int], max_search_radius: int = 10) -> Optional[str]:
    #     """Find accessible network node with expanding search"""
        
    #     # First try: direct lookup
    #     nearest_node = self.spatial_mapper.get_nearest_node(grid_position)
    #     if nearest_node:
    #         return nearest_node
        
    #     # Second try: expanding radius search
    #     for radius in range(1, max_search_radius + 1):
    #         x, y = grid_position
            
    #         for dx in range(-radius, radius + 1):
    #             for dy in range(-radius, radius + 1):
    #                 if dx*dx + dy*dy <= radius*radius:  # Within circular radius
    #                     search_x = max(0, min(self.spatial_mapper.grid_width - 1, x + dx))
    #                     search_y = max(0, min(self.spatial_mapper.grid_height - 1, y + dy))
                        
    #                     search_pos = (search_x, search_y)
    #                     node = self.spatial_mapper.get_nearest_node(search_pos)
    #                     if node:
    #                         return node
        
    #     # Third try: any available node (fallback)
    #     available_nodes = list(self.spatial_mapper.node_to_grid.keys())
    #     if available_nodes:
    #         print(f"Using fallback node for {grid_position}")
    #         return available_nodes[0]
        
    #     return None
    

    # def _convert_to_spatial_route(self, network_route: Dict, 
    #                             origin_grid: Tuple[int, int], 
    #                             destination_grid: Tuple[int, int]) -> List[Tuple[int, int]]:
    #     """Convert network route to spatial coordinates"""
    #     spatial_route = [origin_grid]
        
    #     for node_id in network_route['path']:
    #         if node_id in self.spatial_mapper.node_to_grid:
    #             spatial_route.append(self.spatial_mapper.node_to_grid[node_id])
        
    #     if destination_grid != spatial_route[-1]:
    #         spatial_route.append(destination_grid)
        
    #     return spatial_route
    
    def get_transfer_stations(self):
        """Identify transfer stations where multiple routes intersect - for topology networks"""
        
        transfer_stations = {}
        
        # Find nodes that have edges with different route_ids
        for node in self.active_network.nodes():
            connected_routes = set()
            
            for neighbor in self.active_network.neighbors(node):
                edge_data = self.active_network[node][neighbor]
                route_id = edge_data.get('route_id', 'unknown')
                connected_routes.add(route_id)
            
            # If multiple routes meet here, it's a transfer station
            if len(connected_routes) > 1:
                transfer_stations[node] = {
                    'routes': list(connected_routes),
                    'transfer_time': 0.8,  # Default transfer time
                    'node_type': self.base_network.nodes.get(node, {}).get('node_type', 'unknown')
                }
        
        print(f"✅ Identified {len(transfer_stations)} transfer stations in network")
        return transfer_stations

    def find_network_route(self, origin, destination, mode='public', current_time=0):
        """Simple route finding for spatial modes"""
        
        if mode in ['car', 'bike', 'walk']:
            # Return simple spatial route
            return self._find_spatial_route(origin, destination, mode, current_time)
        else:
            # For public transport, use the router
            try:
                start_node = self.spatial_mapper.get_nearest_node(origin)
                end_node = self.spatial_mapper.get_nearest_node(destination)
                
                if not start_node or not end_node:
                    return None
                    
                # Use the router for public transport
                route_details = self.router.find_shortest_path(start_node, end_node)
                
                if route_details and 'path' in route_details:
                    return {
                        'network_route': route_details,
                        'spatial_route': self._convert_to_spatial_route(route_details, origin, destination),
                        'access_node': start_node,
                        'egress_node': end_node
                    }
                else:
                    return None
                    
            except Exception as e:
                print(f"Public transport routing error: {e}")
                return None
    

    def _find_spatial_route(self, origin, destination, mode, current_time):
        """Find route through spatial grid (for car/bike/walk)"""
        
        # Use the network router for cars to get a proper path that considers congestion
        if mode in ['car', 'uber_like1', 'uber_like2']:
            try:
                start_node = self.spatial_mapper.get_nearest_node(origin)
                end_node = self.spatial_mapper.get_nearest_node(destination)

                if not start_node or not end_node:
                    print(f"Could not find network nodes for car route from {origin} to {destination}")
                    return None

                # Use the router to find a congestion-aware route on the network
                route_details = self.router.find_route_with_congestion(start_node, end_node, current_time, mode)

                if route_details and 'path' in route_details:
                    # Convert the network path back to a spatial path for the agent
                    spatial_route = self._convert_to_spatial_route(route_details, origin, destination)
                    return {
                        'network_route': route_details,
                        'spatial_route': spatial_route,
                        'access_node': start_node,
                        'egress_node': end_node,
                        'mode': mode
                    }
                else:
                     # Fallback to simple grid path if network routing fails
                    print(f"Network routing failed for car from {origin} to {destination}, falling back to grid path.")
                    pass
            except Exception as e:
                print(f"Car routing failed with error: {e}, falling back to simple grid path.")

        # For non-car modes or as a fallback, use Bresenham's line algorithm for a direct path
        spatial_route = []
        x1, y1 = origin
        x2, y2 = destination
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        current = (x1, y1)
        
        while True:
            spatial_route.append(current)
            if current == destination:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy
            current = (x1, y1)

            if len(spatial_route) > 500:  # Increased safety check for longer routes
                print(f"Spatial route generation exceeded safety limit for mode {mode}")
                break
        
        return {
            'spatial_route': spatial_route,
            'mode': mode
        }

    def record_trip_on_network(self, origin: Tuple[int, int], destination: Tuple[int, int], 
                          start_time: int, duration: int, mode: str = 'car'):
        """Record trip for congestion tracking - mode-aware"""
        try:
            route_result = self.find_network_route(origin, destination, mode, start_time)
            if route_result:
                # Record on appropriate system based on mode
                if mode in ['car', 'uber_like1', 'uber_like2']:
                    # Record on grid for car traffic
                    spatial_route = route_result.get('spatial_route', [])
                    self.congestion_model.record_trip_on_grid(spatial_route, start_time, duration, mode)
                else:
                    # Record on network for public transport
                    network_route = route_result.get('network_route', {}).get('path', [])
                    self.congestion_model.record_trip_on_network(network_route, start_time, duration)
        except Exception as e:
            print(f"Error recording trip: {e}")

    
    # def get_network_statistics(self) -> Dict:
    #     """Get network topology statistics"""
    #     graph = self.active_network
        
    #     stats = {
    #         'num_nodes': graph.number_of_nodes(),
    #         'num_edges': graph.number_of_edges(),
    #         'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
    #         'is_connected': nx.is_connected(graph),
    #         'route_calculations': self.route_calculation_count,
    #         'topology_type': self.topology_type
    #     }
        
    #     if nx.is_connected(graph):
    #         stats['diameter'] = nx.diameter(graph)
    #         stats['avg_path_length'] = nx.average_shortest_path_length(graph)
    #     else:
    #         stats['diameter'] = float('inf')
    #         stats['avg_path_length'] = float('inf')
        
    #     return stats
    
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
        try:
            from topology.network_topology import TransportMode
        except ImportError:
            # Fallback if TransportMode not available
            class TransportMode:
                BUS = "bus"
                TRAIN = "train"
        
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