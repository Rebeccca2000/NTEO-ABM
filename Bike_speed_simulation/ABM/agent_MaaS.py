from mesa import Agent
import heapq
import math
from mesa.space import MultiGrid
from ABM.agent_service_provider_initialisation import SubsidyUsageLog, ShareServiceBookingLog, ServiceBookingLog
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, func
import uuid
import random
import networkx as nx

class MaaS(Agent):
    def __init__(self, unique_id, model, service_provider_agent, commuter_agents, 
             network_manager=None, DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=None, 
             BACKGROUND_TRAFFIC_AMOUNT=None, stations=None, routes=None, transfers=None, 
             num_commuters=None, grid_width=None, grid_height=None, CONGESTION_ALPHA=None,
             CONGESTION_BETA=None, CONGESTION_CAPACITY=None, CONGESTION_T_IJ_FREE_FLOW=None,
             subsidy_config=None, schema=None):
    
        super().__init__(unique_id, model)
        
        # Core agent references
        self.service_provider_agent = service_provider_agent
        self.commuter_agents = commuter_agents
        
        # Network topology manager (PRIMARY ROUTING SYSTEM)
        self.network_manager = network_manager
        if not network_manager:
            raise ValueError("NetworkManager is required - legacy routing removed")
        
        # Legacy compatibility - populated from network
        self.stations = stations 
        self.routes = routes 
        self.transfers = transfers
        
        # Grid and spatial parameters
        self.grid_width = grid_width or 100
        self.grid_height = grid_height or 80
        self.grid = MultiGrid(width=self.grid_width, height=self.grid_height, torus=False)
        
        # Database setup
        self.db_engine = create_engine(self.model.db_connection_string)
        self.Session = service_provider_agent.Session
        self.schema = schema
        
        # Configuration parameters
        self.dynamic_maas_surcharge_base_coefficient = DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS
        self.background_traffic_amount = BACKGROUND_TRAFFIC_AMOUNT
        self.num_commuters = num_commuters
        
        # Subsidy management
        self.subsidy_config = subsidy_config
        self.current_subsidy_pool = subsidy_config.total_amount if subsidy_config else 0
        self.last_reset_step = 0
        self.total_subsidies_given = 0
        
        # Performance tracking
        self.routing_performance = {
            'network_routes': 0,
            'cache_hits': 0,
            'failed_routes': 0,
            'routes_calculated': 0,  # For successful route calculations
            'total_time': 0,  # If you're tracking time
            'cache_misses': 0,  # If you're tracking misses
        }
        
        # Simplified cache system
        self._cache = {
            'routes': {},
            'prices': {}
        }
        self._cache_limits = {
            'routes': 200,
            'prices': 100
        }
        
        print(f"MaaS agent initialized with network topology routing only")

    def find_nearest_station_any_mode(self, point):
        """Use network-based station finding"""
        nearest_node = self.network_manager.spatial_mapper.get_nearest_node(point)
        if not nearest_node:
            return None, None
        
        # Determine mode based on network node data
        node_data = self.network_manager.base_network.nodes.get(nearest_node)
        if node_data and node_data.transport_modes:
            primary_mode = node_data.transport_modes[0].value
            return primary_mode, nearest_node
        
        return 'bus', nearest_node  # Default fallback

    def check_is_peak(self, current_ticks):
        if (36 <= current_ticks % 144 < 60) or (90 <= current_ticks % 144 < 114):
            return True
        return False

    def insert_time_varying_traffic(self, session):
        """Insert background traffic using network-based routing"""
        current_step = self.model.get_current_step()
        ticks_in_day = 144
        current_day_tick = current_step % ticks_in_day
        
        # Define traffic intensity based on time of day
        if 36 <= current_day_tick < 48:
            traffic_multiplier = 3.0
        elif 48 <= current_day_tick < 60:
            traffic_multiplier = 2.5
        elif 90 <= current_day_tick < 102:
            traffic_multiplier = 2.5
        elif 102 <= current_day_tick < 114:
            traffic_multiplier = 3.0
        elif (60 <= current_day_tick < 90) or (114 <= current_day_tick < 130):
            traffic_multiplier = 1.0
        else:
            traffic_multiplier = 0.3
        
        adjusted_traffic_amount = int(self.background_traffic_amount * traffic_multiplier)
        
        if adjusted_traffic_amount <= 0:
            return 0
        
        new_bookings = []
        is_peak = self.check_is_peak(current_step)
        base_start_time = current_step
        
        # Generate network-based background traffic
        for _ in range(adjusted_traffic_amount):
            # Generate random start and end grid points
            start_x, start_y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
            end_x, end_y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
            
            start_grid = (start_x, start_y)
            end_grid = (end_x, end_y)
            
            # Use network routing for background traffic
            network_result = self.network_manager.find_network_route(start_grid, end_grid, mode='car', current_time=current_step)
            
            if network_result and 'spatial_route' in network_result:
                route = network_result['spatial_route']
                
                # Calculate duration based on network route
                route_length = len(route) - 1 if len(route) > 1 else 1
                duration = max(1, route_length // 6)
                if is_peak:
                    duration = int(duration * 1.3)
                
                start_time = base_start_time + random.randint(0, 2)
                affected_steps = list(range(start_time, start_time + duration))
                
                # Record traffic on network
                self.network_manager.record_trip_on_network(start_grid, end_grid, start_time, duration)
                
                # Create booking record
                new_booking = ShareServiceBookingLog(
                    commuter_id=-1,
                    request_id=str(uuid.uuid4()),
                    mode_id=0,
                    provider_id=0,
                    company_name="background_traffic",
                    start_time=start_time,
                    duration=duration,
                    affected_steps=affected_steps,
                    route_details=route
                )
                new_bookings.append(new_booking)
        
        if new_bookings:
            session.add_all(new_bookings)
            session.commit()
        
        return adjusted_traffic_amount

    def calculate_single_mode_time_and_price(self, origin, destination, unit_price, unit_speed, mode_id):
        """Calculate route for SINGLE modes (car/bike/walk) using SPATIAL GRID - FIXED"""
        
        cache_key = f"single_{origin}_{destination}_{mode_id}_{unit_price}_{unit_speed}"
        if cache_key in self._cache.get('routes', {}):
            return self._cache['routes'][cache_key]
        
        try:
            current_time = self.model.get_current_step() if hasattr(self, 'model') else 0
            
            # Map mode_id to mode string
            mode_map = {3: 'car', 4: 'bike', 5: 'walk'}
            mode = mode_map.get(mode_id, 'walk')
            
            # Use SPATIAL routing for car/bike/walk (NOT network edges)
            network_result = self.network_manager.find_network_route(
                origin, destination, 
                mode=mode, 
                current_time=current_time
            )
            
            if network_result and 'spatial_route' in network_result:
                spatial_route = network_result['spatial_route']
                
                # Calculate metrics based on spatial path
                total_length = len(spatial_route) - 1 if len(spatial_route) > 1 else 1
                
                # Apply congestion for cars
                if mode_id == 3:  # Car
                    # Use the proper BPR implementation
                    total_time = self.network_manager.congestion_model.get_edge_travel_time(
                        str(origin), str(destination), mode='car'
                    )
                    # Record car trip for congestion modeling
                    duration = max(1, round(total_time))
                    self.network_manager.record_trip_on_network(origin, destination, current_time, duration)
                else:
                    total_time = total_length / unit_speed
                
                total_price = unit_price * total_length
                
                result = (spatial_route, total_time, total_price)
                self._update_cache('routes', cache_key, result)
                return result
            else:
                # Fallback to Manhattan distance
                return self._calculate_manhattan_fallback(origin, destination, unit_price, unit_speed)
                
        except Exception as e:
            print(f"Error in single mode calculation: {e}")
            return self._calculate_manhattan_fallback(origin, destination, unit_price, unit_speed)

    def _calculate_manhattan_fallback(self, origin, destination, unit_price, unit_speed):
        """Simple Manhattan distance fallback"""
        distance = abs(origin[0] - destination[0]) + abs(origin[1] - destination[1])
        route = [origin, destination]
        time = distance / unit_speed
        price = unit_price * distance
        return (route, time, price)
    


    def get_current_usage(self):
        """Calculate active commuters"""
        active_commuters = 0
        for commuter in self.commuter_agents:
            if any(request['status'] == 'active' for request in commuter.requests.values()):
                active_commuters += 1
        return active_commuters

    def calculate_dynamic_MaaS_surcharge(self, payment_scheme):
        coefficients = self.dynamic_maas_surcharge_base_coefficient
        S_base = coefficients['S_base']
        alpha = coefficients['alpha']
        delta = coefficients['delta']

        current_usage = self.get_current_usage()
        system_capacity = self.num_commuters

        UR = current_usage / system_capacity
        S_dynamic = S_base * (1 - alpha * (1 - UR))
        S_dynamic = max(0, min(S_dynamic, S_base * 1.5))

        if payment_scheme == 'PAYG':
            surcharge_percentage = S_dynamic
        elif payment_scheme == 'subscription':
            surcharge_percentage = S_base * delta
        else:
            surcharge_percentage = 0

        return surcharge_percentage

    # def find_optimal_route(self, origin_point, destination_point, route_type='public'):
    #     """Find optimal PUBLIC TRANSPORT route with transfers - FIXED for topologies"""
    #     print(f"\n🔍 DEBUG find_optimal_route:")
    #     print(f"   Origin: {origin_point}, Destination: {destination_point}")
    
    #     cache_key = f"route_{origin_point}_{destination_point}_{route_type}"
    #     if cache_key in self._cache.get('routes', {}):
    #         return self._cache['routes'][cache_key]
        
    #     try:
    #         # Get nearest network nodes (stations)
    #         start_node = self.network_manager.spatial_mapper.get_nearest_node(origin_point)
    #         end_node = self.network_manager.spatial_mapper.get_nearest_node(destination_point)
    #         print(f"   Mapped nodes: {start_node} -> {end_node}")
    #         # Debug: Check available edges from start node
    #         if start_node in self.network_manager.active_network:
    #             neighbors = list(self.network_manager.active_network.neighbors(start_node))
    #             print(f"   Start node neighbors: {neighbors[:5]}...")
                
    #             for neighbor in neighbors[:3]:  # Check first 3 edges
    #                 edge_data = self.network_manager.active_network[start_node][neighbor]
    #                 print(f"   Edge {start_node}->{neighbor}:")
    #                 print(f"     route_id: {edge_data.get('route_id', 'MISSING!')}")
    #                 print(f"     mode: {edge_data.get('transport_mode', 'MISSING!')}")
            
    #         # Continue with existing routing logic...
    #         route_details = self.network_manager.router.find_shortest_path(
    #             start_node, end_node, mode_preference=None
    #         )
            
    #         if route_details and 'path' in route_details:
    #             path = route_details['path']
    #             print(f"   Found path with {len(path)} nodes")
                
    #             # Debug: Check edges in path
    #             for i in range(len(path)-1):
    #                 if path[i] in self.network_manager.active_network:
    #                     if path[i+1] in self.network_manager.active_network[path[i]]:
    #                         edge_data = self.network_manager.active_network[path[i]][path[i+1]]
    #                         route_id = edge_data.get('route_id', 'MISSING')
    #                         print(f"   Path edge {path[i]}->{path[i+1]}: route_id={route_id}")
               
    #         if not start_node or not end_node:
    #             print(f"⚠️ No stations found near {origin_point} -> {destination_point}")
    #             return None
            
    #         # Use Dijkstra with transfer penalties for realistic routing
    #         import heapq
            
    #         # Initialize tracking structures
    #         best_times = {node: float('inf') for node in self.network_manager.active_network.nodes()}
    #         best_times[start_node] = 0
    #         best_paths = {}
    #         best_routes = {}  # Track which route was used to reach each node
            
    #         # Priority queue: (total_time, current_node, previous_route_id)
    #         queue = [(0, start_node, None)]
    #         visited = set()
            
    #         # Transfer penalty (in minutes)
    #         TRANSFER_PENALTY = 0.2  # Configurable based on research needs
            
    #         while queue:
    #             current_time, current_node, previous_route = heapq.heappop(queue)
                
    #             if current_node in visited:
    #                 continue
    #             visited.add(current_node)
                
    #             # Early termination if destination reached
    #             if current_node == end_node:
    #                 break
                
    #             # Explore all edges from current node
    #             for neighbor in self.network_manager.active_network.neighbors(current_node):
    #                 edge_data = self.network_manager.active_network[current_node][neighbor]
                    
    #                 # Extract edge attributes (works with topology-generated edges)
    #                 travel_time = edge_data.get('travel_time', 10)
    #                 route_id = edge_data.get('route_id', 'unknown')
    #                 frequency = edge_data.get('frequency', 15)
                    
    #                 # Calculate wait time (average wait = frequency/2)
    #                 wait_time = frequency / 2 if current_node == start_node else 0
                    
    #                 # Add transfer penalty if changing routes
    #                 transfer_time = 0
    #                 if previous_route and previous_route != route_id:
    #                     transfer_time = TRANSFER_PENALTY
    #                     # print(f"  Transfer at {current_node}: {previous_route} -> {route_id}")
                    
    #                 # Total time to reach neighbor
    #                 arrival_time = current_time + wait_time + travel_time + transfer_time
                    
    #                 # Update if better path found
    #                 if arrival_time < best_times[neighbor]:
    #                     best_times[neighbor] = arrival_time
    #                     best_paths[neighbor] = current_node
    #                     best_routes[neighbor] = route_id
    #                     heapq.heappush(queue, (arrival_time, neighbor, route_id))
            
    #         # Reconstruct path
    #         if end_node not in best_paths:
    #             print(f"⚠️ No public transport path found")
    #             return None
            
    #         path = []
    #         current = end_node
    #         while current != start_node:
    #             path.append(current)
    #             current = best_paths.get(current)
    #             if current is None:
    #                 return None
    #         path.append(start_node)
    #         path.reverse()
            
    #         # Cache and return
    #         self._update_cache('routes', cache_key, path)

    #         return path
            
    #     except Exception as e:
    #         print(f"❌ Public transport routing error: {e}")
    #         return None

    def find_optimal_route(self, origin_point, destination_point, route_type='public'):
        """Find optimal PUBLIC TRANSPORT route with transfers - FIXED for topologies"""
        
        # Initialize cache if needed
        if 'routes' not in self._cache:
            self._cache['routes'] = {}
        
        # Check cache first
        cache_key = f"route_{origin_point}_{destination_point}_{route_type}"
        if cache_key in self._cache['routes']:
            self.routing_performance['cache_hits'] += 1
            return self._cache['routes'][cache_key]
        
        # Debug flag - set to True when debugging
        DEBUG = False  # Change to False in production
        
        if DEBUG:
            print(f"\n🔍 DEBUG find_optimal_route:")
            print(f"   Origin: {origin_point}, Destination: {destination_point}")
        
        try:
            # Step 1: Map grid coordinates to network nodes (stations)
            start_node = self.network_manager.spatial_mapper.get_nearest_node(origin_point)
            end_node = self.network_manager.spatial_mapper.get_nearest_node(destination_point)
            
            if not start_node or not end_node:
                if DEBUG:
                    print(f"⚠️ No stations found near {origin_point} -> {destination_point}")
                return None
            
            if DEBUG:
                print(f"   Mapped nodes: {start_node} -> {end_node}")
                self._debug_node_edges(start_node)
            
            # Step 2: Run Dijkstra's algorithm with transfer penalties
            path = self._dijkstra_with_transfers(start_node, end_node, DEBUG)
            
            if not path:
                if DEBUG:
                    print(f"⚠️ No public transport path found")
                return None
            
            # Step 3: Debug path edges if needed
            if DEBUG:
                self._debug_path_edges(path)
            
            # Cache the result
            self._cache['routes'][cache_key] = path
            self.routing_performance['routes_calculated'] += 1
            
            return path
            
        except Exception as e:
            print(f"❌ Public transport routing error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _dijkstra_with_transfers(self, start_node, end_node, debug=False):
        """Dijkstra's algorithm with transfer penalties for realistic public transport routing"""
        import heapq
        
        # Configuration
        TRANSFER_PENALTY = 0.2  # minutes - configurable for research
        DEFAULT_TRAVEL_TIME = 10  # fallback if edge missing travel_time
        DEFAULT_FREQUENCY = 15  # fallback if edge missing frequency
        
        # Initialize tracking structures
        network = self.network_manager.active_network
        best_times = {node: float('inf') for node in network.nodes()}
        best_times[start_node] = 0
        best_paths = {}
        best_routes = {}  # Track route_id used to reach each node
        
        # Priority queue: (total_time, current_node, previous_route_id)
        queue = [(0, start_node, None)]
        visited = set()
        
        while queue:
            current_time, current_node, previous_route = heapq.heappop(queue)
            
            # Skip if already visited
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # Early termination if destination reached
            if current_node == end_node:
                break
            
            # Explore all neighbors
            if current_node not in network:
                continue
                
            for neighbor in network.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                # Get edge data
                edge_data = network[current_node][neighbor]
                
                # Extract attributes with fallbacks
                travel_time = edge_data.get('travel_time', DEFAULT_TRAVEL_TIME)
                route_id = edge_data.get('route_id', f'UNKNOWN_{current_node}_{neighbor}')
                frequency = edge_data.get('frequency', DEFAULT_FREQUENCY)
                
                # Calculate components of travel time
                # 1. Wait time (only at start)
                wait_time = frequency / 2 if current_node == start_node else 0
                
                # 2. Transfer penalty (if changing routes)
                transfer_time = 0
                if previous_route and previous_route != route_id:
                    transfer_time = TRANSFER_PENALTY
                    if debug and transfer_time > 0:
                        print(f"      Transfer at {current_node}: {previous_route} -> {route_id}")
                
                # 3. Total time to reach neighbor
                arrival_time = current_time + wait_time + travel_time + transfer_time
                
                # Update if better path found
                if arrival_time < best_times[neighbor]:
                    best_times[neighbor] = arrival_time
                    best_paths[neighbor] = current_node
                    best_routes[neighbor] = route_id
                    heapq.heappush(queue, (arrival_time, neighbor, route_id))
        
        # Reconstruct path
        if end_node not in best_paths:
            return None
        
        path = []
        current = end_node
        while current != start_node:
            path.append(current)
            current = best_paths.get(current)
            if current is None:
                return None
        path.append(start_node)
        path.reverse()
        
        return path

    def _debug_node_edges(self, node):
        """Debug helper: Check edges from a node"""
        network = self.network_manager.active_network
        
        if node not in network:
            print(f"   ❌ Node {node} not in network!")
            return
        
        neighbors = list(network.neighbors(node))
        print(f"   Node {node} has {len(neighbors)} neighbors")
        
        # Show first 3 edges
        for neighbor in neighbors[:3]:
            edge_data = network[node][neighbor]
            route_id = edge_data.get('route_id', 'MISSING!')
            mode = edge_data.get('transport_mode', 'MISSING!')
            print(f"     → {neighbor}: route_id={route_id}, mode={mode}")

    def _debug_path_edges(self, path):
        """Debug helper: Check all edges in a path"""
        network = self.network_manager.active_network
        print(f"   Path has {len(path)} nodes")
        
        topology_edges_used = 0
        base_edges_used = 0
        
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if current in network and next_node in network[current]:
                edge_data = network[current][next_node]
                route_id = edge_data.get('route_id', 'MISSING')
                
                # Categorize edge type
                if any(x in route_id for x in ['_DC', '_SW', '_SF', '_EQ', '_TOPOLOGY']):
                    topology_edges_used += 1
                    print(f"     ✅ TOPOLOGY edge: {current}→{next_node}: {route_id}")
                elif any(x in route_id for x in ['T1_', 'T4_', 'T8_', 'BUS_380']):
                    base_edges_used += 1
                    print(f"     Base edge: {current}→{next_node}: {route_id}")
                else:
                    print(f"     ??? Unknown edge: {current}→{next_node}: {route_id}")
        
        print(f"   Path uses {topology_edges_used} topology edges, {base_edges_used} base edges")

    def _update_cache(self, cache_type, key, value):
        """Helper method to update cache with size limits"""
        if cache_type not in self._cache:
            self._cache[cache_type] = {}
        
        self._cache[cache_type][key] = value
        
        # Limit cache size to prevent memory issues
        MAX_CACHE_SIZE = 1000
        if len(self._cache[cache_type]) > MAX_CACHE_SIZE:
            # Remove oldest entries (FIFO)
            keys_to_remove = list(self._cache[cache_type].keys())[:200]
            for k in keys_to_remove:
                del self._cache[cache_type][k]


    def build_detailed_itinerary(self, optimal_path, origin_point, destination_point):
        """Build detailed itinerary from network path - FIXED for topologies"""
        print(f"\n🔍 DEBUG build_detailed_itinerary:")

        if not optimal_path or len(optimal_path) < 2:
            print(f"   ❌ Path too short!")
            return []
        
        detailed_itinerary = []
        
        # Add walking to first station
        first_station = optimal_path[0]
        detailed_itinerary.append(('to station', [origin_point, first_station]))
        
        # Group consecutive segments by route_id for realistic journey segments
        current_route_segments = []
        current_route_id = None
        current_mode = None
        
        for i in range(len(optimal_path) - 1):
            current_node = optimal_path[i]
            next_node = optimal_path[i + 1]
            # Check if edge exists
            if current_node not in self.network_manager.active_network:
                print(f"   ❌ Node {current_node} not in network!")
                continue
                
            if next_node not in self.network_manager.active_network[current_node]:
                print(f"   ❌ No edge {current_node}->{next_node}!")
                continue
            # Get edge data from network
            if self.network_manager.active_network.has_edge(current_node, next_node):
                edge_data = self.network_manager.active_network[current_node][next_node]
                # Debug output
                
                if 'route_id' not in edge_data:
                    print(f"   ⚠️ WARNING: Missing route_id, skipping segment!")
                    continue
                # Extract route information
                route_id = edge_data.get('route_id', 'unknown')
                mode = edge_data.get('transport_mode', 'bus')
                
                
                # Convert mode to string if needed
                if hasattr(mode, 'value'):
                    mode = mode.value
                elif not isinstance(mode, str):
                    mode = str(mode)
                
                # Normalize mode string
                mode = mode.lower()
                if mode not in ['bus', 'train', 'express_bus', 'premium_train']:
                    mode = 'bus'  # Safe default
                
                # Check if this is same route or need to start new segment
                if route_id != current_route_id:
                    # Save previous segment if exists
                    if current_route_segments:
                        detailed_itinerary.append((current_mode, current_route_id, current_route_segments))
                    
                    # Start new segment
                    current_route_id = route_id
                    current_mode = mode
                    current_route_segments = [current_node, next_node]
                else:
                    # Continue current segment
                    current_route_segments.append(next_node)
            else:
                print(f"⚠️ Missing edge data for {current_node} -> {next_node}")
        
        # Add final segment
        if current_route_segments:
            detailed_itinerary.append((current_mode, current_route_id, current_route_segments))
        
        # Add walking from last station
        last_station = optimal_path[-1]
        detailed_itinerary.append(('to destination', [last_station, destination_point]))

 
        return detailed_itinerary

    def _create_fallback_route(self, origin, destination):
        """Create simple walking route when network routing fails"""
        return {
            'path': [origin, destination],
            'total_time': self._calculate_manhattan_distance(origin, destination) / 0.76,  # walking speed
            'mode': 'walk',
            'fallback': True
        }

    def _calculate_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def calculate_total_time_and_price_public(self, payment_scheme, detailed_itinerary, walking_speed, 
                                    bus_stop_speed, train_stop_speed, bus_stop_price, train_stop_price):
        """Calculate public transport time and price using ACTUAL NETWORK DATA"""
        itinerary_hash = hash(str(detailed_itinerary) + payment_scheme)
        cache_key = f"price_time_{itinerary_hash}_{walking_speed}_{bus_stop_speed}_{train_stop_speed}"
        if hasattr(self, '_price_time_cache') and cache_key in self._price_time_cache:
            return self._price_time_cache[cache_key]
        
        total_time = 0
        total_price = 0
        
       
        
        for segment_idx, segment in enumerate(detailed_itinerary):
            try:
                segment_type = segment[0] if len(segment) > 0 else 'unknown'
                
                
                if segment_type == 'to station':
                    # Walking to station
                    origin = segment[1][0] if len(segment) > 1 and len(segment[1]) > 0 else (0, 0)
                    station = segment[1][1] if len(segment) > 1 and len(segment[1]) > 1 else 'CENTRAL'
                    
                    station_coordinates = self.network_manager.spatial_mapper.node_to_grid.get(station, (50, 40))
                    if isinstance(origin, (tuple, list)) and len(origin) >= 2:
                        walk_distance = abs(origin[0] - station_coordinates[0]) + abs(origin[1] - station_coordinates[1])
                    else:
                        walk_distance = 5  # Default
                    
                    walk_time = walk_distance / walking_speed if walking_speed > 0 else walk_distance / 0.76
                    total_time += walk_time
                    print(f"[DEBUG] Walk to station: {walk_distance} units = {walk_time:.2f} time")
                    
                elif segment_type in ['bus', 'train']:
                    # 🎯 USE NETWORK DATA HERE!
                    mode = segment_type
                    route_id = segment[1] if len(segment) > 1 else 'unknown'
                    stations_list = segment[2] if len(segment) > 2 else []
                    
                    
                    if len(stations_list) >= 2:
                        segment_time = 0
                        segment_price = 0
                        
                        # ✅ PROCESS EACH STATION-TO-STATION HOP USING NETWORK EDGES
                        for i in range(len(stations_list) - 1):
                            from_station = stations_list[i]
                            to_station = stations_list[i + 1]
                            
                            # 🎯 GET ACTUAL NETWORK EDGE DATA
                            if self.network_manager.active_network.has_edge(from_station, to_station):
                                edge_data = self.network_manager.active_network[from_station][to_station]
                                
                                # ✅ USE YOUR CAREFULLY CALCULATED travel_time!
                                hop_time = edge_data.get('travel_time', 5.0)
                                hop_frequency = edge_data.get('frequency', 10.0)
                                hop_route_id = edge_data.get('route_id', route_id)
                                
                                # Add waiting time (half the frequency on average)
                                wait_time = hop_frequency / 2.0
                                
                                segment_time += hop_time + wait_time
                                print(f"[DEBUG]   {from_station}->{to_station}: travel={hop_time:.1f}, wait={wait_time:.1f}, route={hop_route_id}")
                                
                            else:
                                # Fallback only if edge doesn't exist
                                fallback_time = 5.0 if mode == 'train' else 3.0
                                segment_time += fallback_time
                                
                        
                        # Calculate price based on number of hops and mode
                        num_hops = len(stations_list) - 1
                        if mode == 'bus':
                            segment_price = min(num_hops * bus_stop_price * 0.5, bus_stop_price * 2.0)  # Cap at 2x base price
                        elif mode == 'train':
                            segment_price = min(num_hops * train_stop_price * 0.3, train_stop_price * 1.5)  # Cap at 1.5x base price
                        
                    else:
                        # Single hop fallback
                        segment_time = 5.0 if mode == 'train' else 3.0
                        segment_price = train_stop_price if mode == 'train' else bus_stop_price
                        print(f"[DEBUG] Single hop {mode}: time={segment_time}, price={segment_price}")
                    
                    total_time += segment_time
                    total_price += segment_price
                    print(f"[DEBUG] Segment total: time={segment_time:.2f}, price={segment_price:.2f}")
                    
                elif segment_type == 'to destination':
                    # Walking from station to destination
                    station = segment[1][0] if len(segment) > 1 and len(segment[1]) > 0 else 'CENTRAL'
                    destination = segment[1][1] if len(segment) > 1 and len(segment[1]) > 1 else (50, 50)
                    
                    station_coordinates = self.network_manager.spatial_mapper.node_to_grid.get(station, (50, 40))
                    if isinstance(destination, (tuple, list)) and len(destination) >= 2:
                        walk_distance = abs(destination[0] - station_coordinates[0]) + abs(destination[1] - station_coordinates[1])
                    else:
                        walk_distance = 5  # Default
                    
                    walk_time = walk_distance / walking_speed if walking_speed > 0 else walk_distance / 0.76
                    total_time += walk_time
                    print(f"[DEBUG] Walk from station: {walk_distance} units = {walk_time:.2f} time")
                    
            except Exception as e:
                print(f"[ERROR] Processing segment {segment_idx}: {e}")
                # Emergency fallback
                total_time += 5.0
                if segment_type in ['bus', 'train']:
                    total_price += 2.5
        
     
        
        # Cache and return
        result = (total_time, total_price, 0)  # No MaaS surcharge for direct public transport
        if not hasattr(self, '_price_time_cache'):
            self._price_time_cache = {}
        self._price_time_cache[cache_key] = result
        
        return result

    # def calculate_total_time_and_price_public(self, payment_scheme, detailed_itinerary, walking_speed, 
    #                                         bus_stop_speed, train_stop_speed, bus_stop_price, train_stop_price):
    #     """Calculate public transport time and price using network data"""
    #     itinerary_hash = hash(str(detailed_itinerary) + payment_scheme)
    #     cache_key = f"price_time_{itinerary_hash}_{walking_speed}_{bus_stop_speed}_{train_stop_speed}"
    #     if hasattr(self, '_price_time_cache') and cache_key in self._price_time_cache:
    #         return self._price_time_cache[cache_key]
        
    #     total_time = 0
    #     total_price = 0
        
    #     for segment in detailed_itinerary:
    #         segment_type = segment[0]
            
    #         if segment_type == 'to station':
    #             # Walking to station - use network routing
    #             origin = segment[1][0]
    #             station = segment[1][1]
                
    #             # Get station coordinates
    #             station_coordinates = self.network_manager.spatial_mapper.node_to_grid.get(station)
    #             if station_coordinates:
    #                 # Use network routing for walking
    #                 network_result = self.network_manager.find_network_route(origin, station_coordinates, mode='walk')
    #                 if network_result and 'spatial_route' in network_result:
    #                     walk_route = network_result['spatial_route']
    #                     walk_distance = len(walk_route) - 1 if len(walk_route) > 1 else 1
    #                 else:
    #                     # Fallback to Manhattan distance
    #                     walk_distance = abs(origin[0] - station_coordinates[0]) + abs(origin[1] - station_coordinates[1])
    #             else:
    #                 walk_distance = 5  # Default
                    
    #             walk_time = walk_distance / walking_speed
    #             total_time += walk_time
                
    #         elif segment_type in ['bus', 'train']:
    #             mode = segment_type
    #             route_id = segment[1] if len(segment) > 1 else 'unknown'
    #             stations_list = segment[2] if len(segment) > 2 else []
                
    #             # Calculate based on number of stations/hops
    #             if len(stations_list) >= 2:
    #                 # Use network edge data if available
    #                 from_station = stations_list[0]
    #                 to_station = stations_list[-1]
                    
    #                 if self.network_manager.active_network.has_edge(from_station, to_station):
    #                     edge_data = self.network_manager.active_network[from_station][to_station]
    #                     segment_time = edge_data.get('travel_time', 5.0)
    #                     # Calculate price based on distance or default
    #                     distance = edge_data.get('distance', 1.0)
    #                     if mode == 'bus':
    #                         segment_price = distance * bus_stop_price * 0.1
    #                     else:  # train
    #                         segment_price = distance * train_stop_price * 0.1
    #                 else:
    #                     # Fallback calculation
    #                     num_stops = 1
    #                     if mode == 'bus':
    #                         segment_time = num_stops * bus_stop_speed
    #                         segment_price = num_stops * bus_stop_price
    #                     else:  # train
    #                         segment_time = num_stops * train_stop_speed
    #                         segment_price = num_stops * train_stop_price
    #             else:
    #                 # Default for single hop
    #                 if mode == 'bus':
    #                     segment_time = bus_stop_speed
    #                     segment_price = bus_stop_price
    #                 else:  # train
    #                     segment_time = train_stop_speed
    #                     segment_price = train_stop_price
                
    #             total_time += segment_time
    #             total_price += segment_price
                
    #         elif segment_type == 'transfer':
    #             # Use transfer time from network or default
    #             transfer_stations = tuple(segment[1]) if len(segment) > 1 else None
    #             if transfer_stations and transfer_stations in self.transfers:
    #                 transfer_time = self.transfers[transfer_stations]
    #             else:
    #                 transfer_time = 1.0  # Default transfer time
    #             total_time += transfer_time
                
    #         elif segment_type == 'to destination':
    #             # Walking from station to destination
    #             station = segment[1][0]
    #             destination = segment[1][1]
                
    #             station_coordinates = self.network_manager.spatial_mapper.node_to_grid.get(station)
    #             if station_coordinates:
    #                 network_result = self.network_manager.find_network_route(station_coordinates, destination, mode='walk')
    #                 if network_result and 'spatial_route' in network_result:
    #                     walk_route = network_result['spatial_route']
    #                     walk_distance = len(walk_route) - 1 if len(walk_route) > 1 else 1
    #                 else:
    #                     walk_distance = abs(station_coordinates[0] - destination[0]) + abs(station_coordinates[1] - destination[1])
    #             else:
    #                 walk_distance = 5  # Default
                    
    #             walk_time = walk_distance / walking_speed
    #             total_time += walk_time
        
    #     # Apply MaaS surcharge
    #     if payment_scheme == 'PAYG':
    #         surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
    #         total_price_with_surcharge = total_price * (1 + surcharge_percentage)
    #         MaaS_surcharge = total_price_with_surcharge - total_price
    #     elif payment_scheme == 'subscription':
    #         surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
    #         total_price_with_surcharge = total_price * (1 + surcharge_percentage)
    #         MaaS_surcharge = total_price_with_surcharge - total_price
    #     else:
    #         total_price_with_surcharge = total_price
    #         MaaS_surcharge = 0
        
    #     result = (total_time, total_price_with_surcharge, MaaS_surcharge)
    #     if not hasattr(self, '_price_time_cache'):
    #         self._price_time_cache = {}
    #     self._price_time_cache[cache_key] = result
        
    #     return result

    def calculate_time_and_price_to_station_or_destination(self, segment, start_time):
        """Calculate segment time/price using network routing"""
        cache_key = f"segment_{start_time}_{segment[0]}_{segment[1][0]}_{segment[1][-1]}"
        if hasattr(self, '_segment_cache') and cache_key in self._segment_cache:
            return self._segment_cache[cache_key]
        
        options = []
        
        try:
            segment_type = segment[0]
            
            if segment_type == 'to station':
                origin = segment[1][0]
                destination_station = segment[1][1]
                
                # Get station coordinates
                destination_coords = self.network_manager.spatial_mapper.node_to_grid.get(destination_station)
                if not destination_coords:
                    return options
                
            elif segment_type == 'to destination':
                origin_station = segment[1][0]
                destination = segment[1][1]
                
                origin_coords = self.network_manager.spatial_mapper.node_to_grid.get(origin_station)
                if not origin_coords:
                    return options
                
                origin = origin_coords
                destination_coords = destination
            else:
                return options
            
            # Calculate Manhattan distance for quick assessment
            manhattan_distance = abs(origin[0] - destination_coords[0]) + abs(origin[1] - destination_coords[1])
            
            # Walking option (always available)
            walk_speed = self.service_provider_agent.get_travel_speed('walk', start_time)
            network_result = self.network_manager.find_network_route(origin, destination_coords, mode='walk')
            if network_result and 'spatial_route' in network_result:
                walk_route = network_result['spatial_route']
                walk_time = len(walk_route) / walk_speed if len(walk_route) > 1 else manhattan_distance / walk_speed
            else:
                walk_route = [origin, destination_coords]
                walk_time = manhattan_distance / walk_speed
            
            options.append((segment_type, segment[1], 'walk', walk_time, 0, walk_route))
            
            # Shared service options (bike and car)
            for mode in ['bike', 'car']:
                price_dict = self.service_provider_agent.get_shared_service_price(mode, start_time)
                if not price_dict:
                    continue
                
                unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                
                # Use network routing
                network_result = self.network_manager.find_network_route(
                    origin, destination_coords, mode=mode, current_time=start_time
                )
                
                if network_result and 'spatial_route' in network_result:
                    route = network_result['spatial_route']
                    time_base = len(route) / unit_speed if len(route) > 1 else manhattan_distance / unit_speed
                else:
                    route = [origin, destination_coords]
                    time_base = manhattan_distance / unit_speed
                
                # Process each company
                for company_name, unit_price in price_dict.items():
                    availability = self.service_provider_agent.check_shared_availability(company_name, start_time)
                    if availability > 0:
                        total_price = time_base * unit_price
                        options.append((segment_type, segment[1], company_name, time_base, total_price, route))
        
        except Exception as e:
            print(f"Error calculating segment options: {e}")
            # Fallback walk option
            if 'origin' in locals() and 'destination_coords' in locals():
                unit_speed = self.service_provider_agent.get_travel_speed('walk', start_time)
                walk_route = [origin, destination_coords]
                walk_time = manhattan_distance / unit_speed if 'manhattan_distance' in locals() else 10
                options.append((segment_type, segment[1], 'walk', walk_time, 0, walk_route))
        
        # Cache result
        if not hasattr(self, '_segment_cache'):
            self._segment_cache = {}
        self._segment_cache[cache_key] = options
        
        # Limit cache size
        if len(self._segment_cache) > 200:
            keys_to_remove = list(self._segment_cache.keys())[:20]
# Limit cache size
        if len(self._segment_cache) > 200:
            keys_to_remove = list(self._segment_cache.keys())[:20]
            for k in keys_to_remove:
                del self._segment_cache[k]
                
        return options

    def options_without_maas(self, request_id, start_time, origin, destination):
        """Generate non-MaaS options using network routing"""
        cache_key = f"options_{request_id}_{start_time}_{origin}_{destination}"
        if cache_key in self._cache['prices']:
            self.routing_performance['cache_hits'] += 1
            cached_options = self._cache['prices'][cache_key]
            self._update_commuter_travel_options(request_id, cached_options, 'without_maas')
            return cached_options
        
        travel_options = {'request_id': request_id}
        
        try:
            # Walking option (always available)
            unit_speed = self.service_provider_agent.get_travel_speed('walk', start_time)
            walk_result = self.calculate_single_mode_time_and_price(origin, destination, 0, unit_speed, 5)
            if walk_result:
                route, time, price = walk_result
                travel_options['walk_route'] = route
                travel_options['walk_time'] = time
                travel_options['walk_price'] = price
                travel_options['walk_availability'] = float('inf')
            
            # Shared service options (bike and car)
            for mode in ['bike', 'car']:
                try:
                    price_dict = self.service_provider_agent.get_shared_service_price(mode, start_time)
                    print(f"    - Price dict: {price_dict}")
                    if not price_dict:
                        continue
                    
                    unit_speed = self.service_provider_agent.get_travel_speed(mode, start_time)
                    mode_id = 4 if mode == 'bike' else 3
                    print(f"    - Unit speed: {unit_speed}, Mode ID: {mode_id}")
                    # Calculate route using network
                    base_result = self.calculate_single_mode_time_and_price(
                        origin, destination, 1.0, unit_speed, mode_id
                    )
                    print(f"    - Base result: {base_result}")
                    if base_result:
                        base_route, base_time, _ = base_result
                        
                        # Process each company
                        for company_name, unit_price in price_dict.items():
                            print(f"      🏢 Checking {company_name}...")
                            availability = self.service_provider_agent.check_shared_availability(company_name, start_time)
                            print(f"        - Availability: {availability}")
                            if availability > 0:
                                total_price = base_time * unit_price
                                option_key = f'{mode}_{company_name}'
                                travel_options[f'{mode}_{company_name}_route'] = base_route
                                travel_options[f'{mode}_{company_name}_time'] = base_time
                                travel_options[f'{mode}_{company_name}_price'] = total_price
                                travel_options[f'{mode}_{company_name}_availability'] = availability
                                print(f"        ✅ Added option: {option_key}")
                                print(f"           Time: {base_time}, Price: {total_price}")
                except Exception as e:
                    print(f"Error calculating {mode} options: {e}")
            
            # Public transport options
            try:
                optimal_path = self.find_optimal_route(origin, destination, 'public')
                print(f"Optimal path: {optimal_path}")
                if optimal_path:
                    detailed_itinerary = self.build_detailed_itinerary(optimal_path, origin, destination)
                    
                    if detailed_itinerary:
                        walking_speed = self.service_provider_agent.get_travel_speed('walk', start_time)
                        bus_speed = self.service_provider_agent.get_travel_speed('bus', start_time)
                        train_speed = self.service_provider_agent.get_travel_speed('train', start_time)
                        bus_price = self.service_provider_agent.get_public_service_price('bus', start_time)
                        train_price = self.service_provider_agent.get_public_service_price('train', start_time)
                        
                        public_time, public_price, _ = self.calculate_total_time_and_price_public(
                            'none', detailed_itinerary, walking_speed, bus_speed, train_speed, bus_price, train_price
                        )
                        
                        travel_options['public_route'] = detailed_itinerary
                        travel_options['public_time'] = public_time
                        travel_options['public_price'] = public_price
                        travel_options['public_availability'] = float('inf')
                
            except Exception as e:
                print(f"Error calculating public transport options: {e}")
            
            # Update commuter's request with options
            self._update_commuter_travel_options(request_id, travel_options, 'without_maas')
            
            # Cache results
            self._update_cache('prices', cache_key, travel_options)
            
            return travel_options
            
        except Exception as e:
            print(f"Error generating travel options: {e}")
            return self._generate_fallback_options(request_id, start_time, origin, destination)

    def maas_options(self, payment_scheme, request_id, start_time, origin, destination):
        """Generate MaaS options using network routing"""
        cache_key = f"maas_{payment_scheme}_{request_id}_{start_time}_{origin}_{destination}"
        if cache_key in self._cache['prices']:
            self.routing_performance['cache_hits'] += 1
            cached_options = self._cache['prices'][cache_key]
            self._update_commuter_travel_options(request_id, cached_options, 'with_maas')
            return cached_options
        
        try:
            # Find optimal public transport route using network
            optimal_path = self.find_optimal_route(origin, destination, 'public')
            if not optimal_path:
                empty_options = []
                self._update_commuter_travel_options(request_id, empty_options, 'with_maas')
                return empty_options
            
            # Build detailed itinerary
            detailed_itinerary = self.build_detailed_itinerary(optimal_path, origin, destination)
            if not detailed_itinerary or len(detailed_itinerary) < 2:
                empty_options = []
                self._update_commuter_travel_options(request_id, empty_options, 'with_maas')
                return empty_options
            
            # Generate MaaS combinations
            maas_options = self._generate_maas_combinations(
                detailed_itinerary, payment_scheme, request_id, start_time
            )
            
            # Update commuter requests
            self._update_commuter_travel_options(request_id, maas_options, 'with_maas')
            
            # Cache results
            self._update_cache('prices', cache_key, maas_options)
            
            return maas_options
            
        except Exception as e:
            print(f"Error generating MaaS options: {e}")
            empty_options = []
            self._update_commuter_travel_options(request_id, empty_options, 'with_maas')
            return empty_options

    def _generate_maas_combinations(self, detailed_itinerary, payment_scheme, request_id, start_time):
        """Generate MaaS option combinations using network routing"""
        
        # Get segment options using network routing
        to_station_options = self.calculate_time_and_price_to_station_or_destination(
            detailed_itinerary[0], start_time
        )
        to_destination_options = self.calculate_time_and_price_to_station_or_destination(
            detailed_itinerary[-1], start_time
        )
        
        # Calculate public transport segment using network data
        bus_speed = self.service_provider_agent.get_travel_speed('bus', start_time)
        train_speed = self.service_provider_agent.get_travel_speed('train', start_time)
        bus_price = self.service_provider_agent.get_public_service_price('bus', start_time)
        train_price = self.service_provider_agent.get_public_service_price('train', start_time)
        
        public_time, public_price, _ = self.calculate_total_time_and_price_public(
            payment_scheme, detailed_itinerary, 0.76, bus_speed, train_speed, bus_price, train_price
        )
        
        # Limit combinations for performance
        max_combinations = 15
        if len(to_station_options) * len(to_destination_options) > max_combinations:
            to_station_options = sorted(to_station_options, key=lambda x: x[3])[:5]
            to_destination_options = sorted(to_destination_options, key=lambda x: x[3])[:5]
        
        maas_options = []
        
        for to_station_option in to_station_options:
            for to_destination_option in to_destination_options:
                # Calculate totals
                total_time = to_station_option[3] + public_time + to_destination_option[3]
                total_price = to_station_option[4] + public_price + to_destination_option[4]
                
                # Apply MaaS surcharge
                final_price, surcharge = self.apply_maas_surcharge(total_price, payment_scheme)
                
                # Create MaaS option
                maas_option = [
                    [to_station_option[2], to_station_option[3], to_station_option[4], to_station_option[5]],
                    [to_destination_option[2], to_destination_option[3], to_destination_option[4], to_destination_option[5]],
                    total_time,
                    final_price,
                    surcharge
                ]
                
                maas_options.append([request_id, detailed_itinerary] + [maas_option])
        
        return maas_options

    def apply_maas_surcharge(self, total_price, payment_scheme):
        """Apply MaaS surcharge with caching"""
        cache_key = f"surcharge_{payment_scheme}_{total_price}"
        if hasattr(self, '_surcharge_cache') and cache_key in self._surcharge_cache:
            return self._surcharge_cache[cache_key]
        
        if payment_scheme == 'PAYG':
            surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_with_surcharge = total_price * (1 + surcharge_percentage)
            maas_surcharge = total_with_surcharge - total_price
        elif payment_scheme == 'subscription':
            surcharge_percentage = self.calculate_dynamic_MaaS_surcharge(payment_scheme)
            total_with_surcharge = total_price * (1 + surcharge_percentage)
            maas_surcharge = total_with_surcharge - total_price
        else:
            total_with_surcharge = total_price
            maas_surcharge = 0
        
        result = (total_with_surcharge, maas_surcharge)
        if not hasattr(self, '_surcharge_cache'):
            self._surcharge_cache = {}
        self._surcharge_cache[cache_key] = result
        
        if len(self._surcharge_cache) > 50:
            self._surcharge_cache.popitem()
        
        return result

    def _update_commuter_travel_options(self, request_id, travel_options, option_type):
        """Update commuter's request with travel options"""
        for commuter in self.commuter_agents:
            
            if request_id in commuter.requests:
                key = f'travel_options_{option_type}'
                commuter.requests[request_id][key] = travel_options
                break

    def _generate_fallback_options(self, request_id, start_time, origin, destination):
        """Generate minimal fallback options when main calculation fails"""
        print(f"Generating fallback options for request {request_id}")
        
        distance = ((destination[0] - origin[0])**2 + (destination[1] - origin[1])**2)**0.5
        walk_time = distance / 0.76
        
        fallback_options = {
            'request_id': request_id,
            'walk_route': [origin, destination],
            'walk_time': walk_time,
            'walk_price': 0,
            'walk_availability': float('inf')
        }
        
        self._update_commuter_travel_options(request_id, fallback_options, 'without_maas')
        return fallback_options

    def _update_cache(self, cache_type: str, key: str, value):
        """Update cache with size limit"""
        if cache_type not in self._cache:
            print(f"Cache type not found: {cache_type}")
            return
        
        self._cache[cache_type][key] = value
        # Check cache size limit
        limit = self._cache_limits.get(cache_type, 100)
        if len(self._cache[cache_type]) > limit:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._cache[cache_type].keys())[:50]
            for old_key in keys_to_remove:
                del self._cache[cache_type][old_key]

    # ===== BOOKING AND SERVICE MANAGEMENT (UNCHANGED) =====
    def book_service(self, request_id, ranked_options, current_step, availability_dict):
        """Book service with optimized database operations (UNCHANGED)"""
        print(f"\n🔍 [DEBUG] book_service called:")
        print(f"  - request_id: {request_id}")
        print(f"  - ranked_options: {ranked_options}")
        print(f"  - current_step: {current_step}")
        # Ensure subsidy pool is managed
        self.manage_subsidy_pool()
        
        # Check for existing booking to avoid duplication
        try:
            with self.Session() as session:
                existing_booking = session.query(ServiceBookingLog).filter_by(
                    request_id=str(request_id)
                ).first()
                
                if existing_booking:
                    print(f"Request {request_id} already has a booking. Skipping.")
                    return True, availability_dict
        except Exception as e:
            print(f"Error checking existing booking: {e}")
        
        booking_made = False
        
        # Find the appropriate commuter
        target_commuter = None
        for commuter in self.commuter_agents:
            if request_id in commuter.requests:
                target_commuter = commuter
                break
        
        if not target_commuter:
            print(f"No commuter found for request {request_id}")
            return False, availability_dict
        
        if target_commuter.requests[request_id]['status'] != 'active':
            return True, availability_dict
        
        # Process ranked options
        for option in ranked_options:
            subsidy = option[-1]
            
            try:
                if "maas" in option[1]:  # MaaS option
                    maas_key = option[1]
                    
                    maas_option = None
                    travel_options = target_commuter.requests[request_id]['travel_options_with_maas']
                    for idx, mo in enumerate(travel_options):
                        if f"maas_{idx}" == maas_key:
                            maas_option = mo
                            break
                    
                    if maas_option:
                        detailed_itinerary = maas_option[1]
                        maas_option_details = maas_option[2]
                        
                        total_time = maas_option_details[2]
                        final_price = maas_option_details[3]
                        maas_surcharge = maas_option_details[4]
                        to_station_info = maas_option_details[0]
                        to_destination_info = maas_option_details[1]
                        
                        # Record booking
                        self.record_maas_or_non_maas_booking(
                            target_commuter,
                            request_id,
                            "MaaS_Bundle",
                            final_price,
                            start_time=target_commuter.requests[request_id]['start_time'],
                            duration=max(1, round(total_time)),
                            route=detailed_itinerary,
                            to_station_info=to_station_info,
                            to_destination_info=to_destination_info,
                            maas_surcharge=maas_surcharge,
                            availability_dict=availability_dict,
                            current_step=current_step,
                            subsidy=subsidy
                        )
                        
                        if self.confirm_booking_maas(target_commuter, request_id):
                            self.record_subsidy_usage(
                                commuter_id=target_commuter.unique_id,
                                request_id=request_id,
                                subsidy_amount=subsidy,
                                mode='maas',
                                timestamp=current_step
                            )
                            booking_made = True
                            break
                else:  # Non-MaaS option
                    probability, mode_company, route, time = option[:4]
                    
                    travel_options = target_commuter.requests[request_id]['travel_options_without_maas']
                    availability_key = mode_company.replace('route', 'availability')
                    price_key = mode_company.replace('route', 'price')
                    
                    if availability_key not in travel_options or price_key not in travel_options:
                        continue
                    
                    availability = travel_options[availability_key]
                    price = travel_options[price_key]
                    
                    if mode_company in ['public_route', 'walk_route']:
                        if self.confirm_booking_non_maas(target_commuter, request_id, mode_company, route, price):
                            self.record_maas_or_non_maas_booking(
                                target_commuter,
                                request_id,
                                mode_company,
                                price,
                                start_time=target_commuter.requests[request_id]['start_time'],
                                duration=max(1, round(time)),
                                route=route,
                                to_station_info=None,
                                to_destination_info=None,
                                maas_surcharge=0,
                                availability_dict=availability_dict,
                                current_step=current_step,
                                subsidy=subsidy
                            )
                            self.record_subsidy_usage(
                                commuter_id=target_commuter.unique_id,
                                request_id=request_id,
                                subsidy_amount=subsidy,
                                mode='public_subsidy',
                                timestamp=current_step
                            )
                            booking_made = True
                            break
                    else:  # Shared service option
                        print(f"\n🚗 [DEBUG] Shared Service Booking Attempt:")
                        print(f"  - mode_company: {mode_company}")
                        print(f"  - request_id: {request_id}")
                        print(f"  - commuter_id: {target_commuter.unique_id}")
                        print(f"  - current_step: {current_step}")
                        
                        # Extract company name
                        try:
                            chosen_company = mode_company.split('_')[1]
                            print(f"  - chosen_company: {chosen_company}")
                        except IndexError as e:
                            print(f"  ❌ ERROR extracting company from mode_company: {e}")
                            continue
                        
                        # Get request details
                        try:
                            start_time = target_commuter.requests[request_id]['start_time']
                            print(f"  - start_time: {start_time}")
                            print(f"  - route: {route}")
                            print(f"  - time: {time}")
                            print(f"  - price: {price}")
                        except KeyError as e:
                            print(f"  ❌ ERROR accessing request details: {e}")
                            continue
                        
                        duration = max(1, round(time))
                        end_check_step = min(current_step + 5, start_time + duration)
                        print(f"  - duration: {duration}")
                        print(f"  - end_check_step: {end_check_step}")
                        
                        # Availability checking with detailed logging
                        print(f"  🔍 Checking availability from step {start_time} to {end_check_step}:")
                        all_steps_available = True
                        steps_to_check = {}
                        failed_step = None
                        failed_reason = None
                        
                        for step in range(start_time, end_check_step + 1):
                            step_key = step - current_step
                            print(f"    - Checking step {step} (step_key: {step_key})")
                            
                            if step_key < 0:
                                all_steps_available = False
                                failed_step = step
                                failed_reason = f"step_key {step_key} < 0"
                                print(f"    ❌ FAIL: step_key {step_key} < 0")
                                break
                                
                            if step_key > 5:
                                all_steps_available = False
                                failed_step = step
                                failed_reason = f"step_key {step_key} > 5"
                                print(f"    ❌ FAIL: step_key {step_key} > 5")
                                break
                            
                            availability_check_key = f'{chosen_company}_{step_key}'
                            current_avail = availability_dict.get(availability_check_key, 0)
                            print(f"    - availability_check_key: {availability_check_key}")
                            print(f"    - current_avail: {current_avail}")
                            
                            if current_avail < 1:
                                all_steps_available = False
                                failed_step = step
                                failed_reason = f"availability {current_avail} < 1 for key {availability_check_key}"
                                print(f"    ❌ FAIL: {failed_reason}")
                                break
                            else:
                                print(f"    ✅ OK: {current_avail} units available")
                                
                            steps_to_check[step_key] = availability_check_key
                        
                        print(f"  📊 Availability Check Result:")
                        print(f"    - all_steps_available: {all_steps_available}")
                        if not all_steps_available:
                            print(f"    - failed_step: {failed_step}")
                            print(f"    - failed_reason: {failed_reason}")
                        else:
                            print(f"    - steps_to_check: {steps_to_check}")
                        
                        # Print current availability_dict for this company
                        company_availability = {k: v for k, v in availability_dict.items() if chosen_company in k}
                        print(f"  📋 Current {chosen_company} availability_dict: {company_availability}")
                        
                        if all_steps_available:
                            print(f"  ✅ All steps available, attempting to confirm booking...")
                            
                            # Try to confirm booking
                            try:
                                booking_confirmed = self.confirm_booking_non_maas(target_commuter, request_id, mode_company, route, price)
                                print(f"  📞 confirm_booking_non_maas result: {booking_confirmed}")
                                
                                if booking_confirmed:
                                    print(f"  🎉 Booking confirmed! Recording booking...")
                                    
                                    # Record the booking
                                    try:
                                        self.record_maas_or_non_maas_booking(
                                            target_commuter,
                                            request_id,
                                            mode_company,
                                            price,
                                            start_time=start_time,
                                            duration=duration,
                                            route=route,
                                            to_station_info=None,
                                            to_destination_info=None,
                                            maas_surcharge=0,
                                            availability_dict=availability_dict,
                                            current_step=current_step,
                                            subsidy=subsidy
                                        )
                                        print(f"  ✅ record_maas_or_non_maas_booking successful")
                                    except Exception as e:
                                        print(f"  ❌ ERROR in record_maas_or_non_maas_booking: {e}")
                                    
                                    # Record subsidy usage
                                    try:
                                        self.record_subsidy_usage(
                                            commuter_id=target_commuter.unique_id,
                                            request_id=request_id,
                                            subsidy_amount=subsidy,
                                            mode=mode_company,
                                            timestamp=current_step
                                        )
                                        print(f"  ✅ record_subsidy_usage successful (subsidy: {subsidy})")
                                    except Exception as e:
                                        print(f"  ❌ ERROR in record_subsidy_usage: {e}")
                                    
                                    # Update availability
                                    if 'Uber' in mode_company or 'Bike' in mode_company:
                                        print(f"  🔄 Updating availability for {len(steps_to_check)} steps...")
                                        for step_key, avail_key in steps_to_check.items():
                                            old_value = availability_dict[avail_key]
                                            availability_dict[avail_key] -= 1
                                            new_value = availability_dict[avail_key]
                                            print(f"    - {avail_key}: {old_value} → {new_value}")
                                    
                                    # Record booking log for shared services
                                    if mode_company not in ['public_route', 'walk_route']:
                                        try:
                                            affected_steps = list(range(start_time, start_time + duration))
                                            print(f"  📝 Recording booking log for {chosen_company}")
                                            print(f"    - affected_steps: {affected_steps}")
                                            
                                            self.service_provider_agent.record_booking_log(
                                                target_commuter.unique_id,
                                                request_id,
                                                chosen_company,
                                                start_time,
                                                duration,
                                                affected_steps,
                                                route
                                            )
                                            print(f"  ✅ record_booking_log successful")
                                        except Exception as e:
                                            print(f"  ❌ ERROR in record_booking_log: {e}")
                                    
                                    booking_made = True
                                    print(f"  🎯 BOOKING SUCCESSFUL! Setting booking_made = True")
                                    break
                                else:
                                    print(f"  ❌ BOOKING FAILED: confirm_booking_non_maas returned False")
                                    
                            except Exception as e:
                                print(f"  ❌ ERROR during booking confirmation: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"  ❌ BOOKING SKIPPED: Not all steps available")
                        
                        print(f"  📋 End of shared service booking attempt\n")
            except Exception as e:
                print(f"Error processing option: {e}")
                continue
        
        if not booking_made:
            print(f"Failed to book any service for request {request_id}")
        
        return booking_made, availability_dict

    # In agent_MaaS.py, modify the record_maas_or_non_maas_booking method:

    def record_maas_or_non_maas_booking(self, commuter, request_id, mode_company, final_total_price, start_time, duration, route, to_station_info, to_destination_info, maas_surcharge, availability_dict, current_step, subsidy):
        """Record booking - FIXED to handle public routes properly"""
        if mode_company == "MaaS_Bundle":
            selected_route = {
                'route': route,
                'price': final_total_price,
                'MaaS_surcharge': maas_surcharge,
                'time': duration,
                'to_station_info': to_station_info,
                'to_destination_info': to_destination_info,
                'route': route,
                'mode': 'MaaS_Bundle',
                'subsidy': subsidy
            }

            commuter.requests[request_id]['selected_route'] = selected_route
            commuter.requests[request_id]['status'] = 'Service Selected'
            
            self.record_service_booking(commuter, request_id, selected_route, subsidy)
            
            to_station_route = next((seg for seg in route if seg[0] == 'to station'), None)
            to_destination_route = next((seg for seg in route if seg[0] == 'to destination'), None)

            if to_station_info and ('Uber' in to_station_info[0] or 'Bike' in to_station_info[0]):
                self.record_share_service_booking(
                    commuter,
                    request_id,
                    to_station_info,
                    start_time,
                    availability_dict,
                    current_step,
                    route_details=to_station_route
                )
            if to_destination_info and ('Uber' in to_destination_info[0] or 'Bike' in to_destination_info[0]):
                self.record_share_service_booking(
                    commuter,
                    request_id,
                    to_destination_info,
                    start_time + duration,
                    availability_dict,
                    current_step,
                    route_details=to_destination_route
                )

        else:
            selected_route = {
                'route': route,
                'price': final_total_price,
                'time': duration,
                'mode': mode_company,
                'subsidy': subsidy
            }

            self.record_service_booking(commuter, request_id, selected_route, subsidy)
            
            # FIXED: Only call record_booking_log for shared services, not public routes
            if mode_company not in ['public_route', 'walk_route'] and ('Uber' in mode_company or 'Bike' in mode_company):
                self.service_provider_agent.record_booking_log(
                    commuter.unique_id,
                    request_id,
                    mode_company,
                    start_time,
                    duration,
                    list(range(start_time, start_time + duration)),
                    route
                )



    def record_share_service_booking(self, commuter, request_id, service_info, start_time, availability_dict, current_step, route_details):
        """Record share service booking (UNCHANGED)"""
        company_name, time, price, detailed_route = service_info
        duration = max(1, round(time))

        affected_steps = list(range(start_time, start_time + duration))

        provider_mapping = {
            'BikeShare1': 3,
            'BikeShare2': 4,
            'UberLike1': 1,
            'UberLike2': 2
        }

        mode_mapping = {
            'BikeShare1': 4,
            'BikeShare2': 4,
            'UberLike1': 3,
            'UberLike2': 3,
            'walk': 5
        }

        provider_id = provider_mapping.get(company_name)
        mode_id = mode_mapping.get(company_name if 'walk' not in company_name else 'walk')

        request_id_str = str(request_id)

        try:
            with self.Session() as session:
                existing_booking = session.query(ShareServiceBookingLog).filter_by(
                    commuter_id=commuter.unique_id,
                    request_id=request_id_str
                ).first()

                if existing_booking:
                    print(f"[INFO] Share service booking already exists for commuter_id={commuter.unique_id}, request_id={request_id_str}. Skipping insert.")
                    return

                new_booking = ShareServiceBookingLog(
                    commuter_id=commuter.unique_id,
                    request_id=request_id_str,
                    mode_id=mode_id,
                    provider_id=provider_id,
                    company_name=company_name,
                    start_time=start_time,
                    duration=duration,
                    affected_steps=affected_steps,
                    route_details=(route_details, detailed_route)
                )
                session.add(new_booking)
                session.commit()

        except Exception as e:
            print(f"[ERROR] Failed to record share service booking: {e}")

        if 'Uber' in company_name or 'Bike' in company_name:
            end_check_step = min(current_step + 5, start_time + duration)
            for step in range(start_time, end_check_step + 1):
                step_key = step - current_step
                availability_deduct_key = f'{company_name}_{step_key}'
                availability_dict[availability_deduct_key] -= 1

    def record_service_booking(self, commuter, request_id, selected_route, subsidy):
        """Record service booking (UNCHANGED)"""
        payment_scheme = commuter.payment_scheme
        start_time = commuter.requests[request_id]['start_time']
        origin_coordinates = commuter.requests[request_id]['origin']
        destination_coordinates = commuter.requests[request_id]['destination']
        
        to_station_info = None
        to_destination_info = None
        
        is_maas_option = isinstance(selected_route, dict) and 'MaaS_surcharge' in selected_route

        if is_maas_option:
            record_company_name = 'MaaS_Bundle'
            route_details = selected_route['route']
            total_price = selected_route['price']
            maas_surcharge = selected_route['MaaS_surcharge']
            total_time = selected_route['time']

            to_station_info = selected_route.get('to_station_info')
            to_destination_info = selected_route.get('to_destination_info')

        else:
            if 'public' in selected_route['mode']:
                record_company_name = 'public'
            elif 'walk' in selected_route['mode']:
                record_company_name = 'walk'
            else:
                record_company_name = selected_route['mode'].split('_')[1]
                    
            route_details = selected_route['route']
            total_price = selected_route['price']
            maas_surcharge = 0
            total_time = selected_route['time']

        new_booking = ServiceBookingLog(
            commuter_id=commuter.unique_id,
            payment_scheme=payment_scheme,
            request_id=str(request_id),
            start_time=start_time,
            record_company_name=record_company_name,
            route_details=route_details,
            total_price=total_price,
            maas_surcharge=maas_surcharge,
            total_time=total_time,
            origin_x=origin_coordinates[0],  # NEW
            origin_y=origin_coordinates[1],  # NEW
            destination_x=destination_coordinates[0],  # NEW
            destination_y=destination_coordinates[1],  # NEW
            to_station=to_station_info,
            to_destination=to_destination_info,
            government_subsidy=subsidy,  # NEW: now simple float
            status='Service Selected'
        )

        try:
            Session = sessionmaker(bind=self.db_engine)
            with Session() as session:
                existing_booking = session.query(ServiceBookingLog).filter_by(
                    commuter_id=commuter.unique_id,
                    payment_scheme=payment_scheme,
                    request_id=str(request_id)
                ).first()

                if existing_booking:
                    print(f"Booking already exists for commuter_id {commuter.unique_id}, request_id {request_id}. Updating status.")
                    existing_booking.status = 'Service Selected'
                    session.commit()
                else:
                    session.add(new_booking)
                    session.commit()

        except SQLAlchemyError as e:
            print(f"[ERROR] Error recording service booking: {e}")
    
    def confirm_booking_maas(self, commuter, request_id):
        """Confirm MaaS booking (UNCHANGED)"""
        result = commuter.accept_service(request_id)
        return result

    def confirm_booking_non_maas(self, commuter, request_id, mode, route, price):
        """Confirm non-MaaS booking (UNCHANGED)"""
        selected_route = {
            'mode': mode,
            'route': route,
            'price': price
        }
        return commuter.accept_service_non_maas(request_id, selected_route)

    # ===== SUBSIDY MANAGEMENT (UNCHANGED) =====
    def manage_subsidy_pool(self):
        """Manage subsidy pool (UNCHANGED)"""
        current_step = self.model.get_current_step()
        current_day = current_step // 144
        current_week = current_day // 7
        day_of_week = current_day % 7

        with self.Session() as session:
            total_used = session.query(func.sum(SubsidyUsageLog.subsidy_amount))\
                .filter(SubsidyUsageLog.week == current_week)\
                .scalar() or 0

        if self.subsidy_config.is_reset_time(current_step, self.last_reset_step):
            self.current_subsidy_pool = self.subsidy_config.total_amount
            self.last_reset_step = current_step
        else:
            self.current_subsidy_pool = self.subsidy_config.total_amount - total_used

        if day_of_week >= 5:
            self.current_subsidy_pool = 0

    def check_subsidy_availability(self, requested_amount):
        """Check subsidy availability (UNCHANGED)"""
        current_step = self.model.get_current_step()
        current_day = current_step // 144
        current_week = current_day // 7
        day_of_week = current_day % 7

        if not self.subsidy_config.is_subsidy_available(day_of_week):
            print("Weekend - No subsidies available")
            return 0

        with self.Session() as session:
            session.expire_all()
            if self.subsidy_config.pool_type == 'weekly':
                total_used = session.query(func.sum(SubsidyUsageLog.subsidy_amount))\
                    .filter(SubsidyUsageLog.week == current_week)\
                    .scalar() or 0
            elif self.subsidy_config.pool_type == 'daily':
                total_used = session.query(func.sum(SubsidyUsageLog.subsidy_amount))\
                    .filter(SubsidyUsageLog.day == current_day)\
                    .scalar() or 0
            elif self.subsidy_config.pool_type == 'monthly':
                total_used = session.query(func.sum(SubsidyUsageLog.subsidy_amount))\
                    .filter(SubsidyUsageLog.month == current_day // 30)\
                    .scalar() or 0

        remaining_pool = self.subsidy_config.total_amount - total_used

        if remaining_pool >= requested_amount:
            return requested_amount
        elif remaining_pool > 0:
            return remaining_pool
        else:
            return 0

    def get_subsidy_statistics(self):
        """Get subsidy statistics (UNCHANGED)"""
        current_step = self.model.get_current_step()
        steps_per_day = 144
        current_day = current_step // steps_per_day
        day_of_week = current_day % 7
        
        stats = {
            'pool_type': self.subsidy_config.pool_type,
            'total_pool': self.subsidy_config.total_amount,
            'remaining_pool': self.current_subsidy_pool,
            'total_given': self.total_subsidies_given,
            'current_day': current_day,
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week],
            'subsidies_available_today': self.subsidy_config.is_subsidy_available(day_of_week)
        }
        
        return stats
    
    def record_subsidy_usage(self, commuter_id, request_id, subsidy_amount, mode, timestamp):
        """Record subsidy usage (UNCHANGED)"""
        day = timestamp // 144
        week = day // 7
        month = day // 30
        
        if hasattr(subsidy_amount, 'item'):
            subsidy_amount = float(subsidy_amount)
        
        with self.Session() as session:
            usage_log = SubsidyUsageLog(
                commuter_id=commuter_id,
                request_id=str(request_id),
                subsidy_amount=subsidy_amount,
                mode=mode,
                timestamp=timestamp,
                day=day,
                week=week,
                month=month,
                period_type=self.subsidy_config.pool_type
            )
            session.add(usage_log)
            session.flush()
            session.commit()


    def log_route_analysis(self, route_result, request_id):
        """Debug function to track shortcut usage - add this method"""
        if not hasattr(self, '_shortcut_stats'):
            self._shortcut_stats = {'total_routes': 0, 'shortcut_routes': 0}
        
        self._shortcut_stats['total_routes'] += 1
        
        # Check if route uses shortcuts
        if 'network_route' in route_result:
            network_path = route_result['network_route']
            shortcuts_used = 0
            
            for i in range(len(network_path) - 1):
                try:
                    edge_data = self.network_manager.active_network.get_edge_data(
                        network_path[i], network_path[i+1]
                    )
                    if edge_data and edge_data.get('edge_type') == 'shortcut':
                        shortcuts_used += 1
                except:
                    continue
            
            if shortcuts_used > 0:
                self._shortcut_stats['shortcut_routes'] += 1
                print(f"🚀 Route {request_id}: Used {shortcuts_used} shortcuts")
        
        # Print stats every 10 routes
        if self._shortcut_stats['total_routes'] % 10 == 0:
            total = self._shortcut_stats['total_routes']
            shortcut = self._shortcut_stats['shortcut_routes']
            percentage = (shortcut / total * 100) if total > 0 else 0
            print(f"📊 Shortcut Usage: {shortcut}/{total} routes ({percentage:.1f}%)")