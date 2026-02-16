import networkx as nx
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from Complete_NTEO.topology.network_topology import SydneyNetworkTopology, TransportMode, NetworkNode
from collections import defaultdict
import math

class SmallWorldTopologyGenerator:
    """
    Implements Watts-Strogatz small-world model with geographic constraints
    for realistic Sydney transport network generation
    """
    
    def __init__(self, base_network: SydneyNetworkTopology):
        self.base_network = base_network
        self.base_nodes = list(base_network.nodes.keys())
        
        # Geographic constraints for realistic rewiring
        self.max_rewire_distance = 40  # Maximum geographic distance for shortcuts
        self.preserve_hierarchy = True  # Maintain transport mode hierarchy
        self.preserve_major_hubs = True  # Don't rewire major hub connections
        
    def generate_small_world_network(self, rewiring_probability: float = 0.1,
                                   initial_neighbors: int = 4,
                                   preserve_geography: bool = True) -> nx.Graph:
        """
        Generate small-world network using Watts-Strogatz model with geographic constraints
        
        Args:
            rewiring_probability: Probability p for rewiring each edge (0.01-0.5)
            initial_neighbors: Number of nearest neighbors in initial regular network
            preserve_geography: Whether to enforce geographic realism in rewiring
        """
        
        print(f"Generating small-world network (p={rewiring_probability:.3f})...")
        
        # Step 1: Create base regular network (ring lattice equivalent)
        regular_graph = self._create_geographic_regular_network(initial_neighbors)
        
        # Step 2: Apply Watts-Strogatz rewiring with geographic constraints
        small_world_graph = self._apply_geographic_rewiring(
            regular_graph, rewiring_probability, preserve_geography
        )
        
        # Step 3: Validate and enhance network
        small_world_graph = self._ensure_network_connectivity(small_world_graph)
        small_world_graph = self._add_transport_hierarchy(small_world_graph)
        
        print(f"Small-world network created: {small_world_graph.number_of_nodes()} nodes, "
              f"{small_world_graph.number_of_edges()} edges")
        
        return small_world_graph
    
    def _determine_transport_modes(self, node1_data, node2_data, distance):
        """Determine appropriate transport modes for connection"""
        # Get available modes for both nodes
        modes1 = set(mode.value for mode in node1_data.transport_modes)
        modes2 = set(mode.value for mode in node2_data.transport_modes)
        
        # Find common modes
        common_modes = modes1.intersection(modes2)
        
        if not common_modes:
            # Default to bus if no common modes
            return [TransportMode.BUS]
        
        # Prefer train for longer distances, bus for shorter
        if distance > 20 and 'train' in common_modes:
            return [TransportMode.TRAIN]
        elif 'bus' in common_modes:
            return [TransportMode.BUS]
        else:
            # Return first available common mode
            mode_priority = ['train', 'bus', 'metro', 'light_rail']
            for mode in mode_priority:
                if mode in common_modes:
                    try:
                        return [getattr(TransportMode, mode.upper())]
                    except AttributeError:
                        continue
            
            return [TransportMode.BUS]  # Final fallback
    
    def _estimate_capacity(self, transport_modes):
        """Estimate capacity based on transport modes"""
        if not transport_modes:
            return 500
        
        mode = transport_modes[0]  # Use primary mode
        
        capacity_map = {
            TransportMode.TRAIN: 2000,
            TransportMode.BUS: 600,
            TransportMode.METRO: 1500,
            TransportMode.FERRY: 300,
            TransportMode.BIKE: 50,
            TransportMode.LIGHT_RAIL: 400,
            TransportMode.WALKING: 100
        }
        
        return capacity_map.get(mode, 500)

    def _can_rewire_edge(self, u: str, v: str) -> bool:
        """Determine if an edge can be rewired based on node importance"""
        
        if not self.preserve_major_hubs:
            return True
        
        # Don't rewire connections involving major hubs
        u_type = self.base_network.nodes[u].node_type
        v_type = self.base_network.nodes[v].node_type
        
        if (u_type.value == 'major_hub' or v_type.value == 'major_hub'):
            return False
        
        return True

    def _find_rewiring_target(self, graph: nx.Graph, u: str, v: str, 
                            preserve_geography: bool) -> Optional[str]:
        """Find suitable target for rewiring with geographic constraints"""
        
        u_coord = self.base_network.nodes[u].coordinates
        candidates = []
        
        for candidate in self.base_nodes:
            if candidate == u or candidate == v:
                continue
            
            if graph.has_edge(u, candidate):
                continue  # Already connected
            
            candidate_coord = self.base_network.nodes[candidate].coordinates
            distance = self._calculate_distance(u_coord, candidate_coord)
            
            # Geographic constraint: limit shortcut distance
            if preserve_geography and distance > self.max_rewire_distance:
                continue
            
            # Transport hierarchy constraint
            if not self._is_valid_transport_connection(u, candidate):
                continue
            
            candidates.append((distance, candidate))
        
        if not candidates:
            return None
        
        # Prefer medium-distance connections (true shortcuts)
        # Not too close (already likely connected) or too far (unrealistic)
        candidates.sort(key=lambda x: abs(x[0] - self.max_rewire_distance * 0.6))
        
        return candidates[0][1]

    def _is_valid_transport_connection(self, node1: str, node2: str) -> bool:
        """Check if transport connection between nodes makes sense"""
        
        if not self.preserve_hierarchy:
            return True
        
        type1 = self.base_network.nodes[node1].node_type
        type2 = self.base_network.nodes[node2].node_type
        
        # Allow connections between:
        # - Any two transport hubs (creates useful shortcuts)
        # - Transport hub to local station
        # - Local stations in same general area
        
        if (type1.value in ['major_hub', 'transport_hub'] or 
            type2.value in ['major_hub', 'transport_hub']):
            return True
        
        # For local connections, check distance
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        distance = self._calculate_distance(coord1, coord2)
        
        return distance <= 25  # Local connections only

    def _add_realistic_edge(self, graph: nx.Graph, node1: str, node2: str, 
                      edge_type: str = "regular", base_data: dict = None):
        """Add edge with realistic transport characteristics and ALL required attributes"""
        
        node1_data = self.base_network.nodes[node1]
        node2_data = self.base_network.nodes[node2]
        
        # Calculate basic edge properties
        distance = self._calculate_distance(node1_data.coordinates, node2_data.coordinates)
        
        # Determine transport modes based on node types and distance
        transport_modes = self._determine_transport_modes(node1_data, node2_data, distance)
        primary_mode = transport_modes[0]  # Get the primary mode
        
        # Calculate travel time (simplified)
        base_time = distance * 2  # Rough estimate: 2 minutes per grid unit
        
        # Adjust time based on edge type
        if edge_type == "shortcut":
            base_time *= 0.7  # Shortcuts are faster
            frequency = 12  # More frequent service for shortcuts
        else:
            frequency = 8   # Regular frequency
        
        # Generate proper route_id (avoid "unknown")
        if edge_type == "shortcut":
            route_id = f"SW_SHORTCUT_{primary_mode.value}_{node1}_{node2}"
        elif edge_type == "hierarchy":
            route_id = f"SW_HIER_{primary_mode.value}_{node1}_{node2}"
        elif edge_type == "connectivity":
            route_id = f"SW_CONN_{primary_mode.value}_{node1}_{node2}"
        else:  # regular
            route_id = f"SW_REG_{primary_mode.value}_{node1}_{node2}"
        
        # Add edge with ALL required attributes for routing system
        graph.add_edge(node1, node2, 
                    # Core routing attributes
                    transport_mode=primary_mode,      # CRITICAL: Single primary mode
                    transport_modes=transport_modes,   # List of all available modes
                    
                    # Performance attributes  
                    travel_time=base_time,
                    distance=distance,
                    capacity=self._estimate_capacity(transport_modes),
                    frequency=frequency,
                    
                    # RAPTOR compatibility
                    route_id=route_id,                # CRITICAL: Proper route ID
                    segment_order=1,                  # CRITICAL: For route sequencing
                    
                    # Small-world specific
                    edge_type=edge_type,              # Track edge origin
                    geographic_distance=distance)

    def _ensure_network_connectivity(self, graph: nx.Graph) -> nx.Graph:
        """Ensure network remains connected after rewiring"""
        
        if nx.is_connected(graph):
            return graph
        
        print("Network disconnected after rewiring - adding connectivity edges...")
        
        # Find disconnected components
        components = list(nx.connected_components(graph))
        main_component = max(components, key=len)
        
        for component in components:
            if component == main_component:
                continue
            
            # Connect each component to main component
            min_distance = float('inf')
            best_connection = None
            
            for node1 in main_component:
                coord1 = self.base_network.nodes[node1].coordinates
                for node2 in component:
                    coord2 = self.base_network.nodes[node2].coordinates
                    distance = self._calculate_distance(coord1, coord2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_connection = (node1, node2)
            
            if best_connection:
                self._add_realistic_edge(graph, best_connection[0], best_connection[1], "connectivity")
        
        return graph

    def _add_transport_hierarchy(self, graph: nx.Graph) -> nx.Graph:
        """Add missing connections to maintain transport hierarchy"""
        
        # Ensure major hubs are well connected
        major_hubs = [n for n in graph.nodes() 
                    if self.base_network.nodes[n].node_type.value == 'major_hub']
        
        for i, hub1 in enumerate(major_hubs):
            for hub2 in major_hubs[i+1:]:
                if not graph.has_edge(hub1, hub2):
                    coord1 = self.base_network.nodes[hub1].coordinates
                    coord2 = self.base_network.nodes[hub2].coordinates
                    distance = self._calculate_distance(coord1, coord2)
                    
                    if distance < 35:  # Connect nearby major hubs
                        self._add_realistic_edge(graph, hub1, hub2, "hierarchy")
        
        return graph
    def _create_geographic_regular_network(self, k: int) -> nx.Graph:
        """Create regular network based on geographic proximity (not ring topology)"""
        
        regular_graph = nx.Graph()
        
        # Add all nodes with their attributes
        for node_id, node_data in self.base_network.nodes.items():
            regular_graph.add_node(node_id, **node_data.__dict__)
        
        # Create connections based on geographic proximity (k-nearest neighbors)
        for node_id in self.base_nodes:
            node_coord = self.base_network.nodes[node_id].coordinates
            
            # Calculate distances to all other nodes
            distances = []
            for other_id in self.base_nodes:
                if other_id != node_id:
                    other_coord = self.base_network.nodes[other_id].coordinates
                    distance = self._calculate_distance(node_coord, other_coord)
                    distances.append((distance, other_id))
            
            # Connect to k nearest neighbors
            distances.sort()
            connections_made = 0
            
            for distance, neighbor_id in distances:
                if connections_made >= k:
                    break
                
                if not regular_graph.has_edge(node_id, neighbor_id):
                    # Add edge with appropriate transport characteristics
                    self._add_realistic_edge(regular_graph, node_id, neighbor_id, "regular")
                    connections_made += 1
        
        return regular_graph
    
    def _apply_geographic_rewiring(self, graph: nx.Graph, p: float, 
                             preserve_geography: bool = True) -> nx.Graph:
        """Apply Watts-Strogatz rewiring with geographic constraints"""
        
        small_world_graph = graph.copy()
        edges_to_process = list(small_world_graph.edges())
        rewired_count = 0
        
        for u, v in edges_to_process:
            if random.random() < p:
                # Decide whether to rewire this edge
                if self._can_rewire_edge(u, v):
                    # Find suitable rewiring target
                    new_target = self._find_rewiring_target(
                        small_world_graph, u, v, preserve_geography
                    )
                    
                    if new_target:
                        # Remove old edge and add new shortcut
                        small_world_graph.remove_edge(u, v)
                        
                        # Add new edge as shortcut
                        self._add_realistic_edge(small_world_graph, u, new_target, "shortcut")
                        rewired_count += 1
                        
                        print(f"Rewired: {u}-{v} → {u}-{new_target}")
        
        print(f"Rewiring complete: {rewired_count} shortcuts created")
        return small_world_graph
    
    def _can_rewire_edge(self, u: str, v: str) -> bool:
        """Determine if an edge can be rewired based on node importance"""
        
        if not self.preserve_major_hubs:
            return True
        
        # Don't rewire connections involving major hubs
        u_type = self.base_network.nodes[u].node_type
        v_type = self.base_network.nodes[v].node_type
        
        if (u_type.value == 'major_hub' or v_type.value == 'major_hub'):
            return False
        
        return True
    
    def _find_rewiring_target(self, graph: nx.Graph, u: str, v: str, 
                            preserve_geography: bool) -> Optional[str]:
        """Find suitable target for rewiring with geographic constraints"""
        
        u_coord = self.base_network.nodes[u].coordinates
        candidates = []
        
        for candidate in self.base_nodes:
            if candidate == u or candidate == v:
                continue
            
            if graph.has_edge(u, candidate):
                continue  # Already connected
            
            candidate_coord = self.base_network.nodes[candidate].coordinates
            distance = self._calculate_distance(u_coord, candidate_coord)
            
            # Geographic constraint: limit shortcut distance
            if preserve_geography and distance > self.max_rewire_distance:
                continue
            
            # Transport hierarchy constraint
            if not self._is_valid_transport_connection(u, candidate):
                continue
            
            candidates.append((distance, candidate))
        
        if not candidates:
            return None
        
        # Prefer medium-distance connections (true shortcuts)
        # Not too close (already likely connected) or too far (unrealistic)
        candidates.sort(key=lambda x: abs(x[0] - self.max_rewire_distance * 0.6))
        
        return candidates[0][1]
    
    def _is_valid_transport_connection(self, node1: str, node2: str) -> bool:
        """Check if transport connection between nodes makes sense"""
        
        if not self.preserve_hierarchy:
            return True
        
        type1 = self.base_network.nodes[node1].node_type
        type2 = self.base_network.nodes[node2].node_type
        
        # Allow connections between:
        # - Any two transport hubs (creates useful shortcuts)
        # - Transport hub to local station
        # - Local stations in same general area
        
        if (type1.value in ['major_hub', 'transport_hub'] or 
            type2.value in ['major_hub', 'transport_hub']):
            return True
        
        # For local connections, check distance
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        distance = self._calculate_distance(coord1, coord2)
        
        return distance <= 25  # Local connections only
    
    def _add_realistic_edge(self, graph: nx.Graph, node1: str, node2: str, 
                      edge_type: str = "regular", base_data: dict = None):
        """Add edge with ALL required attributes for routing system"""
        print(f"🔧 Adding edge: {node1}-{node2} (type: {edge_type})")
        
        node1_data = self.base_network.nodes[node1]
        node2_data = self.base_network.nodes[node2]
        
        # Calculate distance
        distance = self._calculate_distance(node1_data.coordinates, node2_data.coordinates)
        
        # Determine transport modes
        transport_modes = self._determine_transport_modes(node1_data, node2_data, distance)
        primary_mode = transport_modes[0] if transport_modes else TransportMode.BUS
        
        print(f"   Transport modes: {[m.value for m in transport_modes]}, primary: {primary_mode.value}")
        
        # Calculate travel time
        base_time = distance * 2
        if edge_type == "shortcut":
            base_time *= 0.7
            frequency = 12
        else:
            frequency = 8
        
        # Generate route_id (CRITICAL - avoid "unknown")
        mode_name = primary_mode.value if hasattr(primary_mode, 'value') else 'bus'
        route_id = f"SW_{edge_type}_{mode_name}_{node1}_{node2}"
        
        print(f"   Route ID: {route_id}")
        print(f"   Frequency: {frequency}")
        
        # Add edge with ALL required attributes
        edge_attributes = {
            # Core attributes
            'transport_mode': primary_mode,       # CRITICAL: singular mode
            'transport_modes': transport_modes,   # plural modes list
            'edge_type': edge_type,              # Small-world tracking
            
            # Routing attributes
            'travel_time': base_time,
            'distance': distance,
            'capacity': self._estimate_capacity(transport_modes),
            'frequency': frequency,
            
            # RAPTOR requirements (CRITICAL)
            'route_id': route_id,                # Must not be "unknown"
            'segment_order': 1,                  # Required for RAPTOR
            
            # Additional
            'geographic_distance': distance
        }
        
        print(f"   Adding with attributes: {list(edge_attributes.keys())}")
        
        graph.add_edge(node1, node2, **edge_attributes)
        
        # CRITICAL: Verify the edge was added correctly
        if graph.has_edge(node1, node2):
            actual_attrs = graph[node1][node2]
            actual_keys = list(actual_attrs.keys())
            print(f"   ✅ Edge added with keys: {actual_keys}")
            
            # Check specific critical attributes
            if 'transport_mode' in actual_attrs:
                print(f"   ✅ transport_mode: {actual_attrs['transport_mode']}")
            else:
                print(f"   ❌ transport_mode MISSING!")
                
            if 'route_id' in actual_attrs:
                print(f"   ✅ route_id: {actual_attrs['route_id']}")
            else:
                print(f"   ❌ route_id MISSING!")
        else:
            print(f"   ❌ EDGE NOT ADDED TO GRAPH!")
    
    def _ensure_network_connectivity(self, graph: nx.Graph) -> nx.Graph:
        """Ensure network remains connected after rewiring"""
        
        if nx.is_connected(graph):
            return graph
        
        print("Network disconnected after rewiring - adding connectivity edges...")
        
        # Find disconnected components
        components = list(nx.connected_components(graph))
        main_component = max(components, key=len)
        
        for component in components:
            if component == main_component:
                continue
            
            # Connect each component to main component
            min_distance = float('inf')
            best_connection = None
            
            for node1 in main_component:
                coord1 = self.base_network.nodes[node1].coordinates
                for node2 in component:
                    coord2 = self.base_network.nodes[node2].coordinates
                    distance = self._calculate_distance(coord1, coord2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_connection = (node1, node2)
            
            if best_connection:
                self._add_realistic_edge(graph, best_connection[0], best_connection[1], "connectivity")
        
        return graph
    
    def _add_transport_hierarchy(self, graph: nx.Graph) -> nx.Graph:
        """Add missing connections to maintain transport hierarchy"""
        
        # Ensure major hubs are well connected
        major_hubs = [n for n in graph.nodes() 
                     if self.base_network.nodes[n].node_type.value == 'major_hub']
        
        for i, hub1 in enumerate(major_hubs):
            for hub2 in major_hubs[i+1:]:
                if not graph.has_edge(hub1, hub2):
                    coord1 = self.base_network.nodes[hub1].coordinates
                    coord2 = self.base_network.nodes[hub2].coordinates
                    distance = self._calculate_distance(coord1, coord2)
                    
                    if distance < 35:  # Connect nearby major hubs
                        self._add_realistic_edge(graph, hub1, hub2, "hierarchy")
        
        return graph
    
    def _calculate_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between coordinates"""
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    def analyze_small_world_properties(self, graph: nx.Graph) -> Dict:
        """Analyze small-world properties of the generated network"""
        
        if not graph or graph.number_of_nodes() == 0:
            return {'error': 'Empty or invalid graph'}
        
        try:
            # Use graph methods, not base_network methods
            n = graph.number_of_nodes()
            m = graph.number_of_edges()
            
            if not nx.is_connected(graph):
                # Handle disconnected components
                largest_cc = max(nx.connected_components(graph), key=len)
                graph = graph.subgraph(largest_cc).copy()
                n = graph.number_of_nodes()
                m = graph.number_of_edges()
            
            # Calculate small-world metrics
            clustering = nx.average_clustering(graph)
            path_length = nx.average_shortest_path_length(graph)
            
            # Count shortcuts (edges marked as 'shortcut')
            shortcuts = sum(1 for u, v, d in graph.edges(data=True) 
                        if d.get('edge_type') == 'shortcut')
            
            # Calculate small-world coefficient
            if n > 3:  # Need minimum nodes for comparison
                # Create equivalent random graph
                p_random = 2 * m / (n * (n - 1))
                random_graph = nx.erdos_renyi_graph(n, p_random)
                
                if nx.is_connected(random_graph):
                    random_clustering = nx.average_clustering(random_graph)
                    random_path_length = nx.average_shortest_path_length(random_graph)
                    
                    # Small-world metrics
                    gamma = clustering / random_clustering if random_clustering > 0 else 0
                    lambda_val = path_length / random_path_length if random_path_length > 0 else 0
                    sigma = gamma / lambda_val if lambda_val > 0 else 0
                else:
                    gamma = lambda_val = sigma = 0
            else:
                gamma = lambda_val = sigma = 0
            
            # Centrality measures (for connected graphs only)
            try:
                betweenness = nx.betweenness_centrality(graph)
                closeness = nx.closeness_centrality(graph)
            except:
                betweenness = closeness = {}
            
            return {
                'nodes': n,
                'edges': m,
                'avg_degree': 2*m/n if n > 0 else 0,
                'clustering_coefficient': clustering,
                'avg_path_length': path_length,
                'small_world_sigma': sigma,
                'gamma': gamma,
                'lambda': lambda_val,
                'shortcuts_created': shortcuts,
                'shortcut_percentage': shortcuts/m*100 if m > 0 else 0,
                'max_betweenness': max(betweenness.values()) if betweenness else 0,
                'avg_betweenness': sum(betweenness.values())/len(betweenness) if betweenness else 0,
                'betweenness_concentration': np.std(list(betweenness.values())) if betweenness else 0,
                'max_closeness': max(closeness.values()) if closeness else 0,
                'avg_closeness': sum(closeness.values())/len(closeness) if closeness else 0,
                'closeness_concentration': np.std(list(closeness.values())) if closeness else 0
            }
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}

class SmallWorldParameterOptimizer:
    """Find optimal small-world parameters for transport equity"""
    
    def __init__(self, base_network: SydneyNetworkTopology):
        self.base_network = base_network
        self.generator = SmallWorldTopologyGenerator(base_network)
    
    def optimize_for_equity(self, p_range: List[float] = None, 
                          k_range: List[int] = None) -> Dict:
        """Find small-world parameters that optimize transport equity"""
        
        if p_range is None:
            p_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        
        if k_range is None:
            k_range = [3, 4, 5, 6]
        
        results = []
        
        for k in k_range:
            for p in p_range:
                print(f"Testing k={k}, p={p:.3f}...")
                
                # Generate network
                network = self.generator.generate_small_world_network(
                    rewiring_probability=p,
                    initial_neighbors=k
                )
                
                # Analyze properties
                props = self.generator.analyze_small_world_properties(network)
                props['k'] = k
                props['p'] = p
                
                # Calculate equity-relevant metrics
                props['peripheral_benefit'] = self._calculate_peripheral_benefit(network)
                props['hub_dominance'] = self._calculate_hub_dominance(network)
                
                results.append(props)
        
        return results
    
    def _calculate_peripheral_benefit(self, graph: nx.Graph) -> float:
        """Calculate how much peripheral nodes benefit from shortcuts"""
        
        # Identify peripheral nodes (low degree, high distance from major hubs)
        major_hubs = [n for n in graph.nodes() 
                     if self.base_network.nodes[n].node_type.value == 'major_hub']
        
        peripheral_nodes = []
        for node in graph.nodes():
            if graph.degree(node) <= 3:  # Low degree
                # Check distance to nearest major hub
                min_dist = float('inf')
                node_coord = self.base_network.nodes[node].coordinates
                
                for hub in major_hubs:
                    hub_coord = self.base_network.nodes[hub].coordinates
                    dist = ((node_coord[0] - hub_coord[0])**2 + 
                           (node_coord[1] - hub_coord[1])**2)**0.5
                    min_dist = min(min_dist, dist)
                
                if min_dist > 25:  # Far from major hubs
                    peripheral_nodes.append(node)
        
        if not peripheral_nodes or not major_hubs:
            return 0
        
        # Calculate average path length from peripheral to major hubs
        total_benefit = 0
        count = 0
        
        for peripheral in peripheral_nodes:
            for hub in major_hubs:
                try:
                    path_length = nx.shortest_path_length(graph, peripheral, hub)
                    # Benefit = inverse of path length (shorter paths = higher benefit)
                    total_benefit += 1 / path_length if path_length > 0 else 0
                    count += 1
                except nx.NetworkXNoPath:
                    continue
        
        return total_benefit / count if count > 0 else 0
    
    def _calculate_hub_dominance(self, graph: nx.Graph) -> float:
        """Calculate how dominant hubs are (higher = more inequality)"""
        
        betweenness = nx.betweenness_centrality(graph)
        
        # Hub dominance = concentration of betweenness centrality
        values = list(betweenness.values())
        if not values:
            return 0
        
        # Gini coefficient of betweenness centrality
        values.sort()
        n = len(values)
        cumsum = np.cumsum(values)
        
        if cumsum[-1] == 0:
            return 0
        
        gini = (2 * sum((i+1) * val for i, val in enumerate(values))) / (n * cumsum[-1]) - (n+1)/n
        return max(0, gini)