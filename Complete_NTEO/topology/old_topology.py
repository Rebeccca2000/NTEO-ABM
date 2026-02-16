class DegreeConstrainedTopologyGenerator:
    """Generate degree-constrained networks maintaining Sydney structure"""
    
    def __init__(self, base_network: SydneyNetworkTopology):
        self.base_network = base_network
        self.route_counter = {'train': 100, 'bus': 1000}  # Counters for new routes
    
    def _add_realistic_edge(self, graph: nx.Graph, node1: str, node2: str):
        """Add edge with realistic transport characteristics - FIXED VERSION"""
        
        # 🎯 CRITICAL FIX: Check if this edge exists in original Sydney network first
        if self.base_network.graph.has_edge(node1, node2):
            # Preserve original Sydney edge data completely
            original_data = self.base_network.graph[node1][node2]
            graph.add_edge(node1, node2, **original_data)
            print(f"   ✅ Preserved Sydney edge: {node1}->{node2} (route: {original_data.get('route_id', 'unknown')})")
            return

        # For new edges, calculate properties
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        
        type1 = self.base_network.nodes[node1].node_type
        type2 = self.base_network.nodes[node2].node_type
        
        # Create Sydney-style route ID (not generic!)
        if (type1 in [NodeType.MAJOR_HUB, NodeType.TRANSPORT_HUB] and 
            type2 in [NodeType.MAJOR_HUB, NodeType.TRANSPORT_HUB] and distance > 15):
            
            mode = TransportMode.TRAIN
            travel_time = distance * 0.8
            capacity = 2000
            frequency = 12
            
            # 🎯 Create Sydney-style train route ID
            self.route_counter['train'] += 1
            route_id = f"T{self.route_counter['train']}_DEGREE"  # T101_DEGREE, T102_DEGREE, etc.
            
        else:
            mode = TransportMode.BUS
            travel_time = distance * 1.2
            capacity = 600
            frequency = 8
            
            # 🎯 Create Sydney-style bus route ID  
            self.route_counter['bus'] += 1
            route_id = f"BUS_{self.route_counter['bus']}"  # BUS_1001, BUS_1002, etc.
        
        # Add edge with COMPLETE Sydney-style attributes
        edge_data = {
            'transport_mode': mode,
            'travel_time': travel_time,
            'capacity': capacity,
            'frequency': frequency,
            'distance': distance,
            'route_id': route_id,  # 🎯 KEY FIX: Add route_id!
            'segment_order': 1,
            'edge_type': 'degree_generated',
            'weight': travel_time
        }
        
        graph.add_edge(node1, node2, **edge_data)
        print(f"   ➕ Added new edge: {node1}->{node2} (route: {route_id})")

    def generate_degree_constrained_network(self, target_degree: int) -> nx.Graph:
        """Generate degree-constrained network preserving Sydney routes"""
        
        # Start with a copy of the base Sydney network
        graph = self.base_network.graph.copy()
        
        print(f"   Starting with Sydney base: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Classify nodes by importance
        major_hubs = [node_id for node_id, node_data in self.base_network.nodes.items()
                     if node_data.node_type == NodeType.MAJOR_HUB]
        transport_hubs = [node_id for node_id, node_data in self.base_network.nodes.items()
                         if node_data.node_type == NodeType.TRANSPORT_HUB]
        other_nodes = [node_id for node_id, node_data in self.base_network.nodes.items()
                      if node_data.node_type not in [NodeType.MAJOR_HUB, NodeType.TRANSPORT_HUB]]
        
        # Apply degree constraints while preserving Sydney structure
        self._preserve_sydney_backbone(graph, major_hubs, transport_hubs)
        self._adjust_degrees_intelligently(graph, target_degree, major_hubs, transport_hubs, other_nodes)
        
        print(f"   ✅ Degree-constrained network: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges, target_degree={target_degree}")
        
        return graph
    
    def _preserve_sydney_backbone(self, graph: nx.Graph, major_hubs: List[str], transport_hubs: List[str]):
        """Ensure critical Sydney connections are preserved"""
        
        # Critical Sydney connections that must be preserved
        critical_connections = [
            ("CENTRAL", "WYNYARD"),      # T1 line backbone
            ("CENTRAL", "REDFERN"),      # T4 line backbone  
            ("CENTRAL", "TOWN_HALL"),    # CBD core
            ("WYNYARD", "MILSONS_POINT"), # Harbour Bridge
            ("CHATSWOOD", "NORTH_SYDNEY"), # North Shore main
        ]
        
        # Ensure these connections exist with proper route IDs
        for node1, node2 in critical_connections:
            if node1 in graph.nodes() and node2 in graph.nodes():
                if not graph.has_edge(node1, node2):
                    # Find appropriate route ID from existing Sydney network
                    route_id = self._find_appropriate_route_id(node1, node2)
                    self._add_sydney_edge(graph, node1, node2, route_id)
    
    def _adjust_degrees_intelligently(self, graph: nx.Graph, target_degree: int, 
                                    major_hubs: List[str], transport_hubs: List[str], other_nodes: List[str]):
        """Adjust node degrees while maintaining Sydney characteristics"""
        
        # Allow major hubs to exceed target degree (they're important)
        for hub in major_hubs:
            while graph.degree(hub) < target_degree + 2:  # Hubs get bonus connections
                candidates = self._find_connection_candidates(graph, hub, major_hubs + transport_hubs)
                if not candidates:
                    break
                best_candidate = self._select_best_candidate(hub, candidates)
                self._add_intelligent_edge(graph, hub, best_candidate)
        
        # Connect transport hubs to achieve target degree
        for hub in transport_hubs:
            while graph.degree(hub) < target_degree:
                candidates = self._find_connection_candidates(graph, hub, major_hubs + transport_hubs + other_nodes)
                if not candidates:
                    break
                best_candidate = self._select_best_candidate(hub, candidates)
                self._add_intelligent_edge(graph, hub, best_candidate)
        
        # Connect remaining nodes
        for node in other_nodes:
            while graph.degree(node) < target_degree:
                candidates = self._find_connection_candidates(graph, node, list(graph.nodes()))
                if not candidates:
                    break
                best_candidate = self._select_best_candidate(node, candidates)
                self._add_intelligent_edge(graph, node, best_candidate)
    
    def _find_connection_candidates(self, graph: nx.Graph, node: str, potential_nodes: List[str]) -> List[str]:
        """Find valid candidates for connection"""
        
        candidates = []
        for candidate in potential_nodes:
            if (candidate != node and 
                not graph.has_edge(node, candidate) and
                self._is_geographically_reasonable(node, candidate)):
                candidates.append(candidate)
        
        return candidates
    
    def _select_best_candidate(self, node: str, candidates: List[str]) -> str:
        """Select best candidate based on Sydney transport logic"""
        
        node_coord = self.base_network.nodes[node].coordinates
        node_type = self.base_network.nodes[node].node_type
        
        def candidate_score(candidate):
            cand_coord = self.base_network.nodes[candidate].coordinates
            cand_type = self.base_network.nodes[candidate].node_type
            
            # Calculate distance
            distance = np.sqrt((node_coord[0] - cand_coord[0])**2 + (node_coord[1] - cand_coord[1])**2)
            
            # Prefer connections to important hubs
            importance_bonus = 0
            if cand_type in [NodeType.MAJOR_HUB, NodeType.TRANSPORT_HUB]:
                importance_bonus = 20
            
            # Prefer reasonable distances
            distance_penalty = distance * 2
            
            return importance_bonus - distance_penalty
        
        return max(candidates, key=candidate_score)
    
    def _is_geographically_reasonable(self, node1: str, node2: str, max_distance: float = 40) -> bool:
        """Check if connection is geographically reasonable"""
        
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        
        return distance <= max_distance
    
    def _add_intelligent_edge(self, graph: nx.Graph, node1: str, node2: str):
        """Add edge with intelligent Sydney-style route ID assignment"""
        
        # Check if this edge exists in original Sydney network
        if self.base_network.graph.has_edge(node1, node2):
            # Use original Sydney edge data
            original_data = self.base_network.graph[node1][node2]
            graph.add_edge(node1, node2, **original_data)
            return
        
        # Check if this edge can extend an existing route
        extended_route_id = self._find_route_extension(node1, node2, graph)
        if extended_route_id:
            self._add_sydney_edge(graph, node1, node2, extended_route_id, edge_type="degree_extension")
            return
        
        # Create new Sydney-style route
        new_route_id = self._create_new_sydney_route_id(node1, node2)
        self._add_sydney_edge(graph, node1, node2, new_route_id, edge_type="degree_new")
    
    def _find_route_extension(self, node1: str, node2: str, graph: nx.Graph) -> Optional[str]:
        """Find existing route that this edge could extend"""
        
        # Check existing connections from both nodes
        for neighbor in graph.neighbors(node1):
            edge_data = graph[node1][neighbor]
            if 'route_id' in edge_data:
                route_id = edge_data['route_id']
                # Check if this route could logically extend to node2
                if self._can_route_extend(route_id, node1, node2):
                    return route_id
        
        for neighbor in graph.neighbors(node2):
            edge_data = graph[node2][neighbor]
            if 'route_id' in edge_data:
                route_id = edge_data['route_id']
                if self._can_route_extend(route_id, node2, node1):
                    return route_id
        
        return None
    
    def _can_route_extend(self, route_id: str, from_node: str, to_node: str) -> bool:
        """Check if a route can logically be extended"""
        
        # Train routes can extend longer distances
        if 'T1' in route_id or 'T4' in route_id or 'TRAIN' in route_id:
            max_distance = 30
        else:  # Bus routes
            max_distance = 20
        
        return self._is_geographically_reasonable(from_node, to_node, max_distance)
    
    def _create_new_sydney_route_id(self, node1: str, node2: str) -> str:
        """Create new Sydney-style route ID"""
        
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        
        type1 = self.base_network.nodes[node1].node_type
        type2 = self.base_network.nodes[node2].node_type
        
        # Long distance between hubs = train
        if (distance > 20 and 
            type1 in [NodeType.MAJOR_HUB, NodeType.TRANSPORT_HUB] and
            type2 in [NodeType.MAJOR_HUB, NodeType.TRANSPORT_HUB]):
            
            self.route_counter['train'] += 1
            return f"T{self.route_counter['train']}_DEGREE"
        
        # Otherwise = bus
        self.route_counter['bus'] += 1
        return f"BUS_{self.route_counter['bus']}"
    
    def _find_appropriate_route_id(self, node1: str, node2: str) -> str:
        """Find appropriate route ID for critical connections"""
        
        # Map critical connections to appropriate routes
        connection_routes = {
            ("CENTRAL", "WYNYARD"): "T1_WESTERN",
            ("CENTRAL", "REDFERN"): "T4_ILLAWARRA", 
            ("CENTRAL", "TOWN_HALL"): "WALK",
            ("WYNYARD", "MILSONS_POINT"): "T1_NORTH_SHORE",
            ("CHATSWOOD", "NORTH_SYDNEY"): "T1_NORTH_SHORE",
        }
        
        # Try both directions
        if (node1, node2) in connection_routes:
            return connection_routes[(node1, node2)]
        elif (node2, node1) in connection_routes:
            return connection_routes[(node2, node1)]
        
        # Default to creating new route
        return self._create_new_sydney_route_id(node1, node2)
    
    def _add_sydney_edge(self, graph: nx.Graph, node1: str, node2: str, route_id: str, edge_type: str = "preserved"):
        """Add edge with proper Sydney transport attributes"""
        
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        
        # Determine transport mode from route ID
        if 'T1' in route_id or 'T4' in route_id or 'T8' in route_id or 'TRAIN' in route_id:
            transport_mode = TransportMode.TRAIN
            travel_time = distance * 0.8  # Trains are faster
            capacity = 1500
            frequency = 12
        elif 'WALK' in route_id:
            transport_mode = TransportMode.WALKING
            travel_time = distance * 2.0
            capacity = 1000
            frequency = 60
        else:  # Bus
            transport_mode = TransportMode.BUS
            travel_time = distance * 1.2
            capacity = 600
            frequency = 8
        
        # Add edge with complete Sydney-style attributes
        graph.add_edge(node1, node2,
                      transport_mode=transport_mode,
                      travel_time=travel_time,
                      capacity=capacity,
                      frequency=frequency,
                      distance=distance,
                      route_id=route_id,  # 🎯 KEY: Preserve Sydney route ID!
                      segment_order=1,
                      edge_type=edge_type,
                      weight=travel_time)
        

class SmallWorldTopologyGenerator:
    """Generate small-world networks preserving Sydney structure"""
    
    def __init__(self, base_network: SydneyNetworkTopology):
        self.base_network = base_network
        self.route_counter = {'train': 200, 'bus': 2000}  # Different range from degree-constrained
    
    def generate_small_world_network(self, rewiring_probability: float, 
                                   initial_neighbors: int = 4,
                                   preserve_geography: bool = True) -> nx.Graph:
        """Generate small-world network using Watts-Strogatz with Sydney preservation"""
        
        print(f"   Generating small-world network (p={rewiring_probability}, k={initial_neighbors})")
        
        # Step 1: Start with Sydney base network
        graph = self.base_network.graph.copy()
        
        # Step 2: Create regular lattice-like structure while preserving Sydney backbone
        self._create_regular_sydney_structure(graph, initial_neighbors)
        
        # Step 3: Rewire edges to create small-world properties
        self._rewire_edges_with_sydney_preservation(graph, rewiring_probability, preserve_geography)
        
        print(f"   ✅ Small-world network: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges, p={rewiring_probability}")
        
        return graph
    
    def _create_regular_sydney_structure(self, graph: nx.Graph, initial_neighbors: int):
        """Create regular structure while maintaining Sydney backbone"""
        
        # Identify Sydney backbone (critical connections to preserve)
        sydney_backbone = self._identify_sydney_backbone(graph)
        
        # Add regular connections to achieve initial_neighbors degree
        nodes = list(graph.nodes())
        for i, node in enumerate(nodes):
            current_degree = graph.degree(node)
            
            # Calculate how many more connections needed
            target_connections = max(0, initial_neighbors - current_degree)
            
            # Find best candidates for regular connections
            candidates = self._find_regular_candidates(graph, node, nodes, i, initial_neighbors)
            
            # Add connections up to target
            added = 0
            for candidate in candidates:
                if added >= target_connections:
                    break
                if not graph.has_edge(node, candidate):
                    self._add_small_world_edge(graph, node, candidate, "regular")
                    added += 1
    
    def _identify_sydney_backbone(self, graph: nx.Graph) -> set:
        """Identify critical Sydney connections that should never be rewired"""
        
        backbone_edges = set()
        
        # Core Sydney train lines - NEVER rewire these
        critical_routes = ['T1_WESTERN', 'T1_NORTH_SHORE', 'T4_EASTERN', 'T4_ILLAWARRA', 'T8_AIRPORT']
        
        for edge in graph.edges(data=True):
            node1, node2, data = edge
            route_id = data.get('route_id', '')
            
            # Preserve core Sydney train routes
            if any(route in route_id for route in critical_routes):
                backbone_edges.add((node1, node2))
                backbone_edges.add((node2, node1))  # Undirected
            
            # Preserve major hub connections
            if (self._is_major_hub(node1) and self._is_major_hub(node2)):
                backbone_edges.add((node1, node2))
                backbone_edges.add((node2, node1))
        
        print(f"   Protected Sydney backbone: {len(backbone_edges)//2} critical edges")
        return backbone_edges
    
    def _find_regular_candidates(self, graph: nx.Graph, node: str, all_nodes: List[str], 
                               node_index: int, k: int) -> List[str]:
        """Find candidates for regular lattice-like connections"""
        
        candidates = []
        
        # Ring lattice approach: connect to k/2 neighbors on each side
        n = len(all_nodes)
        for j in range(1, k//2 + 1):
            # Right neighbors
            right_idx = (node_index + j) % n
            right_neighbor = all_nodes[right_idx]
            if self._is_geographically_reasonable(node, right_neighbor):
                candidates.append(right_neighbor)
            
            # Left neighbors  
            left_idx = (node_index - j) % n
            left_neighbor = all_nodes[left_idx]
            if self._is_geographically_reasonable(node, left_neighbor):
                candidates.append(left_neighbor)
        
        # Add some geographic neighbors for realism
        node_coord = self.base_network.nodes[node].coordinates
        geographic_neighbors = []
        
        for other_node in all_nodes:
            if other_node != node:
                distance = self._calculate_distance(node, other_node)
                if 5 < distance < 25:  # Medium distance connections
                    geographic_neighbors.append(other_node)
        
        # Sort by distance and add closest ones
        geographic_neighbors.sort(key=lambda x: self._calculate_distance(node, x))
        candidates.extend(geographic_neighbors[:k//2])
        
        return candidates
    
    def _rewire_edges_with_sydney_preservation(self, graph: nx.Graph, 
                                             rewiring_probability: float,
                                             preserve_geography: bool):
        """Rewire edges while preserving Sydney structure"""
        
        sydney_backbone = self._identify_sydney_backbone(graph)
        edges_to_consider = [(u, v) for u, v in graph.edges() 
                           if (u, v) not in sydney_backbone and (v, u) not in sydney_backbone]
        
        rewired_count = 0
        total_eligible = len(edges_to_consider)
        
        print(f"   Rewiring {total_eligible} edges (protecting {len(sydney_backbone)//2} backbone edges)")
        
        for u, v in edges_to_consider:
            if random.random() < rewiring_probability:
                # Store original edge data before rewiring
                original_edge_data = graph[u][v].copy()
                
                # Find rewiring target
                rewiring_target = self._find_rewiring_target(graph, u, v, preserve_geography)
                
                if rewiring_target and not graph.has_edge(u, rewiring_target):
                    # Remove old edge
                    graph.remove_edge(u, v)
                    
                    # Add rewired edge (creates shortcut)
                    self._add_small_world_edge(graph, u, rewiring_target, "shortcut", original_edge_data)
                    rewired_count += 1
        
        print(f"   Rewired {rewired_count}/{total_eligible} edges ({rewired_count/max(total_eligible,1)*100:.1f}%)")
    
    def _find_rewiring_target(self, graph: nx.Graph, u: str, v: str, 
                            preserve_geography: bool) -> Optional[str]:
        """Find suitable target for rewiring that maintains Sydney realism"""
        
        # Get all possible targets (not already connected to u)
        possible_targets = [node for node in graph.nodes() 
                          if node != u and node != v and not graph.has_edge(u, node)]
        
        if not possible_targets:
            return None
        
        # Filter by geography if required
        if preserve_geography:
            u_coord = self.base_network.nodes[u].coordinates
            geographic_targets = []
            
            for target in possible_targets:
                target_coord = self.base_network.nodes[target].coordinates
                distance = np.sqrt((u_coord[0] - target_coord[0])**2 + 
                                 (u_coord[1] - target_coord[1])**2)
                
                # Allow longer shortcuts for small-world effect, but not too crazy
                if distance < 50:  # Max shortcut distance
                    geographic_targets.append(target)
            
            possible_targets = geographic_targets if geographic_targets else possible_targets
        
        # Prefer connections to important hubs for small-world effect
        hub_targets = [target for target in possible_targets 
                      if self._is_major_hub(target) or self._is_transport_hub(target)]
        
        if hub_targets:
            return random.choice(hub_targets)
        else:
            return random.choice(possible_targets) if possible_targets else None
    
    def _add_small_world_edge(self, graph: nx.Graph, node1: str, node2: str, 
                            edge_type: str, base_data: dict = None):
        """Add small-world edge with Sydney-style route ID"""
        
        # Check if original Sydney edge exists
        if self.base_network.graph.has_edge(node1, node2):
            original_data = self.base_network.graph[node1][node2]
            graph.add_edge(node1, node2, **original_data)
            return
        
        # Calculate properties
        distance = self._calculate_distance(node1, node2)
        
        # Determine route type based on edge type and distance
        if edge_type == "shortcut":
            # Shortcuts are typically bus routes (more flexible)
            route_id = self._create_shortcut_route_id(node1, node2)
            transport_mode = TransportMode.BUS
            travel_time = distance * 0.9  # Shortcuts are faster
            capacity = 800  # Higher capacity for shortcuts
            frequency = 15
        else:  # regular
            # Regular connections follow Sydney patterns
            route_id = self._create_regular_route_id(node1, node2)
            if distance > 20 and (self._is_major_hub(node1) or self._is_major_hub(node2)):
                transport_mode = TransportMode.TRAIN
                travel_time = distance * 0.8
                capacity = 1500
                frequency = 12
            else:
                transport_mode = TransportMode.BUS
                travel_time = distance * 1.2
                capacity = 600
                frequency = 8
        
        # Add edge with complete attributes
        graph.add_edge(node1, node2,
                      transport_mode=transport_mode,
                      travel_time=travel_time,
                      capacity=capacity,
                      frequency=frequency,
                      distance=distance,
                      route_id=route_id,  # 🎯 KEY: Sydney-style route ID!
                      segment_order=1,
                      edge_type=edge_type,
                      weight=travel_time)
    
    def _create_shortcut_route_id(self, node1: str, node2: str) -> str:
        """Create route ID for small-world shortcuts"""
        
        # Express bus routes for shortcuts
        self.route_counter['bus'] += 1
        return f"BUS_E{self.route_counter['bus']}"  # Express bus
    
    def _create_regular_route_id(self, node1: str, node2: str) -> str:
        """Create route ID for regular connections"""
        
        distance = self._calculate_distance(node1, node2)
        
        if (distance > 20 and 
            (self._is_major_hub(node1) or self._is_major_hub(node2))):
            # Train for long hub connections
            self.route_counter['train'] += 1
            return f"T{self.route_counter['train']}_SW"  # Small-world train
        else:
            # Regular bus
            self.route_counter['bus'] += 1
            return f"BUS_{self.route_counter['bus']}"
    
    # Utility methods
    def _is_major_hub(self, node: str) -> bool:
        return (node in self.base_network.nodes and 
                self.base_network.nodes[node].node_type == NodeType.MAJOR_HUB)
    
    def _is_transport_hub(self, node: str) -> bool:
        return (node in self.base_network.nodes and 
                self.base_network.nodes[node].node_type == NodeType.TRANSPORT_HUB)
    
    def _calculate_distance(self, node1: str, node2: str) -> float:
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    def _is_geographically_reasonable(self, node1: str, node2: str, max_distance: float = 35) -> bool:
        return self._calculate_distance(node1, node2) <= max_distance
    


class ScaleFreeTopologyGenerator:
    """Generate scale-free networks preserving Sydney structure"""
    
    def __init__(self, base_network: SydneyNetworkTopology):
        self.base_network = base_network
        self.route_counter = {'train': 300, 'bus': 3000}  # Different range from others
    
    def generate_scale_free_network(self, m_edges: int, alpha: float = 1.0,
                                  preserve_geography: bool = True,
                                  hub_preference: float = 2.0) -> nx.Graph:
        """Generate scale-free network using preferential attachment with Sydney preservation"""
        
        print(f"   Generating scale-free network (m={m_edges}, α={alpha})")
        
        # Step 1: Start with Sydney base network as seed
        graph = self.base_network.graph.copy()
        
        # Step 2: Identify existing hubs in Sydney network
        sydney_hubs = self._identify_sydney_hubs(graph)
        
        # Step 3: Apply preferential attachment while preserving Sydney structure
        self._apply_preferential_attachment(graph, m_edges, alpha, sydney_hubs, 
                                          preserve_geography, hub_preference)
        
        print(f"   ✅ Scale-free network: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges, m={m_edges}")
        
        return graph
    
    def _identify_sydney_hubs(self, graph: nx.Graph) -> Dict[str, float]:
        """Identify and rank Sydney hubs by importance"""
        
        hubs = {}
        
        for node_id, node_data in self.base_network.nodes.items():
            importance = 0
            
            # Base importance from node type
            if node_data.node_type == NodeType.MAJOR_HUB:
                importance += 10
            elif node_data.node_type == NodeType.TRANSPORT_HUB:
                importance += 5
            elif node_data.node_type == NodeType.LOCAL_STATION:
                importance += 2
            
            # Additional importance from current connectivity
            if node_id in graph.nodes():
                importance += graph.degree(node_id) * 0.5
            
            # Employment/population weight
            importance += node_data.employment_weight * 3
            importance += node_data.population_weight * 2
            
            hubs[node_id] = importance
        
        # Normalize hub scores
        max_importance = max(hubs.values()) if hubs else 1
        for node in hubs:
            hubs[node] = hubs[node] / max_importance
        
        print(f"   Identified {len(hubs)} Sydney hubs with importance scores")
        return hubs
    
    def _apply_preferential_attachment(self, graph: nx.Graph, m_edges: int, alpha: float,
                                     sydney_hubs: Dict[str, float], preserve_geography: bool,
                                     hub_preference: float):
        """Apply preferential attachment while boosting Sydney hubs"""
        
        nodes_list = list(graph.nodes())
        
        # Calculate initial node attractiveness (combining degree + Sydney importance)
        attractiveness = self._calculate_attractiveness(graph, sydney_hubs, alpha, hub_preference)
        
        # Preferential attachment rounds
        attachment_rounds = max(1, len(nodes_list) // 5)  # Multiple rounds for better distribution
        
        for round_num in range(attachment_rounds):
            print(f"   Attachment round {round_num + 1}/{attachment_rounds}")
            
            # Each round, try to add m_edges connections per node (if beneficial)
            nodes_to_process = nodes_list.copy()
            random.shuffle(nodes_to_process)  # Random order to avoid bias
            
            for node in nodes_to_process:
                current_degree = graph.degree(node)
                
                # Don't over-connect nodes (scale-free should have many low-degree nodes)
                max_new_connections = min(m_edges, max(1, 8 - current_degree))
                
                # Find attachment targets
                targets = self._find_attachment_targets(
                    graph, node, attractiveness, max_new_connections, 
                    preserve_geography, sydney_hubs
                )
                
                # Add connections
                for target in targets:
                    if not graph.has_edge(node, target):
                        self._add_scale_free_edge(graph, node, target, sydney_hubs)
                        
                        # Update attractiveness after adding edge
                        attractiveness[node] += alpha
                        attractiveness[target] += alpha
    
    def _calculate_attractiveness(self, graph: nx.Graph, sydney_hubs: Dict[str, float], 
                                alpha: float, hub_preference: float) -> Dict[str, float]:
        """Calculate node attractiveness for preferential attachment"""
        
        attractiveness = {}
        
        for node in graph.nodes():
            # Base attractiveness from degree (preferential attachment)
            degree_attraction = graph.degree(node) ** alpha
            
            # Sydney hub bonus
            sydney_bonus = sydney_hubs.get(node, 0) * hub_preference
            
            # Geographic centrality bonus (nodes in center of Sydney get bonus)
            if node in self.base_network.nodes:
                coord = self.base_network.nodes[node].coordinates
                # Sydney CBD is around (50, 40), give bonus to central locations
                centrality_distance = np.sqrt((coord[0] - 50)**2 + (coord[1] - 40)**2)
                centrality_bonus = max(0, 2 - centrality_distance / 20)  # Bonus decreases with distance
            else:
                centrality_bonus = 0
            
            attractiveness[node] = degree_attraction + sydney_bonus + centrality_bonus + 1  # +1 to avoid zero
        
        return attractiveness
    
    def _find_attachment_targets(self, graph: nx.Graph, node: str, 
                               attractiveness: Dict[str, float], max_connections: int,
                               preserve_geography: bool, sydney_hubs: Dict[str, float]) -> List[str]:
        """Find targets for preferential attachment"""
        
        # Get potential targets (not already connected)
        potential_targets = [target for target in graph.nodes() 
                           if target != node and not graph.has_edge(node, target)]
        
        # Filter by geography if required
        if preserve_geography:
            geographic_targets = []
            for target in potential_targets:
                if self._is_geographically_reasonable(node, target, max_distance=45):
                    geographic_targets.append(target)
            potential_targets = geographic_targets if geographic_targets else potential_targets
        
        if not potential_targets:
            return []
        
        # Calculate selection probabilities based on attractiveness
        target_probabilities = []
        total_attractiveness = sum(attractiveness.get(target, 1) for target in potential_targets)
        
        for target in potential_targets:
            prob = attractiveness.get(target, 1) / total_attractiveness
            target_probabilities.append(prob)
        
        # Select targets using weighted random selection
        selected_targets = []
        for _ in range(min(max_connections, len(potential_targets))):
            if not potential_targets:
                break
                
            # Weighted random selection
            selected_idx = np.random.choice(len(potential_targets), p=target_probabilities)
            selected_target = potential_targets.pop(selected_idx)
            selected_targets.append(selected_target)
            
            # Remove from probabilities and renormalize
            target_probabilities.pop(selected_idx)
            if target_probabilities:
                prob_sum = sum(target_probabilities)
                target_probabilities = [p / prob_sum for p in target_probabilities]
        
        return selected_targets
    
    def _add_scale_free_edge(self, graph: nx.Graph, node1: str, node2: str, 
                           sydney_hubs: Dict[str, float]):
        """Add scale-free edge with appropriate Sydney route ID"""
        
        # Check if this is an original Sydney edge
        if self.base_network.graph.has_edge(node1, node2):
            original_data = self.base_network.graph[node1][node2]
            graph.add_edge(node1, node2, **original_data)
            return
        
        # Determine edge characteristics based on nodes involved
        distance = self._calculate_distance(node1, node2)
        
        # High-importance hub connections become premium routes
        node1_importance = sydney_hubs.get(node1, 0)
        node2_importance = sydney_hubs.get(node2, 0)
        combined_importance = node1_importance + node2_importance
        
        if combined_importance > 1.5:  # Both high-importance hubs
            # Premium train connection
            route_id = self._create_premium_route_id(node1, node2)
            transport_mode = TransportMode.TRAIN
            travel_time = distance * 0.7  # Premium routes are fastest
            capacity = 2000  # High capacity
            frequency = 15   # High frequency
            
        elif combined_importance > 0.8:  # At least one important hub
            # Express bus connection
            route_id = self._create_express_route_id(node1, node2)
            transport_mode = TransportMode.BUS
            travel_time = distance * 0.9  # Express buses faster than regular
            capacity = 800
            frequency = 12
            
        else:  # Regular connection
            # Standard bus route
            route_id = self._create_standard_route_id(node1, node2)
            transport_mode = TransportMode.BUS
            travel_time = distance * 1.2
            capacity = 600
            frequency = 8
        
        # Add edge with complete Sydney-style attributes
        graph.add_edge(node1, node2,
                      transport_mode=transport_mode,
                      travel_time=travel_time,
                      capacity=capacity,
                      frequency=frequency,
                      distance=distance,
                      route_id=route_id,  # 🎯 KEY: Sydney-style route ID!
                      segment_order=1,
                      edge_type="scale_free",
                      weight=travel_time)
    
    def _create_premium_route_id(self, node1: str, node2: str) -> str:
        """Create premium route ID for major hub connections"""
        self.route_counter['train'] += 1
        return f"T{self.route_counter['train']}_EXPRESS"
    
    def _create_express_route_id(self, node1: str, node2: str) -> str:
        """Create express route ID for important connections"""
        self.route_counter['bus'] += 1
        return f"BUS_X{self.route_counter['bus']}"  # Express bus
    
    def _create_standard_route_id(self, node1: str, node2: str) -> str:
        """Create standard route ID for regular connections"""
        self.route_counter['bus'] += 1
        return f"BUS_{self.route_counter['bus']}"
    
    # Utility methods
    def _calculate_distance(self, node1: str, node2: str) -> float:
        coord1 = self.base_network.nodes[node1].coordinates
        coord2 = self.base_network.nodes[node2].coordinates
        return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    def _is_geographically_reasonable(self, node1: str, node2: str, max_distance: float = 45) -> bool:
        """Check if connection is geographically reasonable (scale-free allows longer connections)"""
        return self._calculate_distance(node1, node2) <= max_distance
    


def __init__(self, topology_type: str = "degree_constrained", 
             degree: int = 3, grid_width: int = 100, grid_height: int = 80,
             rewiring_probability: float = 0.1, initial_neighbors: int = 4,
             attachment_parameter: int = 2, connectivity_level: int = 4,  # ADD THESE
             **kwargs):
        """Initialize with exact same interface as your working system"""
        
        print(f"🏗️ Initializing unified network manager")
        print(f"   Topology: {topology_type}")
        print(f"   Parameter: {degree}")
        print(f"   Grid: {grid_width}×{grid_height}")
        
        # Store configuration
        self.topology_type = topology_type
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Create base network
        from topology.network_topology import SydneyNetworkTopology
        self.base_network = SydneyNetworkTopology()
        self.base_network.initialize_base_sydney_network()
        
        # Generate topology-specific network using available generators
        if topology_type == "degree_constrained":
            # Use your existing DegreeConstrainedTopologyGenerator
            try:
                from topology.network_topology import DegreeConstrainedTopologyGenerator
                generator = DegreeConstrainedTopologyGenerator(self.base_network)
                self.active_network = generator.generate_degree_constrained_network(degree)
                print(f"✅ Degree-constrained network created using DegreeConstrainedTopologyGenerator")
            except Exception as e:
                print(f"⚠️ Failed to create degree-constrained network: {e}")
                # Fallback to base network
                self.active_network = self.base_network.graph if hasattr(self.base_network, 'graph') else self.base_network.get_networkx_graph()
        
        elif topology_type == "small_world":
            # Use unified topology generator for small world
            try:
                from topology.unified_topology_generator import UnifiedTopologyGenerator
                unified_generator = UnifiedTopologyGenerator(self.base_network)
                self.active_network = unified_generator.generate_topology(
                    'small_world', 
                    rewiring_probability,
                    initial_neighbors=initial_neighbors
                )
                print(f"✅ Small-world network created using UnifiedTopologyGenerator")
            except Exception as e:
                print(f"⚠️ Failed to create small-world network: {e}")
                self.active_network = self.base_network.graph if hasattr(self.base_network, 'graph') else self.base_network.get_networkx_graph()
        
        # ADD THESE NEW CASES:
        elif topology_type == "grid":
            try:
                from topology.unified_topology_generator import UnifiedTopologyGenerator
                unified_generator = UnifiedTopologyGenerator(self.base_network)
                self.active_network = unified_generator.generate_topology(
                    'grid', 
                    connectivity_level
                )
                print(f"✅ Grid network created using UnifiedTopologyGenerator")
            except Exception as e:
                print(f"⚠️ Failed to create grid network: {e}")
                self.active_network = self.base_network.graph if hasattr(self.base_network, 'graph') else self.base_network.get_networkx_graph()
        
        elif topology_type == "scale_free":
            try:
                from topology.unified_topology_generator import UnifiedTopologyGenerator
                unified_generator = UnifiedTopologyGenerator(self.base_network)
                self.active_network = unified_generator.generate_topology(
                    'scale_free', 
                    attachment_parameter
                )
                print(f"✅ Scale-free network created using UnifiedTopologyGenerator")
            except Exception as e:
                print(f"⚠️ Failed to create scale-free network: {e}")
                self.active_network = self.base_network.graph if hasattr(self.base_network, 'graph') else self.base_network.get_networkx_graph()
        
        else:
            # Fallback to base network
            print(f"⚠️ Unknown topology type: {topology_type}, using base network")
            self.active_network = self.base_network.graph if hasattr(self.base_network, 'graph') else self.base_network.get_networkx_graph()
        
        # Initialize network components (rest of method unchanged)
        self.spatial_mapper = UnifiedSpatialMapper(self.base_network, grid_width, grid_height)
        self.congestion_model = NetworkCongestionModel(self.active_network)
        self.router = UnifiedNetworkRouter(self.active_network, self.spatial_mapper)
        
        print(f"   📊 Final network: {self.active_network.number_of_nodes()} nodes, {self.active_network.number_of_edges()} edges")
    