from agent_subsidy_pool import SubsidyPoolConfig
import networkx as nx
# In your database.py file, change this line:
# OLD (SQLite):
# DB_CONNECTION_STRING = "sqlite:///mobility_simulation.db"

# NEW (PostgreSQL):
DB_CONNECTION_STRING = "postgresql://username:password@localhost:5432/nteo_abm"


# If you don't have a password set up, use:
DB_CONNECTION_STRING = "postgresql://localhost:5432/nteo_abm"

SIMULATION_STEPS = 144  # One day simulation (144 steps = 24 hours)
# Commuter configuration
num_commuters = 200
income_weights = [0.5, 0.3, 0.2]
health_weights = [0.9, 0.1]
payment_weights = [0.8, 0.2]
disability_weights = [0.2, 0.8]
tech_access_weights = [0.95, 0.05]

age_distribution = {
    (18, 25): 0.2, 
    (26, 35): 0.3, 
    (36, 45): 0.2, 
    (46, 55): 0.15, 
    (56, 65): 0.1, 
    (66, 75): 0.05,
}

# ===== NETWORK TOPOLOGY CONFIGURATION (ONLY) =====
# NETWORK_CONFIG = {
#     'topology_type': 'degree_constrained',  # Options: 'degree_constrained', 'small_world', 'scale_free', 'base_sydney'
#     'degree_constraint': 4,
#     'preserve_hierarchy': True,
#     'grid_width': 100,
#     'grid_height': 80,
#     'sydney_realism': True,  # Create realistic Sydney network structure
# }

# ===== MODE CHOICE AND UTILITY PARAMETERS =====
PENALTY_COEFFICIENTS = {
    'disability_bike_walk': 0.8,
    'age_health_bike_walk': 0.3,
    'no_tech_access_car_bike': 0.1
}

AFFORDABILITY_THRESHOLDS = {
    'low': 25, 'middle': 40, 'high': 130
}

VALUE_OF_TIME = {
    'low': 5, 'middle': 10, 'high': 20
}

FLEXIBILITY_ADJUSTMENTS = {
    'low': 1.15, 'medium': 1.0, 'high': 0.85
}

UTILITY_FUNCTION_BASE_COEFFICIENTS = {
    'beta_C': -0.15, 
    'beta_T': -0.09, 
    'beta_W': -0.04, 
    'beta_A': -0.04, 
    'alpha': -0.01
}

ASC_VALUES = {
    'walk': 1,
    'bike': 1,
    'car': 0,
    'public': 5,
    'maas': 0,
    'default': 0
}

UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS = {
    'beta_C': -0.02,
    'beta_T': -0.09
}

# ===== CONGESTION CONTROL (NETWORK-BASED) =====
CONGESTION_ALPHA = 0.03
CONGESTION_BETA = 1.5
CONGESTION_CAPACITY = 10  # Per network edge
CONGESTION_T_IJ_FREE_FLOW = 1.5
BACKGROUND_TRAFFIC_AMOUNT = 120

# ===== MAAS AND PRICING =====
DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS = {
    'S_base': 0.02,
    'alpha': 0.10,
    'delta': 0.5  
}

public_price_table = {
    'train': {'on_peak': 3, 'off_peak': 2.6},
    'bus': {'on_peak': 2.4, 'off_peak': 2}
}

ALPHA_VALUES = {
    'UberLike1': 0.3,
    'UberLike2': 0.3,
    'BikeShare1': 0.25,
    'BikeShare2': 0.25
}

# Service provider capacity and pricing
UberLike1_capacity = 15
UberLike1_price = 15.5
UberLike2_capacity = 19
UberLike2_price = 16.5
BikeShare1_capacity = 10
BikeShare1_price = 2.5
BikeShare2_capacity = 12
BikeShare2_price = 3

# ===== GOVERNMENT SUBSIDY =====
subsidy_dataset = {
    'low': {'bike': 0.50, 'car': 0.55, 'MaaS_Bundle': 0.5},
    'middle': {'bike': 0.23, 'car': 0.35, 'MaaS_Bundle': 0.35},
    'high': {'bike': 0.1, 'car': 0.05, 'MaaS_Bundle': 0.1},
}

daily_config = SubsidyPoolConfig('daily', 0)
weekly_config = SubsidyPoolConfig('weekly', 22000)
monthly_config = SubsidyPoolConfig('monthly', 80000)

# ===== NETWORK-DERIVED DATA STRUCTURES =====
# These will be populated by the NetworkToRAPTORConverter
stations = {}  # Will be populated from network topology
routes = {}    # Will be populated from network topology
transfers = {} # Will be populated from network topology

def initialize_network_data(network_manager):
    """
    Initialize RAPTOR-compatible data structures from network topology
    """
    global stations, routes, transfers
    
    converter = NetworkToRAPTORConverter(network_manager)
    stations, routes, transfers = converter.extract_raptor_compatible_data()
    
    print(f"Initialized from network: {len(stations.get('train', {}))} train stations, "
          f"{len(stations.get('bus', {}))} bus stops, {len(transfers)} transfers")

# ===== NETWORK-TO-RAPTOR CONVERTER =====
class NetworkToRAPTORConverter:
    def __init__(self, network_manager):
        self.network_manager = network_manager
        self.network = network_manager.active_network
        self.spatial_mapper = network_manager.spatial_mapper

    def extract_raptor_compatible_data(self):
        """Convert network topology to RAPTOR-compatible format"""
        stations = {'train': {}, 'bus': {}}
        routes = {'train': {}, 'bus': {}}
        transfers = {}
        
        # Extract stations from network nodes
        for node_id, node_data in self.network.nodes(data=True):
            grid_coord = self.spatial_mapper.node_to_grid.get(node_id)
            if not grid_coord:
                continue
                
            # Get transport modes for this node
            transport_modes = node_data.get('transport_modes', [])
            
            for mode in transport_modes:
                mode_key = mode.value if hasattr(mode, 'value') else str(mode)
                if mode_key in ['train', 'bus']:
                    stations[mode_key][node_id] = grid_coord
        
        # Extract routes from network edges
        route_segments = {}
        for u, v, edge_data in self.network.edges(data=True):
            route_id = edge_data.get('route_id', f"R_{edge_data.get('transport_mode', 'unknown')}")
            transport_mode = edge_data.get('transport_mode')
            
            if hasattr(transport_mode, 'value'):
                mode_key = transport_mode.value
            else:
                mode_key = str(transport_mode)
                
            if mode_key in ['train', 'bus']:
                if route_id not in route_segments:
                    route_segments[route_id] = {'mode': mode_key, 'segments': []}
                
                segment_order = edge_data.get('segment_order', 0)
                route_segments[route_id]['segments'].append((u, v, segment_order))
        
        # Convert segments to ordered station lists
        for route_id, route_data in route_segments.items():
            mode = route_data['mode']
            segments = sorted(route_data['segments'], key=lambda x: x[2])
            
            if segments:
                # Build station sequence from segments
                station_sequence = [segments[0][0]]  # Start with first station
                for segment in segments:
                    station_sequence.append(segment[1])
                
                routes[mode][route_id] = station_sequence
        
        # Calculate transfers between nodes with multiple transport modes
        for node_id, node_data in self.network.nodes(data=True):
            transport_modes = node_data.get('transport_modes', [])
            
            if len(transport_modes) > 1:
                # Create transfers between different modes at same location
                for i, mode1 in enumerate(transport_modes):
                    for mode2 in transport_modes[i+1:]:
                        mode1_key = mode1.value if hasattr(mode1, 'value') else str(mode1)
                        mode2_key = mode2.value if hasattr(mode2, 'value') else str(mode2)
                        
                        if mode1_key in ['train', 'bus'] and mode2_key in ['train', 'bus']:
                            transfer_time = self._calculate_transfer_time(mode1_key, mode2_key, node_data)
                            transfers[(node_id, node_id)] = transfer_time
        
        # Add inter-station transfers for nearby stations
        self._add_proximity_transfers(transfers, stations)
        
        return stations, routes, transfers
    
    def _calculate_transfer_time(self, mode1, mode2, node_data):
        """Calculate transfer time between modes at a node"""
        base_transfer_time = 0.5
        
        # Different transfer times based on mode combination
        if 'train' in [mode1, mode2] and 'bus' in [mode1, mode2]:
            return base_transfer_time + 0.3  # Train-bus transfer
        else:
            return base_transfer_time  # Same mode or other combinations
    
    def _add_proximity_transfers(self, transfers, stations):
        """Add transfers between nearby stations of different modes"""
        all_stations = {}
        for mode in ['train', 'bus']:
            for station_id, coord in stations[mode].items():
                all_stations[station_id] = (coord, mode)
        
        for station1_id, (coord1, mode1) in all_stations.items():
            for station2_id, (coord2, mode2) in all_stations.items():
                if station1_id != station2_id and mode1 != mode2:
                    # Calculate distance
                    distance = ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
                    
                    # If close enough, add transfer
                    if distance <= 3:
                        transfer_time = 0.5 + distance * 0.2
                        transfers[(station1_id, station2_id)] = transfer_time
                        transfers[(station2_id, station1_id)] = transfer_time


# Add these lines to your existing database.py file

# ===== UPDATED NETWORK TOPOLOGY CONFIGURATION =====
# NETWORK_CONFIG = {
#     # Topology Type Options: 'degree_constrained', 'small_world', 'scale_free', 'base_sydney'
#     'topology_type': 'small_world',  # Changed to small_world for analysis
    
#     # Common parameters
#     'grid_width': 100,
#     'grid_height': 80,
#     'sydney_realism': True,
#     'preserve_hierarchy': True,
    
#     # Degree-constrained specific parameters
#     'degree_constraint': 3,
    
#     # Small-world specific parameters (Watts-Strogatz model)
#     'rewiring_probability': 0.1,          # p ∈ [0.01, 0.5] 
#     'initial_neighbors': 4,               # k parameter for initial regular network
#     'preserve_geography': True,           # Maintain geographic realism in rewiring
#     'max_rewire_distance': 40,           # Maximum distance for shortcuts
    
#     # Scale-free specific parameters (for future use)
#     'attachment_parameter': 2,            # m parameter for Barabási-Albert
#     'initial_complete_graph_size': 5,     # m0 parameter
    
#     # Hybrid topology parameters (for future use)
#     'rail_topology': 'scale_free',        # Hub-based rail network
#     'bus_topology': 'small_world',        # Small-world bus network
#     'local_topology': 'degree_constrained' # Regular local connections
# }

NETWORK_CONFIG = {
    # CHANGE THIS LINE from whatever it was before:
    'topology_type': 'scale_free',        # Changed from 'degree_constrained' or 'small_world'
    
    # Keep these existing parameters:
    'grid_width': 100,
    'grid_height': 80,
    'sydney_realism': True,
    'preserve_hierarchy': True,
    
    # ADD THESE NEW SCALE-FREE PARAMETERS:
    'attachment_parameter': 2,            # m parameter: number of edges per new node (1-3)
    'preferential_strength': 1.0,        # α parameter: preferential attachment strength (0.5-1.5)
    'preserve_geography': True,           # Maintain Sydney geographic constraints
    'max_attachment_distance': 50,       # Maximum distance for connections
    
    # Optional: Remove old parameters you don't need anymore:
    # 'degree_constraint': 3,             # Remove this if it exists
    # 'rewiring_probability': 0.1,        # Remove this if it exists
}
# ===== SMALL-WORLD ANALYSIS CONFIGURATIONS =====

# Define parameter ranges for small-world analysis
SMALL_WORLD_ANALYSIS_CONFIG = {
    'rewiring_probabilities': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'analysis_runs_per_config': 1,  # Single run per config
    'simulation_steps_per_run': SIMULATION_STEPS,  # Use the 144 steps
    'commuters_for_analysis': 40
}

# Research-focused configurations for specific questions
RESEARCH_CONFIGURATIONS = {
    # Question: Do small-world shortcuts improve peripheral accessibility?
    'peripheral_access_study': {
        'topology_type': 'small_world',
        'rewiring_probabilities': [0.0, 0.05, 0.1, 0.2, 0.3],  # Include p=0 (regular) as baseline
        'focus_metric': 'peripheral_benefit_score',
        'peripheral_distance_threshold': 25,
        'analysis_description': 'Impact of small-world shortcuts on peripheral area accessibility'
    },
    
    # Question: What's the optimal balance between efficiency and equity?
    'efficiency_equity_study': {
        'topology_type': 'small_world',
        'rewiring_probabilities': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        'focus_metric': 'efficiency_equity_balance',
        'analysis_description': 'Optimal small-world parameters for transport efficiency and equity'
    },
    
    # Question: How do shortcuts affect different demographic groups?
    'demographic_impact_study': {
        'topology_type': 'small_world',
        'rewiring_probabilities': [0.0, 0.1, 0.3, 0.5],
        'demographic_focus': ['income_level', 'age', 'has_disability'],
        'analysis_description': 'Small-world impact on different demographic groups'
    }
}

# Quick configuration switcher function
def get_small_world_config(p_value=0.1, k_neighbors=4):
    """Get small-world network configuration"""
    config = NETWORK_CONFIG.copy()
    config.update({
        'topology_type': 'small_world',
        'rewiring_probability': p_value,
        'initial_neighbors': k_neighbors,
        'preserve_geography': True
    })
    return config

# def get_degree_constrained_config(degree=3):
#     """Get degree-constrained network configuration"""
#     config = NETWORK_CONFIG.copy()
#     config.update({
#         'topology_type': 'degree_constrained',
#         'degree_constraint': degree
#     })
#     return config

def get_research_config(study_name):
    """Get configuration for specific research study"""
    if study_name in RESEARCH_CONFIGURATIONS:
        base_config = NETWORK_CONFIG.copy()
        research_config = RESEARCH_CONFIGURATIONS[study_name].copy()
        base_config.update(research_config)
        return base_config
    else:
        print(f"Unknown research configuration: {study_name}")
        print(f"Available configurations: {list(RESEARCH_CONFIGURATIONS.keys())}")
        return NETWORK_CONFIG

# ===== UPDATED NETWORK-TO-RAPTOR CONVERTER =====
class NetworkToRAPTORConverter:
    def __init__(self, network_manager):
        self.network_manager = network_manager
        self.network = network_manager.active_network
        self.spatial_mapper = network_manager.spatial_mapper

    def extract_raptor_compatible_data(self):
        """Convert network topology to RAPTOR-compatible format"""
        stations = {'train': {}, 'bus': {}}
        routes = {'train': {}, 'bus': {}}
        transfers = {}
        
        # Extract stations from network nodes
        for node_id, node_data in self.network.nodes(data=True):
            grid_coord = self.spatial_mapper.node_to_grid.get(node_id)
            if not grid_coord:
                continue
                
            # Get transport modes for this node
            transport_modes = node_data.get('transport_modes', [])
            
            for mode in transport_modes:
                mode_key = mode.value if hasattr(mode, 'value') else str(mode)
                if mode_key in ['train', 'bus']:
                    stations[mode_key][node_id] = grid_coord
        
        # Extract routes from network edges
        route_segments = {}
        for u, v, edge_data in self.network.edges(data=True):
            route_id = edge_data.get('route_id', f"R_{edge_data.get('transport_mode', 'unknown')}")
            transport_mode = edge_data.get('transport_mode')
            
            if hasattr(transport_mode, 'value'):
                mode_key = transport_mode.value
            else:
                mode_key = str(transport_mode)
                
            if mode_key in ['train', 'bus']:
                if route_id not in route_segments:
                    route_segments[route_id] = {'mode': mode_key, 'segments': []}
                
                segment_order = edge_data.get('segment_order', 0)
                route_segments[route_id]['segments'].append((u, v, segment_order))
        
        # Convert segments to ordered station lists
        for route_id, route_data in route_segments.items():
            mode = route_data['mode']
            segments = sorted(route_data['segments'], key=lambda x: x[2])
            
            if segments:
                # Build station sequence from segments
                station_sequence = [segments[0][0]]  # Start with first station
                for segment in segments:
                    station_sequence.append(segment[1])
                
                routes[mode][route_id] = station_sequence
        
        # Calculate transfers between nodes with multiple transport modes
        for node_id, node_data in self.network.nodes(data=True):
            transport_modes = node_data.get('transport_modes', [])
            
            if len(transport_modes) > 1:
                # Create transfers between different modes at same location
                for i, mode1 in enumerate(transport_modes):
                    for mode2 in transport_modes[i+1:]:
                        mode1_key = mode1.value if hasattr(mode1, 'value') else str(mode1)
                        mode2_key = mode2.value if hasattr(mode2, 'value') else str(mode2)
                        
                        if mode1_key in ['train', 'bus'] and mode2_key in ['train', 'bus']:
                            transfer_time = self._calculate_transfer_time(mode1_key, mode2_key, node_data)
                            transfers[(node_id, node_id)] = transfer_time
        
        # Add inter-station transfers for nearby stations
        self._add_proximity_transfers(transfers, stations)
        
        return stations, routes, transfers
    
    def _calculate_transfer_time(self, mode1, mode2, node_data):
        """Calculate transfer time between modes at a node"""
        base_transfer_time = 0.5
        
        # Different transfer times based on mode combination
        if 'train' in [mode1, mode2] and 'bus' in [mode1, mode2]:
            return base_transfer_time + 0.3  # Train-bus transfer
        else:
            return base_transfer_time  # Same mode or other combinations
    
    def _add_proximity_transfers(self, transfers, stations):
        """Add transfers between nearby stations of different modes"""
        all_stations = {}
        for mode in ['train', 'bus']:
            for station_id, coord in stations[mode].items():
                all_stations[station_id] = (coord, mode)
        
        for station1_id, (coord1, mode1) in all_stations.items():
            for station2_id, (coord2, mode2) in all_stations.items():
                if station1_id != station2_id and mode1 != mode2:
                    # Calculate distance
                    distance = ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
                    
                    # If close enough, add transfer
                    if distance <= 3:
                        transfer_time = 0.5 + distance * 0.2
                        transfers[(station1_id, station2_id)] = transfer_time
                        transfers[(station2_id, station1_id)] = transfer_time

# Update the initialize_network_data function
def initialize_network_data(network_manager):
    """
    Initialize RAPTOR-compatible data structures from network topology
    """
    global stations, routes, transfers
    
    converter = NetworkToRAPTORConverter(network_manager)
    stations, routes, transfers = converter.extract_raptor_compatible_data()
    
    # Get topology type for logging
    topology_type = "unknown"
    if hasattr(network_manager, 'active_network'):
        # Check for small-world characteristics
        shortcut_edges = sum(1 for u, v, d in network_manager.active_network.edges(data=True) 
                           if d.get('edge_type') == 'shortcut')
        if shortcut_edges > 0:
            topology_type = "small_world"
        # else:
        #     topology_type = "degree_constrained"
    
    print(f"Initialized {topology_type} network: {len(stations.get('train', {}))} train stations, "
          f"{len(stations.get('bus', {}))} bus stops, {len(transfers)} transfers")

# ===== SMALL-WORLD SPECIFIC UTILITIES =====

def analyze_small_world_properties(graph):
    """Quick analysis of small-world properties"""
    if not graph or graph.number_of_nodes() == 0:
        return {}
    
    try:
        # Basic metrics
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        
        if not nx.is_connected(graph):
            return {'error': 'Graph is not connected'}
        
        clustering = nx.average_clustering(graph)
        path_length = nx.average_shortest_path_length(graph)
        
        # Small-world coefficient
        # Create random graph for comparison
        p_random = 2 * m / (n * (n - 1)) if n > 1 else 0
        random_graph = nx.erdos_renyi_graph(n, p_random)
        
        if nx.is_connected(random_graph):
            random_clustering = nx.average_clustering(random_graph)
            random_path_length = nx.average_shortest_path_length(random_graph)
            
            gamma = clustering / random_clustering if random_clustering > 0 else 0
            lambda_val = path_length / random_path_length if random_path_length > 0 else 0
            sigma = gamma / lambda_val if lambda_val > 0 else 0
        else:
            gamma = lambda_val = sigma = 0
        
        # Count shortcuts
        shortcuts = sum(1 for u, v, d in graph.edges(data=True) 
                       if d.get('edge_type') == 'shortcut')
        
        return {
            'nodes': n,
            'edges': m,
            'clustering_coefficient': clustering,
            'avg_path_length': path_length,
            'small_world_sigma': sigma,
            'gamma': gamma,
            'lambda': lambda_val,
            'shortcuts': shortcuts,
            'shortcut_percentage': shortcuts/m*100 if m > 0 else 0
        }
        
    except Exception as e:
        return {'error': str(e)}

def print_network_analysis(network_manager):
    """Print analysis of current network"""
    if hasattr(network_manager, 'active_network'):
        analysis = analyze_small_world_properties(network_manager.active_network)
        
        if 'error' not in analysis:
            print(f"\n📊 NETWORK ANALYSIS:")
            print(f"   Nodes: {analysis['nodes']}, Edges: {analysis['edges']}")
            print(f"   Clustering: {analysis['clustering_coefficient']:.3f}")
            print(f"   Path Length: {analysis['avg_path_length']:.2f}")
            print(f"   Small-World σ: {analysis['small_world_sigma']:.3f}")
            print(f"   Shortcuts: {analysis['shortcuts']} ({analysis['shortcut_percentage']:.1f}%)")
        else:
            print(f"Network analysis error: {analysis['error']}")

# Usage examples in comments:
"""
# Switch to small-world network
NETWORK_CONFIG['topology_type'] = 'small_world'
NETWORK_CONFIG['rewiring_probability'] = 0.1

# Use research configuration
config = get_research_config('peripheral_access_study')

# Analyze specific small-world parameters
sw_config = get_small_world_config(p_value=0.2, k_neighbors=4)
"""