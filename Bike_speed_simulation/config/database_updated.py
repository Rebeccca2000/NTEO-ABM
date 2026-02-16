# database_updated.py
"""
Updated Database Configuration - Network Configuration Removed
Network configuration is now handled by network_config.py and network_factory.py
"""

from ABM.agent_subsidy_pool import SubsidyPoolConfig
import networkx as nx

# ===== DATABASE CONNECTION =====
# PostgreSQL connection for NTEO ABM
DB_CONNECTION_STRING = "postgresql://localhost:5432/nteo_abm"

# ===== SIMULATION PARAMETERS =====
SIMULATION_STEPS = 144  # One day simulation (144 steps = 24 hours)

# ===== COMMUTER CONFIGURATION =====
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

# ===== MODE CHOICE AND UTILITY PARAMETERS =====
PENALTY_COEFFICIENTS = {
    'disability_bike_walk': 0.8,
    'age_health_bike_walk': 0.3,
    'no_tech_access_car_bike': 0.1
}

AFFORDABILITY_THRESHOLDS = {
    'low': 30, 'middle': 50, 'high': 80
}

VALUE_OF_TIME = {
    'low': 10, 'middle': 25, 'high': 50
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
    'walk': 0,
    'bike': 0,
    'car': 0,
    'public': 0,
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
    'bus': {'on_peak': 2.5, 'off_peak': 2.2},
    'metro': {'on_peak': 3.2, 'off_peak': 2.8}
}

# ===== ALPHA VALUES (MODE CHOICE SENSITIVITY) =====
ALPHA_VALUES = {
    'low_income': 0.15,
    'middle_income': 0.10,
    'high_income': 0.05
}


# UberLike and BikeShare Service Configurations
UberLike1_capacity = 50
UberLike1_price = 2.0
UberLike2_capacity = 50  
UberLike2_price = 2.5
BikeShare1_capacity = 30
BikeShare1_price = 0.5
BikeShare2_capacity = 30
BikeShare2_price = 0.7

# ===== GOVERNMENT SUBSIDY =====
subsidy_dataset = {
    'low': {'bike': 0.50, 'car': 0.55, 'MaaS_Bundle': 0.5},
    'middle': {'bike': 0.23, 'car': 0.35, 'MaaS_Bundle': 0.35},
    'high': {'bike': 0.1, 'car': 0.05, 'MaaS_Bundle': 0.1},
}

daily_config = SubsidyPoolConfig('daily', 0)
weekly_config = SubsidyPoolConfig('weekly', 22000)
monthly_config = SubsidyPoolConfig('monthly', 80000)


# ===== NETWORK CONFIG COMPATIBILITY =====
# Import current network configuration
try:
    from config.network_config import get_current_config
    NETWORK_CONFIG = get_current_config()
except ImportError:
    # Fallback if network_config not available
    NETWORK_CONFIG = {
        'topology_type': 'degree_constrained',
        'grid_width': 100,
        'grid_height': 80,
        'degree_constraint': 4,
        'sydney_realism': True
    }

print(f"📊 Added compatibility variables for agent_run_visualisation.py")
print(f"🚗 UberLike services: {UberLike1_capacity}/{UberLike2_capacity} capacity")
print(f"🚲 BikeShare services: {BikeShare1_capacity}/{BikeShare2_capacity} capacity")
print(f"🌐 Network config: {NETWORK_CONFIG.get('topology_type', 'unknown')}")

# ===== NETWORK CONFIGURATION IMPORT =====
# Import network configuration from separate module
try:
    from config.network_config import get_current_config
    
    def get_network_config():
        """Get current network configuration from network_config module"""
        return get_current_config()
    
    # For backward compatibility, create a NETWORK_CONFIG variable
    # This allows existing code to still access network config
    NETWORK_CONFIG = get_current_config()
    
    print("✅ Network configuration imported from network_config.py")
    print(f"   Current topology: {NETWORK_CONFIG.get('topology_type', 'unknown')}")
    
except ImportError as e:
    print(f"⚠️  Warning: Could not import network configuration: {e}")
    print("   Using fallback network configuration")
    
    # Fallback configuration for testing/development
    NETWORK_CONFIG = {
        'topology_type': 'degree_constrained',
        'degree_constraint': 4,
        'grid_width': 100,
        'grid_height': 80,
        'sydney_realism': True,
        'preserve_hierarchy': True
    }

# ===== BACKWARD COMPATIBILITY FUNCTIONS =====
def get_small_world_config(p_value=0.1, k_neighbors=4):
    """
    Backward compatibility function for small-world configuration.
    Now redirects to network_config module.
    """
    try:
        from config.network_config import get_config_for_topology
        return get_config_for_topology('small_world', p_value)
    except ImportError:
        # Fallback
        config = NETWORK_CONFIG.copy()
        config.update({
            'topology_type': 'small_world',
            'rewiring_probability': p_value,
            'initial_neighbors': k_neighbors,
            'preserve_geography': True
        })
        return config

def get_degree_constrained_config(degree=3):
    """
    Backward compatibility function for degree-constrained configuration.
    Now redirects to network_config module.
    """
    try:
        from config.network_config import get_config_for_topology
        return get_config_for_topology('degree_constrained', degree)
    except ImportError:
        # Fallback
        config = NETWORK_CONFIG.copy()
        config.update({
            'topology_type': 'degree_constrained',
            'degree_constraint': degree
        })
        return config

def get_research_config(study_name):
    """
    Backward compatibility function for research configurations.
    Now redirects to network_config module.
    """
    try:
        from config.network_config import get_research_config as get_research_config_new
        return get_research_config_new(study_name)
    except ImportError:
        print(f"⚠️  Warning: Could not get research config for {study_name}")
        return NETWORK_CONFIG

# ===== LEGACY RESEARCH CONFIGURATIONS (DEPRECATED) =====
# These are kept for backward compatibility but are now handled by network_config.py
LEGACY_RESEARCH_CONFIGURATIONS = {
    'small_world_clustering_study': {
        'topology_type': 'small_world',
        'rewiring_probabilities': [0.0, 0.05, 0.1, 0.2, 0.3],
        'focus_metric': 'clustering_coefficient_impact',
        'analysis_description': 'Impact of clustering on transport network efficiency'
    },
    
    'peripheral_access_study': {
        'topology_type': 'small_world',
        'rewiring_probabilities': [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
        'focus_metric': 'peripheral_benefit_score',
        'peripheral_distance_threshold': 25,
        'analysis_description': 'Impact of small-world shortcuts on peripheral area accessibility'
    },
    
    'efficiency_equity_study': {
        'topology_type': 'small_world',
        'rewiring_probabilities': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        'focus_metric': 'efficiency_equity_balance',
        'analysis_description': 'Optimal small-world parameters for transport efficiency and equity'
    },
    
    'demographic_impact_study': {
        'topology_type': 'small_world',
        'rewiring_probabilities': [0.0, 0.1, 0.3, 0.5],
        'demographic_focus': ['income_level', 'age', 'has_disability'],
        'analysis_description': 'Small-world impact on different demographic groups'
    }
}

# ===== NETWORK-TO-RAPTOR CONVERTER (UPDATED) =====
class NetworkToRAPTORConverter:
    """
    Convert network topology to RAPTOR-compatible format.
    Updated to work with the new network configuration system.
    """
    
    def __init__(self, network_manager=None):
        """
        Initialize converter with network manager from network factory.
        
        Args:
            network_manager: Network manager from network_factory.create_network()
        """
        if network_manager is None:
            # Create default network manager using network factory
            try:
                from network_factory import create_network
                network_interface = create_network()
                self.network_manager = network_interface.network_manager
            except ImportError:
                raise ImportError("Cannot create network manager - network_factory not available")
        else:
            self.network_manager = network_manager
        
        self.network = self.network_manager.active_network
        self.spatial_mapper = getattr(self.network_manager, 'spatial_mapper', None)

    def extract_raptor_compatible_data(self):
        """Convert network topology to RAPTOR-compatible format"""
        stations = {'train': {}, 'bus': {}}
        routes = {'train': {}, 'bus': {}}
        transfers = {}
        
        # Extract stations from network nodes
        for node_id, node_data in self.network.nodes(data=True):
            # Get spatial coordinates if available
            if self.spatial_mapper:
                try:
                    grid_coord = self.spatial_mapper.get_grid_coordinates(node_id)
                except:
                    grid_coord = (0, 0)  # Fallback
            else:
                grid_coord = (0, 0)
            
            # Assign to transport modes based on node properties
            node_type = node_data.get('node_type', 'mixed')
            
            if node_type in ['train', 'mixed']:
                stations['train'][f'train_{node_id}'] = grid_coord
            if node_type in ['bus', 'mixed']:
                stations['bus'][f'bus_{node_id}'] = grid_coord
        
        # Extract routes from network edges
        train_route_id = 0
        bus_route_id = 0
        
        for u, v, edge_data in self.network.edges(data=True):
            edge_type = edge_data.get('edge_type', 'mixed')
            
            if edge_type in ['train', 'mixed']:
                route_id = f'train_route_{train_route_id}'
                routes['train'][route_id] = {
                    'stations': [f'train_{u}', f'train_{v}'],
                    'travel_times': [edge_data.get('weight', 1.0)]
                }
                train_route_id += 1
            
            if edge_type in ['bus', 'mixed']:
                route_id = f'bus_route_{bus_route_id}'
                routes['bus'][route_id] = {
                    'stations': [f'bus_{u}', f'bus_{v}'],
                    'travel_times': [edge_data.get('weight', 1.0)]
                }
                bus_route_id += 1
        
        # Add transfers between co-located stations
        self._add_transfers(transfers, stations)
        
        return stations, routes, transfers
    
    def _add_transfers(self, transfers, stations):
        """Add transfers between co-located stations of different modes"""
        base_transfer_time = 0.5
        
        # Add transfers between train and bus stations at same locations
        for train_station, train_coord in stations['train'].items():
            for bus_station, bus_coord in stations['bus'].items():
                # If coordinates are the same or very close
                distance = ((train_coord[0] - bus_coord[0])**2 + 
                           (train_coord[1] - bus_coord[1])**2)**0.5
                
                if distance <= 2:  # Close enough for transfer
                    transfer_time = base_transfer_time + distance * 0.1
                    transfers[(train_station, bus_station)] = transfer_time
                    transfers[(bus_station, train_station)] = transfer_time

# ===== UTILITY FUNCTIONS =====
def update_network_configuration(new_topology_type: str, **kwargs):
    """
    Update the global network configuration.
    This function provides compatibility for existing code.
    """
    global NETWORK_CONFIG
    
    try:
        from config.network_config import switch_network_type, get_current_config
        switch_network_type(new_topology_type)
        NETWORK_CONFIG = get_current_config()
        
        # Update with any additional parameters
        NETWORK_CONFIG.update(kwargs)
        
        print(f"✅ Network configuration updated to {new_topology_type}")
        return True
        
    except ImportError:
        print("⚠️  Warning: Could not update network configuration - using fallback")
        NETWORK_CONFIG['topology_type'] = new_topology_type
        NETWORK_CONFIG.update(kwargs)
        return False

def get_current_network_summary():
    """Get summary of current network configuration"""
    return {
        'topology_type': NETWORK_CONFIG.get('topology_type', 'unknown'),
        'grid_size': f"{NETWORK_CONFIG.get('grid_width', 0)}x{NETWORK_CONFIG.get('grid_height', 0)}",
        'main_parameter': NETWORK_CONFIG.get('degree_constraint') or 
                         NETWORK_CONFIG.get('rewiring_probability') or 
                         NETWORK_CONFIG.get('attachment_parameter') or 
                         'unknown',
        'sydney_realism': NETWORK_CONFIG.get('sydney_realism', False)
    }

# ===== INITIALIZATION MESSAGE =====
def print_database_status():
    """Print current database and network configuration status"""
    print("\n" + "="*60)
    print("DATABASE CONFIGURATION STATUS")
    print("="*60)
    print(f"Database: {DB_CONNECTION_STRING}")
    print(f"Simulation Steps: {SIMULATION_STEPS}")
    print(f"Commuters: {num_commuters}")
    
    network_summary = get_current_network_summary()
    print(f"\nNetwork Configuration:")
    print(f"  - Topology: {network_summary['topology_type']}")
    print(f"  - Grid Size: {network_summary['grid_size']}")
    print(f"  - Main Parameter: {network_summary['main_parameter']}")
    print(f"  - Sydney Realism: {network_summary['sydney_realism']}")
    
    print("\n✅ Database configuration loaded")
    print("📁 Network configuration separated to network_config.py")
    print("🏭 Network creation handled by network_factory.py")

# Print status when module is imported
if __name__ == "__main__":
    print_database_status()
else:
    # Print brief status when imported
    network_type = NETWORK_CONFIG.get('topology_type', 'unknown')
    print(f"📊 Database config loaded | Network: {network_type}")