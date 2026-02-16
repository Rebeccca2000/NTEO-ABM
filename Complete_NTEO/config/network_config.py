# network_config.py
"""
Unified Network Configuration System for NTEO
Implements single control parameter system as per the construction guide.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import warnings

# ===== MASTER CONTROL PARAMETER =====
# Single line controls everything - change this to switch topology types
NETWORK_TYPE = "degree_constrained"  # Options: "degree_constrained", "small_world", "scale_free", "base_sydney"

# ===== LEVEL 1: COMMON PARAMETERS (NEVER CHANGE) =====
# FIXED: Removed preserve_geography from common parameters - it's topology-specific
COMMON_PARAMETERS = {
    # Grid dimensions
    'grid_width': 100,
    'grid_height': 80,
    
    # System settings
    'sydney_realism': True,
    'preserve_hierarchy': True,
    # REMOVED: 'preserve_geography': True,  # This was causing the error!
    
    # Simulation consistency
    'base_seed': 42,  # For reproducible network generation
    'minimum_connectivity': True,  # Ensure all networks are connected
    
    # Performance settings
    'enable_caching': True,
    'max_cache_size': 1000,
}

# ===== LEVEL 2: TOPOLOGY-SPECIFIC PARAMETERS =====
TOPOLOGY_PARAMETERS = {
    'degree_constrained': {
        'degree_values': [3, 4, 5, 6, 7],  # Available degree constraints
        'default_degree': 4,
        'variation_parameter_name': 'degree_constraint',
        'description': 'Fixed connectivity networks with specified node degree',
        'topology_params': ['preserve_geography', 'preserve_major_hubs']  # Parameters for topology generation only
    },
    
    'small_world': {
        'rewiring_probabilities': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        'default_rewiring_probability': 0.1,
        'initial_neighbors': 4,
        'variation_parameter_name': 'rewiring_probability',
        'description': 'Watts-Strogatz small-world networks with shortcuts',
        'topology_params': ['initial_neighbors', 'preserve_geography']  # Parameters for topology generation only
    },
    
    'scale_free': {
        'attachment_parameters': [1, 2, 3, 4, 5],
        'default_attachment_parameter': 2,
        'alpha_range': [0.5, 1.0, 1.5, 2.0],
        'default_alpha': 1.0,
        'variation_parameter_name': 'attachment_parameter',
        'description': 'Barabási-Albert preferential attachment networks',
        'topology_params': ['alpha', 'preserve_geography', 'hub_preference']  # Parameters for topology generation only
    },
    
    'base_sydney': {
        'connectivity_levels': [4, 6, 8],
        'default_connectivity_level': 6,
        'variation_parameter_name': 'connectivity_level',
        'description': 'Base Sydney network with varying connectivity',
        'topology_params': ['grid_width', 'grid_height']  # Parameters for topology generation only
    },

    'grid': {
        'connectivity_levels': [4, 6, 8],
        'default_connectivity_level': 4,
        'variation_parameter_name': 'connectivity_level',
        'description': 'Regular mathematical grid network (baseline)',
        'topology_params': ['grid_width', 'grid_height']
    }
}

# ===== RESEARCH STUDY CONFIGURATIONS =====
RESEARCH_CONFIGURATIONS = {
    'degree_comparison_study': {
        'topology_type': 'degree_constrained',
        'parameter_range': [3, 4, 5, 6, 7],
        'focus_metric': 'network_efficiency',
        'analysis_description': 'Comparative analysis of degree-constrained networks'
    },
    
    'small_world_efficiency_study': {
        'topology_type': 'small_world',
        'parameter_range': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        'focus_metric': 'small_world_coefficient',
        'analysis_description': 'Small-world effect on transport network efficiency'
    },
    
    'scale_free_hub_study': {
        'topology_type': 'scale_free',
        'parameter_range': [1, 2, 3, 4],
        'focus_metric': 'hub_centrality_impact',
        'analysis_description': 'Impact of hub formation on network performance'
    },
    
    'equity_efficiency_balance_study': {
        'topology_type': 'small_world',
        'parameter_range': [0.05, 0.1, 0.15, 0.2],
        'focus_metric': 'efficiency_equity_balance',
        'analysis_description': 'Optimal balance between transport efficiency and equity'
    }
}

@dataclass
class NetworkConfiguration:
    """Complete network configuration for a specific topology and variation"""
    topology_type: str
    variation_parameter: float
    grid_width: int = 100
    grid_height: int = 80
    sydney_realism: bool = True
    preserve_hierarchy: bool = True
    
    def __post_init__(self):
        """Validate configuration on creation"""
        if self.topology_type not in TOPOLOGY_PARAMETERS:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # Add topology-specific parameters
        topology_params = TOPOLOGY_PARAMETERS[self.topology_type]
        param_name = topology_params['variation_parameter_name']
        setattr(self, param_name, self.variation_parameter)

class NetworkConfigurationManager:
    """Manages network configurations and provides standardized interfaces"""
    
    def __init__(self, master_topology_type: str = None):
        self.master_topology_type = master_topology_type or NETWORK_TYPE
        self._validate_topology_type()
    
    def _validate_topology_type(self):
        """Validate that the master topology type is supported"""
        if self.master_topology_type not in TOPOLOGY_PARAMETERS:
            available_types = list(TOPOLOGY_PARAMETERS.keys())
            raise ValueError(f"Unsupported topology type: {self.master_topology_type}. "
                           f"Available types: {available_types}")
    
    def get_base_configuration(self) -> Dict[str, Any]:
        """Get base configuration for the current topology type"""
        config = COMMON_PARAMETERS.copy()
        config['topology_type'] = self.master_topology_type
        
        # Add topology-specific defaults
        topology_params = TOPOLOGY_PARAMETERS[self.master_topology_type]
        
        if self.master_topology_type == 'degree_constrained':
            config['degree_constraint'] = topology_params['default_degree']
        elif self.master_topology_type == 'small_world':
            config['rewiring_probability'] = topology_params['default_rewiring_probability']
            config['initial_neighbors'] = topology_params['initial_neighbors']
        elif self.master_topology_type == 'scale_free':
            config['attachment_parameter'] = topology_params['default_attachment_parameter']
            config['alpha'] = topology_params.get('default_alpha', 1.0)
        elif self.master_topology_type == 'base_sydney':
            config['connectivity_level'] = topology_params.get('default_connectivity_level', 6)
        elif self.master_topology_type == 'grid':  # ADD THIS
            config['connectivity_level'] = topology_params.get('default_connectivity_level', 4)
        
        return config
    
    def get_parameter_range(self, topology_type: str = None) -> List[float]:
        """Get available parameter values for specified topology"""
        topo_type = topology_type or self.master_topology_type
        topology_params = TOPOLOGY_PARAMETERS[topo_type]
        
        if topo_type == 'degree_constrained':
            return topology_params['degree_values']
        elif topo_type == 'small_world':
            return topology_params['rewiring_probabilities']
        elif topo_type == 'scale_free':
            return topology_params['attachment_parameters']
        elif topo_type == 'base_sydney':
            return topology_params['connectivity_levels']
        elif topo_type == 'grid':  # ADD THIS
            return topology_params['connectivity_levels']
        else:
            return [1, 2, 3]  # Default fallback
    
    def switch_topology_type(self, new_topology_type: str):
        """Switch to a different topology type"""
        old_type = self.master_topology_type
        self.master_topology_type = new_topology_type
        
        try:
            self._validate_topology_type()
            print(f"✅ Switched from {old_type} to {new_topology_type}")
        except ValueError as e:
            # Revert if invalid
            self.master_topology_type = old_type
            raise e
    
    def get_variation_configuration(self, variation_parameter: float) -> NetworkConfiguration:
        """Get configuration for a specific parameter variation"""
        config = NetworkConfiguration(
            topology_type=self.master_topology_type,
            variation_parameter=variation_parameter,
            **COMMON_PARAMETERS
        )
        return config
    
    def get_research_study_config(self, study_name: str) -> Dict[str, Any]:
        """Get configuration for a specific research study"""
        if study_name not in RESEARCH_CONFIGURATIONS:
            available_studies = list(RESEARCH_CONFIGURATIONS.keys())
            raise ValueError(f"Unknown research study: {study_name}. "
                           f"Available studies: {available_studies}")
        
        study_config = RESEARCH_CONFIGURATIONS[study_name].copy()
        
        # Override master topology type if study specifies one
        if 'topology_type' in study_config:
            original_type = self.master_topology_type
            self.master_topology_type = study_config['topology_type']
            base_config = self.get_base_configuration()
            self.master_topology_type = original_type  # Restore
            study_config.update(base_config)
        else:
            study_config.update(self.get_base_configuration())
        
        return study_config
    
    def get_topology_description(self) -> str:
        """Get description of current topology type"""
        return TOPOLOGY_PARAMETERS[self.master_topology_type]['description']
    
    def print_current_configuration(self):
        """Print current configuration summary"""
        config = self.get_base_configuration()
        param_range = self.get_parameter_range()
        
        print(f"\n{'='*60}")
        print(f"NETWORK CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        print(f"Topology Type: {self.master_topology_type}")
        print(f"Description: {self.get_topology_description()}")
        print(f"Grid Size: {config['grid_width']} × {config['grid_height']}")
        print(f"Parameter Range: {param_range}")
        print(f"Sydney Realism: {config['sydney_realism']}")
        print(f"Preserve Hierarchy: {config['preserve_hierarchy']}")

# ===== UTILITY FUNCTIONS =====
def get_current_config() -> Dict[str, Any]:
    """Get current network configuration"""
    manager = NetworkConfigurationManager()
    return manager.get_base_configuration()

def get_config_for_topology(topology_type: str, variation_parameter: float = None) -> Dict[str, Any]:
    """
    FIXED: Get configuration for specific topology and parameter.
    Separates topology-generation parameters from model parameters.
    """
    manager = NetworkConfigurationManager(topology_type)
    config = manager.get_base_configuration()
    
    # Set the variation parameter
    topology_params = TOPOLOGY_PARAMETERS[topology_type]
    param_name = topology_params['variation_parameter_name']
    if variation_parameter is not None:
        config[param_name] = variation_parameter
    
    # FIXED: Don't add topology-specific parameters to model config
    # These will be handled separately in network generation
    return config

def get_research_config(study_name: str) -> Dict[str, Any]:
    """Get configuration for a predefined research study"""
    if study_name not in RESEARCH_CONFIGURATIONS:
        available_studies = list(RESEARCH_CONFIGURATIONS.keys())
        raise ValueError(f"Unknown research study: {study_name}. "
                       f"Available studies: {available_studies}")
    
    return RESEARCH_CONFIGURATIONS[study_name].copy()

def switch_network_type(new_type: str):
    """Switch the global network type"""
    global NETWORK_TYPE
    if new_type not in TOPOLOGY_PARAMETERS:
        available_types = list(TOPOLOGY_PARAMETERS.keys())
        raise ValueError(f"Invalid network type: {new_type}. "
                       f"Available types: {available_types}")
    
    NETWORK_TYPE = new_type
    print(f"✅ Global network type switched to: {new_type}")

def get_topology_parameters_for_generation(topology_type: str) -> List[str]:
    """
    FIXED: Get list of parameters that should only be used for topology generation,
    not passed to the ABM model.
    """
    if topology_type in TOPOLOGY_PARAMETERS:
        return TOPOLOGY_PARAMETERS[topology_type].get('topology_params', [])
    return []

def separate_topology_and_model_params(config: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    FIXED: Separate configuration into topology-generation parameters and model parameters.
    
    Returns:
        (topology_params, model_params) - Two separate dictionaries
    """
    topology_type = config.get('topology_type', 'degree_constrained')
    topology_param_names = get_topology_parameters_for_generation(topology_type)
    
    topology_params = {}
    model_params = {}
    
    for key, value in config.items():
        if key in topology_param_names:
            topology_params[key] = value
        else:
            model_params[key] = value
    
    return topology_params, model_params

# ===== BACKWARDS COMPATIBILITY =====
def get_small_world_config(p_value=0.1, k_neighbors=4):
    """Backward compatibility function for small-world configuration"""
    warnings.warn("get_small_world_config is deprecated. Use get_config_for_topology instead.", 
                  DeprecationWarning, stacklevel=2)
    return get_config_for_topology('small_world', p_value)

def get_degree_constrained_config(degree=3):
    """Backward compatibility function for degree-constrained configuration"""
    warnings.warn("get_degree_constrained_config is deprecated. Use get_config_for_topology instead.", 
                  DeprecationWarning, stacklevel=2)
    return get_config_for_topology('degree_constrained', degree)

# ===== VALIDATION FUNCTIONS =====
def validate_configuration(config: Dict[str, Any]) -> bool:
    """Validate a network configuration"""
    required_fields = ['topology_type', 'grid_width', 'grid_height']
    
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field: {field}")
            return False
    
    topology_type = config['topology_type']
    if topology_type not in TOPOLOGY_PARAMETERS:
        print(f"❌ Invalid topology type: {topology_type}")
        return False
    
    # Validate topology-specific parameters
    topology_params = TOPOLOGY_PARAMETERS[topology_type]
    param_name = topology_params['variation_parameter_name']
    
    if param_name not in config:
        print(f"❌ Missing topology parameter: {param_name}")
        return False
    
    print("✅ Configuration validation passed")
    return True

def print_configuration_summary(config: Dict[str, Any] = None):
    """Print summary of current or provided configuration"""
    if config is None:
        config = get_current_config()
    
    print("\n" + "="*60)
    print("NETWORK CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Topology Type: {config.get('topology_type', 'unknown')}")
    print(f"Grid Size: {config.get('grid_width', 0)}x{config.get('grid_height', 0)}")
    print(f"Sydney Realism: {config.get('sydney_realism', False)}")
    
    # Print topology-specific parameters
    topology_type = config.get('topology_type')
    if topology_type in TOPOLOGY_PARAMETERS:
        topology_params = TOPOLOGY_PARAMETERS[topology_type]
        param_name = topology_params['variation_parameter_name']
        param_value = config.get(param_name, 'not set')
        print(f"Main Parameter ({param_name}): {param_value}")
        print(f"Description: {topology_params['description']}")
    
    print("="*60)

# ===== MODULE INITIALIZATION =====
if __name__ == "__main__":
    print("NTEO Network Configuration System")
    print_configuration_summary()
    
    print("\nTesting configuration creation...")
    test_config = get_config_for_topology('degree_constrained', 5)
    validate_configuration(test_config)
    
    print("\nTesting parameter separation...")
    topo_params, model_params = separate_topology_and_model_params(test_config)
    print(f"Topology params: {list(topo_params.keys())}")
    print(f"Model params: {list(model_params.keys())}")
    
    print("\n✅ Network configuration system ready!")
    
# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    # Example 1: Basic usage
    print("Example 1: Basic Configuration")
    manager = NetworkConfigurationManager()
    manager.print_current_configuration()
    
    # Example 2: Switch topology types
    print("\nExample 2: Switching Topology Types")
    manager.switch_topology_type('small_world')
    config = manager.get_base_configuration()
    print(f"Small-world config: {config}")
    
    # Example 3: Get research study configuration
    print("\nExample 3: Research Study Configuration")
    research_config = get_research_config('degree_comparison_study')
    print(f"Research config: {research_config}")
    
    # Example 4: Parameter variations
    print("\nExample 4: Parameter Variations")
    manager.switch_topology_type('degree_constrained')
    for degree in manager.get_parameter_range():
        var_config = manager.get_variation_configuration(degree)
        print(f"Degree-{degree} config: {var_config.degree_constraint}")