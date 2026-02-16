# network_factory.py
"""
Network Factory Pattern for NTEO - FIXED VERSION
Properly filters topology parameters from model parameters.
"""

import networkx as nx
import numpy as np
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import warnings

# FIXED: Use unified topology generator only
from topology.unified_topology_generator import UnifiedTopologyGenerator

# Import your existing network components
try:
    from topology.unified_network_integration import TwoLayerNetworkManager, UnifiedSpatialMapper, NetworkCongestionModel, UnifiedNetworkRouter
    from topology.network_topology import SydneyNetworkTopology
except ImportError as e:
    warnings.warn(f"Some network modules not available: {e}")

try:
    from config.network_config import NetworkConfigurationManager, get_current_config, TOPOLOGY_PARAMETERS
except ImportError as e:
    warnings.warn(f"Network config not available: {e}")
    # Fallback configuration manager
    class NetworkConfigurationManager:
        def __init__(self):
            self.master_topology_type = "degree_constrained"
        
        def get_base_configuration(self):
            return {
                'grid_width': 100,
                'grid_height': 80,
                'sydney_realism': True,
                'preserve_hierarchy': True,
                'preserve_geography': True
            }
    TOPOLOGY_PARAMETERS = {}

class NetworkFactory:
    """
    Single factory that creates appropriate network manager based on topology type.
    Uses ONLY UnifiedTopologyGenerator - no external topology files needed.
    """
    
    def __init__(self):
        try:
            self.config_manager = NetworkConfigurationManager()
        except:
            self.config_manager = NetworkConfigurationManager()  # Use fallback
        
        self._base_network_cache = {}
        self.unified_generator = None
    
    def create_network(self, 
                      network_type: str = None, 
                      variation_parameter: Union[float, int] = None,
                      **kwargs) -> 'TwoLayerNetworkManager':
        """
        Single function that creates any network type using unified generator.
        
        Args:
            network_type: Type of topology ('degree_constrained', 'small_world', 'scale_free', 'base_sydney')
            variation_parameter: Main control parameter for topology
            **kwargs: Additional parameters
        
        Returns:
            Configured TwoLayerNetworkManager ready for ABM
        """
        
        # Use current configuration if not specified
        if network_type is None:
            network_type = getattr(self.config_manager, 'master_topology_type', 'degree_constrained')
        
        # Get base configuration
        try:
            config = self.config_manager.get_base_configuration()
        except:
            # Fallback configuration
            config = {
                'grid_width': 100,
                'grid_height': 80,
                'sydney_realism': True,
                'preserve_hierarchy': True,
                'preserve_geography': True
            }
        
        # Override with any provided kwargs
        config.update(kwargs)
        
        # CRITICAL FIX: Remove all topology-specific parameters before passing to manager
        if network_type in TOPOLOGY_PARAMETERS:
            topology_params_to_remove = TOPOLOGY_PARAMETERS[network_type].get('topology_params', [])
            # Remove these from config before creating manager
            for param in topology_params_to_remove:
                config.pop(param, None)
        
        # Set variation parameter if provided
        if variation_parameter is not None:
            config = self._set_variation_parameter(config, network_type, variation_parameter)
        
        # Initialize unified generator if needed
        if self.unified_generator is None:
            base_network = self._get_base_sydney_network()
            self.unified_generator = UnifiedTopologyGenerator(base_network)
        
        print(f"🏗️ Creating {network_type} network (param: {variation_parameter})")
        
        # Route to appropriate creation method
        if network_type == 'degree_constrained':
            config['topology_type'] = 'degree_constrained'
            return self._create_degree_constrained_network(config)
        elif network_type == 'small_world':
            config['topology_type'] = 'small_world'
            return self._create_small_world_network(config)
        elif network_type == 'scale_free':
            config['topology_type'] = 'scale_free'
            return self._create_scale_free_network(config)
        elif network_type == 'base_sydney':
            config['topology_type'] = 'base_sydney'
            return self._create_base_sydney_network(config)
        elif network_type == 'grid':
            connectivity_level = config.get('connectivity_level', 4)
            config['topology_type'] = 'grid'  
            print(f"[DEBUG] Creating grid network with connectivity={connectivity_level}")
            try:
                graph = self.unified_generator.generate_topology('grid', connectivity_level)
                print(f"[DEBUG] graph successfully generated")
                network_manager = self._create_standard_manager(graph, config)
                print(f"✅ Grid network created (connectivity={connectivity_level})")
                return network_manager
            except Exception as e:
                print(f"❌ Failed to create grid network: {e}")
                raise
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
    
    def _set_variation_parameter(self, config: Dict[str, Any], 
                               network_type: str, 
                               variation_parameter: Union[float, int]) -> Dict[str, Any]:
        """Set the appropriate parameter for each topology type"""
        
        if network_type == 'degree_constrained':
            config['degree_constraint'] = int(variation_parameter)
        elif network_type == 'small_world':
            config['rewiring_probability'] = float(variation_parameter)
        elif network_type == 'scale_free':
            config['attachment_parameter'] = int(variation_parameter)
        elif network_type == 'base_sydney':
            config['connectivity_level'] = int(variation_parameter)
        elif network_type == 'grid':  # ADD THIS
            config['connectivity_level'] = int(variation_parameter)
        
        return config
    
    def _create_degree_constrained_network(self, config: Dict[str, Any]) -> 'TwoLayerNetworkManager':
        """Create degree-constrained network using unified generator"""
        target_degree = config.get('degree_constraint', 4)
        
        # Extract topology-specific parameters from original config
        original_config = self.config_manager.get_base_configuration() if hasattr(self.config_manager, 'get_base_configuration') else {}
        topology_params = {
            'preserve_geography': original_config.get('preserve_geography', False),
            'preserve_major_hubs': original_config.get('preserve_major_hubs', True)
        }
        
        try:
            # Generate using unified generator with topology parameters
            graph = self.unified_generator.generate_topology(
                'degree_constrained', 
                target_degree, 
                **topology_params
            )
            # 🎯 CRITICAL: Verify route IDs are preserved
            self._verify_route_ids(graph, "degree_constrained")
            
            # Create network manager with filtered config
            network_manager = self._create_standard_manager(graph, config)
            
            print(f"✅ Degree-constrained network created (degree={target_degree})")
            print(f"   - Nodes: {network_manager.active_network.number_of_nodes()}")
            print(f"   - Edges: {network_manager.active_network.number_of_edges()}")
            
            return network_manager
            
        except Exception as e:
            print(f"❌ Failed to create degree-constrained network: {e}")
            raise
    
    def _create_small_world_network(self, config: Dict[str, Any]) -> 'TwoLayerNetworkManager':
        """Create small-world network using unified generator"""
        rewiring_prob = config.get('rewiring_probability', 0.1)
        initial_neighbors = config.get('initial_neighbors', 4)
        
        # Extract topology-specific parameters from original config
        original_config = self.config_manager.get_base_configuration() if hasattr(self.config_manager, 'get_base_configuration') else {}
        topology_params = {
            'initial_neighbors': initial_neighbors,
            'preserve_geography': original_config.get('preserve_geography', False)
        }
        
        try:
            # Generate using unified generator with topology parameters
            graph = self.unified_generator.generate_topology(
                'small_world', 
                rewiring_prob,
                **topology_params
            )
            # 🎯 CRITICAL: Verify route IDs are preserved
            self._verify_route_ids(graph, "small_world")
            # Create network manager with filtered config
            network_manager = self._create_standard_manager(graph, config)
            
            print(f"✅ Small-world network created (p={rewiring_prob}, k={initial_neighbors})")
            print(f"   - Nodes: {network_manager.active_network.number_of_nodes()}")
            print(f"   - Edges: {network_manager.active_network.number_of_edges()}")
            
            return network_manager
            
        except Exception as e:
            print(f"❌ Failed to create small-world network: {e}")
            raise
    
    def _create_scale_free_network(self, config: Dict[str, Any]) -> 'TwoLayerNetworkManager':
        """Create scale-free network using unified generator"""
        attachment_param = config.get('attachment_parameter', 2)
        alpha = config.get('alpha', 1.0)
        
        # CRITICAL FIX: Set the topology type in config
        config['topology_type'] = 'scale_free'
        
        # Extract topology-specific parameters from original config
        original_config = self.config_manager.get_base_configuration() if hasattr(self.config_manager, 'get_base_configuration') else {}
        topology_params = {
            'alpha': alpha,
            'preserve_geography': original_config.get('preserve_geography', False)
        }
        
        try:
            # Generate using unified generator with topology parameters
            graph = self.unified_generator.generate_topology(
                'scale_free', 
                attachment_param,
                **topology_params
            )
            # 🎯 CRITICAL: Verify route IDs are preserved
            self._verify_route_ids(graph, "scale_free")
            
            # Create network manager with filtered config
            network_manager = self._create_standard_manager(graph, config)
            
            print(f"✅ Scale-free network created (m={attachment_param}, α={alpha})")
            print(f"   - Nodes: {network_manager.active_network.number_of_nodes()}")
            print(f"   - Edges: {network_manager.active_network.number_of_edges()}")
            
            return network_manager
            
        except Exception as e:
            print(f"❌ Failed to create scale-free network: {e}")
            raise
    def _verify_route_ids(self, graph: nx.Graph, topology_type: str):
        """Verify that all edges have route IDs - CRITICAL VERIFICATION"""
        
        total_edges = graph.number_of_edges()
        edges_with_route_id = 0
        sydney_routes = 0
        new_routes = 0
        
        for u, v, data in graph.edges(data=True):
            if 'route_id' in data and data['route_id'] != 'unknown':
                edges_with_route_id += 1
                route_id = data['route_id']
                
                # Count Sydney vs new routes
                if any(prefix in route_id for prefix in ['T1_', 'T4_', 'T8_', 'BUS_380', 'BUS_E60']):
                    sydney_routes += 1
                else:
                    new_routes += 1
            else:
                print(f"   ⚠️ Missing route_id: {u}->{v}, data: {list(data.keys())}")
        
        # Report results
        success_rate = edges_with_route_id / total_edges * 100 if total_edges > 0 else 0
        
        print(f"   📊 Route ID Verification ({topology_type}):")
        print(f"   - Total edges: {total_edges}")
        print(f"   - Edges with route_id: {edges_with_route_id}/{total_edges} ({success_rate:.1f}%)")
        print(f"   - Sydney routes preserved: {sydney_routes}")
        print(f"   - New {topology_type} routes: {new_routes}")
        
        if success_rate < 100:
            print(f"   ❌ WARNING: {total_edges - edges_with_route_id} edges missing route_id!")
        else:
            print(f"   ✅ SUCCESS: All edges have route IDs!")
        
        return success_rate == 100
    def _create_base_sydney_network(self, config: Dict[str, Any]) -> 'TwoLayerNetworkManager':
        """Create base Sydney network"""
        connectivity_level = config.get('connectivity_level', 6)
        
        try:
            # Generate using unified generator
            graph = self.unified_generator.generate_topology(
                'base_sydney', 
                connectivity_level
            )
            
            # Create network manager with filtered config
            network_manager = self._create_standard_manager(graph, config)
            
            print(f"✅ Base Sydney network created (connectivity={connectivity_level})")
            print(f"   - Nodes: {network_manager.active_network.number_of_nodes()}")
            print(f"   - Edges: {network_manager.active_network.number_of_edges()}")
            
            return network_manager
            
        except Exception as e:
            print(f"❌ Failed to create base Sydney network: {e}")
            raise
    
    def _create_standard_manager(self, graph: nx.Graph, config: Dict[str, Any]) -> 'TwoLayerNetworkManager':
        """Create standard TwoLayerNetworkManager for any topology - COMPLETELY FIXED VERSION"""
        
        # CRITICAL FIX: Get the ACTUAL topology type and parameter from config
        topology_type = config.get('topology_type', 'degree_constrained')
        
        # FIXED: Extract the correct parameter based on the ACTUAL topology type being created
        if topology_type == 'degree_constrained':
            actual_parameter = config.get('degree_constraint', 3)
        elif topology_type == 'small_world':
            actual_parameter = config.get('rewiring_probability', 0.1)
        elif topology_type == 'scale_free':
            actual_parameter = config.get('attachment_parameter', 2)
        elif topology_type == 'base_sydney':
            actual_parameter = config.get('connectivity_level', 6)
        elif topology_type == 'grid':
            actual_parameter = config.get('connectivity_level', 4)
        else:
            actual_parameter = config.get('variation_parameter', 3)  # Fallback
        
        print(f"🔍 DEBUG: topology_type={topology_type}, actual_parameter={actual_parameter}")
        
        # FIX: Create manager_config with both filtered params AND defaults
        # Your config has these keys: ['sydney_realism', 'preserve_hierarchy', 'base_seed', 'minimum_connectivity', 'enable_caching', 'max_cache_size', 'topology_type', 'connectivity_level']
        # But TwoLayerNetworkManager expects: ['grid_width', 'grid_height', 'congestion_alpha', 'congestion_beta', 'capacity_per_edge']
        
        # Start with defaults that TwoLayerNetworkManager needs
        manager_config = {
            'grid_width': 100,
            'grid_height': 80,
            'congestion_alpha': 0.15,
            'congestion_beta': 4.0,
            'capacity_per_edge': 1000
        }
        
        # Override with any values from your config
        allowed_params = ['grid_width', 'grid_height', 'congestion_alpha', 'congestion_beta', 'capacity_per_edge']
        for param in allowed_params:
            if param in config:
                manager_config[param] = config[param]
        
        print(f"🔍 DEBUG: manager_config={manager_config}")
        
        # Get base network for spatial mapping
        base_network = self._get_base_sydney_network()
        print(f"🔍 DEBUG: About to create GraphNetworkManager")
        print(f"   graph type: {type(graph)}")
        print(f"   base_network type: {type(base_network)}")
        print(f"   manager_config type: {type(manager_config)}")
        print(f"   topology_type type: {type(topology_type)}")
        print(f"   actual_parameter type: {type(actual_parameter)}")
        
        # Create a custom network manager that wraps the graph
        class GraphNetworkManager(TwoLayerNetworkManager):
            def __init__(self, network_graph, base_net, filtered_config, topo_type, param):
                print(f"🏗️ Creating TwoLayerNetworkManager with topology_type={topo_type}, parameter={param}")
                print(f"🔍 DEBUG: filtered_config contents: {filtered_config}")
                print(f"🔍 DEBUG: About to call super().__init__")
                self.route_calculation_count = 0      # ← Add this
                self.total_route_calculation_time = 0 # ← Add this
                # FIXED: Complete all topology type cases WITHOUT fallback
                if topo_type == 'degree_constrained':
                    super().__init__(
                        topology_type=topo_type,
                        degree=int(param),  # Use actual degree parameter
                        grid_width=filtered_config.get('grid_width', 100),
                        grid_height=filtered_config.get('grid_height', 80)
                    )
                elif topo_type == 'small_world':
                    super().__init__(
                        topology_type=topo_type,
                        degree=3,  # Default degree for small world
                        rewiring_probability=float(param),  # Use actual rewiring probability
                        grid_width=filtered_config.get('grid_width', 100),
                        grid_height=filtered_config.get('grid_height', 80)
                    )
                elif topo_type == 'scale_free':
                    # FIXED: Use scale_free directly instead of degree_constrained
                    super().__init__(
                        topology_type='scale_free',  # ✅ Use actual topology type
                        attachment_parameter=int(param),  # ✅ Use correct parameter name
                        grid_width=filtered_config.get('grid_width', 100),
                        grid_height=filtered_config.get('grid_height', 80)
                    )
                elif topo_type == 'grid':
                    # FIXED: Use grid directly instead of degree_constrained
                    super().__init__(
                        topology_type='grid',  # ✅ Use actual topology type
                        connectivity_level=int(param),  # ✅ Use correct parameter name
                        grid_width=filtered_config.get('grid_width', 100),
                        grid_height=filtered_config.get('grid_height', 80)
                    )
                elif topo_type == 'base_sydney':
                    super().__init__(
                        topology_type='base_sydney',
                        connectivity_level=int(param),
                        grid_width=filtered_config.get('grid_width', 100),
                        grid_height=filtered_config.get('grid_height', 80)
                    )
                else:
                    # Default case
                    super().__init__(
                        topology_type='degree_constrained',
                        degree=int(param) if isinstance(param, (int, float)) else 3,
                        grid_width=filtered_config.get('grid_width', 100),
                        grid_height=filtered_config.get('grid_height', 80)
                    )
                
                print(f"🔍 DEBUG: super().__init__() completed successfully")
                
                # CRITICAL: Replace the auto-generated network with our custom graph
                self.active_network = network_graph
                self.base_network = base_net
                
                # Store the actual topology type and parameter 
                self.topology_type = topo_type
                self.variation_parameter = param
                
                # Recreate spatial mapper and other components with new network
                from topology.unified_network_integration import UnifiedSpatialMapper, NetworkCongestionModel, UnifiedNetworkRouter
                
                self.spatial_mapper = UnifiedSpatialMapper(base_net, 
                                                        filtered_config.get('grid_width', 100),
                                                        filtered_config.get('grid_height', 80))
                
                # Recreate congestion model and router with the correct network
                self.congestion_model = NetworkCongestionModel(network_graph)
                # Initialize router with the correct constructor for this codebase
                # UnifiedNetworkRouter(network_graph, congestion_model)
                self.router = UnifiedNetworkRouter(network_graph, self.congestion_model)
                
                print(f"✅ Custom network manager created: {network_graph.number_of_nodes()} nodes, {network_graph.number_of_edges()} edges")
                print(f"✅ Custom network manager created")
                print(f"   Network: {network_graph.number_of_nodes()} nodes, {network_graph.number_of_edges()} edges")
                print(f"   Topology: {topo_type} with parameter {param}")
            
            def _update_router_network(self):
                """Update router's network reference after topology changes"""
                if hasattr(self, 'router'):
                    self.router.network = self.active_network
                    if hasattr(self.router, 'congestion_model'):
                        self.router.congestion_model.network = self.active_network
                    # Clear router's cache since network changed
                    if hasattr(self.router, 'route_cache'):
                        self.router.route_cache.clear()
                    if hasattr(self.router, 'shortest_paths_cache'):
                        self.router.shortest_paths_cache.clear()
        
        # Create the custom manager
        try:
            manager = GraphNetworkManager(graph, base_network, manager_config, topology_type, actual_parameter)
            return manager
        except Exception as e:
            print(f"❌ Failed to create GraphNetworkManager: {e}")
            print(f"🔍 DEBUG: Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def _get_base_sydney_network(self):
        """Get or create base Sydney network with validation"""
        
        cache_key = "base_sydney_network"
        
        if cache_key in self._base_network_cache:
            base_network = self._base_network_cache[cache_key]
            if self._validate_cached_network(base_network):
                return base_network
            else:
                print("🔄 Cached base network invalid, recreating...")
        
        # Create new base network
        from topology.network_topology import SydneyNetworkTopology
        base_network = SydneyNetworkTopology()
        base_network.initialize_base_sydney_network()
        
        # Validate creation
        if not self._validate_base_sydney_network(base_network):
            print("❌ Base network creation failed, using minimal fallback")
            base_network = self._create_minimal_sydney_network()
        
        # Cache valid network
        self._base_network_cache[cache_key] = base_network
        print(f"✅ Base Sydney network ready: {base_network.graph.number_of_edges()} edges")
        
        return base_network
    
    def _validate_cached_network(self, network) -> bool:
        """Validate that cached network is still valid"""
        if not hasattr(network, 'graph') or not network.graph:
            return False
        
        return network.graph.number_of_edges() > 10  # Minimum edge count


    def _create_minimal_sydney_network(self):
        """Create minimal Sydney network as absolute fallback"""
        from topology.network_topology import SydneyNetworkTopology, NetworkNode, NetworkEdge, NodeType, TransportMode
        
        minimal_network = SydneyNetworkTopology()
        
        # Add essential nodes
        nodes = [
            NetworkNode("CENTRAL", NodeType.MAJOR_HUB, (50, 40), 0.2, 0.8, 
                    [TransportMode.TRAIN, TransportMode.BUS], "Central Station"),
            NetworkNode("PARRAMATTA", NodeType.TRANSPORT_HUB, (30, 45), 0.4, 0.5,
                    [TransportMode.TRAIN, TransportMode.BUS], "Parramatta"),
            NetworkNode("BONDI_JUNCTION", NodeType.TRANSPORT_HUB, (60, 35), 0.4, 0.3,
                    [TransportMode.BUS], "Bondi Junction"),
        ]
        
        for node in nodes:
            minimal_network._add_node(node)
        
        # Add essential edges
        edges = [
            NetworkEdge("CENTRAL", "PARRAMATTA", TransportMode.TRAIN, 20.0, 1800, 12, 25.0, "T1_WESTERN", 1),
            NetworkEdge("CENTRAL", "BONDI_JUNCTION", TransportMode.BUS, 25.0, 600, 15, 12.0, "BUS_380", 1),
        ]
        
        for edge in edges:
            minimal_network._add_edge(edge)
        
        print("⚠️ Using minimal Sydney network fallback")
        return minimal_network
    
    def _validate_base_sydney_network(self, base_network) -> bool:
        """Validate that base Sydney network has required routes"""
        required_routes = ['T1_WESTERN', 'T4_ILLAWARRA', 'BUS_380']
        found_routes = 0
        
        for u, v, data in base_network.graph.edges(data=True):
            route_id = data.get('route_id', '')
            for required in required_routes:
                if required in route_id:
                    found_routes += 1
                    break
        
        return found_routes >= 2  # At least 2 out of 3 critical routes


# ===== STANDARDIZED INTERFACE =====
class StandardizedNetworkInterface:
    """
    Provides consistent interface for all network types.
    Maps different implementations to a standard API.
    """
    
    def __init__(self, network_manager: 'TwoLayerNetworkManager'):
        self.network_manager = network_manager

    
 
    
    def update_congestion(self, route: list, agent_id: str, add: bool = True):
        """Update network congestion"""
        if hasattr(self.network_manager, 'update_edge_usage'):
            self.network_manager.update_edge_usage(route, agent_id, add)
        elif hasattr(self.network_manager, 'congestion_model'):
            # Fallback for different implementations
            for i in range(len(route) - 1):
                edge = (route[i], route[i+1])
                if add:
                    self.network_manager.congestion_model.add_agent_to_edge(edge, agent_id)
                else:
                    self.network_manager.congestion_model.remove_agent_from_edge(edge, agent_id)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        if hasattr(self.network_manager, 'get_network_stats'):
            return self.network_manager.get_network_stats()
        
        # Fallback implementation
        network = self.network_manager.active_network
        return {
            'num_nodes': network.number_of_nodes(),
            'num_edges': network.number_of_edges(),
            'is_connected': nx.is_connected(network),
            'topology_type': getattr(self.network_manager, 'topology_type', 'unknown')
        }
    
    def get_spatial_position(self, node_id: str) -> tuple:
        """Get spatial position of a network node"""
        if hasattr(self.network_manager, 'spatial_mapper'):
            return self.network_manager.spatial_mapper.get_grid_coordinates(node_id)  # Fixed method name
        return (0, 0)

# ===== CONVENIENCE FUNCTIONS =====
def create_network(network_type: str, variation_parameter: Union[float, int] = None, **kwargs) -> StandardizedNetworkInterface:
    """
    Main entry point for creating networks.
    
    Usage examples:
        create_network('degree_constrained', 4)
        create_network('small_world', 0.1)
        create_network('scale_free', 2)
    """
    factory = NetworkFactory()
    network_manager = factory.create_network(network_type, variation_parameter, **kwargs)
    return StandardizedNetworkInterface(network_manager)

def create_research_network(study_name: str, variation_parameter: Union[float, int] = None) -> StandardizedNetworkInterface:
    """Create network for specific research study"""
    try:
        from network_config import get_research_config
        config = get_research_config(study_name)
        network_type = config['topology_type']
    except:
        # Fallback if config not available
        network_type = 'degree_constrained'
    
    factory = NetworkFactory()
    network_manager = factory.create_network(network_type, variation_parameter)
    return StandardizedNetworkInterface(network_manager)

def batch_create_networks(network_type: str, parameter_values: list) -> Dict[str, StandardizedNetworkInterface]:
    """Create multiple networks for parameter comparison"""
    factory = NetworkFactory()
    networks = {}
    
    for param_value in parameter_values:
        key = f"{network_type}_{param_value}"
        try:
            network_manager = factory.create_network(network_type, param_value)
            networks[key] = StandardizedNetworkInterface(network_manager)
            print(f"✅ Created {key}")
        except Exception as e:
            print(f"❌ Failed to create {key}: {e}")
    
    return networks

# Utility functions for integration with existing system
def create_network_from_config(config_dict: Dict[str, Any]) -> StandardizedNetworkInterface:
    """Create network directly from configuration dictionary"""
    factory = NetworkFactory()
    
    network_type = config_dict.get('network_type', 'degree_constrained')
    variation_param = config_dict.get('variation_parameter', 4)
    
    network_manager = factory.create_network(network_type, variation_param, **config_dict)
    return StandardizedNetworkInterface(network_manager)

# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    print("🧪 UNIFIED NETWORK FACTORY - TESTING ALL TOPOLOGIES")
    print("=" * 60)
    
    # Test 1: Basic factory functionality
    print("\n1️⃣ Testing Network Factory")
    factory = NetworkFactory()
    
    # Test all topology types
    test_configs = [
        ('degree_constrained', 4),
        ('small_world', 0.1),
        ('scale_free', 2),
        ('base_sydney', 6)
    ]
    
    for topology_type, param in test_configs:
        print(f"\n🧪 Testing {topology_type} (param: {param})")
        
        try:
            network_interface = create_network(topology_type, param)
            stats = network_interface.get_network_stats()
            
            print(f"   ✅ Network created successfully")
            print(f"   📊 Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
            print(f"   🔗 Connected: {stats['is_connected']}")
            
            # Test routing
            route = network_interface.find_route((10, 10), (20, 20))
            print(f"   🛣️ Test route length: {len(route) if route else 'No route found'}")
            
        except Exception as e:
            print(f"❌ Failed in _get_base_sydney_network: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 2: Batch creation
    print("\n2️⃣ Testing Batch Network Creation")
    batch_networks = batch_create_networks('degree_constrained', [3, 4, 5])
    print(f"   Created {len(batch_networks)} networks")
    
    print("\n✅ Network factory testing completed!")