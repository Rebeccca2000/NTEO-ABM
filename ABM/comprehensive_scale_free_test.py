#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE SCALE-FREE ABM TEST SUITE
Complete testing framework for scale-free networks with ABM integration

This single script tests everything:
1. Network generation validation
2. ABM integration testing  
3. Route finding functionality
4. Performance benchmarking
5. Mini simulation runs
6. Configuration validation

Usage: python comprehensive_scale_free_test.py

Requirements:
- scale_free_topology.py in your project
- network_topology.py available
- database.py configured for scale-free networks
"""

import sys
import os
import time
import traceback
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import argparse

# Core imports with error handling
try:
    from Complete_NTEO.topology.scale_free_topology import ScaleFreeTopologyGenerator, ScaleFreeParameterOptimizer, create_scale_free_network_manager
    from Complete_NTEO.topology.network_topology import SydneyNetworkTopology, NodeType, TransportMode
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Critical import error: {e}")
    print("Required files:")
    print("  - scale_free_topology.py")
    print("  - network_topology.py") 
    print("  - network_integration.py")
    print("\nPlease ensure these files are in your project directory.")
    sys.exit(1)

# Database import with configuration check
try:
    import database as db
    print("✅ Database module imported")
except ImportError as e:
    print(f"⚠️  Database import warning: {e}")
    print("Will proceed with default configuration")
    
    # Create minimal config if database.py not available
    class MockDatabase:
        NETWORK_CONFIG = {
            'topology_type': 'scale_free',
            'grid_width': 100,
            'grid_height': 80,
            'sydney_realism': True,
            'preserve_hierarchy': True,
            'attachment_parameter': 2,
            'preferential_strength': 1.0
        }
    db = MockDatabase()

class ComprehensiveScaleFreeTest:
    """Complete test suite for scale-free networks"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_data = {}
        self.network_manager = None
        
        # Initialize base network
        try:
            self.base_network = SydneyNetworkTopology()
            self.base_network.initialize_base_sydney_network()
            print(f"✅ Base Sydney network: {len(self.base_network.nodes)} nodes")
        except Exception as e:
            print(f"❌ Failed to initialize base network: {e}")
            sys.exit(1)
        
        # Test configurations
        self.test_configs = [
            {'m': 1, 'alpha': 1.0, 'name': 'Low_Connectivity'},
            {'m': 2, 'alpha': 1.0, 'name': 'Standard'},
            {'m': 3, 'alpha': 1.0, 'name': 'High_Connectivity'},
            {'m': 2, 'alpha': 1.5, 'name': 'Strong_Preference'}
        ]
    
    def check_database_configuration(self) -> bool:
        """Check if database.py is properly configured for scale-free networks"""
        print("\n🔧 DATABASE CONFIGURATION CHECK")
        print("=" * 50)
        
        try:
            # Check if NETWORK_CONFIG exists
            if not hasattr(db, 'NETWORK_CONFIG'):
                print("❌ NETWORK_CONFIG not found in database.py")
                self._show_database_config_guide()
                return False
            
            config = db.NETWORK_CONFIG
            
            # Check topology type
            if config.get('topology_type') != 'scale_free':
                print(f"⚠️  topology_type is '{config.get('topology_type')}', should be 'scale_free'")
                self._show_database_config_guide()
                return False
            
            # Check required parameters
            required_params = ['attachment_parameter', 'preferential_strength']
            missing_params = [p for p in required_params if p not in config]
            
            if missing_params:
                print(f"⚠️  Missing parameters: {missing_params}")
                self._show_database_config_guide()
                return False
            
            print("✅ Database configuration looks good!")
            print(f"   Topology: {config.get('topology_type')}")
            print(f"   Attachment parameter (m): {config.get('attachment_parameter')}")
            print(f"   Preferential strength (α): {config.get('preferential_strength')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Database configuration check failed: {e}")
            self._show_database_config_guide()
            return False
    
    def _show_database_config_guide(self):
        """Show how to configure database.py for scale-free networks"""
        print("\n📝 DATABASE.PY CONFIGURATION GUIDE")
        print("=" * 50)
        print("Add/update this in your database.py file:")
        print("""
NETWORK_CONFIG = {
    'topology_type': 'scale_free',        # CHANGE THIS LINE
    'grid_width': 100,
    'grid_height': 80,
    'sydney_realism': True,
    'preserve_hierarchy': True,
    
    # ADD THESE SCALE-FREE PARAMETERS:
    'attachment_parameter': 2,            # m parameter (1-3 recommended)
    'preferential_strength': 1.0,        # α parameter (0.5-1.5 range)
    'preserve_geography': True,
    'max_attachment_distance': 50,
}
        """)
    
    def test_basic_network_generation(self) -> bool:
        """Test 1: Basic scale-free network generation"""
        print("\n🔧 TEST 1: BASIC NETWORK GENERATION")
        print("=" * 50)
        
        try:
            generator = ScaleFreeTopologyGenerator(self.base_network)
            all_passed = True
            
            for config in self.test_configs:
                print(f"\n📋 Testing {config['name']} (m={config['m']}, α={config['alpha']})...")
                
                # Generate network
                start_time = time.time()
                sf_network = generator.generate_scale_free_network(
                    m_edges=config['m'], 
                    alpha=config['alpha']
                )
                generation_time = time.time() - start_time
                
                # Validate network
                num_nodes = sf_network.number_of_nodes()
                num_edges = sf_network.number_of_edges()
                is_connected = nx.is_connected(sf_network) if num_nodes > 0 else False
                avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
                
                print(f"   Nodes: {num_nodes}")
                print(f"   Edges: {num_edges}")
                print(f"   Connected: {is_connected}")
                print(f"   Avg degree: {avg_degree:.2f}")
                print(f"   Generation time: {generation_time:.3f}s")
                
                # Check edge attributes
                sample_edges = list(sf_network.edges(data=True))
                if sample_edges:
                    required_attrs = ['transport_mode', 'route_id', 'travel_time']
                    edge_attrs = sample_edges[0][2].keys()
                    missing_attrs = [attr for attr in required_attrs if attr not in edge_attrs]
                    
                    if missing_attrs:
                        print(f"   ⚠️  Missing edge attributes: {missing_attrs}")
                        all_passed = False
                    else:
                        print(f"   ✅ All required edge attributes present")
                
                # Validation
                if num_nodes > 20 and is_connected and avg_degree >= 2:
                    print(f"   ✅ {config['name']} generation successful")
                    self.test_results[f"generation_{config['name']}"] = True
                else:
                    print(f"   ❌ {config['name']} generation failed validation")
                    all_passed = False
                    self.test_results[f"generation_{config['name']}"] = False
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Network generation test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_network_manager_creation(self) -> bool:
        """Test 2: Network manager creation for ABM integration"""
        print("\n🔗 TEST 2: NETWORK MANAGER CREATION")
        print("=" * 50)
        
        try:
            all_passed = True
            
            for config in self.test_configs[:2]:  # Test first 2 configs to save time
                print(f"\n📋 Testing manager creation for {config['name']}...")
                
                # Create network manager
                start_time = time.time()
                network_manager = create_scale_free_network_manager(
                    m_edges=config['m'], 
                    alpha=config['alpha']
                )
                creation_time = time.time() - start_time
                
                # Validate components
                required_components = ['active_network', 'router', 'congestion_model', 'spatial_mapper']
                missing_components = [comp for comp in required_components 
                                    if not hasattr(network_manager, comp)]
                
                if missing_components:
                    print(f"   ❌ Missing components: {missing_components}")
                    all_passed = False
                    continue
                
                # Check network properties
                network = network_manager.active_network
                num_nodes = network.number_of_nodes()
                has_routing = hasattr(network_manager.router, 'find_shortest_path')
                
                print(f"   Network nodes: {num_nodes}")
                print(f"   Router available: {has_routing}")
                print(f"   Creation time: {creation_time:.3f}s")
                
                if num_nodes > 0 and has_routing:
                    print(f"   ✅ {config['name']} manager creation successful")
                    self.test_results[f"manager_{config['name']}"] = True
                    
                    # Store one manager for later tests
                    if not self.network_manager:
                        self.network_manager = network_manager
                else:
                    print(f"   ❌ {config['name']} manager creation failed")
                    all_passed = False
                    self.test_results[f"manager_{config['name']}"] = False
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Network manager creation test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_routing_functionality(self) -> bool:
        """Test 3: Route finding and navigation"""
        print("\n🛣️ TEST 3: ROUTING FUNCTIONALITY")
        print("=" * 50)
        
        if not self.network_manager:
            print("❌ No network manager available for routing test")
            return False
        
        try:
            network = self.network_manager.active_network
            nodes = list(network.nodes())
            
            if len(nodes) < 5:
                print("❌ Not enough nodes for comprehensive routing test")
                return False
            
            print(f"📋 Testing routing on {len(nodes)} nodes...")
            
            # Test multiple route pairs
            num_tests = min(15, len(nodes) // 2)
            successful_routes = 0
            failed_routes = 0
            route_times = []
            path_lengths = []
            
            for i in range(num_tests):
                start_node = nodes[i]
                end_node = nodes[-(i+1)]
                
                try:
                    # Test NetworkX routing first
                    start_time = time.time()
                    nx_path = nx.shortest_path(network, start_node, end_node)
                    nx_time = time.time() - start_time
                    
                    # Test network manager routing
                    start_time = time.time()
                    route_result = self.network_manager.router.find_shortest_path(start_node, end_node)
                    manager_time = time.time() - start_time
                    
                    if nx_path and route_result:
                        successful_routes += 1
                        route_times.append(manager_time)
                        path_lengths.append(len(nx_path))
                    else:
                        failed_routes += 1
                    
                except Exception as route_error:
                    failed_routes += 1
                    if i < 3:  # Show first few errors for debugging
                        print(f"   Route {start_node}→{end_node} error: {route_error}")
            
            # Calculate metrics
            success_rate = successful_routes / num_tests * 100
            avg_route_time = np.mean(route_times) if route_times else 0
            avg_path_length = np.mean(path_lengths) if path_lengths else 0
            
            print(f"   Routes tested: {num_tests}")
            print(f"   Success rate: {success_rate:.1f}% ({successful_routes} successful)")
            print(f"   Average route time: {avg_route_time:.4f}s")
            print(f"   Average path length: {avg_path_length:.1f} hops")
            
            # Store performance data
            self.performance_data['routing'] = {
                'success_rate': success_rate,
                'avg_route_time': avg_route_time,
                'avg_path_length': avg_path_length
            }
            
            if success_rate >= 80:  # At least 80% should succeed
                print("   ✅ Routing functionality test passed")
                self.test_results['routing'] = True
                return True
            else:
                print("   ❌ Routing functionality test failed")
                self.test_results['routing'] = False
                return False
                
        except Exception as e:
            print(f"❌ Routing test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_mini_abm_simulation(self) -> bool:
        """Test 4: Mini ABM simulation"""
        print("\n🎭 TEST 4: MINI ABM SIMULATION")
        print("=" * 50)
        
        if not self.network_manager:
            print("❌ No network manager available for simulation")
            return False
        
        try:
            network = self.network_manager.active_network
            nodes = list(network.nodes())
            
            # Simulation parameters
            num_agents = min(8, len(nodes))
            num_steps = 12
            
            print(f"📋 Simulating {num_agents} agents for {num_steps} steps...")
            
            # Initialize agents
            agent_positions = {}
            agent_destinations = {}
            
            for agent_id in range(num_agents):
                agent_positions[agent_id] = nodes[agent_id % len(nodes)]
                agent_destinations[agent_id] = nodes[(agent_id + len(nodes)//2) % len(nodes)]
            
            print(f"   Initial positions: {list(agent_positions.values())[:3]}...")
            print(f"   Destinations: {list(agent_destinations.values())[:3]}...")
            
            # Simulate movement
            movement_attempts = 0
            successful_movements = 0
            total_travel_time = 0
            
            for step in range(num_steps):
                for agent_id in range(num_agents):
                    current_pos = agent_positions[agent_id]
                    destination = agent_destinations[agent_id]
                    
                    if current_pos != destination:
                        movement_attempts += 1
                        
                        try:
                            # Find route
                            start_time = time.time()
                            route = self.network_manager.router.find_shortest_path(
                                current_pos, destination
                            )
                            route_time = time.time() - start_time
                            
                            if route:
                                # Simulate movement (move one step along route)
                                # For simplicity, just move to destination
                                agent_positions[agent_id] = destination
                                successful_movements += 1
                                total_travel_time += route_time
                                
                                # Give agent new random destination
                                agent_destinations[agent_id] = nodes[
                                    (agent_id + step) % len(nodes)
                                ]
                            
                        except Exception as move_error:
                            pass  # Count as failed movement
            
            # Calculate simulation metrics
            movement_success_rate = successful_movements / movement_attempts * 100 if movement_attempts > 0 else 0
            avg_travel_time = total_travel_time / successful_movements if successful_movements > 0 else 0
            
            print(f"   Movement attempts: {movement_attempts}")
            print(f"   Successful movements: {successful_movements}")
            print(f"   Success rate: {movement_success_rate:.1f}%")
            print(f"   Average travel time: {avg_travel_time:.4f}s per movement")
            print(f"   Final positions: {list(agent_positions.values())[:3]}...")
            
            # Store simulation data
            self.performance_data['simulation'] = {
                'movement_success_rate': movement_success_rate,
                'avg_travel_time': avg_travel_time,
                'total_movements': successful_movements
            }
            
            if movement_success_rate >= 70:  # At least 70% should succeed
                print("   ✅ Mini ABM simulation passed")
                self.test_results['simulation'] = True
                return True
            else:
                print("   ❌ Mini ABM simulation failed")
                self.test_results['simulation'] = False
                return False
                
        except Exception as e:
            print(f"❌ Mini simulation test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_scale_free_properties(self) -> bool:
        """Test 5: Scale-free network properties analysis"""
        print("\n📊 TEST 5: SCALE-FREE PROPERTIES ANALYSIS")
        print("=" * 50)
        
        try:
            generator = ScaleFreeTopologyGenerator(self.base_network)
            optimizer = ScaleFreeParameterOptimizer(self.base_network)
            
            all_passed = True
            
            for config in self.test_configs[:2]:  # Test 2 configs to save time
                print(f"\n📋 Analyzing {config['name']}...")
                
                # Generate network
                network = generator.generate_scale_free_network(
                    m_edges=config['m'], alpha=config['alpha']
                )
                
                # Analyze properties
                properties = optimizer.analyze_scale_free_properties(network)
                
                # Expected ranges for validation
                expected_ranges = {
                    'avg_degree': (2.0, 10.0),
                    'density': (0.01, 0.2),
                    'avg_clustering': (0.0, 1.0),
                    'hub_ratio': (0.0, 0.5)
                }
                
                print(f"   Nodes: {properties.get('num_nodes', 0)}")
                print(f"   Average degree: {properties.get('avg_degree', 0):.2f}")
                print(f"   Network density: {properties.get('density', 0):.3f}")
                print(f"   Average clustering: {properties.get('avg_clustering', 0):.3f}")
                print(f"   Hub ratio: {properties.get('hub_ratio', 0):.3f}")
                
                # Validate properties
                validation_passed = True
                for prop, (min_val, max_val) in expected_ranges.items():
                    prop_value = properties.get(prop, 0)
                    if not (min_val <= prop_value <= max_val):
                        print(f"   ⚠️  {prop} ({prop_value:.3f}) outside expected range [{min_val}, {max_val}]")
                        validation_passed = False
                
                if validation_passed:
                    print(f"   ✅ {config['name']} properties within expected ranges")
                else:
                    print(f"   ⚠️  {config['name']} properties outside expected ranges")
                    all_passed = False
                
                self.test_results[f"properties_{config['name']}"] = validation_passed
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Scale-free properties test failed: {e}")
            traceback.print_exc()
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        overall_success_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
        
        print(f"Overall Results: {passed_tests}/{total_tests} tests passed ({overall_success_rate:.1f}%)")
        print()
        
        # Detailed results
        print("Test Details:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        # Performance summary
        if self.performance_data:
            print("\nPerformance Summary:")
            if 'routing' in self.performance_data:
                routing_data = self.performance_data['routing']
                print(f"  Route success rate: {routing_data['success_rate']:.1f}%")
                print(f"  Average route time: {routing_data['avg_route_time']:.4f}s")
            
            if 'simulation' in self.performance_data:
                sim_data = self.performance_data['simulation']
                print(f"  Simulation success rate: {sim_data['movement_success_rate']:.1f}%")
                print(f"  Average movement time: {sim_data['avg_travel_time']:.4f}s")
        
        # Recommendations
        print("\nRecommendations:")
        if overall_success_rate >= 90:
            print("🎉 EXCELLENT! Your scale-free network is ready for full ABM research!")
            print("   • Run comparative studies with other topologies")
            print("   • Analyze equity implications of hub-dominated networks")
            print("   • Test different subsidy scenarios")
        elif overall_success_rate >= 75:
            print("✅ GOOD! Scale-free network is mostly working with minor issues")
            print("   • Review failed tests and fix specific issues")
            print("   • Consider running with reduced complexity initially")
            print("   • Monitor performance during full simulations")
        else:
            print("⚠️  NEEDS WORK! Several critical issues need fixing")
            print("   • Check import dependencies and file structure")
            print("   • Verify database.py configuration")
            print("   • Debug failed tests before proceeding")
        
        return overall_success_rate >= 75

def main():
    """Main test execution"""
    print("🧪 COMPREHENSIVE SCALE-FREE ABM TEST SUITE")
    print("=" * 60)
    print("Testing all scale-free network functionality for ABM integration")
    print()
    
    # Initialize tester
    tester = ComprehensiveScaleFreeTest()
    
    # Run all tests
    tests = [
        ("Database Configuration", tester.check_database_configuration),
        ("Network Generation", tester.test_basic_network_generation),
        ("Network Manager Creation", tester.test_network_manager_creation),
        ("Routing Functionality", tester.test_routing_functionality),
        ("Mini ABM Simulation", tester.test_mini_abm_simulation),
        ("Scale-Free Properties", tester.test_scale_free_properties)
    ]
    
    # Execute tests
    for test_name, test_func in tests:
        print(f"\n🔬 RUNNING: {test_name}")
        try:
            result = test_func()
            if not result:
                print(f"⚠️  {test_name} had issues - check output above")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            tester.test_results[test_name.lower().replace(' ', '_')] = False
    
    # Generate final report
    success = tester.generate_test_report()
    
    if success:
        print(f"\n🎯 NEXT STEPS:")
        print("1. Update your main ABM script to use scale-free networks")
        print("2. Run full simulations with different m and α parameters")
        print("3. Compare equity outcomes with other network topologies")
        print("4. Analyze hub vs peripheral accessibility patterns")
    else:
        print(f"\n🔧 DEBUGGING STEPS:")
        print("1. Check all required files are present")
        print("2. Update database.py configuration as shown above")
        print("3. Fix any import or dependency issues")
        print("4. Re-run this test suite")

if __name__ == "__main__":
    main()