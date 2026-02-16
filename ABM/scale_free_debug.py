#!/usr/bin/env python3
"""
🔍 SCALE-FREE NETWORK DEBUG ANALYSIS - FIXED VERSION
Comprehensive debugging with proper configuration management
"""

import sys
import networkx as nx
from typing import Dict, List, Optional
import json

# Core imports
try:
    from Complete_NTEO.topology.scale_free_topology import ScaleFreeTopologyGenerator, create_scale_free_network_manager
    from Complete_NTEO.topology.network_topology import SydneyNetworkTopology, TransportMode
    import database as db
    from agent_run_visualisation import MobilityModel
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ScaleFreeDebugAnalyzer:
    """Comprehensive debugger for scale-free network issues"""
    
    def __init__(self):
        self.debug_results = {}
        # CRITICAL: Store the original scale-free configuration
        self.scale_free_config = {
            'topology_type': 'scale_free',
            'attachment_parameter': 2,
            'preferential_strength': 1.0,
            'grid_width': 100,
            'grid_height': 80,
            'preserve_geography': True,
            'max_attachment_distance': 50
        }
        
    def ensure_scale_free_config(self):
        """Ensure scale-free configuration is active"""
        print("🔧 Ensuring scale-free configuration...")
        db.NETWORK_CONFIG.update(self.scale_free_config)
        print(f"   Config set to: {db.NETWORK_CONFIG['topology_type']}")
        
    def analyze_network_generation_vs_abm(self):
        """Compare what scale-free generation creates vs what ABM uses"""
        print("\n🔍 ANALYZING: Network Generation vs ABM Usage")
        print("=" * 60)
        
        # STEP 1: Ensure correct configuration
        self.ensure_scale_free_config()
        
        # Step 2: Generate scale-free network directly
        print("📋 Step 1: Direct scale-free generation...")
        base_network = SydneyNetworkTopology()
        base_network.initialize_base_sydney_network()
        
        generator = ScaleFreeTopologyGenerator(base_network)
        sf_network = generator.generate_scale_free_network(m_edges=2, alpha=1.0)
        
        print(f"   Generated scale-free network:")
        print(f"   - Nodes: {sf_network.number_of_nodes()}")
        print(f"   - Edges: {sf_network.number_of_edges()}")
        print(f"   - Connected: {nx.is_connected(sf_network)}")
        
        # Check edge attributes
        sample_edges = list(sf_network.edges(data=True))[:3]
        if sample_edges:
            print(f"   - Sample edge attributes: {list(sample_edges[0][2].keys())}")
            for i, (u, v, data) in enumerate(sample_edges):
                route_id = data.get('route_id', 'MISSING')
                transport_mode = data.get('transport_mode', 'MISSING')
                print(f"     Edge {i+1}: {u}→{v}, route_id={route_id}, mode={transport_mode}")
        
        # Step 3: Create network manager
        print(f"\n📋 Step 2: Network manager creation...")
        network_manager = create_scale_free_network_manager(m_edges=2, alpha=1.0)
        
        manager_network = network_manager.active_network
        print(f"   Network manager network:")
        print(f"   - Nodes: {manager_network.number_of_nodes()}")
        print(f"   - Edges: {manager_network.number_of_edges()}")
        print(f"   - Connected: {nx.is_connected(manager_network)}")
        
        # Step 4: MobilityModel initialization - ENSURE SCALE-FREE CONFIG
        print(f"\n📋 Step 3: MobilityModel initialization...")
        
        # CRITICAL: Restore scale-free config before creating model
        self.ensure_scale_free_config()
        
        try:
            model = MobilityModel(
                db_connection_string=db.DB_CONNECTION_STRING,
                num_commuters=5,  # Very small for debugging
                data_income_weights=db.income_weights,
                data_health_weights=db.health_weights,
                data_payment_weights=db.payment_weights,
                data_age_distribution=db.age_distribution,
                data_disability_weights=db.disability_weights,
                data_tech_access_weights=db.tech_access_weights,
                subsidy_config=db.monthly_config,
                network_config=self.scale_free_config,  # ← Use explicit scale-free config
                ASC_VALUES=db.ASC_VALUES,
                UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS=db.UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
                UTILITY_FUNCTION_BASE_COEFFICIENTS=db.UTILITY_FUNCTION_BASE_COEFFICIENTS,
                PENALTY_COEFFICIENTS=db.PENALTY_COEFFICIENTS,
                AFFORDABILITY_THRESHOLDS=db.AFFORDABILITY_THRESHOLDS,
                FLEXIBILITY_ADJUSTMENTS=db.FLEXIBILITY_ADJUSTMENTS,
                VALUE_OF_TIME=db.VALUE_OF_TIME,
                public_price_table=db.public_price_table,
                ALPHA_VALUES=db.ALPHA_VALUES,
                DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=db.DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
                BACKGROUND_TRAFFIC_AMOUNT=db.BACKGROUND_TRAFFIC_AMOUNT,
                CONGESTION_ALPHA=db.CONGESTION_ALPHA,
                CONGESTION_BETA=db.CONGESTION_BETA,
                CONGESTION_CAPACITY=db.CONGESTION_CAPACITY,
                CONGESTION_T_IJ_FREE_FLOW=db.CONGESTION_T_IJ_FREE_FLOW,
                uber_like1_capacity=db.UberLike1_capacity,
                uber_like1_price=db.UberLike1_price,
                uber_like2_capacity=db.UberLike2_capacity,
                uber_like2_price=db.UberLike2_price,
                bike_share1_capacity=db.BikeShare1_capacity,
                bike_share1_price=db.BikeShare1_price,
                bike_share2_capacity=db.BikeShare2_capacity,
                bike_share2_price=db.BikeShare2_price,
                subsidy_dataset=db.subsidy_dataset,
            )
            
            abm_network = model.network_manager.active_network
            
            print(f"   MobilityModel network:")
            print(f"   - Nodes: {abm_network.number_of_nodes()}")
            print(f"   - Edges: {abm_network.number_of_edges()}")
            
            # Check for small-world contamination
            shortcut_count = sum(1 for u, v, data in abm_network.edges(data=True) 
                               if data.get('edge_type') == 'shortcut')
            
            if shortcut_count > 0:
                print(f"   ❌ ERROR: ABM network has {shortcut_count} shortcuts (small-world contamination)!")
            else:
                print(f"   ✅ SUCCESS: ABM network is pure scale-free (no shortcuts)")
            
            # Store results
            self.debug_results['networks'] = {
                'sf_generated': {'nodes': sf_network.number_of_nodes(), 'edges': sf_network.number_of_edges()},
                'manager': {'nodes': manager_network.number_of_nodes(), 'edges': manager_network.number_of_edges()},
                'abm_final': {'nodes': abm_network.number_of_nodes(), 'edges': abm_network.number_of_edges()},
                'shortcuts_found': shortcut_count
            }
            
            return model
            
        except Exception as e:
            print(f"   ❌ MobilityModel creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_with_other_topologies(self):
        """Compare scale-free behavior with other topology types"""
        print(f"\n⚖️ COMPARING: Scale-Free vs Other Topologies")
        print("=" * 60)
        
        # CRITICAL: Save the original scale-free config
        original_config = db.NETWORK_CONFIG.copy()
        
        topologies_to_test = [
            {'name': 'degree_constrained', 'config': {'topology_type': 'degree_constrained', 'degree_constraint': 3}},
            {'name': 'small_world', 'config': {'topology_type': 'small_world', 'rewiring_probability': 0.1}},
        ]
        
        comparison_results = {}
        
        for topo in topologies_to_test:
            print(f"\n📋 Testing {topo['name']}...")
            
            # Temporarily update config for this topology
            test_config = original_config.copy()
            test_config.update(topo['config'])
            db.NETWORK_CONFIG.update(test_config)
            
            try:
                # Create minimal model with explicit config
                test_model = MobilityModel(
                    db_connection_string=db.DB_CONNECTION_STRING,
                    num_commuters=5,
                    data_income_weights=db.income_weights,
                    data_health_weights=db.health_weights,
                    data_payment_weights=db.payment_weights,
                    data_age_distribution=db.age_distribution,
                    data_disability_weights=db.disability_weights,
                    data_tech_access_weights=db.tech_access_weights,
                    subsidy_config=db.monthly_config,
                    network_config=test_config,  # ← Use explicit test config
                    ASC_VALUES=db.ASC_VALUES,
                    UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS=db.UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
                    UTILITY_FUNCTION_BASE_COEFFICIENTS=db.UTILITY_FUNCTION_BASE_COEFFICIENTS,
                    PENALTY_COEFFICIENTS=db.PENALTY_COEFFICIENTS,
                    AFFORDABILITY_THRESHOLDS=db.AFFORDABILITY_THRESHOLDS,
                    FLEXIBILITY_ADJUSTMENTS=db.FLEXIBILITY_ADJUSTMENTS,
                    VALUE_OF_TIME=db.VALUE_OF_TIME,
                    public_price_table=db.public_price_table,
                    ALPHA_VALUES=db.ALPHA_VALUES,
                    DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=db.DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
                    BACKGROUND_TRAFFIC_AMOUNT=db.BACKGROUND_TRAFFIC_AMOUNT,
                    CONGESTION_ALPHA=db.CONGESTION_ALPHA,
                    CONGESTION_BETA=db.CONGESTION_BETA,
                    CONGESTION_CAPACITY=db.CONGESTION_CAPACITY,
                    CONGESTION_T_IJ_FREE_FLOW=db.CONGESTION_T_IJ_FREE_FLOW,
                    uber_like1_capacity=db.UberLike1_capacity,
                    uber_like1_price=db.UberLike1_price,
                    uber_like2_capacity=db.UberLike2_capacity,
                    uber_like2_price=db.UberLike2_price,
                    bike_share1_capacity=db.BikeShare1_capacity,
                    bike_share1_price=db.BikeShare1_price,
                    bike_share2_capacity=db.BikeShare2_capacity,
                    bike_share2_price=db.BikeShare2_price,
                    subsidy_dataset=db.subsidy_dataset,
                )
                
                network = test_model.network_manager.active_network
                
                # Test movement
                initial_positions = [c.location for c in test_model.commuter_agents if hasattr(c, 'location')]
                movements = 0
                
                for step in range(3):
                    test_model.step()
                    final_positions = [c.location for c in test_model.commuter_agents if hasattr(c, 'location')]
                    
                    step_movements = sum(1 for initial, final in zip(initial_positions, final_positions)
                                       if initial != final)
                    movements += step_movements
                
                comparison_results[topo['name']] = {
                    'nodes': network.number_of_nodes(),
                    'edges': network.number_of_edges(),
                    'movements_in_3_steps': movements,
                    'commuters': len(test_model.commuter_agents)
                }
                
                print(f"   Nodes: {network.number_of_nodes()}, Edges: {network.number_of_edges()}")
                print(f"   Movements in 3 steps: {movements}")
                
            except Exception as topo_error:
                print(f"   Error testing {topo['name']}: {topo_error}")
                comparison_results[topo['name']] = {'error': str(topo_error)}
            
            finally:
                # CRITICAL: Restore original scale-free config after each test
                db.NETWORK_CONFIG.update(original_config)
                print(f"   ✅ Config restored to: {db.NETWORK_CONFIG['topology_type']}")
        
        # Store results
        self.debug_results['topology_comparison'] = comparison_results
    
    def generate_debug_report(self):
        """Generate comprehensive debug report"""
        print(f"\n📋 COMPREHENSIVE DEBUG REPORT")
        print("=" * 70)
        
        print(f"🔍 SUMMARY OF FINDINGS:")
        
        # Network analysis
        if 'networks' in self.debug_results:
            net_data = self.debug_results['networks']
            print(f"\n🌐 Network Analysis:")
            print(f"   Generated scale-free: {net_data['sf_generated']['nodes']} nodes, {net_data['sf_generated']['edges']} edges")
            print(f"   Network manager: {net_data['manager']['nodes']} nodes, {net_data['manager']['edges']} edges")
            print(f"   Final ABM network: {net_data['abm_final']['nodes']} nodes, {net_data['abm_final']['edges']} edges")
            print(f"   Shortcuts detected: {net_data['shortcuts_found']}")
            
            if net_data['shortcuts_found'] > 0:
                print(f"   ❌ ISSUE: ABM network contains small-world shortcuts!")
            else:
                print(f"   ✅ SUCCESS: Pure scale-free network confirmed!")
        
        # Topology comparison
        if 'topology_comparison' in self.debug_results:
            topo_data = self.debug_results['topology_comparison']
            print(f"\n⚖️ Topology Comparison:")
            for topo_name, data in topo_data.items():
                if 'error' not in data:
                    print(f"   {topo_name}: {data.get('movements_in_3_steps', 0)} movements")
                else:
                    print(f"   {topo_name}: ERROR - {data['error']}")
        
        return []

def main():
    """Main debugging execution"""
    print("🔍 SCALE-FREE NETWORK DEBUG ANALYSIS - FIXED VERSION")
    print("="*70)
    print("Comprehensive debugging with proper configuration management")
    print()
    
    analyzer = ScaleFreeDebugAnalyzer()
    
    # Run main analysis (this includes MobilityModel test)
    analyzer.analyze_network_generation_vs_abm()
    
    # Run comparison (with proper config restoration)
    analyzer.compare_with_other_topologies()
    
    # Generate final report
    analyzer.generate_debug_report()
    
    print(f"\n✅ DEBUG COMPLETE: Configuration properly managed throughout analysis")

if __name__ == "__main__":
    main()