#!/usr/bin/env python3
"""
Test script to verify scale-free network generation without small-world interference
"""

import database as db
import networkx as nx
from Complete_NTEO.topology.scale_free_topology import ScaleFreeTopologyGenerator, create_scale_free_network_manager
from Complete_NTEO.topology.network_topology import SydneyNetworkTopology

def test_scale_free_generation():
    """Test that scale-free network is generated without small-world features"""
    
    print("=" * 60)
    print("TESTING SCALE-FREE NETWORK GENERATION")
    print("=" * 60)
    
    # 1. Set configuration explicitly
    print("\n1️⃣ Setting database configuration...")
    db.NETWORK_CONFIG['topology_type'] = 'scale_free'
    db.NETWORK_CONFIG['attachment_parameter'] = 2
    print(f"   Config: {db.NETWORK_CONFIG['topology_type']}")
    
    # 2. Create base network
    print("\n2️⃣ Creating base Sydney network...")
    base_network = SydneyNetworkTopology()
    base_network.initialize_base_sydney_network()
    print(f"   Base network: {len(base_network.nodes)} nodes")
    
    # 3. Generate scale-free network
    print("\n3️⃣ Generating scale-free network...")
    generator = ScaleFreeTopologyGenerator(base_network)
    sf_network = generator.generate_scale_free_network(m_edges=2, alpha=1.0)
    
    print(f"   Nodes: {sf_network.number_of_nodes()}")
    print(f"   Edges: {sf_network.number_of_edges()}")
    
    # 4. Check for small-world indicators
    print("\n4️⃣ Checking for small-world contamination...")
    
    # Check edge types
    edge_types = set()
    shortcut_count = 0
    
    for u, v, data in sf_network.edges(data=True):
        edge_type = data.get('edge_type', 'regular')
        edge_types.add(edge_type)
        if edge_type == 'shortcut':
            shortcut_count += 1
    
    print(f"   Edge types found: {edge_types}")
    print(f"   Shortcut edges: {shortcut_count}")
    
    if shortcut_count > 0:
        print("   ❌ ERROR: Network contains shortcuts! Small-world code is still running!")
    else:
        print("   ✅ SUCCESS: No shortcuts found - pure scale-free network!")
    
    # 5. Verify preferential attachment
    print("\n5️⃣ Verifying scale-free properties...")
    degrees = dict(sf_network.degree())
    degree_dist = {}
    
    for d in degrees.values():
        degree_dist[d] = degree_dist.get(d, 0) + 1
    
    # Find hubs (high-degree nodes)
    high_degree_nodes = [node for node, deg in degrees.items() if deg >= 5]
    print(f"   High-degree hubs (≥5 connections): {len(high_degree_nodes)}")
    
    if high_degree_nodes:
        print("   ✅ Hub structure present - scale-free characteristics confirmed!")
        # Show top 5 hubs
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print("   Top 5 hubs:")
        for node, degree in sorted_nodes:
            print(f"     - {node}: {degree} connections")
    else:
        print("   ⚠️  Warning: No high-degree hubs found")
    
    # 6. Test with MobilityModel
    print("\n6️⃣ Testing with MobilityModel initialization...")
    try:
        from agent_run_visualisation import MobilityModel
        
        # Create minimal config
        test_config = {
            'db_connection_string': db.DB_CONNECTION_STRING,
            'num_commuters': 5,
            'data_income_weights': db.income_weights,
            'data_health_weights': db.health_weights,
            'data_payment_weights': db.payment_weights,
            'data_age_distribution': db.age_distribution,
            'data_disability_weights': db.disability_weights,
            'data_tech_access_weights': db.tech_access_weights,
            'subsidy_config': db.monthly_config,
            'network_config': db.NETWORK_CONFIG,
            # Add all other required parameters with defaults
            'ASC_VALUES': db.ASC_VALUES,
            'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': db.UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
            'UTILITY_FUNCTION_BASE_COEFFICIENTS': db.UTILITY_FUNCTION_BASE_COEFFICIENTS,
            'PENALTY_COEFFICIENTS': db.PENALTY_COEFFICIENTS,
            'AFFORDABILITY_THRESHOLDS': db.AFFORDABILITY_THRESHOLDS,
            'FLEXIBILITY_ADJUSTMENTS': db.FLEXIBILITY_ADJUSTMENTS,
            'VALUE_OF_TIME': db.VALUE_OF_TIME,
            'public_price_table': db.public_price_table,
            'ALPHA_VALUES': db.ALPHA_VALUES,
            'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': db.DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
            'BACKGROUND_TRAFFIC_AMOUNT': db.BACKGROUND_TRAFFIC_AMOUNT,
            'CONGESTION_ALPHA': db.CONGESTION_ALPHA,
            'CONGESTION_BETA': db.CONGESTION_BETA,
            'CONGESTION_CAPACITY': db.CONGESTION_CAPACITY,
            'CONGESTION_T_IJ_FREE_FLOW': db.CONGESTION_T_IJ_FREE_FLOW,
            'uber_like1_capacity': db.UberLike1_capacity,
            'uber_like1_price': db.UberLike1_price,
            'uber_like2_capacity': db.UberLike2_capacity,
            'uber_like2_price': db.UberLike2_price,
            'bike_share1_capacity': db.BikeShare1_capacity,
            'bike_share1_price': db.BikeShare1_price,
            'bike_share2_capacity': db.BikeShare2_capacity,
            'bike_share2_price': db.BikeShare2_price,
            'subsidy_dataset': db.subsidy_dataset,
        }
        
        # Watch for output about network type
        model = MobilityModel(**test_config)
        
        # Check final network
        final_network = model.network_manager.active_network
        print(f"   Final network in model: {final_network.number_of_nodes()} nodes, "
              f"{final_network.number_of_edges()} edges")
        
        # Check for shortcuts again
        final_shortcuts = sum(1 for u, v, d in final_network.edges(data=True) 
                            if d.get('edge_type') == 'shortcut')
        
        if final_shortcuts > 0:
            print(f"   ❌ ERROR: Final network has {final_shortcuts} shortcuts!")
        else:
            print("   ✅ SUCCESS: Model initialized with pure scale-free network!")
            
    except Exception as e:
        print(f"   ⚠️  Model initialization error: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_scale_free_generation()