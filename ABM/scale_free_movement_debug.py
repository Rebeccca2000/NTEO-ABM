#!/usr/bin/env python3
"""
🌐 SCALE-FREE NETWORK MOVEMENT TEST
Test if commuters can move properly through the scale-free network topology
"""

import sys
import uuid
import random
import networkx as nx

# Core imports
try:
    from Complete_NTEO.topology.scale_free_topology import create_scale_free_network_manager
    import database as db
    from agent_run_visualisation import MobilityModel
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ScaleFreeMovementTester:
    """Test commuter movement through scale-free network"""
    
    def __init__(self):
        # Ensure scale-free configuration
        self.scale_free_config = {
            'topology_type': 'scale_free',
            'attachment_parameter': 2,
            'preferential_strength': 1.0,
            'grid_width': 100,
            'grid_height': 80,
            'preserve_geography': True,
            'max_attachment_distance': 50
        }
        db.NETWORK_CONFIG.update(self.scale_free_config)
        
    def test_scale_free_network_properties(self):
        """Test basic scale-free network properties"""
        print("\n🌐 TESTING: Scale-Free Network Properties")
        print("=" * 60)
        
        # Create model
        model = self._create_test_model()
        network = model.network_manager.active_network
        
        print(f"📊 Network Analysis:")
        print(f"   Nodes: {network.number_of_nodes()}")
        print(f"   Edges: {network.number_of_edges()}")
        print(f"   Connected: {nx.is_connected(network)}")
        print(f"   Average degree: {sum(dict(network.degree()).values()) / network.number_of_nodes():.2f}")
        
        # Check for hubs (scale-free characteristic)
        degrees = dict(network.degree())
        high_degree_nodes = [node for node, deg in degrees.items() if deg >= 8]
        print(f"   High-degree hubs (≥8 connections): {len(high_degree_nodes)}")
        
        if high_degree_nodes:
            print(f"   ✅ Scale-free hub structure detected")
            top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            for hub, degree in top_hubs:
                print(f"      {hub}: {degree} connections")
        else:
            print(f"   ⚠️ No clear hub structure detected")
        
        return model
    
    def test_network_accessibility(self, model):
        """Test that all network locations are accessible"""
        print("\n🗺️ TESTING: Network Accessibility")
        print("=" * 60)
        
        network = model.network_manager.active_network
        spatial_mapper = model.network_manager.spatial_mapper
        
        print(f"📊 Accessibility Analysis:")
        print(f"   Network nodes: {network.number_of_nodes()}")
        print(f"   Spatial locations: {len(spatial_mapper.node_to_grid)}")
        print(f"   Grid mappings: {len(spatial_mapper.grid_to_node)}")
        
        # Test random route finding
        nodes = list(network.nodes())
        test_routes = 0
        successful_routes = 0
        
        print(f"\n🧪 Testing route accessibility:")
        for _ in range(10):  # Test 10 random routes
            start = random.choice(nodes)
            end = random.choice(nodes)
            
            if start != end:
                test_routes += 1
                try:
                    path = nx.shortest_path(network, start, end)
                    successful_routes += 1
                    print(f"   ✅ {start} → {end}: {len(path)} hops")
                except nx.NetworkXNoPath:
                    print(f"   ❌ {start} → {end}: NO PATH")
        
        success_rate = successful_routes / test_routes if test_routes > 0 else 0
        print(f"   Route success rate: {success_rate:.1%}")
        
        if success_rate > 0.9:
            print(f"   ✅ Network is well-connected for routing")
        else:
            print(f"   ❌ Network has connectivity issues")
    
    def test_commuter_placement(self, model):
        """Test commuter placement on scale-free network"""
        print("\n🚶 TESTING: Commuter Placement")
        print("=" * 60)
        
        print(f"📊 Commuter Analysis:")
        print(f"   Total commuters: {len(model.commuter_agents)}")
        
        # Check commuter locations
        placed_commuters = 0
        network_locations = list(model.network_manager.spatial_mapper.node_to_grid.values())
        
        for i, commuter in enumerate(model.commuter_agents[:5]):  # Check first 5
            if hasattr(commuter, 'location'):
                location = commuter.location
                placed_commuters += 1
                
                # Check if location is network-accessible
                is_network_accessible = location in network_locations
                print(f"   Commuter {i}: {location} {'✅' if is_network_accessible else '❌'}")
                
                if not is_network_accessible:
                    print(f"      ⚠️ Location not in network!")
        
        print(f"   Commuters with locations: {placed_commuters}/{len(model.commuter_agents)}")
        
        if placed_commuters == len(model.commuter_agents):
            print(f"   ✅ All commuters properly placed")
        else:
            print(f"   ❌ Some commuters missing locations")
    
    def test_manual_travel_requests(self, model):
        """Test manual creation of travel requests through scale-free network"""
        print("\n🎯 TESTING: Manual Travel Requests")
        print("=" * 60)
        
        if not model.commuter_agents:
            print("   ❌ No commuters available for testing")
            return
        
        # Get network locations for testing
        network_locations = list(model.network_manager.spatial_mapper.node_to_grid.values())
        
        if len(network_locations) < 2:
            print("   ❌ Not enough network locations for testing")
            return
        
        # Test commuter
        test_commuter = model.commuter_agents[0]
        print(f"📊 Testing with commuter {test_commuter.unique_id}")
        
        # Create test requests manually
        test_requests = []
        for i in range(3):  # Test 3 different routes
            origin = random.choice(network_locations)
            destination = random.choice(network_locations)
            
            if origin != destination:
                request_id = uuid.uuid4()
                start_time = model.current_step + i + 1
                
                print(f"   Creating request {i+1}: {origin} → {destination}")
                
                try:
                    test_commuter.create_request(request_id, origin, destination, start_time, 'test')
                    test_requests.append(request_id)
                    print(f"   ✅ Request created successfully")
                except Exception as e:
                    print(f"   ❌ Request creation failed: {e}")
        
        # Check if requests were added
        if hasattr(test_commuter, 'requests'):
            print(f"   Total requests after creation: {len(test_commuter.requests)}")
            
            # Show request details
            for req_id, request in test_commuter.requests.items():
                if req_id in test_requests:
                    print(f"      Request: {request.get('origin')} → {request.get('destination')}")
        else:
            print(f"   ❌ Commuter has no requests attribute")
    
    def test_maas_routing(self, model):
        """Test MaaS routing through scale-free network"""
        print("\n🤖 TESTING: MaaS Routing Through Scale-Free Network")
        print("=" * 60)
        
        if not hasattr(model, 'maas_agent'):
            print("   ❌ No MaaS agent available")
            return
        
        maas = model.maas_agent
        network_locations = list(model.network_manager.spatial_mapper.node_to_grid.values())
        
        if len(network_locations) < 2:
            print("   ❌ Not enough locations for routing test")
            return
        
        # Test routing between random locations
        print(f"📊 Testing MaaS routing:")
        
        for i in range(5):  # Test 5 routes
            origin = random.choice(network_locations)
            destination = random.choice(network_locations)
            
            if origin != destination:
                print(f"   Test route {i+1}: {origin} → {destination}")
                
                try:
                    # Test options without MaaS
                    request_id = uuid.uuid4()
                    start_time = model.current_step + 1
                    
                    options_without_maas = maas.options_without_maas(
                        request_id, start_time, origin, destination
                    )
                    
                    print(f"      Options without MaaS: {len(options_without_maas) if options_without_maas else 0}")
                    
                    # Test MaaS options
                    options_with_maas = maas.maas_options(
                        'credit', request_id, start_time, origin, destination
                    )
                    
                    print(f"      Options with MaaS: {len(options_with_maas) if options_with_maas else 0}")
                    
                    if options_without_maas or options_with_maas:
                        print(f"      ✅ Routing successful")
                    else:
                        print(f"      ❌ No routing options found")
                        
                except Exception as e:
                    print(f"      ❌ Routing failed: {e}")
    
    def test_commuter_movement_simulation(self, model):
        """Test actual commuter movement through scale-free network"""
        print("\n🎬 TESTING: Commuter Movement Simulation")
        print("=" * 60)
        
        # Record initial state
        initial_positions = {}
        initial_requests = {}
        
        for i, commuter in enumerate(model.commuter_agents):
            initial_positions[i] = getattr(commuter, 'location', None)
            initial_requests[i] = len(getattr(commuter, 'requests', {}))
        
        print(f"📊 Initial state:")
        print(f"   Commuters tracked: {len(initial_positions)}")
        print(f"   Total initial requests: {sum(initial_requests.values())}")
        
        # Force creation of some travel requests
        print(f"\n🎯 Forcing travel request creation:")
        created_requests = 0
        
        for commuter in model.commuter_agents[:3]:  # Test first 3 commuters
            try:
                result = model.create_time_based_trip(model.current_step, commuter)
                if result:
                    created_requests += 1
                    print(f"   ✅ Request created for commuter {commuter.unique_id}")
            except Exception as e:
                print(f"   ❌ Request creation failed for commuter {commuter.unique_id}: {e}")
        
        print(f"   Total requests created: {created_requests}")
        
        # Run simulation steps
        print(f"\n🎬 Running simulation steps:")
        
        movements_detected = 0
        requests_changes = 0
        
        for step in range(5):  # Run 5 steps
            try:
                print(f"   Step {step + 1}:")
                
                # Record pre-step state
                pre_step_positions = {}
                pre_step_requests = {}
                
                for i, commuter in enumerate(model.commuter_agents):
                    pre_step_positions[i] = getattr(commuter, 'location', None)
                    pre_step_requests[i] = len(getattr(commuter, 'requests', {}))
                
                # Execute step
                model.step()
                
                # Check for changes
                step_movements = 0
                step_request_changes = 0
                
                for i, commuter in enumerate(model.commuter_agents):
                    post_position = getattr(commuter, 'location', None)
                    post_requests = len(getattr(commuter, 'requests', {}))
                    
                    if pre_step_positions[i] != post_position:
                        step_movements += 1
                    
                    if pre_step_requests[i] != post_requests:
                        step_request_changes += 1
                
                movements_detected += step_movements
                requests_changes += step_request_changes
                
                print(f"      Movements: {step_movements}, Request changes: {step_request_changes}")
                
            except Exception as e:
                print(f"      ❌ Step {step + 1} failed: {e}")
        
        # Final analysis
        print(f"\n📊 Movement Analysis:")
        print(f"   Total movements detected: {movements_detected}")
        print(f"   Total request changes: {requests_changes}")
        
        if movements_detected > 0:
            print(f"   ✅ Commuters are moving through scale-free network!")
        else:
            print(f"   ❌ No movement detected - need to investigate further")
        
        if requests_changes > 0:
            print(f"   ✅ Travel requests are being processed")
        else:
            print(f"   ⚠️ No request processing detected")
    
    def _create_test_model(self):
        """Create test model with scale-free network"""
        return MobilityModel(
            db_connection_string=db.DB_CONNECTION_STRING,
            num_commuters=10,  # More commuters for better testing
            data_income_weights=db.income_weights,
            data_health_weights=db.health_weights,
            data_payment_weights=db.payment_weights,
            data_age_distribution=db.age_distribution,
            data_disability_weights=db.disability_weights,
            data_tech_access_weights=db.tech_access_weights,
            subsidy_config=db.monthly_config,
            network_config=self.scale_free_config,
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

def main():
    """Main testing execution"""
    print("🌐 SCALE-FREE NETWORK MOVEMENT TEST")
    print("="*70)
    print("Testing if commuters can move properly through scale-free network topology")
    print()
    
    tester = ScaleFreeMovementTester()
    
    # Test scale-free network integration with ABM
    model = tester.test_scale_free_network_properties()
    tester.test_network_accessibility(model)
    tester.test_commuter_placement(model)
    tester.test_manual_travel_requests(model)
    tester.test_maas_routing(model)
    tester.test_commuter_movement_simulation(model)
    
    print(f"\n🎯 SCALE-FREE MOVEMENT TEST COMPLETE")
    print("This tests YOUR scale-free network topology with YOUR existing ABM!")

if __name__ == "__main__":
    main()