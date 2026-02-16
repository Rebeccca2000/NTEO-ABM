#!/usr/bin/env python3
"""
Debug Small-World Network Generation
Quick diagnostic script to identify and fix the small-world generation issues
"""

import database as db
from Complete_NTEO.topology.network_topology import SydneyNetworkTopology
import traceback

def debug_sydney_network():
    """Debug SydneyNetworkTopology object structure"""
    print("🔍 DEBUGGING SYDNEY NETWORK TOPOLOGY")
    print("="*60)
    
    # Create base network
    base_network = SydneyNetworkTopology()
    base_network.initialize_base_sydney_network()
    
    print("Base network attributes:")
    for attr in dir(base_network):
        if not attr.startswith('_'):
            try:
                value = getattr(base_network, attr)
                if not callable(value):
                    print(f"  {attr}: {type(value)} = {value if len(str(value)) < 100 else str(value)[:100] + '...'}")
            except:
                print(f"  {attr}: <error accessing>")
    
    print(f"\nNetwork nodes: {base_network.graph.number_of_nodes()}")
    print(f"Network edges: {base_network.graph.number_of_edges()}")
    
    # Check node structure
    if base_network.graph.number_of_nodes() > 0:
        sample_node = list(base_network.graph.nodes())[0]
        print(f"\nSample node: {sample_node}")
        print(f"Node data: {base_network.nodes[sample_node]}")
    
    return base_network

def debug_small_world_generation():
    """Debug small-world generation process step by step"""
    print("\n🌐 DEBUGGING SMALL-WORLD GENERATION")
    print("="*60)
    
    try:
        # Step 1: Import generator
        print("Step 1: Importing SmallWorldTopologyGenerator...")
        from small_world_topology import SmallWorldTopologyGenerator
        print("✅ Import successful")
        
        # Step 2: Create base network
        print("\nStep 2: Creating base network...")
        base_network = debug_sydney_network()
        print("✅ Base network created")
        
        # Step 3: Initialize generator
        print("\nStep 3: Initializing generator...")
        generator = SmallWorldTopologyGenerator(base_network)
        print("✅ Generator initialized")
        
        # Step 4: Check generator attributes
        print("\nStep 4: Checking generator state...")
        print(f"  Base nodes: {len(generator.base_nodes) if hasattr(generator, 'base_nodes') else 'missing'}")
        print(f"  Max rewire distance: {generator.max_rewire_distance if hasattr(generator, 'max_rewire_distance') else 'missing'}")
        
        # Step 5: Try manual generation with detailed logging
        print("\nStep 5: Attempting small-world generation...")
        
        # Monkey patch to add debug logging
        original_method = generator.generate_small_world_network
        
        def debug_generate_small_world_network(*args, **kwargs):
            print(f"  Calling generate_small_world_network with args={args}, kwargs={kwargs}")
            try:
                result = original_method(*args, **kwargs)
                print(f"  Generation successful: {result.number_of_nodes()} nodes, {result.number_of_edges()} edges")
                
                # Check edge types
                shortcut_count = 0
                regular_count = 0
                for u, v, data in result.edges(data=True):
                    edge_type = data.get('edge_type', 'unknown')
                    if edge_type == 'shortcut':
                        shortcut_count += 1
                    elif edge_type == 'regular':
                        regular_count += 1
                
                print(f"  Edge analysis: {shortcut_count} shortcuts, {regular_count} regular, {result.number_of_edges() - shortcut_count - regular_count} unclassified")
                return result
                
            except Exception as e:
                print(f"  ❌ Generation failed: {e}")
                print(f"  Full traceback:")
                traceback.print_exc()
                raise
        
        generator.generate_small_world_network = debug_generate_small_world_network
        
        # Generate with default parameters
        small_world_graph = generator.generate_small_world_network(
            rewiring_probability=0.1,
            initial_neighbors=4,
            preserve_geography=True
        )
        
        print("✅ Small-world generation successful!")
        return small_world_graph
        
    except Exception as e:
        print(f"❌ Small-world generation failed: {e}")
        print("\nFull error traceback:")
        traceback.print_exc()
        return None

def debug_network_manager():
    """Debug network manager integration"""
    print("\n🔧 DEBUGGING NETWORK MANAGER INTEGRATION")
    print("="*60)
    
    try:
        from network_integration import TwoLayerNetworkManager
        
        # Test creating network manager with small-world config
        network_manager = TwoLayerNetworkManager(
            topology_type='small_world',
            grid_width=db.NETWORK_CONFIG['grid_width'],
            grid_height=db.NETWORK_CONFIG['grid_height']
        )
        
        print(f"✅ Network manager created")
        print(f"  Active network: {network_manager.active_network.number_of_nodes()} nodes")
        print(f"  Active network: {network_manager.active_network.number_of_edges()} edges")
        
        # Check edge types in active network
        shortcut_count = 0
        regular_count = 0
        unclassified_count = 0
        
        for u, v, data in network_manager.active_network.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            if edge_type == 'shortcut':
                shortcut_count += 1
            elif edge_type == 'regular':
                regular_count += 1
            else:
                unclassified_count += 1
        
        print(f"  Edge analysis: {shortcut_count} shortcuts, {regular_count} regular, {unclassified_count} unclassified")
        
        if shortcut_count == 0:
            print("❌ CRITICAL: No shortcuts found in network manager!")
            print("🔧 This confirms small-world generation is failing in the network manager")
        
        return network_manager
        
    except Exception as e:
        print(f"❌ Network manager creation failed: {e}")
        traceback.print_exc()
        return None

def propose_fixes():
    """Propose specific fixes based on debug results"""
    print("\n🛠️ PROPOSED FIXES")
    print("="*60)
    
    print("Based on debug results, try these fixes in order:")
    print()
    print("1. FIX INTERFACE MISMATCH:")
    print("   - Check if SydneyNetworkTopology has .nodes vs .stations")
    print("   - Update SmallWorldTopologyGenerator to use correct attribute")
    print()
    print("2. FIX SILENT FAILURE:")
    print("   - Add proper error handling in SmallWorldTopologyGenerator")
    print("   - Ensure failures propagate to user instead of falling back silently")
    print()
    print("3. FIX EDGE LABELING:")
    print("   - Ensure rewired edges are marked with edge_type='shortcut'")
    print("   - Original edges should be marked with edge_type='regular'")
    print()
    print("4. TEST STEP-BY-STEP:")
    print("   - Test base network creation")
    print("   - Test regular network generation")
    print("   - Test rewiring process")
    print("   - Test final small-world properties")

if __name__ == "__main__":
    print("🔍 SMALL-WORLD DEBUG DIAGNOSTIC")
    print("="*60)
    
    # Run all debug tests
    base_network = debug_sydney_network()
    small_world_network = debug_small_world_generation()
    network_manager = debug_network_manager()
    
    # Propose fixes
    propose_fixes()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE - Review output above for specific issues")
    print("="*60)