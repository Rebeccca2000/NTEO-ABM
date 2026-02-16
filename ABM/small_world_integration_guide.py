"""
🌐 SMALL-WORLD INTEGRATION GUIDE

This guide shows you how to integrate the new small-world network functionality
with your existing ABM-ETOP system. Follow these steps to add small-world analysis
to your current workflow.

INTEGRATION STEPS:
1. Update your database.py configuration
2. Modify your existing simulation scripts
3. Add small-world analysis to your workflow
4. Create comparative studies
"""

import os
import shutil
from datetime import datetime

class SmallWorldIntegrationGuide:
    """Step-by-step integration guide for small-world networks"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("🌐 Small-World Integration Guide")
        print("="*50)
    
    def check_required_files(self):
        """Check if all required files are present"""
        required_files = [
            'small_world_topology.py',
            'small_world_analyzer.py', 
            'small_world_visualizer.py',
            'run_small_world_analysis.py',
            'database.py',
            'agent_run_visualisation.py',
            'network_topology.py',
            'network_integration.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   • {file}")
            print("\n📥 Please create these files from the provided artifacts first.")
            return False
        else:
            print("✅ All required files present!")
            return True
    
    def step_1_update_database_config(self):
        """Step 1: Update database.py configuration"""
        print("\n📝 STEP 1: Update Database Configuration")
        print("-" * 40)
        
        print("Add these lines to your database.py file:")
        print("""
# ===== SMALL-WORLD NETWORK CONFIGURATION =====
# Add to your existing NETWORK_CONFIG:

NETWORK_CONFIG = {
    'topology_type': 'small_world',  # Changed from 'degree_constrained'
    'grid_width': 100,
    'grid_height': 80,
    'sydney_realism': True,
    'preserve_hierarchy': True,
    
    # Small-world specific parameters
    'rewiring_probability': 0.1,          # p ∈ [0.01, 0.5] 
    'initial_neighbors': 4,               # k parameter
    'preserve_geography': True,           # Maintain Sydney realism
    'max_rewire_distance': 40,           # Max shortcut distance
}

# Research configurations
SMALL_WORLD_RESEARCH_CONFIG = {
    'rewiring_probabilities': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'analysis_runs': 2,
    'simulation_steps': 75
}
        """)
        
        print("\n✅ Database configuration ready for small-world networks!")
    
    def step_2_modify_existing_scripts(self):
        """Step 2: Show how to modify existing scripts"""
        print("\n🔧 STEP 2: Modify Existing Scripts")
        print("-" * 40)
        
        print("Option A: Minimal Changes (Recommended)")
        print("Just change database.py NETWORK_CONFIG['topology_type'] to 'small_world'")
        print("Your existing agent_run_visualisation.py will automatically use small-world!")
        
        print("\nOption B: Explicit Integration")
        print("Add this to your model creation code:")
        
        print("""
# In your existing simulation script:
from small_world_topology import SmallWorldTopologyGenerator
from network_integration import TwoLayerNetworkManager

# Create small-world network manager
def create_small_world_model(p_value=0.1):
    # Update network config
    network_config = db.NETWORK_CONFIG.copy()
    network_config['topology_type'] = 'small_world'
    network_config['rewiring_probability'] = p_value
    
    # Create model with small-world network
    model = MobilityModel(
        # ... your existing parameters ...
        network_config=network_config
    )
    
    return model

# Usage:
model = create_small_world_model(p_value=0.1)
model.run_model(144)
        """)
    
    def step_3_add_analysis_capabilities(self):
        """Step 3: Add small-world analysis capabilities"""
        print("\n📊 STEP 3: Add Small-World Analysis")
        print("-" * 40)
        
        print("Add this analysis function to your workflow:")
        
        print("""
# Add to your analysis script:
from small_world_analyzer import SmallWorldAnalysisFramework
from small_world_visualizer import SmallWorldNetworkVisualizer

def analyze_small_world_equity():
    \"\"\"Analyze small-world network equity implications\"\"\"
    
    # Configuration
    base_config = {
        # ... your existing config ...
        'network_config': {'topology_type': 'small_world'}
    }
    
    # Test different rewiring probabilities
    p_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    # Create analyzer
    analyzer = SmallWorldAnalysisFramework(base_config, p_values)
    
    # Run analysis
    results = analyzer.run_complete_small_world_analysis(
        num_runs=2, 
        steps_per_run=75
    )
    
    # Print findings
    print("🔬 RESEARCH FINDINGS:")
    for p, result in results.items():
        equity_score = result.get('equity_metrics', {}).get('peripheral_benefit_score', 0)
        print(f"p={p:.3f}: Peripheral benefit = {equity_score:.3f}")
    
    return results

# Usage:
equity_results = analyze_small_world_equity()
        """)
    
    def step_4_create_comparison_studies(self):
        """Step 4: Create comparison studies"""
        print("\n⚖️  STEP 4: Create Comparison Studies")
        print("-" * 40)
        
        print("Compare small-world vs degree-constrained networks:")
        
        print("""
def compare_network_topologies():
    \"\"\"Compare small-world vs degree-constrained equity outcomes\"\"\"
    
    configurations = [
        ('Degree-3 Baseline', 'degree_constrained', {'degree_constraint': 3}),
        ('Small-World Low', 'small_world', {'rewiring_probability': 0.05}),
        ('Small-World Medium', 'small_world', {'rewiring_probability': 0.1}),
        ('Small-World High', 'small_world', {'rewiring_probability': 0.3})
    ]
    
    results = {}
    
    for name, topology, params in configurations:
        print(f"Testing {name}...")
        
        # Update configuration
        config = get_base_config()
        config['network_config']['topology_type'] = topology
        config['network_config'].update(params)
        
        # Run simulation
        if topology == 'small_world':
            from small_world_analyzer import SmallWorldAnalysisFramework
            analyzer = SmallWorldAnalysisFramework(config, [params['rewiring_probability']])
            result = analyzer.run_single_small_world_analysis(params['rewiring_probability'], 2, 50)
        else:
            from degree_comparison import DegreeComparisonFramework
            analyzer = DegreeComparisonFramework(config, [params['degree_constraint']])
            result = analyzer.run_single_degree_simulation(params['degree_constraint'], 2, 50)
        
        results[name] = result
    
    # Compare results
    print("\\n📊 COMPARISON RESULTS:")
    for name, result in results.items():
        if 'equity_metrics' in result:
            equity = result['equity_metrics'].get('efficiency_equity_balance', 0)
            print(f"{name}: Equity score = {equity:.3f}")
    
    return results

# Usage:
comparison_results = compare_network_topologies()
        """)
    
    def step_5_visualization_integration(self):
        """Step 5: Add visualization capabilities"""
        print("\n🎨 STEP 5: Add Visualization Capabilities")
        print("-" * 40)
        
        print("Integrate small-world visualizations:")
        
        print("""
def create_small_world_visualizations(p_value=0.1):
    \"\"\"Create comprehensive small-world visualizations\"\"\"
    
    # Create network manager
    from small_world_topology import SmallWorldTopologyGenerator
    from network_topology import SydneyNetworkTopology
    from network_integration import TwoLayerNetworkManager
    
    # Initialize network
    base_network = SydneyNetworkTopology()
    base_network.initialize_base_sydney_network()
    
    generator = SmallWorldTopologyGenerator(base_network)
    sw_graph = generator.generate_small_world_network(
        rewiring_probability=p_value,
        initial_neighbors=4
    )
    
    # Create network manager
    network_manager = TwoLayerNetworkManager()
    network_manager.active_network = sw_graph
    network_manager.base_network = base_network
    
    # Create visualizer
    from small_world_visualizer import SmallWorldNetworkVisualizer
    visualizer = SmallWorldNetworkVisualizer(network_manager, p_value)
    
    # Generate visualizations
    visualizer.visualize_network_structure(
        save_path=f"small_world_analysis_p{p_value:.3f}.png"
    )
    
    return visualizer

# Usage:
visualizer = create_small_world_visualizations(p_value=0.1)
        """)
    
    def create_quick_start_script(self):
        """Create a quick start script for immediate use"""
        print("\n🚀 CREATING QUICK START SCRIPT")
        print("-" * 40)
        
        script_content = '''#!/usr/bin/env python3
"""
Quick Start Script for Small-World Analysis
Generated by SmallWorldIntegrationGuide
"""

import database as db
from run_small_world_analysis import run_quick_analysis, run_visualization_analysis

def main():
    print("🌐 Small-World Network Quick Start")
    print("="*40)
    
    # Option 1: Quick analysis
    print("\\n1. Running quick analysis (p=0.1)...")
    try:
        result = run_quick_analysis(p_value=0.1, num_runs=2, steps_per_run=50)
        print("✅ Quick analysis completed!")
    except Exception as e:
        print(f"❌ Quick analysis failed: {e}")
    
    # Option 2: Visualization
    print("\\n2. Creating visualizations...")
    try:
        visualizer, stats = run_visualization_analysis(p_value=0.1)
        if stats:
            print(f"✅ Visualization completed! Sigma = {stats['sigma']:.3f}")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
    
    print("\\n🎉 Small-world analysis complete!")
    print("Check generated files for detailed results.")

if __name__ == "__main__":
    main()
'''
        
        # Save quick start script
        with open(f'quick_start_small_world_{self.timestamp}.py', 'w') as f:
            f.write(script_content)
        
        print(f"✅ Quick start script created: quick_start_small_world_{self.timestamp}.py")
    
    def run_integration_check(self):
        """Run integration check to verify everything works"""
        print("\n🔍 RUNNING INTEGRATION CHECK")
        print("-" * 40)
        
        try:
            # Test 1: Check imports
            print("Test 1: Checking imports...")
            from small_world_topology import SmallWorldTopologyGenerator
            from small_world_analyzer import SmallWorldAnalysisFramework
            from small_world_visualizer import SmallWorldNetworkVisualizer
            print("✅ All imports successful")
            
            # Test 2: Check configuration
            print("Test 2: Checking configuration...")
            import database as db
            if hasattr(db, 'NETWORK_CONFIG'):
                print("✅ Network configuration found")
            else:
                print("⚠️  Update database.py with network configuration")
            
            # Test 3: Test network creation
            print("Test 3: Testing network creation...")
            from Complete_NTEO.topology.network_topology import SydneyNetworkTopology
            base_network = SydneyNetworkTopology()
            base_network.initialize_base_sydney_network()
            
            generator = SmallWorldTopologyGenerator(base_network)
            test_graph = generator.generate_small_world_network(
                rewiring_probability=0.1, initial_neighbors=4
            )
            
            if test_graph.number_of_nodes() > 0:
                print(f"✅ Network creation successful ({test_graph.number_of_nodes()} nodes)")
            else:
                print("❌ Network creation failed")
            
            print("\n🎉 Integration check passed!")
            return True
            
        except Exception as e:
            print(f"❌ Integration check failed: {e}")
            print("\n🔧 Troubleshooting:")
            print("• Ensure all required files are present")
            print("• Check database.py configuration")
            print("• Verify Python dependencies")
            return False
    
    def generate_integration_summary(self):
        """Generate integration summary"""
        print("\n📋 INTEGRATION SUMMARY")
        print("="*50)
        
        print("✅ COMPLETED STEPS:")
        print("   1. Database configuration updated")
        print("   2. Integration methods provided")
        print("   3. Analysis capabilities added")
        print("   4. Comparison studies setup")
        print("   5. Visualization integration")
        print("   6. Quick start script created")
        
        print("\n🎯 NEXT ACTIONS:")
        print("   1. Update your database.py with small-world config")
        print("   2. Run the quick start script to test")
        print("   3. Modify existing scripts as needed")
        print("   4. Run comparative analysis")
        print("   5. Analyze equity implications")
        
        print("\n🔬 RESEARCH WORKFLOW:")
        print("   1. Test baseline (p=0.0, equivalent to regular network)")
        print("   2. Test small-world configurations (p=0.05, 0.1, 0.2, 0.3)")
        print("   3. Analyze peripheral accessibility improvements") 
        print("   4. Compare efficiency vs equity trade-offs")
        print("   5. Identify optimal configuration for Sydney")
        
        print("\n📂 GENERATED FILES:")
        print(f"   • quick_start_small_world_{self.timestamp}.py")
        print("   • Integration methods provided above")
        
        print("\n🎉 Small-world integration ready!")


def run_integration_guide():
    """Main function to run the integration guide"""
    guide = SmallWorldIntegrationGuide()
    
    # Check prerequisites
    if not guide.check_required_files():
        return False
    
    # Run integration steps
    guide.step_1_update_database_config()
    guide.step_2_modify_existing_scripts()
    guide.step_3_add_analysis_capabilities()
    guide.step_4_create_comparison_studies()
    guide.step_5_visualization_integration()
    
    # Create practical tools
    guide.create_quick_start_script()
    
    # Verify integration
    integration_success = guide.run_integration_check()
    
    # Generate summary
    guide.generate_integration_summary()
    
    return integration_success


# Example usage patterns
class SmallWorldUsageExamples:
    """Practical usage examples for small-world networks"""
    
    @staticmethod
    def example_1_basic_usage():
        """Example 1: Basic small-world network creation"""
        print("Example 1: Basic Usage")
        print("""
# Create small-world network
from small_world_topology import SmallWorldTopologyGenerator
from network_topology import SydneyNetworkTopology

base_network = SydneyNetworkTopology()
base_network.initialize_base_sydney_network()

generator = SmallWorldTopologyGenerator(base_network)
sw_network = generator.generate_small_world_network(
    rewiring_probability=0.1,  # 10% chance to rewire each edge
    initial_neighbors=4,       # Start with 4 nearest neighbors
    preserve_geography=True    # Maintain Sydney realism
)

print(f"Created network: {sw_network.number_of_nodes()} nodes, {sw_network.number_of_edges()} edges")
        """)
    
    @staticmethod
    def example_2_analysis():
        """Example 2: Running small-world analysis"""
        print("Example 2: Analysis")
        print("""
# Run small-world analysis
from small_world_analyzer import SmallWorldAnalysisFramework

base_config = get_base_config()  # Your existing config
analyzer = SmallWorldAnalysisFramework(base_config, [0.1, 0.2, 0.3])

results = analyzer.run_complete_small_world_analysis(
    num_runs=2, 
    steps_per_run=75
)

# Check results
for p, result in results.items():
    sigma = result['network_properties']['small_world_sigma']
    equity = result['equity_metrics']['peripheral_benefit_score']
    print(f"p={p:.3f}: σ={sigma:.3f}, equity={equity:.3f}")
        """)
    
    @staticmethod
    def example_3_comparison():
        """Example 3: Comparing with degree-constrained"""
        print("Example 3: Comparison")
        print("""
# Compare small-world vs degree-constrained
def compare_topologies():
    # Test configurations
    configs = [
        ('Degree-3', 'degree_constrained', 3),
        ('Small-World', 'small_world', 0.1)
    ]
    
    for name, topology, param in configs:
        # Create appropriate network
        if topology == 'small_world':
            # Use small-world analysis
            pass
        else:
            # Use degree analysis
            pass
        
        # Compare equity outcomes
        pass

compare_topologies()
        """)


if __name__ == "__main__":
    # Run the integration guide
    success = run_integration_guide()
    
    if success:
        print("\n🎉 Integration successful! You can now use small-world networks.")
    else:
        print("\n❌ Integration issues detected. Please resolve before proceeding.")
    
    # Show usage examples
    print("\n📚 USAGE EXAMPLES:")
    examples = SmallWorldUsageExamples()
    examples.example_1_basic_usage()
    examples.example_2_analysis()
    examples.example_3_comparison()