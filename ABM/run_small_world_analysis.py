#!/usr/bin/env python3
"""
🌐 SMALL-WORLD NETWORK ANALYSIS - MAIN EXECUTION SCRIPT

This script implements Step 1 of your NTEO proposal:
- Small-World Network Implementation using Watts-Strogatz model
- Geographic realism maintained for Sydney transport networks
- Research question: Do small-world shortcuts improve equity by reducing travel times for peripheral areas?

Usage:
    python run_small_world_analysis.py [options]

Options:
    --quick     : Quick analysis with fewer parameters
    --full      : Full comprehensive analysis  
    --visual    : Focus on visualization
    --compare   : Compare with degree-constrained baseline
"""

import argparse
import sys
import os
import traceback
from datetime import datetime

# Import required modules
import database as db
from small_world_analyzer import SmallWorldAnalysisFramework, run_small_world_analysis
from small_world_visualizer import SmallWorldNetworkVisualizer, create_small_world_visualization
from small_world_topology import SmallWorldTopologyGenerator, SmallWorldParameterOptimizer

def main():
    print("🌐 SMALL-WORLD NETWORK ANALYSIS FOR TRANSPORT EQUITY")
    print("="*70)
    print("NTEO Project - Step 1: Small-World Implementation (Watts-Strogatz)")
    print("Research Question: Do shortcuts improve peripheral area accessibility?")
    print("="*70)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Small-World Network Analysis")
    parser.add_argument('--quick', action='store_true', help='Quick analysis')
    parser.add_argument('--full', action='store_true', help='Full comprehensive analysis')
    parser.add_argument('--visual', action='store_true', help='Visualization focus')
    parser.add_argument('--compare', action='store_true', help='Compare with baseline')
    parser.add_argument('--p', type=float, default=0.2, help='Rewiring probability (default: 0.2)')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per configuration')
    parser.add_argument('--steps', type=int, default=db.SIMULATION_STEPS, help='Simulation steps per run')
    
    args = parser.parse_args()
    
    # Determine analysis type
    if not any([args.quick, args.full, args.visual, args.compare]):
        # Default: ask user
        print("\nChoose analysis type:")
        print("1. Quick Analysis (test p=0.1 only)")
        print("2. Full Analysis (test multiple p values)")
        print("3. Visualization Focus (create detailed visualizations)")
        print("4. Comparison Study (small-world vs degree-constrained)")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            args.quick = True
        elif choice == '2':
            args.full = True
        elif choice == '3':
            args.visual = True
        elif choice == '4':
            args.compare = True
        else:
            print("Invalid choice. Running quick analysis.")
            args.quick = True
    
    try:
        # Execute selected analysis
        if args.quick:
            run_quick_analysis(args.p, args.runs, args.steps)
        elif args.full:
            run_full_analysis(args.runs, args.steps)
        elif args.visual:
            run_visualization_analysis(args.p)
        elif args.compare:
            run_comparison_analysis(args.runs, args.steps)
        
        print("\n✅ Analysis completed successfully!")
        print("\n📋 Next Steps:")
        print("• Review generated visualizations and reports")
        print("• Examine equity implications for peripheral areas")
        print("• Proceed to Scale-Free network implementation (Step 2)")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        print("• Check database connection in database.py")
        print("• Ensure all required files are present")
        print("• Try reducing --runs or --steps parameters")
        print("\nFull error traceback:")
        traceback.print_exc()
        return 1
    
    return 0

def run_quick_analysis(p_value=0.1, num_runs=1, steps_per_run=db.SIMULATION_STEPS):
    """Quick analysis for single p-value (recommended for initial testing)"""
    print(f"\n🚀 RUNNING QUICK SMALL-WORLD ANALYSIS")
    print(f"Parameters: p={p_value:.3f}, runs={num_runs}, steps={steps_per_run}")
    print("-" * 50)
    
    # Prepare configuration
    base_config = get_base_config()
    
    # Create analyzer with single p-value
    analyzer = SmallWorldAnalysisFramework(base_config, [p_value])
    
    # Run analysis
    result = analyzer.run_single_small_world_analysis(
        p_value, num_runs, steps_per_run
    )
    
    # Generate quick report
    print_quick_results(result, p_value)
    
    # Save results
    analyzer.save_results()
    
    # Create basic visualization
    if 'network_properties' in result:
        print("\n📊 Creating visualization...")
        create_quick_visualization(result, p_value)
    
    return result

def run_full_analysis(num_runs=1, steps_per_run=db.SIMULATION_STEPS):
    """Full analysis across multiple p-values (comprehensive study)"""
    print(f"\n🔬 RUNNING COMPREHENSIVE SMALL-WORLD ANALYSIS")
    print(f"Parameters: runs={num_runs}, steps={steps_per_run}")
    print("-" * 50)
    
    # Use standard research configuration
    base_config = get_base_config()
    
    # Define research-focused p-values
    p_values = db.SMALL_WORLD_ANALYSIS_CONFIG['rewiring_probabilities']
    print(f"Testing rewiring probabilities: {p_values}")
    
    # Create analyzer
    analyzer = SmallWorldAnalysisFramework(base_config, p_values)
    
    # Run complete analysis
    results = analyzer.run_complete_small_world_analysis(num_runs, steps_per_run)
    
    # Generate comprehensive findings
    print_comprehensive_findings(results)
    
    return results

def run_visualization_analysis(p_value=0.5):
    """Focus on creating detailed visualizations"""
    print(f"\n🎨 CREATING DETAILED SMALL-WORLD VISUALIZATIONS")
    print(f"Parameters: p={p_value:.3f}")
    print("-" * 50)
    
    try:
        # Create network manager with small-world topology
        network_manager = create_small_world_network_manager(p_value)
        
        # Create visualizer
        visualizer = SmallWorldNetworkVisualizer(network_manager, p_value)
        
        # Generate comprehensive visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"small_world_detailed_p{p_value:.3f}_{timestamp}.png"
        
        print("Creating detailed network visualization...")
        visualizer.visualize_network_structure(save_path=save_path)
        
        # Print network analysis
        print("\n📊 NETWORK ANALYSIS SUMMARY:")
        stats = visualizer._calculate_comprehensive_stats()
        
        print(f"Small-World Coefficient (σ): {stats['sigma']:.3f}")
        print(f"Clustering Coefficient: {stats['clustering']:.3f}")
        print(f"Average Path Length: {stats['avg_path_length']:.2f}")
        print(f"Shortcuts Created: {stats['shortcuts']} ({stats['shortcut_percentage']:.1f}%)")
        print(f"Peripheral Nodes: {stats['peripheral_nodes']} ({stats['peripheral_percentage']:.1f}%)")
        print(f"Accessibility Gini: {stats['accessibility_gini']:.3f}")
        
        # Research implications
        print(f"\n🔬 RESEARCH IMPLICATIONS:")
        if stats['sigma'] > 1:
            print("✅ Network exhibits small-world properties")
        else:
            print("⚠️  Limited small-world effect detected")
        
        if stats['peripheral_benefit'] > 0.1:
            print("✅ Shortcuts appear to benefit peripheral areas")
        else:
            print("⚠️  Limited benefit for peripheral areas")
        
        if stats['hierarchy_preserved']:
            print("✅ Transport hierarchy maintained")
        else:
            print("⚠️  Transport hierarchy may be disrupted")
        
        return visualizer, stats
        
    except Exception as e:
        print(f"Visualization error: {e}")
        traceback.print_exc()
        return None, None

def run_comparison_analysis(num_runs=2, steps_per_run=50):
    """Compare small-world with degree-constrained baseline"""
    print(f"\n⚖️  RUNNING COMPARISON ANALYSIS")
    print("Comparing Small-World vs Degree-Constrained Networks")
    print("-" * 50)
    
    # Test configurations
    configs = [
        ('Degree-3 (Baseline)', 'degree_constrained', {'degree_constraint': 3}),
        ('Small-World p=0.05', 'small_world', {'rewiring_probability': 0.05}),
        ('Small-World p=0.1', 'small_world', {'rewiring_probability': 0.1}),
        ('Small-World p=0.2', 'small_world', {'rewiring_probability': 0.2})
    ]
    
    results = {}
    
    for name, topology_type, params in configs:
        print(f"\n🧪 Testing {name}...")
        
        try:
            base_config = get_base_config()
            base_config['network_config']['topology_type'] = topology_type
            base_config['network_config'].update(params)
            
            if topology_type == 'small_world':
                analyzer = SmallWorldAnalysisFramework(base_config, [params['rewiring_probability']])
                result = analyzer.run_single_small_world_analysis(
                    params['rewiring_probability'], num_runs, steps_per_run
                )
            else:
                # Use degree comparison framework for baseline
                from degree_comparison import DegreeComparisonFramework
                degree_analyzer = DegreeComparisonFramework(base_config, [params['degree_constraint']])
                result = degree_analyzer.run_single_degree_simulation(
                    params['degree_constraint'], num_runs, steps_per_run
                )
            
            results[name] = result
            print(f"✅ {name} completed")
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    # Generate comparison report
    print_comparison_results(results)
    
    return results

def create_small_world_network_manager(p_value=0.2, k_neighbors=4):
    """Create network manager with small-world topology"""
    from Complete_NTEO.topology.network_topology import SydneyNetworkTopology
    from small_world_topology import SmallWorldTopologyGenerator
    from network_integration import TwoLayerNetworkManager, NetworkSpatialMapper, NetworkEdgeCongestionModel, NetworkRouter
    
    # Create base Sydney network
    base_network = SydneyNetworkTopology()
    base_network.initialize_base_sydney_network()
    
    # Generate small-world network
    generator = SmallWorldTopologyGenerator(base_network)
    small_world_graph = generator.generate_small_world_network(
        rewiring_probability=p_value,
        initial_neighbors=k_neighbors,
        preserve_geography=True
    )
    
    # Create custom network manager
    class SmallWorldNetworkManager(TwoLayerNetworkManager):
        def __init__(self, sw_graph, base_net):
            self.base_network = base_net
            self.active_network = sw_graph
            
            # Initialize components
            self.spatial_mapper = NetworkSpatialMapper(base_net, 100, 80)
            self.congestion_model = NetworkEdgeCongestionModel(sw_graph)
            self.router = NetworkRouter(sw_graph, self.congestion_model)
            self.route_calculation_count = 0
            
            # Ensure connectivity
            self._ensure_network_connectivity()
    
    return SmallWorldNetworkManager(small_world_graph, base_network)

def get_base_config():
    """Get base configuration for small-world analysis"""
    return {
        'db_connection_string': db.DB_CONNECTION_STRING,
        'data_income_weights': db.income_weights,
        'data_health_weights': db.health_weights,
        'data_payment_weights': db.payment_weights,
        'data_age_distribution': db.age_distribution,
        'data_disability_weights': db.disability_weights,
        'data_tech_access_weights': db.tech_access_weights,
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
        'subsidy_config': db.daily_config,
        'network_config': db.NETWORK_CONFIG.copy()
    }

def print_quick_results(result, p_value):
    """Print results from quick analysis"""
    print(f"\n📊 QUICK ANALYSIS RESULTS (p={p_value:.3f})")
    print("=" * 50)
    
    if 'network_properties' in result:
        props = result['network_properties']
        print(f"🌐 Network Properties:")
        print(f"   • Nodes: {props.get('nodes', 'N/A')}")
        print(f"   • Edges: {props.get('edges', 'N/A')}")
        print(f"   • Small-World Coefficient (σ): {props.get('small_world_sigma', 0):.3f}")
        print(f"   • Clustering: {props.get('clustering_coefficient', 0):.3f}")
        print(f"   • Path Length: {props.get('avg_path_length', 0):.2f}")
        print(f"   • Shortcuts: {props.get('shortcuts_created', 0)} ({props.get('shortcut_percentage', 0):.1f}%)")
    
    if 'equity_metrics' in result:
        equity = result['equity_metrics']
        print(f"\n🏛️ Equity Analysis:")
        print(f"   • Peripheral Benefit Score: {equity.get('peripheral_benefit_score', 0):.3f}")
        print(f"   • Efficiency-Equity Balance: {equity.get('efficiency_equity_balance', 0):.3f}")
        print(f"   • Centrality Inequality: {equity.get('centrality_inequality', 0):.3f}")
    
    if 'performance_metrics' in result:
        perf = result['performance_metrics']
        successful_runs = len([r for r in result.get('runs', []) if r.get('success', False)])
        print(f"\n⚡ Performance:")
        print(f"   • Successful Runs: {successful_runs}/{len(result.get('runs', []))}")
        print(f"   • Avg Runtime: {perf.get('run_time_seconds_mean', 0):.2f}s")
    
    # Research implications
    print(f"\n🔬 RESEARCH IMPLICATIONS:")
    if result.get('network_properties', {}).get('small_world_sigma', 0) > 1:
        print("   ✅ Network exhibits small-world properties")
    else:
        print("   ⚠️  Limited small-world effect")
    
    if result.get('equity_metrics', {}).get('peripheral_benefit_score', 0) > 0.1:
        print("   ✅ Shortcuts appear to benefit peripheral areas")
    else:
        print("   ⚠️  Limited benefit for peripheral areas")

def print_comprehensive_findings(results):
    """Print findings from comprehensive analysis"""
    print(f"\n🔬 COMPREHENSIVE ANALYSIS FINDINGS")
    print("=" * 60)
    
    if not results:
        print("No results to analyze.")
        return
    
    # Find optimal configurations
    best_sigma = 0
    best_equity = 0
    best_p_sigma = 0
    best_p_equity = 0
    
    for p_value, result in results.items():
        if 'performance_metrics' in result and 'equity_metrics' in result:
            sigma = result['performance_metrics'].get('small_world_sigma_mean', 0)
            equity = result['equity_metrics'].get('efficiency_equity_balance', 0)
            
            if sigma > best_sigma:
                best_sigma = sigma
                best_p_sigma = p_value
            
            if equity > best_equity:
                best_equity = equity
                best_p_equity = p_value
    
    print(f"🏆 OPTIMAL CONFIGURATIONS:")
    print(f"   • Best Small-World Effect: p={best_p_sigma:.3f} (σ={best_sigma:.3f})")
    print(f"   • Best Equity Balance: p={best_p_equity:.3f} (score={best_equity:.3f})")
    
    # Research conclusions
    print(f"\n📚 RESEARCH CONCLUSIONS:")
    print(f"   • Small-world shortcuts {'DO' if best_equity > 0.1 else 'DO NOT'} significantly improve transport equity")
    print(f"   • Optimal rewiring probability for Sydney: p≈{best_p_equity:.3f}")
    print(f"   • {'Strong' if best_sigma > 2 else 'Moderate' if best_sigma > 1 else 'Weak'} small-world effect achieved")

def print_comparison_results(results):
    """Print results from comparison analysis"""
    print(f"\n⚖️  COMPARISON ANALYSIS RESULTS")
    print("=" * 60)
    
    comparison_table = []
    for name, result in results.items():
        if 'error' in result:
            comparison_table.append([name, "FAILED", "N/A", "N/A", "N/A"])
        else:
            # Extract metrics (handle both small-world and degree results)
            if 'network_properties' in result:
                # Small-world result
                sigma = result.get('network_properties', {}).get('small_world_sigma', 0)
                clustering = result.get('network_properties', {}).get('clustering_coefficient', 0)
                path_length = result.get('network_properties', {}).get('avg_path_length', 0)
                equity_score = result.get('equity_metrics', {}).get('efficiency_equity_balance', 0)
            else:
                # Degree-constrained result
                sigma = 0  # No small-world properties
                clustering = result.get('network_stats', {}).get('clustering_coefficient', 0)
                path_length = result.get('network_stats', {}).get('avg_path_length', 0)
                equity_score = 0  # Calculate from available metrics
            
            comparison_table.append([
                name,
                f"{sigma:.3f}",
                f"{clustering:.3f}",
                f"{path_length:.2f}",
                f"{equity_score:.3f}"
            ])
    
    print(f"{'Configuration':<20} {'Sigma':<8} {'Cluster':<8} {'PathLen':<8} {'Equity':<8}")
    print("-" * 60)
    for row in comparison_table:
        print(f"{row[0]:<20} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8}")
    
    print(f"\n📊 KEY INSIGHTS:")
    print(f"   • Small-world networks show different properties than regular networks")
    print(f"   • Comparison helps identify optimal topology for transport equity")
    print(f"   • Results inform infrastructure planning decisions")

def create_quick_visualization(result, p_value):
    """Create quick visualization for single result"""
    try:
        # Create network manager
        network_manager = create_small_world_network_manager(p_value)
        
        # Create simple visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"quick_small_world_p{p_value:.3f}_{timestamp}.png"
        
        fig = create_small_world_visualization(network_manager, p_value, save_path)
        print(f"📸 Quick visualization saved: {save_path}")
        
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    sys.exit(main())