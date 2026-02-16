# run_bike_speed_analysis.py
"""
Simple runner script for bike speed analysis.
Execute this after implementing the service provider modifications.
"""

import sys
import os
from pathlib import Path

# Add your project root to path
project_root = Path(__file__).parent
repo_root = project_root.parent
# Ensure the repository root is on sys.path so top-level packages (e.g. `config`, `testing`) can be imported
sys.path.insert(0, str(repo_root.resolve()))

# Import the analyzer
from bike_speed_analysis import BikeSpeedAnalyzer

def main():
    """Run the bike speed sensitivity analysis."""
    
    print("🚀 NTEO Bike Speed Analysis")
    print("="*50)
    print("Testing different bike speeds on baseline public transport network")
    print("Metrics: Mode Choice Equity, Travel Time Equity, System Efficiency")
    print()
    
    # Create analyzer with custom configuration if needed
    analyzer = BikeSpeedAnalyzer(
        output_dir="results/bike_speed_analysis"
    )
    
    # Optional: Customize analysis parameters
    analyzer.bike_speeds = [ 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]  # Speed range

    analyzer.num_runs_per_speed = 5  # Adjust based on computational resources
    analyzer.simulation_steps = 144  # Full day simulation
    
    # Run analysis
    try:
        results = analyzer.run_bike_speed_analysis()
        
        # Display summary results
        print_analysis_summary(results)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Check that you have implemented the service provider modifications!")
        return False
    
    return True

def print_analysis_summary(results):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("🎯 ANALYSIS SUMMARY")
    print("="*60)
    
    # Basic stats
    total_simulations = sum(len(speed_data['runs']) for speed_data in results['speed_results'].values())
    print(f"Total simulations completed: {total_simulations}")
    print(f"Speed range tested: {min(results['speed_results'].keys())} - {max(results['speed_results'].keys())}")
    print(f"Analysis execution time: {results['execution_time']:.1f} seconds")
    print()
    
    # Optimal speeds
    if 'optimal_speeds' in results['statistical_analysis']:
        optimal = results['statistical_analysis']['optimal_speeds']
        print("🏆 OPTIMAL BIKE SPEEDS:")
        print(f"  Best Mode Choice Equity: {optimal['best_mode_choice_equity']['speed']} (score: {optimal['best_mode_choice_equity']['value']:.3f})")
        print(f"  Best Travel Time Equity: {optimal['best_travel_time_equity']['speed']} (score: {optimal['best_travel_time_equity']['value']:.3f})")
        print(f"  Best System Efficiency: {optimal['best_system_efficiency']['speed']} (score: {optimal['best_system_efficiency']['value']:.1f})")
        print()
    
    # Speed sensitivity trends
    if 'speed_sensitivity' in results['statistical_analysis']:
        sensitivity = results['statistical_analysis']['speed_sensitivity']
        print("📈 SPEED SENSITIVITY TRENDS:")
        for metric, data in sensitivity.items():
            correlation = data['correlation_with_speed']
            interpretation = data['interpretation']
            print(f"  {metric}: {interpretation} (r={correlation:.3f})")
        print()
    
    # Key insights
    print("💡 KEY INSIGHTS:")
    insights = generate_insights(results)
    for insight in insights:
        print(f"  • {insight}")
    
    print("\n📊 Check the results/bike_speed_analysis/ folder for detailed visualizations and data!")

def generate_insights(results):
    """Generate key insights from the analysis."""
    insights = []
    
    if 'speed_sensitivity' not in results['statistical_analysis']:
        return ["Analysis incomplete - check for errors in simulation runs"]
    
    sensitivity = results['statistical_analysis']['speed_sensitivity']
    
    # Analyze mode choice equity trend
    if 'final_mode_choice_equity' in sensitivity:
        corr = sensitivity['final_mode_choice_equity']['correlation_with_speed']
        if abs(corr) > 0.5:
            trend = "increases" if corr > 0 else "decreases"
            insights.append(f"Mode choice equity {trend} significantly with bike speed (strong correlation)")
        else:
            insights.append("Mode choice equity shows weak sensitivity to bike speed changes")
    
    # Analyze travel time equity trend  
    if 'final_travel_time_equity' in sensitivity:
        corr = sensitivity['final_travel_time_equity']['correlation_with_speed']
        if abs(corr) > 0.5:
            trend = "increases" if corr > 0 else "decreases"
            insights.append(f"Travel time equity {trend} significantly with bike speed")
        else:
            insights.append("Travel time equity relatively stable across bike speeds")
    
    # Analyze system efficiency trend
    if 'final_system_efficiency' in sensitivity:
        corr = sensitivity['final_system_efficiency']['correlation_with_speed']
        if abs(corr) > 0.5:
            trend = "improves" if corr > 0 else "deteriorates"
            insights.append(f"System efficiency {trend} significantly with faster bikes")
        else:
            insights.append("System efficiency shows limited response to bike speed")
    
    # Check for trade-offs
    optimal_speeds = results['statistical_analysis'].get('optimal_speeds', {})
    if optimal_speeds:
        equity_speed = optimal_speeds['best_mode_choice_equity']['speed']
        efficiency_speed = optimal_speeds['best_system_efficiency']['speed']
        
        if equity_speed != efficiency_speed:
            insights.append(f"Trade-off detected: equity optimized at speed {equity_speed}, efficiency at {efficiency_speed}")
        else:
            insights.append(f"Alignment found: both equity and efficiency optimized at speed {equity_speed}")
    
    return insights

if __name__ == "__main__":
    main()