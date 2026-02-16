#!/usr/bin/env python3
"""
Run Simple NTEO Aggregation
Focus on clean average results for the three key metrics
"""

import sys
from pathlib import Path

# Add research folder to path
project_root = Path(__file__).parent
research_path = project_root / "research"
sys.path.insert(0, str(research_path))

# Import the simple aggregator (save the SimpleMetricsAggregator code as SimpleMetricsAggregator.py)
try:
    from research.SimpleMetricsAggregator import run_simple_aggregation
except ImportError:
    print("❌ Please save the SimpleMetricsAggregator code as 'SimpleMetricsAggregator.py' in the research/ folder")
    sys.exit(1)

def main():
    print("🎯 NTEO Simple Metrics Aggregation - Multiple Directories")
    print("="*60)
    print("Focus: Clean averages for Mode Choice Equity, Travel Time Equity, System Efficiency")
    print("Scanning: wctr_test_results, wctr_test_results_1, wctr_test_results_2, etc.")
    print("")
    
    # Your base directory name - it will find all numbered variations
    results_base_directory = "wctr_test_results"  # This will find wctr_test_results, wctr_test_results_1, etc.
    output_directory = "combined_metrics_results"
    
    # Check if at least the base directory exists
    base_path = Path(results_base_directory)
    if not base_path.exists():
        print(f"❌ Base results directory '{results_base_directory}' not found!")
        print(f"   Current directory: {Path.cwd()}")
        return
    
    # Run the simple aggregation
    try:
        print(f"🚀 Processing ALL result directories matching: {results_base_directory}*")
        results = run_simple_aggregation(results_base_directory, output_directory)
        if results and 'data' in results:
            print("\n🎉 SUCCESS! Clean metrics aggregation completed.")
            print("\n📊 QUICK RESULTS PREVIEW:")
            
            # Show quick preview of the averages
            df = results['data']
            if not df.empty:
                print("-" * 60)
                for _, row in df.iterrows():
                    topology = row['topology'].replace('_', ' ').title()
                    print(f"\n{topology}:")
                    
                    # Mode Choice Equity
                    if 'final_mode_choice_equity_mean' in row:
                        mce_mean = row['final_mode_choice_equity_mean']
                        mce_count = row['final_mode_choice_equity_count']
                        print(f"  Mode Choice Equity:  {mce_mean:.3f} ({mce_count} runs)")
                    
                    # Travel Time Equity  
                    if 'final_travel_time_equity_mean' in row:
                        tte_mean = row['final_travel_time_equity_mean']
                        tte_count = row['final_travel_time_equity_count']
                        print(f"  Travel Time Equity:  {tte_mean:.3f} ({tte_count} runs)")
                    
                    # System Efficiency
                    if 'final_system_efficiency_mean' in row:
                        se_mean = row['final_system_efficiency_mean']
                        se_count = row['final_system_efficiency_count']
                        print(f"  System Efficiency:   {se_mean:.0f} ({se_count} runs)")
                
                print("-" * 60)
            
            print(f"\n📁 Full results saved in: {output_directory}/")
            print("   - Summary table (TXT)")
            print("   - Comparison plot (PNG)")  
            print("   - Raw data (CSV)")
            
        else:
            print("❌ No results generated - check your JSON files")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()