#!/usr/bin/env python3
"""
Script to run NTEO Results Aggregation
Place this file in your main project directory and run it
"""

import sys
from pathlib import Path

# Add the research folder to Python path so we can import the aggregator
project_root = Path(__file__).parent
research_path = project_root / "research"
sys.path.insert(0, str(research_path))

# Now import the aggregator
from research.NTEOResultsAggregator import run_aggregation_with_verification

def main():
    print("🚀 NTEO Results Aggregation")
    print("="*50)
    
    # Define paths
    results_directory = "wctr_test_results"  # Your results folder
    output_directory = "aggregated_analysis"  # Where to save aggregated results
    
    # Check if results directory exists
    results_path = Path(results_directory)
    if not results_path.exists():
        print(f"❌ Results directory '{results_directory}' not found!")
        print(f"   Current working directory: {Path.cwd()}")
        print(f"   Looking for: {results_path.absolute()}")
        print("\n💡 Make sure you're running this script from the correct directory")
        return
    
    # Check for JSON files
    json_files = list(results_path.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in '{results_directory}'!")
        print(f"   Directory contents: {list(results_path.iterdir())}")
        return
    
    print(f"✅ Found {len(json_files)} JSON files in '{results_directory}'")
    
    # Run the aggregation
    try:
        final_report = run_aggregation_with_verification(
            results_directory=results_directory,
            output_directory=output_directory
        )
        
        if final_report:
            print(f"\n🎉 Aggregation completed successfully!")
            print(f"📁 Check results in: {output_directory}/")
            
            # Show key findings
            if 'statistical_adequacy' in final_report['metadata']:
                adequacy = final_report['metadata']['statistical_adequacy']
                print(f"\n📊 Statistical Summary:")
                print(f"   - Total runs processed: {final_report['metadata']['total_runs_found']}")
                print(f"   - Statistically adequate: {'YES' if adequacy['overall_adequate'] else 'NO'}")
                
                if not adequacy['overall_adequate']:
                    print(f"   - Recommendations for improvement:")
                    for rec in adequacy['recommendations'][:3]:  # Show first 3
                        print(f"     {rec}")
        else:
            print("❌ Aggregation failed - check the output above for details")
            
    except Exception as e:
        print(f"❌ Error during aggregation: {e}")
        print(f"   Make sure NTEOResultsAggregator.py is in the research/ folder")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()