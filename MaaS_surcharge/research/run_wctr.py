# run_wctr.py
import sys
sys.path.append('.')  # Fix path first

from research.nteo_research_runner import NTEOResearchRunner

def main():
    print("🎯 NTEO WCTR Statistical Analysis")
    print("=" * 50)
    print("Choose option:")
    print("1. Full WCTR Analysis (25 runs) - Journal Quality ~4-6 hours")
    print("2. Quick Test (5 runs) - ~1 hour") 
    print("3. Mini Test (2 runs) - ~15 minutes")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        print("🚀 Running FULL WCTR Statistical Analysis...")
        runner = NTEOResearchRunner("wctr_full_results")
        
        topology_configs = {
            'grid': [4, 6, 8], 
            'degree_constrained': [3, 4, 5, 6, 7],
            'small_world': [0.0, 0.1, 0.2, 0.3, 0.5],
            'scale_free': [1, 2, 3, 4]
        }

        results = runner.run_wctr_statistical_study(num_runs=25, topology_configs=topology_configs, steps_per_run=144)
        
    elif choice == "2":
        print("🚀 Running Quick Test 2...")
        runner = NTEOResearchRunner("wctr_test_results_1")
        
        topology_configs = {
            'degree_constrained': [0, 2, 4, 6, 8, 12,16],
            'small_world': [0.0,0.2,0.4,0.6,0.8,1.2,1.6],
            'scale_free': [0,2, 4,6, 8, 12,16]
        }
        # topology_configs = {
        #     'grid': [4, 6], 
        #     # 'degree_constrained': [1,5],
        #     # 'small_world': [0.1,0.5],
        #     # 'scale_free': [1,5] 
        # }
        # topology_configs = {
        #     # 'grid': [4, 6], 
        #     'degree_constrained': [5],
        #     # 'small_world': [0.1,0.5],
        #     # 'scale_free': [2,11]
        # }
        results= runner.run_wctr_statistical_study(num_runs=5, topology_configs=topology_configs, steps_per_run=144)
      
    else:
        print("🚀 Running Mini Test...")
        runner = NTEOResearchRunner("wctr_mini_results")
        
        topology_configs = {
            # 'grid': [4], 
            'degree_constrained': [3, 4],
            'small_world': [0.1, 0.2],
            'scale_free': [1, 2]
        }
        

        results = runner.run_wctr_statistical_study(num_runs=1, topology_configs=topology_configs, steps_per_run=144)
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📁 Check results in: {runner.output_dir}")

if __name__ == "__main__":
    main()