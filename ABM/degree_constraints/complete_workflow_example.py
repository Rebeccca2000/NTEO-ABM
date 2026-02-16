# complete_workflow_example.py
"""
🎯 COMPLETE WORKFLOW: How to use both old and new tracking systems

This shows you exactly how to:
1. Run your normal simulation with live tracking
2. Then use the old agent_tracker for detailed analysis
3. Show numbered edges and mode-specific route popularity
"""

import database as db
from agent_run_visualisation import MobilityModel

# Import the new live tracking (after you create the files)
try:
    from live_agent_tracker import add_live_tracking_to_model
    from ABM.degree_constraints.route_heatmap_visualizer import add_live_route_visualization
    NEW_SYSTEM_AVAILABLE = True
except ImportError:
    print("⚠️  New tracking files not found. Create them first!")
    NEW_SYSTEM_AVAILABLE = False

def run_simulation_with_live_tracking():
    """Method 1: Run simulation with LIVE tracking"""
    print("🚀 RUNNING SIMULATION WITH LIVE TRACKING")
    print("="*50)
    
    # Create your model exactly as before
    model = MobilityModel(
        db_connection_string=db.DB_CONNECTION_STRING,
        num_commuters=db.num_commuters,  # Small for testing
        data_income_weights=db.income_weights,
        data_health_weights=db.health_weights,
        data_payment_weights=db.payment_weights,
        data_age_distribution=db.age_distribution,
        data_disability_weights=db.disability_weights,
        data_tech_access_weights=db.tech_access_weights,
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
        subsidy_config=db.daily_config,
        network_config=db.NETWORK_CONFIG
    )
    
    if NEW_SYSTEM_AVAILABLE:
        # 🆕 ADD LIVE TRACKING (just 2 lines!)
        tracker = add_live_tracking_to_model(model)
        route_viz = add_live_route_visualization(model)
        
        print("✅ Live tracking added!")
        
        # Run simulation - now you see live stats!
        print("🏃 Running simulation with live tracking...")
        model.run_model(db.SIMULATION_STEPS)
        
        # Show final route visualization with multiple modes
        print("📊 Showing final multi-mode route heatmap...")
        route_viz.update_visualization()
        
        # Print detailed edge summary
        print("📋 Printing detailed edge usage summary...")
        route_viz.print_edge_summary()
        
        # Print summary
        print("\n🎯 LIVE TRACKING SUMMARY:")
        stats = tracker.get_live_statistics()
        print(f"  Final step: {stats['step']}")
        print(f"  Active agents: {stats['active_agents']}")
        print(f"  Available modes: {stats['available_modes']}")
        
        print("\n🔥 TOP 5 POPULAR EDGES (ALL MODES):")
        popular_edges = tracker.get_popular_edges_by_mode('all', 5)
        for i, (edge_num, count) in enumerate(popular_edges, 1):
            edge_desc = tracker.get_edge_description(edge_num)
            print(f"  {i}. {edge_desc}: {count} uses")
        
        # Show mode-specific top edges
        print("\n🚗 MODE-SPECIFIC POPULAR EDGES:")
        for mode in ['walk', 'bike', 'car', 'public', 'MaaS_Bundle']:
            mode_edges = tracker.get_popular_edges_by_mode(mode, 3)
            if mode_edges:
                print(f"  {mode.upper()}:")
                for i, (edge_num, count) in enumerate(mode_edges, 1):
                    edge_desc = tracker.get_edge_description(edge_num)
                    print(f"    {i}. {edge_desc}: {count} uses")
            else:
                print(f"  {mode.upper()}: No routes recorded")
        
        return model, tracker, route_viz
    else:
        # Run without live tracking
        print("⚠️  Running without live tracking (files not found)")
        model.run_model(50)
        return model, None, None

def run_post_simulation_analysis():
    """Method 2: Use OLD agent_tracker for detailed analysis AFTER simulation"""
    print("\n📈 RUNNING POST-SIMULATION ANALYSIS")
    print("="*50)
    
    # Import and use the old system
    from agent_tracker import create_agent_tracker
    
    # Create tracker that reads from database
    tracker = create_agent_tracker(degree_constraint=3)
    
    print("✅ Post-simulation tracker created!")
    
    # Generate analysis reports
    tracker.generate_summary_report()
    
    # Analyze patterns
    print("\n📊 Analyzing mode choice patterns...")
    df = tracker.analyze_mode_choice_patterns()
    
    # Track individual journeys
    print("\n👥 Tracking individual journeys...")
    tracker.track_individual_journeys(max_commuters=3)
    
    return tracker

def complete_workflow():
    """Complete workflow: Live tracking DURING + Analysis AFTER"""
    print("🎯 COMPLETE AGENT TRACKING WORKFLOW")
    print("="*60)
    
    # Step 1: Run simulation with live tracking
    model, live_tracker, route_viz = run_simulation_with_live_tracking()
    
    # Step 2: Detailed post-simulation analysis
    post_tracker = run_post_simulation_analysis()
    
    print("\n✅ WORKFLOW COMPLETE!")
    print("\nWhat you got:")
    print("  🎯 Live tracking during simulation")
    print("  🗺️  Multi-mode route heatmap visualization") 
    print("  📊 Numbered edge usage analysis")
    print("  📈 Detailed post-simulation analysis")
    print("  👥 Individual journey tracking")
    print("\n📋 Key Features:")
    print("  • Network edges numbered for easy reference")
    print("  • Separate visualizations for each transport mode")
    print("  • Real-time edge usage tracking")
    print("  • Mode-specific route popularity analysis")
    
    return model, live_tracker, route_viz, post_tracker

# Different ways to run it
def option_1_live_only():
    """Just live tracking during simulation"""
    print("🎯 OPTION 1: Live tracking only")
    model, tracker, viz = run_simulation_with_live_tracking()
    
def option_2_analysis_only():
    """Just post-simulation analysis (your current approach)"""
    print("📈 OPTION 2: Post-simulation analysis only")
    tracker = run_post_simulation_analysis()

def option_3_both():
    """Both live and post-simulation (recommended)"""
    print("🔥 OPTION 3: Both systems (recommended)")
    complete_workflow()

def quick_demo():
    """Quick demo showing edge numbering and mode-specific analysis"""
    print("🚀 Quick demo - running edge analysis system...")
    
    if NEW_SYSTEM_AVAILABLE:
        print("✅ New files detected - running complete workflow")
        model, live_tracker, route_viz, post_tracker = complete_workflow()
        
        # Additional edge analysis
        if live_tracker:
            print("\n🔍 ADDITIONAL EDGE ANALYSIS:")
            print(f"Total network edges: {len(live_tracker.edge_to_number)}")
            print(f"Edges actually used: {len(live_tracker.edge_usage_by_mode['all'])}")
            
            # Show edge efficiency
            all_edges = live_tracker.get_popular_edges_by_mode('all', 10)
            if all_edges:
                total_usage = sum(count for _, count in all_edges)
                print(f"Total edge uses: {total_usage}")
                print(f"Network utilization: {len(all_edges)}/{len(live_tracker.edge_to_number)} edges used "
                      f"({len(all_edges)/len(live_tracker.edge_to_number)*100:.1f}%)")
        
        # Save visualization
        if route_viz:
            route_viz.save_current_state("final_multi_mode_analysis.png")
            print("💾 Saved final analysis as 'final_multi_mode_analysis.png'")
            
    else:
        print("⚠️  New files not found - running post-analysis only")
        option_2_analysis_only()

if __name__ == "__main__":
    print("🎯 AGENT TRACKING COMPLETE WORKFLOW")
    print("="*50)
    
    print("Choose your option:")
    print("1. Live tracking only (during simulation)")
    print("2. Post-analysis only (your current method)")  
    print("3. Both systems (recommended)")
    print("4. Quick demo with edge analysis")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        option_1_live_only()
    elif choice == '2':
        option_2_analysis_only()
    elif choice == '3':
        option_3_both()
    elif choice == '4':
        quick_demo()
    else:
        print("❌ Invalid choice - running quick demo")
        quick_demo()
        
    print("\n🎯 NEXT STEPS:")
    if not NEW_SYSTEM_AVAILABLE:
        print("1. Create the 3 new files from the artifacts")
        print("2. Run this script again")
    else:
        print("1. ✅ Integrate live tracking into your main simulation")
        print("2. ✅ Use numbered edges for network analysis")
        print("3. ✅ Analyze mode-specific route patterns")
        print("4. ✅ Use both systems for comprehensive analysis")
        print("\n📊 Analysis Features Available:")
        print("  • Real-time edge usage tracking")
        print("  • Mode-specific route visualizations")
        print("  • Network efficiency analysis")
        print("  • Edge numbering for easy reference")