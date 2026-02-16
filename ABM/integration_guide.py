# integration_guide.py - Complete Live Agent Tracking Setup

"""
🎯 LIVE AGENT TRACKING INTEGRATION GUIDE

This guide shows you how to add live agent movement tracking and route visualization 
to your existing ABM-ETOP model.
"""

import database as db
from agent_run_visualisation import MobilityModel
from live_agent_tracker import add_live_tracking_to_model
from ABM.degree_constraints.route_heatmap_visualizer import add_live_route_visualization, create_live_dashboard
import matplotlib.pyplot as plt

# Fix for the original SQL error
def fix_sql_error_in_agent_tracker():
    """
    Quick fix for the PostgreSQL RANDOM() error in agent_tracker.py
    Replace the problematic query with this corrected version:
    """
    
    corrected_query = """
    -- FIXED: PostgreSQL requires DISTINCT columns in ORDER BY
    SELECT commuter_id, RANDOM() as rand_val
    FROM (SELECT DISTINCT commuter_id FROM service_booking_log) t
    ORDER BY rand_val
    LIMIT :limit
    """
    
    print("🔧 SQL Error Fix:")
    print("Replace the track_individual_journeys query with:")
    print(corrected_query)
    print()

def setup_live_tracking_basic():
    """Setup 1: Basic live tracking (no visualization)"""
    print("🚀 SETUP 1: Basic Live Tracking")
    print("=" * 50)
    
    # Create your model as normal
    model = MobilityModel(
        db_connection_string=db.DB_CONNECTION_STRING,
        num_commuters=30,  # Small number for testing
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
    
    # Add live tracking
    tracker = add_live_tracking_to_model(model)
    
    print("✅ Live tracking added!")
    print("Now run: model.run_model(50) to see live stats every 10 steps")
    
    return model, tracker

def setup_route_visualization():
    """Setup 2: Route visualization"""
    print("\n🗺️ SETUP 2: Route Visualization")
    print("=" * 50)
    
    # Get model with basic tracking
    model, tracker = setup_live_tracking_basic()
    
    # Add route visualization
    route_viz = add_live_route_visualization(model)
    
    print("✅ Route visualization added!")
    print("Run simulation and call route_viz.update_visualization() to see routes")
    
    # Run some steps to generate data
    print("Running 20 steps to generate route data...")
    model.run_model(20)
    
    # Show visualization
    route_viz.update_visualization()
    plt.show()
    
    return model, route_viz

def setup_full_dashboard():
    """Setup 3: Full live dashboard"""
    print("\n📊 SETUP 3: Full Live Dashboard")
    print("=" * 50)
    
    # Get model with tracking
    model, tracker = setup_live_tracking_basic()
    
    # Create full dashboard
    dashboard = create_live_dashboard(model)
    
    print("✅ Full dashboard created!")
    
    # Run simulation with live updates
    print("Running simulation with live dashboard updates...")
    
    for step in range(30):
        model.step()
        
        # Update dashboard every 5 steps
        if step % 5 == 0:
            dashboard.update_dashboard()
            plt.pause(0.1)  # Brief pause to see updates
    
    plt.show()
    
    return model, dashboard

def setup_mesa_integration():
    """Setup 4: Mesa web interface integration"""
    print("\n🌐 SETUP 4: Mesa Web Interface")
    print("=" * 50)
    
    from mesa.visualization.modules import CanvasGrid
    from mesa.visualization.ModularVisualization import ModularServer
    from live_agent_tracker import enhanced_agent_portrayal, LiveVisualizationElement
    
    # Model parameters
    model_params = {
        "db_connection_string": db.DB_CONNECTION_STRING,
        "num_commuters": 30,
        "data_income_weights": db.income_weights,
        "data_health_weights": db.health_weights,
        "data_payment_weights": db.payment_weights,
        "data_age_distribution": db.age_distribution,
        "data_disability_weights": db.disability_weights,
        "data_tech_access_weights": db.tech_access_weights,
        "ASC_VALUES": db.ASC_VALUES,
        "UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS": db.UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
        "UTILITY_FUNCTION_BASE_COEFFICIENTS": db.UTILITY_FUNCTION_BASE_COEFFICIENTS,
        "PENALTY_COEFFICIENTS": db.PENALTY_COEFFICIENTS,
        "AFFORDABILITY_THRESHOLDS": db.AFFORDABILITY_THRESHOLDS,
        "FLEXIBILITY_ADJUSTMENTS": db.FLEXIBILITY_ADJUSTMENTS,
        "VALUE_OF_TIME": db.VALUE_OF_TIME,
        "public_price_table": db.public_price_table,
        "ALPHA_VALUES": db.ALPHA_VALUES,
        "DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS": db.DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
        "BACKGROUND_TRAFFIC_AMOUNT": db.BACKGROUND_TRAFFIC_AMOUNT,
        "CONGESTION_ALPHA": db.CONGESTION_ALPHA,
        "CONGESTION_BETA": db.CONGESTION_BETA,
        "CONGESTION_CAPACITY": db.CONGESTION_CAPACITY,
        "CONGESTION_T_IJ_FREE_FLOW": db.CONGESTION_T_IJ_FREE_FLOW,
        "uber_like1_capacity": db.UberLike1_capacity,
        "uber_like1_price": db.UberLike1_price,
        "uber_like2_capacity": db.UberLike2_capacity,
        "uber_like2_price": db.UberLike2_price,
        "bike_share1_capacity": db.BikeShare1_capacity,
        "bike_share1_price": db.BikeShare1_price,
        "bike_share2_capacity": db.BikeShare2_capacity,
        "bike_share2_price": db.BikeShare2_price,
        "subsidy_dataset": db.subsidy_dataset,
        "subsidy_config": db.daily_config,
        "network_config": db.NETWORK_CONFIG
    }
    
    # Create enhanced model creator
    def create_model_with_tracking(**params):
        model = MobilityModel(**params)
        # Add live tracking
        add_live_tracking_to_model(model)
        return model
    
    # Enhanced grid with live tracking
    grid = CanvasGrid(enhanced_agent_portrayal, 100, 80, 800, 640)
    
    # Live stats element
    class LiveStatsElement:
        def render(self, model):
            if hasattr(model, 'live_tracker'):
                tracker = model.live_tracker
                stats = tracker.get_live_statistics()
                popular_routes = tracker.get_popular_routes(3)
                
                html = f"""
                <div style='font-family: monospace;'>
                <h3>🎯 Live Tracking - Step {stats['step']}</h3>
                <p><b>Active Agents:</b> {stats['active_agents']}/{stats['total_agents']}</p>
                <p><b>Popular Routes:</b></p>
                <ul>
                """
                
                for (origin, dest), count in popular_routes:
                    html += f"<li>({origin[0]:.0f},{origin[1]:.0f}) → ({dest[0]:.0f},{dest[1]:.0f}): {count} trips</li>"
                
                html += "</ul></div>"
                return html
            return "<p>Tracker loading...</p>"
        
        def __init__(self):
            pass
    
    # Create server
    server = ModularServer(
        create_model_with_tracking,
        [grid, LiveStatsElement()],
        "Live Agent Movement Tracker",
        model_params
    )
    
    print("✅ Mesa server created!")
    print("Run: server.launch() to start web interface")
    print("Then go to: http://localhost:8521")
    
    return server

# Quick setup functions
def quick_setup_basic():
    """Quick setup for basic live tracking"""
    model, tracker = setup_live_tracking_basic()
    
    print("\n🎯 QUICK START COMMANDS:")
    print("model.run_model(50)  # Run 50 steps with live stats")
    print("tracker.get_popular_routes()  # Get popular routes")
    print("tracker.get_live_statistics()  # Get current stats")
    
    return model, tracker

def quick_setup_visualization():
    """Quick setup for route visualization"""
    model, route_viz = setup_route_visualization()
    
    print("\n🎯 VISUALIZATION COMMANDS:")
    print("route_viz.update_visualization()  # Update route heatmap")
    print("route_viz.save_current_state()  # Save as PNG")
    
    return model, route_viz

# Main execution
if __name__ == "__main__":
    print("🎯 LIVE AGENT TRACKING INTEGRATION GUIDE")
    print("=" * 60)
    
    # Show the SQL fix first
    fix_sql_error_in_agent_tracker()
    
    print("Choose your setup:")
    print("1. Basic live tracking (console stats)")
    print("2. Route visualization (matplotlib)")
    print("3. Full dashboard (advanced)")
    print("4. Mesa web interface")
    
    choice = input("\nEnter choice (1-4) or 'quick' for demo: ").strip().lower()
    
    if choice == '1':
        model, tracker = setup_live_tracking_basic()
    elif choice == '2':
        model, route_viz = setup_route_visualization()
    elif choice == '3':
        model, dashboard = setup_full_dashboard()
    elif choice == '4':
        server = setup_mesa_integration()
    elif choice == 'quick':
        print("\n🚀 Running quick demo...")
        model, tracker = quick_setup_basic()
        print("\n📊 Running 30 steps...")
        model.run_model(30)
        print("\n📈 Final statistics:")
        print(tracker.get_live_statistics())
        print("\n🔥 Popular routes:")
        for route, count in tracker.get_popular_routes(5):
            origin, dest = route
            print(f"  ({origin[0]:.0f},{origin[1]:.0f}) → ({dest[0]:.0f},{dest[1]:.0f}): {count} trips")
    else:
        print("Invalid choice. Run again and choose 1-4 or 'quick'")