import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import seaborn as sns
from collections import defaultdict
import database as db
from network_integration import TwoLayerNetworkManager
import warnings
warnings.filterwarnings('ignore')

class AgentMovementTracker:
    def __init__(self, db_connection_string, network_manager=None):
        self.db_engine = create_engine(db_connection_string)
        self.Session = sessionmaker(bind=self.db_engine)
        
        # Create network manager if not provided
        if network_manager is None:
            print("Creating network manager for visualization...")
            self.network_manager = TwoLayerNetworkManager(
                topology_type="degree_constrained",
                degree=3,  # Default degree
                grid_width=100,
                grid_height=80
            )
        else:
            self.network_manager = network_manager
            
        # Check database connection
        self._check_database_connection()
        
    def _check_database_connection(self):
        """Check if database has required tables and data"""
        try:
            with self.Session() as session:
                # Check if tables exist and have data
                commuter_count = session.execute(text("SELECT COUNT(*) FROM commuter_info_log")).scalar()
                booking_count = session.execute(text("SELECT COUNT(*) FROM service_booking_log")).scalar()
                
                print(f"Database check:")
                print(f"  - Commuters in database: {commuter_count}")
                print(f"  - Bookings in database: {booking_count}")
                
                if commuter_count == 0:
                    print("⚠️  WARNING: No commuter data found. Run a simulation first!")
                if booking_count == 0:
                    print("⚠️  WARNING: No booking data found. Limited analysis possible.")
                    
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            print("Make sure you've run a simulation first to populate the database.")
    
    def get_agent_data_at_step(self, step):
        """Get all agent positions and modes at a specific step"""
        try:
            with self.Session() as session:
                # Get commuter positions - handle both string and direct coordinate columns
                commuter_query = text("""
                    SELECT commuter_id, 
                           COALESCE(location_x, 50) as x,
                           COALESCE(location_y, 40) as y,
                           income_level,
                           requests, 
                           services_owned
                    FROM commuter_info_log
                """)
                commuters = session.execute(commuter_query).fetchall()
                
                # Get active bookings at this step
                booking_query = text("""
                    SELECT commuter_id, record_company_name, start_time, total_time, 
                           route_details, status
                    FROM service_booking_log
                    WHERE start_time <= :step 
                    AND start_time + total_time >= :step
                    AND status = 'Service Selected'
                """)
                bookings = session.execute(booking_query, {"step": step}).fetchall()
                
                return commuters, bookings
                
        except Exception as e:
            print(f"Error getting agent data for step {step}: {e}")
            return [], []
    
    def visualize_agent_movements(self, start_step=0, end_step=100, interval=1000, save_animation=True):
        """Create animated visualization of agent movements"""
        print(f"Creating animation from step {start_step} to {end_step}...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Check if we have data
        test_commuters, test_bookings = self.get_agent_data_at_step(start_step)
        if not test_commuters:
            print("❌ No data available for visualization. Run a simulation first!")
            return None
        
        def animate(frame):
            current_step = start_step + frame
            ax1.clear()
            ax2.clear()
            
            # Left plot: Spatial movement
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 80)
            ax1.set_title(f'Agent Movements - Step {current_step}')
            ax1.grid(True, alpha=0.3)
            
            # Add network overlay
            if hasattr(self.network_manager, 'spatial_mapper'):
                for node_id, coord in self.network_manager.spatial_mapper.node_to_grid.items():
                    ax1.plot(coord[0], coord[1], 'ko', markersize=2, alpha=0.3)
            
            commuters, bookings = self.get_agent_data_at_step(current_step)
            
            # Track modes and positions
            mode_counts = defaultdict(int)
            
            # Create booking lookup
            booking_lookup = {booking.commuter_id: booking.record_company_name 
                            for booking in bookings}
            
            for commuter in commuters:
                try:
                    x, y = commuter.x, commuter.y
                    income_level = getattr(commuter, 'income_level', 'unknown')
                    
                    # Determine current mode from bookings
                    current_mode = booking_lookup.get(commuter.commuter_id, 'stationary')
                    
                    # Color by mode, shape by income
                    if current_mode == 'walk':
                        color, marker = 'green', 'o'
                    elif 'UberLike' in current_mode:
                        color, marker = 'red', '^'
                    elif 'BikeShare' in current_mode:
                        color, marker = 'blue', 's'
                    elif current_mode == 'public':
                        color, marker = 'orange', 'D'
                    elif current_mode == 'MaaS_Bundle':
                        color, marker = 'purple', '*'
                    else:
                        color, marker = 'gray', '.'
                    
                    # Size by income level
                    if income_level == 'high':
                        size = 50
                    elif income_level == 'middle':
                        size = 35
                    else:
                        size = 25
                    
                    ax1.scatter(x, y, c=color, marker=marker, s=size, alpha=0.7, 
                              edgecolors='black', linewidth=0.5)
                    mode_counts[current_mode] += 1
                    
                except Exception as e:
                    continue
            
            # Right plot: Mode distribution
            if mode_counts:
                modes = list(mode_counts.keys())
                counts = list(mode_counts.values())
                colors = ['green' if m=='walk' else 'red' if 'Uber' in m else 
                         'blue' if 'Bike' in m else 'orange' if m=='public' else 
                         'purple' if m=='MaaS_Bundle' else 'gray' for m in modes]
                
                bars = ax2.bar(modes, counts, color=colors, alpha=0.7)
                ax2.set_ylabel('Number of Agents')
                ax2.set_title(f'Mode Distribution - Step {current_step}')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'{int(height)}', ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(0.5, 0.5, 'No active trips', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=16)
            
            plt.tight_layout()
        
        # Create animation
        frames = min(end_step - start_step, 50)  # Limit frames for performance
        print(f"Generating {frames} frames...")
        
        ani = animation.FuncAnimation(fig, animate, frames=frames, 
                                    interval=interval, repeat=False)
        
        if save_animation:
            print("Saving animation... (this may take a while)")
            ani.save(f'agent_movements_{start_step}_{end_step}.mp4', writer='pillow', fps=2)
            print("Animation saved!")
        
        plt.show()
        return ani
    
    def analyze_mode_choice_patterns(self):
        """Analyze mode choice patterns by demographics"""
        print("Analyzing mode choice patterns...")
        
        try:
            with self.Session() as session:
                # Get comprehensive booking data
                query = text("""
                    SELECT 
                        sbl.commuter_id,
                        sbl.record_company_name,
                        sbl.total_time,
                        sbl.total_price,
                        sbl.start_time,
                        COALESCE(sbl.government_subsidy, 0) as government_subsidy,
                        cil.income_level,
                        cil.age,
                        cil.has_disability,
                        cil.payment_scheme
                    FROM service_booking_log sbl
                    JOIN commuter_info_log cil ON sbl.commuter_id = cil.commuter_id
                    WHERE sbl.status = 'Service Selected'
                """)
                
                df = pd.read_sql(query, session.bind)
                
        except Exception as e:
            print(f"Error querying database: {e}")
            return None
        
        if df.empty:
            print("❌ No booking data found!")
            return None
        
        print(f"Analyzing {len(df)} trips from {df['commuter_id'].nunique()} commuters")
        
        # Create comprehensive analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        try:
            # 1. Mode choice by income level
            ax1 = axes[0, 0]
            mode_income = pd.crosstab(df['record_company_name'], df['income_level'], normalize='columns')
            mode_income.plot(kind='bar', ax=ax1, stacked=True)
            ax1.set_title('Mode Choice by Income Level')
            ax1.set_ylabel('Proportion')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # 2. Mode choice by payment scheme
            ax2 = axes[0, 1]
            mode_payment = pd.crosstab(df['record_company_name'], df['payment_scheme'], normalize='columns')
            mode_payment.plot(kind='bar', ax=ax2, stacked=True)
            ax2.set_title('Mode Choice by Payment Scheme')
            ax2.set_ylabel('Proportion')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # 3. Trip duration by mode
            ax3 = axes[0, 2]
            df.boxplot(column='total_time', by='record_company_name', ax=ax3)
            ax3.set_title('Trip Duration by Mode')
            ax3.set_ylabel('Time (steps)')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # 4. Trip cost by mode
            ax4 = axes[1, 0]
            df.boxplot(column='total_price', by='record_company_name', ax=ax4)
            ax4.set_title('Trip Cost by Mode')
            ax4.set_ylabel('Price ($)')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            
            # 5. Subsidies by income
            ax5 = axes[1, 1]
            subsidy_data = df[df['government_subsidy'] > 0]
            if not subsidy_data.empty:
                subsidy_data.boxplot(column='government_subsidy', by='income_level', ax=ax5)
                ax5.set_title('Government Subsidies by Income Level')
                ax5.set_ylabel('Subsidy Amount ($)')
            else:
                ax5.text(0.5, 0.5, 'No subsidy data', ha='center', va='center', transform=ax5.transAxes)
            
            # 6. Demographics summary
            ax6 = axes[1, 2]
            demo_summary = f"""
Demographics Summary:
• Total Trips: {len(df)}
• Unique Commuters: {df['commuter_id'].nunique()}

Income Distribution:
{df['income_level'].value_counts().to_string()}

Payment Schemes:
{df['payment_scheme'].value_counts().to_string()}

Top Modes:
{df['record_company_name'].value_counts().head().to_string()}

Average Subsidies:
${df['government_subsidy'].mean():.2f}
            """
            
            ax6.text(0.05, 0.95, demo_summary, transform=ax6.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            ax6.axis('off')
            
        except Exception as e:
            print(f"Error creating plots: {e}")
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def track_individual_journeys(self, commuter_ids=None, max_commuters=5):
        """Track individual commuter journeys over time"""
        print(f"Tracking individual journeys (max {max_commuters} commuters)...")
        
        try:
            with self.Session() as session:
                if commuter_ids is None:
                    # Get random sample of active commuters
                    query = text("""
                        SELECT commuter_id
                        FROM (
                            SELECT DISTINCT commuter_id, RANDOM() as rand_val 
                            FROM service_booking_log
                        ) t
                        ORDER BY rand_val
                        LIMIT :limit
                    """)
                    result = session.execute(query, {"limit": max_commuters}).fetchall()
                    commuter_ids = [r.commuter_id for r in result]
                
                if not commuter_ids:
                    print("❌ No commuters found!")
                    return
                
                print(f"Tracking {len(commuter_ids)} commuters: {commuter_ids}")
                
                fig, axes = plt.subplots(len(commuter_ids), 1, figsize=(15, 4*len(commuter_ids)))
                if len(commuter_ids) == 1:
                    axes = [axes]
                
                for idx, commuter_id in enumerate(commuter_ids):
                    ax = axes[idx]
                    
                    # Get journey data for this commuter
                    journey_query = text("""
                        SELECT start_time, total_time, record_company_name, total_price
                        FROM service_booking_log
                        WHERE commuter_id = :commuter_id
                        ORDER BY start_time
                    """)
                    
                    journeys = session.execute(journey_query, {"commuter_id": commuter_id}).fetchall()
                    
                    if journeys:
                        times = [j.start_time for j in journeys]
                        modes = [j.record_company_name for j in journeys]
                        durations = [j.total_time for j in journeys]
                        
                        # Create timeline
                        y_pos = 0.5
                        colors_used = set()
                        
                        for i, (time, mode, duration) in enumerate(zip(times, modes, durations)):
                            # Color by mode
                            if mode == 'walk':
                                color = 'green'
                            elif 'Uber' in mode:
                                color = 'red'
                            elif 'Bike' in mode:
                                color = 'blue'
                            elif mode == 'public':
                                color = 'orange'
                            elif mode == 'MaaS_Bundle':
                                color = 'purple'
                            else:
                                color = 'gray'
                            
                            ax.barh(y_pos, duration, left=time, height=0.3, 
                                   color=color, alpha=0.7, 
                                   label=mode if mode not in colors_used else "")
                            colors_used.add(mode)
                            
                            # Add mode label
                            if duration > 5:  # Only label longer trips
                                ax.text(time + duration/2, y_pos, mode, 
                                       ha='center', va='center', fontsize=8, rotation=0)
                        
                        ax.set_title(f'Commuter {commuter_id} Journey Timeline ({len(journeys)} trips)')
                        ax.set_xlabel('Simulation Step')
                        ax.set_ylim(0, 1)
                        ax.set_yticks([])
                        
                        # Add legend
                        if colors_used:
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax.text(0.5, 0.5, f'No journeys found for Commuter {commuter_id}', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Commuter {commuter_id} - No Data')
                
        except Exception as e:
            print(f"Error tracking journeys: {e}")
            return
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("AGENT MOVEMENT ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        try:
            with self.Session() as session:
                # Basic statistics
                commuter_count = session.execute(text("SELECT COUNT(DISTINCT commuter_id) FROM commuter_info_log")).scalar()
                trip_count = session.execute(text("SELECT COUNT(*) FROM service_booking_log")).scalar()
                
                # Mode usage statistics
                mode_stats = session.execute(text("""
                    SELECT record_company_name, COUNT(*) as count, AVG(total_price) as avg_price
                    FROM service_booking_log 
                    GROUP BY record_company_name 
                    ORDER BY count DESC
                """)).fetchall()
                
                # Income distribution
                income_stats = session.execute(text("""
                    SELECT income_level, COUNT(*) as count
                    FROM commuter_info_log 
                    GROUP BY income_level
                """)).fetchall()
                
                print(f"📊 BASIC STATISTICS:")
                print(f"   Total Commuters: {commuter_count}")
                print(f"   Total Trips: {trip_count}")
                print(f"   Avg Trips per Commuter: {trip_count/commuter_count:.2f}")
                
                print(f"\n🚗 MODE USAGE:")
                for mode, count, avg_price in mode_stats:
                    print(f"   {mode}: {count} trips (${avg_price:.2f} avg)")
                
                print(f"\n💰 INCOME DISTRIBUTION:")
                for income, count in income_stats:
                    print(f"   {income}: {count} commuters ({count/commuter_count*100:.1f}%)")
                
        except Exception as e:
            print(f"Error generating summary: {e}")


def create_agent_tracker(degree_constraint=3):
    """Helper function to create a properly configured agent tracker"""
    print(f"Creating Agent Tracker with degree-{degree_constraint} network...")
    
    # Create network manager
    network_manager = TwoLayerNetworkManager(
        topology_type="degree_constrained",
        degree=degree_constraint,
        grid_width=100,
        grid_height=80
    )
    
    # Create tracker
    tracker = AgentMovementTracker(db.DB_CONNECTION_STRING, network_manager)
    
    return tracker


# Usage example and main execution
if __name__ == "__main__":
    print("🚀 Starting Agent Movement Tracker")
    
    try:
        # Create tracker with default network
        tracker = create_agent_tracker(degree_constraint=3)
        
        # Generate summary report
        tracker.generate_summary_report()
        
        # Analyze mode choice patterns
        print("\n📈 Analyzing mode choice patterns...")
        df = tracker.analyze_mode_choice_patterns()
        
        # Track some individual journeys
        print("\n👥 Tracking individual journeys...")
        tracker.track_individual_journeys(max_commuters=3)
        
        # Create visualization (optional - comment out if too slow)
        print("\n🎬 Creating movement visualization...")
        print("Note: Animation can take time. Set save_animation=True to save as file.")
        
        # Uncomment the next line to create animation
        # ani = tracker.visualize_agent_movements(start_step=0, end_step=20, interval=500)
        
        print("✅ Agent tracking analysis completed!")
        
    except Exception as e:
        print(f"❌ Error running agent tracker: {e}")
        print("\n💡 TROUBLESHOOTING:")
        print("1. Make sure you've run a simulation first to populate the database")
        print("2. Check your database connection string in database.py")
        print("3. Ensure required tables exist (commuter_info_log, service_booking_log)")