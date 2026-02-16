import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

class RouteHeatmapVisualizer:
    """Visualize popular routes as heatmaps on the network by transport mode"""
    
    def __init__(self, model):
        self.model = model
        self.network_manager = model.network_manager
        self.live_tracker = getattr(model, 'live_tracker', None)
        
        # Create figure for live visualization - 5 mode-specific plots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(24, 16))
        self.axes = self.axes.flatten()  # Make it easier to index
        
        # Define the modes we want to visualize
        self.modes_to_show = ['walk', 'bike', 'car', 'public', 'MaaS_Bundle']
        self.mode_colors = {
            'walk': 'green',
            'bike': 'blue', 
            'car': 'red',
            'public': 'orange',
            'MaaS_Bundle': 'purple'
        }
        
        self.setup_base_networks()
        
        print("🗺️ Multi-Mode Route Heatmap Visualizer initialized!")
    
    def setup_base_networks(self):
        """Setup the base network visualization for all subplots"""
        for idx, mode in enumerate(self.modes_to_show):
            ax = self.axes[idx]
            ax.clear()
            ax.set_xlim(0, self.model.grid_width)
            ax.set_ylim(0, self.model.grid_height)
            ax.set_aspect('equal')
            ax.set_title(f'{mode.title()} Routes', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Draw network nodes
            for node_id, coord in self.network_manager.spatial_mapper.node_to_grid.items():
                node_data = self.network_manager.base_network.nodes.get(node_id)
                if node_data:
                    if node_data.node_type.value == 'major_hub':
                        ax.plot(coord[0], coord[1], 'ro', markersize=6, alpha=0.7)
                    elif node_data.node_type.value == 'transport_hub':
                        ax.plot(coord[0], coord[1], 'bo', markersize=4, alpha=0.7)
                    else:
                        ax.plot(coord[0], coord[1], 'go', markersize=3, alpha=0.5)
            
            # Draw network edges (base)
            for u, v in self.network_manager.active_network.edges():
                coord1 = self.network_manager.spatial_mapper.node_to_grid.get(u)
                coord2 = self.network_manager.spatial_mapper.node_to_grid.get(v)
                if coord1 and coord2:
                    ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 
                           'k-', alpha=0.2, linewidth=0.5)
        
        # Use the last subplot for overall statistics
        self.setup_statistics_plot()
    
    def setup_statistics_plot(self):
        """Setup the statistics subplot"""
        ax = self.axes[5]  # Last subplot
        ax.clear()
        ax.set_title('Edge Usage Statistics', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def update_visualization(self):
        """Update all mode-specific visualizations"""
        if not self.live_tracker:
            print("No live tracker available!")
            return
        
        # Setup base networks
        self.setup_base_networks()
        
        # Update each mode-specific visualization
        for idx, mode in enumerate(self.modes_to_show):
            self.draw_mode_specific_routes(mode, self.axes[idx])
        
        # Update statistics
        self.update_statistics_plot()
        
        plt.tight_layout()
        plt.draw()
    
    def draw_mode_specific_routes(self, mode, ax):
        """Draw routes for a specific mode"""
        if not self.live_tracker:
            return
        
        # Get edge usage for this mode
        popular_edges = self.live_tracker.get_popular_edges_by_mode(mode, 15)
        
        if not popular_edges:
            # Show "No data" message
            ax.text(0.5, 0.5, f'No {mode} routes yet', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, style='italic', alpha=0.7)
            return
        
        # Normalize usage counts for this mode
        max_usage = max(count for _, count in popular_edges) if popular_edges else 1
        
        # Get mode-specific color
        base_color = self.mode_colors.get(mode, 'gray')
        
        # Draw edges with usage intensity
        edge_lines = []
        for i, (edge_num, count) in enumerate(popular_edges):
            if edge_num in self.live_tracker.number_to_edge:
                edge_info = self.live_tracker.number_to_edge[edge_num]
                coord1, coord2 = edge_info['coords']
                
                # Calculate line properties
                intensity = count / max_usage
                line_width = 2 + (intensity * 8)  # 2-10 pixels wide
                alpha = 0.4 + (intensity * 0.6)   # 0.4-1.0 transparency
                
                # Create color gradient
                if base_color == 'green':
                    color = plt.cm.Greens(0.3 + intensity * 0.7)
                elif base_color == 'blue':
                    color = plt.cm.Blues(0.3 + intensity * 0.7)
                elif base_color == 'red':
                    color = plt.cm.Reds(0.3 + intensity * 0.7)
                elif base_color == 'orange':
                    color = plt.cm.Oranges(0.3 + intensity * 0.7)
                elif base_color == 'purple':
                    color = plt.cm.Purples(0.3 + intensity * 0.7)
                else:
                    color = base_color
                
                # Draw edge
                line = ax.plot([coord1[0], coord2[0]], 
                              [coord1[1], coord2[1]], 
                              color=color, 
                              linewidth=line_width, 
                              alpha=alpha,
                              zorder=3)[0]
                edge_lines.append(line)
                
                # Add edge number annotation for top 5 edges
                if i < 5:
                    mid_x = (coord1[0] + coord2[0]) / 2
                    mid_y = (coord1[1] + coord2[1]) / 2
                    ax.annotate(f'{edge_num}\n({count})', (mid_x, mid_y), 
                               fontsize=8, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.2", 
                                       facecolor='white', alpha=0.9),
                               zorder=4)
        
        # Add mode-specific statistics
        total_usage = sum(count for _, count in popular_edges)
        ax.text(0.02, 0.98, f'{mode.title()}\nTotal: {total_usage} trips\nEdges: {len(popular_edges)}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def update_statistics_plot(self):
        """Update the statistics plot with MODE-SPECIFIC edge usage data"""
        ax = self.axes[5]
        ax.clear()
        ax.set_title('Top Edges by Transport Mode', fontsize=14, fontweight='bold')
        
        if not self.live_tracker:
            ax.text(0.5, 0.5, 'No tracker data', ha='center', va='center', 
                   transform=ax.transAxes)
            return
        
        # Create mode-specific statistics text
        stats_text = ""
        
        # Show top edges for each mode
        for mode in self.modes_to_show:
            mode_edges = self.live_tracker.get_popular_edges_by_mode(mode, 3)  # Top 3 per mode
            total = sum(count for _, count in mode_edges)
            
            if mode_edges:
                stats_text += f"🚗 {mode.upper()} ({total} total):\n"
                for i, (edge_num, count) in enumerate(mode_edges, 1):
                    if edge_num in self.live_tracker.number_to_edge:
                        edge_info = self.live_tracker.number_to_edge[edge_num]
                        description = edge_info['description']
                        stats_text += f"  {i}. Edge {edge_num}: {count} uses\n"
                        stats_text += f"     {description}\n"
                stats_text += "\n"
            else:
                stats_text += f"🚗 {mode.upper()}: No routes yet\n\n"
        
        # Add overall statistics
        stats = self.live_tracker.get_live_statistics()
        all_edges = self.live_tracker.get_popular_edges_by_mode('all', 10)
        
        stats_text += f"🎯 OVERALL SUMMARY:\n"
        stats_text += f"Step: {stats['step']}\n"
        stats_text += f"Active: {stats['active_agents']}/{stats['total_agents']}\n"
        stats_text += f"Total edges used: {len(all_edges)}\n"
        stats_text += f"Network edges: {len(self.live_tracker.edge_to_number)}\n"
        
        # Calculate utilization
        if len(self.live_tracker.edge_to_number) > 0:
            utilization = len(all_edges) / len(self.live_tracker.edge_to_number) * 100
            stats_text += f"Utilization: {utilization:.1f}%\n"
        
        # Add mode totals summary
        stats_text += f"\n📊 MODE TOTALS:\n"
        for mode in self.modes_to_show:
            mode_edges = self.live_tracker.get_popular_edges_by_mode(mode, 100)  # Get all
            total = sum(count for _, count in mode_edges)
            edge_count = len(mode_edges)
            stats_text += f"{mode:8s}: {total:3d} trips, {edge_count:2d} edges\n"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.9))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def save_current_state(self, filename="multi_mode_route_heatmap.png"):
        """Save current visualization state"""
        self.update_visualization()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📸 Saved multi-mode route heatmap as {filename}")
    
    def print_edge_summary(self):
        """Print detailed edge usage summary to console"""
        if not self.live_tracker:
            print("No live tracker available!")
            return
        
        print("\n" + "="*80)
        print("🎯 DETAILED MODE-SPECIFIC EDGE USAGE SUMMARY")
        print("="*80)
        
        # Print top edges for each mode with more details
        for mode in self.modes_to_show:
            edges = self.live_tracker.get_popular_edges_by_mode(mode, 8)  # Top 8 per mode
            total_trips = sum(count for _, count in edges)
            
            if edges:
                print(f"\n🚗 {mode.upper()} ROUTES - {total_trips} total trips, {len(edges)} edges used:")
                for i, (edge_num, count) in enumerate(edges, 1):
                    if edge_num in self.live_tracker.number_to_edge:
                        edge_info = self.live_tracker.number_to_edge[edge_num]
                        percentage = (count / total_trips * 100) if total_trips > 0 else 0
                        print(f"  {i:2d}. Edge {edge_num:2d}: {count:3d} uses ({percentage:4.1f}%) - {edge_info['description']}")
            else:
                print(f"\n🚗 {mode.upper()}: No routes recorded")
        
        # Print overall network utilization
        all_edges = self.live_tracker.get_popular_edges_by_mode('all', 20)
        total_network_edges = len(self.live_tracker.edge_to_number)
        used_edges = len(all_edges)
        utilization = (used_edges / total_network_edges * 100) if total_network_edges > 0 else 0
        
        print(f"\n🌟 NETWORK UTILIZATION ANALYSIS:")
        print(f"   Total network edges: {total_network_edges}")
        print(f"   Edges actually used: {used_edges}")
        print(f"   Network utilization: {utilization:.1f}%")
        
        if all_edges:
            total_usage = sum(count for _, count in all_edges)
            print(f"   Total edge crossings: {total_usage}")
            print(f"   Average uses per active edge: {total_usage/used_edges:.1f}")
        
        # Print busiest edges overall
        print(f"\n🔥 TOP 10 BUSIEST EDGES (ALL MODES):")
        for i, (edge_num, count) in enumerate(all_edges[:10], 1):
            if edge_num in self.live_tracker.number_to_edge:
                edge_info = self.live_tracker.number_to_edge[edge_num]
                print(f"  {i:2d}. Edge {edge_num:2d}: {count:3d} uses - {edge_info['description']}")
        
        # Print mode comparison
        print(f"\n📊 MODE COMPARISON:")
        mode_stats = []
        for mode in self.modes_to_show:
            mode_edges = self.live_tracker.get_popular_edges_by_mode(mode, 100)
            total_trips = sum(count for _, count in mode_edges)
            unique_edges = len(mode_edges)
            mode_stats.append((mode, total_trips, unique_edges))
        
        # Sort by total trips
        mode_stats.sort(key=lambda x: x[1], reverse=True)
        
        for i, (mode, trips, edges) in enumerate(mode_stats, 1):
            avg_per_edge = trips / edges if edges > 0 else 0
            print(f"  {i}. {mode:12s}: {trips:3d} trips, {edges:2d} edges, {avg_per_edge:4.1f} avg/edge")
        
        print("="*80)

class LiveDashboard:
    """Complete live dashboard combining multiple visualizations"""
    
    def __init__(self, model):
        self.model = model
        self.live_tracker = getattr(model, 'live_tracker', None)
        
        # Create dashboard layout - use the multi-mode visualizer
        self.route_viz = RouteHeatmapVisualizer(model)
        
    def update_dashboard(self):
        """Update dashboard - delegate to route visualizer"""
        self.route_viz.update_visualization()
        
    def save_dashboard(self, filename="live_dashboard.png"):
        """Save dashboard state"""
        self.route_viz.save_current_state(filename)

# Easy integration functions
def add_live_route_visualization(model):
    """Add live route visualization to your model"""
    from live_agent_tracker import add_live_tracking_to_model
    
    # Check if tracking already exists
    if not hasattr(model, 'live_tracker'):
        tracker = add_live_tracking_to_model(model)
    else:
        tracker = model.live_tracker
    
    # Add route visualizer
    route_viz = RouteHeatmapVisualizer(model)
    
    # DON'T override step again - just use the existing enhanced step
    model.route_visualizer = route_viz
    
    return route_viz

def create_live_dashboard(model):
    """Create live dashboard for your model"""
    from live_agent_tracker import add_live_tracking_to_model
    
    # Add tracking if not already present
    if not hasattr(model, 'live_tracker'):
        add_live_tracking_to_model(model)
    
    # Create dashboard
    dashboard = LiveDashboard(model)
    
    # Add to model
    model.live_dashboard = dashboard
    
    return dashboard

if __name__ == "__main__":
    print("🗺️ Multi-Mode Route Heatmap Visualizer")
    print("Usage:")
    print("  viz = add_live_route_visualization(your_model)")
    print("  dashboard = create_live_dashboard(your_model)")