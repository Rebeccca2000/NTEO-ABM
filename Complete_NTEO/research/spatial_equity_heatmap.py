"""
CLEAN VERSION of spatial_equity_heatmap.py - Only Essential Functions
DELETE all the other functions and replace with this clean version
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

class SpatialEquityVisualizer:
    """Clean spatial equity visualization focused on parameter comparison"""
    
    def __init__(self, grid_width=100, grid_height=80):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.sydney_regions = self._define_sydney_regions()
        
    def _define_sydney_regions(self):
        """Define Sydney geographical regions with exact coordinates"""
        return {
            'CBD_Inner': {
                'bounds': (45, 55, 35, 50),
                'center': (50, 42),
                'color': '#FF6B6B'
            },
            'Western_Sydney': {
                'bounds': (0, 35, 15, 60),
                'center': (17, 40),
                'color': '#4ECDC4'
            },
            'Eastern_Suburbs': {
                'bounds': (55, 85, 20, 65),
                'center': (70, 35),
                'color': '#45B7D1'
            },
            'North_Shore': {
                'bounds': (45, 75, 50, 80),
                'center': (58, 65),
                'color': '#96CEB4'
            },
            'South_Sydney': {
                'bounds': (30, 70, 0, 35),
                'center': (50, 20),
                'color': '#FECA57'
            }
        }
    
    # KEEP: This function is useful for single topology analysis
    def create_comprehensive_heatmap(self, spatial_equity_data, topology_name, save_path=None):
        """Create comprehensive GEOGRAPHIC heat map visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Geographic Spatial Equity Analysis - {topology_name.title()} Network', 
                     fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        metrics = [
            ('mode_choice_equity', 'Mode Choice Equity'),
            ('travel_time_equity', 'Travel Time Equity'), 
            ('system_efficiency', 'System Efficiency'),
            ('commuter_count', 'Commuter Density')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes_flat[idx]
            self._create_geographic_heatmap(ax, spatial_equity_data, metric, title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Geographic spatial equity heatmap saved: {save_path}")
        
        return fig
    
    # KEEP: Helper function for geographic visualization
    def _create_geographic_heatmap(self, ax, spatial_equity_data, metric, title):
        """Create individual GEOGRAPHIC heat map for a specific metric"""
        
        # Create 100x80 grid for Sydney geography
        grid = np.zeros((self.grid_height, self.grid_width))
        
        # Fill regions with metric values based on actual geographic bounds
        for region_name, data in spatial_equity_data.items():
            if region_name in self.sydney_regions:
                region_info = self.sydney_regions[region_name]
                bounds = region_info['bounds']
                x_min, x_max, y_min, y_max = bounds
                
                metric_value = data.get(metric, 0.0)
                
                # Fill the actual geographic area in the grid
                grid[y_min:y_max, x_min:x_max] = metric_value
        
        # Create the geographic heat map
        im = ax.imshow(grid, cmap='RdYlBu_r', aspect='auto', origin='lower', 
                      extent=[0, self.grid_width, 0, self.grid_height])
        
        # Add region boundaries
        for region_name, region_info in self.sydney_regions.items():
            bounds = region_info['bounds']
            center = region_info['center']
            x_min, x_max, y_min, y_max = bounds
            
            # Draw boundary rectangle
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                   linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            
            # Add region label
            ax.text(center[0], center[1], region_name.replace('_', '\n'), 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label=title, shrink=0.8)
        ax.set_title(title)
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        
    # NEW: Your main parameter comparison function
    def create_parameter_comparison_analysis(self, all_configs_data, topology_name, save_path=None):
        """Create parameter comparison analysis showing all configurations side by side"""
        
        config_names = list(all_configs_data.keys())
        n_configs = len(config_names)
        
        if n_configs == 0:
            print("No configuration data provided")
            return None
        
        # Create figure with 3 rows (Network Structure, Shortcut Impact, Accessibility)
        fig, axes = plt.subplots(3, n_configs, figsize=(5*n_configs, 15))
        fig.suptitle(f'{topology_name.title()} Parameter Comparison Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Handle single configuration case
        if n_configs == 1:
            axes = axes.reshape(3, 1)
        
        # Extract parameter values for subplot titles
        parameter_values = []
        for config_name in config_names:
            # Extract parameter value from config name (e.g., "degree_constrained_4" -> "4")
            if '_' in config_name:
                param_val = config_name.split('_')[-1]
                parameter_values.append(param_val)
            else:
                parameter_values.append(config_name)
        
        for i, config_name in enumerate(config_names):
            config_data = all_configs_data[config_name]
            param_val = parameter_values[i]
            
            # Row 1: Network Structure Analysis
            ax_network = axes[0, i]
            self._plot_parameter_network_structure(
                ax_network, config_data, f"{topology_name} = {param_val}"
            )
            
            # Row 2: Shortcut Impact Analysis  
            ax_shortcut = axes[1, i]
            self._plot_parameter_shortcut_impact(
                ax_shortcut, config_data, f"Shortcuts - {param_val}"
            )
            
            # Row 3: Accessibility Heatmap
            ax_access = axes[2, i]
            self._plot_parameter_accessibility_heatmap(
                ax_access, config_data, f"Accessibility - {param_val}"
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Parameter comparison saved: {save_path}")
        
        return fig

    def _plot_parameter_network_structure(self, ax, config_data, title):
        """Plot network structure for one parameter configuration"""
        
        network_structure = config_data.get('network_structure', {})
        
        if not network_structure:
            ax.text(0.5, 0.5, 'No Network Data', ha='center', va='center', transform=ax.transAxes)
            return
        
        nodes = network_structure.get('nodes', {})
        shortcuts = network_structure.get('shortcuts', [])
        original_edges = network_structure.get('original_edges', [])
        
        # Plot original edges (blue)
        for edge in original_edges:
            from_node, to_node = edge['from_node'], edge['to_node']
            if from_node in nodes and to_node in nodes:
                from_coords = nodes[from_node]['coordinates']
                to_coords = nodes[to_node]['coordinates']
                ax.plot([from_coords[0], to_coords[0]], [from_coords[1], to_coords[1]], 
                       'b-', alpha=0.4, linewidth=2)
        
        # Plot shortcuts (red) with thickness based on count
        for edge in shortcuts:
            from_node, to_node = edge['from_node'], edge['to_node']
            if from_node in nodes and to_node in nodes:
                from_coords = nodes[from_node]['coordinates']
                to_coords = nodes[to_node]['coordinates']
                ax.plot([from_coords[0], to_coords[0]], [from_coords[1], to_coords[1]], 
                       'r-', alpha=0.8, linewidth=3)
        
        # Plot nodes with size based on degree
        for node_id, node_data in nodes.items():
            coords = node_data['coordinates']
            degree = node_data['degree']
            node_type = node_data['node_type']
            
            # Size based on degree
            marker_size = max(6, min(15, degree * 2))
            
            # Color based on type
            if 'major' in node_type.lower():
                color = 'red'
                marker = 's'
            elif 'transport' in node_type.lower():
                color = 'orange'
                marker = 'o'
            else:
                color = 'blue'
                marker = '^'
            
            ax.plot(coords[0], coords[1], marker, color=color, markersize=marker_size, 
                   markeredgecolor='black')
        
        # Add statistics
        num_shortcuts = len(shortcuts)
        num_original = len(original_edges)
        total_edges = num_shortcuts + num_original
        shortcut_ratio = (num_shortcuts / total_edges * 100) if total_edges > 0 else 0
        
        stats_text = f"Shortcuts: {num_shortcuts}\nRatio: {shortcut_ratio:.1f}%\nNodes: {len(nodes)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 80)
        ax.set_title(title, fontweight='bold')

    def _plot_parameter_shortcut_impact(self, ax, config_data, title):
        """Plot shortcut impact for one parameter configuration - IMPROVED DEBUG"""
        
        network_structure = config_data.get('network_structure', {})
        spatial_equity = config_data.get('spatial_equity', {})
        
        # DEBUG: Print what data we have

        
        if not network_structure:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get shortcut data
        shortcuts = network_structure.get('shortcuts', [])
        edge_congestion = network_structure.get('edge_congestion', {})
        nodes = network_structure.get('nodes', {})
        
        # DEBUG: Print first few shortcuts
        for i, shortcut in enumerate(shortcuts[:3]):
            print(f"   Shortcut {i}: {shortcut['from_node']}->{shortcut['to_node']} ({shortcut['route_id']})")
        
        if len(shortcuts) == 0:
            ax.text(0.5, 0.5, f'No Shortcuts Found\n(Total edges: {len(network_structure.get("edges", []))})', 
                ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create impact grid
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 80, 40)
        X, Y = np.meshgrid(x, y)
        impact_values = np.zeros_like(X)
        
        # Calculate impact based on shortcut locations and usage
        impact_count = 0
        for edge in shortcuts:
            from_node, to_node = edge['from_node'], edge['to_node']
            edge_key = f"{from_node}_{to_node}"
            usage = edge_congestion.get(edge_key, 1)  # Default usage
            
            if from_node in nodes and to_node in nodes:
                from_coords = nodes[from_node]['coordinates']
                to_coords = nodes[to_node]['coordinates']
                
                # Create impact area around shortcuts
                for i in range(len(X)):
                    for j in range(len(X[i])):
                        dist = min(
                            np.sqrt((X[i,j] - from_coords[0])**2 + (Y[i,j] - from_coords[1])**2),
                            np.sqrt((X[i,j] - to_coords[0])**2 + (Y[i,j] - to_coords[1])**2)
                        )
                        if dist < 15:
                            impact_values[i,j] += usage * (15 - dist) / 15
                            impact_count += 1
        
        # Create heatmap if we have impact
        max_impact = np.max(impact_values)
        print(f"   Max impact value: {max_impact}, Impact points: {impact_count}")
        
        if max_impact > 0:
            im = ax.contourf(X, Y, impact_values, levels=10, cmap='RdYlBu_r')
            plt.colorbar(im, ax=ax, label='Impact', shrink=0.8)
            
            # Add shortcut lines
            for edge in shortcuts:
                from_node, to_node = edge['from_node'], edge['to_node']
                if from_node in nodes and to_node in nodes:
                    from_coords = nodes[from_node]['coordinates']
                    to_coords = nodes[to_node]['coordinates']
                    ax.plot([from_coords[0], to_coords[0]], [from_coords[1], to_coords[1]], 
                        'r-', alpha=0.8, linewidth=2, label='Shortcut' if edge == shortcuts[0] else '')
        else:
            ax.text(0.5, 0.5, f'No Shortcut Impact\n({len(shortcuts)} shortcuts found but no geographic impact)', 
                ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 80)
        ax.set_title(f"{title} ({len(shortcuts)} shortcuts)")

    def _plot_parameter_accessibility_heatmap(self, ax, config_data, title):
        """Plot accessibility heatmap for one parameter configuration"""
        
        spatial_equity = config_data.get('spatial_equity', {})
        
        if not spatial_equity:
            ax.text(0.5, 0.5, 'No Spatial Data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create accessibility grid
        grid = np.zeros((self.grid_height, self.grid_width))
        
        # Fill regions with system efficiency values
        for region_name, data in spatial_equity.items():
            if region_name in self.sydney_regions:
                region_info = self.sydney_regions[region_name]
                bounds = region_info['bounds']
                x_min, x_max, y_min, y_max = bounds
                
                accessibility_value = data.get('system_efficiency', 0.0)
                grid[y_min:y_max, x_min:x_max] = accessibility_value
        
        # Create heatmap
        if np.max(grid) > 0:
            im = ax.imshow(grid, cmap='RdYlGn', aspect='auto', origin='lower', 
                          extent=[0, self.grid_width, 0, self.grid_height])
            plt.colorbar(im, ax=ax, label='Accessibility', shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'No Accessibility Data', ha='center', va='center', transform=ax.transAxes)
        
        # Add region boundaries and labels
        for region_name, region_info in self.sydney_regions.items():
            bounds = region_info['bounds']
            center = region_info['center']
            x_min, x_max, y_min, y_max = bounds
            
            # Draw boundary
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                   linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            
            # Add label
            ax.text(center[0], center[1], region_name.replace('_', '\n'), 
                   ha='center', va='center', fontsize=6, fontweight='bold')
        
        ax.set_xlim(0, self.grid_width)
        ax.set_ylim(0, self.grid_height)
        ax.set_title(title)