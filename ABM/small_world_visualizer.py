import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from collections import defaultdict

class SmallWorldNetworkVisualizer:
    """
    Specialized visualizer for small-world transport networks
    Focuses on showing shortcuts, accessibility changes, and equity implications
    """
    
    def __init__(self, network_manager, rewiring_probability=0.1):
        self.network_manager = network_manager
        self.graph = network_manager.active_network
        self.base_network = network_manager.base_network
        self.spatial_mapper = network_manager.spatial_mapper
        self.rewiring_probability = rewiring_probability
        
        # Color schemes for different visualizations
        self.colors = {
            'regular': '#2E86C1',      # Blue for regular edges
            'shortcut': '#E74C3C',     # Red for shortcuts
            'hierarchy': '#F39C12',    # Orange for hierarchy edges
            'connectivity': '#27AE60', # Green for connectivity edges
            'major_hub': '#8E44AD',    # Purple for major hubs
            'transport_hub': '#E67E22', # Orange for transport hubs
            'local_station': '#16A085', # Teal for local stations
            'peripheral': '#E74C3C'     # Red for peripheral areas
        }
        
        print(f"🎨 Small-World Visualizer initialized (p={rewiring_probability:.3f})")
    
    def visualize_network_structure(self, save_path=None, show_labels=True):
        """Create comprehensive network structure visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Network Overview with Edge Types
        self._plot_network_overview(axes[0, 0], show_labels)
        
        # 2. Shortcut Analysis
        self._plot_shortcut_analysis(axes[0, 1])
        
        # 3. Accessibility Heat Map
        self._plot_accessibility_heatmap(axes[1, 0])
        
        # 4. Network Statistics
        self._plot_network_statistics(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📸 Saved visualization: {save_path}")
        
        plt.show()
        
        return fig
    
    def _plot_network_overview(self, ax, show_labels):
        """Plot network overview showing different edge types and node hierarchies"""
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 80)
        ax.set_aspect('equal')
        ax.set_title(f'Small-World Network Structure (p={self.rewiring_probability:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot edges by type
        edge_counts = {'regular': 0, 'shortcut': 0, 'hierarchy': 0, 'connectivity': 0}
        
        for u, v, data in self.graph.edges(data=True):
            u_coord = self.spatial_mapper.node_to_grid.get(u)
            v_coord = self.spatial_mapper.node_to_grid.get(v)
            
            if u_coord and v_coord:
                edge_type = data.get('edge_type', 'regular')
                edge_counts[edge_type] += 1
                
                color = self.colors.get(edge_type, self.colors['regular'])
                
                if edge_type == 'shortcut':
                    # Emphasize shortcuts
                    ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 
                           color=color, linewidth=3, alpha=0.8, zorder=3)
                elif edge_type == 'hierarchy':
                    ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 
                           color=color, linewidth=2, alpha=0.7, zorder=2)
                else:
                    ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 
                           color=color, linewidth=1, alpha=0.5, zorder=1)
        
        # Plot nodes by hierarchy
        node_counts = {'major_hub': 0, 'transport_hub': 0, 'local_station': 0, 'other': 0}
        
        for node_id, coord in self.spatial_mapper.node_to_grid.items():
            node_data = self.base_network.nodes.get(node_id)
            if node_data:
                node_type = node_data.node_type.value
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
                
                if node_type == 'major_hub':
                    ax.plot(coord[0], coord[1], 'o', color=self.colors['major_hub'], 
                           markersize=12, zorder=5, markeredgecolor='black', markeredgewidth=1)
                    if show_labels:
                        ax.annotate(node_id, (coord[0], coord[1]), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=8, 
                                  bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                elif node_type == 'transport_hub':
                    ax.plot(coord[0], coord[1], 's', color=self.colors['transport_hub'], 
                           markersize=8, zorder=4, markeredgecolor='black', markeredgewidth=0.5)
                else:
                    ax.plot(coord[0], coord[1], '^', color=self.colors['local_station'], 
                           markersize=6, zorder=3, alpha=0.7)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors['shortcut'], linewidth=3, label=f'Shortcuts ({edge_counts["shortcut"]})'),
            plt.Line2D([0], [0], color=self.colors['regular'], linewidth=1, label=f'Regular ({edge_counts["regular"]})'),
            plt.Line2D([0], [0], color=self.colors['hierarchy'], linewidth=2, label=f'Hierarchy ({edge_counts["hierarchy"]})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['major_hub'], 
                      markersize=8, label=f'Major Hubs ({node_counts.get("major_hub", 0)})'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['transport_hub'], 
                      markersize=6, label=f'Transport Hubs ({node_counts.get("transport_hub", 0)})'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=self.colors['local_station'], 
                      markersize=6, label=f'Local Stations ({node_counts.get("local_station", 0)})')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    def _plot_shortcut_analysis(self, ax):
        """Analyze and visualize shortcut effectiveness"""
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 80)
        ax.set_aspect('equal')
        ax.set_title('Shortcut Impact Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Find shortcuts and analyze their impact
        shortcuts = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'shortcut':
                u_coord = self.spatial_mapper.node_to_grid.get(u)
                v_coord = self.spatial_mapper.node_to_grid.get(v)
                if u_coord and v_coord:
                    distance = np.sqrt((u_coord[0] - v_coord[0])**2 + (u_coord[1] - v_coord[1])**2)
                    shortcuts.append((u, v, u_coord, v_coord, distance))
        
        # Draw base network lightly
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') != 'shortcut':
                u_coord = self.spatial_mapper.node_to_grid.get(u)
                v_coord = self.spatial_mapper.node_to_grid.get(v)
                if u_coord and v_coord:
                    ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 
                           color='lightgray', linewidth=0.5, alpha=0.3)
        
        # Highlight shortcuts with impact analysis
        if shortcuts:
            # Color shortcuts by their length (longer = more impactful)
            distances = [s[4] for s in shortcuts]
            norm = Normalize(vmin=min(distances), vmax=max(distances))
            cmap = plt.cm.Reds
            
            for u, v, u_coord, v_coord, distance in shortcuts:
                color = cmap(norm(distance))
                ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 
                       color=color, linewidth=4, alpha=0.8, zorder=3)
                
                # Add impact annotation for longest shortcuts
                if distance > np.percentile(distances, 75):  # Top 25% longest
                    mid_x, mid_y = (u_coord[0] + v_coord[0])/2, (u_coord[1] + v_coord[1])/2
                    ax.annotate(f'{distance:.1f}', (mid_x, mid_y), 
                              bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8),
                              fontsize=8, ha='center')
            
            # Add colorbar for shortcut distances
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label('Shortcut Distance', rotation=270, labelpad=15)
        
        # Draw nodes
        for node_id, coord in self.spatial_mapper.node_to_grid.items():
            node_data = self.base_network.nodes.get(node_id)
            if node_data:
                node_type = node_data.node_type.value
                if node_type == 'major_hub':
                    ax.plot(coord[0], coord[1], 'o', color=self.colors['major_hub'], 
                           markersize=10, zorder=5)
                else:
                    ax.plot(coord[0], coord[1], 'o', color='darkgray', 
                           markersize=4, zorder=4, alpha=0.7)
        
        # Add statistics
        if shortcuts:
            stats_text = f"""
Shortcut Analysis:
• Total shortcuts: {len(shortcuts)}
• Avg shortcut length: {np.mean(distances):.1f}
• Max shortcut length: {max(distances):.1f}
• Shortcut efficiency: {len(shortcuts)/self.graph.number_of_edges()*100:.1f}%
            """
        else:
            stats_text = "No shortcuts found"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    def _plot_accessibility_heatmap(self, ax):
        """Create accessibility heat map showing impact of small-world structure"""
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 80)
        ax.set_aspect('equal')
        ax.set_title('Accessibility Heat Map', fontsize=14, fontweight='bold')
        
        # Calculate accessibility for each node (inverse of average path length to all other nodes)
        accessibility_scores = {}
        
        try:
            # Calculate all-pairs shortest paths
            path_lengths = dict(nx.all_pairs_shortest_path_length(self.graph))
            
            for node in self.graph.nodes():
                if node in path_lengths:
                    lengths = list(path_lengths[node].values())
                    if lengths:
                        avg_path_length = np.mean(lengths)
                        # Accessibility = inverse of average path length
                        accessibility_scores[node] = 1 / avg_path_length if avg_path_length > 0 else 0
                    else:
                        accessibility_scores[node] = 0
                else:
                    accessibility_scores[node] = 0
        
        except Exception as e:
            print(f"Warning: Could not calculate accessibility scores: {e}")
            # Fallback: use degree centrality
            centrality = nx.degree_centrality(self.graph)
            accessibility_scores = centrality
        
        if accessibility_scores:
            # Normalize scores
            max_score = max(accessibility_scores.values()) if accessibility_scores.values() else 1
            min_score = min(accessibility_scores.values()) if accessibility_scores.values() else 0
            
            # Create heat map
            for node_id, coord in self.spatial_mapper.node_to_grid.items():
                if node_id in accessibility_scores:
                    score = accessibility_scores[node_id]
                    normalized_score = (score - min_score) / (max_score - min_score) if max_score > min_score else 0
                    
                    # Color from blue (low accessibility) to red (high accessibility)
                    color = plt.cm.RdYlBu_r(normalized_score)
                    size = 50 + normalized_score * 150  # Size based on accessibility
                    
                    ax.scatter(coord[0], coord[1], c=[color], s=size, 
                             alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)
        
        # Draw network edges lightly
        for u, v in self.graph.edges():
            u_coord = self.spatial_mapper.node_to_grid.get(u)
            v_coord = self.spatial_mapper.node_to_grid.get(v)
            if u_coord and v_coord:
                ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 
                       color='lightgray', linewidth=0.5, alpha=0.3)
        
        # Identify peripheral areas (low accessibility)
        if accessibility_scores:
            threshold = np.percentile(list(accessibility_scores.values()), 25)  # Bottom 25%
            peripheral_nodes = [node for node, score in accessibility_scores.items() if score <= threshold]
            
            # Highlight peripheral areas
            for node in peripheral_nodes:
                coord = self.spatial_mapper.node_to_grid.get(node)
                if coord:
                    circle = plt.Circle((coord[0], coord[1]), 3, 
                                      color=self.colors['peripheral'], fill=False, 
                                      linewidth=2, linestyle='--', alpha=0.8)
                    ax.add_patch(circle)
            
            # Add accessibility statistics
            stats_text = f"""
Accessibility Analysis:
• High access nodes: {sum(1 for s in accessibility_scores.values() if s > np.percentile(list(accessibility_scores.values()), 75))}
• Low access nodes: {len(peripheral_nodes)}
• Accessibility range: {min_score:.3f} - {max_score:.3f}
• Std deviation: {np.std(list(accessibility_scores.values())):.3f}

🔴 Dashed circles = Peripheral areas
            """
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.9))
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Accessibility Score', rotation=270, labelpad=15)
    
    def _plot_network_statistics(self, ax):
        """Plot comprehensive network statistics"""
        
        ax.axis('off')
        ax.set_title('Network Analysis Statistics', fontsize=14, fontweight='bold')
        
        # Calculate comprehensive statistics
        stats = self._calculate_comprehensive_stats()
        
        # Create formatted statistics display
        stats_text = f"""
📊 SMALL-WORLD NETWORK STATISTICS (p={self.rewiring_probability:.3f})

🌐 BASIC PROPERTIES:
   • Nodes: {stats['nodes']}
   • Edges: {stats['edges']}
   • Average Degree: {stats['avg_degree']:.2f}
   • Density: {stats['density']:.4f}

🔗 SMALL-WORLD METRICS:
   • Clustering Coefficient: {stats['clustering']:.3f}
   • Average Path Length: {stats['avg_path_length']:.2f}
   • Small-World Coefficient (σ): {stats['sigma']:.3f}
   • Gamma (C/C_random): {stats['gamma']:.3f}
   • Lambda (L/L_random): {stats['lambda']:.3f}

⚡ CONNECTIVITY ANALYSIS:
   • Network Diameter: {stats['diameter']}
   • Is Connected: {stats['is_connected']}
   • Number of Components: {stats['components']}
   • Largest Component: {stats['largest_component_size']} nodes

🎯 SHORTCUT ANALYSIS:
   • Total Shortcuts: {stats['shortcuts']}
   • Shortcut Percentage: {stats['shortcut_percentage']:.1f}%
   • Avg Shortcut Length: {stats['avg_shortcut_length']:.2f}

🏛️ HIERARCHY PRESERVATION:
   • Major Hub Connections: {stats['major_hub_edges']}
   • Transport Hub Connections: {stats['transport_hub_edges']}
   • Local Station Connections: {stats['local_station_edges']}

📈 CENTRALITY MEASURES:
   • Max Betweenness: {stats['max_betweenness']:.3f}
   • Avg Betweenness: {stats['avg_betweenness']:.3f}
   • Betweenness Std: {stats['betweenness_std']:.3f}
   • Max Closeness: {stats['max_closeness']:.3f}

🎯 EQUITY IMPLICATIONS:
   • Peripheral Nodes: {stats['peripheral_nodes']} ({stats['peripheral_percentage']:.1f}%)
   • Accessibility Range: {stats['accessibility_range']:.3f}
   • Accessibility Gini: {stats['accessibility_gini']:.3f}

💡 RESEARCH INSIGHTS:
   {"✅ Good small-world properties" if stats['sigma'] > 1 else "❌ Limited small-world effect"}
   {"✅ Shortcuts benefit peripheral areas" if stats['peripheral_benefit'] > 0.1 else "⚠️ Limited peripheral benefit"}
   {"✅ Maintains transport hierarchy" if stats['hierarchy_preserved'] else "⚠️ Hierarchy disrupted"}
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    
    def _calculate_comprehensive_stats(self):
        """Calculate comprehensive network statistics"""
        
        graph = self.graph
        stats = {}
        
        # Basic properties
        stats['nodes'] = graph.number_of_nodes()
        stats['edges'] = graph.number_of_edges()
        stats['avg_degree'] = 2 * stats['edges'] / stats['nodes'] if stats['nodes'] > 0 else 0
        stats['density'] = nx.density(graph)
        
        # Connectivity
        stats['is_connected'] = nx.is_connected(graph)
        stats['components'] = nx.number_connected_components(graph)
        if stats['is_connected']:
            stats['diameter'] = nx.diameter(graph)
            stats['avg_path_length'] = nx.average_shortest_path_length(graph)
        else:
            largest_cc = max(nx.connected_components(graph), key=len)
            largest_subgraph = graph.subgraph(largest_cc)
            stats['diameter'] = nx.diameter(largest_subgraph)
            stats['avg_path_length'] = nx.average_shortest_path_length(largest_subgraph)
        
        stats['largest_component_size'] = len(max(nx.connected_components(graph), key=len))
        
        # Small-world metrics
        stats['clustering'] = nx.average_clustering(graph)
        
        # Calculate small-world coefficient
        try:
            n = stats['nodes']
            m = stats['edges']
            p = 2 * m / (n * (n - 1)) if n > 1 else 0
            
            random_graph = nx.erdos_renyi_graph(n, p)
            if nx.is_connected(random_graph):
                C_random = nx.average_clustering(random_graph)
                L_random = nx.average_shortest_path_length(random_graph)
                
                stats['gamma'] = stats['clustering'] / C_random if C_random > 0 else 0
                stats['lambda'] = stats['avg_path_length'] / L_random if L_random > 0 else 0
                stats['sigma'] = stats['gamma'] / stats['lambda'] if stats['lambda'] > 0 else 0
            else:
                stats['gamma'] = stats['lambda'] = stats['sigma'] = 0
        except:
            stats['gamma'] = stats['lambda'] = stats['sigma'] = 0
        
        # Shortcut analysis
        shortcuts = [data for u, v, data in graph.edges(data=True) 
                    if data.get('edge_type') == 'shortcut']
        stats['shortcuts'] = len(shortcuts)
        stats['shortcut_percentage'] = (len(shortcuts) / stats['edges'] * 100) if stats['edges'] > 0 else 0
        
        if shortcuts:
            shortcut_lengths = [data.get('distance', 0) for data in shortcuts]
            stats['avg_shortcut_length'] = np.mean(shortcut_lengths)
        else:
            stats['avg_shortcut_length'] = 0
        
        # Hierarchy analysis
        edge_counts = {'major_hub': 0, 'transport_hub': 0, 'local_station': 0}
        for u, v in graph.edges():
            u_type = self.base_network.nodes[u].node_type.value
            v_type = self.base_network.nodes[v].node_type.value
            
            if 'major_hub' in [u_type, v_type]:
                edge_counts['major_hub'] += 1
            elif 'transport_hub' in [u_type, v_type]:
                edge_counts['transport_hub'] += 1
            else:
                edge_counts['local_station'] += 1
        
        stats['major_hub_edges'] = edge_counts['major_hub']
        stats['transport_hub_edges'] = edge_counts['transport_hub']
        stats['local_station_edges'] = edge_counts['local_station']
        stats['hierarchy_preserved'] = edge_counts['major_hub'] > 0
        
        # Centrality measures
        try:
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            
            stats['max_betweenness'] = max(betweenness.values()) if betweenness else 0
            stats['avg_betweenness'] = np.mean(list(betweenness.values())) if betweenness else 0
            stats['betweenness_std'] = np.std(list(betweenness.values())) if betweenness else 0
            stats['max_closeness'] = max(closeness.values()) if closeness else 0
        except:
            stats['max_betweenness'] = stats['avg_betweenness'] = stats['betweenness_std'] = stats['max_closeness'] = 0
        
        # Accessibility and equity analysis
        try:
            if stats['is_connected']:
                path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
                accessibility_scores = []
                
                for node in graph.nodes():
                    if node in path_lengths:
                        lengths = list(path_lengths[node].values())
                        if lengths:
                            avg_length = np.mean(lengths)
                            accessibility = 1 / avg_length if avg_length > 0 else 0
                            accessibility_scores.append(accessibility)
                
                if accessibility_scores:
                    stats['accessibility_range'] = max(accessibility_scores) - min(accessibility_scores)
                    stats['accessibility_gini'] = self._calculate_gini(accessibility_scores)
                    
                    # Peripheral analysis
                    threshold = np.percentile(accessibility_scores, 25)
                    peripheral_count = sum(1 for score in accessibility_scores if score <= threshold)
                    stats['peripheral_nodes'] = peripheral_count
                    stats['peripheral_percentage'] = (peripheral_count / len(accessibility_scores) * 100)
                    stats['peripheral_benefit'] = np.mean([score for score in accessibility_scores if score <= threshold])
                else:
                    stats['accessibility_range'] = stats['accessibility_gini'] = 0
                    stats['peripheral_nodes'] = stats['peripheral_percentage'] = stats['peripheral_benefit'] = 0
            else:
                stats['accessibility_range'] = stats['accessibility_gini'] = 0
                stats['peripheral_nodes'] = stats['peripheral_percentage'] = stats['peripheral_benefit'] = 0
        except:
            stats['accessibility_range'] = stats['accessibility_gini'] = 0
            stats['peripheral_nodes'] = stats['peripheral_percentage'] = stats['peripheral_benefit'] = 0
        
        return stats
    
    def _calculate_gini(self, values):
        """Calculate Gini coefficient"""
        if not values or len(values) == 0:
            return 0
        
        values = np.array(values)
        values = values[~np.isnan(values)]
        
        if len(values) == 0 or np.all(values == 0):
            return 0
        
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        
        if cumsum[-1] == 0:
            return 0
        
        gini = (2 * np.sum((np.arange(1, n + 1)) * values)) / (n * cumsum[-1]) - (n + 1) / n
        return max(0, gini)
    
    def create_small_world_comparison(self, other_networks, save_path=None):
        """Compare this small-world network with other networks"""
        
        n_networks = len(other_networks) + 1
        fig, axes = plt.subplots(2, (n_networks + 1) // 2, figsize=(6 * n_networks, 12))
        
        if n_networks == 1:
            axes = [axes]
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        # Plot this network
        self._plot_comparison_network(axes[0, 0], "Current", self.rewiring_probability)
        
        # Plot other networks
        for i, (p_value, network_manager) in enumerate(other_networks):
            row = (i + 1) // ((n_networks + 1) // 2)
            col = (i + 1) % ((n_networks + 1) // 2)
            
            visualizer = SmallWorldNetworkVisualizer(network_manager, p_value)
            visualizer._plot_comparison_network(axes[row, col], f"p={p_value:.3f}", p_value)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_comparison_network(self, ax, title, p_value):
        """Plot network for comparison purposes"""
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 80)
        ax.set_aspect('equal')
        ax.set_title(f'{title} (p={p_value:.3f})', fontsize=12, fontweight='bold')
        
        # Plot edges
        for u, v, data in self.graph.edges(data=True):
            u_coord = self.spatial_mapper.node_to_grid.get(u)
            v_coord = self.spatial_mapper.node_to_grid.get(v)
            
            if u_coord and v_coord:
                edge_type = data.get('edge_type', 'regular')
                
                if edge_type == 'shortcut':
                    ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 
                           color=self.colors['shortcut'], linewidth=2, alpha=0.8)
                else:
                    ax.plot([u_coord[0], v_coord[0]], [u_coord[1], v_coord[1]], 
                           color=self.colors['regular'], linewidth=0.5, alpha=0.5)
        
        # Plot nodes
        for node_id, coord in self.spatial_mapper.node_to_grid.items():
            node_data = self.base_network.nodes.get(node_id)
            if node_data:
                node_type = node_data.node_type.value
                
                if node_type == 'major_hub':
                    ax.plot(coord[0], coord[1], 'o', color=self.colors['major_hub'], markersize=8)
                elif node_type == 'transport_hub':
                    ax.plot(coord[0], coord[1], 's', color=self.colors['transport_hub'], markersize=6)
                else:
                    ax.plot(coord[0], coord[1], '^', color=self.colors['local_station'], markersize=4, alpha=0.7)


def create_small_world_visualization(network_manager, rewiring_probability=0.1, save_path=None):
    """Convenience function to create small-world visualization"""
    
    visualizer = SmallWorldNetworkVisualizer(network_manager, rewiring_probability)
    return visualizer.visualize_network_structure(save_path)


if __name__ == "__main__":
    print("🎨 Small-World Network Visualizer")
    print("Usage: visualizer = SmallWorldNetworkVisualizer(network_manager, p_value)")
    print("       visualizer.visualize_network_structure()")