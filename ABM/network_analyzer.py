import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Complete_NTEO.topology.network_topology import SydneyNetworkTopology, DegreeConstrainedTopologyGenerator
from network_integration import TwoLayerNetworkManager
import seaborn as sns
from collections import Counter

class NetworkTopologyAnalyzer:
    def __init__(self, degree_constraint=3):
        self.degree = degree_constraint
        self.network_manager = TwoLayerNetworkManager(
            topology_type="degree_constrained",
            degree=degree_constraint,
            grid_width=100,
            grid_height=80
        )
        
    def visualize_network_structure(self, save_path=None):
        """Create comprehensive network visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Network topology with node types
        ax1 = axes[0, 0]
        pos = {node: self.network_manager.base_network.nodes[node].coordinates 
               for node in self.network_manager.active_network.nodes()}
        
        # Color nodes by type
        node_colors = []
        node_sizes = []
        for node in self.network_manager.active_network.nodes():
            node_data = self.network_manager.base_network.nodes[node]
            if node_data.node_type.value == 'major_hub':
                node_colors.append('red')
                node_sizes.append(300)
            elif node_data.node_type.value == 'transport_hub':
                node_colors.append('orange')
                node_sizes.append(200)
            else:
                node_colors.append('lightblue')
                node_sizes.append(100)
        
        # Color edges by transport mode
        edge_colors = []
        edge_widths = []
        for u, v in self.network_manager.active_network.edges():
            edge_data = self.network_manager.active_network[u][v]
            mode = edge_data.get('transport_mode')
            if hasattr(mode, 'value'):
                mode_str = mode.value
            else:
                mode_str = str(mode)
                
            if mode_str == 'train':
                edge_colors.append('green')
                edge_widths.append(3)
            elif mode_str == 'bus':
                edge_colors.append('blue')
                edge_widths.append(2)
            else:
                edge_colors.append('gray')
                edge_widths.append(1)
        
        nx.draw_networkx(self.network_manager.active_network, pos, ax=ax1,
                        node_color=node_colors, node_size=node_sizes,
                        edge_color=edge_colors, width=edge_widths,
                        with_labels=False, alpha=0.8)
        ax1.set_title(f'Network Topology (Degree {self.degree})\nRed=Major Hub, Orange=Transport Hub, Blue=Local')
        ax1.set_aspect('equal')
        
        # 2. Degree distribution
        ax2 = axes[0, 1]
        degrees = [d for n, d in self.network_manager.active_network.degree()]
        degree_counts = Counter(degrees)
        
        ax2.bar(degree_counts.keys(), degree_counts.values(), color='skyblue', alpha=0.7)
        ax2.set_xlabel('Node Degree')
        ax2.set_ylabel('Number of Nodes')
        ax2.set_title(f'Degree Distribution\nTarget: {self.degree}, Actual Mean: {np.mean(degrees):.2f}')
        ax2.grid(True, alpha=0.3)
        
        # 3. Spatial mapping overlay
        ax3 = axes[1, 0]
        # Show grid coordinates
        spatial_pos = self.network_manager.spatial_mapper.node_to_grid
        grid_x = [coord[0] for coord in spatial_pos.values()]
        grid_y = [coord[1] for coord in spatial_pos.values()]
        
        scatter = ax3.scatter(grid_x, grid_y, c=node_colors, s=node_sizes, alpha=0.7)
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, 80)
        ax3.set_xlabel('Grid X')
        ax3.set_ylabel('Grid Y')
        ax3.set_title('Spatial Mapping to Grid')
        ax3.grid(True, alpha=0.3)
        
        # 4. Network statistics
        ax4 = axes[1, 1]
        stats = self.get_network_statistics()
        
        stats_text = f"""
Network Statistics (Degree {self.degree}):
• Nodes: {stats['num_nodes']}
• Edges: {stats['num_edges']}
• Average Degree: {stats['avg_degree']:.2f}
• Clustering Coefficient: {stats['clustering_coefficient']:.3f}
• Is Connected: {stats['is_connected']}
• Diameter: {stats['diameter']}
• Average Path Length: {stats['avg_path_length']:.2f}

Degree Distribution:
• Min: {stats['degree_distribution']['min']}
• Max: {stats['degree_distribution']['max']}
• Std: {stats['degree_distribution']['std']:.2f}
• Variance: {stats['degree_distribution']['variance']:.2f}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats
    
    def analyze_route_connectivity(self):
        """Analyze how routes connect through the network"""
        route_analysis = {}
        
        # Count routes by mode
        mode_counts = {}
        route_lengths = {}
        
        for u, v, data in self.network_manager.active_network.edges(data=True):
            mode = data.get('transport_mode')
            if hasattr(mode, 'value'):
                mode_str = mode.value
            else:
                mode_str = str(mode)
            
            if mode_str not in mode_counts:
                mode_counts[mode_str] = 0
                route_lengths[mode_str] = []
                
            mode_counts[mode_str] += 1
            route_lengths[mode_str].append(data.get('travel_time', 0))
        
        print("\n=== ROUTE CONNECTIVITY ANALYSIS ===")
        for mode, count in mode_counts.items():
            avg_time = np.mean(route_lengths[mode]) if route_lengths[mode] else 0
            print(f"{mode.upper()}: {count} edges, avg travel time: {avg_time:.1f}")
        
        return mode_counts, route_lengths
    
    def get_network_statistics(self):
        """Get comprehensive network statistics"""
        graph = self.network_manager.active_network
        
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
            'clustering_coefficient': nx.average_clustering(graph),
            'is_connected': nx.is_connected(graph),
        }
        
        if nx.is_connected(graph):
            stats['diameter'] = nx.diameter(graph)
            stats['avg_path_length'] = nx.average_shortest_path_length(graph)
        else:
            stats['diameter'] = float('inf')
            stats['avg_path_length'] = float('inf')
        
        degrees = [d for n, d in graph.degree()]
        stats['degree_distribution'] = {
            'min': min(degrees),
            'max': max(degrees),
            'std': np.std(degrees),
            'variance': np.var(degrees)
        }
        
        return stats

# Usage example
if __name__ == "__main__":
    # Analyze degree-3 network
    analyzer = NetworkTopologyAnalyzer(degree_constraint=3)
    stats = analyzer.visualize_network_structure(save_path="network_degree_3.png")
    analyzer.analyze_route_connectivity()