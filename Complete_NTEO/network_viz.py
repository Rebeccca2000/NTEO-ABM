import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from typing import Tuple


class NetworkAnalyzer:
    """Single responsibility: Calculate network metrics."""
    
    @staticmethod
    def calculate_metrics(graph: nx.Graph) -> dict:
        """Calculate key network metrics."""
        metrics = {}
        
        # Clustering coefficient
        metrics['clustering'] = nx.average_clustering(graph)
        
        # Average shortest path length
        if nx.is_connected(graph):
            metrics['avg_path'] = nx.average_shortest_path_length(graph)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            metrics['avg_path'] = nx.average_shortest_path_length(subgraph)
        
        # Centrality measures
        centrality = nx.degree_centrality(graph)
        metrics['max_centrality'] = max(centrality.values())
        metrics['centrality_var'] = np.var(list(centrality.values()))
        
        return metrics
    
    @staticmethod
    def print_metrics(graph: nx.Graph, network_type: str = "Network"):
        """Print network metrics in a formatted way."""
        metrics = NetworkAnalyzer.calculate_metrics(graph)
        
        print(f"\n{network_type} Metrics:")
        print(f"Clustering: {metrics['clustering']:.3f}")
        print(f"Avg Path: {metrics['avg_path']:.2f}")
        print(f"Max Centrality: {metrics['max_centrality']:.3f}")
        print(f"Centrality Var: {metrics['centrality_var']:.3f}")


class NetworkVisualizer:
    """Single responsibility: Generate different network topologies."""
    
    def __init__(self, node_count: int = 20, target_degree: int = 4):
        self.node_count = node_count
        self.target_degree = target_degree
        self.graph = None
    
    def generate_degree_constrained_network(self) -> nx.Graph:
        """Generate a degree-constrained network."""
        self.graph = nx.Graph()
        nodes = list(range(self.node_count))
        self.graph.add_nodes_from(nodes)
        
        # Create degree-constrained connections
        for node in nodes:
            current_degree = self.graph.degree(node)
            connections_needed = self.target_degree - current_degree
            
            if connections_needed > 0:
                candidates = [n for n in nodes 
                            if n != node 
                            and not self.graph.has_edge(node, n)
                            and self.graph.degree(n) < self.target_degree]
                
                # Connect to random candidates
                np.random.shuffle(candidates)
                for candidate in candidates[:connections_needed]:
                    if self.graph.degree(candidate) < self.target_degree:
                        self.graph.add_edge(node, candidate)
        
        return self.graph
    
    def generate_small_world_network(self, rewiring_probability: float = 0.1) -> nx.Graph:
        """Generate small-world network using Watts-Strogatz algorithm."""
        if self.node_count < self.target_degree:
            raise ValueError("Node count must be >= target degree")
        
        # Start with regular ring lattice
        self.graph = self._create_regular_ring(self.node_count, self.target_degree)
        
        # Rewire edges with given probability
        self._rewire_edges(rewiring_probability)
        
        return self.graph
    
    def generate_scale_free_network(self, m_edges: int = 2) -> nx.Graph:
        """Generate scale-free network using Barabási-Albert preferential attachment."""
        if self.node_count < m_edges + 1:
            raise ValueError("Node count must be > m_edges")
        
        # Create initial complete graph with m_edges+1 nodes
        self.graph = nx.complete_graph(m_edges + 1)
        
        # Add remaining nodes using preferential attachment
        for new_node in range(m_edges + 1, self.node_count):
            # Calculate attachment probabilities based on current degrees
            degrees = dict(self.graph.degree())
            total_degree = sum(degrees.values())
            
            if total_degree == 0:
                # Fallback: random attachment
                targets = random.sample(list(self.graph.nodes()), m_edges)
            else:
                # Preferential attachment
                targets = []
                candidates = list(self.graph.nodes())
                
                for _ in range(m_edges):
                    if not candidates:
                        break
                    
                    # Calculate probabilities proportional to degree
                    candidate_degrees = [degrees[node] for node in candidates]
                    candidate_total = sum(candidate_degrees)
                    
                    if candidate_total == 0:
                        # All candidates have degree 0, use uniform random
                        target = random.choice(candidates)
                    else:
                        # Normalize probabilities to ensure they sum to 1
                        probs = [deg / candidate_total for deg in candidate_degrees]
                        target = np.random.choice(candidates, p=probs)
                    
                    targets.append(target)
                    candidates.remove(target)  # Avoid multiple edges to same node
            
            # Add new node and connect to targets
            self.graph.add_node(new_node)
            for target in targets:
                self.graph.add_edge(new_node, target)
        
        return self.graph
    
    def _create_regular_ring(self, n: int, k: int) -> nx.Graph:
        """Create regular ring lattice where each node connects to k nearest neighbors."""
        graph = nx.Graph()
        nodes = list(range(n))
        graph.add_nodes_from(nodes)
        
        # Connect each node to k/2 neighbors on each side
        for node in nodes:
            for j in range(1, k // 2 + 1):
                neighbor = (node + j) % n
                graph.add_edge(node, neighbor)
        
        return graph
    
    def _rewire_edges(self, probability: float):
        """Rewire edges with given probability to create small-world properties."""
        edges_to_rewire = list(self.graph.edges())
        nodes = list(self.graph.nodes())
        
        for u, v in edges_to_rewire:
            if np.random.random() < probability:
                # Remove original edge
                self.graph.remove_edge(u, v)
                
                # Find new target (avoid self-loops and existing edges)
                candidates = [n for n in nodes 
                            if n != u and not self.graph.has_edge(u, n)]
                
                if candidates:
                    new_target = np.random.choice(candidates)
                    self.graph.add_edge(u, new_target)
    
    def get_layout_positions(self, layout_type: str = "spring") -> dict:
        """Generate layout positions."""
        if layout_type == "circular":
            return nx.circular_layout(self.graph)
        else:
            return nx.spring_layout(self.graph, k=1, iterations=50, seed=42)


class NetworkRenderer:
    """Single responsibility: Render network visualization."""
    
    def __init__(self, figsize: Tuple[float, float] = (10, 8)):
        self.figsize = figsize
    
    def render(self, graph: nx.Graph, positions: dict, save_path: str = None, 
               network_type: str = "degree_constrained"):
        """Render the network with transparent background."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Configure transparent background
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        
        if network_type == "small_world":
            self._render_small_world(graph, positions, ax)
        elif network_type == "scale_free":
            self._render_scale_free(graph, positions, ax)
        else:
            self._render_standard(graph, positions, ax)
        
        # Remove axes and margins
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _render_standard(self, graph: nx.Graph, positions: dict, ax):
        """Render standard network (degree-constrained)."""
        node_size = 800
        node_color = '#4A90E2'  # Blue
        edge_color = '#666666'
        edge_width = 1.5
        
        nx.draw_networkx_nodes(graph, positions, 
                             node_size=node_size,
                             node_color=node_color,
                             alpha=0.8,
                             ax=ax)
        
        nx.draw_networkx_edges(graph, positions,
                             edge_color=edge_color,
                             width=edge_width,
                             alpha=0.6,
                             ax=ax)
    
    def _render_small_world(self, graph: nx.Graph, positions: dict, ax):
        """Render small-world network with ring structure and shortcuts."""
        # Identify regular ring edges vs shortcuts
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        
        ring_edges = []
        shortcut_edges = []
        
        for u, v in graph.edges():
            # Check if edge is part of regular ring (adjacent nodes in circular layout)
            node_diff = abs(u - v)
            if node_diff == 1 or node_diff == n - 1:  # Adjacent or wrap-around edge
                ring_edges.append((u, v))
            else:
                shortcut_edges.append((u, v))
        
        # Draw nodes (orange/yellow like in image)
        nx.draw_networkx_nodes(graph, positions,
                             node_size=1000,
                             node_color='#FF8C42',  # Orange
                             alpha=0.9,
                             ax=ax)
        
        # Draw ring edges (thicker, darker)
        if ring_edges:
            nx.draw_networkx_edges(graph, positions,
                                 edgelist=ring_edges,
                                 edge_color='#333333',
                                 width=2.5,
                                 alpha=0.8,
                                 ax=ax)
        
        # Draw shortcut edges (thinner, lighter)
        if shortcut_edges:
            nx.draw_networkx_edges(graph, positions,
                                 edgelist=shortcut_edges,
                                 edge_color='#666666',
                                 width=1.5,
                                 alpha=0.5,
                                 ax=ax)
    
    def _render_scale_free(self, graph: nx.Graph, positions: dict, ax):
        """Render scale-free network with variable node sizes based on degree."""
        # Calculate degrees for node sizing
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        min_degree = min(degrees.values()) if degrees else 1
        
        # Scale node sizes based on degree (hubs larger)
        min_size = 300
        max_size = 1500
        node_sizes = []
        
        for node in graph.nodes():
            degree = degrees[node]
            # Logarithmic scaling for better visual distinction
            if max_degree > min_degree:
                normalized = (np.log(degree + 1) - np.log(min_degree + 1)) / \
                           (np.log(max_degree + 1) - np.log(min_degree + 1))
            else:
                normalized = 0.5
            size = min_size + (max_size - min_size) * normalized
            node_sizes.append(size)
        
        # Draw nodes with variable sizes (green like in image)
        nx.draw_networkx_nodes(graph, positions,
                             node_size=node_sizes,
                             node_color='#4CAF50',  # Green
                             alpha=0.8,
                             ax=ax)
        
        # Draw edges (standard styling)
        nx.draw_networkx_edges(graph, positions,
                             edge_color='#666666',
                             width=1.2,
                             alpha=0.5,
                             ax=ax)


class NetworkGenerator:
    """Main coordinator following Open/Closed principle."""
    
    def __init__(self, node_count: int = 20, target_degree: int = 4):
        self.visualizer = NetworkVisualizer(node_count, target_degree)
        self.renderer = NetworkRenderer()
        self.analyzer = NetworkAnalyzer()
    
    def create_degree_constrained_visualization(self, save_path: str = None):
        """Generate and render degree-constrained network."""
        graph = self.visualizer.generate_degree_constrained_network()
        positions = self.visualizer.get_layout_positions("spring")
        self.renderer.render(graph, positions, save_path, "degree_constrained")
        return graph
    
    def create_small_world_visualization(self, rewiring_probability: float = 0.1, save_path: str = None):
        """Generate and render small-world network with circular layout."""
        graph = self.visualizer.generate_small_world_network(rewiring_probability)
        positions = self.visualizer.get_layout_positions("circular")
        self.renderer.render(graph, positions, save_path, "small_world")
        return graph
    
    def create_scale_free_visualization(self, m_edges: int = 2, save_path: str = None):
        """Generate and render scale-free network with spring layout."""
        graph = self.visualizer.generate_scale_free_network(m_edges)
        positions = self.visualizer.get_layout_positions("spring")
        self.renderer.render(graph, positions, save_path, "scale_free")
        return graph


# Usage Examples
if __name__ == "__main__":
    # Create networks similar to reference images
    generator = NetworkGenerator(node_count=20, target_degree=4)
    analyzer = NetworkAnalyzer()
    
    print("Generating scale-free network...")
    sf_network = generator.create_scale_free_visualization(m_edges=2, save_path="scale_free.png")
    analyzer.print_metrics(sf_network, "Scale-Free Network")
    
    print("\nGenerating small-world network...")
    sw_network = generator.create_small_world_visualization(0.3, "small_world.png")
    analyzer.print_metrics(sw_network, "Small-World Network")
    
    print("\nGenerating degree-constrained network...")
    dc_network = generator.create_degree_constrained_visualization("degree_constrained.png")
    analyzer.print_metrics(dc_network, "Degree-Constrained Network")