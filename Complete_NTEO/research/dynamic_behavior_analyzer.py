import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Tuple


class NetworkVisualizer:
    """Single responsibility: Generate and visualize degree-constrained networks."""
    
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
    
    def get_layout_positions(self) -> dict:
        """Generate layout positions using spring layout."""
        return nx.spring_layout(self.graph, k=1, iterations=50, seed=42)


class NetworkRenderer:
    """Single responsibility: Render network visualization."""
    
    def __init__(self, figsize: Tuple[float, float] = (10, 8)):
        self.figsize = figsize
    
    def render(self, graph: nx.Graph, positions: dict, save_path: str = None):
        """Render the network with transparent background."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Configure transparent background
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        
        # Node styling
        node_size = 800
        node_color = '#4A90E2'  # Blue similar to image
        edge_color = '#666666'
        edge_width = 1.5
        
        # Draw network
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
        
        # Remove axes and margins
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, transparent=True, dpi=300, bbox_inches='tight')
        
        plt.show()


class NetworkGenerator:
    """Main coordinator following Open/Closed principle."""
    
    def __init__(self, node_count: int = 20, target_degree: int = 4):
        self.visualizer = NetworkVisualizer(node_count, target_degree)
        self.renderer = NetworkRenderer()
    
    def create_degree_constrained_visualization(self, save_path: str = None):
        """Generate and render degree-constrained network."""
        graph = self.visualizer.generate_degree_constrained_network()
        positions = self.visualizer.get_layout_positions()
        self.renderer.render(graph, positions, save_path)
        return graph
    
    def create_small_world_visualization(self, rewiring_probability: float = 0.1, save_path: str = None):
        """Generate and render small-world network."""
        graph = self.visualizer.generate_small_world_network(rewiring_probability)
        positions = self.visualizer.get_layout_positions()
        self.renderer.render(graph, positions, save_path)
        return graph


# Usage Examples
if __name__ == "__main__":
    generator = NetworkGenerator(node_count=18, target_degree=4)
    
    # Generate degree-constrained network
    print("Generating degree-constrained network...")
    dc_network = generator.create_degree_constrained_visualization("degree_constrained.png")
    
    # Generate small-world networks with different rewiring probabilities
    print("Generating small-world networks...")
    
    # Low rewiring (more regular)
    sw_low = generator.create_small_world_visualization(0.05, "small_world_low.png")
    
    # Medium rewiring (classic small-world)
    sw_medium = generator.create_small_world_visualization(0.3, "small_world_medium.png")
    
    # High rewiring (more random)
    sw_high = generator.create_small_world_visualization(0.8, "small_world_high.png")