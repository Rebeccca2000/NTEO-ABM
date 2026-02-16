#!/usr/bin/env python3
"""
Small-World Public Transport Route Definition Checker
Understand exactly how public transport routes are created in your small-world network
"""

import database as db
from network_integration import TwoLayerNetworkManager
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class SmallWorldRouteInspector:
    """Inspect how public transport routes are defined in small-world networks"""
    
    def __init__(self, rewiring_p=0.1):
        # Initialize small-world network
        self.network_manager = TwoLayerNetworkManager(
            topology_type='small_world',
            degree=3,
            grid_width=100,
            grid_height=80
        )
        
        # Initialize RAPTOR data structures
        db.initialize_network_data(self.network_manager)
        
        self.rewiring_p = rewiring_p
        print(f"🔍 Small-World Route Inspector (p={rewiring_p})")
    
    def inspect_route_creation_process(self):
        """Show exactly how routes are created from the small-world network"""
        print("\n" + "="*70)
        print("ROUTE CREATION PROCESS INSPECTION")
        print("="*70)
        
        network = self.network_manager.active_network
        
        print("Step 1: Network Structure Analysis")
        print("-" * 40)
        
        # Analyze network nodes by transport mode
        mode_nodes = defaultdict(list)
        for node, data in network.nodes(data=True):
            transport_modes = data.get('transport_modes', [])
            for mode in transport_modes:
                mode_key = mode.value if hasattr(mode, 'value') else str(mode)
                mode_nodes[mode_key].append(node)
        
        for mode, nodes in mode_nodes.items():
            print(f"  {mode}: {len(nodes)} nodes")
            if nodes:
                sample_nodes = nodes[:3]
                for node in sample_nodes:
                    coord = self.network_manager.spatial_mapper.node_to_grid.get(node)
                    print(f"    {node}: {coord}")
        
        print("\nStep 2: Edge Analysis for Route Building")
        print("-" * 40)
        
        # Analyze edges by transport mode and type
        edge_analysis = defaultdict(lambda: {'regular': 0, 'shortcut': 0, 'routes': defaultdict(list)})
        
        for u, v, data in network.edges(data=True):
            transport_mode = data.get('transport_mode')
            edge_type = data.get('edge_type', 'regular')
            route_id = data.get('route_id', 'unknown')
            
            mode_key = transport_mode.value if hasattr(transport_mode, 'value') else str(transport_mode)
            
            edge_analysis[mode_key][edge_type] += 1
            edge_analysis[mode_key]['routes'][route_id].append((u, v))
        
        for mode, analysis in edge_analysis.items():
            print(f"  {mode}:")
            print(f"    Regular edges: {analysis['regular']}")
            print(f"    Shortcut edges: {analysis['shortcut']}")
            print(f"    Routes: {len(analysis['routes'])}")
            
            # Show sample routes
            for route_id, edges in list(analysis['routes'].items())[:2]:
                print(f"      {route_id}: {len(edges)} segments")
        
        return edge_analysis
    
    def inspect_raptor_conversion(self):
        """Show how NetworkX edges become RAPTOR routes"""
        print("\n" + "="*70)
        print("RAPTOR CONVERSION INSPECTION")
        print("="*70)
        
        stations = db.stations
        routes = db.routes
        transfers = db.transfers
        
        print("Step 3: Stations from Network Nodes")
        print("-" * 40)
        
        for mode, station_dict in stations.items():
            print(f"  {mode} stations: {len(station_dict)}")
            
            # Show sample stations with coordinates
            sample_stations = list(station_dict.items())[:5]
            for station_id, coord in sample_stations:
                print(f"    {station_id}: {coord}")
        
        print("\nStep 4: Routes from Network Edges")
        print("-" * 40)
        
        for mode, route_dict in routes.items():
            print(f"  {mode} routes: {len(route_dict)}")
            
            # Show detailed route structure
            for route_id, station_sequence in list(route_dict.items())[:3]:
                print(f"    {route_id}:")
                print(f"      Length: {len(station_sequence)} stations")
                print(f"      Sequence: {station_sequence[:5]}...")
                
                # Check if route forms a valid path
                if len(station_sequence) >= 2:
                    start_coord = stations[mode].get(station_sequence[0])
                    end_coord = stations[mode].get(station_sequence[-1])
                    print(f"      Start: {start_coord}")
                    print(f"      End: {end_coord}")
                    
                    # Calculate route span
                    if start_coord and end_coord:
                        route_distance = np.sqrt((end_coord[0] - start_coord[0])**2 + 
                                               (end_coord[1] - start_coord[1])**2)
                        print(f"      Span: {route_distance:.1f} units")
        
        print(f"\nStep 5: Transfer Connections")
        print("-" * 40)
        print(f"  Total transfers: {len(transfers)}")
        
        # Show sample transfers
        sample_transfers = list(transfers.items())[:5]
        for (from_station, to_station), transfer_time in sample_transfers:
            from_coord = None
            to_coord = None
            
            # Find coordinates for transfer stations
            for mode, station_dict in stations.items():
                if from_station in station_dict:
                    from_coord = station_dict[from_station]
                if to_station in station_dict:
                    to_coord = station_dict[to_station]
            
            print(f"  {from_station} ↔ {to_station}: {transfer_time:.1f}min")
            if from_coord and to_coord:
                transfer_distance = np.sqrt((from_coord[0] - to_coord[0])**2 + 
                                          (from_coord[1] - to_coord[1])**2)
                print(f"    Distance: {transfer_distance:.1f} units")
    
    def test_actual_route_usage(self):
        """Test if the defined routes actually work for pathfinding"""
        print("\n" + "="*70)
        print("ROUTE FUNCTIONALITY TEST")
        print("="*70)
        
        stations = db.stations
        routes = db.routes
        
        # Test route connectivity for each mode
        for mode, route_dict in routes.items():
            print(f"\nTesting {mode} routes:")
            
            working_routes = 0
            for route_id, station_sequence in route_dict.items():
                
                if len(station_sequence) < 2:
                    print(f"  {route_id}: ❌ Too short ({len(station_sequence)} stations)")
                    continue
                
                # Check if all stations exist
                missing_stations = [s for s in station_sequence if s not in stations[mode]]
                if missing_stations:
                    print(f"  {route_id}: ❌ Missing stations: {missing_stations[:2]}...")
                    continue
                
                # Check route connectivity
                network = self.network_manager.active_network
                route_connected = True
                
                for i in range(len(station_sequence) - 1):
                    current_station = station_sequence[i]
                    next_station = station_sequence[i + 1]
                    
                    # Check if there's a network path between consecutive stations
                    if not network.has_edge(current_station, next_station):
                        # Try to find any path
                        try:
                            path = nx.shortest_path(network, current_station, next_station)
                            if len(path) > 3:  # Too many hops between consecutive stations
                                route_connected = False
                                break
                        except nx.NetworkXNoPath:
                            route_connected = False
                            break
                
                if route_connected:
                    working_routes += 1
                    start_coord = stations[mode][station_sequence[0]]
                    end_coord = stations[mode][station_sequence[-1]]
                    route_span = np.sqrt((end_coord[0] - start_coord[0])**2 + 
                                       (end_coord[1] - start_coord[1])**2)
                    print(f"  {route_id}: ✅ Working ({len(station_sequence)} stations, {route_span:.1f} span)")
                else:
                    print(f"  {route_id}: ❌ Connectivity issues")
            
            success_rate = working_routes / len(route_dict) * 100 if route_dict else 0
            print(f"  {mode} route success rate: {working_routes}/{len(route_dict)} ({success_rate:.1f}%)")
    
    def visualize_small_world_routes(self):
        """Create visualization showing how routes work in small-world network"""
        print("\n" + "="*70)
        print("SMALL-WORLD ROUTE VISUALIZATION")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        network = self.network_manager.active_network
        stations = db.stations
        routes = db.routes
        
        # Plot 1: Network with shortcuts highlighted
        pos = {}
        for node, data in network.nodes(data=True):
            coord = self.network_manager.spatial_mapper.node_to_grid.get(node)
            if coord:
                pos[node] = coord
        
        if pos:
            # Draw regular edges
            regular_edges = [(u, v) for u, v, d in network.edges(data=True) 
                           if d.get('edge_type') != 'shortcut']
            shortcut_edges = [(u, v) for u, v, d in network.edges(data=True) 
                            if d.get('edge_type') == 'shortcut']
            
            nx.draw_networkx_nodes(network, pos, node_size=20, node_color='lightblue', ax=axes[0,0])
            nx.draw_networkx_edges(network, pos, edgelist=regular_edges, 
                                 edge_color='gray', width=0.5, ax=axes[0,0])
            nx.draw_networkx_edges(network, pos, edgelist=shortcut_edges, 
                                 edge_color='red', width=1.5, ax=axes[0,0])
            
            axes[0,0].set_title(f'Small-World Network\n({len(shortcut_edges)} shortcuts)')
            axes[0,0].set_xlim(0, 100)
            axes[0,0].set_ylim(0, 80)
        
        # Plot 2: Train route visualization
        if 'train' in stations and 'train' in routes:
            train_coords = []
            for station_id, coord in stations['train'].items():
                train_coords.append(coord)
            
            if train_coords:
                train_x, train_y = zip(*train_coords)
                axes[0,1].scatter(train_x, train_y, c='red', s=30, label='Train Stations')
                
                # Draw train routes
                for route_id, station_sequence in list(routes['train'].items())[:3]:
                    route_coords = []
                    for station in station_sequence:
                        if station in stations['train']:
                            route_coords.append(stations['train'][station])
                    
                    if len(route_coords) >= 2:
                        route_x, route_y = zip(*route_coords)
                        axes[0,1].plot(route_x, route_y, 'r-', alpha=0.7, linewidth=2)
                
                axes[0,1].set_title(f'Train Routes\n({len(routes["train"])} routes)')
                axes[0,1].legend()
        
        # Plot 3: Bus route visualization
        if 'bus' in stations and 'bus' in routes:
            bus_coords = []
            for station_id, coord in stations['bus'].items():
                bus_coords.append(coord)
            
            if bus_coords:
                bus_x, bus_y = zip(*bus_coords)
                axes[1,0].scatter(bus_x, bus_y, c='blue', s=20, label='Bus Stops')
                
                # Draw bus routes
                for route_id, station_sequence in list(routes['bus'].items())[:5]:
                    route_coords = []
                    for station in station_sequence:
                        if station in stations['bus']:
                            route_coords.append(stations['bus'][station])
                    
                    if len(route_coords) >= 2:
                        route_x, route_y = zip(*route_coords)
                        axes[1,0].plot(route_x, route_y, 'b-', alpha=0.5, linewidth=1)
                
                axes[1,0].set_title(f'Bus Routes\n({len(routes["bus"])} routes)')
                axes[1,0].legend()
        
        # Plot 4: Combined system
        if stations:
            for mode, station_dict in stations.items():
                coords = list(station_dict.values())
                if coords:
                    x_coords, y_coords = zip(*coords)
                    color = 'red' if mode == 'train' else 'blue'
                    size = 40 if mode == 'train' else 20
                    axes[1,1].scatter(x_coords, y_coords, c=color, s=size, 
                                    label=f'{mode.title()} ({len(coords)})', alpha=0.7)
            
            axes[1,1].set_title('Complete Transport System')
            axes[1,1].legend()
        
        for ax in axes.flat:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 80)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('small_world_routes_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as 'small_world_routes_analysis.png'")
    
    def complete_inspection(self):
        """Run complete route definition inspection"""
        print("🔍 COMPLETE SMALL-WORLD ROUTE INSPECTION")
        print("=" * 80)
        
        edge_analysis = self.inspect_route_creation_process()
        self.inspect_raptor_conversion()
        self.test_actual_route_usage()
        self.visualize_small_world_routes()
        
        print("\n" + "="*80)
        print("ROUTE DEFINITION SUMMARY")
        print("="*80)
        
        # Summary of findings
        network = self.network_manager.active_network
        shortcuts = sum(1 for _, _, d in network.edges(data=True) if d.get('edge_type') == 'shortcut')
        total_edges = network.number_of_edges()
        
        print(f"Network Structure:")
        print(f"  Total edges: {total_edges}")
        print(f"  Shortcut edges: {shortcuts} ({shortcuts/total_edges*100:.1f}%)")
        
        stations = db.stations
        routes = db.routes
        
        total_stations = sum(len(s) for s in stations.values())
        total_routes = sum(len(r) for r in routes.values())
        
        print(f"\nPublic Transport System:")
        print(f"  Total stations: {total_stations}")
        print(f"  Total routes: {total_routes}")
        print(f"  Transfer connections: {len(db.transfers)}")
        
        print(f"\nRoute Creation Process:")
        print(f"  1. NetworkX graph defines network topology")
        print(f"  2. Nodes with transport_modes become stations")
        print(f"  3. Edges with route_id become route segments") 
        print(f"  4. NetworkToRAPTORConverter builds ordered station sequences")
        print(f"  5. Multi-modal nodes create transfer connections")

def inspect_small_world_routes(p=0.1):
    """Quick inspection of small-world route definitions"""
    inspector = SmallWorldRouteInspector(p)
    return inspector.complete_inspection()

if __name__ == "__main__":
    inspect_small_world_routes(0.1)