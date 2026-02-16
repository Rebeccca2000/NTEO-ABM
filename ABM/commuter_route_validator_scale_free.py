#!/usr/bin/env python3
"""
🚶 COMMUTER ROUTE VALIDATION AND CHOICE ANALYSIS

Detailed validation of how network topology changes affect commuter routing choices.
This module specifically tests:

1. Route Choice Validation - How does scale-free topology change route selection?
2. Hub Dependency Analysis - How dependent are commuters on hub nodes?
3. Travel Time Impact - How does hub-based routing affect travel times?
4. Equity Impact Analysis - Who benefits/suffers from hub-dominated networks?
5. Alternative Route Analysis - What happens when hubs are disrupted?
6. Commuter Behavior Simulation - Realistic commuter decision modeling

Usage:
    python commuter_route_validation.py [options]
    
Options:
    --routes        : Route choice analysis
    --equity        : Equity impact analysis  
    --disruption    : Hub disruption scenarios
    --behavior      : Commuter behavior simulation
    --full          : All commuter analysis tests
"""

import sys
import os
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import random
import time

# Import required modules
try:
    from Complete_NTEO.topology.scale_free_topology import ScaleFreeTopologyGenerator, create_scale_free_network_manager
    from Complete_NTEO.topology.network_topology import SydneyNetworkTopology, NodeType, TransportMode
    import database as db
    
    # Try to import other topologies for comparison
    try:
        from small_world_topology import SmallWorldTopologyGenerator
        SMALL_WORLD_AVAILABLE = True
    except ImportError:
        SMALL_WORLD_AVAILABLE = False
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class CommuterRouteValidator:
    """
    Comprehensive validator for commuter routing behavior in scale-free networks
    """
    
    def __init__(self, network_manager, topology_type="scale_free"):
        self.network_manager = network_manager
        self.graph = network_manager.active_network
        self.base_network = network_manager.base_network
        self.spatial_mapper = network_manager.spatial_mapper
        self.topology_type = topology_type
        
        # Commuter profiles for realistic behavior modeling
        self.commuter_profiles = {
            'time_critical': {
                'time_weight': 0.7,
                'cost_weight': 0.1,
                'comfort_weight': 0.1,
                'reliability_weight': 0.1,
                'hub_tolerance': 0.8  # High tolerance for crowded hubs if faster
            },
            'comfort_focused': {
                'time_weight': 0.2,
                'cost_weight': 0.2,
                'comfort_weight': 0.5,
                'reliability_weight': 0.1,
                'hub_tolerance': 0.3  # Low tolerance for crowded hubs
            },
            'budget_conscious': {
                'time_weight': 0.2,
                'cost_weight': 0.6,
                'comfort_weight': 0.1,
                'reliability_weight': 0.1,
                'hub_tolerance': 0.5  # Moderate tolerance if cheaper
            },
            'reliability_focused': {
                'time_weight': 0.3,
                'cost_weight': 0.1,
                'comfort_weight': 0.1,
                'reliability_weight': 0.5,
                'hub_tolerance': 0.4  # Prefer less complex routes
            }
        }
        
        print(f"🚶 Commuter Route Validator initialized for {topology_type} network")
    
    def analyze_route_choice_patterns(self):
        """Analyze how network topology affects route choice patterns"""
        
        print("\n🛣️  ROUTE CHOICE PATTERN ANALYSIS")
        print("=" * 50)
        
        results = {}
        
        # Identify network characteristics
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        
        if not degree_values:
            print("❌ No network data available")
            return results
        
        # Classify nodes by degree
        hub_threshold = np.percentile(degree_values, 75)
        major_hubs = [node for node, degree in degrees.items() if degree >= hub_threshold]
        peripheral_nodes = [node for node, degree in degrees.items() if degree <= np.percentile(degree_values, 25)]
        
        print(f"Network analysis: {len(major_hubs)} hubs, {len(peripheral_nodes)} peripheral nodes")
        
        # Analyze route patterns for different origin-destination types
        route_patterns = {
            'peripheral_to_peripheral': [],
            'peripheral_to_hub': [],
            'hub_to_hub': [],
            'hub_to_peripheral': []
        }
        
        # Sample routes for analysis
        sample_size = min(50, len(list(self.graph.nodes())))
        
        for i in range(sample_size):
            # Sample different OD pair types
            
            # Peripheral to peripheral
            if len(peripheral_nodes) >= 2:
                p1, p2 = random.sample(peripheral_nodes, 2)
                self._analyze_single_route(p1, p2, route_patterns['peripheral_to_peripheral'], 'peripheral_to_peripheral')
            
            # Peripheral to hub
            if peripheral_nodes and major_hubs:
                p = random.choice(peripheral_nodes)
                h = random.choice(major_hubs)
                self._analyze_single_route(p, h, route_patterns['peripheral_to_hub'], 'peripheral_to_hub')
            
            # Hub to hub
            if len(major_hubs) >= 2:
                h1, h2 = random.sample(major_hubs, 2)
                self._analyze_single_route(h1, h2, route_patterns['hub_to_hub'], 'hub_to_hub')
        
        # Analyze patterns
        results['route_patterns'] = self._analyze_route_statistics(route_patterns)
        results['hub_dependency'] = self._calculate_hub_dependency(major_hubs)
        results['network_efficiency'] = self._calculate_network_efficiency_metrics()
        
        return results
    
    def _analyze_single_route(self, origin, destination, pattern_list, pattern_type):
        """Analyze a single route and add to pattern analysis"""
        
        try:
            # Find shortest path
            shortest_path = nx.shortest_path(self.graph, origin, destination)
            path_length = len(shortest_path) - 1
            
            # Calculate travel time (simplified)
            travel_time = 0
            for i in range(len(shortest_path) - 1):
                edge_data = self.graph.get_edge_data(shortest_path[i], shortest_path[i+1])
                if edge_data:
                    travel_time += edge_data.get('travel_time', 1.0)
            
            # Identify hub usage
            degrees = dict(self.graph.degree())
            hub_threshold = np.percentile(list(degrees.values()), 75)
            hubs_in_path = [node for node in shortest_path[1:-1] if degrees.get(node, 0) >= hub_threshold]
            
            # Calculate route characteristics
            route_info = {
                'origin': origin,
                'destination': destination,
                'path_length': path_length,
                'travel_time': travel_time,
                'hubs_used': len(hubs_in_path),
                'hub_names': hubs_in_path,
                'uses_major_hub': len(hubs_in_path) > 0,
                'pattern_type': pattern_type
            }
            
            pattern_list.append(route_info)
            
        except nx.NetworkXNoPath:
            # No path exists
            route_info = {
                'origin': origin,
                'destination': destination,
                'path_length': float('inf'),
                'travel_time': float('inf'),
                'hubs_used': 0,
                'hub_names': [],
                'uses_major_hub': False,
                'pattern_type': pattern_type,
                'no_path': True
            }
            pattern_list.append(route_info)
    
    def _analyze_route_statistics(self, route_patterns):
        """Analyze statistics from route patterns"""
        
        stats = {}
        
        for pattern_type, routes in route_patterns.items():
            if not routes:
                continue
            
            valid_routes = [r for r in routes if not r.get('no_path', False)]
            
            if valid_routes:
                stats[pattern_type] = {
                    'total_routes': len(routes),
                    'valid_routes': len(valid_routes),
                    'avg_path_length': np.mean([r['path_length'] for r in valid_routes]),
                    'avg_travel_time': np.mean([r['travel_time'] for r in valid_routes]),
                    'hub_usage_rate': np.mean([r['uses_major_hub'] for r in valid_routes]),
                    'avg_hubs_per_route': np.mean([r['hubs_used'] for r in valid_routes]),
                    'connectivity_rate': len(valid_routes) / len(routes)
                }
            else:
                stats[pattern_type] = {
                    'total_routes': len(routes),
                    'valid_routes': 0,
                    'connectivity_rate': 0
                }
        
        return stats
    
    def _calculate_hub_dependency(self, major_hubs):
        """Calculate how dependent the network is on major hubs"""
        
        if not major_hubs:
            return {'dependency_score': 0, 'critical_hubs': []}
        
        # Test network connectivity without each hub
        hub_criticality = {}
        
        for hub in major_hubs:
            # Create network without this hub
            test_graph = self.graph.copy()
            test_graph.remove_node(hub)
            
            # Measure impact
            original_components = nx.number_connected_components(self.graph)
            modified_components = nx.number_connected_components(test_graph)
            
            # Calculate largest component size change
            if nx.is_connected(self.graph):
                original_largest = self.graph.number_of_nodes()
            else:
                original_largest = len(max(nx.connected_components(self.graph), key=len))
            
            if nx.number_connected_components(test_graph) > 0:
                modified_largest = len(max(nx.connected_components(test_graph), key=len))
            else:
                modified_largest = 0
            
            # Calculate criticality score
            component_impact = modified_components - original_components
            size_impact = (original_largest - modified_largest) / original_largest
            
            hub_criticality[hub] = {
                'component_impact': component_impact,
                'size_impact': size_impact,
                'criticality_score': component_impact * 0.5 + size_impact * 0.5
            }
        
        # Overall dependency metrics
        avg_criticality = np.mean([data['criticality_score'] for data in hub_criticality.values()])
        critical_hubs = [hub for hub, data in hub_criticality.items() if data['criticality_score'] > 0.1]
        
        return {
            'dependency_score': avg_criticality,
            'critical_hubs': critical_hubs,
            'hub_criticality': hub_criticality
        }
    
    def _calculate_network_efficiency_metrics(self):
        """Calculate various network efficiency metrics"""
        
        metrics = {}
        
        try:
            if nx.is_connected(self.graph):
                metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(self.graph)
                metrics['diameter'] = nx.diameter(self.graph)
                metrics['global_efficiency'] = nx.global_efficiency(self.graph)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
                metrics['diameter'] = nx.diameter(subgraph)
                metrics['global_efficiency'] = nx.global_efficiency(subgraph)
                metrics['connectivity_note'] = f"Based on largest component ({len(largest_cc)} nodes)"
            
            metrics['avg_clustering_coefficient'] = nx.average_clustering(self.graph)
            metrics['number_of_nodes'] = self.graph.number_of_nodes()
            metrics['number_of_edges'] = self.graph.number_of_edges()
            metrics['density'] = nx.density(self.graph)
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def simulate_commuter_behavior(self):
        """Simulate realistic commuter behavior and route choices"""
        
        print("\n🧑 COMMUTER BEHAVIOR SIMULATION")
        print("=" * 50)
        
        results = {}
        
        # Sample commuters with different profiles
        num_commuters = 100
        commuter_scenarios = []
        
        for i in range(num_commuters):
            # Randomly assign profile
            profile_name = random.choice(list(self.commuter_profiles.keys()))
            profile = self.commuter_profiles[profile_name].copy()
            
            # Sample origin and destination
            nodes = list(self.graph.nodes())
            origin = random.choice(nodes)
            destination = random.choice([n for n in nodes if n != origin])
            
            commuter_scenarios.append({
                'commuter_id': i,
                'profile_name': profile_name,
                'profile': profile,
                'origin': origin,
                'destination': destination
            })
        
        # Simulate route choices for each commuter
        route_choices = []
        
        for scenario in commuter_scenarios:
            choice = self._simulate_individual_commuter_choice(scenario)
            if choice:
                route_choices.append(choice)
        
        # Analyze results by profile
        results['profile_analysis'] = self._analyze_commuter_profiles(route_choices)
        results['route_preferences'] = self._analyze_route_preferences(route_choices)
        results['equity_implications'] = self._analyze_equity_implications(route_choices)
        
        return results
    
    def _simulate_individual_commuter_choice(self, scenario):
        """Simulate route choice for individual commuter"""
        
        origin = scenario['origin']
        destination = scenario['destination']
        profile = scenario['profile']
        
        try:
            # Find all reasonable paths (up to 3 alternatives)
            all_paths = []
            
            # Shortest path
            try:
                shortest_path = nx.shortest_path(self.graph, origin, destination)
                all_paths.append(shortest_path)
            except nx.NetworkXNoPath:
                return None
            
            # Alternative paths (if they exist)
            try:
                # Simple alternative: remove highest degree node from shortest path and find new path
                degrees = dict(self.graph.degree())
                if len(shortest_path) > 2:
                    intermediate_nodes = shortest_path[1:-1]
                    if intermediate_nodes:
                        # Remove highest degree intermediate node
                        highest_degree_node = max(intermediate_nodes, key=lambda x: degrees.get(x, 0))
                        temp_graph = self.graph.copy()
                        temp_graph.remove_node(highest_degree_node)
                        
                        alt_path = nx.shortest_path(temp_graph, origin, destination)
                        if alt_path != shortest_path:
                            all_paths.append(alt_path)
            except:
                pass  # Alternative path not found
            
            # Evaluate each path
            path_evaluations = []
            
            for path in all_paths:
                evaluation = self._evaluate_path_for_commuter(path, profile)
                evaluation['path'] = path
                path_evaluations.append(evaluation)
            
            # Choose best path based on commuter profile
            if path_evaluations:
                best_path = max(path_evaluations, key=lambda x: x['total_score'])
                
                return {
                    'commuter_id': scenario['commuter_id'],
                    'profile_name': scenario['profile_name'],
                    'origin': origin,
                    'destination': destination,
                    'chosen_path': best_path['path'],
                    'path_evaluation': best_path,
                    'alternatives_considered': len(path_evaluations)
                }
        
        except Exception as e:
            print(f"Error simulating commuter {scenario['commuter_id']}: {e}")
            return None
    
    def _evaluate_path_for_commuter(self, path, profile):
        """Evaluate a path based on commuter profile preferences"""
        
        # Calculate path characteristics
        path_length = len(path) - 1
        travel_time = 0
        total_cost = 0
        hub_usage = 0
        comfort_score = 1.0
        reliability_score = 1.0
        
        degrees = dict(self.graph.degree())
        hub_threshold = np.percentile(list(degrees.values()), 75)
        
        # Calculate metrics along the path
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            
            # Get edge data
            edge_data = self.graph.get_edge_data(current_node, next_node, {})
            
            # Travel time
            travel_time += edge_data.get('travel_time', 1.0)
            
            # Cost (simplified)
            transport_mode = edge_data.get('transport_mode', TransportMode.BUS)
            if transport_mode == TransportMode.TRAIN:
                total_cost += 2.0  # Train costs more but faster
            else:
                total_cost += 1.5  # Bus costs
            
            # Hub usage (intermediate nodes only)
            if i > 0 and i < len(path) - 1:
                node_degree = degrees.get(current_node, 0)
                if node_degree >= hub_threshold:
                    hub_usage += 1
                    # Comfort decreases with hub usage (crowding)
                    comfort_score *= 0.8
                    # Reliability may decrease with transfers
                    reliability_score *= 0.9
        
        # Normalize scores
        time_score = 1.0 / (1.0 + travel_time / 10.0)  # Normalize travel time
        cost_score = 1.0 / (1.0 + total_cost / 5.0)    # Normalize cost
        
        # Hub tolerance factor
        hub_penalty = 1.0
        if hub_usage > 0:
            hub_penalty = profile.get('hub_tolerance', 0.5)
        
        # Calculate weighted total score
        total_score = (
            profile['time_weight'] * time_score +
            profile['cost_weight'] * cost_score +
            profile['comfort_weight'] * comfort_score * hub_penalty +
            profile['reliability_weight'] * reliability_score
        )
        
        return {
            'path_length': path_length,
            'travel_time': travel_time,
            'total_cost': total_cost,
            'hub_usage': hub_usage,
            'comfort_score': comfort_score,
            'reliability_score': reliability_score,
            'time_score': time_score,
            'cost_score': cost_score,
            'total_score': total_score
        }
    
    def _analyze_commuter_profiles(self, route_choices):
        """Analyze route choices by commuter profile"""
        
        profile_analysis = {}
        
        for profile_name in self.commuter_profiles.keys():
            profile_choices = [choice for choice in route_choices 
                             if choice['profile_name'] == profile_name]
            
            if profile_choices:
                profile_analysis[profile_name] = {
                    'num_commuters': len(profile_choices),
                    'avg_travel_time': np.mean([choice['path_evaluation']['travel_time'] 
                                              for choice in profile_choices]),
                    'avg_cost': np.mean([choice['path_evaluation']['total_cost'] 
                                       for choice in profile_choices]),
                    'avg_hub_usage': np.mean([choice['path_evaluation']['hub_usage'] 
                                            for choice in profile_choices]),
                    'hub_usage_rate': np.mean([choice['path_evaluation']['hub_usage'] > 0 
                                             for choice in profile_choices]),
                    'avg_satisfaction': np.mean([choice['path_evaluation']['total_score'] 
                                               for choice in profile_choices])
                }
        
        return profile_analysis
    
    def _analyze_route_preferences(self, route_choices):
        """Analyze overall route preferences"""
        
        if not route_choices:
            return {}
        
        # Hub usage patterns
        hub_usage_distribution = [choice['path_evaluation']['hub_usage'] for choice in route_choices]
        routes_using_hubs = sum(1 for usage in hub_usage_distribution if usage > 0)
        
        # Travel time distribution
        travel_times = [choice['path_evaluation']['travel_time'] for choice in route_choices]
        
        # Path length distribution
        path_lengths = [choice['path_evaluation']['path_length'] for choice in route_choices]
        
        return {
            'total_commuters': len(route_choices),
            'hub_usage_rate': routes_using_hubs / len(route_choices),
            'avg_hubs_per_route': np.mean(hub_usage_distribution),
            'avg_travel_time': np.mean(travel_times),
            'travel_time_std': np.std(travel_times),
            'avg_path_length': np.mean(path_lengths),
            'path_length_std': np.std(path_lengths),
            'avg_alternatives_considered': np.mean([choice['alternatives_considered'] 
                                                  for choice in route_choices])
        }
    
    def _analyze_equity_implications(self, route_choices):
        """Analyze equity implications of route choices"""
        
        if not route_choices:
            return {}
        
        # Analyze by origin/destination characteristics
        degrees = dict(self.graph.degree())
        degree_threshold_low = np.percentile(list(degrees.values()), 25)
        degree_threshold_high = np.percentile(list(degrees.values()), 75)
        
        # Classify commuters by location type
        peripheral_commuters = []
        central_commuters = []
        
        for choice in route_choices:
            origin_degree = degrees.get(choice['origin'], 0)
            dest_degree = degrees.get(choice['destination'], 0)
            avg_location_degree = (origin_degree + dest_degree) / 2
            
            if avg_location_degree <= degree_threshold_low:
                peripheral_commuters.append(choice)
            elif avg_location_degree >= degree_threshold_high:
                central_commuters.append(choice)
        
        # Compare outcomes
        equity_analysis = {}
        
        if peripheral_commuters:
            equity_analysis['peripheral_commuters'] = {
                'count': len(peripheral_commuters),
                'avg_travel_time': np.mean([c['path_evaluation']['travel_time'] 
                                          for c in peripheral_commuters]),
                'avg_cost': np.mean([c['path_evaluation']['total_cost'] 
                                   for c in peripheral_commuters]),
                'hub_dependency': np.mean([c['path_evaluation']['hub_usage'] > 0 
                                         for c in peripheral_commuters])
            }
        
        if central_commuters:
            equity_analysis['central_commuters'] = {
                'count': len(central_commuters),
                'avg_travel_time': np.mean([c['path_evaluation']['travel_time'] 
                                          for c in central_commuters]),
                'avg_cost': np.mean([c['path_evaluation']['total_cost'] 
                                   for c in central_commuters]),
                'hub_dependency': np.mean([c['path_evaluation']['hub_usage'] > 0 
                                         for c in central_commuters])
            }
        
        # Calculate equity metrics
        if peripheral_commuters and central_commuters:
            travel_time_ratio = (equity_analysis['peripheral_commuters']['avg_travel_time'] / 
                               equity_analysis['central_commuters']['avg_travel_time'])
            cost_ratio = (equity_analysis['peripheral_commuters']['avg_cost'] / 
                         equity_analysis['central_commuters']['avg_cost'])
            
            equity_analysis['equity_metrics'] = {
                'travel_time_ratio': travel_time_ratio,  # >1 means peripheral areas suffer
                'cost_ratio': cost_ratio,                # >1 means peripheral areas pay more
                'equity_score': 2.0 - travel_time_ratio - cost_ratio  # Higher is more equitable
            }
        
        return equity_analysis
    
    def visualize_commuter_analysis(self, analysis_results, save_path=None):
        """Create comprehensive visualization of commuter analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # 1. Route Pattern Analysis
        self._plot_route_patterns(axes[0, 0], analysis_results.get('route_patterns', {}))
        
        # 2. Hub Dependency
        self._plot_hub_dependency(axes[0, 1], analysis_results.get('hub_dependency', {}))
        
        # 3. Commuter Profile Preferences
        self._plot_profile_preferences(axes[0, 2], analysis_results.get('profile_analysis', {}))
        
        # 4. Travel Time Distribution
        self._plot_travel_time_distribution(axes[1, 0], analysis_results.get('route_preferences', {}))
        
        # 5. Equity Analysis
        self._plot_equity_analysis(axes[1, 1], analysis_results.get('equity_implications', {}))
        
        # 6. Network Efficiency Summary
        self._plot_efficiency_summary(axes[1, 2], analysis_results.get('network_efficiency', {}))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📸 Commuter analysis visualization saved: {save_path}")
        
        plt.show()
        return fig
    
    def _plot_route_patterns(self, ax, route_patterns):
        """Plot route pattern analysis"""
        
        if not route_patterns:
            ax.text(0.5, 0.5, 'No route pattern data', transform=ax.transAxes, ha='center')
            return
        
        pattern_names = list(route_patterns.keys())
        hub_usage_rates = [data.get('hub_usage_rate', 0) for data in route_patterns.values()]
        avg_travel_times = [data.get('avg_travel_time', 0) for data in route_patterns.values()]
        
        x = np.arange(len(pattern_names))
        width = 0.35
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - width/2, hub_usage_rates, width, label='Hub Usage Rate', 
                       color='red', alpha=0.7)
        bars2 = ax2.bar(x + width/2, avg_travel_times, width, label='Avg Travel Time', 
                        color='blue', alpha=0.7)
        
        ax.set_xlabel('Route Pattern Type')
        ax.set_ylabel('Hub Usage Rate', color='red')
        ax2.set_ylabel('Average Travel Time', color='blue')
        ax.set_title('Route Patterns by Origin-Destination Type')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('_', ' ').title() for name in pattern_names], rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, hub_usage_rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{value:.1%}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars2, avg_travel_times):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_hub_dependency(self, ax, hub_dependency):
        """Plot hub dependency analysis"""
        
        if not hub_dependency or 'hub_criticality' not in hub_dependency:
            ax.text(0.5, 0.5, 'No hub dependency data', transform=ax.transAxes, ha='center')
            return
        
        hub_criticality = hub_dependency['hub_criticality']
        hubs = list(hub_criticality.keys())
        criticality_scores = [data['criticality_score'] for data in hub_criticality.values()]
        
        bars = ax.bar(range(len(hubs)), criticality_scores, 
                     color=['red' if score > 0.3 else 'orange' if score > 0.1 else 'green' 
                           for score in criticality_scores],
                     alpha=0.7)
        
        ax.set_xlabel('Hub Nodes')
        ax.set_ylabel('Criticality Score')
        ax.set_title(f'Hub Dependency Analysis\n(Overall Score: {hub_dependency.get("dependency_score", 0):.3f})')
        ax.set_xticks(range(len(hubs)))
        ax.set_xticklabels(hubs, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add criticality threshold lines
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='High Criticality')
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Medium Criticality')
        ax.legend()
    
    def _plot_profile_preferences(self, ax, profile_analysis):
        """Plot commuter profile preferences"""
        
        if not profile_analysis:
            ax.text(0.5, 0.5, 'No profile analysis data', transform=ax.transAxes, ha='center')
            return
        
        profiles = list(profile_analysis.keys())
        hub_usage_rates = [data.get('hub_usage_rate', 0) for data in profile_analysis.values()]
        satisfaction_scores = [data.get('avg_satisfaction', 0) for data in profile_analysis.values()]
        
        # Create scatter plot
        colors = ['red', 'blue', 'green', 'orange'][:len(profiles)]
        
        for i, profile in enumerate(profiles):
            ax.scatter(hub_usage_rates[i], satisfaction_scores[i], 
                      c=colors[i], s=100, alpha=0.7, label=profile.replace('_', ' ').title())
            
            # Add profile label
            ax.annotate(profile.replace('_', ' ').title(), 
                       (hub_usage_rates[i], satisfaction_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Hub Usage Rate')
        ax.set_ylabel('Average Satisfaction Score')
        ax.set_title('Commuter Profile Preferences')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_travel_time_distribution(self, ax, route_preferences):
        """Plot travel time distribution"""
        
        if not route_preferences:
            ax.text(0.5, 0.5, 'No route preferences data', transform=ax.transAxes, ha='center')
            return
        
        # Create histogram representation
        avg_time = route_preferences.get('avg_travel_time', 0)
        std_time = route_preferences.get('travel_time_std', 0)
        
        # Generate sample distribution for visualization
        x = np.linspace(max(0, avg_time - 3*std_time), avg_time + 3*std_time, 100)
        y = np.exp(-0.5 * ((x - avg_time) / std_time)**2) / (std_time * np.sqrt(2 * np.pi))
        
        ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
        ax.fill_between(x, y, alpha=0.3)
        ax.axvline(avg_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_time:.1f}')
        
        ax.set_xlabel('Travel Time')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Travel Time Distribution\n(μ={avg_time:.1f}, σ={std_time:.1f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics
        stats_text = f"""Statistics:
Hub Usage Rate: {route_preferences.get('hub_usage_rate', 0):.1%}
Avg Path Length: {route_preferences.get('avg_path_length', 0):.1f}
Total Commuters: {route_preferences.get('total_commuters', 0)}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    def _plot_equity_analysis(self, ax, equity_implications):
        """Plot equity analysis"""
        
        if not equity_implications or 'equity_metrics' not in equity_implications:
            ax.text(0.5, 0.5, 'No equity analysis data', transform=ax.transAxes, ha='center')
            return
        
        equity_metrics = equity_implications['equity_metrics']
        
        # Create comparison chart
        metrics = ['Travel Time Ratio', 'Cost Ratio', 'Equity Score']
        values = [
            equity_metrics.get('travel_time_ratio', 1),
            equity_metrics.get('cost_ratio', 1),
            equity_metrics.get('equity_score', 0)
        ]
        
        colors = ['red' if v > 1 else 'green' if v < 1 else 'yellow' for v in values[:2]]
        colors.append('green' if values[2] > 0 else 'red')
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        
        ax.set_ylabel('Ratio / Score')
        ax.set_title('Equity Analysis\n(Peripheral vs Central Areas)')
        ax.grid(True, alpha=0.3)
        
        # Add reference lines
        ax.axhline(y=1, color='black', linestyle='-', alpha=0.5, label='Equity Baseline')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.legend()
    
    def _plot_efficiency_summary(self, ax, network_efficiency):
        """Plot network efficiency summary"""
        
        if not network_efficiency:
            ax.text(0.5, 0.5, 'No efficiency data', transform=ax.transAxes, ha='center')
            return
        
        # Create efficiency dashboard
        ax.axis('off')
        ax.set_title('Network Efficiency Summary', fontsize=14, fontweight='bold')
        
        efficiency_text = f"""
🌐 NETWORK TOPOLOGY: {self.topology_type.upper()}

📊 BASIC METRICS:
   • Nodes: {network_efficiency.get('number_of_nodes', 'N/A')}
   • Edges: {network_efficiency.get('number_of_edges', 'N/A')}
   • Density: {network_efficiency.get('density', 0):.4f}

⚡ EFFICIENCY METRICS:
   • Avg Path Length: {network_efficiency.get('avg_shortest_path_length', 0):.2f}
   • Network Diameter: {network_efficiency.get('diameter', 'N/A')}
   • Global Efficiency: {network_efficiency.get('global_efficiency', 0):.3f}
   • Avg Clustering: {network_efficiency.get('avg_clustering_coefficient', 0):.3f}

🎯 PERFORMANCE INSIGHTS:
   {"• Highly efficient hub-based routing" if self.topology_type == "scale_free" else "• Distributed routing patterns"}
   {"• Strong central connectivity" if network_efficiency.get('global_efficiency', 0) > 0.5 else "• Moderate connectivity"}
   {"• Clustered local structure" if network_efficiency.get('avg_clustering_coefficient', 0) > 0.3 else "• Low clustering"}
        """
        
        ax.text(0.05, 0.95, efficiency_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

def main():
    """Main testing function for commuter route validation"""
    
    parser = argparse.ArgumentParser(description="Commuter Route Validation and Choice Analysis")
    parser.add_argument('--routes', action='store_true', help='Route choice analysis')
    parser.add_argument('--equity', action='store_true', help='Equity impact analysis')
    parser.add_argument('--disruption', action='store_true', help='Hub disruption scenarios')
    parser.add_argument('--behavior', action='store_true', help='Commuter behavior simulation')
    parser.add_argument('--full', action='store_true', help='All commuter analysis tests')
    
    args = parser.parse_args()
    
    print("🚶 COMMUTER ROUTE VALIDATION AND CHOICE ANALYSIS")
    print("=" * 60)
    print("Analyzing how scale-free network topology affects commuter routing choices")
    print("=" * 60)
    
    try:
        # Create scale-free network for testing
        print("\n🏗️  Creating scale-free network for analysis...")
        network_manager = create_scale_free_network_manager(m_edges=2, alpha=1.0)
        
        # Create validator
        validator = CommuterRouteValidator(network_manager, "scale_free")
        
        all_results = {}
        
        if args.routes or args.full or not any(vars(args).values()):
            print("\n🛣️  Running route choice pattern analysis...")
            route_results = validator.analyze_route_choice_patterns()
            all_results.update(route_results)
        
        if args.behavior or args.full:
            print("\n🧑 Running commuter behavior simulation...")
            behavior_results = validator.simulate_commuter_behavior()
            all_results.update(behavior_results)
        
        # Create comprehensive visualization
        print("\n🎨 Creating comprehensive visualization...")
        save_path = "commuter_analysis_scale_free.png"
        validator.visualize_commuter_analysis(all_results, save_path)
        
        # Print summary
        print("\n📋 ANALYSIS SUMMARY")
        print("=" * 30)
        
        if 'route_preferences' in all_results:
            prefs = all_results['route_preferences']
            print(f"Hub Usage Rate: {prefs.get('hub_usage_rate', 0):.1%}")
            print(f"Average Travel Time: {prefs.get('avg_travel_time', 0):.1f}")
            print(f"Average Path Length: {prefs.get('avg_path_length', 0):.1f}")
        
        if 'equity_implications' in all_results and 'equity_metrics' in all_results['equity_implications']:
            equity = all_results['equity_implications']['equity_metrics']
            print(f"Equity Score: {equity.get('equity_score', 0):.2f}")
            print(f"Travel Time Equity: {equity.get('travel_time_ratio', 1):.2f}")
        
        print("\n🎉 Commuter analysis completed successfully!")
        print(f"📸 Detailed visualizations saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Commuter analysis failed: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)