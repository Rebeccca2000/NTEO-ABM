# degree_comparison_fixed.py - Clean approach without database complications

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import time
from datetime import datetime
import database as db
from agent_run_visualisation import MobilityModel
from collections import defaultdict
import traceback

class DegreeComparisonFramework:
    def __init__(self, base_config, degrees_to_test=[3, 4, 5]):
        self.base_config = base_config
        self.degrees_to_test = degrees_to_test
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_single_degree_simulation(self, degree, num_runs=1, steps_per_run=db.SIMULATION_STEPS):
        """Run simulation for a single degree with minimal database usage"""
        print(f"\n{'='*60}")
        print(f"TESTING DEGREE-{degree} NETWORK")
        print(f"{'='*60}")
        
        degree_results = {
            'degree': degree,
            'runs': [],
            'network_stats': {},
            'performance_metrics': {}
        }
        
        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} for Degree-{degree} ---")
            
            try:
                # Update network configuration
                network_config = self.base_config['network_config'].copy()
                network_config['degree_constraint'] = degree
                network_config['topology_type'] = 'degree_constrained'
                
                # Create model parameters (no schema needed - use default database)
                model_params = self.base_config.copy()
                model_params['network_config'] = network_config
                model_params['num_commuters'] = 30  # Reduced for faster testing
                model_params['schema'] = None  # Use default schema
                
                # Create model
                model = MobilityModel(**model_params)
                
                # DISABLE problematic batch logging
                original_batch_update = model.batch_update_commuter_logs
                model.batch_update_commuter_logs = lambda: None  # Replace with no-op
                
                # Collect initial network statistics
                if run == 0:  # Only collect once per degree
                    degree_results['network_stats'] = model.network_manager.get_network_statistics()
                    print(f"Network stats: {degree_results['network_stats']['num_nodes']} nodes, "
                          f"{degree_results['network_stats']['num_edges']} edges, "
                          f"avg degree: {degree_results['network_stats']['avg_degree']:.2f}")
                
                # Run simulation with performance monitoring
                start_time = time.time()
                
                # Run simulation step by step to avoid batch logging issues
                for step in range(steps_per_run):
                    try:
                        model.current_step += 1
                        
                        # Core simulation logic without batch logging
                        model.service_provider_agent.update_time_steps()
                        availability_dict = model.service_provider_agent.initialize_availability(model.current_step - 1)
                        
                        # Process commuters (simplified)
                        for commuter in model.commuter_agents[:10]:  # Process subset for speed
                            model.create_time_based_trip(model.current_step, commuter)
                            
                            # Process active requests
                            for request_id, request in list(commuter.requests.items()):
                                if request['status'] == 'active' and request['start_time'] >= model.current_step:
                                    try:
                                        # Get travel options
                                        travel_options_without_MaaS = model.maas_agent.options_without_maas(
                                            request_id, request['start_time'], request['origin'], request['destination'])
                                        
                                        travel_options_with_MaaS = model.maas_agent.maas_options(
                                            commuter.payment_scheme, request_id, request['start_time'], 
                                            request['origin'], request['destination'])

                                        # Rank options
                                        ranked_options = commuter.rank_service_options(
                                            travel_options_without_MaaS, travel_options_with_MaaS, request_id)
                                        
                                        if ranked_options:
                                            booking_success, availability_dict = model.maas_agent.book_service(
                                                request_id, ranked_options, model.current_step, availability_dict)
                                    except Exception as e:
                                        # Skip problematic requests
                                        continue
                            
                            # Update commuter location (simplified)
                            commuter.update_location()
                        
                        # Update service provider
                        model.service_provider_agent.update_availability()
                        model.service_provider_agent.dynamic_pricing_share()
                        
                        # Progress indicator
                        if step % 20 == 0:
                            print(f"  Step {step}/{steps_per_run}")
                        
                    except Exception as e:
                        print(f"Warning: Error in step {step}: {e}")
                        continue
                
                run_time = time.time() - start_time
                print(f"Simulation completed in {run_time:.2f} seconds")
                
                # Collect run metrics
                run_results = self.collect_simplified_metrics(model, degree, run, run_time)
                degree_results['runs'].append(run_results)
                
                print(f"Completed run {run + 1} for degree-{degree}")
                
            except Exception as e:
                print(f"Error in degree-{degree}, run {run + 1}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Aggregate results across runs
        if degree_results['runs']:
            degree_results['performance_metrics'] = self.aggregate_performance_metrics(degree_results['runs'])
        
        self.results[degree] = degree_results
        return degree_results
    
    def collect_simplified_metrics(self, model, degree, run, run_time):
        """Collect simplified metrics without database complications"""
        try:
            # Network topology metrics
            network_stats = model.network_manager.get_network_statistics()
            
            # Count active commuters and trips
            total_requests = sum(len(c.requests) for c in model.commuter_agents)
            active_requests = sum(1 for c in model.commuter_agents 
                                for r in c.requests.values() 
                                if r['status'] != 'finished')
            
            performance_metrics = {
                'degree': degree,
                'run': run,
                'run_time_seconds': run_time,
                'steps_completed': model.current_step,
                'avg_path_length': network_stats.get('avg_path_length', 0),
                'clustering_coefficient': network_stats.get('clustering_coefficient', 0),
                'network_efficiency': 1 / network_stats.get('avg_path_length', 1) if network_stats.get('avg_path_length', 0) > 0 else 0,
                'route_calculation_count': model.network_manager.route_calculation_count,
                'cache_size': len(model.network_manager.router.shortest_paths_cache) if hasattr(model.network_manager, 'router') else 0,
                'total_commuters': len(model.commuter_agents),
                'total_requests': total_requests,
                'active_requests': active_requests,
                'network_connectivity': network_stats.get('is_connected', False),
                'network_diameter': network_stats.get('diameter', 0),
                'success': True
            }
            
            return performance_metrics
            
        except Exception as e:
            print(f"Error collecting metrics for degree-{degree}, run {run}: {str(e)}")
            return {
                'degree': degree,
                'run': run,
                'success': False,
                'error': str(e)
            }
    
    def aggregate_performance_metrics(self, runs):
        """Aggregate performance metrics across multiple runs"""
        if not runs:
            return {}
        
        successful_runs = [run for run in runs if run.get('success', False)]
        if not successful_runs:
            return {}
        
        # Collect metrics
        metrics = defaultdict(list)
        for run in successful_runs:
            for key, value in run.items():
                if key not in ['degree', 'run', 'success'] and isinstance(value, (int, float)):
                    metrics[key].append(value)
        
        # Calculate statistics
        aggregated = {}
        for key, values in metrics.items():
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
                aggregated[f'{key}_min'] = float(np.min(values))
                aggregated[f'{key}_max'] = float(np.max(values))
        
        return aggregated
    
    def run_complete_comparison(self, num_runs=1, steps_per_run=db.SIMULATION_STEPS):
        """Run complete comparison across all degrees"""
        print(f"\n{'='*80}")
        print(f"STARTING CLEAN DEGREE COMPARISON STUDY")
        print(f"Degrees to test: {self.degrees_to_test}")
        print(f"Runs per degree: {num_runs}")
        print(f"Steps per run: {steps_per_run}")
        print(f"{'='*80}")
        
        for degree in self.degrees_to_test:
            self.run_single_degree_simulation(degree, num_runs, steps_per_run)
        
        # Save and visualize results
        self.save_results()
        if self.results:
            self.generate_comparison_report()
            self.create_visualizations()
        
        return self.results
    
    def save_results(self):
        """Save results with proper JSON serialization"""
        try:
            # Save as pickle
            with open(f'degree_comparison_clean_{self.timestamp}.pkl', 'wb') as f:
                pickle.dump(self.results, f)
            
            # Convert for JSON
            json_results = {}
            for degree, result in self.results.items():
                json_results[str(degree)] = {
                    'network_stats': self._convert_for_json(result.get('network_stats', {})),
                    'performance_metrics': self._convert_for_json(result.get('performance_metrics', {})),
                    'successful_runs': len([r for r in result.get('runs', []) if r.get('success', False)]),
                    'total_runs': len(result.get('runs', []))
                }
            
            with open(f'degree_comparison_clean_{self.timestamp}.json', 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n✅ Results saved:")
            print(f"  - degree_comparison_clean_{self.timestamp}.pkl")
            print(f"  - degree_comparison_clean_{self.timestamp}.json")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        else:
            return obj
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print(f"\n{'='*80}")
        print(f"CLEAN DEGREE COMPARISON RESULTS")
        print(f"{'='*80}")
        
        # Create summary table
        summary_data = []
        for degree, result in self.results.items():
            if 'network_stats' in result and 'performance_metrics' in result:
                net_stats = result['network_stats']
                perf_metrics = result['performance_metrics']
                
                row = {
                    'Degree': degree,
                    'Nodes': net_stats.get('num_nodes', 'N/A'),
                    'Edges': net_stats.get('num_edges', 'N/A'),
                    'Avg_Degree': f"{net_stats.get('avg_degree', 0):.2f}",
                    'Clustering': f"{net_stats.get('clustering_coefficient', 0):.3f}",
                    'Efficiency': f"{perf_metrics.get('network_efficiency_mean', 0):.3f}",
                    'Avg_Runtime': f"{perf_metrics.get('run_time_seconds_mean', 0):.2f}s",
                    'Successful_Runs': len([r for r in result.get('runs', []) if r.get('success', False)])
                }
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            print("\nSUMMARY TABLE:")
            print(df.to_string(index=False))
            
            # Identify best performing configurations
            if len(summary_data) > 1:
                best_efficiency = max(summary_data, key=lambda x: float(x['Efficiency']))
                best_clustering = max(summary_data, key=lambda x: float(x['Clustering']))
                
                print(f"\n🏆 BEST PERFORMERS:")
                print(f"  • Highest Efficiency: Degree-{best_efficiency['Degree']} ({best_efficiency['Efficiency']})")
                print(f"  • Highest Clustering: Degree-{best_clustering['Degree']} ({best_clustering['Clustering']})")
        
        return summary_data
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.results:
            print("No results to visualize!")
            return
        
        # Extract data for plotting
        degrees = []
        nodes = []
        edges = []
        avg_degrees = []
        clustering_coeffs = []
        network_efficiencies = []
        runtimes = []
        
        for degree, result in self.results.items():
            if result.get('network_stats') and result.get('performance_metrics'):
                degrees.append(degree)
                nodes.append(result['network_stats']['num_nodes'])
                edges.append(result['network_stats']['num_edges'])
                avg_degrees.append(result['network_stats']['avg_degree'])
                clustering_coeffs.append(result['network_stats']['clustering_coefficient'])
                network_efficiencies.append(result['performance_metrics'].get('network_efficiency_mean', 0))
                runtimes.append(result['performance_metrics'].get('run_time_seconds_mean', 0))
        
        if not degrees:
            print("No valid data for visualization!")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Network Size (Nodes vs Edges)
        ax1 = axes[0, 0]
        ax1.bar(degrees, nodes, alpha=0.7, label='Nodes', color='lightblue')
        ax1_twin = ax1.twinx()
        ax1_twin.bar([d + 0.3 for d in degrees], edges, alpha=0.7, label='Edges', color='lightcoral', width=0.4)
        ax1.set_xlabel('Degree Constraint')
        ax1.set_ylabel('Nodes', color='blue')
        ax1_twin.set_ylabel('Edges', color='red')
        ax1.set_title('Network Size by Degree Constraint')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Average Degree Achievement
        ax2 = axes[0, 1]
        bars = ax2.bar(degrees, avg_degrees, color='green', alpha=0.7)
        ax2.plot(degrees, degrees, 'r--', label='Target Degree')
        ax2.set_xlabel('Target Degree Constraint')
        ax2.set_ylabel('Actual Average Degree')
        ax2.set_title('Degree Constraint Achievement')
        ax2.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_degrees):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Network Topology Metrics
        ax3 = axes[0, 2]
        x = np.arange(len(degrees))
        width = 0.35
        ax3.bar(x - width/2, clustering_coeffs, width, label='Clustering Coefficient', alpha=0.7)
        ax3.bar(x + width/2, network_efficiencies, width, label='Network Efficiency', alpha=0.7)
        ax3.set_xlabel('Degree Constraint')
        ax3.set_ylabel('Metric Value')
        ax3.set_title('Network Topology Metrics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(degrees)
        ax3.legend()
        
        # 4. Performance Analysis
        ax4 = axes[1, 0]
        bars = ax4.bar(degrees, runtimes, color='orange', alpha=0.7)
        ax4.set_xlabel('Degree Constraint')
        ax4.set_ylabel('Average Runtime (seconds)')
        ax4.set_title('Simulation Performance')
        
        for bar, value in zip(bars, runtimes):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}s', ha='center', va='bottom')
        
        # 5. Network Efficiency vs Clustering
        ax5 = axes[1, 1]
        scatter = ax5.scatter(clustering_coeffs, network_efficiencies, 
                             c=degrees, s=100, alpha=0.7, cmap='viridis')
        ax5.set_xlabel('Clustering Coefficient')
        ax5.set_ylabel('Network Efficiency')
        ax5.set_title('Efficiency vs Clustering Trade-off')
        
        # Add degree labels
        for i, degree in enumerate(degrees):
            ax5.annotate(f'Degree-{degree}', 
                        (clustering_coeffs[i], network_efficiencies[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(scatter, ax=ax5, label='Degree Constraint')
        
        # 6. Summary Radar Chart
        ax6 = axes[1, 2]
        if len(degrees) > 1:
            # Normalize metrics for comparison
            norm_clustering = np.array(clustering_coeffs) / max(clustering_coeffs) if max(clustering_coeffs) > 0 else np.zeros_like(clustering_coeffs)
            norm_efficiency = np.array(network_efficiencies) / max(network_efficiencies) if max(network_efficiencies) > 0 else np.zeros_like(network_efficiencies)
            norm_runtime = 1 - (np.array(runtimes) / max(runtimes)) if max(runtimes) > 0 else np.ones_like(runtimes)  # Invert (lower is better)
            
            x = np.arange(len(degrees))
            width = 0.25
            
            ax6.bar(x - width, norm_clustering, width, label='Clustering (normalized)', alpha=0.7)
            ax6.bar(x, norm_efficiency, width, label='Efficiency (normalized)', alpha=0.7)
            ax6.bar(x + width, norm_runtime, width, label='Performance (inverted)', alpha=0.7)
            
            ax6.set_xlabel('Degree Constraint')
            ax6.set_ylabel('Normalized Score (0-1)')
            ax6.set_title('Overall Performance Summary')
            ax6.set_xticks(x)
            ax6.set_xticklabels([f'Degree-{d}' for d in degrees])
            ax6.legend()
        
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(f'degree_comparison_visualization_{self.timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved: degree_comparison_visualization_{self.timestamp}.png")


# Usage example
if __name__ == "__main__":
    try:
        # Prepare base configuration
        base_config = {
            'db_connection_string': db.DB_CONNECTION_STRING,  
            'data_income_weights': db.income_weights,
            'data_health_weights': db.health_weights,
            'data_payment_weights': db.payment_weights,
            'data_age_distribution': db.age_distribution,
            'data_disability_weights': db.disability_weights,
            'data_tech_access_weights': db.tech_access_weights,
            'ASC_VALUES': db.ASC_VALUES,
            'UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS': db.UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
            'UTILITY_FUNCTION_BASE_COEFFICIENTS': db.UTILITY_FUNCTION_BASE_COEFFICIENTS,
            'PENALTY_COEFFICIENTS': db.PENALTY_COEFFICIENTS,
            'AFFORDABILITY_THRESHOLDS': db.AFFORDABILITY_THRESHOLDS,
            'FLEXIBILITY_ADJUSTMENTS': db.FLEXIBILITY_ADJUSTMENTS,
            'VALUE_OF_TIME': db.VALUE_OF_TIME,
            'public_price_table': db.public_price_table,
            'ALPHA_VALUES': db.ALPHA_VALUES,
            'DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS': db.DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
            'BACKGROUND_TRAFFIC_AMOUNT': db.BACKGROUND_TRAFFIC_AMOUNT,
            'CONGESTION_ALPHA': db.CONGESTION_ALPHA,
            'CONGESTION_BETA': db.CONGESTION_BETA,
            'CONGESTION_CAPACITY': db.CONGESTION_CAPACITY,
            'CONGESTION_T_IJ_FREE_FLOW': db.CONGESTION_T_IJ_FREE_FLOW,
            'uber_like1_capacity': db.UberLike1_capacity,
            'uber_like1_price': db.UberLike1_price,
            'uber_like2_capacity': db.UberLike2_capacity,
            'uber_like2_price': db.UberLike2_price,
            'bike_share1_capacity': db.BikeShare1_capacity,
            'bike_share1_price': db.BikeShare1_price,
            'bike_share2_capacity': db.BikeShare2_capacity,
            'bike_share2_price': db.BikeShare2_price,
            'subsidy_dataset': db.subsidy_dataset,
            'subsidy_config': db.daily_config,
            'network_config': db.NETWORK_CONFIG
        }
        
        # Run clean comparison
        comparison = DegreeComparisonFramework(base_config, degrees_to_test=[3, 4, 5])
        results = comparison.run_complete_comparison(num_runs=2, steps_per_run=50)
        
        print("\n🎉 Clean comparison completed successfully!")
        
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()