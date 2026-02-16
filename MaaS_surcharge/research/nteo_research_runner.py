# nteo_research_runner.py
"""
NTEO Research Runner - Comparative Network Topology Studies
Implements automated research studies using the unified network configuration system.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import networkx as nx
# Import the new unified system
from config.network_config import NetworkConfigurationManager, get_research_config
from config.network_factory import NetworkFactory, batch_create_networks
from testing.abm_initialization import create_nteo_model, MobilityModelNTEO
import config.database_updated as db
TOPOLOGY_COLORS = {
        'degree_constrained': '#2E86AB',  # Blue
        'small_world': '#A23B72',        # Purple  
        'scale_free': '#F18F01'          # Orange
    }

class NTEOResearchRunner:
    """
    Automated research runner for network topology comparative studies.
    Implements the research framework outlined in the construction guide.
    """
    
    def __init__(self, output_dir: str = "nteo_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.config_manager = NetworkConfigurationManager()
        self.network_factory = NetworkFactory()
        
        self.results_cache = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🔬 NTEO Research Runner initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def run_single_topology_study(self, 
                                topology_type: str,
                                parameter_values: List[float],
                                num_runs: int = 3,
                                steps_per_run: int = 144,
                                num_commuters: int = 140) -> Dict[str, Any]:
        """
        Run comparative study for a single topology type across different parameter values.
        
        Args:
            topology_type: Type of network topology
            parameter_values: List of parameter values to test
            num_runs: Number of simulation runs per parameter value
            steps_per_run: Number of simulation steps per run
            num_commuters: Number of commuter agents
        
        Returns:
            Dictionary containing all results
        """
        
        print(f"\n{'='*60}")
        print(f"SINGLE TOPOLOGY STUDY: {topology_type.upper()}")
        print(f"{'='*60}")
        print(f"Parameters: {parameter_values}")
        print(f"Runs per parameter: {num_runs}")
        print(f"Steps per run: {steps_per_run}")
        
        study_results = {
            'topology_type': topology_type,
            'parameter_values': parameter_values,
            'num_runs': num_runs,
            'steps_per_run': steps_per_run,
            'num_commuters': num_commuters,
            'results': {},
            'summary_stats': {},
            'study_timestamp': self.timestamp
        }
        
        total_combinations = len(parameter_values) * num_runs
        current_combination = 0
        
        for param_value in parameter_values:
            param_key = f"{topology_type}_{param_value}"
            print(f"\n🧪 Testing {param_key}")
            
            param_results = {
                'parameter_value': param_value,
                'runs': [],
                'network_stats': {},
                'performance_metrics': {}
            }
            
            for run_idx in range(num_runs):
                current_combination += 1
                progress = (current_combination / total_combinations) * 100
                
                print(f"   📊 Run {run_idx + 1}/{num_runs} ({progress:.1f}% total progress)")
                model = None  # Initialize model variable
                try:
                    # Create model with specific topology and parameter
                    model = create_nteo_model(
                        topology_type=topology_type,
                        variation_parameter=param_value,
                        num_commuters=num_commuters
                    )
                    if model:
                        # Collect initial network statistics (only once per parameter)
                        if run_idx == 0:
                            param_results['network_stats'] = model.network_interface.get_network_stats()
                        
                        # Run simulation
                        run_data = self._run_simulation(model, steps_per_run, f"{param_key}_run{run_idx}")
                        param_results['runs'].append(run_data)
                        
                        print(f"     ✅ Completed - Mode Equity: {run_data['final_mode_choice_equity']:.3f}, " +
                                f"Time Equity: {run_data['final_travel_time_equity']:.3f}, " +
                                f"Efficiency: {run_data['final_system_efficiency']:.1f}")
                    else:
                        raise Exception("Model creation returned None")
                except Exception as e:
                    print(f"❌ Failed in run_single_topology_study: {e}")
                    param_results['runs'].append({'error': str(e)})
                finally:
                    # CRITICAL: Always cleanup
                    if model:
                        model.cleanup()
                        del model  # Force garbage collection
            
            # Calculate summary statistics for this parameter
            param_results['performance_metrics'] = self._calculate_parameter_summary(param_results['runs'])
            study_results['results'][param_key] = param_results
        
        # Calculate overall study summary
        study_results['summary_stats'] = self._calculate_study_summary(study_results['results'])
        
        # Save results
        self._save_study_results(study_results, f"{topology_type}_study_{self.timestamp}")
        
        # CHANGED: Generate parameter comparison plots (not averaged spatial plots)
        try:
            if self._generate_parameter_comparison_plots(study_results):
                print("✅ Parameter comparison plots generated")
            else:
                print("⚠️ No configuration data available for visualization")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate parameter comparison plots: {e}")

        print(f"\n✅ {topology_type} study completed!")
        print(f"📊 Results saved to {self.output_dir}")
        return study_results
    
 
    def run_multi_topology_comparison(self,
                                    topology_configs: Dict[str, List[float]],
                                    num_runs: int = 3,
                                    steps_per_run: int = 144,
                                    num_commuters: int = 140) -> Dict[str, Any]:
        """
        Run comparative study across multiple topology types.
        
        Args:
            topology_configs: Dict mapping topology types to parameter value lists
                            e.g., {'degree_constrained': [3, 4, 5], 'small_world': [0.1, 0.2, 0.3]}
            num_runs: Number of simulation runs per configuration
            steps_per_run: Number of simulation steps per run
            num_commuters: Number of commuter agents
        
        Returns:
            Dictionary containing comparative results
        """
        
        print(f"\n{'='*60}")
        print(f"MULTI-TOPOLOGY COMPARISON STUDY")
        print(f"{'='*60}")
        print(f"Topologies: {list(topology_configs.keys())}")
        
        comparison_results = {
            'topology_configs': topology_configs,
            'num_runs': num_runs,
            'steps_per_run': steps_per_run,
            'num_commuters': num_commuters,
            'topology_results': {},
            'comparative_analysis': {},
            'study_timestamp': self.timestamp
        }
        
        # Run study for each topology type
        for topology_type, parameter_values in topology_configs.items():
            print(f"\n🔄 Processing {topology_type} topology...")
            
            topology_results = self.run_single_topology_study(
                topology_type=topology_type,
                parameter_values=parameter_values,
                num_runs=num_runs,
                steps_per_run=steps_per_run,
                num_commuters=num_commuters
            )
            
            comparison_results['topology_results'][topology_type] = topology_results
              
            try:
                self.create_spatial_equity_heatmaps(topology_results['results'], topology_type)
            except Exception as e:
                print(f"⚠️ Warning: Could not generate spatial heatmaps for {topology_type}: {e}")
        # Perform comparative analysis
        comparison_results['comparative_analysis'] = self._perform_comparative_analysis(
            comparison_results['topology_results']
        )
        
        # Save comparative results
        self._save_study_results(comparison_results, f"multi_topology_comparison_{self.timestamp}")
        
        # Generate comparative visualizations
        self._generate_comparative_plots(comparison_results)
        
        print(f"\n🎯 Multi-topology comparison completed!")
        print(f"📈 Comparative analysis and plots generated")
        
        return comparison_results
    
    def run_research_study(self, study_name: str,
                          num_runs: int = 3,
                          steps_per_run: int = 144,
                          num_commuters: int = 150) -> Dict[str, Any]:
        """
        Run a predefined research study from network_config.py
        
        Args:
            study_name: Name of research study from RESEARCH_CONFIGURATIONS
            num_runs: Number of simulation runs per parameter value
            steps_per_run: Number of simulation steps per run
            num_commuters: Number of commuter agents
        
        Returns:
            Study results dictionary
        """
        
        print(f"\n{'='*60}")
        print(f"RESEARCH STUDY: {study_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Get research configuration
            research_config = get_research_config(study_name)
            topology_type = research_config['topology_type']
            parameter_range = research_config['parameter_range']
            
            print(f"🔬 Study: {research_config.get('analysis_description', 'No description')}")
            print(f"🌐 Topology: {topology_type}")
            print(f"📊 Parameters: {parameter_range}")
            
            # Run the study
            results = self.run_single_topology_study(
                topology_type=topology_type,
                parameter_values=parameter_range,
                num_runs=num_runs,
                steps_per_run=steps_per_run,
                num_commuters=num_commuters
            )
            
            # Add research-specific metadata
            results['research_study_name'] = study_name
            results['research_config'] = research_config
            
            # Generate research-specific analysis
            results['research_analysis'] = self._perform_research_specific_analysis(
                results, research_config
            )
            
            print(f"\n🎯 Research study '{study_name}' completed!")
            return results
            
        except Exception as e:
            print(f"❌ Research study failed: {e}")
            raise
    
    def generate_spatial_analysis(self, model, topology_name, save_dir="my_results"):
        """Generate spatial analysis with working data"""
        
        # Get spatial data using your FIXED method
        spatial_data = model.calculate_spatial_equity_distributions()
        
        # Get actual network nodes for visualization
        network_nodes = {}
        if hasattr(model, 'network_manager') and hasattr(model.network_manager, 'base_network'):
            for node_id, node_obj in model.network_manager.base_network.nodes.items():
                network_nodes[node_id] = {
                    'coordinates': node_obj.coordinates,
                    'type': node_obj.node_type.value,
                    'zone_name': node_obj.zone_name
                }
        
        # Create visualization
        try:
            from spatial_equity_heatmap import SpatialEquityVisualizer
        except ImportError:
            print("⚠️ SpatialEquityVisualizer not found, skipping visualization")
            return spatial_data, None, None
        
        visualizer = SpatialEquityVisualizer()
        
        # Generate heat maps
        heatmap_path = f"{save_dir}/spatial_equity_{topology_name}_heatmap.png"
        fig1 = visualizer.create_comprehensive_heatmap(spatial_data, topology_name, heatmap_path)
        
        # Generate accessibility analysis
        access_path = f"{save_dir}/accessibility_{topology_name}_analysis.png"
        fig2 = visualizer.create_accessibility_analysis({
            'spatial_equity': spatial_data,
            'network_nodes': network_nodes,
            'topology_name': topology_name
        }, access_path)
        
        return spatial_data, fig1, fig2
    
    def _run_simulation(self, model: MobilityModelNTEO, num_steps: int, run_id: str) -> Dict[str, Any]:
        """Run simulation with spatial equity tracking"""
        
        start_time = time.time()
        initial_stats = model.get_model_summary()
        initial_network_structure = self._export_network_structure(model)
    
        step_data = []
     
        
        for step in range(num_steps):
            try:
                model.step()
                
                if step % 10 == 0:
                    # Your existing step metrics
                    step_metrics = {
                        'step': step,
                        'route_calculations': model.route_calculation_count,
                        'mode_choice_equity': model.calculate_mode_choice_equity(),
                        'travel_time_equity': model.calculate_travel_time_equity(),
                        'system_efficiency': model.calculate_system_efficiency()
                    }
                    step_data.append(step_metrics)
                    
                    
            except Exception as step_error:
                print(f"⚠️ Error at step {step}: {step_error}")
                break
        
        # Collect final state with spatial analysis
        final_stats = model.get_model_summary()
       

        # Get spatial equity (with error handling)
        try:
            final_spatial_equity = model.calculate_spatial_equity_distributions()
        except Exception as e:
            print(f"⚠️ Could not calculate spatial equity: {e}")
            final_spatial_equity = {}

        # Export final network structure with usage data
        final_network_structure = self._export_network_structure(model)
        execution_time = time.time() - start_time
    
        
        return {
            'run_id': run_id,
            'execution_time': execution_time,
            'num_steps': num_steps,
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'step_data': step_data,
            'final_spatial_equity': final_spatial_equity,
            # NEW: Add real network structure data
            'initial_network_structure': initial_network_structure,
            'final_network_structure': final_network_structure,
            'final_route_calculations': model.route_calculation_count,
            'final_mode_choice_equity': model.calculate_mode_choice_equity(),
            'final_travel_time_equity': model.calculate_travel_time_equity(),
            'final_system_efficiency': model.calculate_system_efficiency(),
            'avg_route_calculation_time': final_stats.get('avg_route_time', 0.0)
        }
    


    def get_topology_color(self, topology_name):
        """Get consistent color for topology across all plots"""
        TOPOLOGY_COLORS = {
            'degree_constrained': '#2E86AB',  # Blue
            'small_world': '#A23B72',        # Purple  
            'scale_free': '#F18F01'          # Orange
        }
        return TOPOLOGY_COLORS.get(topology_name, '#666666')

    def get_topology_label(self, topology_name):
        """Get consistent label for topology"""
        return topology_name.replace('_', ' ').title()

    def scale_parameter_for_display(self, topology_name, param_value):
        """Scale parameters for consistent display across topologies"""
        if topology_name == 'small_world':
            return param_value * 10  # Scale 0.1-0.6 to 1-6
        else:
            return param_value  # Keep 1-6 as is

    def unscale_parameter_from_display(topology_name, display_value):
        """Convert display parameter back to actual value"""
        if topology_name == 'small_world':
            return display_value / 10
        else:
            return display_value
    def _calculate_parameter_summary(self, runs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate summary statistics for multiple runs of the same parameter"""
        
        # Filter out failed runs
        successful_runs = [run for run in runs if 'error' not in run]
        
        if not successful_runs:
            return {'error': 'All runs failed'}
        
        # Extract metrics from successful runs
        metrics = {
            'mode_choice_equity': [run['final_mode_choice_equity'] for run in successful_runs],
            'travel_time_equity': [run['final_travel_time_equity'] for run in successful_runs],
            'system_efficiency': [run['final_system_efficiency'] for run in successful_runs],
            'execution_time': [run['execution_time'] for run in successful_runs],
            'route_calculations': [run['final_route_calculations'] for run in successful_runs],
            'avg_route_time': [run['avg_route_calculation_time'] for run in successful_runs]
        }
        
        # Calculate summary statistics
        summary = {}
        for metric_name, values in metrics.items():
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)
        
        summary['successful_runs'] = len(successful_runs)
        summary['failed_runs'] = len(runs) - len(successful_runs)
        
        return summary
    
    def _calculate_study_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall study summary statistics"""
        
        summary = {
            'total_configurations': len(results),
            'successful_configurations': 0,
            'best_equity_config': None,
            'best_efficiency_config': None,
            'parameter_performance': {}
        }
        
        best_equity = float('inf')  # NEW: lower is better for equity metrics
        best_efficiency = float('inf')
        
        for config_name, config_results in results.items():
            if 'error' not in config_results['performance_metrics']:
                summary['successful_configurations'] += 1
                
                # Check for best equity
                mode_choice_equity_mean = config_results['performance_metrics']['mode_choice_equity_mean']
                if mode_choice_equity_mean < best_equity:  # Note: < because lower is better
                    best_equity = mode_choice_equity_mean
                    summary['best_equity_config'] = {
                        'config': config_name,
                        'mode_choice_equity': mode_choice_equity_mean,
                        'travel_time_equity': config_results['performance_metrics']['travel_time_equity_mean'],
                        'system_efficiency': config_results['performance_metrics']['system_efficiency_mean'],
                        'parameter_value': config_results['parameter_value']
                    }
                
                # Check for best efficiency (lowest execution time)
                efficiency_mean = config_results['performance_metrics']['execution_time_mean']
                if efficiency_mean < best_efficiency:
                    best_efficiency = efficiency_mean
                    summary['best_efficiency_config'] = {
                        'config': config_name,
                        'execution_time': efficiency_mean,
                        'parameter_value': config_results['parameter_value']
                    }
                
                # Store parameter performance
                summary['parameter_performance'][config_results['parameter_value']] = {
                    'mode_choice_equity': config_results['performance_metrics']['mode_choice_equity_mean'],
                    'travel_time_equity': config_results['performance_metrics']['travel_time_equity_mean'],
                    'system_efficiency': config_results['performance_metrics']['system_efficiency_mean'],
                    'execution_time': efficiency_mean
                }
                        
        return summary
    
    def _perform_comparative_analysis(self, topology_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis across different topology types"""
        
        analysis = {
            'topology_rankings': {},
            'performance_comparison': {},
            'trade_off_analysis': {}
        }
        
        # Extract best performance from each topology
        topology_performance = {}
        for topology_name, results in topology_results.items():
            best_config = results['summary_stats']['best_equity_config']
            if best_config:
                topology_performance[topology_name] = {
                    'mode_choice_equity': best_config['mode_choice_equity'],
                    'travel_time_equity': best_config['travel_time_equity'], 
                    'system_efficiency': best_config['system_efficiency'],
                    'best_parameter': best_config['parameter_value'],
                    'execution_time': results['summary_stats']['best_efficiency_config']['execution_time']
                }
        
        # Rank topologies by each metric (lower is better for equity metrics)
        analysis['topology_rankings']['mode_choice_equity'] = sorted(
            topology_performance.items(), key=lambda x: x[1]['mode_choice_equity'])
        analysis['topology_rankings']['travel_time_equity'] = sorted(
            topology_performance.items(), key=lambda x: x[1]['travel_time_equity'])
        analysis['topology_rankings']['system_efficiency'] = sorted(
            topology_performance.items(), key=lambda x: x[1]['system_efficiency'])
        analysis['topology_rankings']['execution_time'] = sorted(
            topology_performance.items(), key=lambda x: x[1]['execution_time'])
        
        # Performance comparison matrix
        analysis['performance_comparison'] = topology_performance
        
        # Multi-dimensional trade-off analysis
        analysis['trade_off_analysis'] = {
            'best_mode_choice_equity': analysis['topology_rankings']['mode_choice_equity'][0][0],
            'best_travel_time_equity': analysis['topology_rankings']['travel_time_equity'][0][0],
            'best_system_efficiency': analysis['topology_rankings']['system_efficiency'][0][0],
            'best_execution_time': analysis['topology_rankings']['execution_time'][0][0],
            'metric_leaders': {
                'mode_choice_equity': [t[0] for t in analysis['topology_rankings']['mode_choice_equity'][:2]],
                'travel_time_equity': [t[0] for t in analysis['topology_rankings']['travel_time_equity'][:2]],
                'system_efficiency': [t[0] for t in analysis['topology_rankings']['system_efficiency'][:2]]
            }
        }
        
        return analysis
    
    def _find_best_balanced_topology(self, topology_performance: Dict[str, Any]) -> str:
        """Find topology with best balance between equity and efficiency"""
        
        best_score = -1
        best_topology = None
        
        # Normalize metrics and calculate balanced score
        equity_values = [perf['best_equity'] for perf in topology_performance.values()]
        efficiency_values = [perf['efficiency'] for perf in topology_performance.values()]
        
        max_equity = max(equity_values) if equity_values else 1
        min_efficiency = min(efficiency_values) if efficiency_values else 1
        
        for topology, perf in topology_performance.items():
            # Normalize equity (higher is better) and efficiency (lower is better)
            norm_equity = perf['best_equity'] / max_equity
            norm_efficiency = min_efficiency / perf['efficiency']
            
            # Balanced score (equal weight to equity and efficiency)
            balanced_score = (norm_equity + norm_efficiency) / 2
            
            if balanced_score > best_score:
                best_score = balanced_score
                best_topology = topology
        
        return best_topology
    
    def _perform_research_specific_analysis(self, results: Dict[str, Any], 
                                          research_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis specific to the research study focus"""
        
        focus_metric = research_config.get('focus_metric', 'equity_index')
        analysis = {'focus_metric': focus_metric}
        
        # Extract focus metric values across parameters
        metric_values = {}
        for config_name, config_results in results['results'].items():
            if 'error' not in config_results['performance_metrics']:
                param_value = config_results['parameter_value']
                
                if focus_metric == 'efficiency_equity_balance':
                    # Calculate balanced score
                    equity = config_results['performance_metrics']['equity_index_mean']
                    efficiency = 1 / config_results['performance_metrics']['execution_time_mean']
                    metric_values[param_value] = (equity + efficiency) / 2
                else:
                    # Use direct metric
                    metric_key = f'{focus_metric}_mean'
                    if metric_key in config_results['performance_metrics']:
                        metric_values[param_value] = config_results['performance_metrics'][metric_key]
        
        # Find optimal parameter value
        if metric_values:
            optimal_param = max(metric_values.items(), key=lambda x: x[1])
            analysis['optimal_parameter'] = {
                'value': optimal_param[0],
                'metric_score': optimal_param[1]
            }
            
            # Calculate parameter sensitivity
            analysis['parameter_sensitivity'] = {
                'values': list(metric_values.keys()),
                'scores': list(metric_values.values()),
                'range': max(metric_values.values()) - min(metric_values.values()),
                'coefficient_of_variation': np.std(list(metric_values.values())) / np.mean(list(metric_values.values()))
            }
        
        return analysis
    
    def _save_study_results(self, results: Dict[str, Any], filename: str):
        """Save study results to JSON file"""
        
        output_path = self.output_dir / f"{filename}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {output_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save results to {output_path}: {e}")
    
    
    # def _generate_comparative_plots(self, comparison_results: Dict[str, Any]):
    #     """Generate comprehensive visualization plots for all 3 metrics"""
        
    #     try:
    #         plt.style.use('seaborn-v0_8')
    #     except:
    #         plt.style.use('default')
        
    #     # Create figure with more subplots for comprehensive analysis
    #     fig = plt.figure(figsize=(20, 15))
    #     fig.suptitle('NTEO Network Topology Analysis - Complete Research Framework', 
    #                 fontsize=16, fontweight='bold')
        
    #     # Plot 1: All 3 metrics comparison (2x2 grid, top-left)
    #     ax1 = plt.subplot(2, 3, 1)
    #     self._plot_equity_comparison(ax1, comparison_results)
        
    #     # Plot 2: Execution time comparison (top-middle)
    #     ax2 = plt.subplot(2, 3, 2)
    #     self._plot_efficiency_comparison(ax2, comparison_results)
        
    #     # Plot 3: 3D trade-off analysis (top-right)
    #     ax3 = plt.subplot(2, 3, 3, projection='3d')
    #     self._plot_tradeoff_analysis_3d(ax3, comparison_results)
        
    #     # Plot 4-6: Parameter sensitivity for each metric (bottom row)
    #     self._plot_parameter_sensitivity_detailed(fig, comparison_results)
        
    #     plt.tight_layout()
        
    #     # Save comprehensive plot
    #     plot_path = self.output_dir / f"comprehensive_analysis_{self.timestamp}.png"
    #     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    #     plt.close() 
        
    #     # Create separate detailed plots for each metric
    #     self._create_individual_metric_plots(comparison_results)
        
    #     print(f"📊 Comprehensive plots saved to {plot_path}")

    def _generate_comparative_plots(self, comparison_results: Dict[str, Any]):
        """Generate comprehensive visualization plots for all 3 metrics (MODIFY EXISTING)"""
        
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        # Create figure with more subplots for comprehensive analysis
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('NTEO Network Topology Analysis - Complete Research Framework', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: All 3 metrics comparison (2x2 grid, top-left)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_equity_comparison(ax1, comparison_results)
        
        # Plot 2: Execution time comparison (top-middle)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_efficiency_comparison(ax2, comparison_results)
        
        # Plot 3: 3D trade-off analysis (top-right)
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        self._plot_tradeoff_analysis_3d(ax3, comparison_results)
        
        # Plot 4-6: Enhanced parameter sensitivity for each metric (bottom row)
        self._plot_parameter_sensitivity_detailed(fig, comparison_results)  # THIS IS NOW ENHANCED!
        
        plt.tight_layout()
        
        # Save comprehensive plot
        plot_path = self.output_dir / f"comprehensive_analysis_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close() 
        
        # Create separate detailed plots for each metric
        self._create_individual_metric_plots(comparison_results)
        
        # NEW: Create multi-dimensional analysis
        self._create_multidimensional_analysis(comparison_results)
        
        print(f"📊 Comprehensive plots saved to {plot_path}")
        print(f"✅ Enhanced analysis with confidence intervals complete!")

    def _plot_parameter_sensitivity_detailed(self, fig, comparison_results):
        """Create enhanced parameter sensitivity plots with confidence intervals and consistent colors/scaling"""
        import numpy as np
        from scipy import stats
        
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        metric_titles = ['Mode Choice Equity', 'Travel Time Equity', 'System Efficiency']
        
        for metric_idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = plt.subplot(2, 3, 4 + metric_idx)
            
            for topology_name, results in comparison_results['topology_results'].items():
                if 'results' not in results:
                    continue
                
                # Extract data with confidence intervals from individual runs
                parameters = []
                means = []
                ci_lowers = []
                ci_uppers = []
                
                # Get data for each parameter value from individual runs
                for config_name, config_data in results['results'].items():
                    if 'runs' in config_data and config_data['runs']:
                        param_value = config_data['parameter_value']
                        
                        # APPLY PARAMETER SCALING FOR DISPLAY
                        display_param = self.scale_parameter_for_display(topology_name, param_value)
                        
                        # Extract metric values from all runs
                        run_values = []
                        for run in config_data['runs']:
                            if f'final_{metric}' in run:
                                run_values.append(run[f'final_{metric}'])
                        
                        if len(run_values) >= 1:  # At least 1 run
                            parameters.append(display_param)  # Use scaled parameter
                            mean_val = np.mean(run_values)
                            means.append(mean_val)
                            
                            # Calculate 95% confidence interval
                            if len(run_values) > 1:
                                sem = stats.sem(run_values)  # Standard error of mean
                                ci = stats.t.interval(0.95, len(run_values)-1, 
                                                    loc=mean_val, scale=sem)
                                ci_lowers.append(ci[0])
                                ci_uppers.append(ci[1])
                            else:
                                # Single run - no CI, just use the value
                                ci_lowers.append(mean_val)
                                ci_uppers.append(mean_val)
                
                if parameters and means:
                    # Sort by parameter value for proper line plotting
                    sorted_data = sorted(zip(parameters, means, ci_lowers, ci_uppers))
                    if sorted_data:
                        parameters, means, ci_lowers, ci_uppers = zip(*sorted_data)
                        
                        # USE CONSISTENT COLORS
                        color = self.get_topology_color(topology_name)
                        label = self.get_topology_label(topology_name)
                        
                        # Plot mean line with enhanced styling
                        ax.plot(parameters, means, 'o-', color=color, 
                            label=label, linewidth=3, markersize=8, alpha=0.9,
                            markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
                        
                        # Add confidence interval shading
                        ax.fill_between(parameters, ci_lowers, ci_uppers, 
                                    color=color, alpha=0.25, linewidth=0)
            
            # Enhanced formatting for conference presentation
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Parameter Value (Scaled for Comparison)', fontsize=11)
            
            ylabel = title
            if 'equity' in metric:
                ylabel += '\n(Lower = Better)'
            ax.set_ylabel(ylabel, fontsize=11)
            
            ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add some padding to y-axis for better readability
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - (ymax-ymin)*0.05, ymax + (ymax-ymin)*0.05)

    def create_enhanced_sensitivity_analysis(self, comparison_results):
        """Create standalone enhanced sensitivity analysis plot"""
        import matplotlib.pyplot as plt
        
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14
        })
        
        # Create figure with better layout for audience understanding
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Network Topology Parameter Sensitivity Analysis\n(95% Confidence Intervals)', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Publication-ready colors
        colors = {
            'degree_constrained': '#2E86AB',  # Blue
            'small_world': '#A23B72',        # Purple  
            'scale_free': '#F18F01'          # Orange
        }
        
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        metric_titles = ['Mode Choice Equity', 'Travel Time Equity', 'System Efficiency']
        
        for metric_idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[metric_idx]
            
            for topology_name, results in comparison_results['topology_results'].items():
                if 'results' not in results:
                    continue
                
                # Extract data with confidence intervals
                parameters = []
                means = []
                ci_lowers = []
                ci_uppers = []
                
                for config_name, config_data in results['results'].items():
                    if 'runs' in config_data and config_data['runs']:
                        param_value = config_data['parameter_value']
                        
                        # Extract metric values from all runs
                        run_values = []
                        for run in config_data['runs']:
                            if f'final_{metric}' in run:
                                run_values.append(run[f'final_{metric}'])
                        
                        if len(run_values) >= 1:
                            parameters.append(param_value)
                            mean_val = np.mean(run_values)
                            means.append(mean_val)
                            
                            # Calculate 95% confidence interval
                            if len(run_values) > 1:
                                import scipy.stats as stats
                                sem = stats.sem(run_values)
                                ci = stats.t.interval(0.95, len(run_values)-1, 
                                                    loc=mean_val, scale=sem)
                                ci_lowers.append(ci[0])
                                ci_uppers.append(ci[1])
                            else:
                                # Single run - no CI
                                ci_lowers.append(mean_val)
                                ci_uppers.append(mean_val)
                
                if parameters and means:
                    # Sort by parameter value
                    sorted_data = sorted(zip(parameters, means, ci_lowers, ci_uppers))
                    if sorted_data:
                        parameters, means, ci_lowers, ci_uppers = zip(*sorted_data)
                        
                        color = colors.get(topology_name, '#666666')
                        label = topology_name.replace('_', ' ').title()
                        
                        # Plot with enhanced styling
                        ax.plot(parameters, means, 'o-', color=color, 
                            label=label, linewidth=3, markersize=8, alpha=0.9,
                            markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
                        
                        # Confidence interval shading
                        ax.fill_between(parameters, ci_lowers, ci_uppers, 
                                    color=color, alpha=0.25, linewidth=0)
            
            # Enhanced formatting
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Parameter Value', fontsize=11)
            
            ylabel = title
            if 'equity' in metric:
                ylabel += '\n(Lower = Better)'
            ax.set_ylabel(ylabel, fontsize=11)
            
            ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add some padding to y-axis
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - (ymax-ymin)*0.05, ymax + (ymax-ymin)*0.05)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / f"enhanced_sensitivity_analysis_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Enhanced sensitivity analysis saved: {save_path}")
        return save_path

    def _create_multidimensional_analysis(self, comparison_results):
        """Create multi-dimensional analysis with better plots"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # Set publication style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('default')
        
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14
        })
        
        # Create figure with clean conference layout
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Multi-Dimensional Performance Analysis', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        # Plot 1: Enhanced Parameter Sensitivity (spans 2 columns)
        ax1 = plt.subplot(2, 3, (1, 2))
        self._plot_enhanced_sensitivity_standalone(ax1, comparison_results)
        
        # Plot 2: 3D Performance Space
        ax2 = plt.subplot(2, 3, 3, projection='3d')
        self._plot_3d_performance_space(ax2, comparison_results)
        
        # Plot 3: IMPROVED - Normalized Performance Comparison (replaces confusing trade-off)
        ax3 = plt.subplot(2, 3, 4)
        self._plot_normalized_performance_comparison(ax3, comparison_results)
        
        # Plot 4: Parameter Robustness Analysis
        ax4 = plt.subplot(2, 3, 5)
        self._plot_parameter_robustness(ax4, comparison_results)
        
        # Plot 5: Performance Summary
        ax5 = plt.subplot(2, 3, 6)
        self._plot_performance_summary(ax5, comparison_results)
        
        plt.tight_layout()
        
        # Save the multi-dimensional analysis
        save_path = self.output_dir / f"multidimensional_analysis_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Multi-dimensional analysis saved: {save_path}")
        return save_path

    def _plot_enhanced_sensitivity_standalone(self, ax, comparison_results):
        """Plot enhanced sensitivity analysis in a single subplot with consistent colors/scaling"""
        metric = 'mode_choice_equity'  # Focus on the most important metric
        
        for topology_name, results in comparison_results['topology_results'].items():
            if 'results' not in results:
                continue
            
            parameters = []
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for config_name, config_data in results['results'].items():
                if 'runs' in config_data and config_data['runs']:
                    param_value = config_data['parameter_value']
                    
                    # APPLY PARAMETER SCALING
                    display_param = self.scale_parameter_for_display(topology_name, param_value)
                    
                    run_values = []
                    for run in config_data['runs']:
                        if f'final_{metric}' in run:
                            run_values.append(run[f'final_{metric}'])
                    
                    if len(run_values) >= 1:
                        parameters.append(display_param)  # Use scaled parameter
                        mean_val = np.mean(run_values)
                        means.append(mean_val)
                        
                        if len(run_values) > 1:
                            import scipy.stats as stats
                            sem = stats.sem(run_values)
                            ci = stats.t.interval(0.95, len(run_values)-1, 
                                                loc=mean_val, scale=sem)
                            ci_lowers.append(ci[0])
                            ci_uppers.append(ci[1])
                        else:
                            ci_lowers.append(mean_val)
                            ci_uppers.append(mean_val)
            
            if parameters and means:
                sorted_data = sorted(zip(parameters, means, ci_lowers, ci_uppers))
                if sorted_data:
                    parameters, means, ci_lowers, ci_uppers = zip(*sorted_data)
                    
                    # USE CONSISTENT COLORS
                    color = self.get_topology_color(topology_name)
                    label = self.get_topology_label(topology_name)
                    
                    ax.plot(parameters, means, 'o-', color=color, 
                        label=label, linewidth=3, markersize=8)
                    ax.fill_between(parameters, ci_lowers, ci_uppers, 
                                color=color, alpha=0.25)
        
        ax.set_title('Mode Choice Equity Parameter Sensitivity\n(with 95% Confidence Intervals)', 
                    fontweight='bold')
        ax.set_xlabel('Parameter Value (Scaled for Comparison)')
        ax.set_ylabel('Mode Choice Equity\n(Lower = Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_3d_performance_space(self, ax, comparison_results):
        """Plot 3D performance space with parameter trajectories and consistent colors"""
        for topology_name, results in comparison_results['topology_results'].items():
            if 'results' not in results:
                continue
                
            mode_choice_vals = []
            travel_time_vals = []
            system_eff_vals = []
            params = []
            
            for config_name, config_data in results['results'].items():
                if 'runs' in config_data and config_data['runs']:
                    param_value = config_data['parameter_value']
                    
                    # Get mean values across runs
                    mc_values = [run.get('final_mode_choice_equity', 0) for run in config_data['runs'] 
                            if 'final_mode_choice_equity' in run]
                    tt_values = [run.get('final_travel_time_equity', 0) for run in config_data['runs'] 
                            if 'final_travel_time_equity' in run]
                    se_values = [run.get('final_system_efficiency', 0) for run in config_data['runs'] 
                            if 'final_system_efficiency' in run]
                    
                    if mc_values and tt_values and se_values:
                        mode_choice_vals.append(np.mean(mc_values))
                        travel_time_vals.append(np.mean(tt_values))
                        system_eff_vals.append(np.mean(se_values))
                        # APPLY PARAMETER SCALING for size/color coding
                        params.append(self.scale_parameter_for_display(topology_name, param_value))
            
            if mode_choice_vals:
                # USE CONSISTENT COLORS
                color = self.get_topology_color(topology_name)
                label = self.get_topology_label(topology_name)
                
                # Plot points
                ax.scatter(mode_choice_vals, travel_time_vals, system_eff_vals,
                        c=color, s=100, alpha=0.7, label=label)
                
                # Connect with trajectory line (if more than one point)
                if len(mode_choice_vals) > 1:
                    ax.plot(mode_choice_vals, travel_time_vals, system_eff_vals,
                        color=color, alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Mode Choice Equity')
        ax.set_ylabel('Travel Time Equity')
        ax.set_zlabel('System Efficiency')
        ax.set_title('3D Performance Space\n(Parameter Trajectories)')
        ax.legend()


    def _plot_normalized_performance_comparison(self, ax, comparison_results):
        """Plot normalized performance comparison - much clearer than trade-off"""
        import numpy as np
        
        # Collect all data for normalization
        all_data = {'mode_choice_equity': [], 'travel_time_equity': [], 'system_efficiency': []}
        topology_data = {}
        
        for topology_name, results in comparison_results['topology_results'].items():
            if 'results' not in results:
                continue
            
            topology_data[topology_name] = {'mode_choice_equity': [], 'travel_time_equity': [], 'system_efficiency': []}
            
            for config_name, config_data in results['results'].items():
                if 'runs' in config_data and config_data['runs']:
                    # Calculate means for this parameter
                    mc_values = [run.get('final_mode_choice_equity', 0) for run in config_data['runs'] 
                            if 'final_mode_choice_equity' in run]
                    tt_values = [run.get('final_travel_time_equity', 0) for run in config_data['runs'] 
                            if 'final_travel_time_equity' in run]
                    se_values = [run.get('final_system_efficiency', 0) for run in config_data['runs'] 
                            if 'final_system_efficiency' in run]
                    
                    if mc_values and tt_values and se_values:
                        mc_mean = np.mean(mc_values)
                        tt_mean = np.mean(tt_values)
                        se_mean = np.mean(se_values)
                        
                        # Store for this topology
                        topology_data[topology_name]['mode_choice_equity'].append(mc_mean)
                        topology_data[topology_name]['travel_time_equity'].append(tt_mean)
                        topology_data[topology_name]['system_efficiency'].append(se_mean)
                        
                        # Store for global normalization
                        all_data['mode_choice_equity'].append(mc_mean)
                        all_data['travel_time_equity'].append(tt_mean)
                        all_data['system_efficiency'].append(se_mean)
        
        # Calculate normalization parameters
        if not all_data['mode_choice_equity']:
            ax.text(0.5, 0.5, 'No data available for comparison', 
                ha='center', va='center', transform=ax.transAxes)
            return
        
        # Normalize to 0-100 scale (for equity: lower is better, so invert)
        def normalize_metric(values, lower_is_better=True):
            if not values:
                return []
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return [50] * len(values)  # All same, so 50%
            
            if lower_is_better:
                # Invert: lower raw values become higher scores
                return [100 * (max_val - val) / (max_val - min_val) for val in values]
            else:
                # Higher raw values become higher scores
                return [100 * (val - min_val) / (max_val - min_val) for val in values]
        
        # Normalize all metrics
        norm_mc_all = normalize_metric(all_data['mode_choice_equity'], lower_is_better=True)
        norm_tt_all = normalize_metric(all_data['travel_time_equity'], lower_is_better=True)
        norm_se_all = normalize_metric(all_data['system_efficiency'], lower_is_better=False)
        
        # Calculate topology averages
        topology_names = []
        mc_avgs = []
        tt_avgs = []
        se_avgs = []
        overall_scores = []
        
        idx = 0
        for topology_name, data in topology_data.items():
            if data['mode_choice_equity']:  # Has data
                n_points = len(data['mode_choice_equity'])
                
                # Get normalized scores for this topology
                mc_norm = norm_mc_all[idx:idx+n_points]
                tt_norm = norm_tt_all[idx:idx+n_points]
                se_norm = norm_se_all[idx:idx+n_points]
                
                topology_names.append(self.get_topology_label(topology_name))
                mc_avgs.append(np.mean(mc_norm))
                tt_avgs.append(np.mean(tt_norm))
                se_avgs.append(np.mean(se_norm))
                
                # Overall score (equal weights)
                overall_score = (np.mean(mc_norm) + np.mean(tt_norm) + np.mean(se_norm)) / 3
                overall_scores.append(overall_score)
                
                idx += n_points
        
        if not topology_names:
            ax.text(0.5, 0.5, 'No valid data for comparison', 
                ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create grouped bar chart
        x = np.arange(len(topology_names))
        width = 0.2
        
        colors = [self.get_topology_color(name.lower().replace(' ', '_')) for name in topology_names]
        
        bars1 = ax.bar(x - width, mc_avgs, width, label='Mode Choice Equity', 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, tt_avgs, width, label='Travel Time Equity', 
                    color=colors, alpha=0.5, edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x + width, se_avgs, width, label='System Efficiency', 
                    color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add overall performance line
        ax2 = ax.twinx()
        line = ax2.plot(x, overall_scores, 'ko-', linewidth=3, markersize=8, 
                    label='Overall Score', markerfacecolor='white', markeredgewidth=2)
        ax2.set_ylabel('Overall Performance Score (0-100)', fontsize=11)
        ax2.set_ylim(0, 100)
        
        # Formatting
        ax.set_xlabel('Network Topology', fontsize=11)
        ax.set_ylabel('Normalized Performance Score\n(0-100, Higher = Better)', fontsize=11)
        ax.set_title('Normalized Performance Comparison\n(All Metrics Scaled 0-100)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(topology_names)
        ax.set_ylim(0, 100)
        
        # Legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Add overall score labels
        for i, score in enumerate(overall_scores):
            ax2.text(i, score + 3, f'{score:.0f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)

    def _plot_parameter_robustness(self, ax, comparison_results):
        """Plot coefficient of variation (robustness) analysis with consistent colors"""
        topology_names = []
        cv_values = {'mode_choice': [], 'travel_time': [], 'system_eff': []}
        topology_orig_names = []
        
        for topology_name, results in comparison_results['topology_results'].items():
            if 'results' not in results:
                continue
                
            topology_names.append(self.get_topology_label(topology_name))
            topology_orig_names.append(topology_name)
            
            # Calculate CV across parameters
            mc_means = []
            tt_means = []
            se_means = []
            
            for config_name, config_data in results['results'].items():
                if 'runs' in config_data and config_data['runs']:
                    mc_values = [run.get('final_mode_choice_equity', 0) for run in config_data['runs'] 
                            if 'final_mode_choice_equity' in run]
                    tt_values = [run.get('final_travel_time_equity', 0) for run in config_data['runs'] 
                            if 'final_travel_time_equity' in run]
                    se_values = [run.get('final_system_efficiency', 0) for run in config_data['runs'] 
                            if 'final_system_efficiency' in run]
                    
                    if mc_values and tt_values and se_values:
                        mc_means.append(np.mean(mc_values))
                        tt_means.append(np.mean(tt_values))
                        se_means.append(np.mean(se_values))
            
            # Calculate CV (std/mean)
            if mc_means:
                cv_values['mode_choice'].append(np.std(mc_means) / np.mean(mc_means) if np.mean(mc_means) != 0 else 0)
                cv_values['travel_time'].append(np.std(tt_means) / np.mean(tt_means) if np.mean(tt_means) != 0 else 0)  
                cv_values['system_eff'].append(np.std(se_means) / np.mean(se_means) if np.mean(se_means) != 0 else 0)
            else:
                cv_values['mode_choice'].append(0)
                cv_values['travel_time'].append(0)
                cv_values['system_eff'].append(0)
        
        if not topology_names:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        x = np.arange(len(topology_names))
        width = 0.25
        
        # USE CONSISTENT COLORS for each topology
        for i, (topo_name, orig_name) in enumerate(zip(topology_names, topology_orig_names)):
            color = self.get_topology_color(orig_name)
            
            ax.bar(x[i] - width, cv_values['mode_choice'][i], width, 
                color=color, alpha=0.7, label='Mode Choice Equity' if i == 0 else "")
            ax.bar(x[i], cv_values['travel_time'][i], width, 
                color=color, alpha=0.5, label='Travel Time Equity' if i == 0 else "")
            ax.bar(x[i] + width, cv_values['system_eff'][i], width, 
                color=color, alpha=0.9, label='System Efficiency' if i == 0 else "")
        
        ax.set_xlabel('Network Topology')
        ax.set_ylabel('Coefficient of Variation\n(Lower = More Robust)')
        ax.set_title('Parameter Robustness Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(topology_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_summary(self, ax, comparison_results):
        """Plot performance summary text with scaled parameters"""
        ax.text(0.1, 0.9, 'Performance Summary:', fontsize=12, fontweight='bold', 
            transform=ax.transAxes)
        
        summary_text = ""
        
        for topology_name, results in comparison_results['topology_results'].items():
            if 'summary_stats' in results and 'best_equity_config' in results['summary_stats']:
                best_config = results['summary_stats']['best_equity_config']
                if best_config:
                    # Get the actual parameter value (not scaled)
                    actual_param = best_config.get('parameter_value', 'N/A')
                    
                    summary_text += f"{self.get_topology_label(topology_name)}:\n"
                    summary_text += f"  Best Parameter: {actual_param}\n"
                    summary_text += f"  Mode Choice Eq: {best_config.get('mode_choice_equity', 0):.3f}\n"
                    summary_text += f"  Travel Time Eq: {best_config.get('travel_time_equity', 0):.3f}\n"
                    summary_text += f"  System Eff: {best_config.get('system_efficiency', 0):.0f}\n\n"
        
        ax.text(0.1, 0.8, summary_text, fontsize=10, transform=ax.transAxes, 
            verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _create_individual_metric_plots(self, comparison_results):
        """Create separate detailed plots for each metric with consistent colors and parameter scaling"""
        
        metrics = {
            'mode_choice_equity': 'Mode Choice Equity Analysis',
            'travel_time_equity': 'Travel Time Equity Analysis',
            'system_efficiency': 'System Efficiency Analysis'
        }
        
        for metric, title in metrics.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Plot 1: Bar comparison across topologies
            topologies = []
            values = []
            topology_orig_names = []
            
            for topology_name, results in comparison_results['topology_results'].items():
                best_config = results['summary_stats']['best_equity_config']
                if best_config and metric in best_config:
                    topologies.append(self.get_topology_label(topology_name))  # Consistent labels
                    topology_orig_names.append(topology_name)
                    values.append(best_config[metric])
            
            if topologies and values:
                # USE CONSISTENT COLORS
                colors = [self.get_topology_color(orig_name) for orig_name in topology_orig_names]
                
                bars = axes[0,0].bar(topologies, values, color=colors, alpha=0.8,
                                edgecolor='black', linewidth=0.5)
                axes[0,0].set_title(f'{title} by Topology')
                axes[0,0].set_ylabel(metric.replace('_', ' ').title())
                axes[0,0].set_xlabel('Network Topology')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    axes[0,0].text(bar.get_x() + bar.get_width()/2, 
                                bar.get_height() + max(values)*0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Parameter sensitivity with scaling and consistent colors
            for topology_name, results in comparison_results['topology_results'].items():
                if 'results' not in results:
                    continue
                    
                # Get data from individual runs for better accuracy
                parameters = []
                metric_values = []
                
                for config_name, config_data in results['results'].items():
                    if 'runs' in config_data and config_data['runs']:
                        param_value = config_data['parameter_value']
                        
                        # APPLY PARAMETER SCALING
                        display_param = self.scale_parameter_for_display(topology_name, param_value)
                        
                        # Get metric values from runs
                        run_values = []
                        for run in config_data['runs']:
                            if f'final_{metric}' in run:
                                run_values.append(run[f'final_{metric}'])
                        
                        if run_values:
                            parameters.append(display_param)
                            metric_values.append(np.mean(run_values))  # Use mean of runs
                
                if parameters and metric_values:
                    # Sort by parameter for proper line plotting
                    sorted_data = sorted(zip(parameters, metric_values))
                    if sorted_data:
                        parameters, metric_values = zip(*sorted_data)
                        
                        # USE CONSISTENT COLORS
                        color = self.get_topology_color(topology_name)
                        label = self.get_topology_label(topology_name)
                        
                        axes[0,1].plot(parameters, metric_values, 'o-', 
                                    color=color, label=label, linewidth=2.5, markersize=6)
            
            axes[0,1].set_title(f'{title} Parameter Sensitivity')
            axes[0,1].set_xlabel('Parameter Value (Scaled for Comparison)')
            axes[0,1].set_ylabel(metric.replace('_', ' ').title())
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3 & 4: Additional analysis (keep existing error handling)
            try:
                self._plot_detailed_statistics(axes[1,0], comparison_results, metric)
            except Exception as e:
                print(f"⚠ Error in detailed statistics for {metric}: {e}")
                axes[1,0].text(0.5, 0.5, f'{title}\nDetailed Statistics\nError: {str(e)[:50]}...', 
                            ha='center', va='center', transform=axes[1,0].transAxes)

            try:
                self._plot_correlation_analysis(axes[1,1], comparison_results, metric)
            except Exception as e:
                print(f"⚠ Error in correlation analysis for {metric}: {e}")
                axes[1,1].text(0.5, 0.5, f'{title}\nCorrelation Analysis\nError: {str(e)[:50]}...', 
                            ha='center', va='center', transform=axes[1,1].transAxes)

            plt.tight_layout()
            
            # Save individual metric plot
            plot_path = self.output_dir / f"{metric}_analysis_{self.timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()


    def _plot_equity_comparison(self, ax, comparison_results):
        """Plot all three research metrics comparison with consistent colors"""
        
        topologies = []
        mode_choice_equity = []
        travel_time_equity = []
        system_efficiency = []
        
        for topology_name, results in comparison_results['topology_results'].items():
            best_config = results['summary_stats']['best_equity_config']
            if best_config:
                topologies.append(self.get_topology_label(topology_name))
                mode_choice_equity.append(best_config['mode_choice_equity'])
                travel_time_equity.append(best_config['travel_time_equity'])
                system_efficiency.append(best_config['system_efficiency'])
        
        # Create grouped bar chart for all 3 metrics
        x = np.arange(len(topologies))
        width = 0.25
        
        # USE CONSISTENT COLORS
        colors = [self.get_topology_color(topo.lower().replace(' ', '_')) for topo in topologies]
        
        bars1 = ax.bar(x - width, mode_choice_equity, width, label='Mode Choice Equity', 
                    alpha=0.8, color=colors)
        bars2 = ax.bar(x, travel_time_equity, width, label='Travel Time Equity', 
                    alpha=0.6, color=colors)
        bars3 = ax.bar(x + width, system_efficiency, width, label='System Efficiency', 
                    alpha=1.0, color=colors)
        
        ax.set_title('Research Metrics by Topology')
        ax.set_ylabel('Metric Values (Lower = Better for Equity)')
        ax.set_xlabel('Network Topology')
        ax.set_xticks(x)
        ax.set_xticklabels(topologies)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars, values in [(bars1, mode_choice_equity), (bars2, travel_time_equity), (bars3, system_efficiency)]:
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)

    
    def _plot_efficiency_comparison(self, ax, comparison_results):
        """Plot execution time comparison with consistent colors"""
        topologies = []
        efficiency_scores = []
        topology_orig_names = []
        
        for topology_name, results in comparison_results['topology_results'].items():
            best_config = results['summary_stats']['best_efficiency_config']
            if best_config:
                topologies.append(self.get_topology_label(topology_name))  # Use consistent labels
                topology_orig_names.append(topology_name)
                efficiency_scores.append(best_config['execution_time'])
        
        # USE CONSISTENT COLORS - one color per topology
        colors = [self.get_topology_color(orig_name) for orig_name in topology_orig_names]
        
        bars = ax.bar(topologies, efficiency_scores, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
        ax.set_title('Execution Time by Topology')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_xlabel('Network Topology')
        
        # Add value labels on bars
        for bar, score in zip(bars, efficiency_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency_scores)*0.02,
                    f'{score:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_parameter_sensitivity(self, ax, comparison_results):
        """Plot parameter sensitivity analysis for all 3 metrics"""
        
        # Create subplots for each metric
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=14)
        
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        metric_titles = ['Mode Choice Equity', 'Travel Time Equity', 'System Efficiency']
        
        for metric_idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            for topology_name, results in comparison_results['topology_results'].items():
                parameter_performance = results['summary_stats']['parameter_performance']
                
                if parameter_performance:
                    params = list(parameter_performance.keys())
                    metric_values = [perf[metric] for perf in parameter_performance.values()]
                    
                    axes[metric_idx].plot(params, metric_values, marker='o', 
                                        label=topology_name, linewidth=2)
            
            axes[metric_idx].set_title(title)
            axes[metric_idx].set_xlabel('Parameter Value')
            axes[metric_idx].set_ylabel(f'{title} (Lower = Better)' if 'equity' in metric.lower() else title)
            axes[metric_idx].legend()
            axes[metric_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig  # Return figure for saving
    
    def _plot_tradeoff_analysis_3d(self, ax, comparison_results):
        """Plot 3D trade-off between all three metrics with consistent colors"""
        from mpl_toolkits.mplot3d import Axes3D
        
        topologies = []
        mode_choice_equity = []
        travel_time_equity = []
        system_efficiency = []
        topology_orig_names = []
        
        # Try to get data from comparative_analysis first, fallback to best_equity_config
        for topology_name, results in comparison_results['topology_results'].items():
            performance = None
            
            # First try comparative_analysis
            if ('comparative_analysis' in comparison_results and 
                'performance_comparison' in comparison_results['comparative_analysis']):
                performance = comparison_results['comparative_analysis']['performance_comparison'].get(topology_name)
            
            # Fallback to best_equity_config
            if not performance and 'summary_stats' in results:
                performance = results['summary_stats'].get('best_equity_config')
            
            if performance:
                topologies.append(self.get_topology_label(topology_name))  # Consistent labels
                topology_orig_names.append(topology_name)
                mode_choice_equity.append(performance['mode_choice_equity'])
                travel_time_equity.append(performance['travel_time_equity'])
                system_efficiency.append(performance['system_efficiency'])
        
        if not topologies:
            ax.text(0.5, 0.5, 'No data available for 3D analysis', 
                ha='center', va='center', transform=ax.transAxes)
            return
        
        # USE CONSISTENT COLORS
        for i, (topology, orig_name) in enumerate(zip(topologies, topology_orig_names)):
            color = self.get_topology_color(orig_name)
            
            ax.scatter(mode_choice_equity[i], travel_time_equity[i], system_efficiency[i],
                    s=150, alpha=0.8, c=color, label=topology, 
                    edgecolors='black', linewidth=1)
            
            # Add topology labels
            ax.text(mode_choice_equity[i], travel_time_equity[i], system_efficiency[i],
                f'  {topology}', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Mode Choice Equity\n(Lower = Better)')
        ax.set_ylabel('Travel Time Equity\n(Lower = Better)')
        ax.set_zlabel('System Efficiency\n(Higher = Better)')
        ax.set_title('3D Performance Space\n(Best Configuration per Topology)')
        ax.legend(loc='best')
        
        # Improve 3D plot readability
        ax.grid(True, alpha=0.3)

    def _plot_tradeoff_analysis(self, ax, comparison_results):
        """Plot trade-off between equity and efficiency"""
        
        topologies = []
        equity_scores = []
        efficiency_scores = []
        
        for topology_name, results in comparison_results['topology_results'].items():
            performance = comparison_results['comparative_analysis']['performance_comparison'].get(topology_name)
            if performance:
                topologies.append(topology_name)
                equity_scores.append(performance['best_equity'])
                efficiency_scores.append(1 / performance['efficiency'])  # Invert for "higher is better"
        
        scatter = ax.scatter(efficiency_scores, equity_scores, s=100, alpha=0.7, 
                           c=range(len(topologies)), cmap='viridis')
        
        # Add topology labels
        for i, topology in enumerate(topologies):
            ax.annotate(topology, (efficiency_scores[i], equity_scores[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_title('Equity vs Efficiency Trade-off')
        ax.set_xlabel('Efficiency Score (1/execution_time)')
        ax.set_ylabel('Equity Index')
        ax.grid(True, alpha=0.3)

    # Add this to your nteo_research_runner.py

    def run_statistical_validation_study(self, topology_type: str, 
                                    num_runs: int = 25,
                                    significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Run statistically robust study for journal publication
        
        Args:
            topology_type: 'degree_constrained', 'small_world', or 'scale_free'
            num_runs: Number of simulation runs per parameter (min 20 for journals)
            significance_level: Alpha level for statistical tests (typically 0.05)
        """
        print(f"🔬 Statistical Validation Study: {topology_type}")
        print(f"📊 Running {num_runs} simulations per parameter...")
        
        # Get parameter range from config
        from config.network_config import NetworkConfigurationManager
        config_manager = NetworkConfigurationManager()
        config_manager.switch_topology_type(topology_type)
        parameters = config_manager.get_parameter_range()
        
        study_results = {
            'topology_type': topology_type,
            'num_runs_per_parameter': num_runs,
            'significance_level': significance_level,
            'parameters_tested': parameters,
            'raw_data': {},
            'statistical_analysis': {},
            'study_timestamp': self.timestamp
        }
        
        # Run simulations for each parameter
        for param_idx, param_value in enumerate(parameters):
            print(f"\n📈 Parameter {param_value} ({param_idx+1}/{len(parameters)})")
            
            param_results = {
                'parameter_value': param_value,
                'raw_runs': [],
                'statistics': {}
            }
            
            # Run multiple simulations
            for run_idx in range(num_runs):
                try:
                    print(f"  🔄 Run {run_idx+1}/{num_runs}...", end="")
                    
                    # Initialize model
                    model = MobilityModelNTEO(
                        topology_type=topology_type,
                        variation_parameter=param_value,
                        num_commuters=140  # Standard from your config
                    )
                    
                    # Run simulation
                    for step in range(144):  # Your standard 144 steps
                        model.step()
                    
                    # Collect metrics
                    run_data = {
                        'run_id': f"{topology_type}_{param_value}_run_{run_idx}",
                        'mode_choice_equity': model.calculate_mode_choice_equity(),
                        'travel_time_equity': model.calculate_travel_time_equity(),
                        'system_efficiency': model.calculate_system_efficiency(),
                        'spatial_equity': model.calculate_spatial_equity_distributions()
                    }
                    
                    param_results['raw_runs'].append(run_data)
                    print(" ✅")
                    
                except Exception as e:
                    print(f" ❌ Failed: {e}")
                    continue
            
            # Calculate parameter statistics
            if param_results['raw_runs']:
                param_results['statistics'] = self._calculate_parameter_statistics(
                    param_results['raw_runs'], significance_level
                )
                study_results['raw_data'][str(param_value)] = param_results
        
        # Cross-parameter statistical analysis
        study_results['statistical_analysis'] = self._calculate_cross_parameter_statistics(
            study_results['raw_data'], significance_level
        )
        
        # Save results
        filename = f"statistical_validation_{topology_type}_{self.timestamp}"
        self._save_study_results(study_results, filename)
        
        return study_results
    
    def _calculate_parameter_statistics(self, runs: List[Dict], alpha: float = 0.05) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a single parameter"""
        import scipy.stats as stats
        import numpy as np
        
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        statistics = {
            'sample_size': len(runs),
            'alpha_level': alpha,
            'metrics': {}
        }
        
        for metric in metrics:
            values = [run[metric] for run in runs if metric in run]
            
            if len(values) >= 10:  # Minimum for reliable statistics
                # Descriptive statistics
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)  # Sample standard deviation
                sem_val = stats.sem(values)  # Standard error of mean
                
                # Confidence interval
                ci_lower, ci_upper = stats.t.interval(
                    confidence=1-alpha,
                    df=len(values)-1,
                    loc=mean_val,
                    scale=sem_val
                )
                
                # Normality test (important for ANOVA assumptions)
                if len(values) >= 8:
                    shapiro_stat, shapiro_p = stats.shapiro(values)
                    is_normal = shapiro_p > alpha
                else:
                    shapiro_stat, shapiro_p, is_normal = None, None, None
                
                # Outlier detection using IQR method
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = [v for v in values if v < lower_bound or v > upper_bound]
                
                statistics['metrics'][metric] = {
                    'descriptive': {
                        'mean': mean_val,
                        'median': np.median(values),
                        'std': std_val,
                        'sem': sem_val,
                        'min': np.min(values),
                        'max': np.max(values),
                        'range': np.max(values) - np.min(values),
                        'cv': std_val / mean_val if mean_val != 0 else float('inf')  # Coefficient of variation
                    },
                    'inferential': {
                        'confidence_interval_95': {
                            'lower': ci_lower,
                            'upper': ci_upper,
                            'margin_of_error': (ci_upper - ci_lower) / 2
                        },
                        'normality_test': {
                            'shapiro_statistic': shapiro_stat,
                            'shapiro_p_value': shapiro_p,
                            'is_normal': is_normal
                        }
                    },
                    'quality_control': {
                        'outlier_count': len(outliers),
                        'outlier_values': outliers,
                        'outlier_percentage': len(outliers) / len(values) * 100
                    }
                }
        
        return statistics

    def _calculate_cross_parameter_statistics(self, raw_data: Dict, alpha: float = 0.05) -> Dict[str, Any]:
        """Perform ANOVA and post-hoc tests across parameters"""
        import scipy.stats as stats
        from scipy.stats import tukey_hsd
        
        analysis = {
            'anova_results': {},
            'effect_sizes': {},
            'post_hoc_tests': {},
            'practical_significance': {}
        }
        
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        
        for metric in metrics:
            # Collect data for ANOVA
            groups = []
            group_labels = []
            
            for param_str, param_data in raw_data.items():
                if 'raw_runs' in param_data:
                    values = [run[metric] for run in param_data['raw_runs'] if metric in run]
                    if len(values) >= 5:  # Minimum group size
                        groups.append(values)
                        group_labels.append(float(param_str))
            
            if len(groups) >= 2:  # Need at least 2 groups for ANOVA
                # One-way ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Effect size (eta-squared)
                total_n = sum(len(group) for group in groups)
                ss_between = sum(len(group) * (np.mean(group) - np.mean([v for group in groups for v in group]))**2 for group in groups)
                ss_total = sum(sum((val - np.mean([v for group in groups for v in group]))**2 for val in group) for group in groups)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                # Cohen's interpretation of effect size
                if eta_squared < 0.01:
                    effect_size_interpretation = "negligible"
                elif eta_squared < 0.06:
                    effect_size_interpretation = "small"
                elif eta_squared < 0.14:
                    effect_size_interpretation = "medium"
                else:
                    effect_size_interpretation = "large"
                
                analysis['anova_results'][metric] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'is_significant': p_value < alpha,
                    'degrees_of_freedom': (len(groups) - 1, total_n - len(groups)),
                    'group_count': len(groups),
                    'total_sample_size': total_n
                }
                
                analysis['effect_sizes'][metric] = {
                    'eta_squared': eta_squared,
                    'interpretation': effect_size_interpretation,
                    'practical_significance': eta_squared > 0.06  # Medium or larger effect
                }
                
                # Post-hoc tests (if ANOVA is significant)
                if p_value < alpha and len(groups) > 2:
                    try:
                        # Tukey HSD test
                        tukey_result = tukey_hsd(*groups)
                        
                        analysis['post_hoc_tests'][metric] = {
                            'test_type': 'tukey_hsd',
                            'pairwise_comparisons': [],
                            'significant_pairs': []
                        }
                        
                        # Extract pairwise comparisons
                        for i in range(len(groups)):
                            for j in range(i+1, len(groups)):
                                comparison = {
                                    'group_1': group_labels[i],
                                    'group_2': group_labels[j],
                                    'mean_diff': np.mean(groups[i]) - np.mean(groups[j]),
                                    'p_value': tukey_result.pvalue[i, j],
                                    'is_significant': tukey_result.pvalue[i, j] < alpha,
                                    'ci_lower': tukey_result.pvalue[i, j] - 1.96 * np.sqrt(tukey_result.pvalue[i, j] * (1 - tukey_result.pvalue[i, j]) / min(len(groups[i]), len(groups[j]))),
                                    'ci_upper': tukey_result.pvalue[i, j] + 1.96 * np.sqrt(tukey_result.pvalue[i, j] * (1 - tukey_result.pvalue[i, j]) / min(len(groups[i]), len(groups[j])))
                                }
                                
                                analysis['post_hoc_tests'][metric]['pairwise_comparisons'].append(comparison)
                                
                                if comparison['is_significant']:
                                    analysis['post_hoc_tests'][metric]['significant_pairs'].append(
                                        f"{group_labels[i]} vs {group_labels[j]}"
                                    )
                    
                    except Exception as e:
                        analysis['post_hoc_tests'][metric] = {'error': str(e)}
        
        return analysis
    
    def compare_topologies_statistically(self, 
                                   topology_results: Dict[str, Dict], 
                                   alpha: float = 0.05) -> Dict[str, Any]:
        """Compare different topology types statistically"""
        import scipy.stats as stats
        
        comparison = {
            'topology_comparison': {},
            'best_performers': {},
            'statistical_significance': {},
            'effect_sizes': {},
            'recommendations': {}
        }
        
        topologies = list(topology_results.keys())
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        
        for metric in metrics:
            # Collect all data for each topology
            topology_data = {}
            
            for topology in topologies:
                all_values = []
                if 'raw_data' in topology_results[topology]:
                    for param_data in topology_results[topology]['raw_data'].values():
                        if 'raw_runs' in param_data:
                            values = [run[metric] for run in param_data['raw_runs'] if metric in run]
                            all_values.extend(values)
                
                if all_values:
                    topology_data[topology] = all_values
            
            if len(topology_data) >= 2:
                # Statistical comparison between topologies
                topology_names = list(topology_data.keys())
                topology_values = list(topology_data.values())
                
                # ANOVA across topologies
                f_stat, p_value = stats.f_oneway(*topology_values)
                
                # Pairwise t-tests with Bonferroni correction
                pairwise_tests = {}
                bonferroni_alpha = alpha / (len(topologies) * (len(topologies) - 1) / 2)
                
                for i, topo1 in enumerate(topology_names):
                    for j, topo2 in enumerate(topology_names[i+1:], i+1):
                        t_stat, t_p = stats.ttest_ind(
                            topology_data[topo1], 
                            topology_data[topo2],
                            equal_var=False  # Welch's t-test
                        )
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            (np.var(topology_data[topo1]) + np.var(topology_data[topo2])) / 2
                        )
                        cohens_d = (np.mean(topology_data[topo1]) - np.mean(topology_data[topo2])) / pooled_std
                        
                        pairwise_tests[f"{topo1}_vs_{topo2}"] = {
                            't_statistic': t_stat,
                            'p_value': t_p,
                            'significant_bonferroni': t_p < bonferroni_alpha,
                            'significant_uncorrected': t_p < alpha,
                            'cohens_d': cohens_d,
                            'effect_size_interpretation': self._interpret_cohens_d(cohens_d),
                            'mean_difference': np.mean(topology_data[topo1]) - np.mean(topology_data[topo2])
                        }
                
                comparison['statistical_significance'][metric] = {
                    'anova_f': f_stat,
                    'anova_p': p_value,
                    'anova_significant': p_value < alpha,
                    'pairwise_tests': pairwise_tests
                }
                
                # Best performer identification
                topology_means = {topo: np.mean(values) for topo, values in topology_data.items()}
                
                if metric in ['mode_choice_equity', 'travel_time_equity']:  # Lower is better
                    best_topology = min(topology_means, key=topology_means.get)
                    ranking = sorted(topology_means.items(), key=lambda x: x[1])
                else:  # Higher is better (system_efficiency)
                    best_topology = max(topology_means, key=topology_means.get)
                    ranking = sorted(topology_means.items(), key=lambda x: x[1], reverse=True)
                
                comparison['best_performers'][metric] = {
                    'best_topology': best_topology,
                    'best_mean': topology_means[best_topology],
                    'ranking': ranking,
                    'performance_gap': abs(ranking[0][1] - ranking[-1][1])
                }
        
        # Generate practical recommendations
        comparison['recommendations'] = self._generate_topology_recommendations(comparison)
        
        return comparison

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _calculate_detailed_statistics(self, comparison_results: Dict[str, Any], metric: str) -> Dict[str, Any]:
        """
        Calculate advanced statistical analysis for a specific metric across all topologies
        
        Args:
            comparison_results: Results from topology comparison
            metric: Target metric ('mode_choice_equity', 'travel_time_equity', 'system_efficiency')
        
        Returns:
            Dictionary with detailed statistical analysis
        """
        detailed_stats = {
            'metric': metric,
            'topology_analysis': {},
            'convergence_analysis': {},
            'variability_analysis': {},
            'distribution_analysis': {}
        }
        
        # 1. TOPOLOGY-SPECIFIC ANALYSIS
        for topology_name, results in comparison_results['topology_results'].items():
            if 'results' not in results:
                continue
                
            topology_stats = {
                'parameter_sensitivity': {},
                'convergence_metrics': {},
                'run_consistency': {}
            }
            
            # Extract all metric values across parameters and runs
            all_values = []
            parameter_values = []
            
            for config_name, config_data in results['results'].items():
                if 'runs' in config_data:
                    param_value = config_data['parameter_value']
                    
                    # Get final metric values from each run
                    run_finals = []
                    for run in config_data['runs']:
                        if 'error' not in run:
                            final_metric_key = f'final_{metric}'
                            if final_metric_key in run:
                                run_finals.append(run[final_metric_key])
                                all_values.append(run[final_metric_key])
                                parameter_values.append(param_value)
                    
                    if run_finals:
                        # Parameter-specific statistics
                        topology_stats['parameter_sensitivity'][param_value] = {
                            'mean': np.mean(run_finals),
                            'std': np.std(run_finals),
                            'cv': np.std(run_finals) / np.mean(run_finals) if np.mean(run_finals) != 0 else 0,
                            'range': np.max(run_finals) - np.min(run_finals),
                            'run_count': len(run_finals)
                        }
            
            # 2. CONVERGENCE ANALYSIS - analyze step data for convergence patterns
            convergence_data = {}
            for config_name, config_data in results['results'].items():
                if 'runs' in config_data:
                    param_value = config_data['parameter_value']
                    convergence_steps = []
                    
                    for run in config_data['runs']:
                        if 'step_data' in run and 'error' not in run:
                            # Extract metric evolution over steps
                            steps = [step_data['step'] for step_data in run['step_data']]
                            metric_values = [step_data.get(metric, 0) for step_data in run['step_data']]
                            
                            # Find convergence point (when metric stabilizes)
                            convergence_step = self._find_convergence_point(steps, metric_values)
                            if convergence_step is not None:
                                convergence_steps.append(convergence_step)
                    
                    if convergence_steps:
                        convergence_data[param_value] = {
                            'avg_convergence_step': np.mean(convergence_steps),
                            'convergence_consistency': 1.0 - (np.std(convergence_steps) / np.mean(convergence_steps)) if np.mean(convergence_steps) != 0 else 0
                        }
            
            topology_stats['convergence_metrics'] = convergence_data
            
            # 3. DISTRIBUTION ANALYSIS
            if all_values:
                topology_stats['distribution'] = {
                    'skewness': stats.skew(all_values),
                    'kurtosis': stats.kurtosis(all_values),
                    'normality_test': stats.normaltest(all_values)[1] if len(all_values) > 8 else None,  # p-value
                    'outlier_count': len([v for v in all_values if abs(v - np.mean(all_values)) > 2 * np.std(all_values)])
                }
            
            detailed_stats['topology_analysis'][topology_name] = topology_stats
        
        # 4. CROSS-TOPOLOGY COMPARISON
        topology_means = {}
        topology_stds = {}
        
        for topology_name, topology_data in detailed_stats['topology_analysis'].items():
            if 'parameter_sensitivity' in topology_data:
                means = [data['mean'] for data in topology_data['parameter_sensitivity'].values()]
                stds = [data['std'] for data in topology_data['parameter_sensitivity'].values()]
                
                if means:
                    topology_means[topology_name] = np.mean(means)
                    topology_stds[topology_name] = np.mean(stds)
        
        # Statistical significance tests between topologies
        detailed_stats['topology_comparison'] = {
            'means': topology_means,
            'variability': topology_stds,
            'ranking': sorted(topology_means.items(), key=lambda x: x[1]) if topology_means else []
        }
        
        return detailed_stats

    def create_statistical_visualizations(self, statistical_results: Dict, save_dir: str = "statistical_plots"):
        """Create publication-quality statistical visualizations"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        
        # Ensure save directory exists
        Path(save_dir).mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot 1: Confidence Intervals Comparison
        self._plot_confidence_intervals(statistical_results, f"{save_dir}/confidence_intervals.png")
        
        # Plot 2: ANOVA Results Visualization
        self._plot_anova_results(statistical_results, f"{save_dir}/anova_analysis.png")
        
        # Plot 3: Effect Sizes
        self._plot_effect_sizes(statistical_results, f"{save_dir}/effect_sizes.png")
        
        # Plot 4: Box Plots with Statistical Annotations
        self._plot_statistical_boxplots(statistical_results, f"{save_dir}/statistical_boxplots.png")

    def _plot_confidence_intervals(self, results: Dict, save_path: str):
        """Plot means with confidence intervals"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Extract data
            parameters = []
            means = []
            ci_lowers = []
            ci_uppers = []
            
            if 'raw_data' in results:
                for param_str, param_data in results['raw_data'].items():
                    if 'statistics' in param_data and metric in param_data['statistics']['metrics']:
                        stats_data = param_data['statistics']['metrics'][metric]
                        
                        parameters.append(float(param_str))
                        means.append(stats_data['descriptive']['mean'])
                        
                        ci_data = stats_data['inferential']['confidence_interval_95']
                        ci_lowers.append(ci_data['lower'])
                        ci_uppers.append(ci_data['upper'])
            
            if means:
                # Sort by parameter value
                sorted_data = sorted(zip(parameters, means, ci_lowers, ci_uppers))
                parameters, means, ci_lowers, ci_uppers = zip(*sorted_data)
                
                # Plot with error bars
                ax.errorbar(parameters, means, 
                        yerr=[np.array(means) - np.array(ci_lowers), 
                                np.array(ci_uppers) - np.array(means)],
                        fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
                
                ax.set_xlabel(f'{results["topology_type"].replace("_", " ").title()} Parameter')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()}\n(95% Confidence Intervals)')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confidence intervals plot saved: {save_path}")
    def _find_convergence_point(self, steps: List[int], values: List[float], 
                            stability_threshold: float = 0.05, min_stable_steps: int = 10) -> int:
        """
        Find the step where a metric converges (becomes stable)
        
        Args:
            steps: List of simulation steps
            values: List of metric values corresponding to steps
            stability_threshold: Maximum relative change to consider stable
            min_stable_steps: Minimum number of stable steps required
        
        Returns:
            Step number where convergence occurs, or None if no convergence
        """
        if len(values) < min_stable_steps + 1:
            return None
        
        for i in range(len(values) - min_stable_steps):
            # Check stability window
            window_values = values[i:i + min_stable_steps]
            
            if len(window_values) == 0:
                continue
                
            # Calculate relative changes in the window
            mean_val = np.mean(window_values)
            if mean_val == 0:
                continue
                
            relative_changes = [abs(v - mean_val) / mean_val for v in window_values]
            
            # Check if all changes are below threshold
            if all(change <= stability_threshold for change in relative_changes):
                return steps[i] if i < len(steps) else None
        
        return None

    def _calculate_correlation_analysis(self, comparison_results: Dict[str, Any], metric: str) -> Dict[str, Any]:
        """
        Calculate correlation analysis between metrics, parameters, and performance
        
        Args:
            comparison_results: Results from topology comparison  
            metric: Target metric for analysis
        
        Returns:
            Dictionary with correlation analysis results
        """
        correlation_analysis = {
            'metric': metric,
            'cross_metric_correlations': {},
            'parameter_correlations': {},
            'temporal_correlations': {},
            'efficiency_tradeoffs': {}
        }
        
        # 1. CROSS-METRIC CORRELATIONS
        # Collect data for all three metrics across all topologies
        metric_data = {
            'mode_choice_equity': [],
            'travel_time_equity': [],
            'system_efficiency': [],
            'execution_time': [],
            'parameter_values': [],
            'topology_types': []
        }
        
        for topology_name, results in comparison_results['topology_results'].items():
            if 'results' not in results:
                continue
                
            for config_name, config_data in results['results'].items():
                if 'runs' in config_data and 'performance_metrics' in config_data:
                    param_value = config_data['parameter_value']
                    perf_metrics = config_data['performance_metrics']
                    
                    # Only include if all metrics are available
                    required_metrics = ['mode_choice_equity_mean', 'travel_time_equity_mean', 
                                    'system_efficiency_mean', 'execution_time_mean']
                    
                    if all(metric_key in perf_metrics for metric_key in required_metrics):
                        metric_data['mode_choice_equity'].append(perf_metrics['mode_choice_equity_mean'])
                        metric_data['travel_time_equity'].append(perf_metrics['travel_time_equity_mean'])
                        metric_data['system_efficiency'].append(perf_metrics['system_efficiency_mean'])
                        metric_data['execution_time'].append(perf_metrics['execution_time_mean'])
                        metric_data['parameter_values'].append(param_value)
                        metric_data['topology_types'].append(topology_name)
        
        # Calculate cross-metric correlations
        if len(metric_data['mode_choice_equity']) > 3:  # Need at least 4 points for correlation
            metrics_for_corr = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency', 'execution_time']
            
            correlation_matrix = {}
            for i, metric1 in enumerate(metrics_for_corr):
                correlation_matrix[metric1] = {}
                for j, metric2 in enumerate(metrics_for_corr):
                    if i != j:
                        # Calculate both Pearson and Spearman correlations
                        try:
                            pearson_corr, pearson_p = pearsonr(metric_data[metric1], metric_data[metric2])
                            spearman_corr, spearman_p = spearmanr(metric_data[metric1], metric_data[metric2])
                            
                            correlation_matrix[metric1][metric2] = {
                                'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
                                'spearman': {'correlation': spearman_corr, 'p_value': spearman_p}
                            }
                        except Exception as e:
                            correlation_matrix[metric1][metric2] = {'error': str(e)}
            
            correlation_analysis['cross_metric_correlations'] = correlation_matrix
        
        # 2. PARAMETER-PERFORMANCE CORRELATIONS
        parameter_corr = {}
        for topology_name, results in comparison_results['topology_results'].items():
            if 'results' not in results:
                continue
                
            # Collect parameter vs metric data for this topology
            params = []
            metric_vals = []
            
            for config_name, config_data in results['results'].items():
                if 'performance_metrics' in config_data:
                    param_value = config_data['parameter_value']
                    metric_mean_key = f'{metric}_mean'
                    
                    if metric_mean_key in config_data['performance_metrics']:
                        params.append(param_value)
                        metric_vals.append(config_data['performance_metrics'][metric_mean_key])
            
            if len(params) > 2:  # Need at least 3 points
                try:
                    pearson_corr, pearson_p = pearsonr(params, metric_vals)
                    spearman_corr, spearman_p = spearmanr(params, metric_vals)
                    
                    parameter_corr[topology_name] = {
                        'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
                        'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
                        'data_points': len(params)
                    }
                except Exception as e:
                    parameter_corr[topology_name] = {'error': str(e)}
        
        correlation_analysis['parameter_correlations'] = parameter_corr
        
        # 3. TEMPORAL CORRELATIONS (analyzing step-by-step changes)
        temporal_analysis = {}
        for topology_name, results in comparison_results['topology_results'].items():
            if 'results' not in results:
                continue
                
            # Analyze temporal patterns within runs
            lag_correlations = []
            trend_strengths = []
            
            for config_name, config_data in results['results'].items():
                if 'runs' in config_data:
                    for run in config_data['runs']:
                        if 'step_data' in run and 'error' not in run:
                            metric_series = [step_data.get(metric, 0) for step_data in run['step_data']]
                            
                            if len(metric_series) > 5:
                                # Calculate lag-1 autocorrelation
                                if len(metric_series) > 1:
                                    lag1_corr = np.corrcoef(metric_series[:-1], metric_series[1:])[0, 1]
                                    if not np.isnan(lag1_corr):
                                        lag_correlations.append(lag1_corr)
                                
                                # Calculate trend strength (correlation with time)
                                time_steps = list(range(len(metric_series)))
                                if len(time_steps) > 1:
                                    trend_corr = np.corrcoef(time_steps, metric_series)[0, 1]
                                    if not np.isnan(trend_corr):
                                        trend_strengths.append(trend_corr)
            
            if lag_correlations and trend_strengths:
                temporal_analysis[topology_name] = {
                    'avg_autocorrelation': np.mean(lag_correlations),
                    'avg_trend_strength': np.mean(trend_strengths),
                    'temporal_stability': 1.0 - np.std(lag_correlations) if len(lag_correlations) > 1 else 1.0
                }
        
        correlation_analysis['temporal_correlations'] = temporal_analysis
        
        # 4. EFFICIENCY-EQUITY TRADEOFF ANALYSIS
        if len(metric_data['mode_choice_equity']) > 3:
            # Analyze tradeoffs between equity metrics and efficiency
            tradeoff_analysis = {}
            
            # Mode choice equity vs system efficiency
            if metric_data['mode_choice_equity'] and metric_data['system_efficiency']:
                try:
                    equity_eff_corr, equity_eff_p = pearsonr(metric_data['mode_choice_equity'], 
                                                        metric_data['system_efficiency'])
                    tradeoff_analysis['mode_equity_vs_efficiency'] = {
                        'correlation': equity_eff_corr,
                        'p_value': equity_eff_p,
                        'interpretation': 'negative_tradeoff' if equity_eff_corr < -0.3 else 'positive_synergy' if equity_eff_corr > 0.3 else 'no_clear_relationship'
                    }
                except:
                    pass
            
            # Travel time equity vs system efficiency  
            if metric_data['travel_time_equity'] and metric_data['system_efficiency']:
                try:
                    time_eff_corr, time_eff_p = pearsonr(metric_data['travel_time_equity'], 
                                                    metric_data['system_efficiency'])
                    tradeoff_analysis['time_equity_vs_efficiency'] = {
                        'correlation': time_eff_corr,
                        'p_value': time_eff_p,
                        'interpretation': 'negative_tradeoff' if time_eff_corr < -0.3 else 'positive_synergy' if time_eff_corr > 0.3 else 'no_clear_relationship'
                    }
                except:
                    pass
            
            correlation_analysis['efficiency_tradeoffs'] = tradeoff_analysis
        
        return correlation_analysis

   
    def _average_spatial_data(self, spatial_data_list):
        """Average spatial equity data across multiple runs"""
        if not spatial_data_list:
            return {}
        
        averaged_data = {}
        all_regions = set()
        
        # Collect all regions
        for spatial_data in spatial_data_list:
            all_regions.update(spatial_data.keys())
        
        # Average each metric for each region
        for region in all_regions:
            region_metrics = []
            for spatial_data in spatial_data_list:
                if region in spatial_data:
                    region_metrics.append(spatial_data[region])
            
            if region_metrics:
                averaged_data[region] = {}
                metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
                
                for metric in metrics:
                    values = [rm.get(metric, 0) for rm in region_metrics if isinstance(rm.get(metric, 0), (int, float))]
                    if values:
                        averaged_data[region][metric] = sum(values) / len(values)
                    else:
                        averaged_data[region][metric] = 0
        
        return averaged_data
    def _plot_detailed_statistics(self, ax, comparison_results: Dict[str, Any], metric: str):
        """
        Plot detailed statistical analysis for a metric
        """
        detailed_stats = self._calculate_detailed_statistics(comparison_results, metric)
        
        # Create a multi-panel statistical summary
        ax.clear()
        
        # Plot 1: Distribution comparison across topologies
        topology_names = []
        means = []
        stds = []
        skewness_vals = []
        
        for topology, stats in detailed_stats['topology_analysis'].items():
            if 'distribution' in stats:
                topology_names.append(topology)
                
                # Get overall mean from parameter sensitivity
                if 'parameter_sensitivity' in stats:
                    param_means = [data['mean'] for data in stats['parameter_sensitivity'].values()]
                    means.append(np.mean(param_means) if param_means else 0)
                    stds.append(np.mean([data['std'] for data in stats['parameter_sensitivity'].values()]) if param_means else 0)
                else:
                    means.append(0)
                    stds.append(0)
                    
                skewness_vals.append(stats['distribution']['skewness'])
        
        if topology_names:
            # Create bar plot with error bars
            x_pos = np.arange(len(topology_names))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                        color=['#2E86AB', '#A23B72', '#F18F01'][:len(topology_names)])
     

            ax.set_xlabel('Network Topology')
            ax.set_ylabel(f'{metric.replace("_", " ").title()}')
            ax.set_title(f'Statistical Distribution Analysis\n{metric.replace("_", " ").title()}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(topology_names, rotation=45)
            
            # Add skewness annotations
            for i, (bar, skew) in enumerate(zip(bars, skewness_vals)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.01,
                    f'Skew: {skew:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Add convergence info if available
            conv_info = []
            for topology, stats in detailed_stats['topology_analysis'].items():
                if 'convergence_metrics' in stats and stats['convergence_metrics']:
                    avg_conv = np.mean([data['avg_convergence_step'] for data in stats['convergence_metrics'].values()])
                    conv_info.append(f'{topology}: {avg_conv:.0f} steps')
            
            if conv_info:
                ax.text(0.02, 0.98, 'Avg Convergence:\n' + '\n'.join(conv_info), 
                    transform=ax.transAxes, va='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'{metric.replace("_", " ").title()}\nNo Statistical Data Available', 
                ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(True, alpha=0.3)

    def _plot_correlation_analysis(self, ax, comparison_results: Dict[str, Any], metric: str):
        """
        Plot correlation analysis for a metric
        """
        correlation_analysis = self._calculate_correlation_analysis(comparison_results, metric)
        
        ax.clear()
        
        # Create correlation heatmap if we have cross-metric correlations
        if 'cross_metric_correlations' in correlation_analysis and correlation_analysis['cross_metric_correlations']:
            correlations = correlation_analysis['cross_metric_correlations']
            
            # Build correlation matrix for heatmap
            metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency', 'execution_time']
            corr_matrix = np.zeros((len(metrics), len(metrics)))
            
            for i, metric1 in enumerate(metrics):
                for j, metric2 in enumerate(metrics):
                    if metric1 in correlations and metric2 in correlations[metric1]:
                        if 'pearson' in correlations[metric1][metric2]:
                            corr_matrix[i, j] = correlations[metric1][metric2]['pearson']['correlation']
                    elif i == j:
                        corr_matrix[i, j] = 1.0
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            
            # Add labels
            ax.set_xticks(range(len(metrics)))
            ax.set_yticks(range(len(metrics)))
            ax.set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45, ha='right')
            ax.set_yticklabels([m.replace('_', '\n') for m in metrics])
            
            # Add correlation values as text
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                                ha='center', va='center', fontsize=8,
                                color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
            
            ax.set_title(f'Cross-Metric Correlations\n(Focus: {metric.replace("_", " ").title()})')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Pearson Correlation')
            
            # Add parameter correlation info
            if 'parameter_correlations' in correlation_analysis:
                param_corr_text = []
                for topology, corr_data in correlation_analysis['parameter_correlations'].items():
                    if 'pearson' in corr_data:
                        corr_val = corr_data['pearson']['correlation']
                        param_corr_text.append(f'{topology}: r={corr_val:.2f}')
                
                if param_corr_text:
                    ax.text(1.02, 0.5, 'Parameter\nCorrelations:\n' + '\n'.join(param_corr_text), 
                        transform=ax.transAxes, va='center', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        else:
            ax.text(0.5, 0.5, f'{metric.replace("_", " ").title()}\nCorrelation Analysis\n(Insufficient Data)', 
                ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(True, alpha=0.3)


    def create_spatial_equity_heatmaps(self, results_data, topology_type):
        """Generate spatial equity heatmaps for TRB paper"""
        # Extract spatial data from results
        spatial_data = {}
        for config_name, config_results in results_data.items():
            print(f"🔍 Config {config_name} keys: {list(config_results.keys())}")
            if 'runs' in config_results:
                for run_idx, run in enumerate(config_results['runs']):
                    print(f"🔍 Run {run_idx} keys: {list(run.keys())}")
                    if 'final_spatial_equity' in run and 'error' not in run:
                        spatial_data[config_name] = run['final_spatial_equity']
                        print(f"✅ Found spatial data for {config_name}")
                        break
            else:
                print(f"❌ No 'runs' found in {config_name}")
        
        if not spatial_data:
            print("No spatial data available for visualization")
            return
        
        # Create heatmap visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Spatial Equity Distribution - {topology_type.replace("_", " ").title()} Network', fontsize=16)
        
        regions = ['CBD_Inner', 'Western_Sydney', 'Eastern_Suburbs', 'North_Shore', 'South_Sydney']
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        
        for i, metric in enumerate(metrics):
            for j, config_name in enumerate(list(spatial_data.keys())[:2]):  # Show first 2 configs
                ax = axes[j, i]
                
                # Extract metric values by region
                region_values = []
                region_labels = []
                
                for region in regions:
                    if region in spatial_data[config_name]:
                        region_data = spatial_data[config_name][region]
                        if metric in region_data and not isinstance(region_data[metric], str):
                            region_values.append(region_data[metric])
                            region_labels.append(region.replace('_', ' '))
                
                if region_values:
                    # Create bar plot for regional comparison
                    bars = ax.bar(region_labels, region_values)
                    ax.set_title(f'{config_name}: {metric.replace("_", " ").title()}')
                    ax.set_ylabel('Equity Score')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Color code bars by performance
                    if metric in ['mode_choice_equity', 'travel_time_equity']:  # Lower is better
                        colors = ['green' if v < np.median(region_values) else 'red' for v in region_values]
                    else:  # Higher is better for efficiency
                        colors = ['red' if v < np.median(region_values) else 'green' for v in region_values]
                    
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = self.output_dir / f"{topology_type}_spatial_equity_heatmap_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Spatial equity heatmap saved: {output_path}")
        return output_path
    
    def run_wctr_statistical_study(self, num_runs, topology_configs, steps_per_run) -> Dict[str, Any]:
        """
        Run complete WCTR statistical study - EVERYTHING AT ONCE
        """
        print(f"\n🎯 WCTR STATISTICAL ANALYSIS - {num_runs} runs per parameter")
        print(f"{'='*60}")
        

        # Run enhanced analysis with statistics
        results = self.run_multi_topology_comparison(
            topology_configs=topology_configs,
            num_runs=num_runs,  # This is the key change - more runs
            steps_per_run=steps_per_run,
            num_commuters=140
        )
        
        # 🔍 DEBUG: Check what results actually contains
        print(f"\n🔍 DEBUG: Results keys = {list(results.keys())}")
        print(f"🔍 DEBUG: Results type = {type(results)}")
        for key, value in results.items():
            print(f"🔍 DEBUG: results['{key}'] = {type(value)}")
            if isinstance(value, dict) and len(value) < 10:
                print(f"        -> {list(value.keys())}")
        # Add statistical analysis
        results['statistical_analysis'] = self._calculate_wctr_statistics(results)
        
        # 🔥 ADD THIS SECTION - PARAMETER COMPARISON FOR EACH TOPOLOGY 🔥
        # print(f"\n📊 Generating parameter comparison visualizations...")
        # for topology_type in topology_configs.keys():
        #     print(f"  Creating parameter comparison for {topology_type}...")
            
        #     # Extract results for this specific topology
        #     topology_results = {
        #         'topology_type': topology_type,
        #         'results': {}
        #     }
            
            # # Generate parameter comparison visualization
            # # 🔧 FIX: Use 'topology_results' instead of 'results'
            # if topology_type in results['topology_results']:
            #     topology_data = results['topology_results'][topology_type]
                
            #     # 🔍 DEBUG: Check the structure of topology_data
            #     print(f"🔍 DEBUG: topology_data for {topology_type} = {type(topology_data)}")
            #     print(f"🔍 DEBUG: topology_data keys/values = {list(topology_data.items()) if isinstance(topology_data, dict) else topology_data}")
                
            #     if isinstance(topology_data, dict):
            #         if 'results' in topology_data:
            #             for param_val, param_data in topology_data['results'].items():
            #                 config_name = f"{topology_type}_{param_val}"
            #                 topology_results['results'][config_name] = param_data
            #     else:
            #         print(f"⚠️ Unexpected topology_data type: {type(topology_data)}")
                
            #     # Generate parameter comparison visualization
            #     if topology_results['results']:
            #         fig = self.generate_parameter_comparison_visualizations(topology_results)
            #         if fig:
            #             print(f"    ✅ Parameter comparison saved for {topology_type}")
            #     else:
            #         print(f"    ⚠️ No data found for {topology_type}")
            # else:
            #     print(f"    ⚠️ {topology_type} not found in topology_results")
        # ✨ EXISTING CROSS-TOPOLOGY ANALYSIS ✨
        results = self._add_cross_topology_analysis(results)
        self._save_cross_topology_report(results)
        self.create_cross_topology_visualizations(results)
        
        # Generate statistical plots
        self._create_wctr_plots(results)
        
        # Save statistical report
        self._save_wctr_report(results)
        
        print(f"\n🎉 WCTR Analysis Complete! Check {self.output_dir}")
        return results

    def _calculate_wctr_statistics(self, results: Dict) -> Dict:
        """Calculate ANOVA, confidence intervals, effect sizes"""
        import scipy.stats as stats
        
        stats_results = {'anova': {}, 'confidence_intervals': {}, 'cross_topology': {}}
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        
        # For each topology, calculate ANOVA across parameters
        for topology_name, topology_data in results['topology_results'].items():
            stats_results['anova'][topology_name] = {}
            stats_results['confidence_intervals'][topology_name] = {}
            
            for metric in metrics:
                # Collect data groups for ANOVA
                groups = []
                group_labels = []
                
                if 'results' in topology_data:
                    for param_key, param_data in topology_data['results'].items():
                        if 'runs' in param_data:
                            values = [run[f'final_{metric}'] for run in param_data['runs'] 
                                    if f'final_{metric}' in run]
                            if len(values) >= 5:
                                groups.append(values)
                                group_labels.append(param_data.get('parameter_value', param_key))
                
                if len(groups) >= 2:
                    # ANOVA
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    # Effect size
                    total_mean = np.mean([val for group in groups for val in group])
                    ss_between = sum(len(group) * (np.mean(group) - total_mean)**2 for group in groups)
                    ss_total = sum(sum((val - total_mean)**2 for val in group) for group in groups)
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    stats_results['anova'][topology_name][metric] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'eta_squared': eta_squared,
                        'effect_size': 'large' if eta_squared > 0.14 else 'medium' if eta_squared > 0.06 else 'small'
                    }
                    
                    # Confidence intervals
                    ci_data = {}
                    for group, label in zip(groups, group_labels):
                        mean_val = np.mean(group)
                        sem_val = stats.sem(group)
                        ci_lower, ci_upper = stats.t.interval(0.95, len(group)-1, loc=mean_val, scale=sem_val)
                        ci_data[str(label)] = {
                            'mean': mean_val, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
                            'sample_size': len(group)
                        }
                    stats_results['confidence_intervals'][topology_name][metric] = ci_data
        
        return stats_results

    def _create_wctr_plots(self, results: Dict):
        """Create publication-quality statistical plots"""
        
        # Plot 1: Confidence intervals for all topologies
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        colors = {'degree_constrained': '#2E86AB', 'small_world': '#A23B72', 'scale_free': '#F18F01'}

        for col, metric in enumerate(metrics):
            ax = axes[col]
            
            for topology in results['topology_results'].keys():
                if ('statistical_analysis' in results and 
                    topology in results['statistical_analysis']['confidence_intervals']):
                    
                    ci_data = results['statistical_analysis']['confidence_intervals'][topology].get(metric, {})
                    
                    if ci_data:
                        param_keys = list(ci_data.keys())
                        params = [float(p) for p in param_keys]
                        means = [ci_data[p]['mean'] for p in param_keys]
                        ci_lowers = [ci_data[p]['ci_lower'] for p in param_keys]
                        ci_uppers = [ci_data[p]['ci_upper'] for p in param_keys]
                        
                        # Sort by parameter
                        sorted_data = sorted(zip(params, means, ci_lowers, ci_uppers))
                        params, means, ci_lowers, ci_uppers = zip(*sorted_data)
                        
                        ax.errorbar(params, means, 
                                yerr=[np.array(means) - np.array(ci_lowers), 
                                        np.array(ci_uppers) - np.array(means)],
                                fmt='o-', label=topology.replace('_', ' ').title(),
                                capsize=5, linewidth=2, color=colors.get(topology, 'gray'))
            
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()}\n(95% Confidence Intervals)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f"wctr_statistical_analysis_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Statistical plots saved: {save_path}")

    def _add_cross_topology_analysis(self, results: Dict) -> Dict:
        """
        Add cross-topology statistical comparison - THE MISSING PIECE!
        This compares topology performance directly (your main research question)
        """
        import scipy.stats as stats
        from scipy.stats import tukey_hsd
        
        print("📊 Running cross-topology statistical analysis...")
        
        cross_topology_stats = {
            'topology_comparison': {},
            'pairwise_tests': {},
            'best_performers': {},
            'practical_significance': {}
        }
        
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        
        for metric in metrics:
            print(f"  📈 Analyzing {metric}...")
            
            # Collect ALL data by topology (pooled across parameters)
            topology_data = {}
            
            for topology_name, topology_results in results['topology_results'].items():
                all_values = []
                
                # Pool all runs across all parameters for this topology
                if 'results' in topology_results:
                    for param_data in topology_results['results'].values():
                        if 'runs' in param_data:
                            values = [run[f'final_{metric}'] for run in param_data['runs'] 
                                    if f'final_{metric}' in run]
                            all_values.extend(values)
                
                if all_values:
                    topology_data[topology_name] = all_values
            
            if len(topology_data) >= 2:
                # MAIN RESEARCH QUESTION: Are topologies significantly different?
                topology_names = list(topology_data.keys())
                topology_values = list(topology_data.values())
                
                # Overall ANOVA
                f_stat, p_value = stats.f_oneway(*topology_values)
                
                cross_topology_stats['topology_comparison'][metric] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'topology_means': {name: np.mean(values) for name, values in topology_data.items()},
                    'topology_stds': {name: np.std(values) for name, values in topology_data.items()},
                    'sample_sizes': {name: len(values) for name, values in topology_data.items()}
                }
                
                # Pairwise comparisons (which topologies differ?)
                pairwise_results = {}
                for i, topo1 in enumerate(topology_names):
                    for j, topo2 in enumerate(topology_names[i+1:], i+1):
                        
                        # Independent t-test
                        t_stat, t_p = stats.ttest_ind(
                            topology_data[topo1], 
                            topology_data[topo2],
                            equal_var=False  # Welch's t-test
                        )
                        
                        # Effect size (Cohen's d)
                        mean1, mean2 = np.mean(topology_data[topo1]), np.mean(topology_data[topo2])
                        std1, std2 = np.std(topology_data[topo1]), np.std(topology_data[topo2])
                        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                        
                        # Effect size interpretation
                        if abs(cohens_d) < 0.2:
                            effect_interpretation = "negligible"
                        elif abs(cohens_d) < 0.5:
                            effect_interpretation = "small"
                        elif abs(cohens_d) < 0.8:
                            effect_interpretation = "medium"
                        else:
                            effect_interpretation = "large"
                        
                        pairwise_results[f"{topo1}_vs_{topo2}"] = {
                            't_statistic': t_stat,
                            'p_value': t_p,
                            'significant': t_p < 0.05,
                            'cohens_d': cohens_d,
                            'effect_size': effect_interpretation,
                            'mean_difference': mean1 - mean2
                        }
                
                cross_topology_stats['pairwise_tests'][metric] = pairwise_results
                
                # Best performer identification
                topology_means = cross_topology_stats['topology_comparison'][metric]['topology_means']
                
                if metric in ['mode_choice_equity', 'travel_time_equity']:  # Lower is better
                    best_topology = min(topology_means, key=topology_means.get)
                    worst_topology = max(topology_means, key=topology_means.get)
                    ranking = sorted(topology_means.items(), key=lambda x: x[1])
                else:  # Higher is better (system_efficiency)
                    best_topology = max(topology_means, key=topology_means.get)
                    worst_topology = min(topology_means, key=topology_means.get)
                    ranking = sorted(topology_means.items(), key=lambda x: x[1], reverse=True)
                
                cross_topology_stats['best_performers'][metric] = {
                    'best_topology': best_topology,
                    'best_mean': topology_means[best_topology],
                    'worst_topology': worst_topology,
                    'worst_mean': topology_means[worst_topology],
                    'performance_gap': abs(topology_means[best_topology] - topology_means[worst_topology]),
                    'ranking': ranking
                }
        
        # Add to main results
        results['cross_topology_analysis'] = cross_topology_stats
        
        print("✅ Cross-topology analysis complete!")
        return results

    def _save_cross_topology_report(self, results: Dict):
        """Save cross-topology analysis results"""
        if 'cross_topology_analysis' not in results:
            return
        
        report_path = self.output_dir / f"cross_topology_analysis_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("CROSS-TOPOLOGY STATISTICAL ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            cross_stats = results['cross_topology_analysis']
            
            f.write("TOPOLOGY PERFORMANCE COMPARISON:\n")
            f.write("-" * 40 + "\n")
            
            for metric, comparison in cross_stats['topology_comparison'].items():
                f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
                f.write(f"  Overall ANOVA: F={comparison['f_statistic']:.3f}, ")
                f.write(f"p={comparison['p_value']:.4f} ")
                f.write(f"({'SIGNIFICANT' if comparison['significant'] else 'NOT SIGNIFICANT'})\n")
                
                f.write(f"  Topology Means:\n")
                for topo, mean_val in comparison['topology_means'].items():
                    f.write(f"    {topo}: {mean_val:.3f} ± {comparison['topology_stds'][topo]:.3f}\n")
            
            f.write(f"\nBEST PERFORMERS:\n")
            f.write("-" * 20 + "\n")
            for metric, best_data in cross_stats['best_performers'].items():
                f.write(f"{metric}: {best_data['best_topology']} ")
                f.write(f"(mean: {best_data['best_mean']:.3f})\n")
            
            f.write(f"\nPAIRWISE COMPARISONS:\n")
            f.write("-" * 25 + "\n")
            for metric, pairwise in cross_stats['pairwise_tests'].items():
                f.write(f"\n{metric}:\n")
                for comparison, stats_data in pairwise.items():
                    f.write(f"  {comparison}: ")
                    f.write(f"p={stats_data['p_value']:.4f}, ")
                    f.write(f"Cohen's d={stats_data['cohens_d']:.3f} ")
                    f.write(f"({stats_data['effect_size']} effect), ")
                    f.write(f"{'SIG' if stats_data['significant'] else 'NS'}\n")
        
        print(f"✅ Cross-topology report saved: {report_path}")


    def _save_wctr_report(self, results: Dict):
        """Save statistical summary for WCTR paper"""
        report_path = self.output_dir / f"wctr_statistical_summary_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("WCTR STATISTICAL ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            if 'statistical_analysis' in results:
                f.write("ANOVA RESULTS (Parameter Effects Within Topologies):\n")
                f.write("-" * 50 + "\n")
                
                for topology, metrics in results['statistical_analysis']['anova'].items():
                    f.write(f"\n{topology.upper().replace('_', ' ')}:\n")
                    for metric, anova_data in metrics.items():
                        sig_text = "SIGNIFICANT" if anova_data['significant'] else "NOT SIGNIFICANT"
                        f.write(f"  {metric}: F={anova_data['f_statistic']:.3f}, ")
                        f.write(f"p={anova_data['p_value']:.4f} ({sig_text}), ")
                        f.write(f"η²={anova_data['eta_squared']:.3f} ({anova_data['effect_size']} effect)\n")
            
            # Sample sizes
            f.write(f"\nSAMPLE SIZES:\n")
            f.write("-" * 20 + "\n")
            for topology, topology_data in results['topology_results'].items():
                total_runs = sum(len(param_data.get('runs', [])) 
                            for param_data in topology_data.get('results', {}).values())
                f.write(f"{topology}: {total_runs} total simulation runs\n")
            
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✅ Statistical report saved: {report_path}")

    # Replace your create_cross_topology_visualizations method with this corrected version:

    def create_cross_topology_visualizations(self, results: Dict):
        """Create publication-quality cross-topology comparison plots with proper scaling"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        if 'cross_topology_analysis' not in results:
            print("⚠️ No cross-topology analysis found")
            return
        
        cross_stats = results['cross_topology_analysis']
        
        # FIXED: Use 2x3 grid properly
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Cross-Topology Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Normalized Topology Means Comparison
        self._plot_normalized_topology_means(axes[0, 0], cross_stats)
        
        # Plot 2-4: Original Values (Separate Metrics) - FIXED INDEXING
        metric_axes = [axes[0, 1], axes[0, 2], axes[1, 0]]  # Three specific subplot positions
        self._plot_separate_metric_comparisons(metric_axes, cross_stats)
        
        # Plot 5: Box Plots by Topology (Normalized)
        self._plot_normalized_boxplots(axes[1, 1], results)
        
        # Plot 6: Trade-off Analysis (Normalized)  
        self._plot_normalized_tradeoff(axes[1, 2], cross_stats)
        
        plt.tight_layout()
        save_path = self.output_dir / f"cross_topology_comparison_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Cross-topology visualizations saved: {save_path}")

    def _plot_separate_metric_comparisons(self, axes_list, cross_stats):
        """Plot each metric separately with appropriate scales"""
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        metric_labels = ['Mode Choice Equity\n(Lower = Better)', 'Travel Time Equity\n(Lower = Better)', 'System Efficiency\n(Higher = Better)']
        topologies = ['degree_constrained', 'small_world', 'scale_free']
        colors = ['#2E86AB', '#A23B72', '#F18F01']

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            if idx >= len(axes_list):  # Safety check
                break
                
            ax = axes_list[idx]  # Use the passed axes list
            
            if metric in cross_stats['topology_comparison']:
                comp_data = cross_stats['topology_comparison'][metric]
                means = comp_data['topology_means']
                stds = comp_data['topology_stds']
                
                # Create bar plot
                topo_names = [t.replace('_', ' ').title() for t in topologies]
                topo_means = [means.get(t, 0) for t in topologies]
                topo_stds = [stds.get(t, 0) for t in topologies]
                
                bars = ax.bar(topo_names, topo_means, yerr=topo_stds, 
                            color=colors, alpha=0.8, capsize=5)
                
                # Add value labels on bars
                for bar, mean_val, std_val in zip(bars, topo_means, topo_stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std_val,
                        f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
            else:
                # Handle case where metric data is missing
                ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_ylabel(label)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


    def _plot_topology_means(self, ax, cross_stats):
        """Plot topology means with error bars"""
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        topologies = ['degree_constrained', 'small_world', 'scale_free']
        
        x = np.arange(len(metrics))
        width = 0.25
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, topology in enumerate(topologies):
            means = []
            stds = []
            
            for metric in metrics:
                if metric in cross_stats['topology_comparison']:
                    comp_data = cross_stats['topology_comparison'][metric]
                    means.append(comp_data['topology_means'].get(topology, 0))
                    stds.append(comp_data['topology_stds'].get(topology, 0))
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x + i*width, means, width, yerr=stds, 
                label=topology.replace('_', ' ').title(), 
                color=colors[i], alpha=0.8, capsize=5)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Performance')
        ax.set_title('Topology Performance Comparison\n(Error bars = ±1 SD)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _normalize_metric_values(self, cross_stats, metric):
        """Normalize metric values to 0-1 scale for comparison"""
        if metric not in cross_stats['topology_comparison']:
            return {}, {}
        
        comp_data = cross_stats['topology_comparison'][metric]
        means = comp_data['topology_means']
        stds = comp_data['topology_stds']
        
        # Normalize based on metric type
        if metric in ['mode_choice_equity', 'travel_time_equity']:
            # Lower is better - normalize so 0 = worst, 1 = best
            max_val = max(means.values())
            min_val = min(means.values())
            range_val = max_val - min_val if max_val != min_val else 1
            
            normalized_means = {topo: 1 - (val - min_val) / range_val for topo, val in means.items()}
            normalized_stds = {topo: std / range_val for topo, std in stds.items()}
        else:
            # Higher is better - normalize so 1 = best, 0 = worst
            max_val = max(means.values())
            min_val = min(means.values())
            range_val = max_val - min_val if max_val != min_val else 1
            
            normalized_means = {topo: (val - min_val) / range_val for topo, val in means.items()}
            normalized_stds = {topo: std / range_val for topo, std in stds.items()}
        
        return normalized_means, normalized_stds

    def _plot_normalized_topology_means(self, ax, cross_stats):
        """Plot normalized topology means (all metrics 0-1 scale)"""
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        topologies = ['degree_constrained', 'small_world', 'scale_free']
        
        x = np.arange(len(metrics))
        width = 0.25
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for i, topology in enumerate(topologies):
            normalized_means = []
            normalized_stds = []
            
            for metric in metrics:
                norm_means, norm_stds = self._normalize_metric_values(cross_stats, metric)
                normalized_means.append(norm_means.get(topology, 0))
                normalized_stds.append(norm_stds.get(topology, 0))
            
            ax.bar(x + i*width, normalized_means, width, yerr=normalized_stds, 
                label=topology.replace('_', ' ').title(), 
                color=colors[i], alpha=0.8, capsize=5)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Normalized Performance (0-1)')
        ax.set_title('Normalized Performance Comparison\n(Higher = Better for All Metrics)')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)


    def _plot_normalized_boxplots(self, ax, results):
        """Create normalized box plots comparing topologies"""
        # Collect all data for normalization
        all_data = {}
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        
        for metric in metrics:
            all_data[metric] = []
            
            for topology_name, topology_results in results['topology_results'].items():
                if 'results' in topology_results:
                    for param_data in topology_results['results'].values():
                        if 'runs' in param_data:
                            for run in param_data['runs']:
                                value = run.get(f'final_{metric}', 0)
                                all_data[metric].append(value)
        
        # Calculate normalization parameters
        norm_params = {}
        for metric in metrics:
            if all_data[metric]:
                min_val = min(all_data[metric])
                max_val = max(all_data[metric])
                range_val = max_val - min_val if max_val != min_val else 1
                norm_params[metric] = {'min': min_val, 'max': max_val, 'range': range_val}
        
        # Collect normalized data for plotting
        plot_data = []
        
        for topology_name, topology_results in results['topology_results'].items():
            if 'results' in topology_results:
                for param_data in topology_results['results'].values():
                    if 'runs' in param_data:
                        for run in param_data['runs']:
                            row = {'topology': topology_name.replace('_', ' ').title()}
                            
                            for metric in metrics:
                                value = run.get(f'final_{metric}', 0)
                                if metric in norm_params:
                                    if metric in ['mode_choice_equity', 'travel_time_equity']:
                                        # Lower is better - invert normalization
                                        normalized = 1 - (value - norm_params[metric]['min']) / norm_params[metric]['range']
                                    else:
                                        # Higher is better
                                        normalized = (value - norm_params[metric]['min']) / norm_params[metric]['range']
                                    row[f'normalized_{metric}'] = normalized
                            
                            plot_data.append(row)
        
        if plot_data:
            import pandas as pd
            df = pd.DataFrame(plot_data)
            
            # Plot normalized mode choice equity as primary metric
            df.boxplot(column='normalized_mode_choice_equity', by='topology', ax=ax)
            ax.set_title('Normalized Mode Choice Equity by Topology\n(Higher = Better)')
            ax.set_xlabel('Topology')
            ax.set_ylabel('Normalized Equity Score (0-1)')
            ax.set_ylim(0, 1)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_normalized_tradeoff(self, ax, cross_stats):
        """Plot normalized equity vs efficiency trade-off"""
        if 'topology_comparison' not in cross_stats:
            ax.text(0.5, 0.5, 'No trade-off data available', 
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        topo_comp = cross_stats['topology_comparison']
        topologies = ['degree_constrained', 'small_world', 'scale_free']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # Calculate normalized equity (combine mode choice and travel time)
        equity_scores = []
        efficiency_scores = []
        topo_labels = []
        
        for topology in topologies:
            # Get normalized equity (average of mode choice and travel time)
            if 'mode_choice_equity' in topo_comp and 'travel_time_equity' in topo_comp:
                mode_means, _ = self._normalize_metric_values(cross_stats, 'mode_choice_equity')
                travel_means, _ = self._normalize_metric_values(cross_stats, 'travel_time_equity')
                
                equity_avg = (mode_means.get(topology, 0) + travel_means.get(topology, 0)) / 2
                equity_scores.append(equity_avg)
            
            # Get normalized efficiency
            if 'system_efficiency' in topo_comp:
                eff_means, _ = self._normalize_metric_values(cross_stats, 'system_efficiency')
                efficiency_scores.append(eff_means.get(topology, 0))
            
            topo_labels.append(topology.replace('_', ' ').title())
        
        if len(equity_scores) == len(efficiency_scores) == len(topo_labels):
            for i, (topo, eq, eff) in enumerate(zip(topo_labels, equity_scores, efficiency_scores)):
                ax.scatter(eff, eq, s=200, c=colors[i], alpha=0.7, label=topo)
                ax.annotate(topo, (eff, eq), xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Normalized System Efficiency (0-1)')
            ax.set_ylabel('Normalized Average Equity (0-1)')
            ax.set_title('Equity vs Efficiency Trade-off\n(Higher = Better for Both)')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add diagonal line for reference
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Balance')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for trade-off analysis', 
                    ha='center', va='center', transform=ax.transAxes)

    # For the parameter sensitivity plots, also add this normalization method:

    def create_normalized_parameter_sensitivity_plots(self, results: Dict):
        """Create parameter sensitivity plots with normalized y-axes"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        colors = {'degree_constrained': '#2E86AB', 'small_world': '#A23B72', 'scale_free': '#F18F01'}
        
        # Collect all data for normalization
        all_metric_data = {metric: [] for metric in metrics}
        
        for topology_name, topology_data in results['topology_results'].items():
            if 'results' in topology_data:
                for param_data in topology_data['results'].values():
                    if 'runs' in param_data:
                        for run in param_data['runs']:
                            for metric in metrics:
                                value = run.get(f'final_{metric}', 0)
                                all_metric_data[metric].append(value)
        
        # Calculate global normalization parameters
        norm_params = {}
        for metric in metrics:
            if all_metric_data[metric]:
                min_val = min(all_metric_data[metric])
                max_val = max(all_metric_data[metric])
                range_val = max_val - min_val if max_val != min_val else 1
                norm_params[metric] = {'min': min_val, 'max': max_val, 'range': range_val}
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for topology_name, topology_data in results['topology_results'].items():
                if 'results' in topology_data:
                    params = []
                    normalized_means = []
                    normalized_ci_lowers = []
                    normalized_ci_uppers = []
                    
                    for param_key in sorted(topology_data['results'].keys(), key=float):
                        param_data = topology_data['results'][param_key]
                        if 'runs' in param_data and len(param_data['runs']) > 0:
                            # Get raw values
                            values = [run[f'final_{metric}'] for run in param_data['runs'] 
                                    if f'final_{metric}' in run]
                            
                            if values and metric in norm_params:
                                # Normalize values
                                if metric in ['mode_choice_equity', 'travel_time_equity']:
                                    # Lower is better - invert
                                    normalized_values = [1 - (v - norm_params[metric]['min']) / norm_params[metric]['range'] for v in values]
                                else:
                                    # Higher is better
                                    normalized_values = [(v - norm_params[metric]['min']) / norm_params[metric]['range'] for v in values]
                                
                                # Calculate statistics on normalized values
                                mean_norm = np.mean(normalized_values)
                                std_norm = np.std(normalized_values)
                                ci_lower = mean_norm - 1.96 * std_norm / np.sqrt(len(normalized_values))
                                ci_upper = mean_norm + 1.96 * std_norm / np.sqrt(len(normalized_values))
                                
                                params.append(float(param_key))
                                normalized_means.append(mean_norm)
                                normalized_ci_lowers.append(ci_lower)
                                normalized_ci_uppers.append(ci_upper)
                    
                    if params:
                        # Sort by parameter value
                        sorted_data = sorted(zip(params, normalized_means, normalized_ci_lowers, normalized_ci_uppers))
                        params, normalized_means, normalized_ci_lowers, normalized_ci_uppers = zip(*sorted_data)
                        
                        ax.errorbar(params, normalized_means, 
                                yerr=[np.array(normalized_means) - np.array(normalized_ci_lowers), 
                                        np.array(normalized_ci_uppers) - np.array(normalized_means)],
                                fmt='o-', label=topology_name.replace('_', ' ').title(),
                                capsize=5, linewidth=2, color=colors.get(topology_name, 'gray'))
            
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel(f'Normalized {metric.replace("_", " ").title()}\n(Higher = Better)')
            ax.set_title(f'Parameter Sensitivity: {metric.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        save_path = self.output_dir / f"normalized_parameter_sensitivity_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Normalized parameter sensitivity plots saved: {save_path}")
    def _plot_topology_boxplots(self, ax, results):
        """Create box plots comparing topologies"""
        # Collect all data for box plots
        plot_data = []
        
        for topology_name, topology_results in results['topology_results'].items():
            if 'results' in topology_results:
                for param_data in topology_results['results'].values():
                    if 'runs' in param_data:
                        for run in param_data['runs']:
                            plot_data.append({
                                'topology': topology_name.replace('_', ' ').title(),
                                'mode_choice_equity': run.get('final_mode_choice_equity', 0),
                                'travel_time_equity': run.get('final_travel_time_equity', 0),
                                'system_efficiency': run.get('final_system_efficiency', 0)
                            })
        
        if plot_data:
            import pandas as pd
            df = pd.DataFrame(plot_data)
            
            # Plot mode choice equity as example
            df.boxplot(column='mode_choice_equity', by='topology', ax=ax)
            ax.set_title('Mode Choice Equity Distribution by Topology')
            ax.set_xlabel('Topology')
            ax.set_ylabel('Mode Choice Equity (lower = better)')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def _plot_effect_sizes_heatmap(self, ax, cross_stats):
        """Plot Cohen's d effect sizes as heatmap"""
        topologies = ['degree_constrained', 'small_world', 'scale_free']
        metrics = ['mode_choice_equity', 'travel_time_equity', 'system_efficiency']
        
        # Create effect size matrix
        effect_matrix = np.zeros((len(topologies), len(topologies)))
        
        for i, topo1 in enumerate(topologies):
            for j, topo2 in enumerate(topologies):
                if i != j:
                    # Find effect size from pairwise comparisons
                    for metric in metrics:
                        if metric in cross_stats['pairwise_tests']:
                            pairwise = cross_stats['pairwise_tests'][metric]
                            comparison_key = f"{topo1}_vs_{topo2}"
                            if comparison_key in pairwise:
                                effect_matrix[i, j] = abs(pairwise[comparison_key]['cohens_d'])
                                break
        
        # Create heatmap
        im = ax.imshow(effect_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        
        # Add labels
        ax.set_xticks(range(len(topologies)))
        ax.set_yticks(range(len(topologies)))
        ax.set_xticklabels([t.replace('_', '\n') for t in topologies])
        ax.set_yticklabels([t.replace('_', '\n') for t in topologies])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Cohen's d (Effect Size)")
        ax.set_title('Effect Sizes Between Topologies\n(Darker = Larger Effect)')
        
        # Add text annotations
        for i in range(len(topologies)):
            for j in range(len(topologies)):
                if effect_matrix[i, j] > 0:
                    ax.text(j, i, f'{effect_matrix[i, j]:.2f}', 
                        ha="center", va="center", color="white" if effect_matrix[i, j] > 0.5 else "black")

    def _plot_equity_efficiency_tradeoff(self, ax, cross_stats):
        """Plot equity vs efficiency trade-off"""
        if 'best_performers' not in cross_stats:
            ax.text(0.5, 0.5, 'No trade-off data available', 
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract data for trade-off plot
        topologies = []
        equity_scores = []  # Lower is better, so we'll invert
        efficiency_scores = []
        
        for metric, best_data in cross_stats['best_performers'].items():
            if metric == 'mode_choice_equity':
                # Use travel time equity as main equity metric (or combine both)
                pass
        
        # Get means from topology comparison
        if 'topology_comparison' in cross_stats:
            topo_comp = cross_stats['topology_comparison']
            
            for topology in ['degree_constrained', 'small_world', 'scale_free']:
                topologies.append(topology.replace('_', ' ').title())
                
                # Use mode choice equity (inverted so higher = better)
                if 'mode_choice_equity' in topo_comp:
                    equity = 1 / (1 + topo_comp['mode_choice_equity']['topology_means'].get(topology, 1))
                    equity_scores.append(equity)
                
                # Use system efficiency
                if 'system_efficiency' in topo_comp:
                    efficiency = topo_comp['system_efficiency']['topology_means'].get(topology, 1500)
                    efficiency_scores.append(efficiency)
        
        if len(topologies) == len(equity_scores) == len(efficiency_scores):
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            for i, (topo, eq, eff) in enumerate(zip(topologies, equity_scores, efficiency_scores)):
                ax.scatter(eff, eq, s=200, c=colors[i], alpha=0.7, label=topo)
                ax.annotate(topo, (eff, eq), xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('System Efficiency (higher = better)')
            ax.set_ylabel('Equity Index (higher = better)')
            ax.set_title('Equity vs Efficiency Trade-off')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for trade-off analysis', 
                    ha='center', va='center', transform=ax.transAxes)

    # def _export_network_structure(self, model) -> Dict[str, Any]:
    #     """Export detailed network structure for visualization"""
        
    #     graph = model.network_manager.active_network   # Get the actual NetworkX graph
        
    #     # Extract real network structure
    #     network_data = {
    #         'nodes': {},
    #         'edges': [],
    #         'shortcuts': [],
    #         'original_edges': [],
    #         'topology_analysis': {},
    #         'edge_congestion': {},
    #         'node_importance': {}
    #     }
        
    #     # Export node data with real attributes
    #     for node_id, node_attrs in graph.nodes(data=True):
    #         network_data['nodes'][node_id] = {
    #             'node_type': str(node_attrs.get('node_type', 'unknown')),
    #             'coordinates': node_attrs.get('coordinates', [0, 0]),
    #             'employment_weight': node_attrs.get('employment_weight', 0),
    #             'population_weight': node_attrs.get('population_weight', 0),
    #             'degree': graph.degree(node_id)
    #         }
        
    #     # Export edge data with real attributes
    #     for u, v, edge_attrs in graph.edges(data=True):
    #         edge_data = {
    #             'from_node': u,
    #             'to_node': v,
    #             'transport_mode': str(edge_attrs.get('transport_mode', 'unknown')),
    #             'travel_time': edge_attrs.get('travel_time', 0),
    #             'capacity': edge_attrs.get('capacity', 0),
    #             'route_id': edge_attrs.get('route_id', ''),
    #             'edge_type': edge_attrs.get('edge_type', 'unknown'),
    #             'distance': edge_attrs.get('distance', 0)
    #         }
            
    #         network_data['edges'].append(edge_data)
            
    #         # Classify edges as shortcuts vs original
    #         if edge_attrs.get('edge_type') == 'shortcut' or 'NEW_' in edge_attrs.get('route_id', ''):
    #             network_data['shortcuts'].append(edge_data)
    #         else:
    #             network_data['original_edges'].append(edge_data)
        
    #     # Calculate real network metrics
    #     network_data['topology_analysis'] = {
    #         'num_shortcuts': len(network_data['shortcuts']),
    #         'num_original': len(network_data['original_edges']),
    #         'shortcut_ratio': len(network_data['shortcuts']) / len(network_data['edges']) if network_data['edges'] else 0,
    #         'clustering_coefficient': nx.average_clustering(graph) if len(graph) > 0 else 0,
    #         'is_connected': nx.is_connected(graph),
    #         'diameter': nx.diameter(graph) if nx.is_connected(graph) else None
    #     }
        
    #     # Calculate node importance (centrality measures)
    #     if len(graph) > 0:
    #         betweenness = nx.betweenness_centrality(graph)
    #         degree_centrality = nx.degree_centrality(graph)
            
    #         for node_id in graph.nodes():
    #             network_data['node_importance'][node_id] = {
    #                 'betweenness_centrality': betweenness.get(node_id, 0),
    #                 'degree_centrality': degree_centrality.get(node_id, 0),
    #                 'is_hub': graph.degree(node_id) > np.mean([graph.degree(n) for n in graph.nodes()])
    #             }
        
    #     # Get edge usage statistics from commuters if available
    #     if hasattr(model, 'commuters'):
    #         edge_usage = {}
    #         for commuter in model.commuters:
    #             if hasattr(commuter, 'current_route') and commuter.current_route:
    #                 for i in range(len(commuter.current_route) - 1):
    #                     edge_key = f"{commuter.current_route[i]}_{commuter.current_route[i+1]}"
    #                     edge_usage[edge_key] = edge_usage.get(edge_key, 0) + 1
            
    #         network_data['edge_congestion'] = edge_usage
        
    #     return network_data     
    
    def _generate_parameter_comparison_plots(self, study_results):
        """Generate parameter comparison plots for ALL configurations"""
        
        try:
            from spatial_equity_heatmap import SpatialEquityVisualizer
            print("✅ SpatialEquityVisualizer imported successfully")
        except ImportError as e:
            print(f"⚠️ SpatialEquityVisualizer not found: {e}")
            return False
        
        # Extract ALL parameter configurations, not just the first one
        all_configs_data = {}
        topology_name = study_results['topology_type']
        
        for config_name, config_results in study_results['results'].items():
            runs = config_results.get('runs', [])
            if runs:
                # Use first run's data for each configuration
                run_data = runs[0]
                if 'final_spatial_equity' in run_data and run_data['final_spatial_equity']:
                    all_configs_data[config_name] = {
                        'spatial_equity': run_data['final_spatial_equity'],
                        'network_structure': run_data.get('final_network_structure', {}),
                        'performance_metrics': {
                            'mode_choice_equity': run_data.get('final_mode_choice_equity', 0),
                            'travel_time_equity': run_data.get('final_travel_time_equity', 0),
                            'system_efficiency': run_data.get('final_system_efficiency', 0)
                        }
                    }
        
        if len(all_configs_data) == 0:
            print("⚠️ No configuration data found")
            return False
        
        print(f"📊 Found {len(all_configs_data)} configurations: {list(all_configs_data.keys())}")
        
        # Create parameter comparison visualization
        visualizer = SpatialEquityVisualizer()
        
        # Create the parameter comparison plot
        comparison_path = f"{self.output_dir}/{topology_name}_parameter_comparison_{self.timestamp}.png"
        fig = visualizer.create_parameter_comparison_analysis(
            all_configs_data, topology_name, comparison_path
        )
        
        if fig:
            print(f"✅ Parameter comparison visualization saved: {comparison_path}")
            return True
        
        return False
    
    def _export_network_structure(self, model) -> Dict[str, Any]:
        """Export detailed network structure for visualization - FIXED VERSION"""
        
        graph = model.network_manager.active_network  # Correct path to NetworkX graph
        
        # Extract real network structure
        network_data = {
            'nodes': {},
            'edges': [],
            'shortcuts': [],
            'original_edges': [],
            'topology_analysis': {},
            'edge_congestion': {},
            'node_importance': {}
        }
        
        # Export node data with coordinates
        for node_id in graph.nodes():
            node_attrs = graph.nodes[node_id]
            # Get coordinates from base network if not in graph
            coordinates = node_attrs.get('coordinates', [0, 0])
            if coordinates == [0, 0] and hasattr(model.network_manager, 'base_network'):
                if node_id in model.network_manager.base_network.nodes:
                    coordinates = model.network_manager.base_network.nodes[node_id].coordinates
            
            network_data['nodes'][node_id] = {
                'coordinates': coordinates,
                'node_type': str(node_attrs.get('node_type', 'unknown')),
                'degree': graph.degree(node_id)
            }
        
        # CRITICAL FIX: Improved edge classification
        base_sydney_routes = {'T1_WESTERN', 'T4_ILLAWARRA', 'T8_AIRPORT', 
                            'BUS_380', 'BUS_E60', 'BUS_143', 'BUS_400', 'BUS_450'}
        
        print(f"🔍 DEBUGGING: Processing {graph.number_of_edges()} edges...")
        shortcut_count = 0
        original_count = 0
        
        for u, v, edge_attrs in graph.edges(data=True):
            edge_data = {
                'from_node': u,
                'to_node': v,
                'transport_mode': str(edge_attrs.get('transport_mode', 'unknown')),
                'travel_time': edge_attrs.get('travel_time', 0),
                'capacity': edge_attrs.get('capacity', 0),
                'route_id': edge_attrs.get('route_id', ''),
                'edge_type': edge_attrs.get('edge_type', 'unknown'),
                'distance': edge_attrs.get('distance', 0)
            }
            
            network_data['edges'].append(edge_data)
            
            # IMPROVED SHORTCUT CLASSIFICATION
            route_id = edge_attrs.get('route_id', '')
            edge_type = edge_attrs.get('edge_type', '')
            
            # Multiple criteria for shortcut detection
            is_shortcut = (
                edge_type == 'shortcut' or 
                edge_type == 'degree_shortcut' or
                'SW' in route_id or  # Small-world shortcuts
                'SF' in route_id or  # Scale-free shortcuts  
                'NEW_' in route_id or
                'E20' in route_id or  # Express routes
                'PT_' in route_id or  # Premium train
                'EX_' in route_id or  # Express bus
                (route_id not in base_sydney_routes and route_id != '')
            )
            
            if is_shortcut:
                network_data['shortcuts'].append(edge_data)
                shortcut_count += 1
     
            else:
                network_data['original_edges'].append(edge_data)
                original_count += 1
        
        print(f"📊 CLASSIFICATION RESULT: {shortcut_count} shortcuts, {original_count} original")
        
        # Calculate network metrics
        network_data['topology_analysis'] = {
            'num_shortcuts': len(network_data['shortcuts']),
            'num_original': len(network_data['original_edges']),
            'shortcut_ratio': len(network_data['shortcuts']) / len(network_data['edges']) if network_data['edges'] else 0,
            'total_edges': len(network_data['edges'])
        }
        
        # CRITICAL: Add edge usage statistics
        edge_usage = {}
        if hasattr(model, 'commuters'):
            for commuter in model.commuters:
                if hasattr(commuter, 'current_route') and commuter.current_route:
                    for i in range(len(commuter.current_route) - 1):
                        edge_key = f"{commuter.current_route[i]}_{commuter.current_route[i+1]}"
                        edge_usage[edge_key] = edge_usage.get(edge_key, 0) + 1
        
        # Add default usage for shortcuts that weren't used
        for shortcut in network_data['shortcuts']:
            edge_key = f"{shortcut['from_node']}_{shortcut['to_node']}"
            if edge_key not in edge_usage:
                edge_usage[edge_key] = 1  # Default usage for visualization
        
        network_data['edge_congestion'] = edge_usage
        
        return network_data



        

# ===== CONVENIENCE FUNCTIONS =====
def run_degree_comparison_study(degrees: List[int] = [3, 4, 5, 6, 7], **kwargs):
    """Quick function to run degree-constrained comparison study"""
    runner = NTEOResearchRunner()
    results = runner.run_single_topology_study('degree_constrained', degrees, **kwargs)
    
    return runner.run_single_topology_study('degree_constrained', degrees, **kwargs)

def run_small_world_study(rewiring_probs: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5], **kwargs):
    """Quick function to run small-world comparison study"""
    runner = NTEOResearchRunner()
    return runner.run_single_topology_study('small_world', rewiring_probs, **kwargs)

def run_scale_free_study(attachment_params: List[int] = [1, 2, 3], **kwargs):
    """Quick function to run scale-free comparison study"""
    runner = NTEOResearchRunner()
    return runner.run_single_topology_study('scale_free', attachment_params, **kwargs)

def run_full_nteo_comparison(**kwargs):
    """Run comprehensive NTEO comparison across all topology types"""
    runner = NTEOResearchRunner()
    
    topology_configs = {
        'grid': [4, 6, 8], 
        'degree_constrained': [3, 4, 5, 6],
        'small_world': [0.1, 0.2, 0.3],
        'scale_free': [1, 2, 3]
    }
    
    return runner.run_multi_topology_comparison(topology_configs, **kwargs)



# ===== MAIN EXECUTION WITH PATH FIX =====
if __name__ == "__main__":
    # Fix Python path issue (same as your working command)
    import sys
    sys.path.append('.')
    
    print("🎯 NTEO WCTR Statistical Analysis")
    print("=" * 50)
    print("Choose option:")
    print("1. Full WCTR Analysis (25 runs) - Journal Quality ~4-6 hours")
    print("2. Quick Test (5 runs) - ~1 hour") 
    print("3. Mini Test (2 runs) - ~15 minutes")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    # Import here after path is fixed
    from research.nteo_research_runner import NTEOResearchRunner
    
    if choice == "1":
        print("🚀 Running FULL WCTR Statistical Analysis...")
        runner = NTEOResearchRunner("wctr_full_results")
        
        topology_configs = {
            'grid': [4, 6, 8], 
            'degree_constrained': [3, 4, 5, 6, 7],
            'small_world': [0.0, 0.1, 0.2, 0.3, 0.5],
            'scale_free': [1, 2, 3, 4]
        }
        
        results = runner.run_multi_topology_comparison(
            topology_configs, num_runs=25, steps_per_run=144
        )
        
    elif choice == "2":
        print("🚀 Running Quick Test...")
        runner = NTEOResearchRunner("wctr_test_results")
        
        topology_configs = {
            'grid': [4, 6, 8], 
            'degree_constrained': [3, 4, 5],
            'small_world': [0.1, 0.2, 0.3],
            'scale_free': [1, 2, 3]
        }
        
        results = runner.run_multi_topology_comparison(
            topology_configs, num_runs=5, steps_per_run=72
        )
        
    else:
        print("🚀 Running Mini Test...")
        runner = NTEOResearchRunner("wctr_mini_results")
        
        topology_configs = {
            'grid': [4, 6], 
            'degree_constrained': [3, 4],
            'small_world': [0.1, 0.2],
            'scale_free': [1, 2]
        }
        
        results = runner.run_multi_topology_comparison(
            topology_configs, num_runs=2, steps_per_run=50
        )
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📁 Check results in: {runner.output_dir}")
    print(f"📊 All visualizations generated")
    print(f"📄 Results saved as JSON")
