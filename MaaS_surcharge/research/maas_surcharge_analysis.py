# maas_surcharge_analysis.py
"""
MaaS Surcharge Impact Analysis
Tests different surcharge values on base_sydney network and measures system metrics.
Focus: S_base parameter impact on mode_choice_equity, travel_time_equity, and system_efficiency
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so sibling packages (like `testing`) can be imported
# when the script is executed directly (python3 research/maas_surcharge_analysis.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import copy

# Import your simulation system
from testing.abm_initialization import create_nteo_model, MobilityModelNTEO
import config.database_updated as db

class MaaSSurchargeAnalyzer:
    """Analyzer for MaaS surcharge impact on system metrics"""
    
    def __init__(self, output_dir: str = "maas_surcharge_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🔬 MaaS Surcharge Analyzer initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def run_surcharge_analysis(self, 
                             s_base_values: List[float],
                             num_runs: int = 3,
                             steps_per_run: int = 144,
                             num_commuters: int = 200) -> Dict[str, Any]:
        """
        Run surcharge analysis with different S_base values
        
        Args:
            s_base_values: List of S_base values to test (main surcharge parameter)
            num_runs: Number of simulation runs per S_base value
            steps_per_run: Steps per simulation (144 = 24 hours)
            num_commuters: Number of commuters in simulation
        """
        
        print(f"\n🚀 Starting MaaS Surcharge Analysis")
        print(f"📊 S_base values: {s_base_values}")
        print(f"🔄 Runs per value: {num_runs}")
        print(f"👥 Commuters: {num_commuters}")
        
        results = {
            'analysis_type': 'maas_surcharge_impact',
            'network_type': 'base_sydney',
            'timestamp': self.timestamp,
            'parameters': {
                's_base_values': s_base_values,
                'num_runs': num_runs,
                'steps_per_run': steps_per_run,
                'num_commuters': num_commuters
            },
            'raw_data': {},
            'summary_stats': {}
        }
        
        total_runs = len(s_base_values) * num_runs
        current_run = 0
        
        # Test each S_base value
        for s_base_idx, s_base in enumerate(s_base_values):
            print(f"\n🎯 Testing S_base = {s_base} ({s_base_idx+1}/{len(s_base_values)})")
            
            s_base_results = {
                'parameter_value': s_base,
                'runs': [],
                'metrics': {}
            }
            
            # Multiple runs for statistical significance
            for run_idx in range(num_runs):
                current_run += 1
                progress = (current_run / total_runs) * 100
                
                print(f"   🔄 Run {run_idx + 1}/{num_runs} ({progress:.1f}% complete)")
                
                model = None
                try:
                    # Create modified surcharge coefficients
                    modified_coefficients = self._create_modified_surcharge_config(s_base)
                    
                    # Create model with base_sydney network and modified surcharge
                    model = create_nteo_model(
                        topology_type='base_sydney',
                        variation_parameter=6,  # Default connectivity for base_sydney
                        num_commuters=num_commuters,
                        dynamic_maas_surcharge_base_coefficients=modified_coefficients
                    )
                    
                    if model:
                        # Run simulation
                        run_data = self._run_simulation(model, steps_per_run, f"s_base_{s_base}_run_{run_idx}")
                        s_base_results['runs'].append(run_data)
                        
                        print(f"     ✅ Completed - Mode Equity: {run_data['final_mode_choice_equity']:.3f}, " +
                              f"Time Equity: {run_data['final_travel_time_equity']:.3f}, " +
                              f"Efficiency: {run_data['final_system_efficiency']:.1f}")
                    else:
                        print("     ❌ Model creation failed")
                        
                except Exception as e:
                    print(f"     ❌ Run failed: {str(e)}")
                    s_base_results['runs'].append({'error': str(e)})
                finally:
                    if model:
                        try:
                            model.cleanup()
                            del model
                        except:
                            pass
            
            # Calculate statistics for this S_base value
            if s_base_results['runs']:
                s_base_results['metrics'] = self._calculate_run_statistics(s_base_results['runs'])
                
            results['raw_data'][str(s_base)] = s_base_results
        
        # Calculate overall summary statistics
        results['summary_stats'] = self._calculate_summary_statistics(results['raw_data'])
        
        # Generate visualizations
        self._generate_surcharge_plots(results)
        
        # Save results
        self._save_results(results, f"maas_surcharge_analysis_{self.timestamp}")
        
        print(f"\n✅ MaaS Surcharge Analysis completed!")
        print(f"📊 Results saved to {self.output_dir}")
        
        return results
    
    def _create_modified_surcharge_config(self, s_base: float) -> Dict[str, float]:
        """Create modified surcharge configuration with new S_base value"""
        
        # Get original coefficients from your database config
        original_coefficients = copy.deepcopy(db.DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS)
        
        # Modify only S_base, keep alpha and delta the same
        modified_coefficients = original_coefficients.copy()
        modified_coefficients['S_base'] = s_base
        
        print(f"     🔧 Modified surcharge config: S_base={s_base}, "
              f"alpha={modified_coefficients.get('alpha', 'N/A')}, "
              f"delta={modified_coefficients.get('delta', 'N/A')}")
        
        return modified_coefficients
    
    def _run_simulation(self, model: MobilityModelNTEO, num_steps: int, run_id: str) -> Dict[str, Any]:
        """Run simulation and collect metrics"""
        
        start_time = time.time()
        
        # Run simulation steps
        for step in range(num_steps):
            model.step()
        
        execution_time = time.time() - start_time
        
        # Collect final metrics (your 3 defined metrics)
        try:
            final_metrics = {
                'run_id': run_id,
                'execution_time': execution_time,
                'final_mode_choice_equity': model.calculate_mode_choice_equity(),
                'final_travel_time_equity': model.calculate_travel_time_equity(), 
                'final_system_efficiency': model.calculate_system_efficiency(),
                'steps_completed': num_steps
            }
            
            # Collect mode share data
            mode_share_data = self._collect_mode_share_data(model)
            final_metrics['mode_share'] = mode_share_data
            
            # Calculate mode share percentages
            total_trips = sum(mode_share_data.values())
            mode_percentages = {}
            if total_trips > 0:
                for mode, count in mode_share_data.items():
                    mode_percentages[mode] = (count / total_trips) * 100
            final_metrics['mode_percentages'] = mode_percentages
            
            # Additional MaaS-specific metrics if available
            try:
                if hasattr(model, 'maas_agent'):
                    maas_stats = self._collect_maas_statistics(model)
                    final_metrics.update(maas_stats)
            except Exception as e:
                print(f"     ⚠️ Could not collect MaaS stats: {e}")
                
        except Exception as e:
            print(f"     ❌ Error collecting metrics: {e}")
            final_metrics = {
                'run_id': run_id,
                'execution_time': execution_time,
                'error': str(e)
            }
        
        return final_metrics
    
    def _collect_maas_statistics(self, model: MobilityModelNTEO) -> Dict[str, float]:
        """Collect MaaS-specific statistics"""
        
        maas_stats = {}
        
        try:
            if hasattr(model, 'maas_agent') and model.maas_agent:
                # MaaS usage statistics
                total_requests = 0
                maas_requests = 0
                
                for commuter in model.commuter_agents:
                    for request_id, request_data in commuter.requests.items():
                        total_requests += 1
                        if request_data.get('chosen_mode') == 'maas':
                            maas_requests += 1
                
                maas_stats['maas_adoption_rate'] = maas_requests / total_requests if total_requests > 0 else 0
                maas_stats['total_transport_requests'] = total_requests
                maas_stats['maas_requests'] = maas_requests
                
        except Exception as e:
            print(f"     ⚠️ Error collecting MaaS stats: {e}")
        
        return maas_stats
    
    def _collect_mode_share_data(self, model: MobilityModelNTEO) -> Dict[str, int]:
        """Collect mode share data from commuter requests"""
        
        mode_counts = {
            'walk': 0,
            'bike': 0, 
            'public': 0,
            'car': 0,
            'maas': 0
        }
        
        try:
            for commuter in model.commuter_agents:
                for request_id, request_data in commuter.requests.items():
                    if 'selected_route' in request_data and request_data['selected_route']:
                        mode = request_data['selected_route'].get('mode', '')
                        
                        # Map mode names to standard categories
                        if mode == 'walk_route' or mode.startswith('walk'):
                            mode_counts['walk'] += 1
                        elif 'bike' in mode:
                            mode_counts['bike'] += 1
                        elif mode == 'public_route' or 'public' in mode:
                            mode_counts['public'] += 1
                        elif 'car' in mode:
                            mode_counts['car'] += 1
                        elif mode == 'MaaS_Bundle' or 'maas' in mode:
                            mode_counts['maas'] += 1
                            
        except Exception as e:
            print(f"     ⚠️ Error collecting mode share: {e}")
        
        return mode_counts
    
    def _calculate_run_statistics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for multiple runs"""
        
        # Filter out error runs
        valid_runs = [run for run in runs if 'error' not in run]
        
        if not valid_runs:
            return {'error': 'No valid runs', 'run_count': len(runs)}
        
        metrics = ['final_mode_choice_equity', 'final_travel_time_equity', 'final_system_efficiency']
        stats = {'valid_runs': len(valid_runs)}
        
        # Calculate statistics for main metrics
        for metric in metrics:
            values = [run[metric] for run in valid_runs if metric in run]
            if values:
                stats[f'{metric}_mean'] = np.mean(values)
                stats[f'{metric}_std'] = np.std(values)
                stats[f'{metric}_min'] = np.min(values)
                stats[f'{metric}_max'] = np.max(values)
        
        # Calculate mode share statistics
        modes = ['walk', 'bike', 'public', 'car', 'maas']
        mode_share_stats = {}
        
        for mode in modes:
            percentages = []
            for run in valid_runs:
                if 'mode_percentages' in run and mode in run['mode_percentages']:
                    percentages.append(run['mode_percentages'][mode])
            
            if percentages:
                mode_share_stats[f'{mode}_percentage_mean'] = np.mean(percentages)
                mode_share_stats[f'{mode}_percentage_std'] = np.std(percentages)
        
        stats['mode_share_stats'] = mode_share_stats
        
        return stats
    
    def _calculate_summary_statistics(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary statistics across all S_base values"""
        
        summary = {
            'parameters_tested': len(raw_data),
            'optimal_values': {},
            'sensitivity_analysis': {}
        }
        
        metrics = ['final_mode_choice_equity', 'final_travel_time_equity', 'final_system_efficiency']
        
        for metric in metrics:
            metric_data = []
            s_base_values = []
            
            for s_base_str, s_base_data in raw_data.items():
                if 'metrics' in s_base_data and f'{metric}_mean' in s_base_data['metrics']:
                    s_base_values.append(float(s_base_str))
                    metric_data.append(s_base_data['metrics'][f'{metric}_mean'])
            
            if metric_data:
                # Find optimal S_base for this metric
                if metric == 'final_system_efficiency':
                    # Higher efficiency is better
                    optimal_idx = np.argmax(metric_data)
                else:
                    # Lower equity values are better (closer to 0)
                    optimal_idx = np.argmin(metric_data)
                
                summary['optimal_values'][metric] = {
                    'optimal_s_base': s_base_values[optimal_idx],
                    'optimal_value': metric_data[optimal_idx]
                }
                
                # Sensitivity analysis
                summary['sensitivity_analysis'][metric] = {
                    'range': max(metric_data) - min(metric_data),
                    'coefficient_of_variation': np.std(metric_data) / np.mean(metric_data) if np.mean(metric_data) != 0 else 0
                }
        
        return summary
    
    def _generate_surcharge_plots(self, results: Dict[str, Any]):
        """Generate visualization plots"""
        
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MaaS Surcharge Impact Analysis - Base Sydney Network', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        s_base_values = []
        mode_equity_means = []
        mode_equity_stds = []
        time_equity_means = []
        time_equity_stds = []
        efficiency_means = []
        efficiency_stds = []
        
        for s_base_str, s_base_data in results['raw_data'].items():
            if 'metrics' in s_base_data:
                metrics = s_base_data['metrics']
                s_base_values.append(float(s_base_str))
                
                mode_equity_means.append(metrics.get('final_mode_choice_equity_mean', 0))
                mode_equity_stds.append(metrics.get('final_mode_choice_equity_std', 0))
                
                time_equity_means.append(metrics.get('final_travel_time_equity_mean', 0))
                time_equity_stds.append(metrics.get('final_travel_time_equity_std', 0))
                
                efficiency_means.append(metrics.get('final_system_efficiency_mean', 0))
                efficiency_stds.append(metrics.get('final_system_efficiency_std', 0))
        
        # Sort by S_base values for proper plotting
        if s_base_values:
            sorted_data = sorted(zip(s_base_values, mode_equity_means, mode_equity_stds,
                                   time_equity_means, time_equity_stds, 
                                   efficiency_means, efficiency_stds))
            s_base_values, mode_equity_means, mode_equity_stds, time_equity_means, time_equity_stds, efficiency_means, efficiency_stds = zip(*sorted_data)
        
        # Plot 1: Mode Choice Equity
        axes[0,0].errorbar(s_base_values, mode_equity_means, yerr=mode_equity_stds, 
                          marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        axes[0,0].set_title('Mode Choice Equity vs MaaS Surcharge', fontweight='bold')
        axes[0,0].set_xlabel('S_base (MaaS Surcharge Parameter)')
        axes[0,0].set_ylabel('Mode Choice Equity (Lower = Better)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Travel Time Equity  
        axes[0,1].errorbar(s_base_values, time_equity_means, yerr=time_equity_stds,
                          marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='orange')
        axes[0,1].set_title('Travel Time Equity vs MaaS Surcharge', fontweight='bold')
        axes[0,1].set_xlabel('S_base (MaaS Surcharge Parameter)')
        axes[0,1].set_ylabel('Travel Time Equity (Lower = Better)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: System Efficiency
        axes[1,0].errorbar(s_base_values, efficiency_means, yerr=efficiency_stds,
                          marker='^', capsize=5, capthick=2, linewidth=2, markersize=8, color='green')
        axes[1,0].set_title('System Efficiency vs MaaS Surcharge', fontweight='bold') 
        axes[1,0].set_xlabel('S_base (MaaS Surcharge Parameter)')
        axes[1,0].set_ylabel('System Efficiency (Higher = Better)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Summary comparison
        axes[1,1].plot(s_base_values, mode_equity_means, 'o-', label='Mode Equity', linewidth=2)
        axes[1,1].plot(s_base_values, time_equity_means, 's-', label='Time Equity', linewidth=2)
        # Normalize efficiency for comparison (scale to 0-1)
        if efficiency_means:
            norm_efficiency = [(x - min(efficiency_means)) / (max(efficiency_means) - min(efficiency_means)) 
                              if max(efficiency_means) != min(efficiency_means) else 0.5 
                              for x in efficiency_means]
            axes[1,1].plot(s_base_values, norm_efficiency, '^-', label='Efficiency (normalized)', linewidth=2)
        axes[1,1].set_title('All Metrics Comparison', fontweight='bold')
        axes[1,1].set_xlabel('S_base (MaaS Surcharge Parameter)')
        axes[1,1].set_ylabel('Metric Values')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"maas_surcharge_analysis_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Analysis plots saved to {plot_path}")
        
        # Create summary table
        self._create_summary_table(results)
    
    def _create_summary_table(self, results: Dict[str, Any]):
        """Create and save summary table including mode share data"""
        
        table_data = []
        
        for s_base_str, s_base_data in results['raw_data'].items():
            if 'metrics' in s_base_data:
                metrics = s_base_data['metrics']
                row = {
                    'S_base': float(s_base_str),
                    'Mode_Equity_Mean': metrics.get('final_mode_choice_equity_mean', 0),
                    'Mode_Equity_Std': metrics.get('final_mode_choice_equity_std', 0),
                    'Time_Equity_Mean': metrics.get('final_travel_time_equity_mean', 0),
                    'Time_Equity_Std': metrics.get('final_travel_time_equity_std', 0),
                    'Efficiency_Mean': metrics.get('final_system_efficiency_mean', 0),
                    'Efficiency_Std': metrics.get('final_system_efficiency_std', 0),
                    'Valid_Runs': metrics.get('valid_runs', 0)
                }
                
                # Add mode share data
                if 'mode_share_stats' in metrics:
                    mode_stats = metrics['mode_share_stats']
                    for mode in ['walk', 'bike', 'public', 'car', 'maas']:
                        row[f'{mode.title()}_Share_%'] = mode_stats.get(f'{mode}_percentage_mean', 0)
                        row[f'{mode.title()}_Share_Std'] = mode_stats.get(f'{mode}_percentage_std', 0)
                
                table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            df = df.sort_values('S_base')
            
            # Save as CSV
            csv_path = self.output_dir / f"surcharge_summary_table_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False, float_format='%.4f')
            
            print(f"📋 Summary table saved to {csv_path}")
            print("\nSummary Preview:")
            print(df.round(4).to_string(index=False))
            
            # Create separate mode share summary
            mode_share_columns = [col for col in df.columns if '_Share_%' in col and '_Std' not in col]
            if mode_share_columns:
                mode_df = df[['S_base'] + mode_share_columns].round(2)
                print(f"\nMode Share Summary:")
                print(mode_df.to_string(index=False))
    
    def _save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file"""
        
        output_path = self.output_dir / f"{filename}.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {output_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save results to {output_path}: {e}")


def main():
    """Main function to run MaaS surcharge analysis"""
    
    # Initialize analyzer
    analyzer = MaaSSurchargeAnalyzer()
    
    # Define S_base values to test (focus on the most influential parameter)
    # Start with small values and increase to see impact
    s_base_values = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    print("🎯 MaaS Surcharge Analysis Configuration:")
    print(f"   Network: base_sydney (fixed)")
    print(f"   Parameter: S_base (most influential)")
    print(f"   Values to test: {s_base_values}")
    print(f"   Metrics: mode_choice_equity, travel_time_equity, system_efficiency")
    print(f"   Mode Share: walk, bike, public, car, maas (with pie charts)")
    print(f"   Output: 3 plots + summary table + mode share analysis")
    
    # Run analysis
    results = analyzer.run_surcharge_analysis(
        s_base_values=s_base_values,
        num_runs=3,  # Adjust based on computational resources
        steps_per_run=144,  # 24 hours simulation
        num_commuters=200  # From your config
    )
    
    # Print optimal values
    if 'optimal_values' in results['summary_stats']:
        print("\n🏆 OPTIMAL SURCHARGE VALUES:")
        for metric, optimal_data in results['summary_stats']['optimal_values'].items():
            print(f"   {metric}: S_base = {optimal_data['optimal_s_base']:.3f} "
                  f"(value = {optimal_data['optimal_value']:.3f})")
    
    return results


if __name__ == "__main__":
    results = main()