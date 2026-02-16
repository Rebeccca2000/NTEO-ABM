#!/usr/bin/env python3
"""
Simple NTEO Metrics Aggregator
Focus on clean average results for the three key metrics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from typing import List

class SimpleMetricsAggregator:
    """
    Simplified aggregator focused on getting clean average results
    for Mode Choice Equity, Travel Time Equity, and System Efficiency
    """
    
    def __init__(self, results_base_directory: str):
        """
        Initialize aggregator to scan multiple result directories
        
        Args:
            results_base_directory: Base directory containing result folders
                                (e.g., if you have wctr_test_results, wctr_test_results_1, etc.)
        """
        self.base_dir = Path(results_base_directory).parent  # Parent directory containing all result folders
        self.base_name = Path(results_base_directory).name   # Base name pattern (e.g., "wctr_test_results")
        
        self.metrics = [
            'final_mode_choice_equity',
            'final_travel_time_equity', 
            'final_system_efficiency'
        ]
        self.metric_labels = [
            'Mode Choice Equity',
            'Travel Time Equity',
            'System Efficiency'
        ]
        
        # Find all matching directories
        self.result_dirs = self._find_all_result_directories()
        print(f"🔍 Found {len(self.result_dirs)} result directories:")
        for dir_path in self.result_dirs:
            print(f"   - {dir_path.name}")

    def _find_all_result_directories(self) -> List[Path]:
        """Find all directories matching the base pattern"""
        result_dirs = []
        
        # Add the base directory if it exists
        base_path = self.base_dir / self.base_name
        if base_path.exists() and base_path.is_dir():
            result_dirs.append(base_path)
        
        # Find numbered variations (e.g., wctr_test_results_1, wctr_test_results_2, etc.)
        for item in self.base_dir.iterdir():
            if (item.is_dir() and 
                item.name.startswith(f"{self.base_name}_") and 
                item.name[len(f"{self.base_name}_"):].isdigit()):
                result_dirs.append(item)
        
        # Sort by number for consistent ordering
        def sort_key(path):
            if path.name == self.base_name:
                return 0  # Base directory comes first
            else:
                return int(path.name.split('_')[-1])
        
        result_dirs.sort(key=sort_key)
        return result_dirs
        
    def get_topology_averages(self) -> pd.DataFrame:
        """
        Get simple topology averages for the three key metrics from ALL result directories
        
        Returns:
            DataFrame with topology averages and confidence intervals
        """
        print("🔍 Scanning ALL result directories for JSON files...")
        
        # Collect all data from all directories
        all_data = []
        total_files = 0
        
        for result_dir in self.result_dirs:
            json_files = list(result_dir.glob("*.json"))
            total_files += len(json_files)
            
            print(f"📁 Processing {result_dir.name}: {len(json_files)} JSON files")
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract runs from different file structures
                    runs = self._extract_runs_from_file(data, file_path)
                    all_data.extend(runs)
                    
                except Exception as e:
                    print(f"⚠️ Skipped {file_path.name}: {e}")
        
        print(f"✅ Processed {total_files} JSON files from {len(self.result_dirs)} directories")
        print(f"✅ Extracted {len(all_data)} total simulation runs")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        if df.empty:
            print("❌ No data found!")
            return pd.DataFrame()
        
        # Show run count breakdown by topology
        print(f"\n📊 Run count by topology:")
        for topology in df['topology'].unique():
            count = len(df[df['topology'] == topology])
            print(f"   - {topology}: {count} runs")
        
        # Calculate topology averages (same as before)
        results = []
        
        for topology in df['topology'].unique():
            topology_data = df[df['topology'] == topology]
            
            row = {'topology': topology}
            
            for metric, label in zip(self.metrics, self.metric_labels):
                if metric in topology_data.columns:
                    values = topology_data[metric].dropna()
                    
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        sem_val = stats.sem(values) if len(values) > 1 else 0
                        
                        # 95% confidence interval
                        if len(values) > 1:
                            ci_lower, ci_upper = stats.t.interval(
                                0.95, len(values)-1, 
                                loc=mean_val, scale=sem_val
                            )
                        else:
                            ci_lower = ci_upper = mean_val
                        
                        row.update({
                            f'{metric}_mean': mean_val,
                            f'{metric}_std': std_val,
                            f'{metric}_sem': sem_val,
                            f'{metric}_ci_lower': ci_lower,
                            f'{metric}_ci_upper': ci_upper,
                            f'{metric}_count': len(values)
                        })
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _extract_runs_from_file(self, data: dict, file_path: Path) -> list:
        """Extract individual simulation runs from a JSON file"""
        runs = []
        
        # Handle multi-topology comparison files
        if 'topology_results' in data:
            for topology_type, topology_data in data['topology_results'].items():
                if 'results' in topology_data:
                    for param_data in topology_data['results'].values():
                        for run in param_data.get('runs', []):
                            if self._has_required_metrics(run):
                                run_data = {'topology': topology_type}
                                for metric in self.metrics:
                                    run_data[metric] = run.get(metric, None)
                                runs.append(run_data)
        
        # Handle individual topology study files
        elif 'results' in data and 'topology_type' in data:
            topology_type = data['topology_type']
            for param_data in data['results'].values():
                for run in param_data.get('runs', []):
                    if self._has_required_metrics(run):
                        run_data = {'topology': topology_type}
                        for metric in self.metrics:
                            run_data[metric] = run.get(metric, None)
                        runs.append(run_data)
        
        # Try to infer topology from filename
        elif 'results' in data:
            topology_type = self._infer_topology_from_filename(file_path)
            if topology_type:
                for param_data in data['results'].values():
                    for run in param_data.get('runs', []):
                        if self._has_required_metrics(run):
                            run_data = {'topology': topology_type}
                            for metric in self.metrics:
                                run_data[metric] = run.get(metric, None)
                            runs.append(run_data)
        
        return runs
    
    def _has_required_metrics(self, run: dict) -> bool:
        """Check if run has the required metrics"""
        return any(metric in run for metric in self.metrics)
    
    def _infer_topology_from_filename(self, file_path: Path) -> str:
        """Infer topology type from filename"""
        filename = file_path.name.lower()
        if 'degree_constrained' in filename:
            return 'degree_constrained'
        elif 'small_world' in filename:
            return 'small_world'
        elif 'scale_free' in filename:
            return 'scale_free'
        return None
    
    def create_summary_table(self, df: pd.DataFrame) -> str:
        """Create a clean summary table"""
        if df.empty:
            return "No data available"
        
        summary_lines = []
        summary_lines.append("NTEO TOPOLOGY PERFORMANCE SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        for _, row in df.iterrows():
            topology = row['topology'].replace('_', ' ').title()
            summary_lines.append(f"{topology}:")
            
            for metric, label in zip(self.metrics, self.metric_labels):
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                count_key = f'{metric}_count'
                ci_lower_key = f'{metric}_ci_lower'
                ci_upper_key = f'{metric}_ci_upper'
                
                if mean_key in row:
                    mean_val = row[mean_key]
                    std_val = row[std_key]
                    count_val = row[count_key]
                    ci_lower = row[ci_lower_key]
                    ci_upper = row[ci_upper_key]
                    
                    summary_lines.append(f"  {label}:")
                    summary_lines.append(f"    Mean: {mean_val:.3f}")
                    summary_lines.append(f"    Std:  {std_val:.3f}")
                    summary_lines.append(f"    95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                    summary_lines.append(f"    Sample size: {count_val} runs")
            
            summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def create_comparison_plot(self, df: pd.DataFrame, output_path: str = None) -> str:
        """Create parameter sensitivity line plots like the example image"""
        
        # First get parameter-level data instead of just topology averages
        print("🔍 Extracting parameter-level data for sensitivity analysis...")
        parameter_data = self._get_parameter_level_data()
        
        if not parameter_data:
            print("❌ No parameter data found")
            return None
        
        # Set up the plot - 3 subplots for 3 metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Network Topology Parameter Sensitivity Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Colors matching your image
        colors = {
            'degree_constrained': '#2E86AB',  # Blue
            'small_world': '#A23B72',        # Purple  
            'scale_free': '#F18F01'          # Orange
        }
        
        labels = {
            'degree_constrained': 'Degree Constrained',
            'small_world': 'Small World',
            'scale_free': 'Scale Free'
        }
        
        for i, (metric, metric_label) in enumerate(zip(self.metrics, self.metric_labels)):
            ax = axes[i]
            
            # Plot each topology
            for topology in ['degree_constrained', 'small_world', 'scale_free']:
                if topology in parameter_data:
                    topo_data = parameter_data[topology]
                    
                    if metric in topo_data and topo_data[metric]:
                        params = list(topo_data[metric].keys())
                        values = list(topo_data[metric].values())
                        # Scale small world parameters by 10
                        if topology == 'small_world':
                            params = [float(p) * 10 for p in params]
                        else:
                            params = [float(p) for p in params]
                        # Sort by parameter value
                        sorted_data = sorted(zip(params, values))
                        params, values = zip(*sorted_data) if sorted_data else ([], [])
                        
                        if params and values:
                            ax.plot(params, values, 
                                marker='o', linewidth=2, markersize=6,
                                label=labels[topology], 
                                color=colors[topology])
            
            ax.set_xlabel('Parameter Value (Scaled for Comparison)')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{metric_label} Parameter Sensitivity')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = f"parameter_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Parameter sensitivity plots saved: {output_path}")
        return output_path

    def _get_parameter_level_data(self) -> dict:
        """Extract parameter-level averages from ALL directories for each topology and metric"""
        print("🔄 Processing parameter-level data from all directories...")
        
        # Structure: {topology: {metric: {parameter: mean_value}}}
        parameter_data = {}
        total_files = 0
        
        for result_dir in self.result_dirs:
            json_files = list(result_dir.glob("*.json"))
            total_files += len(json_files)
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract parameter data from different file structures
                    param_runs = self._extract_parameter_runs_from_file(data, file_path)
                    
                    for topology, params in param_runs.items():
                        if topology not in parameter_data:
                            parameter_data[topology] = {}
                        
                        for param_value, runs in params.items():
                            for metric in self.metrics:
                                if metric not in parameter_data[topology]:
                                    parameter_data[topology][metric] = {}
                                
                                # Calculate mean for this topology-parameter-metric combination
                                values = [run[metric] for run in runs if metric in run and run[metric] is not None]
                                
                                if values:
                                    if param_value not in parameter_data[topology][metric]:
                                        parameter_data[topology][metric][param_value] = []
                                    
                                    parameter_data[topology][metric][param_value].extend(values)
                
                except Exception as e:
                    print(f"⚠️ Skipped {file_path.name}: {e}")
        
        print(f"✅ Processed {total_files} files for parameter analysis")
        
        # Convert lists to means
        for topology in parameter_data:
            for metric in parameter_data[topology]:
                for param in parameter_data[topology][metric]:
                    values = parameter_data[topology][metric][param]
                    parameter_data[topology][metric][param] = np.mean(values)
                    print(f"   {topology} param {param} {metric}: {len(values)} values -> mean {np.mean(values):.3f}")
        
        return parameter_data

    def _extract_parameter_runs_from_file(self, data: dict, file_path: Path) -> dict:
        """Extract runs grouped by topology and parameter value"""
        param_runs = {}
        
        # Handle multi-topology comparison files
        if 'topology_results' in data:
            for topology_type, topology_data in data['topology_results'].items():
                if 'results' in topology_data:
                    if topology_type not in param_runs:
                        param_runs[topology_type] = {}
                    
                    for param_key, param_data in topology_data['results'].items():
                        parameter_value = param_data.get('parameter_value', param_key)
                        
                        runs = param_data.get('runs', [])
                        valid_runs = [run for run in runs if self._has_required_metrics(run)]
                        
                        if valid_runs:
                            param_runs[topology_type][parameter_value] = valid_runs
        
        # Handle individual topology study files  
        elif 'results' in data and 'topology_type' in data:
            topology_type = data['topology_type']
            param_runs[topology_type] = {}
            
            for param_key, param_data in data['results'].items():
                parameter_value = param_data.get('parameter_value', param_key)
                
                runs = param_data.get('runs', [])
                valid_runs = [run for run in runs if self._has_required_metrics(run)]
                
                if valid_runs:
                    param_runs[topology_type][parameter_value] = valid_runs
        
        return param_runs
        
    def run_simple_analysis(self, output_dir: str = "simple_results") -> dict:
        """Run complete simple analysis"""
        
        print("\n" + "="*60)
        print("SIMPLE NTEO METRICS AGGREGATION")
        print("="*60)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get topology averages
        df = self.get_topology_averages()
        
        if df.empty:
            print("❌ No data found for analysis")
            return {}
        
        # Create summary table
        summary_text = self.create_summary_table(df)
        print("\n" + summary_text)
        
        # Save summary to file
        summary_file = output_path / f"topology_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
            f.write(f"\nGenerated: {datetime.now().isoformat()}")
        
        # Create comparison plot
        plot_file = output_path / f"topology_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.create_comparison_plot(df, str(plot_file))
        
        # Save data to CSV
        csv_file = output_path / f"topology_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n✅ Analysis complete!")
        print(f"📁 Results saved in: {output_dir}/")
        print(f"📄 Summary: {summary_file}")
        print(f"📊 Plot: {plot_file}")
        print(f"📈 Data: {csv_file}")
        
        return {
            'data': df,
            'summary': summary_text,
            'files': {
                'summary': str(summary_file),
                'plot': str(plot_file),
                'data': str(csv_file)
            }
        }

def run_simple_aggregation(results_directory: str, output_directory: str = "simple_results"):
    """
    Run simple aggregation focused on clean average results
    """
    aggregator = SimpleMetricsAggregator(results_directory)
    return aggregator.run_simple_analysis(output_directory)

# Direct usage example
if __name__ == "__main__":
    # Use this to run directly
    results = run_simple_aggregation("wctr_test_results", "clean_results")