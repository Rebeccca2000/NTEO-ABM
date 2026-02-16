import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class NTEOResultsAggregator:
    """
    Statistical aggregator for NTEO simulation results
    
    CRITICAL: This addresses statistical inadequacy in your current 3-run approach
    """
    
    def __init__(self, results_directory: str, output_directory: str = None):
        self.results_dir = Path(results_directory)
        self.output_dir = Path(output_directory) if output_directory else Path("aggregated_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistical significance threshold
        self.alpha = 0.05
        self.min_runs_for_stats = 10  # Minimum for reliable statistics
        
        print(f"🔍 Scanning {self.results_dir} for JSON files...")
        self.result_files = list(self.results_dir.glob("*.json"))
        
        # Identify different file types based on your naming convention
        self.file_types = {
            'individual_studies': [],
            'multi_topology_comparisons': [],
            'other_files': []
        }
        
        for file_path in self.result_files:
            if '_study_' in file_path.name:
                self.file_types['individual_studies'].append(file_path)
            elif 'multi_topology_comparison_' in file_path.name:
                self.file_types['multi_topology_comparisons'].append(file_path)
            else:
                self.file_types['other_files'].append(file_path)
        
        print(f"📁 Found {len(self.result_files)} JSON files:")
        print(f"   - Individual studies: {len(self.file_types['individual_studies'])}")
        print(f"   - Multi-topology comparisons: {len(self.file_types['multi_topology_comparisons'])}")
        print(f"   - Other files: {len(self.file_types['other_files'])}")
        
        # List individual study files by topology type
        topology_counts = {}
        for file_path in self.file_types['individual_studies']:
            # Extract topology type from filename (e.g., "degree_constrained_study_...")
            parts = file_path.name.split('_study_')
            if len(parts) >= 2:
                topology_type = parts[0]
                topology_counts[topology_type] = topology_counts.get(topology_type, 0) + 1
        
        if topology_counts:
            print(f"   Topology breakdown:")
            for topology, count in topology_counts.items():
                print(f"     - {topology}: {count} files")
        
    def aggregate_all_results(self) -> Dict[str, Any]:
        """
        Main aggregation pipeline - combines all simulation runs
        
        Returns:
            Comprehensive aggregated analysis with statistical validation
        """
        print("\n" + "="*60)
        print("NTEO STATISTICAL AGGREGATION PIPELINE")
        print("="*60)
        
        # Step 1: Load and validate all result files
        raw_data = self._load_all_results()
        
        # Step 2: CRITICAL - Check statistical adequacy
        adequacy_report = self._validate_statistical_adequacy(raw_data)
        
        # Step 3: Aggregate runs across files
        aggregated_data = self._aggregate_simulation_runs(raw_data)
        
        # Step 4: Calculate robust statistics with confidence intervals
        statistical_analysis = self._calculate_robust_statistics(aggregated_data)
        
        # Step 5: Perform cross-topology ANOVA and effect size analysis
        comparative_analysis = self._perform_statistical_tests(aggregated_data)
        
        # Step 6: Generate publication-ready visualizations
        visualization_paths = self._create_publication_plots(aggregated_data, statistical_analysis)
        
        # Compile final report
        final_report = {
            'metadata': {
                'aggregation_timestamp': datetime.now().isoformat(),
                'source_files': [str(f.name) for f in self.result_files],
                'total_runs_found': sum(len(data.get('runs', [])) for data in raw_data.values()),
                'statistical_adequacy': adequacy_report
            },
            'aggregated_data': aggregated_data,
            'statistical_analysis': statistical_analysis,
            'comparative_analysis': comparative_analysis,
            'visualization_paths': visualization_paths
        }
        
        # Save comprehensive report
        self._save_final_report(final_report)
        
    def list_detected_files(self) -> Dict[str, List[str]]:
        """
        List all detected files with their exact names for verification
        
        Returns:
            Dictionary with file categories and their exact filenames
        """
        detected_files = {
            'individual_studies': {},
            'multi_topology_comparisons': [],
            'other_files': []
        }
        
        # Group individual studies by topology type
        for file_path in self.file_types['individual_studies']:
            filename = file_path.name
            # Extract topology type from filename pattern
            topology_type = filename.split('_study_')[0]
            
            if topology_type not in detected_files['individual_studies']:
                detected_files['individual_studies'][topology_type] = []
            
            detected_files['individual_studies'][topology_type].append(filename)
        
        # Multi-topology comparison files
        for file_path in self.file_types['multi_topology_comparisons']:
            detected_files['multi_topology_comparisons'].append(file_path.name)
        
        # Other files
        for file_path in self.file_types['other_files']:
            detected_files['other_files'].append(file_path.name)
        
        # Print detailed breakdown
        print("\n" + "="*60)
        print("DETAILED FILE DETECTION REPORT")
        print("="*60)
        
        print(f"\n📊 INDIVIDUAL TOPOLOGY STUDIES:")
        if detected_files['individual_studies']:
            for topology, files in detected_files['individual_studies'].items():
                print(f"   {topology}:")
                for file in sorted(files):
                    print(f"     - {file}")
        else:
            print("   ⚠️ No individual study files found")
        
        print(f"\n🔄 MULTI-TOPOLOGY COMPARISON FILES:")
        if detected_files['multi_topology_comparisons']:
            for file in sorted(detected_files['multi_topology_comparisons']):
                print(f"   - {file}")
        else:
            print("   ⚠️ No multi-topology comparison files found")
        
        print(f"\n📁 OTHER JSON FILES:")
        if detected_files['other_files']:
            for file in sorted(detected_files['other_files']):
                print(f"   - {file}")
        else:
            print("   ℹ️ No other JSON files found")
        
        print("\n" + "="*60)
        
        return detected_files
    
    def aggregate_all_results(self) -> Dict[str, Any]:
        """
        Main aggregation pipeline - combines all simulation runs
        
        Returns:
            Comprehensive aggregated analysis with statistical validation
        """
        print("\n" + "="*60)
        print("NTEO STATISTICAL AGGREGATION PIPELINE")
        print("="*60)
        
        # Step 0: Show detailed file detection
        detected_files = self.list_detected_files()
        
        # Step 1: Load and validate all result files
        raw_data = self._load_all_results()
        
        # Step 2: CRITICAL - Check statistical adequacy
        adequacy_report = self._validate_statistical_adequacy(raw_data)
        
        # Step 3: Aggregate runs across files
        aggregated_data = self._aggregate_simulation_runs(raw_data)
        
        # Step 4: Calculate robust statistics with confidence intervals
        statistical_analysis = self._calculate_robust_statistics(aggregated_data)
        
        # Step 5: Perform cross-topology ANOVA and effect size analysis
        comparative_analysis = self._perform_statistical_tests(aggregated_data)
        
        # Step 6: Generate publication-ready visualizations
        visualization_paths = self._create_publication_plots(aggregated_data, statistical_analysis)
        
        # Compile final report
        final_report = {
            'metadata': {
                'aggregation_timestamp': datetime.now().isoformat(),
                'source_files': [str(f.name) for f in self.result_files],
                'detected_files': detected_files,
                'total_runs_found': sum(len(data.get('runs', [])) for data in raw_data.values()),
                'statistical_adequacy': adequacy_report
            },
            'aggregated_data': aggregated_data,
            'statistical_analysis': statistical_analysis,
            'comparative_analysis': comparative_analysis,
            'visualization_paths': visualization_paths
        }
        
        # Save comprehensive report
        self._save_final_report(final_report)
        
        return final_report
    
    def _load_all_results(self) -> Dict[str, Any]:
        """Load and parse all JSON result files with correct structure handling"""
        raw_data = {}
        
        # Process individual topology study files
        for file_path in self.file_types['individual_studies']:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract topology type from filename
                topology_type = file_path.name.split('_study_')[0]
                file_key = f"{topology_type}_{file_path.stem.split('_')[-1]}"  # Include timestamp
                
                # Ensure the data structure is consistent
                if 'topology_type' not in data:
                    data['topology_type'] = topology_type
                
                raw_data[file_key] = data
                runs_count = self._count_total_runs(data)
                print(f"✅ Loaded {file_key}: {runs_count} runs")
                
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")
        
        # Process multi-topology comparison files
        for file_path in self.file_types['multi_topology_comparisons']:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                file_key = file_path.stem
                raw_data[file_key] = data
                
                runs_count = self._count_total_runs(data)
                print(f"✅ Loaded {file_key}: {runs_count} runs")
                
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")
        
        # Process other JSON files
        for file_path in self.file_types['other_files']:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                file_key = file_path.stem
                raw_data[file_key] = data
                
                runs_count = self._count_total_runs(data)
                if runs_count > 0:
                    print(f"✅ Loaded {file_key}: {runs_count} runs")
                else:
                    print(f"ℹ️ Loaded {file_key}: (metadata/configuration file)")
                
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")
        
        return raw_data
    
    def _count_total_runs(self, data: Dict) -> int:
        """Count total simulation runs in a result file"""
        total_runs = 0
        
        # Handle different file structures
        if 'topology_results' in data:
            # Multi-topology comparison file
            for topology_data in data['topology_results'].values():
                if 'results' in topology_data:
                    for param_data in topology_data['results'].values():
                        total_runs += len(param_data.get('runs', []))
        elif 'results' in data:
            # Single topology study file
            for param_data in data['results'].values():
                total_runs += len(param_data.get('runs', []))
        
        return total_runs
    
    def _validate_statistical_adequacy(self, raw_data: Dict) -> Dict[str, Any]:
        """CRITICAL: Validate whether you have enough runs for statistical significance"""
        adequacy_report = {
            'overall_adequate': True,
            'topology_analysis': {},
            'recommendations': []
        }
        
        topology_run_counts = {}
        
        for file_key, data in raw_data.items():
            if 'topology_results' in data:
                for topology_type, topology_data in data['topology_results'].items():
                    if topology_type not in topology_run_counts:
                        topology_run_counts[topology_type] = {}
                    
                    if 'results' in topology_data:
                        for param_key, param_data in topology_data['results'].items():
                            runs = param_data.get('runs', [])
                            if param_key not in topology_run_counts[topology_type]:
                                topology_run_counts[topology_type][param_key] = 0
                            topology_run_counts[topology_type][param_key] += len(runs)
        
        # Analyze adequacy for each topology-parameter combination
        for topology, params in topology_run_counts.items():
            topology_adequate = True
            param_analysis = {}
            
            for param, run_count in params.items():
                is_adequate = run_count >= self.min_runs_for_stats
                param_analysis[param] = {
                    'run_count': run_count,
                    'adequate': is_adequate,
                    'power_estimate': self._estimate_statistical_power(run_count)
                }
                
                if not is_adequate:
                    topology_adequate = False
                    adequacy_report['recommendations'].append(
                        f"⚠️ {topology} parameter {param}: {run_count} runs "
                        f"(need {self.min_runs_for_stats}+ for robust statistics)"
                    )
            
            adequacy_report['topology_analysis'][topology] = {
                'overall_adequate': topology_adequate,
                'parameters': param_analysis
            }
            
            if not topology_adequate:
                adequacy_report['overall_adequate'] = False
        
        # Print adequacy report
        print("\n📊 STATISTICAL ADEQUACY ASSESSMENT:")
        print("-" * 40)
        
        if adequacy_report['overall_adequate']:
            print("✅ Sufficient runs for statistical analysis")
        else:
            print("❌ INSUFFICIENT RUNS FOR ROBUST STATISTICS")
            for rec in adequacy_report['recommendations']:
                print(rec)
        
        return adequacy_report
    
    def _estimate_statistical_power(self, n: int, effect_size: float = 0.5, alpha: float = 0.05) -> float:
        """Estimate statistical power for detecting medium effect sizes"""
        # Simplified power calculation for t-test
        if n < 3:
            return 0.0
        
        # Cohen's d = 0.5 (medium effect)
        # This is a rough approximation
        import math
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        non_centrality = effect_size * math.sqrt(n / 2)
        power = 1 - stats.t.cdf(t_critical, n - 1, non_centrality)
        
        return max(0.0, min(1.0, power))
    
    def _aggregate_simulation_runs(self, raw_data: Dict) -> Dict[str, Any]:
        """Aggregate simulation runs across all files by topology and parameter"""
        aggregated = {}
        
        for file_key, data in raw_data.items():
            print(f"🔄 Processing {file_key}...")
            
            # Handle multi-topology files
            if 'topology_results' in data:
                for topology_type, topology_data in data['topology_results'].items():
                    if topology_type not in aggregated:
                        aggregated[topology_type] = {}
                    
                    if 'results' in topology_data:
                        for param_key, param_data in topology_data['results'].items():
                            if param_key not in aggregated[topology_type]:
                                aggregated[topology_type][param_key] = {
                                    'runs': [],
                                    'parameter_value': param_data.get('parameter_value')
                                }
                            
                            # Aggregate runs
                            runs = param_data.get('runs', [])
                            aggregated[topology_type][param_key]['runs'].extend(runs)
            
            # Handle single topology files  
            elif 'results' in data and 'topology_type' in data:
                topology_type = data['topology_type']
                if topology_type not in aggregated:
                    aggregated[topology_type] = {}
                
                for param_key, param_data in data['results'].items():
                    if param_key not in aggregated[topology_type]:
                        aggregated[topology_type][param_key] = {
                            'runs': [],
                            'parameter_value': param_data.get('parameter_value')
                        }
                    
                    runs = param_data.get('runs', [])
                    aggregated[topology_type][param_key]['runs'].extend(runs)
        
        # Print aggregation summary
        print("\n📈 AGGREGATION SUMMARY:")
        print("-" * 30)
        
        for topology, params in aggregated.items():
            total_runs = sum(len(param_data['runs']) for param_data in params.values())
            print(f"{topology}: {len(params)} parameters, {total_runs} total runs")
            
            for param_key, param_data in params.items():
                run_count = len(param_data['runs'])
                status = "✅" if run_count >= self.min_runs_for_stats else "⚠️"
                print(f"  {param_key}: {run_count} runs {status}")
        
        return aggregated
    
    def _calculate_robust_statistics(self, aggregated_data: Dict) -> Dict[str, Any]:
        """Calculate comprehensive statistics with confidence intervals"""
        metrics = ['final_mode_choice_equity', 'final_travel_time_equity', 'final_system_efficiency']
        statistics = {}
        
        for topology, params in aggregated_data.items():
            statistics[topology] = {}
            
            for param_key, param_data in params.items():
                runs = param_data['runs']
                param_stats = {
                    'sample_size': len(runs),
                    'parameter_value': param_data.get('parameter_value'),
                    'metrics': {}
                }
                
                for metric in metrics:
                    values = [run.get(metric, 0) for run in runs if metric in run]
                    
                    if len(values) >= 2:
                        # Descriptive statistics
                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1)
                        sem_val = stats.sem(values)
                        
                        # Confidence interval
                        if len(values) >= 3:
                            ci_lower, ci_upper = stats.t.interval(
                                confidence=0.95,
                                df=len(values)-1,
                                loc=mean_val,
                                scale=sem_val
                            )
                        else:
                            ci_lower, ci_upper = mean_val, mean_val
                        
                        # Normality test
                        if len(values) >= 8:
                            _, normality_p = stats.shapiro(values)
                            is_normal = normality_p > self.alpha
                        else:
                            normality_p, is_normal = None, None
                        
                        param_stats['metrics'][metric] = {
                            'mean': mean_val,
                            'std': std_val,
                            'sem': sem_val,
                            'median': np.median(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'confidence_interval_95': {
                                'lower': ci_lower,
                                'upper': ci_upper,
                                'margin_of_error': (ci_upper - ci_lower) / 2
                            },
                            'normality_test': {
                                'p_value': normality_p,
                                'is_normal': is_normal
                            }
                        }
                
                statistics[topology][param_key] = param_stats
        
        return statistics
    
    def _perform_statistical_tests(self, aggregated_data: Dict) -> Dict[str, Any]:
        """Perform ANOVA and pairwise comparisons"""
        from scipy.stats import f_oneway
        from itertools import combinations
        
        metrics = ['final_mode_choice_equity', 'final_travel_time_equity', 'final_system_efficiency']
        comparative_analysis = {
            'cross_topology_anova': {},
            'pairwise_comparisons': {},
            'effect_sizes': {}
        }
        
        for metric in metrics:
            print(f"\n🧮 Analyzing {metric}...")
            
            # Collect data by topology (pooling across parameters)
            topology_data = {}
            for topology, params in aggregated_data.items():
                topology_values = []
                for param_data in params.values():
                    runs = param_data['runs']
                    values = [run.get(metric, 0) for run in runs if metric in run]
                    topology_values.extend(values)
                
                if topology_values:
                    topology_data[topology] = topology_values
            
            # Perform ANOVA if we have enough topologies and data
            if len(topology_data) >= 2:
                topology_names = list(topology_data.keys())
                topology_values_list = [topology_data[name] for name in topology_names]
                
                # Check minimum sample sizes
                valid_topologies = [(name, values) for name, values in zip(topology_names, topology_values_list) 
                                  if len(values) >= 3]
                
                if len(valid_topologies) >= 2:
                    f_stat, p_value = f_oneway(*[values for _, values in valid_topologies])
                    
                    # Calculate effect size (eta-squared)
                    ss_total = sum(np.var(values, ddof=1) * (len(values) - 1) 
                                 for _, values in valid_topologies)
                    ss_between = sum(len(values) * (np.mean(values) - np.mean([v for _, vals in valid_topologies for v in vals]))**2 
                                   for _, values in valid_topologies)
                    eta_squared = ss_between / (ss_between + ss_total) if (ss_between + ss_total) > 0 else 0
                    
                    comparative_analysis['cross_topology_anova'][metric] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'eta_squared': eta_squared,
                        'effect_size': self._interpret_effect_size(eta_squared),
                        'topology_means': {name: np.mean(values) for name, values in valid_topologies},
                        'topology_stds': {name: np.std(values, ddof=1) for name, values in valid_topologies}
                    }
                    
                    # Pairwise t-tests with Bonferroni correction
                    pairwise_results = {}
                    for (name1, values1), (name2, values2) in combinations(valid_topologies, 2):
                        if len(values1) >= 3 and len(values2) >= 3:
                            t_stat, p_val = stats.ttest_ind(values1, values2)
                            
                            # Cohen's d
                            pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) + 
                                                (len(values2) - 1) * np.var(values2, ddof=1)) / 
                                               (len(values1) + len(values2) - 2))
                            cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
                            
                            pairwise_results[f"{name1}_vs_{name2}"] = {
                                't_statistic': t_stat,
                                'p_value': p_val,
                                'significant': p_val < self.alpha,
                                'cohens_d': abs(cohens_d),
                                'effect_size': self._interpret_cohens_d(abs(cohens_d))
                            }
                    
                    comparative_analysis['pairwise_comparisons'][metric] = pairwise_results
        
        return comparative_analysis
    
    def _interpret_effect_size(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size"""
        if eta_squared < 0.01:
            return "negligible"
        elif eta_squared < 0.06:
            return "small"
        elif eta_squared < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _create_publication_plots(self, aggregated_data: Dict, statistics: Dict) -> List[str]:
        """Create publication-ready visualizations"""
        viz_paths = []
        
        # 1. Cross-topology performance comparison with error bars
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Network Topology Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['final_mode_choice_equity', 'final_travel_time_equity', 'final_system_efficiency']
        metric_labels = ['Mode Choice Equity', 'Travel Time Equity', 'System Efficiency']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            
            topologies = []
            means = []
            errors = []
            
            for topology, params in statistics.items():
                # Pool across parameters for topology-level analysis
                topology_values = []
                for param_stats in params.values():
                    if metric in param_stats['metrics']:
                        runs = param_stats['sample_size']
                        if runs >= 3:  # Only include adequate samples
                            topology_values.append(param_stats['metrics'][metric]['mean'])
                
                if topology_values:
                    topologies.append(topology.replace('_', ' ').title())
                    topology_mean = np.mean(topology_values)
                    topology_error = np.std(topology_values) / np.sqrt(len(topology_values))
                    means.append(topology_mean)
                    errors.append(topology_error)
            
            if topologies:
                bars = ax.bar(topologies, means, yerr=errors, capsize=5, alpha=0.7)
                ax.set_title(label)
                ax.set_ylabel('Performance Score')
                
                # Color code bars
                colors = plt.cm.Set3(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                # Add value labels
                for bar, mean_val in zip(bars, means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(errors)*0.1,
                           f'{mean_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        path1 = self.output_dir / 'topology_performance_comparison.png'
        plt.savefig(path1, dpi=300, bbox_inches='tight')
        plt.close()
        viz_paths.append(str(path1))
        
        print(f"✅ Performance comparison plot saved: {path1}")
        
        return viz_paths
    
    def _save_final_report(self, report: Dict):
        """Save comprehensive aggregation report"""
        # Save JSON report
        json_path = self.output_dir / f'aggregated_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save human-readable summary
        txt_path = self.output_dir / 'aggregation_summary.txt'
        with open(txt_path, 'w') as f:
            f.write("NTEO STATISTICAL AGGREGATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Adequacy assessment
            f.write("STATISTICAL ADEQUACY ASSESSMENT:\n")
            f.write("-" * 35 + "\n")
            adequacy = report['metadata']['statistical_adequacy']
            f.write(f"Overall adequate: {'YES' if adequacy['overall_adequate'] else 'NO'}\n\n")
            
            if adequacy['recommendations']:
                f.write("Recommendations:\n")
                for rec in adequacy['recommendations']:
                    f.write(f"  {rec}\n")
            
            f.write(f"\nTotal simulation runs processed: {report['metadata']['total_runs_found']}\n")
            f.write(f"Source files: {len(report['metadata']['source_files'])}\n")
            
            # Statistical results
            if 'comparative_analysis' in report and 'cross_topology_anova' in report['comparative_analysis']:
                f.write("\nCROSS-TOPOLOGY ANOVA RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                for metric, anova_results in report['comparative_analysis']['cross_topology_anova'].items():
                    f.write(f"\n{metric.replace('final_', '').replace('_', ' ').title()}:\n")
                    f.write(f"  F-statistic: {anova_results['f_statistic']:.3f}\n")
                    f.write(f"  p-value: {anova_results['p_value']:.6f}\n")
                    f.write(f"  Significant: {'YES' if anova_results['significant'] else 'NO'}\n")
                    f.write(f"  Effect size (η²): {anova_results['eta_squared']:.3f} ({anova_results['effect_size']})\n")
            
            f.write(f"\nGenerated: {report['metadata']['aggregation_timestamp']}\n")
        
        print(f"📊 Final report saved: {json_path}")
        print(f"📄 Summary saved: {txt_path}")

# Usage function with file verification
def run_aggregation_with_verification(results_directory: str, output_directory: str = None):
    """
    Main function to run the aggregation pipeline with detailed file verification
    
    This handles the specific NTEO file naming conventions:
    - degree_constrained_study_YYYYMMDD_HHMMSS.json
    - small_world_study_YYYYMMDD_HHMMSS.json  
    - scale_free_study_YYYYMMDD_HHMMSS.json
    - multi_topology_comparison_YYYYMMDD_HHMMSS.json
    
    Args:
        results_directory: Path to directory containing JSON result files
        output_directory: Path for output (default: "aggregated_results")
    """
    print("🔍 NTEO Results Aggregation with File Verification")
    print("="*60)
    
    aggregator = NTEOResultsAggregator(results_directory, output_directory)
    
    # Check if we have the expected file types
    expected_patterns = [
        "*_study_*.json",  # Individual topology studies
        "multi_topology_comparison_*.json"  # Multi-topology comparisons
    ]
    
    found_patterns = {}
    for pattern in expected_patterns:
        matching_files = list(Path(results_directory).glob(pattern))
        found_patterns[pattern] = len(matching_files)
        print(f"📁 {pattern}: {len(matching_files)} files found")
    
    if sum(found_patterns.values()) < 2:
        print("❌ Need at least 2 result files for meaningful aggregation")
        return None
    
    # Run complete aggregation
    print(f"\n🚀 Starting aggregation of {len(aggregator.result_files)} files...")
    final_report = aggregator.aggregate_all_results()
    
    # Print final summary
    print("\n" + "="*60)
    print("AGGREGATION COMPLETE")
    print("="*60)
    
    adequacy = final_report['metadata']['statistical_adequacy']
    if adequacy['overall_adequate']:
        print("✅ Results are statistically adequate for publication")
    else:
        print("⚠️ WARNING: Results may not be statistically robust")
        print("Consider running more simulations before publication")
        
        # Show specific recommendations
        for rec in adequacy['recommendations'][:5]:  # Show first 5 recommendations
            print(f"   {rec}")
    
    print(f"📁 Results saved in: {aggregator.output_dir}")
    print(f"📊 Files processed: {len(final_report['metadata']['source_files'])}")
    
    return final_report

# Usage function
def run_aggregation(results_directory: str, output_directory: str = None):
    """
    Simple wrapper for the main aggregation function
    """
    return run_aggregation_with_verification(results_directory, output_directory)