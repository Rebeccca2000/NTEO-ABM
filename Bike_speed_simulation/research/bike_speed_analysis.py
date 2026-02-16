# bike_speed_analysis.py
"""
Bike Speed Parameter Analysis for NTEO
Tests different bike speeds on baseline public transport network (Grid topology)
Measures impact on mode choice equity, travel time equity, and system efficiency

Author: NTEO Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
import json
from pathlib import Path

# Import your existing modules

from config.network_config import NetworkConfigurationManager
from config.database_updated import *
from testing.abm_initialization import MobilityModelNTEO


class BikeSpeedAnalyzer:
    """
    Analyzes impact of bike speed variations on transport equity metrics.
    
    Follows SOLID principles:
    - Single Responsibility: Only handles bike speed analysis
    - Open/Closed: Extensible for other speed parameters
    - Liskov Substitution: Compatible with base analysis framework
    - Interface Segregation: Clean interface for speed testing
    - Dependency Inversion: Depends on abstractions, not concrete implementations
    """
    
    def __init__(self, output_dir: str = "results/bike_speed_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration for baseline network (Grid topology)
        self.baseline_config = {
            'topology_type': 'base_sydney',  # Use base Sydney network, not grid
            'connectivity_level': 6,  # Standard baseline connectivity
            'grid_width': 100,
            'grid_height': 80,
            'sydney_realism': True
        }
        
        # Analysis parameters
        self.bike_speeds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # Range of speeds to test
        self.num_runs_per_speed = 5  # Multiple runs for statistical reliability
        self.simulation_steps = 144  # Standard simulation length
        
        # Results storage
        self.results = {}
        
    def run_bike_speed_analysis(self) -> Dict:
        """
        Main analysis method - tests different bike speeds systematically.
        
        Returns:
            Dict: Complete analysis results with statistical summaries
        """
        print("🚴 Starting Bike Speed Analysis on Baseline Public Transport Network")
        print("="*70)
        print(f"Network: {self.baseline_config['topology_type']}")
        print(f"Speed range: {self.bike_speeds}")
        print(f"Runs per speed: {self.num_runs_per_speed}")
        print()
        
        analysis_start_time = time.time()
        
        for speed_idx, bike_speed in enumerate(self.bike_speeds):
            print(f"🔄 Testing Bike Speed: {bike_speed} ({speed_idx+1}/{len(self.bike_speeds)})")
            
            speed_results = {
                'bike_speed': bike_speed,
                'runs': [],
                'statistics': {}
            }
            
            # Run multiple simulations for this speed
            for run_idx in range(self.num_runs_per_speed):
                print(f"   Run {run_idx+1}/{self.num_runs_per_speed}...", end="")
                
                try:
                    # Create model with modified bike speed
                    model = self._create_model_with_bike_speed(bike_speed)
                    
                    # Run simulation
                    run_data = self._run_single_simulation(model, bike_speed, run_idx)
                    speed_results['runs'].append(run_data)
                    
                    print(f" ✅ (Equity: {run_data['final_mode_choice_equity']:.3f})")
                    
                except Exception as e:
                    print(f" ❌ Failed: {e}")
                    continue
            
            # Calculate statistics for this speed
            if speed_results['runs']:
                speed_results['statistics'] = self._calculate_speed_statistics(speed_results['runs'])
                self.results[bike_speed] = speed_results
            
            print()
        
        # Cross-speed analysis
        analysis_results = {
            'analysis_type': 'bike_speed_sensitivity',
            'baseline_network': self.baseline_config,
            'speed_results': self.results,
            'statistical_analysis': self._perform_cross_speed_analysis(),
            'execution_time': time.time() - analysis_start_time
        }
        
        # Save and visualize results
        self._save_results(analysis_results)
        self._create_visualizations(analysis_results)
        
        print("✅ Bike Speed Analysis Complete!")
        return analysis_results
    
    def _create_model_with_bike_speed(self, bike_speed: float) -> MobilityModelNTEO:
        """
        Creates model with specified bike speed.
        
        CRITICAL: This requires modifying the ServiceProviderAgent to accept
        configurable bike speed rather than hardcoded value.
        """
        # Create network configuration
        config_manager = NetworkConfigurationManager()
        config_manager.switch_topology_type('degree_constrained')
        network_config = config_manager.get_base_configuration()
        
        # Create model with standard parameters but custom bike speed
        model = MobilityModelNTEO(
            db_connection_string=DB_CONNECTION_STRING,
            num_commuters=num_commuters,
            income_weights=income_weights,
            health_weights=health_weights,
            payment_weights=payment_weights,
            disability_weights=disability_weights,
            tech_access_weights=tech_access_weights,
            age_distribution=age_distribution,
            penalty_coefficients=PENALTY_COEFFICIENTS,
            affordability_thresholds=AFFORDABILITY_THRESHOLDS,
            value_of_time=VALUE_OF_TIME,
            flexibility_adjustments=FLEXIBILITY_ADJUSTMENTS,
            asc_values=ASC_VALUES,
            utility_function_base_coefficients=UTILITY_FUNCTION_BASE_COEFFICIENTS,
            utility_function_high_income_car_coefficients=UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
            public_price_table=public_price_table,
            alpha_values=ALPHA_VALUES,
            dynamic_maas_surcharge_base_coefficients=DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
            background_traffic_amount=BACKGROUND_TRAFFIC_AMOUNT,
            congestion_alpha=CONGESTION_ALPHA,
            congestion_beta=CONGESTION_BETA,
            congestion_capacity=CONGESTION_CAPACITY,
            congestion_t_ij_free_flow=CONGESTION_T_IJ_FREE_FLOW,
            uber_like1_capacity=UberLike1_capacity,
            uber_like1_price=UberLike1_price,
            uber_like2_capacity=UberLike2_capacity,
            uber_like2_price=UberLike2_price,
            bike_share1_capacity=BikeShare1_capacity,
            bike_share1_price=BikeShare1_price,
            bike_share2_capacity=BikeShare2_capacity,
            bike_share2_price=BikeShare2_price,
            subsidy_dataset=subsidy_dataset,
            subsidy_config=daily_config,
            network_config=network_config
        )

        # Attach bike speed as a model attribute so agents can access it if needed
        try:
            model.bike_speed = bike_speed
        except Exception:
            # Fallback: ignore if model object is not mutable in this way
            pass
        
        
        return model
    
    def _run_single_simulation(self, model: MobilityModelNTEO, bike_speed: float, run_idx: int) -> Dict:
        """Run a single simulation and collect metrics."""
        start_time = time.time()
        
        # Run simulation
        for step in range(self.simulation_steps):
            model.step()
        
        # Collect final metrics
        final_metrics = {
            'run_id': f"bike_speed_{bike_speed}_run_{run_idx}",
            'bike_speed': bike_speed,
            'execution_time': time.time() - start_time,
            'final_mode_choice_equity': model.calculate_mode_choice_equity(),
            'final_travel_time_equity': model.calculate_travel_time_equity(),
            'final_system_efficiency': model.calculate_system_efficiency(),
            'num_steps': self.simulation_steps,
            'model_summary': model.get_model_summary()
        }
        
        return final_metrics
    
    def _calculate_speed_statistics(self, runs: List[Dict]) -> Dict:
        """Calculate statistical summary for a specific bike speed."""
        metrics = ['final_mode_choice_equity', 'final_travel_time_equity', 'final_system_efficiency']
        
        statistics = {}
        for metric in metrics:
            values = [run[metric] for run in runs if metric in run]
            if values:
                statistics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'n_samples': len(values)
                }
        
        return statistics
    
    def _perform_cross_speed_analysis(self) -> Dict:
        """Analyze trends across different bike speeds."""
        if len(self.results) < 2:
            return {'error': 'Insufficient data for cross-speed analysis'}
        
        analysis = {}
        metrics = ['final_mode_choice_equity', 'final_travel_time_equity', 'final_system_efficiency']
        
        # Prepare data for analysis
        speed_data = {}
        for metric in metrics:
            speeds = []
            values = []
            
            for bike_speed, results in self.results.items():
                if 'statistics' in results and metric in results['statistics']:
                    speeds.append(bike_speed)
                    values.append(results['statistics'][metric]['mean'])
            
            if len(speeds) >= 3:  # Need at least 3 points for trend analysis
                # Calculate correlation with bike speed
                correlation = np.corrcoef(speeds, values)[0, 1] if len(speeds) > 1 else 0
                
                # Calculate trend (linear fit)
                if len(speeds) >= 2:
                    trend_coeff = np.polyfit(speeds, values, 1)[0]  # Slope of linear fit
                else:
                    trend_coeff = 0
                
                speed_data[metric] = {
                    'speeds': speeds,
                    'values': values,
                    'correlation_with_speed': correlation,
                    'trend_coefficient': trend_coeff,
                    'interpretation': self._interpret_trend(correlation, trend_coeff, metric)
                }
        
        analysis['speed_sensitivity'] = speed_data
        
        # Find optimal bike speed
        analysis['optimal_speeds'] = self._find_optimal_speeds()
        
        return analysis
    
    def _interpret_trend(self, correlation: float, trend_coeff: float, metric: str) -> str:
        """Interpret the relationship between bike speed and metric."""
        corr_strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
        direction = "positive" if trend_coeff > 0 else "negative"
        
        if "equity" in metric.lower():
            equity_interpretation = "worse" if trend_coeff > 0 else "better" 
            return f"{corr_strength} {direction} trend (faster bikes → {equity_interpretation} equity)"
        else:
            # System efficiency: lower = better (consistent with equity metrics)
            efficiency_interpretation = "worse" if trend_coeff > 0 else "better"
            return f"{corr_strength} {direction} trend (faster bikes → {efficiency_interpretation} efficiency)"
    
    def _find_optimal_speeds(self) -> Dict:
        """Find optimal bike speeds for different objectives."""
        optimal_speeds = {}
        
        # Best mode choice equity (lowest value)
        best_mode_equity_speed = None
        best_mode_equity_value = float('inf')
        
        # Best travel time equity (lowest value)
        best_time_equity_speed = None
        best_time_equity_value = float('inf')
        
        # Best system efficiency (lowest value - consistent with "lower = better")
        best_efficiency_speed = None
        best_efficiency_value = float('inf')
        
        for bike_speed, results in self.results.items():
            if 'statistics' not in results:
                continue
                
            stats = results['statistics']
            
            # Mode choice equity
            if 'final_mode_choice_equity' in stats:
                value = stats['final_mode_choice_equity']['mean']
                if value < best_mode_equity_value:
                    best_mode_equity_value = value
                    best_mode_equity_speed = bike_speed
            
            # Travel time equity  
            if 'final_travel_time_equity' in stats:
                value = stats['final_travel_time_equity']['mean']
                if value < best_time_equity_value:
                    best_time_equity_value = value
                    best_time_equity_speed = bike_speed
            
            # System efficiency (lower = better)
            if 'final_system_efficiency' in stats:
                value = stats['final_system_efficiency']['mean']
                if value < best_efficiency_value:
                    best_efficiency_value = value
                    best_efficiency_speed = bike_speed
        
        return {
            'best_mode_choice_equity': {'speed': best_mode_equity_speed, 'value': best_mode_equity_value},
            'best_travel_time_equity': {'speed': best_time_equity_speed, 'value': best_time_equity_value},
            'best_system_efficiency': {'speed': best_efficiency_speed, 'value': best_efficiency_value}
        }
    
    def _create_visualizations(self, analysis_results: Dict):
        """Create comprehensive visualizations of bike speed analysis."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bike Speed Impact on Transport System Metrics\nBaseline Public Transport Network (Grid Topology)', 
                    fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        speeds = []
        mode_equity_means = []
        mode_equity_stds = []
        time_equity_means = []
        time_equity_stds = []
        efficiency_means = []
        efficiency_stds = []
        
        for bike_speed in sorted(self.results.keys()):
            results = self.results[bike_speed]
            if 'statistics' not in results:
                continue
                
            stats = results['statistics']
            speeds.append(bike_speed)
            
            # Mode choice equity
            if 'final_mode_choice_equity' in stats:
                mode_equity_means.append(stats['final_mode_choice_equity']['mean'])
                mode_equity_stds.append(stats['final_mode_choice_equity']['std'])
            else:
                mode_equity_means.append(np.nan)
                mode_equity_stds.append(0)
            
            # Travel time equity
            if 'final_travel_time_equity' in stats:
                time_equity_means.append(stats['final_travel_time_equity']['mean'])
                time_equity_stds.append(stats['final_travel_time_equity']['std'])
            else:
                time_equity_means.append(np.nan)
                time_equity_stds.append(0)
            
            # System efficiency
            if 'final_system_efficiency' in stats:
                efficiency_means.append(stats['final_system_efficiency']['mean'])
                efficiency_stds.append(stats['final_system_efficiency']['std'])
            else:
                efficiency_means.append(np.nan)
                efficiency_stds.append(0)
        
        # Plot 1: Mode Choice Equity
        axes[0, 0].errorbar(speeds, mode_equity_means, yerr=mode_equity_stds, 
                           marker='o', capsize=5, linewidth=2, markersize=8)
        axes[0, 0].set_title('Mode Choice Equity vs Bike Speed\n(Lower = Better)', fontweight='bold')
        axes[0, 0].set_xlabel('Bike Speed (unit length/step)')
        axes[0, 0].set_ylabel('Mode Choice Equity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Travel Time Equity
        axes[0, 1].errorbar(speeds, time_equity_means, yerr=time_equity_stds, 
                           marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_title('Travel Time Equity vs Bike Speed\n(Lower = Better)', fontweight='bold')
        axes[0, 1].set_xlabel('Bike Speed (unit length/step)')
        axes[0, 1].set_ylabel('Travel Time Equity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: System Efficiency
        axes[1, 0].errorbar(speeds, efficiency_means, yerr=efficiency_stds, 
                           marker='^', capsize=5, linewidth=2, markersize=8, color='green')
        axes[1, 0].set_title('System Efficiency vs Bike Speed\n(Lower = Better)', fontweight='bold')
        axes[1, 0].set_xlabel('Bike Speed (unit length/step)')
        axes[1, 0].set_ylabel('System Efficiency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Combined normalized metrics
        if len(speeds) > 1:
            # Normalize metrics for comparison (0-1 scale) - ALL metrics: lower = better
            norm_mode_equity = 1 - (np.array(mode_equity_means) - np.nanmin(mode_equity_means)) / (np.nanmax(mode_equity_means) - np.nanmin(mode_equity_means)) if not np.all(np.isnan(mode_equity_means)) else [0] * len(speeds)
            norm_time_equity = 1 - (np.array(time_equity_means) - np.nanmin(time_equity_means)) / (np.nanmax(time_equity_means) - np.nanmin(time_equity_means)) if not np.all(np.isnan(time_equity_means)) else [0] * len(speeds)
            norm_efficiency = 1 - (np.array(efficiency_means) - np.nanmin(efficiency_means)) / (np.nanmax(efficiency_means) - np.nanmin(efficiency_means)) if not np.all(np.isnan(efficiency_means)) else [0] * len(speeds)
            
            axes[1, 1].plot(speeds, norm_mode_equity, marker='o', linewidth=2, label='Mode Choice Equity')
            axes[1, 1].plot(speeds, norm_time_equity, marker='s', linewidth=2, label='Travel Time Equity')
            axes[1, 1].plot(speeds, norm_efficiency, marker='^', linewidth=2, label='System Efficiency')
            axes[1, 1].set_title('Normalized Metrics Comparison\n(Higher = Better)', fontweight='bold')
            axes[1, 1].set_xlabel('Bike Speed (unit length/step)')
            axes[1, 1].set_ylabel('Normalized Score (0-1)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'bike_speed_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: {plot_path}")
    
    def _save_results(self, analysis_results: Dict):
        """Save complete analysis results."""
        # Save JSON results
        json_path = self.output_dir / 'bike_speed_analysis_results.json'
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_data = []
        for bike_speed, results in self.results.items():
            if 'statistics' not in results:
                continue
                
            stats = results['statistics']
            row = {'bike_speed': bike_speed}
            
            for metric in ['final_mode_choice_equity', 'final_travel_time_equity', 'final_system_efficiency']:
                if metric in stats:
                    row[f'{metric}_mean'] = stats[metric]['mean']
                    row[f'{metric}_std'] = stats[metric]['std']
                    row[f'{metric}_n'] = stats[metric]['n_samples']
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / 'bike_speed_summary.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"📁 Results saved to: {json_path}")
        print(f"📊 Summary CSV saved to: {csv_path}")


# ===== CRITICAL MODIFICATION NEEDED =====
class EnhancedServiceProviderAgent:
    """
    Enhanced service provider that accepts configurable bike speed.
    
    You need to modify your existing agent_service_provider.py with this functionality:
    """
    
    def __init__(self, *args, bike_speed: float = 2.0, **kwargs):
        # Initialize with your existing code
        self.bike_speed = bike_speed
        # ... rest of your existing __init__ code
    
    def set_bike_speed(self, new_bike_speed: float):
        """Allow dynamic modification of bike speed for analysis."""
        self.bike_speed = new_bike_speed
        print(f"🚴 Bike speed updated to: {new_bike_speed}")
    
    def get_travel_speed(self, mode, current_ticks):
        """Modified version of your existing method with configurable bike speed."""
        if mode == 'bus':
            return 3
        elif mode == 'train':
            return 5
        elif mode == 'walk':
            return 1.5
        elif mode == 'bike':
            return self.bike_speed  # <-- KEY CHANGE: Use configurable speed
        elif mode == 'car':
            if self.check_is_peak(current_ticks):
                return 6.5
            else:
                return 7.5
        else:
            print(f"❌Unknown mode: {mode}")
            return 0


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BikeSpeedAnalyzer(output_dir="results/bike_speed_analysis")
    
    # Run analysis
    results = analyzer.run_bike_speed_analysis()
    
    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    
    if 'optimal_speeds' in results['statistical_analysis']:
        optimal = results['statistical_analysis']['optimal_speeds']
        print(f"Best Mode Choice Equity: Speed {optimal['best_mode_choice_equity']['speed']} (score: {optimal['best_mode_choice_equity']['value']:.3f})")
        print(f"Best Travel Time Equity: Speed {optimal['best_travel_time_equity']['speed']} (score: {optimal['best_travel_time_equity']['value']:.3f})")
        print(f"Best System Efficiency: Speed {optimal['best_system_efficiency']['speed']} (score: {optimal['best_system_efficiency']['value']:.1f})")
    
    if 'speed_sensitivity' in results['statistical_analysis']:
        sensitivity = results['statistical_analysis']['speed_sensitivity']
        for metric, data in sensitivity.items():
            print(f"{metric}: {data['interpretation']}")