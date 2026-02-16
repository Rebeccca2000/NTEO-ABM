import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import pairwise_distances
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class TransportEquityAnalyzer:
    def __init__(self, db_connection_string, network_manager):
        self.db_engine = create_engine(db_connection_string)
        self.Session = sessionmaker(bind=self.db_engine)
        self.network_manager = network_manager
        
    def calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for inequality measurement"""
        if len(values) == 0:
            return 0
        
        values = np.array(values)
        values = values[~np.isnan(values)]  # Remove NaN values
        
        if len(values) == 0 or np.all(values == 0):
            return 0
        
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
        
        return max(0, gini)  # Ensure non-negative
    
    def get_accessibility_by_demographics(self):
        """Calculate accessibility metrics by demographic groups"""
        with self.Session() as session:
            query = text("""
                SELECT 
                    c.commuter_id,
                    c.income_level,
                    c.age,
                    c.has_disability,
                    c.payment_scheme,
                    c.location_x,
                    c.location_y,
                    COUNT(s.commuter_id) as total_trips,
                    AVG(s.total_time) as avg_trip_time,
                    AVG(s.total_price) as avg_trip_cost,
                    AVG(COALESCE(s.government_subsidy, 0)) as avg_subsidy,
                    STRING_AGG(DISTINCT s.record_company_name, ', ') as modes_used
                FROM commuter_info_log c
                LEFT JOIN service_booking_log s ON c.commuter_id = s.commuter_id
                WHERE s.status IN ('Service Selected', 'finished')  
                GROUP BY c.commuter_id, c.income_level, c.age, c.has_disability, 
                        c.payment_scheme, c.location_x, c.location_y
                HAVING COUNT(s.commuter_id) > 0
            """)
            
            return pd.read_sql(query, session.bind)
        
    def calculate_spatial_equity_metrics(self):
        """Calculate spatial equity metrics across the network"""
        df = self.get_accessibility_by_demographics()
        
        if df.empty:
            print("No data available for equity analysis!")
            return {}
        
        # Parse location data
        df['x_coord'] = df['location_x']  # Direct access
        df['y_coord'] = df['location_y']  # Direct access
            
        # Calculate equity metrics
        equity_metrics = {}
        
        # 1. Gini coefficients
        equity_metrics['gini_trip_time'] = self.calculate_gini_coefficient(df['avg_trip_time'])
        equity_metrics['gini_trip_cost'] = self.calculate_gini_coefficient(df['avg_trip_cost'])
        equity_metrics['gini_subsidies'] = self.calculate_gini_coefficient(df['avg_subsidy'])
        equity_metrics['gini_trip_frequency'] = self.calculate_gini_coefficient(df['total_trips'])
        
        # 2. Income-based equity
        income_groups = df.groupby('income_level')
        equity_metrics['income_time_ratio'] = (
            income_groups['avg_trip_time'].mean()['high'] / 
            income_groups['avg_trip_time'].mean()['low']
        ) if 'high' in income_groups.groups and 'low' in income_groups.groups else 1.0
        
        equity_metrics['income_cost_ratio'] = (
            income_groups['avg_trip_cost'].mean()['high'] / 
            income_groups['avg_trip_cost'].mean()['low']
        ) if 'high' in income_groups.groups and 'low' in income_groups.groups else 1.0
        
        # 3. Disability-based equity
        if df['has_disability'].sum() > 0:
            disability_groups = df.groupby('has_disability')
            equity_metrics['disability_time_penalty'] = (
                disability_groups['avg_trip_time'].mean()[True] / 
                disability_groups['avg_trip_time'].mean()[False]
            ) if True in disability_groups.groups and False in disability_groups.groups else 1.0
        else:
            equity_metrics['disability_time_penalty'] = 1.0
        
        # 4. Spatial clustering metrics
        coordinates = df[['x_coord', 'y_coord']].values
        times = df['avg_trip_time'].values
        costs = df['avg_trip_cost'].values
        
        # Calculate spatial autocorrelation (Moran's I approximation)
        if len(coordinates) > 3:
            distance_matrix = pairwise_distances(coordinates)
            weights = 1 / (distance_matrix + 1e-10)  # Inverse distance weights
            np.fill_diagonal(weights, 0)
            
            # Moran's I for trip times
            equity_metrics['spatial_autocorr_time'] = self.calculate_morans_i(times, weights)
            equity_metrics['spatial_autocorr_cost'] = self.calculate_morans_i(costs, weights)
        else:
            equity_metrics['spatial_autocorr_time'] = 0
            equity_metrics['spatial_autocorr_cost'] = 0
        
        return equity_metrics, df
    
    def calculate_morans_i(self, values, weights):
        """Calculate Moran's I spatial autocorrelation statistic"""
        n = len(values)
        if n <= 1:
            return 0
        
        values = np.array(values)
        mean_val = np.mean(values)
        
        # Calculate Moran's I
        numerator = 0
        denominator = np.sum((values - mean_val) ** 2)
        w_sum = np.sum(weights)
        
        if denominator == 0 or w_sum == 0:
            return 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
        
        morans_i = (n / w_sum) * (numerator / denominator)
        return morans_i
    
    def calculate_network_based_equity(self):
        """Calculate equity metrics based on network topology"""
        # Get network centrality measures
        graph = self.network_manager.active_network
        
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(graph)
        closeness = nx.closeness_centrality(graph)
        degree_centrality = nx.degree_centrality(graph)
        
        # Map to spatial coordinates
        centrality_data = []
        for node in graph.nodes():
            if node in self.network_manager.spatial_mapper.node_to_grid:
                coord = self.network_manager.spatial_mapper.node_to_grid[node]
                centrality_data.append({
                    'node': node,
                    'x': coord[0],
                    'y': coord[1],
                    'betweenness': betweenness[node],
                    'closeness': closeness[node],
                    'degree_centrality': degree_centrality[node]
                })
        
        centrality_df = pd.DataFrame(centrality_data)
        
        # Calculate equity metrics for network access
        network_equity = {}
        
        if not centrality_df.empty:
            network_equity['gini_betweenness'] = self.calculate_gini_coefficient(centrality_df['betweenness'])
            network_equity['gini_closeness'] = self.calculate_gini_coefficient(centrality_df['closeness'])
            network_equity['gini_degree_centrality'] = self.calculate_gini_coefficient(centrality_df['degree_centrality'])
            
            # Calculate accessibility variance across spatial locations
            network_equity['spatial_access_variance'] = np.var(centrality_df['closeness'])
            network_equity['network_coverage'] = len(centrality_df) / (100 * 80)  # Proportion of grid covered
        
        return network_equity, centrality_df
    
    def comprehensive_equity_analysis(self):
        """Run comprehensive equity analysis"""
        print("=== COMPREHENSIVE TRANSPORT EQUITY ANALYSIS ===\n")
        
        # Get demographic-based equity
        spatial_equity, demo_df = self.calculate_spatial_equity_metrics()
        
        # Get network-based equity
        network_equity, centrality_df = self.calculate_network_based_equity()
        
        # Print results
        print("1. DEMOGRAPHIC EQUITY METRICS:")
        print(f"   • Gini Coefficient (Trip Time): {spatial_equity['gini_trip_time']:.3f}")
        print(f"   • Gini Coefficient (Trip Cost): {spatial_equity['gini_trip_cost']:.3f}")
        print(f"   • Gini Coefficient (Subsidies): {spatial_equity['gini_subsidies']:.3f}")
        print(f"   • Gini Coefficient (Trip Frequency): {spatial_equity['gini_trip_frequency']:.3f}")
        print(f"   • Income Time Ratio (High/Low): {spatial_equity['income_time_ratio']:.3f}")
        print(f"   • Income Cost Ratio (High/Low): {spatial_equity['income_cost_ratio']:.3f}")
        print(f"   • Disability Time Penalty: {spatial_equity['disability_time_penalty']:.3f}")
        
        print("\n2. SPATIAL EQUITY METRICS:")
        print(f"   • Spatial Autocorrelation (Time): {spatial_equity['spatial_autocorr_time']:.3f}")
        print(f"   • Spatial Autocorrelation (Cost): {spatial_equity['spatial_autocorr_cost']:.3f}")
        
        print("\n3. NETWORK EQUITY METRICS:")
        print(f"   • Gini Coefficient (Betweenness): {network_equity['gini_betweenness']:.3f}")
        print(f"   • Gini Coefficient (Closeness): {network_equity['gini_closeness']:.3f}")
        print(f"   • Gini Coefficient (Degree Centrality): {network_equity['gini_degree_centrality']:.3f}")
        print(f"   • Spatial Access Variance: {network_equity['spatial_access_variance']:.6f}")
        print(f"   • Network Coverage: {network_equity['network_coverage']:.3f}")
        
        # Create visualization
        self.visualize_equity_analysis(spatial_equity, network_equity, demo_df, centrality_df)
        
        # Combine all metrics
        all_metrics = {**spatial_equity, **network_equity}
        
        return all_metrics
    
    def visualize_equity_analysis(self, spatial_equity, network_equity, demo_df, centrality_df):
        """Create comprehensive equity visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Gini coefficients comparison
        ax1 = axes[0, 0]
        gini_metrics = {
            'Trip Time': spatial_equity['gini_trip_time'],
            'Trip Cost': spatial_equity['gini_trip_cost'],
            'Subsidies': spatial_equity['gini_subsidies'],
            'Trip Frequency': spatial_equity['gini_trip_frequency'],
            'Betweenness': network_equity['gini_betweenness'],
            'Closeness': network_equity['gini_closeness']
        }
        
        bars = ax1.bar(range(len(gini_metrics)), list(gini_metrics.values()), 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange', 'purple'])
        ax1.set_xticks(range(len(gini_metrics)))
        ax1.set_xticklabels(list(gini_metrics.keys()), rotation=45, ha='right')
        ax1.set_ylabel('Gini Coefficient')
        ax1.set_title('Equity Metrics (Lower = More Equitable)')
        ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='High Inequality Threshold')
        ax1.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Income equity visualization
        ax2 = axes[0, 1]
        if not demo_df.empty:
            income_stats = demo_df.groupby('income_level')[['avg_trip_time', 'avg_trip_cost']].mean()
            income_stats.plot(kind='bar', ax=ax2)
            ax2.set_title('Average Trip Metrics by Income Level')
            ax2.set_ylabel('Value')
            ax2.legend()
            plt.setp(ax2.get_xticklabels(), rotation=0)
        
        # 3. Spatial distribution of trip times
        ax3 = axes[0, 2]
        if not demo_df.empty:
            scatter = ax3.scatter(demo_df['x_coord'], demo_df['y_coord'], 
                                c=demo_df['avg_trip_time'], cmap='RdYlBu_r', 
                                s=50, alpha=0.7)
            ax3.set_xlabel('X Coordinate')
            ax3.set_ylabel('Y Coordinate')
            ax3.set_title('Spatial Distribution of Average Trip Times')
            plt.colorbar(scatter, ax=ax3, label='Avg Trip Time')
        
        # 4. Network centrality distribution
        ax4 = axes[1, 0]
        if not centrality_df.empty:
            scatter = ax4.scatter(centrality_df['x'], centrality_df['y'], 
                                c=centrality_df['closeness'], cmap='viridis', 
                                s=100, alpha=0.7)
            ax4.set_xlabel('X Coordinate')
            ax4.set_ylabel('Y Coordinate')
            ax4.set_title('Network Closeness Centrality Distribution')
            plt.colorbar(scatter, ax=ax4, label='Closeness Centrality')
        
        # 5. Mode choice equity
        ax5 = axes[1, 1]
        if not demo_df.empty:
            # Analyze mode diversity
            mode_counts = defaultdict(int)
            for modes_str in demo_df['modes_used'].dropna():
                for mode in modes_str.split(', '):
                    mode_counts[mode.strip()] += 1
            
            if mode_counts:
                modes = list(mode_counts.keys())
                counts = list(mode_counts.values())
                ax5.pie(counts, labels=modes, autopct='%1.1f%%', startangle=90)
                ax5.set_title('Mode Choice Distribution')
        
        # 6. Equity summary
        ax6 = axes[1, 2]
        
        # Calculate overall equity score (lower is better)
        overall_score = np.mean([
            spatial_equity['gini_trip_time'],
            spatial_equity['gini_trip_cost'],
            network_equity['gini_closeness']
        ])
        
        summary_text = f"""
EQUITY SUMMARY:

Overall Equity Score: {overall_score:.3f}
(0 = Perfect Equity, 1 = Maximum Inequality)

Key Findings:
• Trip Time Inequality: {'HIGH' if spatial_equity['gini_trip_time'] > 0.3 else 'MODERATE' if spatial_equity['gini_trip_time'] > 0.2 else 'LOW'}
• Trip Cost Inequality: {'HIGH' if spatial_equity['gini_trip_cost'] > 0.3 else 'MODERATE' if spatial_equity['gini_trip_cost'] > 0.2 else 'LOW'}
• Network Access Inequality: {'HIGH' if network_equity['gini_closeness'] > 0.3 else 'MODERATE' if network_equity['gini_closeness'] > 0.2 else 'LOW'}

Income Equity:
• High-to-Low Time Ratio: {spatial_equity['income_time_ratio']:.2f}x
• High-to-Low Cost Ratio: {spatial_equity['income_cost_ratio']:.2f}x

Network Coverage: {network_equity['network_coverage']*100:.1f}%
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax6.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return overall_score

# Usage example for degree comparison
def compare_degree_equity(degrees=[3, 5, 7], db_connection_string=None):
    """Compare equity metrics across different degree constraints"""
    results = {}
    
    for degree in degrees:
        print(f"\n{'='*50}")
        print(f"ANALYZING DEGREE-{degree} NETWORK")
        print(f"{'='*50}")
        
        # You would need to run separate simulations for each degree
        # For now, this shows the framework
        # network_manager = TwoLayerNetworkManager(
        #     topology_type="degree_constrained",
        #     degree=degree,
        #     grid_width=100,
        #     grid_height=80
        # )
        
        # analyzer = TransportEquityAnalyzer(db_connection_string, network_manager)
        # results[degree] = analyzer.comprehensive_equity_analysis()
    
    return results

if __name__ == "__main__":
    # Initialize analyzer with your network manager
    # analyzer = TransportEquityAnalyzer(db.DB_CONNECTION_STRING, network_manager)
    # metrics = analyzer.comprehensive_equity_analysis()
    pass