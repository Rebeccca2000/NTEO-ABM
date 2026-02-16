# abm_initialization.py
"""
Modified ABM Initialization - Uses Network Factory Pattern
Eliminates dual routing system and uses only Network Topology System (Two Layers).
FIXED VERSION with complete trip processing logic.
"""

from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker, scoped_session
import config.database_updated as db  # Use updated database without network config
from ABM.agent_service_provider_initialisation import reset_database
from ABM.agent_service_provider_initialisation import TABLE_PREFIX
from config.network_factory import StandardizedNetworkInterface
from config.network_config import NetworkConfigurationManager
from topology.unified_network_integration import TwoLayerNetworkManager
from topology.network_topology import SydneyNetworkTopology
from ABM.agent_commuter import Commuter  # Not CommuterAgent
from ABM.agent_MaaS import MaaS  # Not MaaSAgent  
from ABM.agent_service_provider import ServiceProvider
import numpy as np
import warnings
import math
import random
import statistics
from   topology.network_topology import NodeType

class StationAgent(Agent):
    """Mesa-compatible station agent for network nodes"""
    
    def __init__(self, unique_id, model, station_type="mixed", network_node_id=None):
        super().__init__(unique_id, model)  # Critical: Proper Mesa Agent initialization
        self.station_type = station_type
        self.network_node_id = network_node_id
        self.agent_type = "station"
        # Note: pos attribute will be set automatically by grid.place_agent()
    
    def get_current_step(self):
        """Get current simulation step - required by MaaS agent"""
        return self.model.current_step
    
    def step(self):
        """Required Mesa agent step method"""
        pass

class MobilityModelNTEO(Model):
    """
    Modified Mobility Model that uses ONLY the Network Topology System.
    Eliminates the dual routing system as per the construction guide.
    FIXED VERSION with complete simulation logic.
    """
    
    def __init__(self, db_connection_string, num_commuters, income_weights, health_weights, 
                payment_weights, disability_weights, tech_access_weights, age_distribution, 
                penalty_coefficients, affordability_thresholds, value_of_time, 
                flexibility_adjustments, asc_values, utility_function_base_coefficients, 
                utility_function_high_income_car_coefficients, public_price_table, 
                alpha_values, dynamic_maas_surcharge_base_coefficients, 
                background_traffic_amount, congestion_alpha, congestion_beta, 
                congestion_capacity, congestion_t_ij_free_flow, 
                uber_like1_capacity, uber_like1_price, uber_like2_capacity, uber_like2_price, 
                bike_share1_capacity, bike_share1_price, bike_share2_capacity, bike_share2_price, 
                subsidy_dataset, subsidy_config, network_config=None, schema=None):
        
        print("🚀 Initializing NTEO Mobility Model with Network Factory")
        print("="*60)
        
        # ===== DATABASE SETUP =====
        self.current_step = 0  # Initialize step counter to 0
        self.db_engine = create_engine(db_connection_string)
        self.db_connection_string = db_connection_string  # Store for agents
        self.schema = schema
        # Add after database engine setup:
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        self.session = self.Session()

        # Reset database (like your working ABM)
        reset_database(self.db_engine, self.session,
            uber_like1_capacity, uber_like1_price, 
            uber_like2_capacity, uber_like2_price, 
            bike_share1_capacity, bike_share1_price, 
            bike_share2_capacity, bike_share2_price, self.schema)
        # Initialize random number generator
        super().__init__(seed=None)
        
        # ===== NETWORK TOPOLOGY SYSTEM =====
        print("\n🔧 Network Topology Initialization")
        print("-"*40)
        
        # Determine topology type and parameter from network_config
        if network_config:
            topology_type = network_config.get('topology_type', 'degree_constrained')
            variation_parameter = self._extract_variation_parameter(network_config, topology_type)
        else:
            topology_type = 'degree_constrained'
            variation_parameter = 4
        
        # Use the convenience function from network_factory module
        print(f"🌐 Creating {topology_type} network topology...")
        from config.network_factory import create_network
        self.network_interface = create_network(
            network_type=topology_type,
            variation_parameter=variation_parameter,
            **{k: v for k, v in (network_config or {}).items() if k not in ['topology_type']}
        )
        
        

        # Extract network manager from interface and add required methods
        self.network_manager = self.network_interface.network_manager
        self.debug_network_edges()
        
        if not hasattr(self.network_manager, 'record_trip_on_network'):
            def record_trip_on_network(origin, destination, start_time, duration):
                # Simple implementation for congestion tracking
                try:
                    route = self.network_interface.find_route(origin, destination)
                    if route:
                        self.network_interface.update_congestion(route, f"trip_{start_time}", True)
                except:
                    pass
            
            self.network_manager.record_trip_on_network = record_trip_on_network
        
        print("✅ Network created successfully!")
        print(f"   📊 Network Statistics:")
        print(f"   - Nodes: {self.network_interface.get_network_stats()['num_nodes']}")
        print(f"   - Edges: {self.network_interface.get_network_stats()['num_edges']}")
        print(f"   - Connected: {self.network_interface.get_network_stats()['is_connected']}")

        # ADD THIS DEBUG CODE HERE:
        print(f"   🔍 Checking route_id data in edges...")
        edge_count = 0
        edges_with_route_id = 0
        edges_without_route_id = 0

        for edge in self.network_manager.active_network.edges(data=True):
            from_node, to_node, data = edge
            edge_count += 1
            if 'route_id' in data:
                edges_with_route_id += 1
            
            else:
                edges_without_route_id += 1
                print(f"❌ ERROR: Edge {from_node}->{to_node} missing route_id")
                print(f"   Available keys: {list(data.keys())}")
            
            # Only show first 10 edges to avoid spam
            if edge_count >= 10:
                break

        print(f"   📊 Route ID Summary:")
        print(f"   - Edges with route_id: {edges_with_route_id}")
        print(f"   - Edges without route_id: {edges_without_route_id}")
        # Extract topology information for summary
        self.topology_type = topology_type
        self.variation_parameter = variation_parameter
        
        # ===== GRID SETUP =====
        self.grid = MultiGrid(100, 80, True)  # Standard Sydney grid size
        
        # ===== SIMULATION TRACKING =====
        self.current_step = 0
        self.maas_agent = None  # Reference to main MaaS agent
        
        # ===== AGENT ATTRIBUTES CONFIGURATION =====
        self.num_commuters = num_commuters
        self.income_weights = income_weights
        self.health_weights = health_weights
        self.payment_weights = payment_weights
        self.disability_weights = disability_weights
        self.tech_access_weights = tech_access_weights
        self.age_distribution = age_distribution
        
        # ===== MODEL PARAMETERS =====
        self.penalty_coefficients = penalty_coefficients
        self.affordability_thresholds = affordability_thresholds
        self.value_of_time = value_of_time
        self.flexibility_adjustments = flexibility_adjustments
        self.asc_values = asc_values
        self.utility_function_base_coefficients = utility_function_base_coefficients
        self.utility_function_high_income_car_coefficients = utility_function_high_income_car_coefficients
        self.public_price_table = public_price_table
        self.alpha_values = alpha_values
        self.dynamic_maas_surcharge_base_coefficients = dynamic_maas_surcharge_base_coefficients
        
        # ===== CONGESTION PARAMETERS =====
        self.background_traffic_amount = background_traffic_amount
        self.congestion_alpha = congestion_alpha
        self.congestion_beta = congestion_beta
        self.congestion_capacity = congestion_capacity
        self.congestion_t_ij_free_flow = congestion_t_ij_free_flow
        
        # ===== SERVICE CAPACITIES AND PRICES =====
        self.uber_like1_capacity = uber_like1_capacity
        self.uber_like1_price = uber_like1_price
        self.uber_like2_capacity = uber_like2_capacity
        self.uber_like2_price = uber_like2_price
        self.bike_share1_capacity = bike_share1_capacity
        self.bike_share1_price = bike_share1_price
        self.bike_share2_capacity = bike_share2_capacity
        self.bike_share2_price = bike_share2_price
        
        # ===== SUBSIDY CONFIGURATION =====
        self.subsidy_dataset = subsidy_dataset
        self.subsidy_config = subsidy_config
        
        # ===== AGENT SCHEDULERS =====
        self.schedule_commuters = RandomActivation(self)
        self.schedule_stations = RandomActivation(self)
        self.schedule_maas = RandomActivation(self)
        
        # ===== INITIALIZE AGENTS =====
        print(f"\n👥 Initializing {num_commuters} commuter agents...")
        self._create_commuter_agents()
        self.log_commuter_info_to_database()  # ← MISSING DATABASE POPULATION

        print(f"🚉 Initializing station agents...")
        self._create_station_agents()
        
        print(f"🏢 Initializing service provider...")
        self._create_service_provider_agent()
        
        print(f"🚗 Initializing MaaS service agents...")
        self._create_maas_agents()
        
        # ===== ENSURE COMMUTER ATTRIBUTES =====
        for commuter in self.commuter_agents:
            if not hasattr(commuter, 'requests'):
                commuter.requests = {}
            if not hasattr(commuter, 'location'):
                commuter.location = (self.random.randrange(100), self.random.randrange(80))
        
        # ===== DATA COLLECTION =====
        self.setup_data_collection()
        
        # ===== PERFORMANCE TRACKING =====
        self.route_calculation_count = 0
        self.total_route_calculation_time = 0
        


    def debug_network_edges(self):
        """Debug: Check all edges for required attributes"""
      
        missing_route_id = []
        topology_edges = []
        base_edges = []
        
        for u, v, data in self.network_manager.active_network.edges(data=True):
            # Check for route_id
            if 'route_id' not in data:
                missing_route_id.append((u, v))
                print(f"❌ Missing route_id: {u}->{v}")
                print(f"   Available attrs: {list(data.keys())}")
            else:
                # Categorize by edge type
                route_id = data['route_id']
                if any(x in route_id for x in ['T1_', 'T4_', 'BUS_380']):
                    base_edges.append((u, v, route_id))
                else:
                    topology_edges.append((u, v, route_id))
                 
        

        


    def _extract_variation_parameter(self, network_config, topology_type):
        """Extract the variation parameter for the given topology type"""
        if topology_type == 'degree_constrained':
            return network_config.get('degree_constraint', 3)
        elif topology_type == 'small_world':
            return network_config.get('rewiring_probability', 0.1)
        elif topology_type == 'scale_free':
            return network_config.get('attachment_parameter', 2)
        elif topology_type == 'base_sydney':
            return network_config.get('connectivity_level', 6)
        elif topology_type == 'grid':
            return network_config.get('grid_connectivity', 4)
        else:
            return None
    
    def _create_commuter_agents(self):
        """Create commuter agents with HILDA-based spatial demographic distribution"""
        self.commuter_agents = []
        
        # HILDA-BASED SPATIAL DEMOGRAPHICS FOR SYDNEY
        # Based on HILDA Survey 2021 spatial income distribution
        sydney_spatial_demographics = {
            # High income areas (Eastern suburbs, North Shore, Inner West premium)
            'high_income_zones': {
                'grid_ranges': [
                    [(55, 85), (45, 75)],  # Eastern suburbs & North Shore
                    [(45, 55), (50, 65)],  # Premium Inner West
                    [(40, 50), (55, 70)]   # Upper North Shore
                ],
                'income_weights': [0.15, 0.35, 0.50],  # low, middle, high
                'population_density': 0.25
            },
            # Middle income areas (Inner suburbs, Sutherland, Hills)
            'middle_income_zones': {
                'grid_ranges': [
                    [(45, 55), (35, 50)],  # Inner Sydney
                    [(50, 70), (15, 35)],  # Sutherland Shire
                    [(55, 75), (65, 80)]   # Hills District
                ],
                'income_weights': [0.25, 0.55, 0.20],
                'population_density': 0.35
            },
            # Lower income areas (Western Sydney, South-West)  
            'low_income_zones': {
                'grid_ranges': [
                    [(0, 35), (20, 60)],   # Western Sydney
                    [(35, 50), (0, 25)],   # South-West Sydney
                    [(70, 85), (0, 25)]    # Far South
                ],
                'income_weights': [0.60, 0.30, 0.10],
                'population_density': 0.40
            }
        }
        
        for i in range(self.num_commuters):
            # SPATIAL DEMOGRAPHIC SAMPLING
            zone_type = self.random.choices(
                ['high_income_zones', 'middle_income_zones', 'low_income_zones'],
                weights=[0.25, 0.35, 0.40]  # Population distribution across zones
            )[0]
            
            zone_info = sydney_spatial_demographics[zone_type]
            
            # Sample location within chosen zone
            # grid_range = self.random.choice(zone_info['grid_ranges'])
            # x_range, y_range = grid_range
            # x = self.random.randint(x_range[0], min(x_range[1], self.grid.width-1))
            # y = self.random.randint(y_range[0], min(y_range[1], self.grid.height-1))
            # commuter_location = (x, y)
            grid_range = self.random.choice(zone_info['grid_ranges'])
            x_range, y_range = grid_range
            # ===== NEW: GET STATIONS FROM EXISTING NETWORK =====
            major_stations = {}
            transport_stations = {}

            # Extract from your network manager
            if hasattr(self, 'network_manager') and self.network_manager and hasattr(self.network_manager, 'base_network'):
                for node_id, node_data in self.network_manager.base_network.nodes.items():
                    if node_data.node_type == NodeType.MAJOR_HUB:
                        major_stations[node_id] = node_data.coordinates
                    elif node_data.node_type == NodeType.TRANSPORT_HUB:
                        transport_stations[node_id] = node_data.coordinates

            # Combine for access weighting (ALL important stations)
            all_important_stations = {**major_stations, **transport_stations}

            if i == 0:  # Print only once
                print(f"   📍 Found {len(major_stations)} major hubs and {len(transport_stations)} transport hubs for access weighting")

            # ===== NEW: TRANSPORT ACCESS WEIGHTED ORIGIN SELECTION =====
            if all_important_stations:
                # Generate multiple candidate locations and weight by station access
                station_weights = []
                candidate_locations = []
                
                for _ in range(10):  # Try 10 candidate locations
                    x = self.random.randint(x_range[0], min(x_range[1], self.grid.width-1))
                    y = self.random.randint(y_range[0], min(y_range[1], self.grid.height-1))
                    
                    # Calculate access weight using ALL your defined stations
                    min_station_dist = min(
                        ((x-sx)**2 + (y-sy)**2)**0.5 
                        for sx, sy in all_important_stations.values()
                    )
                    
                    # Higher weight = closer to station
                    access_weight = max(0.1, 1.0 / (1.0 + min_station_dist/10))
                    
                    # Check if closest station is a major hub (extra weight)
                    if major_stations:
                        closest_major_dist = min(
                            ((x-sx)**2 + (y-sy)**2)**0.5 
                            for sx, sy in major_stations.values()
                        )
                        if closest_major_dist <= min_station_dist + 2:  # Close to major hub
                            access_weight *= 1.5  # Bonus for major hubs
                    
                    candidate_locations.append((x, y))
                    station_weights.append(access_weight)
                
                # Choose location with station access bias
                commuter_location = self.random.choices(candidate_locations, weights=station_weights)[0]
            else:
                # Fallback to random if network not available
                x = self.random.randint(x_range[0], min(x_range[1], self.grid.width-1))
                y = self.random.randint(y_range[0], min(y_range[1], self.grid.height-1))
                commuter_location = (x, y)
            # Sample demographics based on spatial zone
            age = self._sample_age()
            income_level = self.random.choices(
                ['low', 'middle', 'high'], 
                weights=zone_info['income_weights']
            )[0]
            health_status = self._sample_health_status()
            payment_scheme = self._sample_payment_method()
            has_disability = self._sample_disability_status()
            tech_access = self._sample_tech_access()
            
            # Create commuter agent (rest unchanged)
            commuter = Commuter(
                unique_id=i + 2,
                model=self,
                commuter_location=commuter_location,
                age=age,
                income_level=income_level,
                has_disability=has_disability,
                tech_access=tech_access,
                health_status=health_status,
                payment_scheme=payment_scheme,
                ASC_VALUES=self.asc_values,
                UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS=self.utility_function_high_income_car_coefficients,
                UTILITY_FUNCTION_BASE_COEFFICIENTS=self.utility_function_base_coefficients,
                PENALTY_COEFFICIENTS=self.penalty_coefficients,
                AFFORDABILITY_THRESHOLDS=self.affordability_thresholds,
                FLEXIBILITY_ADJUSTMENTS=self.flexibility_adjustments,
                VALUE_OF_TIME=self.value_of_time,
                subsidy_dataset=self.subsidy_dataset
            )
            
            self.schedule_commuters.add(commuter)
            self.commuter_agents.append(commuter)
            self.grid.place_agent(commuter, commuter_location)
        
       
    
    def _create_station_agents(self):
        """Create station agents at network nodes"""
        
        
        self.station_agents = []
        node_count = 0
        
        try:
            # Get nodes from network manager
            network_nodes = list(self.network_manager.active_network.nodes())
        
            
            for node_id in network_nodes[:10]:  # Start with just 10 stations to test
                try:
                    # Get grid coordinates for this node
                    grid_pos = self.network_manager.spatial_mapper.node_to_grid.get(node_id, (50, 40))
                    
                    # Create simple station agent
                    station = StationAgent(
                        unique_id=f"station_{node_id}",
                        model=self,
                        station_type="mixed",
                        network_node_id=node_id
                    )
                    
                    self.station_agents.append(station)
                    self.schedule_stations.add(station)
                    self.grid.place_agent(station, grid_pos)
                    node_count += 1
                    
                    
                except Exception as e:
                    print(f"❌ [WARNING] Failed to create station for node {node_id}: {e}")
                    continue
        
        except Exception as e:
            print(f"❌ [ERROR] Failed to access network nodes: {e}")
            return
        
      
    
    def _create_service_provider_agent(self):
        """Create the service provider agent using original constructor"""
        self.service_provider_agent = ServiceProvider(
            unique_id="service_provider",
            model=self,
            db_connection_string=self.db_connection_string,
            public_price_table=self.public_price_table,
            ALPHA_VALUES=self.alpha_values,
            bike_speed=getattr(self, 'bike_speed', 2.0),
            schema=self.schema
        )
        
        self.schedule_maas.add(self.service_provider_agent)
       # FIX: Ensure shared services are initialized for topology system
        self.service_provider_agent.initialize_availability(0)  # Initialize at step 0
        
        # VERIFY: Check if shared services are properly set up
        with self.service_provider_agent.Session() as session:
            for table_name in ['UberLike1', 'UberLike2', 'BikeShare1', 'BikeShare2']:
                table_class = self.service_provider_agent.get_service_table(table_name)
                count = session.query(table_class).count()
                print(f"[DEBUG] {table_name} has {count} rows after initialization")
        
     
    
    def _create_maas_agents(self):
        """Create a single properly configured MaaS agent like in the working version"""
        # Create main MaaS agent with all required parameters
        self.maas_agent = MaaS(
            unique_id="maas_main",
            model=self,
            service_provider_agent=self.service_provider_agent,
            commuter_agents=self.commuter_agents,
            network_manager=self.network_manager,
            DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=self.dynamic_maas_surcharge_base_coefficients,
            BACKGROUND_TRAFFIC_AMOUNT=self.background_traffic_amount,
            stations=getattr(self, 'stations', []),  # Legacy compatibility
            routes=getattr(self, 'routes', []),      # Legacy compatibility  
            transfers=getattr(self, 'transfers', []), # Legacy compatibility
            num_commuters=self.num_commuters,
            grid_width=100,  # Use grid dimensions
            grid_height=80,
            CONGESTION_ALPHA=self.congestion_alpha,
            CONGESTION_BETA=self.congestion_beta,
            CONGESTION_CAPACITY=self.congestion_capacity,
            CONGESTION_T_IJ_FREE_FLOW=self.congestion_t_ij_free_flow,
            subsidy_config=self.subsidy_config,
            schema=self.schema
        )
        
        self.schedule_maas.add(self.maas_agent)
     
    
    def log_commuter_info_to_database(self):
        """Log all commuter information to database with robust location handling"""
        try:
            session = self.Session()
            
            # Import the table class
            from ABM.agent_service_provider_initialisation import CommuterInfoLog
            
            # Clear existing records first
            session.query(CommuterInfoLog).delete()
            
            # Log each commuter's info
            for commuter in self.commuter_agents:
                # Use commuter location directly (we know it exists from debug output)
                location_x, location_y = float(commuter.location[0]), float(commuter.location[1])
                
                commuter_info = CommuterInfoLog(
                    commuter_id=commuter.unique_id,
                    location_x=location_x,
                    location_y=location_y,
                    age=commuter.age,
                    income_level=commuter.income_level,
                    has_disability=commuter.has_disability,
                    tech_access=commuter.tech_access,
                    health_status=commuter.health_status,
                    payment_scheme=commuter.payment_scheme
                )
                session.add(commuter_info)
            
            session.commit()
            session.close()
            
            
        except Exception as e:
            print(f"❌ Error logging commuter info: {e}")
            print(f"[DEBUG] Full error details:")
            import traceback
            traceback.print_exc()
            if 'session' in locals():
                session.close()


    def debug_database_tables(self):
        """Debug function to check database table contents"""
        try:
            session = self.Session()
            
            if self.schema:
                # Check both tables using raw SQL (apply TABLE_PREFIX)
                commuter_table = f"{TABLE_PREFIX}commuter_info_log"
                booking_table = f"{TABLE_PREFIX}service_booking_log"
                commuter_count = session.execute(text(f"SELECT COUNT(*) FROM {commuter_table}")).fetchone()[0]
                booking_count = session.execute(text(f"SELECT COUNT(*) FROM {booking_table}")).fetchone()[0]
                
            
            else:
                # Check using ORM
                from ABM.agent_service_provider_initialisation import ServiceBookingLog, CommuterInfoLog
                commuter_count = session.query(CommuterInfoLog).count()
                booking_count = session.query(ServiceBookingLog).count()
                
            
            session.close()
            
        except Exception as e:
            print(f"❌ Debug error: {e}")
            if 'session' in locals():
                session.close()
    def setup_data_collection(self):
        """Setup data collection for the model"""
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Commuters": lambda m: m.schedule_commuters.get_agent_count(),
                "Total_Stations": lambda m: m.schedule_stations.get_agent_count(),
                "Network_Nodes": lambda m: len(list(m.network_manager.active_network.nodes())),
                "Network_Edges": lambda m: len(list(m.network_manager.active_network.edges())),
                "Route_Calculations": lambda m: m.route_calculation_count
            },
            agent_reporters={
                "Location": lambda a: a.pos if hasattr(a, 'pos') else None,
                "Type": lambda a: type(a).__name__
            }
        )
    
    # ===== TRIP GENERATION AND PROCESSING METHODS =====
    def should_create_trip(self, commuter, current_step):
        """Enhanced probability-based trip generation"""
        ticks_in_day = 144
        current_day_tick = current_step % ticks_in_day
        current_day = current_step // ticks_in_day
        day_of_week = current_day % 7
        is_weekend = day_of_week >= 5
        
        # Base probability (increase from your current 0.4)
        base_probability = 0.6  # Higher baseline
        
        # Demographic multipliers
        if commuter.income_level == 'high':
            base_probability *= 1.4
        elif commuter.income_level == 'low':
            base_probability *= 0.9
            
        # Age/accessibility adjustments
        if commuter.age >= 65 or commuter.has_disability:
            base_probability *= 0.8
        
        # Time-based multipliers (this is key for realism)
        if not is_weekend:
            # Morning peak
            if 40 <= current_day_tick <= 56:  # 7:30-9:20 AM
                time_multiplier = 2.5
            # Evening peak  
            elif 96 <= current_day_tick <= 112:  # 4:40-6:40 PM
                time_multiplier = 2.0
            # Midday
            elif 60 <= current_day_tick <= 90:
                time_multiplier = 1.2
            else:
                time_multiplier = 0.6
        else:
            # Weekend midday peak
            if 60 <= current_day_tick <= 90:
                time_multiplier = 1.5
            else:
                time_multiplier = 0.8
        
        # Remove active travel blocking - allow overlapping trips
        # People often have multiple commitments
        
        final_probability = base_probability * time_multiplier
        return self.random.random() < final_probability

    def get_time_period(self, day_tick):
        """Get time period for given day tick"""
        if 36 <= day_tick < 60:
            return 'morning_peak'
        elif 60 <= day_tick < 90:
            return 'midday'
        elif 90 <= day_tick < 114:
            return 'evening_peak'
        elif 114 <= day_tick < 140:
            return 'evening'
        else:
            return 'night'

    def all_requests_finished(self, commuter):
        """Check if all commuter requests are finished"""
        return all(request['status'] in ['finished', 'expired'] 
                  for request in commuter.requests.values())

    def create_time_based_trip(self, current_step, commuter):
        """Create trip requests with realistic timing patterns"""
        if commuter.requests:
            has_active = any(r['status'] in ['active', 'Service Selected'] 
                            for r in commuter.requests.values())
            if has_active:
                return False
        
        # Only check trip creation periodically
        if (current_step + hash(commuter.unique_id)) % 2 != 0:
            return False
        
        # Only create new trips if commuter isn't already traveling
        if not commuter.requests or self.all_requests_finished(commuter):
            if not self.should_create_trip(commuter, current_step):
                
                return False
                
            # Get time context
            ticks_in_day = 144
            current_day_tick = current_step % ticks_in_day
            current_day = current_step // ticks_in_day
            day_of_week = current_day % 7
            is_weekend = day_of_week >= 5
            
            # Calculate trip probability
            base_probability = 0.45
            
            if commuter.income_level == 'high':
                base_probability *= 1.5
            elif commuter.income_level == 'low':
                base_probability *= 0.8
                
            if commuter.age >= 65 or commuter.has_disability:
                base_probability *= 0.7
                
            if commuter.payment_scheme == 'subscription':
                base_probability *= 1.3
            
            # Time-based multiplier
            time_multiplier = 1.0
            
            if not is_weekend:
                # Morning peak (8am = tick 48)
                morning_peak_center = 48
                morning_intensity = math.exp(-0.5 * ((current_day_tick - morning_peak_center) / 8) ** 2)
                
                # Evening peak (5:30pm = tick 105)
                evening_peak_center = 105
                evening_intensity = math.exp(-0.5 * ((current_day_tick - evening_peak_center) / 10) ** 2)
                
                time_multiplier = max(morning_intensity * 3.0, evening_intensity * 2.5, 0.2)
            else:
                # Weekend midday peak (noon = tick 72)
                midday_peak_center = 72
                midday_intensity = math.exp(-0.5 * ((current_day_tick - midday_peak_center) / 16) ** 2)
                time_multiplier = midday_intensity * 1.5
            
            # Final probability
            trip_probability = base_probability * time_multiplier
            
            if self.random.random() < trip_probability:
                # Determine trip purpose
                if not is_weekend and current_day_tick < 60:  # Morning weekday
                    purpose_weights = {'work': 0.7, 'school': 0.2, 'shopping': 0.05, 'medical': 0.03, 'leisure': 0.02}
                elif not is_weekend and current_day_tick >= 90 and current_day_tick < 114:  # Evening weekday
                    purpose_weights = {'work': 0.1, 'school': 0.05, 'shopping': 0.3, 'leisure': 0.5, 'medical': 0.05}
                elif is_weekend:
                    purpose_weights = {'shopping': 0.4, 'leisure': 0.4, 'medical': 0.05, 'work': 0.1, 'school': 0.05}
                else:
                    purpose_weights = {'work': 0.2, 'school': 0.1, 'shopping': 0.3, 'medical': 0.2, 'leisure': 0.2}
                
                purposes = list(purpose_weights.keys())
                weights = list(purpose_weights.values())
                travel_purpose = self.random.choices(purposes, weights=weights)[0]
                
                # Generate destination
                origin = commuter.location if hasattr(commuter, 'location') else (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height))
                destination = self.get_purpose_based_destination(travel_purpose, origin, commuter)
                
                # Set timing
                min_delay = 1
                max_delay = 5
                start_time = current_step + min_delay + self.random.randint(0, max_delay)
                
                if start_time <= current_step + 5:
                    # Create request
                    # Create request using the proper ABM method
                    request_id = f"{commuter.unique_id}_{current_step}_{self.random.randint(1000, 9999)}"
                    commuter.create_request(request_id, origin, destination, start_time, travel_purpose)
                    return True
        return False

    def get_purpose_based_destination(self, purpose, origin, commuter):
        """Generate destination with return trips and trip chaining logic"""
        
        # ===== ESTABLISH HOME LOCATION (FIRST TIME ONLY) =====
        if not hasattr(commuter, '_home_location'):
            commuter._home_location = commuter.location  # Where they started = home
        
        # ===== GET DISTRICTS FROM EXISTING NETWORK =====
        districts = {}
        
        if hasattr(self, 'network_manager') and self.network_manager and hasattr(self.network_manager, 'base_network'):
            # Major employment centers
            for node_id, node_data in self.network_manager.base_network.nodes.items():
                if node_data.node_type == NodeType.MAJOR_HUB:
                    districts[node_id] = {
                        'center': node_data.coordinates,
                        'attraction': {
                            'work': node_data.employment_weight,
                            'other': node_data.population_weight * 0.8
                        },
                        'radius': 8,
                        'type': 'major'
                    }
            
            # Secondary centers
            important_transport_hubs = ['BONDI_JUNCTION', 'NORTH_SYDNEY', 'HORNSBY', 'BLACKTOWN', 'HURSTVILLE']
            for node_id, node_data in self.network_manager.base_network.nodes.items():
                if (node_data.node_type == NodeType.TRANSPORT_HUB and 
                    node_id in important_transport_hubs):
                    districts[node_id] = {
                        'center': node_data.coordinates,
                        'attraction': {
                            'work': node_data.employment_weight * 0.7,
                            'other': node_data.population_weight * 1.2
                        },
                        'radius': 6,
                        'type': 'secondary'
                    }
        
        # Fallback districts
        if not districts:
            districts = {
                'CBD': {'center': (50, 40), 'attraction': {'work': 0.8, 'other': 0.3}, 'radius': 8, 'type': 'major'},
                'PARRAMATTA': {'center': (25, 30), 'attraction': {'work': 0.6, 'other': 0.4}, 'radius': 6, 'type': 'major'},
            }
        
        # ===== DISTANCE CALCULATION HELPER =====
        def distance_between(loc1, loc2):
            return ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5
        
        # ===== RETURN TRIP DETECTION =====
        home_distance = distance_between(origin, commuter._home_location)
        
        # Check if currently at work
        at_work = False
        if hasattr(commuter, '_work_location'):
            work_distance = distance_between(origin, commuter._work_location)
            at_work = work_distance <= 5  # Within 5 units of work = at work
        
        # Check if far from home
        far_from_home = home_distance > 10
        
        # ===== TIME-CONTEXT DETECTION =====
        current_step = getattr(self, 'schedule', {})
        if hasattr(current_step, 'steps'):
            step_num = current_step.steps
        else:
            step_num = 0
        
        ticks_in_day = 144
        current_day_tick = step_num % ticks_in_day
        
        # Time periods
        is_morning_peak = 40 <= current_day_tick <= 60   # 7:30-10:00 AM
        is_evening_peak = 90 <= current_day_tick <= 120  # 4:30-8:00 PM
        is_late_evening = current_day_tick > 120         # After 8 PM
        
        # ===== RETURN HOME LOGIC =====
        # High probability of returning home if:
        # 1. Evening/late evening AND far from home
        # 2. At work during evening peak
        # 3. Purpose is explicitly work but it's evening (return commute)
        
        return_home_probability = 0.0
        
        if far_from_home:
            if is_evening_peak:
                return_home_probability = 0.7  # High chance in evening
            elif is_late_evening:
                return_home_probability = 0.9  # Very high chance late evening
            elif at_work and current_day_tick > 85:  # At work after 4 PM
                return_home_probability = 0.8
        
        # Override for evening "work" trips (these are returns)
        if purpose == 'work' and is_evening_peak and far_from_home:
            return_home_probability = 0.85
        
        # Return home if triggered
        if self.random.random() < return_home_probability:
            return commuter._home_location
        
        # ===== WORK DESTINATION (CONSISTENT) =====
        if purpose in ['work', 'school'] and not is_evening_peak:
            if not hasattr(commuter, '_work_location'):
                # First-time work destination assignment
                district_probs = []
                district_names = []
                
                for district_name, district_info in districts.items():
                    prob = district_info['attraction']['work']
                    
                    # Distance decay from home (not current location)
                    dist = distance_between(commuter._home_location, district_info['center'])
                    prob = prob / (1.0 + dist/50)
                    
                    # Demographic preferences
                    if commuter.income_level == 'high':
                        if district_name in ['CENTRAL', 'CHATSWOOD', 'NORTH_SYDNEY']:
                            prob *= 1.8
                    elif commuter.income_level == 'low':
                        if district_name in ['PARRAMATTA', 'BLACKTOWN', 'LIVERPOOL']:
                            prob *= 1.5
                    
                    district_probs.append(prob)
                    district_names.append(district_name)
                
                # Add local work option
                district_probs.append(0.3)
                district_names.append('LOCAL')
                
                # Select work district
                chosen_district = self.random.choices(district_names, weights=district_probs)[0]
                
                if chosen_district == 'LOCAL':
                    # Local work near home
                    angle = self.random.uniform(0, 2 * math.pi)
                    r = self.random.uniform(5, 15)
                    work_x = max(0, min(self.grid.width-1, int(commuter._home_location[0] + r * math.cos(angle))))
                    work_y = max(0, min(self.grid.height-1, int(commuter._home_location[1] + r * math.sin(angle))))
                    commuter._work_location = (work_x, work_y)
                else:
                    # Work in chosen district
                    district_info = districts[chosen_district]
                    center = district_info['center']
                    radius = district_info['radius']
                    
                    angle = self.random.uniform(0, 2 * math.pi)
                    r = self.random.uniform(0, radius)
                    
                    work_x = max(0, min(self.grid.width-1, int(center[0] + r * math.cos(angle))))
                    work_y = max(0, min(self.grid.height-1, int(center[1] + r * math.sin(angle))))
                    commuter._work_location = (work_x, work_y)
                
                commuter._work_district = chosen_district
            
            return commuter._work_location
        
        # ===== OTHER DESTINATIONS (SHOPPING, LEISURE, MEDICAL) =====
        else:
            # Add local area as strong option
            districts['LOCAL'] = {
                'center': origin,  # From current location
                'attraction': {'other': 0.6},
                'radius': 12,
                'type': 'local'
            }
            
            district_probs = []
            district_names = []
            
            for district_name, district_info in districts.items():
                prob = district_info['attraction'].get('other', 0.2)
                
                # Distance decay from current location (not home)
                dist = distance_between(origin, district_info['center'])
                
                if district_name == 'LOCAL':
                    prob = prob  # No distance decay for local
                else:
                    prob = prob / (1.0 + dist/25)  # Stronger preference for nearby
                
                # Time-based adjustments
                if district_name == 'LOCAL' and (is_evening_peak or is_late_evening):
                    prob *= 1.5  # Prefer local in evening
                
                district_probs.append(prob)
                district_names.append(district_name)
            
            # Select destination district
            chosen_district = self.random.choices(district_names, weights=district_probs)[0]
            district_info = districts[chosen_district]
            
            center = district_info['center']
            radius = district_info['radius']
            
            angle = self.random.uniform(0, 2 * math.pi)
            r = self.random.uniform(0, radius)
            
            dest_x = max(0, min(self.grid.width-1, int(center[0] + r * math.cos(angle))))
            dest_y = max(0, min(self.grid.height-1, int(center[1] + r * math.sin(angle))))
            
            return (dest_x, dest_y)
    
    def debug_booking_status_sql(self):
        """Efficient SQL-based booking status debug - much better than Python loops!"""
        try:
            session = self.Session()
            
            if self.schema:
                # Raw SQL approach (when using schema) - apply TABLE_PREFIX
                booking_table = f"{TABLE_PREFIX}service_booking_log"
                commuter_table = f"{TABLE_PREFIX}commuter_info_log"
                booking_status_query = text(f"""
                    SELECT 
                        status,
                        COUNT(*) as count,
                        COUNT(DISTINCT commuter_id) as unique_commuters
                    FROM {booking_table}
                    GROUP BY status
                """)
                status_results = session.execute(booking_status_query).fetchall()
                
                # Total commuters vs commuters with bookings
                total_commuters_query = text(f"SELECT COUNT(*) FROM {commuter_table}")
                total_commuters = session.execute(total_commuters_query).fetchone()[0]
                
                commuters_with_bookings_query = text(f"SELECT COUNT(DISTINCT commuter_id) FROM {booking_table}")
                commuters_with_bookings = session.execute(commuters_with_bookings_query).fetchone()[0]
                
            else:
                # ORM approach
                from ABM.agent_service_provider_initialisation import ServiceBookingLog, CommuterInfoLog
                from sqlalchemy import func
                
                status_results = session.query(
                    ServiceBookingLog.status,
                    func.count(ServiceBookingLog.request_id).label('count'),
                    func.count(ServiceBookingLog.commuter_id.distinct()).label('unique_commuters')
                ).group_by(ServiceBookingLog.status).all()
                
                total_commuters = session.query(CommuterInfoLog).count()
                commuters_with_bookings = session.query(ServiceBookingLog.commuter_id.distinct()).count()
            
            # Print results
            print(f"📊 STEP {self.current_step} SQL BOOKING STATUS:")
            print(f"   - Total commuters in system: {total_commuters}")
            print(f"   - Commuters with booking attempts: {commuters_with_bookings}")
            print(f"   - Commuters with NO bookings: {total_commuters - commuters_with_bookings}")
            
            for status_row in status_results:
                if hasattr(status_row, '_mapping'):  # Raw SQL result
                    status, count, unique_commuters = status_row.status, status_row.count, status_row.unique_commuters
                else:  # ORM result
                    status, count, unique_commuters = status_row[0], status_row[1], status_row[2]
                print(f"   - {status}: {count} bookings ({unique_commuters} unique commuters)")
            
            session.close()
            
        except Exception as e:
            print(f"❌ SQL debug error: {e}")
            if 'session' in locals():
                session.close()

    def debug_network_topology_sql(self):
        """Debug network topology differences using SQL"""
        try:
            session = self.Session()
            
            # Get topology info from network manager
            network_stats = self.network_interface.get_network_stats()
            requested_topology = getattr(self, 'topology_type', 'unknown')
            actual_topology = network_stats.get('topology_type', 'unknown')
            
            print(f"🔍 NETWORK TOPOLOGY DEBUG:")
           
            
            # Check if routes are actually using the network topology
            if self.schema:
                booking_table = f"{TABLE_PREFIX}service_booking_log"
                route_analysis_query = text(f"""
                    SELECT 
                        COUNT(*) as total_routes,
                        AVG(CAST(total_time AS FLOAT)) as avg_travel_time,
                        COUNT(DISTINCT record_company_name) as unique_modes
                    FROM {booking_table} 
                    WHERE total_time IS NOT NULL
                """)
                route_stats = session.execute(route_analysis_query).fetchone()
                
                print(f"   - Total routes calculated: {route_stats.total_routes if route_stats else 0}")
                print(f"   - Average travel time: {route_stats.avg_travel_time:.2f}" if route_stats and route_stats.avg_travel_time else "   - Average travel time: N/A")
                print(f"   - Unique transport modes: {route_stats.unique_modes if route_stats else 0}")
            
            # Check for topology-specific routing patterns
            sample_edges = list(self.network_manager.active_network.edges(data=True))[:3]
            print(f"   - Sample edge data keys: {list(sample_edges[0][2].keys()) if sample_edges else 'No edges'}")
            
            session.close()
            
        except Exception as e:
            print(f"❌ Network topology debug error: {e}")
            if 'session' in locals():
                session.close()

    def debug_commuter_activity_sql(self):
        """Debug why commuters aren't creating/completing bookings"""
        try:
            session = self.Session()
            
            if self.schema:
                # Check commuter distribution by characteristics (apply TABLE_PREFIX)
                commuter_table = f"{TABLE_PREFIX}commuter_info_log"
                booking_table = f"{TABLE_PREFIX}service_booking_log"
                commuter_breakdown_query = text(f"""
                    SELECT 
                        income_level,
                        COUNT(*) as total_commuters,
                        COUNT(DISTINCT sbl.commuter_id) as commuters_with_bookings
                    FROM {commuter_table} cil
                    LEFT JOIN {booking_table} sbl ON cil.commuter_id = sbl.commuter_id
                    GROUP BY income_level
                """)
                commuter_breakdown = session.execute(commuter_breakdown_query).fetchall()
                
                for row in commuter_breakdown:
                    income = row.income_level
                    total = row.total_commuters
                    with_bookings = row.commuters_with_bookings or 0
                    booking_rate = (with_bookings / total * 100) if total > 0 else 0
                    
            
            session.close()
            
        except Exception as e:
            print(f"❌ Commuter activity debug error: {e}")
            if 'session' in locals():
                session.close()
    def step(self):
        """Execute one step of the model with full trip processing logic"""
      
        # Initialize availability if needed
        if hasattr(self, 'service_provider_agent'):
            self.service_provider_agent.initialize_availability(self.current_step)
        
        # Process commuters - separate active from inactive
        active_commuters = []
        inactive_commuters = []
        
        for commuter in self.commuter_agents:
            needs_processing = (
                any(r['status'] == 'active' and r['start_time'] <= self.current_step + 5 
                    for r in commuter.requests.values()) or
                any(r['status'] == 'Service Selected' for r in commuter.requests.values()) or
                (self.current_step + hash(commuter.unique_id)) % 5 == 0
            )
            
            if needs_processing:
                active_commuters.append(commuter)
            else:
                inactive_commuters.append(commuter)
        
        # Process active commuters
        for commuter in active_commuters:
            self.create_time_based_trip(self.current_step, commuter)
            
            # Clean up stale requests
            for request_id, request in list(commuter.requests.items()):
                if request['status'] == 'active' and request['start_time'] < self.current_step:
                    request['status'] = 'expired'
            
            # Process valid active requests
            for request_id, request in list(commuter.requests.items()):
                try:
                    if request['status'] == 'active' and request['start_time'] >= self.current_step:
                        # Use the main MaaS agent
                        if self.maas_agent:
                            # Process travel options
                            travel_options_without_MaaS = self.maas_agent.options_without_maas(
                                request_id, request['start_time'], request['origin'], request['destination'])
                            
                            travel_options_with_MaaS = self.maas_agent.maas_options(
                                commuter.payment_scheme, request_id, request['start_time'], 
                                request['origin'], request['destination'])
                            
                            ranked_options = commuter.rank_service_options(
                                travel_options_without_MaaS, travel_options_with_MaaS, request_id)
                            
                            if ranked_options:
                                # *** CRITICAL FIX: Get properly formatted availability_dict ***
                                current_availability_dict = self.service_provider_agent.initialize_availability(self.current_step)
                                print(f"[DEBUG] Using availability_dict: {current_availability_dict}")
                                
                                # Try to book service with correct availability_dict
                                booking_success, _ = self.maas_agent.book_service(
                                    request_id, ranked_options, self.current_step, current_availability_dict)
                                
                                # Track route calculations
                                self.route_calculation_count += 1
                                
                                if not booking_success:
                                    # Mark as expired if booking failed
                                    request['status'] = 'expired'
                                    print(f"[DEBUG] Booking failed for request {request_id}")
                                else:
                                    # Booking successful - status should be updated by book_service
                                    print(f"[DEBUG] Booking successful for request {request_id}")
                            else:
                                # No viable options available
                                request['status'] = 'expired'
                                print(f"[DEBUG] No viable options for request {request_id}")
                except Exception as e:
                    # Handle errors gracefully
                    request['status'] = 'expired'
                    print(f"❌ Error processing request {request_id}: {e}")
            
            # Update commuter location and status
            if hasattr(commuter, 'update_location'):
                commuter.update_location()
            if hasattr(commuter, 'check_travel_status'):
                commuter.check_travel_status()
        
        # Process inactive commuters
        for commuter in inactive_commuters:
            if any(r['status'] == 'Service Selected' for r in commuter.requests.values()):
                if hasattr(commuter, 'update_location'):
                    commuter.update_location()
        
        # Update service provider
        if hasattr(self, 'service_provider_agent'):
            self.service_provider_agent.update_time_steps()  # Add this line
            self.service_provider_agent.update_availability()
            if hasattr(self.service_provider_agent, 'dynamic_pricing_share'):
                self.service_provider_agent.dynamic_pricing_share()
        
        # Step schedulers for any remaining agent logic
        self.schedule_commuters.step()
        self.schedule_stations.step()
        self.schedule_maas.step()
        self.current_step += 1
        # ===== ADD SQL DEBUG CALLS HERE =====
        # Debug every 10 steps to avoid spam
        if self.current_step % 10 == 0:
            self.debug_booking_status_sql()
            self.debug_network_topology_sql()
        
        # Less frequent detailed analysis
        if self.current_step % 50 == 0:
            self.debug_commuter_activity_sql()
        
        # Collect data
        self.datacollector.collect(self)

    # ===== ROUTING INTERFACE (NETWORK TOPOLOGY ONLY) =====

    
    def update_congestion(self, route, agent_id, add=True):
        """Update congestion on network edges"""
        self.network_interface.update_congestion(route, agent_id, add)
    
    # ===== CALCULATION METHODS FOR RESEARCH =====

    def _calculate_equity_index(self):
        """Calculate transport equity index"""
        try:
            # Simple equity calculation based on successful bookings vs attempts
            total_requests = sum(len(commuter.requests) for commuter in self.commuter_agents)
            successful_bookings = sum(1 for commuter in self.commuter_agents 
                                     for request in commuter.requests.values() 
                                     if request['status'] == 'Service Selected')
            
            if total_requests == 0:
                return 0.0
            
            success_rate = successful_bookings / total_requests
            
            # Calculate inequality across income groups
            income_success = {'low': [], 'middle': [], 'high': []}
            for commuter in self.commuter_agents:
                if commuter.requests:
                    commuter_success = sum(1 for r in commuter.requests.values() 
                                         if r['status'] == 'Service Selected') / len(commuter.requests)
                    income_success[commuter.income_level].append(commuter_success)
            
            # Calculate coefficient of variation across income groups
            group_means = []
            for income_level, successes in income_success.items():
                if successes:
                    group_means.append(sum(successes) / len(successes))
            
            if len(group_means) > 1:
                cv = statistics.stdev(group_means) / max(statistics.mean(group_means), 0.001)
                equity_index = success_rate * (1 - cv)  # Higher equity = lower coefficient of variation
            else:
                equity_index = success_rate
            
            return max(0.0, min(equity_index, 1.0))
        except:
            return 0.0

    def get_current_step(self):
        """Get current simulation step"""
        return self.current_step
    
    def calculate_spatial_equity_distributions(self):
        """Calculate spatial equity using WORKING aggregate methods + commuter distribution"""
        
        # Define Sydney regions based on your actual network nodes
        sydney_regions = {
            'CBD_Inner': {
                'bounds': (45, 55, 35, 50),
                'major_nodes': ['CENTRAL', 'TOWN_HALL', 'WYNYARD', 'CIRCULAR_QUAY', 'MARTIN_PLACE'],
                'center': (50, 42)
            },
            'Western_Sydney': {
                'bounds': (0, 35, 15, 60), 
                'major_nodes': ['PARRAMATTA', 'BLACKTOWN', 'PENRITH', 'LIVERPOOL', 'BANKSTOWN'],
                'center': (20, 35)
            },
            'Eastern_Suburbs': {
                'bounds': (55, 85, 20, 65),
                'major_nodes': ['BONDI_JUNCTION', 'RANDWICK', 'MAROUBRA', 'EASTGARDENS'],
                'center': (68, 40)
            },
            'North_Shore': {
                'bounds': (45, 75, 50, 80),
                'major_nodes': ['CHATSWOOD', 'HORNSBY', 'NORTH_SYDNEY', 'CROWS_NEST', 'MANLY'],
                'center': (58, 65)
            },
            'South_Sydney': {
                'bounds': (30, 70, 0, 35),
                'major_nodes': ['SUTHERLAND', 'HURSTVILLE', 'KOGARAH', 'ROCKDALE'],
                'center': (50, 20)
            }
        }
        
        # Get WORKING aggregate metrics (we know these work!)
        global_mode_equity = self.calculate_mode_choice_equity()
        global_travel_equity = self.calculate_travel_time_equity()
        global_efficiency = self.calculate_system_efficiency()
        
        # Normalize mode equity (fix the 1.333 issue)
        if global_mode_equity > 1.0:
            global_mode_equity = global_mode_equity / 10.0  # Reasonable normalization
        
        # Get commuter spatial distribution
        regional_equity = {}
        total_commuters = len(self.commuter_agents)
        
        for region_name, region_info in sydney_regions.items():
            bounds = region_info['bounds']
            min_x, max_x, min_y, max_y = bounds
            
            # Count commuters in this region
            region_commuters = []
            for commuter in self.commuter_agents:
                # Use commuter.location if available, otherwise pos
                loc = getattr(commuter, 'location', getattr(commuter, 'pos', (50, 40)))
                x, y = loc[0], loc[1]
                
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    region_commuters.append(commuter)
            
            region_population = len(region_commuters)
            
            if region_population > 0:
                # Calculate regional fraction 
                population_fraction = region_population / total_commuters
                
                # Get regional activity level (completed trips)
                regional_trips = 0
                for commuter in region_commuters:
                    regional_trips += sum(1 for r in commuter.requests.values() 
                                        if r['status'] in ['Service Selected', 'finished'])
                
                # Calculate activity-adjusted metrics using working aggregate values
                activity_multiplier = max(0.1, min(2.0, regional_trips / max(region_population, 1)))
                
                # Regional demographic weights
                income_weights = {'low': 0, 'middle': 0, 'high': 0}
                for commuter in region_commuters:
                    income_weights[commuter.income_level] += 1
                
                # Income inequality factor (higher inequality = higher equity scores)
                total_regional = sum(income_weights.values())
                if total_regional > 0:
                    income_shares = [income_weights[level]/total_regional for level in ['low', 'middle', 'high']]
                    # Gini-like coefficient for income distribution
                    inequality_factor = 1.0 + 0.5 * np.std(income_shares)
                else:
                    inequality_factor = 1.0
                
                # Calculate realistic regional metrics
                regional_equity[region_name] = {
                    'mode_choice_equity': min(1.0, global_mode_equity * inequality_factor * activity_multiplier),
                    'travel_time_equity': global_travel_equity * population_fraction * activity_multiplier,
                    'system_efficiency': global_efficiency * population_fraction * activity_multiplier,
                    'commuter_count': region_population,
                    'activity_level': regional_trips,
                    'demographic_mix': income_weights
                }
            else:
                # Empty region
                regional_equity[region_name] = {
                    'mode_choice_equity': 0.0,
                    'travel_time_equity': 0.0,
                    'system_efficiency': 0.0,
                    'commuter_count': 0,
                    'activity_level': 0,
                    'demographic_mix': {'low': 0, 'middle': 0, 'high': 0}
                }
        
        return regional_equity

    def _calculate_regional_equity_metrics(self, region_commuters):
        """Calculate equity metrics for specific region"""
        try:
            # Mode Choice Equity for region
            income_mode_shares = {'low': {}, 'middle': {}, 'high': {}}
            
            for commuter in region_commuters:
                if commuter.requests:
                    for request in commuter.requests.values():
                        if request['status'] == 'Service Selected':
                            mode = request.get('service_type', 'unknown')
                            if mode not in income_mode_shares[commuter.income_level]:
                                income_mode_shares[commuter.income_level][mode] = 0
                            income_mode_shares[commuter.income_level][mode] += 1
            
            # Calculate mode choice equity (simplified version of your main calculation)
            mode_choice_equity = self._calculate_mode_choice_disparity(income_mode_shares)
            
            # Travel Time Equity for region  
            travel_times = []
            for commuter in region_commuters:
                if commuter.requests:
                    for request in commuter.requests.values():
                        if 'travel_time' in request:
                            travel_times.append(request['travel_time'])
            
            if travel_times:
                mean_travel_time = sum(travel_times) / len(travel_times)
                travel_time_equity = sum(abs(t - mean_travel_time) for t in travel_times)
            else:
                travel_time_equity = 0.0
                
            # System Efficiency for region
            total_travel_time = sum(travel_times) if travel_times else 0
            
            return {
                'mode_choice_equity': mode_choice_equity,
                'travel_time_equity': travel_time_equity, 
                'system_efficiency': total_travel_time,
                'commuter_count': len(region_commuters),
                'sample_income_distribution': {
                    income: len([c for c in region_commuters if c.income_level == income])
                    for income in ['low', 'middle', 'high']
                }
            }
            
        except Exception as e:
            print(f"Regional equity calculation failed: {e}")
            return {'error': str(e)}

    def _calculate_mode_choice_disparity(self, income_mode_shares):
        """Simplified mode choice disparity calculation for regions"""
        try:
            all_modes = set()
            for income_modes in income_mode_shares.values():
                all_modes.update(income_modes.keys())
            
            if not all_modes:
                return 0.0
                
            total_disparity = 0
            for mode in all_modes:
                mode_counts = [income_mode_shares[income].get(mode, 0) for income in ['low', 'middle', 'high']]
                if sum(mode_counts) > 0:
                    mode_shares = [count/max(sum(income_mode_shares[income].values()), 1) for income, count in zip(['low', 'middle', 'high'], mode_counts)]
                    mean_share = sum(mode_shares) / len(mode_shares)
                    total_disparity += sum(abs(share - mean_share) for share in mode_shares)
            
            return total_disparity
            
        except Exception as e:
            return 0.0
    
    # ===== AGENT ATTRIBUTE SAMPLING =====
    def _sample_income_level(self):
        """Sample income level based on weights"""
        levels = ['low', 'middle', 'high']
        return self.random.choices(levels, weights=self.income_weights)[0]
    
    def _sample_health_status(self):
        """Sample health/mobility status"""
        statuses = ['good', 'poor']
        return self.random.choices(statuses, weights=self.health_weights)[0]
    
    def _sample_payment_method(self):
        """Sample payment method"""
        methods = ['PAYG', 'subscription']
        return self.random.choices(methods, weights=self.payment_weights)[0]
    
    def _sample_disability_status(self):
        """Sample disability status"""
        return self.random.choices([True, False], weights=self.disability_weights)[0]
    
    def _sample_tech_access(self):
        """Sample technology access"""
        return self.random.choices([True, False], weights=self.tech_access_weights)[0]
    
    def _sample_age(self):
        """Sample age based on distribution"""
        age_ranges = list(self.age_distribution.keys())
        weights = list(self.age_distribution.values())
        selected_range = self.random.choices(age_ranges, weights=weights)[0]
        return self.random.randint(selected_range[0], selected_range[1])
    
    def get_model_summary(self):
        """Get summary of model configuration and status"""
        stats = self.network_interface.get_network_stats()
        
        return {
            'model_type': 'NTEO Mobility Model',
            'network_topology': self.topology_type,
            'network_stats': stats,
            'total_commuters': self.num_commuters,
            'total_stations': self.schedule_stations.get_agent_count(),
            'total_maas_services': self.schedule_maas.get_agent_count(),
            'routing_system': 'Network Topology Only (Dual System Eliminated)',
            'route_calculations': self.route_calculation_count,
            'avg_route_time': self.total_route_calculation_time / max(self.route_calculation_count, 1)
        }
    
    def calculate_mode_choice_equity(self):
        """
        Calculate Mode Choice Equity using the EXACT approach from mode_share_optimization.py
        """
        try:
            session = self.Session()


            # ADD THIS DEBUG BLOCK
            if self.schema:
                booking_table = f"{TABLE_PREFIX}service_booking_log"
                commuter_table = f"{TABLE_PREFIX}commuter_info_log"
                booking_count = session.execute(text(f"SELECT COUNT(*) FROM {booking_table}")).fetchone()[0]
                commuter_count = session.execute(text(f"SELECT COUNT(*) FROM {commuter_table}")).fetchone()[0]
            else:
                from ABM.agent_service_provider_initialisation import ServiceBookingLog, CommuterInfoLog
                booking_count = session.query(ServiceBookingLog).count()
                commuter_count = session.query(CommuterInfoLog).count()
  
            if commuter_count == 0:
                print(f"[DEBUG] ❌ CommuterInfoLog is EMPTY! This is why equity = 0")
                session.close()
                return 0.0
            
            income_levels = ['low', 'middle', 'high']
            
            # Get mode shares and trip counts for each income level (EXACT same logic)
            mode_shares = {}
            total_trips_by_income = {}
            all_trips = 0
            
        
            for income_level in income_levels:
                mode_shares[income_level] = {}
                
                # Use the EXACT same query pattern as your working code
                if self.schema:
                    booking_table = f"{TABLE_PREFIX}service_booking_log"
                    commuter_table = f"{TABLE_PREFIX}commuter_info_log"
                    query = text(f"""
                        SELECT record_company_name, COUNT(request_id) as count
                        FROM {booking_table} JOIN {commuter_table} 
                        ON {booking_table}.commuter_id = {commuter_table}.commuter_id
                        WHERE {commuter_table}.income_level = :income_level
                        GROUP BY record_company_name
                    """)
                    mode_choices = session.execute(query, {"income_level": income_level}).fetchall()
                else:
                    from ABM.agent_service_provider_initialisation import ServiceBookingLog, CommuterInfoLog
                    mode_choices = session.query(
                        ServiceBookingLog.record_company_name,
                        func.count(ServiceBookingLog.request_id).label('count')
                    ).join(
                        CommuterInfoLog,
                        ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
                    ).filter(
                        CommuterInfoLog.income_level == income_level
                    ).group_by(
                        ServiceBookingLog.record_company_name
                    ).all()

                # Calculate total trips for this income level (EXACT same logic)
                total_trips_by_income[income_level] = sum(choice[1] for choice in mode_choices) or 1
                all_trips += total_trips_by_income[income_level]
                
                # Calculate mode shares with EXACT same mode normalization
                for mode, count in mode_choices:
                    # EXACT same normalization logic as your working code
                    if 'Bike' in mode or mode == 'bike':
                        category = 'bike'
                    elif 'Uber' in mode or mode == 'car':
                        category = 'car'
                    elif mode == 'MaaS_Bundle':
                        category = 'MaaS'
                    elif mode == 'public':
                        category = 'public'
                    elif mode == 'walk':
                        category = 'walk'
                    else:
                        category = mode
                    
                    if category in mode_shares[income_level]:
                        mode_shares[income_level][category] += count / total_trips_by_income[income_level]
                    else:
                        mode_shares[income_level][category] = count / total_trips_by_income[income_level]
            
            # EXACT same logic to ensure all modes represented
            all_modes = set()
            for income_shares in mode_shares.values():
                all_modes.update(income_shares.keys())
            
            for income_level in income_levels:
                for mode in all_modes:
                    if mode not in mode_shares[income_level]:
                        mode_shares[income_level][mode] = 0
            
            # Calculate average mode share for each mode (EXACT same logic)
            avg_mode_shares = {}
            for mode in all_modes:
                shares = [mode_shares[income][mode] for income in income_levels]
                avg_mode_shares[mode] = sum(shares) / len(shares)
            
            # Calculate MAE using EXACT same approach
            total_mae = 0
            
            for income_level in income_levels:
                for mode in all_modes:
                    mode_share = mode_shares[income_level].get(mode, 0)  # sᵢ,ₘ
                    avg_share = avg_mode_shares.get(mode, 0)             # s̄ₘ
                    
                    # φ(|sᵢ,ₘ - s̄ₘ|) = simple absolute difference
                    abs_diff = abs(mode_share - avg_share)
                    total_mae += abs_diff  # Sum across ALL mode-income combinations
            
            session.close()
            return total_mae
            
        except Exception as e:
            print(f"Error calculating mode choice equity: {e}")
            if 'session' in locals():
                session.close()
            return 0.0

    def calculate_travel_time_equity(self):
        """
        Calculate Travel Time Equity using EXACT approach from travel_time_equity_optimization.py
        """
        try:
            session = self.Session()

            # ADD THIS DEBUG BLOCK  
            if self.schema:
                commuter_table = f"{TABLE_PREFIX}commuter_info_log"
                commuter_count = session.execute(text(f"SELECT COUNT(*) FROM {commuter_table}")).fetchone()[0]
            else:
                from ABM.agent_service_provider_initialisation import CommuterInfoLog
                commuter_count = session.query(CommuterInfoLog).count()
            
            print(f" [DEBUG] Travel Time Equity - CommuterInfoLog entries: {commuter_count}")
            
            if commuter_count == 0:
                print(f"[DEBUG] ❌ CommuterInfoLog is EMPTY! This is why travel time equity = 0")
                session.close()
                return 0.0
            
            income_levels = ['low', 'middle', 'high']
            
            travel_times_by_income = {}
            trip_counts_by_income = {}
            total_travel_time = 0
            total_trips = 0
            
            # EXACT same query pattern as your working code
            for income_level in income_levels:
                if self.schema:
                    booking_table = f"{TABLE_PREFIX}service_booking_log"
                    commuter_table = f"{TABLE_PREFIX}commuter_info_log"
                    query = text(f"""
                        SELECT sbl.total_time, COUNT(sbl.request_id) as trip_count
                        FROM {booking_table} sbl 
                        JOIN {commuter_table} cil ON sbl.commuter_id = cil.commuter_id
                        WHERE cil.income_level = :income_level AND sbl.total_time IS NOT NULL
                        GROUP BY sbl.total_time
                    """)
                    travel_times = session.execute(query, {"income_level": income_level}).fetchall()
                else:
                    from ABM.agent_service_provider_initialisation import ServiceBookingLog, CommuterInfoLog
                    travel_times = session.query(
                        ServiceBookingLog.total_time,
                        func.count(ServiceBookingLog.request_id).label('trip_count')
                    ).join(
                        CommuterInfoLog,
                        ServiceBookingLog.commuter_id == CommuterInfoLog.commuter_id
                    ).filter(
                        CommuterInfoLog.income_level == income_level,
                        ServiceBookingLog.total_time != None
                    ).group_by(
                        ServiceBookingLog.total_time
                    ).all()
                
                # EXACT same calculation logic
                income_total_time = sum(time * count for time, count in travel_times)
                income_trip_count = sum(count for _, count in travel_times)
                
                travel_times_by_income[income_level] = income_total_time
                trip_counts_by_income[income_level] = income_trip_count
                
                total_travel_time += income_total_time
                total_trips += income_trip_count
            
            # EXACT same deviation calculation
            if total_trips > 0:
                overall_avg_travel_time = total_travel_time / total_trips  # t̄
                
                total_deviation = 0
                for income_level in income_levels:
                    if trip_counts_by_income[income_level] > 0:
                        avg_time_i = travel_times_by_income[income_level] / trip_counts_by_income[income_level]  # tᵢ
                    else:
                        avg_time_i = 0
                    
                    deviation = abs(avg_time_i - overall_avg_travel_time)  # |tᵢ - t̄|
                    total_deviation += deviation  # Σᵢ |tᵢ - t̄|
                
                session.close()
                return total_deviation
            
            session.close()
            return 0.0
            
        except Exception as e:
            print(f"Error calculating travel time equity: {e}")
            if 'session' in locals():
                session.close()
            return 0.0

    def calculate_system_efficiency(self):
        """
        Calculate System Efficiency using EXACT approach from total_system_travel_time_optimization.py
        """
        try:
            session = self.Session()
            
            # EXACT same query as your working code
            if self.schema:
                booking_table = f"{TABLE_PREFIX}service_booking_log"
                query = text(f"""
                    SELECT SUM(total_time) as total_system_time
                    FROM {booking_table}
                    WHERE total_time IS NOT NULL
                """)
                result = session.execute(query).fetchone()
            else:
                from ABM.agent_service_provider_initialisation import ServiceBookingLog
                result = session.query(
                    func.sum(ServiceBookingLog.total_time).label('total_system_time')
                ).filter(
                    ServiceBookingLog.total_time != None
                ).first()
            
            total_system_time = float(result.total_system_time) if result and result.total_system_time else 0
            
            session.close()
            return total_system_time
            
        except Exception as e:
            print(f"Error calculating system efficiency: {e}")
            if 'session' in locals():
                session.close()
            return 0.0
        
        
    def cleanup(self):
        """Properly close all database connections"""
        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
            if hasattr(self, 'Session') and self.Session:
                self.Session.remove()  # Critical for scoped_session
            if hasattr(self, 'db_engine') and self.db_engine:
                self.db_engine.dispose()  # Close connection pool
            print("🧹 Database connections cleaned up")
        except Exception as e:
            print(f"❌ Cleanup error: {e}")

# ===== CONVENIENCE FUNCTION FOR CREATING NTEO MODELS =====
def create_nteo_model(topology_type='degree_constrained', variation_parameter=4, 
                     num_commuters=250, **kwargs):
    """
    Convenience function to create NTEO model with specified network topology.
    """
    
    print(f"\n🔨 Creating NTEO model with {topology_type} topology")
    
    # Set up network configuration
    from config.network_config import get_config_for_topology, separate_topology_and_model_params
    network_config = get_config_for_topology(topology_type, variation_parameter)
    
    # CRITICAL FIX: Separate topology params from model params
    topology_params, model_params = separate_topology_and_model_params(network_config)
    
    
    # Use default database parameters if not provided
    default_params = {
        'db_connection_string': db.DB_CONNECTION_STRING,
        'num_commuters': num_commuters,
        'income_weights': db.income_weights,
        'health_weights': db.health_weights,
        'payment_weights': db.payment_weights,
        'disability_weights': db.disability_weights,
        'tech_access_weights': db.tech_access_weights,
        'age_distribution': db.age_distribution,
        'penalty_coefficients': db.PENALTY_COEFFICIENTS,
        'affordability_thresholds': db.AFFORDABILITY_THRESHOLDS,
        'value_of_time': db.VALUE_OF_TIME,
        'flexibility_adjustments': db.FLEXIBILITY_ADJUSTMENTS,
        'asc_values': db.ASC_VALUES,
        'utility_function_base_coefficients': db.UTILITY_FUNCTION_BASE_COEFFICIENTS,
        'utility_function_high_income_car_coefficients': db.UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
        'public_price_table': db.public_price_table,
        'alpha_values': db.ALPHA_VALUES,
        'dynamic_maas_surcharge_base_coefficients': db.DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
        'background_traffic_amount': db.BACKGROUND_TRAFFIC_AMOUNT,
        'congestion_alpha': db.CONGESTION_ALPHA,
        'congestion_beta': db.CONGESTION_BETA,
        'congestion_capacity': db.CONGESTION_CAPACITY,
        'congestion_t_ij_free_flow': db.CONGESTION_T_IJ_FREE_FLOW,
        'uber_like1_capacity': 50,
        'uber_like1_price': 4.0,
        'uber_like2_capacity': 50,
        'uber_like2_price': 5.5,
        'bike_share1_capacity': 30,
        'bike_share1_price': 3.5,
        'bike_share2_capacity': 30,
        'bike_share2_price': 3,
        'subsidy_dataset': db.subsidy_dataset,
        'subsidy_config': db.daily_config,
        'network_config': model_params,
    }

    # Override with provided kwargs
    default_params.update(kwargs)
    
    # Create and return model
    model = MobilityModelNTEO(**default_params)

    
    return model


