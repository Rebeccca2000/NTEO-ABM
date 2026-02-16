# agent_run_visualisation.py - FIXED FOR UNIFIED NETWORK SYSTEM
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import TextElement
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa import Agent
from sqlalchemy.orm import sessionmaker, scoped_session
import uuid
import random
import math
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
from config.network_factory import NetworkFactory
# ===== FIXED IMPORTS FOR UNIFIED SYSTEM =====
import sys
import os

# Add paths for new unified system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'topology'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ABM'))

# Import new unified database configuration
import config.database_updated as db  # CRITICAL CHANGE: Use database_updated instead of database

# Import agent classes (they will be fixed separately)
from agent_service_provider import ServiceProvider
from agent_commuter import Commuter
from agent_MaaS import MaaS
from agent_service_provider_initialisation import reset_database, CommuterInfoLog

# Import new unified network system
from testing.abm_initialization import create_nteo_model, MobilityModelNTEO  # NEW: Use unified ABM

# Import all variables from unified database
from config.database_updated import (
    num_commuters, income_weights, health_weights, payment_weights, 
    age_distribution, disability_weights, tech_access_weights,
    DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS, DB_CONNECTION_STRING, 
    SIMULATION_STEPS, CONGESTION_ALPHA, CONGESTION_BETA, 
    CONGESTION_CAPACITY, CONGESTION_T_IJ_FREE_FLOW, 
    ASC_VALUES, UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS, 
    UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS, 
    AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, 
    VALUE_OF_TIME, public_price_table, ALPHA_VALUES, 
    BACKGROUND_TRAFFIC_AMOUNT,
    # Service provider variables (add these after updating database_updated.py)
    UberLike1_capacity, UberLike1_price,
    UberLike2_capacity, UberLike2_price, 
    BikeShare1_capacity, BikeShare1_price,
    BikeShare2_capacity, BikeShare2_price,
    # Subsidy configurations
    subsidy_dataset, daily_config, weekly_config, monthly_config,
    # Network configuration
    NETWORK_CONFIG
)
class StationAgent(Agent):  # Change from plain class to Agent subclass
    """Mesa-compatible station agent for network nodes"""
    def __init__(self, unique_id, model, location, mode, station_type="mixed", network_node_id=None):
        super().__init__(unique_id, model)  # Initialize Mesa Agent
        self.station_type = station_type
        self.network_node_id = network_node_id
        self.agent_type = "station"
        self.location = location
        self.mode = mode
    def step(self):
        pass

class MobilityModel(Model):
    def __init__(self, db_connection_string, num_commuters, 
                data_income_weights, data_health_weights, data_payment_weights, 
                data_age_distribution, data_disability_weights, data_tech_access_weights,
                ASC_VALUES, UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS, 
                UTILITY_FUNCTION_BASE_COEFFICIENTS, PENALTY_COEFFICIENTS, 
                AFFORDABILITY_THRESHOLDS, FLEXIBILITY_ADJUSTMENTS, VALUE_OF_TIME,
                public_price_table, ALPHA_VALUES, DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS, 
                BACKGROUND_TRAFFIC_AMOUNT, CONGESTION_ALPHA, CONGESTION_BETA, 
                CONGESTION_CAPACITY, CONGESTION_T_IJ_FREE_FLOW, 
                uber_like1_capacity, uber_like1_price, uber_like2_capacity, uber_like2_price, 
                bike_share1_capacity, bike_share1_price, bike_share2_capacity, bike_share2_price, 
                subsidy_dataset, subsidy_config, network_config=None, schema=None):
    
        # ===== DATABASE SETUP =====
        self.db_engine = create_engine(db_connection_string)
        self.schema = schema
        
        if self.schema:
            self.engine = create_engine(db_connection_string)
            with self.engine.connect() as connection:
                connection.execute(text(f"SET search_path TO {self.schema}"))
        
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        self.session = self.Session()
        
        # Reset database
        reset_database(self.db_engine, self.session,
                    uber_like1_capacity, uber_like1_price, 
                    uber_like2_capacity, uber_like2_price, 
                    bike_share1_capacity, bike_share1_price, 
                    bike_share2_capacity, bike_share2_price, self.schema)
        
        self.db_connection_string = db_connection_string
        

        # ===== NETWORK TOPOLOGY INITIALIZATION (UPDATED) =====
        if network_config is None:
            network_config = NETWORK_CONFIG  # Use updated import

        # Store config
        self.network_config = network_config
        self.grid_width = network_config['grid_width']
        self.grid_height = network_config['grid_height']

        topology_type = network_config['topology_type']  # ✅ NO DEFAULT - keep as is!
        print(f"🔧 Initializing {topology_type} network topology...")

        # Use network factory instead of direct TwoLayerNetworkManager calls
        network_factory = NetworkFactory()

        # Extract variation parameter based on topology type (keep your existing logic)
        if topology_type == "degree_constrained":
            variation_parameter = network_config.get('degree_constraint', 3)
        elif topology_type == "small_world":
            variation_parameter = network_config.get('rewiring_probability', 0.1)
        elif topology_type == "scale_free":
            variation_parameter = network_config.get('attachment_parameter', 2)
        elif topology_type == "grid":
            variation_parameter = network_config.get('connectivity_level', 6)
        else:
            # For any topology type you add later
            variation_parameter = network_config.get('variation_parameter', 1)


        # Create network using factory - this replaces all your if/elif statements
        self.network_manager = network_factory.create_network(
            network_type=topology_type,
            variation_parameter=variation_parameter,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            congestion_alpha=CONGESTION_ALPHA,        # ✅ Correct parameter name
            congestion_beta=CONGESTION_BETA,          # ✅ Correct parameter name  
            capacity_per_edge=CONGESTION_CAPACITY,    # ✅ Correct parameter name
            sydney_realism=network_config.get('sydney_realism', True),
            preserve_hierarchy=network_config.get('preserve_hierarchy', True)
        )


        
        # ===== GRID AND SCHEDULING SETUP =====
        self.grid = MultiGrid(network_config['grid_width'], network_config['grid_height'], torus=False)
        self.schedule = RandomActivation(self)
        
        # Store configuration
        self.network_config = network_config
        self.grid_width = network_config['grid_width']
        self.grid_height = network_config['grid_height']
        self.num_commuters = num_commuters
        self.current_step = 0
        
        # ===== STORE PARAMETERS =====
        self.data_income_weights = data_income_weights
        self.data_health_weights = data_health_weights
        self.data_payment_weights = data_payment_weights
        self.data_age_distribution = data_age_distribution
        self.data_disability_weights = data_disability_weights
        self.data_tech_access_weights = data_tech_access_weights
        self.subsidy_config = subsidy_config
        
        # Model parameters
        self.asc_values = ASC_VALUES
        self.utility_function_high_income_car_coefficients = UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS
        self.utility_function_base_coefficients = UTILITY_FUNCTION_BASE_COEFFICIENTS
        self.penalty_coefficients = PENALTY_COEFFICIENTS
        self.affordability_thresholds = AFFORDABILITY_THRESHOLDS
        self.flexibility_adjustments = FLEXIBILITY_ADJUSTMENTS
        self.value_of_time = VALUE_OF_TIME
        self.alpha_values = ALPHA_VALUES
        self.public_price_table = public_price_table
        
        # Congestion parameters
        self.congestion_alpha = CONGESTION_ALPHA
        self.congestion_beta = CONGESTION_BETA
        self.congestion_capacity = CONGESTION_CAPACITY
        self.conjestion_t_ij_free_flow = CONGESTION_T_IJ_FREE_FLOW
        self.dynamic_maas_surcharge_base_coefficient = DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS
        self.background_traffic_amount = BACKGROUND_TRAFFIC_AMOUNT
        
        # ===== INITIALIZE AGENTS =====
        self._initialize_service_provider()
        self._initialize_station_agents()
        self._initialize_commuter_agents()
        self._initialize_maas_agent()
        


    def _initialize_service_provider(self):
        """Initialize the service provider agent"""
        self.service_provider_agent = ServiceProvider(
            unique_id='service_provider_1', 
            model=self, 
            db_connection_string=self.db_connection_string,
            ALPHA_VALUES=self.alpha_values, 
            public_price_table=self.public_price_table,
            schema=self.schema
        )
        self.schedule.add(self.service_provider_agent)    

    def _initialize_station_agents(self):
        """Initialize station agents based on network topology"""
        for node_id, grid_coord in self.network_manager.spatial_mapper.node_to_grid.items():
            node_data = self.network_manager.base_network.nodes[node_id]
            
            # Determine primary transport mode for visualization
            if hasattr(node_data, 'transport_modes') and node_data.transport_modes:
                primary_mode = node_data.transport_modes[0].value
            else:
                primary_mode = 'bus'  # Default
            
            station_agent = StationAgent(
                unique_id=f"{primary_mode}_{node_id}", 
                model=self, 
                location=grid_coord, 
                mode=primary_mode
            )
            self.grid.place_agent(station_agent, grid_coord)
            self.schedule.add(station_agent)

    def _initialize_commuter_agents(self):
        """Initialize commuter agents with NETWORK-ACCESSIBLE placement only"""
        self.commuter_agents = []
        
        # Define distributions
        income_levels = ['low', 'middle', 'high']
        health_statuses = ['good', 'poor']
        payment_schemes = ['PAYG', 'subscription']
        
        # Setup age distribution
        cumulative_age_weights = self._setup_age_distribution()
        
        for i in range(self.num_commuters):
            # Generate commuter attributes
            income_level = random.choices(income_levels, self.data_income_weights)[0]
            health_status = random.choices(health_statuses, self.data_health_weights)[0]
            payment_scheme = random.choices(payment_schemes, self.data_payment_weights)[0]
            age = self._get_random_age(cumulative_age_weights)
            has_disability = random.choices([True, False], self.data_disability_weights)[0]
            tech_access = random.choices([True, False], self.data_tech_access_weights)[0]

            # NETWORK-ACCESSIBLE location selection ONLY
            commuter_location = self._get_network_accessible_location()

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
                subsidy_dataset=subsidy_dataset
            )
            
            self.commuter_agents.append(commuter)
            self.schedule.add(commuter)
            self.grid.place_agent(commuter, commuter_location)
            self.record_commuter_info(commuter)

    def _get_network_accessible_location(self):
        """Get location that's accessible to the network"""
        # Get all network nodes
        available_nodes = list(self.network_manager.spatial_mapper.node_to_grid.values())
        
        if not available_nodes:
            # Emergency fallback
            return (50, 40)
        
        # Choose a node weighted by population
        node_weights = []
        node_locations = []
        
        for node_id, grid_coord in self.network_manager.spatial_mapper.node_to_grid.items():
            node_data = self.network_manager.base_network.nodes.get(node_id)
            if node_data:
                weight = max(0.1, node_data.population_weight)  # Minimum weight of 0.1
                node_weights.append(weight)
                node_locations.append(grid_coord)
        
        if not node_locations:
            return available_nodes[0]
        
        # Weighted random selection
        chosen_location = random.choices(node_locations, weights=node_weights)[0]
        
        # Add some variation around the chosen node (within access radius)
        x, y = chosen_location
        variation_radius = 2
        
        new_x = max(0, min(self.grid_width - 1, x + random.randint(-variation_radius, variation_radius)))
        new_y = max(0, min(self.grid_height - 1, y + random.randint(-variation_radius, variation_radius)))
        
        return (new_x, new_y)

    def _setup_age_distribution(self):
        """Setup cumulative age distribution for sampling"""
        cumulative_age_weights = []
        current_weight = 0
        for age_range, weight in self.data_age_distribution.items():
            current_weight += weight
            cumulative_age_weights.append((age_range, current_weight))
        return cumulative_age_weights

    def _get_random_age(self, cumulative_age_weights):
        """Get random age based on distribution"""
        rnd = random.random()
        for age_range, cumulative_weight in cumulative_age_weights:
            if rnd <= cumulative_weight:
                return random.randint(age_range[0], age_range[1])
        return random.randint(18, 75)  # Fallback    

    def _initialize_maas_agent(self):
        """Initialize MaaS agent with network manager (REQUIRED)"""
        self.maas_agent = MaaS(
            unique_id="maas_1", 
            model=self,
            service_provider_agent=self.service_provider_agent,
            commuter_agents=self.commuter_agents,
            network_manager=self.network_manager,  # REQUIRED
            DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=self.dynamic_maas_surcharge_base_coefficient,
            BACKGROUND_TRAFFIC_AMOUNT=self.background_traffic_amount,
            stations=db.stations,  # Legacy compatibility
            routes=db.routes,      # Legacy compatibility
            transfers=db.transfers, # Legacy compatibility
            num_commuters=self.num_commuters,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            CONGESTION_ALPHA=self.congestion_alpha,
            CONGESTION_BETA=self.congestion_beta,
            CONGESTION_CAPACITY=self.congestion_capacity,
            CONGESTION_T_IJ_FREE_FLOW=self.conjestion_t_ij_free_flow,
            subsidy_config=self.subsidy_config,
            schema=self.schema
        )
        self.schedule.add(self.maas_agent)

    # ===== ALL OTHER METHODS REMAIN UNCHANGED =====
    def record_commuter_info(self, commuter):
        """Record commuter information in database (UNCHANGED)"""
        new_commuter_info = CommuterInfoLog(
        commuter_id=commuter.unique_id,
        location_x=commuter.location[0],  # NEW: separate x coordinate
        location_y=commuter.location[1],  # NEW: separate y coordinate
        age=commuter.age,
        income_level=commuter.income_level,
        has_disability=1 if commuter.has_disability else 0,
        tech_access=1 if commuter.tech_access else 0,
        health_status=commuter.health_status,
        payment_scheme=commuter.payment_scheme
        )
        
        with self.Session() as session:
            existing_commuter = session.query(CommuterInfoLog).filter_by(
                commuter_id=commuter.unique_id
            ).first()
            
            if not existing_commuter:
                session.add(new_commuter_info)
                session.commit()

    def update_commuter_info_log(self, commuter):
        """Update commuter information in database"""
        requests_with_str_keys = {str(k): {**v, 'request_id': str(v['request_id'])} 
                                for k, v in commuter.requests.items()}
        services_owned_with_str_keys = {str(k): v for k, v in commuter.services_owned.items()}

        with self.Session() as session:
            commuter_log = session.query(CommuterInfoLog).filter_by(commuter_id=commuter.unique_id).first()

            if not commuter_log:
                commuter_log = CommuterInfoLog(
                    commuter_id=commuter.unique_id,
                    location_x=commuter.location[0],  # NEW
                    location_y=commuter.location[1],  # NEW
                    age=commuter.age,
                    income_level=commuter.income_level,
                    has_disability=1 if commuter.has_disability else 0,
                    tech_access=1 if commuter.tech_access else 0,
                    health_status=commuter.health_status,
                    payment_scheme=commuter.payment_scheme,
                    requests=str(requests_with_str_keys),
                    services_owned=str(services_owned_with_str_keys)
                )
                session.add(commuter_log)
            else:
                commuter_log.location_x = commuter.location[0]  # NEW
                commuter_log.location_y = commuter.location[1]  # NEW
                commuter_log.requests = str(requests_with_str_keys)
                commuter_log.services_owned = str(services_owned_with_str_keys)
            
            session.commit()

    def check_is_peak(self, current_ticks):
        if (36 <= current_ticks % 144 < 60) or (90 <= current_ticks % 144 < 114):
            return True
        return False

    def should_create_trip(self, commuter, current_step):
        """Determine if commuter should create trip (UNCHANGED)"""
        # Get time context
        ticks_in_day = 144
        current_day_tick = current_step % ticks_in_day
        current_day = current_step // ticks_in_day
        day_of_week = current_day % 7  # 0-4 weekday, 5-6 weekend
        is_weekend = day_of_week >= 5
        
        # Count trips in the current day
        trips_in_current_day = sum(1 for request in commuter.requests.values() 
                                if request['start_time'] // ticks_in_day == current_day)
        
        # Determine the current time period
        if 36 <= current_day_tick < 60:  # 6:30am-10am
            time_period = 'morning_peak'
        elif 60 <= current_day_tick < 90:  # 10am-3pm
            time_period = 'midday'
        elif 90 <= current_day_tick < 114:  # 3pm-7pm
            time_period = 'evening_peak'
        elif 114 <= current_day_tick < 126:  # 7pm-9pm
            time_period = 'evening'
        else:  # Overnight
            time_period = 'night'
        
        # Count trips in the current time period (same day)
        if time_period == 'morning_peak':
            trips_in_period = sum(1 for request in commuter.requests.values() 
                                if request['start_time'] // ticks_in_day == current_day 
                                and 36 <= request['start_time'] % ticks_in_day < 60)
        elif time_period == 'evening_peak':
            trips_in_period = sum(1 for request in commuter.requests.values() 
                                if request['start_time'] // ticks_in_day == current_day 
                                and 90 <= request['start_time'] % ticks_in_day < 114)
        else:
            # For other periods, just check within a 24-tick window (4 hours)
            period_start = max(0, current_day_tick - 12)
            period_end = min(ticks_in_day - 1, current_day_tick + 12)
            trips_in_period = sum(1 for request in commuter.requests.values() 
                                if request['start_time'] // ticks_in_day == current_day 
                                and period_start <= request['start_time'] % ticks_in_day < period_end)
        
        # Base daily limits depending on payment scheme and demographics
        if commuter.payment_scheme == 'subscription':
            base_daily_limit = 8  # Subscription users make more trips
        else:  # PAYG
            base_daily_limit = 6
        
        # Demographic adjustments
        if commuter.income_level == 'high':
            base_daily_limit += 2  # High income makes more trips
        elif commuter.income_level == 'low':
            base_daily_limit -= 1  # Low income makes fewer trips
        
        if commuter.age >= 65 or commuter.has_disability:
            base_daily_limit -= 1  # Older/disabled people may make fewer trips
        
        # Weekend adjustments
        if is_weekend:
            # Fewer trips on weekends, especially for work commuters
            base_daily_limit = max(3, base_daily_limit - 2)
        
        # Determine max trips per time period
        if time_period == 'morning_peak' or time_period == 'evening_peak':
            # During peak, allow more trips for commuters with subscription
            if commuter.payment_scheme == 'subscription':
                period_limit = 3
            else:
                period_limit = 2
        elif time_period == 'midday':
            # Some commuters may do multiple midday trips (lunch, meetings)
            period_limit = 2
        elif time_period == 'evening':
            # Evening activities
            period_limit = 2
        else:  # night
            # Fewer trips at night
            period_limit = 1
        
        # Check if commuter is already traveling (has active non-finished trips)
        has_active_travel = any(request['status'] not in ['finished', 'expired'] 
                            for request in commuter.requests.values())
        if has_active_travel:
            return False  # Can't start a new trip while already traveling
        
        # Special case: If commuter hasn't made any trips yet today, 
        # always allow at least one trip
        if trips_in_current_day == 0:
            # But consider time of day - most people start their day in morning
            if time_period == 'morning_peak' or (is_weekend and time_period == 'midday'):
                return True
            elif time_period == 'night':
                # Very few trips start during night
                return random.random() < 0.1
            else:
                # Some probability of starting first trip later in the day
                return random.random() < 0.4
        
        # Check against limits
        return (trips_in_current_day < base_daily_limit and 
                trips_in_period < period_limit)

    def all_requests_finished(self, commuter):
        return all(request['status'] in ['finished', 'expired'] for request in commuter.requests.values())

    def create_time_based_trip(self, current_step, commuter):
        """Create trip requests with realistic timing patterns (UNCHANGED)"""
        # Add this at the very beginning
        if commuter.requests:
            # Check if commuter has any active requests
            has_active = any(r['status'] in ['active', 'Service Selected'] 
                            for r in commuter.requests.values())
            if has_active:
                return False  # Skip if already traveling
        
        # Only check trip creation every 3-5 steps to reduce calculations
        if (current_step + commuter.unique_id) % 3 != 0:
            return False
        
        # Only create new trips if commuter isn't already traveling
        if not commuter.requests or self.all_requests_finished(commuter):
            # Skip if daily trip limit reached
            if not self.should_create_trip(commuter, current_step):
                return False
                
            # Get time of day context (144 ticks = 1 day, tick 0 = midnight)
            ticks_in_day = 144
            current_day_tick = current_step % ticks_in_day
            current_day = current_step // ticks_in_day
            day_of_week = current_day % 7  # 0-4 weekday, 5-6 weekend
            is_weekend = day_of_week >= 5
            
            # Baseline probabilities influenced by demographics
            base_probability = 0.05
            
            # Income level adjustments (higher income = more trips)
            if commuter.income_level == 'high':
                base_probability *= 1.5
            elif commuter.income_level == 'low':
                base_probability *= 0.8
                
            # Age/disability adjustments
            if commuter.age >= 65 or commuter.has_disability:
                base_probability *= 0.7
                
            # PAYG vs Subscription adjustments
            if commuter.payment_scheme == 'subscription':
                base_probability *= 1.3
                
            # Time of day probability distribution (using normal curves around peak times)
            time_multiplier = 1.0
            
            # Morning peak (centered at 8am = tick 48)
            if not is_weekend:
                morning_peak_center = 48
                morning_intensity = math.exp(-0.5 * ((current_day_tick - morning_peak_center) / 8) ** 2)
                
                # Evening peak (centered at 5:30pm = tick 105)
                evening_peak_center = 105
                evening_intensity = math.exp(-0.5 * ((current_day_tick - evening_peak_center) / 10) ** 2)
                
                # Combine the peaks
                time_multiplier = max(morning_intensity * 3.0, evening_intensity * 2.5, 0.2)
            else:
                # Weekend pattern (midday peak)
                midday_peak_center = 72  # Noon
                midday_intensity = math.exp(-0.5 * ((current_day_tick - midday_peak_center) / 16) ** 2)
                time_multiplier = midday_intensity * 1.5
            
            # Final probability
            trip_probability = base_probability * time_multiplier
            
            # Decide whether to create a trip
            if random.random() < trip_probability:
                # Determine trip purpose based on time of day
                if not is_weekend and current_day_tick < 60:  # Morning on weekday
                    purpose_weights = {'work': 0.7, 'school': 0.2, 'shopping': 0.05, 'medical': 0.03, 'leisure': 0.02}
                elif not is_weekend and current_day_tick >= 90 and current_day_tick < 114:  # Evening on weekday
                    purpose_weights = {'work': 0.1, 'school': 0.05, 'shopping': 0.3, 'leisure': 0.5, 'medical': 0.05}
                elif is_weekend:  # Weekend
                    purpose_weights = {'shopping': 0.4, 'leisure': 0.4, 'medical': 0.05, 'work': 0.1, 'school': 0.05}
                else:  # Middle of weekday
                    purpose_weights = {'work': 0.2, 'school': 0.1, 'shopping': 0.3, 'medical': 0.2, 'leisure': 0.2}
                
                # Select purpose based on weights
                purposes = list(purpose_weights.keys())
                weights = list(purpose_weights.values())
                travel_purpose = random.choices(purposes, weights=weights)[0]
                
                # Generate destination based on purpose
                origin = commuter.location
                destination = self.get_purpose_based_destination(travel_purpose, origin, commuter)
                
                # Set trip timing - more realistic start time distribution
                # Most people don't plan trips exactly at current time - add a buffer
                min_delay = 1
                max_delay = 5
                
                if travel_purpose in ['work', 'school'] and current_day_tick < 30:
                    # Early morning work/school trips might be planned further ahead
                    max_delay = 4
                    
                start_time = current_step + min_delay + random.randint(0, max_delay)
                
                # Double-check that the start_time is valid before creating the request
                if start_time > current_step + 5:
                    # If somehow we ended up with a time too far ahead, adjust it
                    start_time = current_step + 5
                # Create the request
                request_id = uuid.uuid4()
                commuter.create_request(request_id, origin, destination, start_time, travel_purpose)
                return True
            
            return False

    def get_purpose_based_destination(self, purpose, origin, commuter):
        """Generate a realistic destination based on trip purpose - NETWORK ACCESSIBLE ONLY"""
        
        # Get all network-accessible locations
        available_locations = list(self.network_manager.spatial_mapper.node_to_grid.values())
        
        if not available_locations:
            # Emergency fallback to origin area
            return self._get_nearby_accessible_location(origin)
        
        # Filter locations by purpose and accessibility
        suitable_locations = []
        
        if purpose == 'work' or purpose == 'school':
            # Work/school: prefer major hubs and employment zones
            for node_id, grid_coord in self.network_manager.spatial_mapper.node_to_grid.items():
                node_data = self.network_manager.base_network.nodes.get(node_id)
                if node_data and (node_data.employment_weight > 0.3 or 
                                node_data.node_type.value in ['major_hub', 'transport_hub']):
                    suitable_locations.append(grid_coord)
                    
        elif purpose == 'shopping':
            # Shopping: prefer transport hubs and mixed-use areas
            for node_id, grid_coord in self.network_manager.spatial_mapper.node_to_grid.items():
                node_data = self.network_manager.base_network.nodes.get(node_id)
                if node_data and (node_data.node_type.value in ['transport_hub', 'major_hub'] or
                                (node_data.employment_weight > 0.2 and node_data.population_weight > 0.2)):
                    suitable_locations.append(grid_coord)
                    
        elif purpose == 'leisure':
            # Leisure: prefer areas with population density
            for node_id, grid_coord in self.network_manager.spatial_mapper.node_to_grid.items():
                node_data = self.network_manager.base_network.nodes.get(node_id)
                if node_data and node_data.population_weight > 0.3:
                    suitable_locations.append(grid_coord)
                    
        elif purpose == 'medical':
            # Medical: prefer major hubs (hospitals)
            for node_id, grid_coord in self.network_manager.spatial_mapper.node_to_grid.items():
                node_data = self.network_manager.base_network.nodes.get(node_id)
                if node_data and node_data.node_type.value == 'major_hub':
                    suitable_locations.append(grid_coord)
        
        # If no suitable locations found, use any accessible location
        if not suitable_locations:
            suitable_locations = available_locations
        
        # Remove locations too close to origin
        min_distance = 5
        distant_locations = []
        for loc in suitable_locations:
            distance = ((loc[0] - origin[0])**2 + (loc[1] - origin[1])**2)**0.5
            if distance >= min_distance:
                distant_locations.append(loc)
        
        if not distant_locations:
            distant_locations = suitable_locations
        
        if not distant_locations:
            return self._get_nearby_accessible_location(origin)
        
        # Choose destination
        destination = random.choice(distant_locations)
        
        # Add small variation around the chosen node
        x, y = destination
        variation_radius = 1
        new_x = max(0, min(self.grid_width - 1, x + random.randint(-variation_radius, variation_radius)))
        new_y = max(0, min(self.grid_height - 1, y + random.randint(-variation_radius, variation_radius)))
        
        return (new_x, new_y)

    def _get_nearby_accessible_location(self, origin):
        """Get accessible location near origin"""
        available_locations = list(self.network_manager.spatial_mapper.node_to_grid.values())
        
        if not available_locations:
            return origin
        
        # Find closest accessible location
        min_distance = float('inf')
        closest_location = available_locations[0]
        
        for loc in available_locations:
            distance = ((loc[0] - origin[0])**2 + (loc[1] - origin[1])**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_location = loc
        
        return closest_location
            
    def get_current_step(self):
        return self.current_step
    
    def step(self):
        """Main simulation step (UNCHANGED LOGIC)"""
        self.service_provider_agent.update_time_steps()
        self.current_step += 1
        
        # Update service provider
        
        availability_dict = self.service_provider_agent.initialize_availability(self.current_step - 1)
        
        # Process commuters
        active_commuters = []
        inactive_commuters = []

        for commuter in self.commuter_agents:
            needs_processing = (
                any(r['status'] == 'active' and r['start_time'] <= self.current_step + 5 
                    for r in commuter.requests.values()) or
                any(r['status'] == 'Service Selected' for r in commuter.requests.values()) or
                (self.current_step + commuter.unique_id) % 5 == 0
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
                        travel_options_without_MaaS = self.maas_agent.options_without_maas(
                            request_id, request['start_time'], request['origin'], request['destination'])
                        print(f"[DEBUG] Got {len(travel_options_without_MaaS)} traditional options")
                        travel_options_with_MaaS = self.maas_agent.maas_options(
                            commuter.payment_scheme, request_id, request['start_time'], 
                            request['origin'], request['destination'])
                        print(f"[DEBUG] Got {len(travel_options_with_MaaS)} MaaS options")
                        ranked_options = commuter.rank_service_options(
                            travel_options_without_MaaS, travel_options_with_MaaS, request_id)
                        
                        if ranked_options:
                            booking_success, availability_dict = self.maas_agent.book_service(
                                request_id, ranked_options, self.current_step, availability_dict)
                            print(f"[DEBUG] Booking successful: {booking_success}")
                            if not booking_success:
                                print(f"Booking for request {request_id} was not successful.")
                        else:
                            print(f"No viable options for request {request_id}.")
                except Exception as e:
                    print(f"Error processing request {request_id}: {str(e)}")
                    
            commuter.update_location()
            commuter.check_travel_status()

        # Process inactive commuters
        for commuter in inactive_commuters:
            if any(r['status'] == 'Service Selected' for r in commuter.requests.values()):
                commuter.update_location()

        # Batch update commuter logs periodically
        if self.current_step % 10 == 0 or self.current_step == SIMULATION_STEPS:
            self.batch_update_commuter_logs()
        
        # Update availability and pricing
        self.service_provider_agent.update_availability()
        self.service_provider_agent.dynamic_pricing_share()
        
        # Insert background traffic using network routing
        with self.Session() as session:
            self.maas_agent.insert_time_varying_traffic(session)
        
        self.schedule.step()

    def batch_update_commuter_logs(self):
        """Batch update all commuter logs"""
        with self.Session() as session:
            for commuter in self.commuter_agents:
                requests_with_str_keys = {str(k): {**v, 'request_id': str(v['request_id'])} 
                                        for k, v in commuter.requests.items()}
                services_owned_with_str_keys = {str(k): v for k, v in commuter.services_owned.items()}
                
                session.query(CommuterInfoLog).filter_by(
                    commuter_id=commuter.unique_id
                ).update({
                    'location_x': commuter.location[0],  # NEW
                    'location_y': commuter.location[1],  # NEW
                    'requests': str(requests_with_str_keys),
                    'services_owned': str(services_owned_with_str_keys)
                })
            session.commit()

    def run_model(self, num_steps):
        for _ in range(num_steps):
            self.step()


def agent_portrayal(agent):
    portrayal = {}
    if isinstance(agent, Commuter):
        color = ""
        if agent.income_level == 'low':
            color = "green"
        elif agent.income_level == 'middle':
            color = "blue"
        else:  # high income
            color = "red"
        
        if agent.current_mode is None:
            portrayal = {"Shape": "circle", "Color": color, "Filled": "true", "r": 0.7, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
        else:
            if agent.current_mode == 'walk':
                portrayal = {"Shape": "circle", "Color": color, "Filled": "true", "r": 0.5, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
            elif agent.current_mode == 'bike':
                portrayal = {"Shape": "arrowHead", "Color": color, "Filled": "true","scale": 0.5,"heading_x": 0, "heading_y": 1, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
            elif agent.current_mode == 'car':
                portrayal = {"Shape": "arrowHead", "Color": color, "Filled": "true","scale": 0.5,"heading_x": 0, "heading_y": -1, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
            elif agent.current_mode == 'bus':
                portrayal = {"Shape": "rect", "Color": color, "Filled": "true", "w": 0.5, "h": 0.5, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}
            elif agent.current_mode == 'train':
                portrayal = {"Shape": "rect", "Color": color, "Filled": "true", "w": 0.8, "h": 0.3, "Layer": 1, "text": str(agent.unique_id), "text_color": "white"}

    elif isinstance(agent, StationAgent):
        color = "yellow" if agent.mode == 'train' else "orange"
        portrayal = {"Shape": "rect", "Color": color, "Filled": "true", "w": 0.1, "h": 0.3, "Layer": 0}
    return portrayal


class CommuteCountElement(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return (
            "Number of commuters: " + str(len(model.commuter_agents)) + "<br>" +
            f"Network topology: {model.network_config['topology_type']}<br>" +
            f"Grid size: {model.grid_width}x{model.grid_height}<br>" +
            f"Network nodes: {model.network_manager.active_network.number_of_nodes()}<br>" +
            f"Network edges: {model.network_manager.active_network.number_of_edges()}<br>" +
            "<span style='color: green;'>●</span> Low income<br>" +
            "<span style='color: blue;'>●</span> Middle income<br>" +
            "<span style='color: red;'>●</span> High income<br>" +
            "<span style='color: black;'>▲</span> Bike<br>" +
            "<span style='color: black;'>▼</span> Car<br>" +
            "<span style='color: black;'>■</span> Bus<br>" +
            "<span style='color: black;'>▭</span> Train<br>" +
            "<span style='color: black;'>●</span> Walk"
        )

# Visualization setup
grid = CanvasGrid(agent_portrayal, NETWORK_CONFIG['grid_width'], NETWORK_CONFIG['grid_height'], 1500, 1500)

# Runner Code for the MobilityModel
if __name__ == "__main__":
    model = MobilityModel(
        db_connection_string=DB_CONNECTION_STRING,
        num_commuters=num_commuters,
        data_income_weights=income_weights,
        data_health_weights=health_weights,
        data_payment_weights=payment_weights,
        data_age_distribution=age_distribution,
        data_disability_weights=disability_weights,
        data_tech_access_weights=tech_access_weights,
        ASC_VALUES=ASC_VALUES,
        UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS=UTILITY_FUNCTION_HIGH_INCOME_CAR_COEFFICIENTS,
        UTILITY_FUNCTION_BASE_COEFFICIENTS=UTILITY_FUNCTION_BASE_COEFFICIENTS,
        PENALTY_COEFFICIENTS=PENALTY_COEFFICIENTS,
        AFFORDABILITY_THRESHOLDS=AFFORDABILITY_THRESHOLDS,
        FLEXIBILITY_ADJUSTMENTS=FLEXIBILITY_ADJUSTMENTS,
        VALUE_OF_TIME=VALUE_OF_TIME,
        public_price_table=public_price_table,
        ALPHA_VALUES=ALPHA_VALUES,
        DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS=DYNAMIC_MAAS_SURCHARGE_BASE_COEFFICIENTS,
        BACKGROUND_TRAFFIC_AMOUNT=BACKGROUND_TRAFFIC_AMOUNT,
        CONGESTION_ALPHA=CONGESTION_ALPHA,
        CONGESTION_BETA=CONGESTION_BETA,
        CONGESTION_CAPACITY=CONGESTION_CAPACITY,
        CONGESTION_T_IJ_FREE_FLOW=CONGESTION_T_IJ_FREE_FLOW,
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
        network_config=NETWORK_CONFIG
    )
    model.run_model(SIMULATION_STEPS)