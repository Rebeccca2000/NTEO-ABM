from mesa import Agent
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from ABM.agent_service_provider_initialisation import TransportModes, UberLike1,UberLike2, BikeShare1, BikeShare2, ShareServiceBookingLog, TransportModes
from sqlalchemy.exc import SQLAlchemyError
from mesa.time import RandomActivation
# from database_01 import public_price_table, ALPHA_VALUES
class ServiceProvider(Agent):
    def __init__(self, unique_id, model, db_connection_string, public_price_table, ALPHA_VALUES, bike_speed, schema=None):
        super().__init__(unique_id, model)
        self.db_engine = create_engine(db_connection_string)
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        self.schedule = RandomActivation(self)
        self.db_connection_string = db_connection_string
        self.alpha_values = ALPHA_VALUES
        self.public_price_table = public_price_table
        self.bike_speed = bike_speed
        self.schema = schema  # Store schema name
        # If a schema is provided, set the search_path on a short-lived session so
        # subsequent sessions see the correct tables and avoid UndefinedTable errors.
        if self.schema:
            try:
                with self.Session() as _s:
                    _s.execute(text(f"SET search_path TO {self.schema}"))
                    _s.commit()
            except Exception as e:
                try:
                    _s.rollback()
                except Exception:
                    pass
                print(f"[DEBUG] Could not set search_path in __init__: {e}")
    def step(self):
        pass

    def update_time_steps(self):
        try:
            with self.Session() as session:
                for table_class in [UberLike1, UberLike2, BikeShare1, BikeShare2]:
                    services = session.query(table_class).all()
                    for service in services:
                        # Check if the new step_count already exists
                        existing_service = session.query(table_class).filter_by(
                            provider_id=service.provider_id,
                            step_count=service.step_count + 1
                        ).first()

                        if existing_service:
                            # Update the existing record only if necessary
                            pass  # Do nothing as per new requirements
                        else:
                            # Insert the new record
                            new_service = table_class(
                                provider_id=service.provider_id,
                                mode_id=service.mode_id,
                                company_name=service.company_name,
                                base_price=service.base_price,
                                capacity=service.capacity,
                                current_price_0=service.base_price,  # Initialize with base price
                                current_price_1=service.base_price,
                                current_price_2=service.base_price,
                                current_price_3=service.base_price,
                                current_price_4=service.base_price,
                                current_price_5=service.base_price,
                                availability_0=service.capacity,  # Initialize with full capacity
                                availability_1=service.capacity,
                                availability_2=service.capacity,
                                availability_3=service.capacity,
                                availability_4=service.capacity,
                                availability_5=service.capacity,
                                step_count=service.step_count + 1
                            )
                            session.add(new_service)

                session.commit()
        except SQLAlchemyError as e:
            print(f"Error updating time steps: {e}")


        
    def check_shared_availability(self, company_name, start_time):
        """
        Check the availability of shared transport service at the given start_time.
        """
        print(f"\n🔍 [DEBUG] check_shared_availability:")
        print(f"  - company_name: {company_name}")
        print(f"  - start_time: {start_time}")
        
        current_step = self.model.get_current_step()
        print(f"  - current_step: {current_step}")
        
        if start_time < current_step:
            print(f"❌Warning: Requested start time {start_time} is before current step {current_step}")
            return False
        
        if start_time > current_step + 5:
            print(f"❌Warning: Requested start time {start_time} is too far in future (current: {current_step})")
            return False

        try:
            future_step = start_time - current_step
            availability_column = f'availability_{future_step}'
            
            with self.Session() as session:
                if self.schema:
                    session.execute(text(f"SET search_path TO {self.schema}"))
                    session.commit()

                table_class = self.get_service_table(company_name)
                if not table_class:
                    print(f"No service table found for company: {company_name}")
                    return False

                # DEBUG: Check what step_count values exist
                total_rows = session.query(table_class).count()
                
                # FIX: Use the same step_count logic as get_shared_service_price
                # The pricing method works with step_count = current_step - 1, so use that
                query_step_count = current_step  # CHANGED: Match pricing logic
                
                step_rows = session.query(table_class).filter_by(step_count=query_step_count).all()
                print(f"  [DEBUG] Table {table_class.__tablename__} total_rows={total_rows}, rows_at_step_{query_step_count}={len(step_rows)}")

                service = session.query(table_class).filter_by(
                    company_name=company_name,
                    step_count=query_step_count  # CHANGED: Use query_step_count instead of current_step
                ).first()
                
                if service:
                    availability = getattr(service, availability_column, 0)
                    print(f"  ✅ Found service record - {availability_column}: {availability}")
                    return availability > 0
                else:
                    print(f"  ❌ No service record found for {company_name} at step {query_step_count}")
                    
                    # DEBUG: Show what step_counts are available
                    all_steps = session.query(table_class.step_count).filter_by(company_name=company_name).distinct().all()
                    available_steps = [row[0] for row in all_steps]
                    print(f"  [DEBUG] Available step_counts for {company_name}: {available_steps}")
                    
                    return False

        except Exception as e:
            print(f"Error checking availability for {company_name}: {e}")
            return False
    
    # Peak hours: Morning: 6:30am - 10am; Evening: 3pm - 7pm. 
    def check_is_peak(self, current_ticks):
        if (36 <= current_ticks % 144 < 60) or (90 <= current_ticks % 144 < 114):
            return True
        return False

    def get_travel_speed(self, mode, current_ticks):
        if mode == 'bus':
            return 3 #step/stop
        elif mode == 'train': #For 1 station, how many step is needed.
            return 5 #step/stop
        elif mode == 'walk': # For 1 unit length, how long do they need - > which means, for 
            return 1.5 #unit length/step
        elif mode == 'bike':
            return self.bike_speed  # unit length/step
        elif mode == 'car':
            if self.check_is_peak(current_ticks):
                return 6.5
            else:
                return 7.5
        else:
            print(f"❌Unknown mode: {mode}")
            return 0  # Default speed if mode is unknown
        
    def get_public_service_price(self, mode, time):
        try:
            if mode in self.public_price_table:
                if self.check_is_peak(time):
                    return self.public_price_table[mode]['on_peak']
                else:
                    return self.public_price_table[mode]['off_peak']
            else:
                print(f"❌No pricing information available for mode: {mode}")
                return None
        except Exception as e:
            print(f"❌Error retrieving pricing: {e}")
            return None

            
    def get_shared_service_price(self, mode, start_time):
        """
        Get the current price of shared transport service for the given mode and start_time.
        :param mode: Type of transport mode.
        :param start_time: The start time for the booking (can be current or up to 5 steps ahead).
        :return: Dictionary with company names as keys and their current prices as values.
        """
        current_step = self.model.get_current_step()
        if current_step is None:
            return None

        if start_time < current_step:
            raise ValueError(f"get_shared_service_price start_time ({start_time}) must be at least the current step ({current_step})")
        elif start_time > current_step + 5:
            raise ValueError("start_time must be less than current step + 5 steps")
        
        future_step = start_time - current_step
        if future_step < 0 or future_step > 5:
            raise ValueError("Invalid start_time calculation for future_step.")

        price_column = f'current_price_{future_step}'
        step_count_to_query = current_step # Remove the -1 logic

        try:
            with self.Session() as session:
                mode_id = session.query(TransportModes.mode_id).filter_by(mode_type=mode).scalar()
                if mode_id is None:
                    print(f"❌No mode_id found for mode: {mode}")
                    return None

                price_dict = {}
                if mode_id == 3:
                    service_tables = ['UberLike1', 'UberLike2']
                elif mode_id == 4:
                    service_tables = ['BikeShare1', 'BikeShare2']
                else:
                    print(f"❌No service tables found for mode_id: {mode_id}")
                    return None

                for table_name in service_tables:
                    table_class = self.get_service_table(table_name)
                    
                    # FIX: Try multiple step_count values to find data
                    services = []
                    for step_offset in [0, 1, -1]:  # Try current, +1, -1
                        test_step = current_step + step_offset
                        test_services = session.query(table_class).filter_by(
                            mode_id=mode_id, 
                            step_count=test_step
                        ).all()
                        if test_services:
                            services = test_services
                            print(f"[DEBUG] Found {len(services)} services for mode_id={mode_id} at step_count={test_step}")
                            break
                    
                    if not services:
                        print(f"❌[DEBUG] No services found for {table_name} mode_id={mode_id} around step {current_step}")
                        continue
                        
                    for service in services:
                        future_step = start_time - current_step
                        price_column = f'current_price_{max(0, min(5, future_step))}'
                        price_dict[service.company_name] = getattr(service, price_column)
                        
                return price_dict if price_dict else None
        except SQLAlchemyError as e:
            print(f"Error retrieving current price: {e}")
            return None


    def record_booking_log(self, commuter_id, request_id, company_name, start_time, duration, affected_steps, route_details):
        # Extract the actual company name from the mode string
        with self.Session() as session:
            if self.schema:
                session.execute(text(f"SET search_path TO {self.schema}"))
                session.commit()
        if '_' in company_name:
            if company_name.split('_')[0] == 'car' or company_name.split('_')[0] == 'bike':
                # Split 'car_UberLike1_route' and get 'UberLike1'
                company_name = company_name.split('_')[1]
            else:
                print(f"This is either a {company_name} so no need to record in share services table")
                return  # Add return here to exit the function for non-shared services
        
        provider_table = self.get_service_table(company_name)
        if provider_table is None:
            print(f"No provider table found for service name: {company_name}")
            return

        with self.Session() as session:
            try:
                # First check if a booking already exists for this request_id
                existing_booking = session.query(ShareServiceBookingLog).filter_by(
                    request_id=str(request_id)
                ).first()

                if existing_booking:
                    print(f"Booking already exists for request_id {request_id}. Skipping record.")
                    return

                # If no existing booking, proceed with creating new booking
                service = session.query(provider_table).filter_by(company_name=company_name).first()
                if service:
                    mode_id = service.mode_id
                    provider_id = service.provider_id

                    booking_log = ShareServiceBookingLog(
                        commuter_id=commuter_id,
                        request_id=str(request_id),
                        mode_id=mode_id,
                        provider_id=provider_id,
                        company_name=company_name,
                        start_time=start_time,
                        duration=duration,
                        affected_steps=affected_steps,
                        route_details=route_details if route_details else []
                    )
                    session.add(booking_log)
                    session.commit()
                    #print(f"Successfully recorded booking for request_id {request_id}")
                else:
                    print(f"❌No service found for company name: {company_name} in table: {provider_table}")
                    
            except Exception as e:
                print(f"❌Error recording booking log: {e}")
                session.rollback()
    def get_service_table(self, service_name):
        service_tables = {
            'UberLike1': UberLike1,
            'UberLike2': UberLike2,
            'BikeShare1': BikeShare1,
            'BikeShare2': BikeShare2
        }
        return service_tables.get(service_name)
    
    def update_availability(self):
        current_step = self.model.get_current_step()
        if current_step is None:
            return

        try:
            with self.Session() as session:
                booking_logs = session.query(ShareServiceBookingLog).all()

                service_tables = ['UberLike1', 'UberLike2', 'BikeShare1', 'BikeShare2']

                for service_table_name in service_tables:
                    service_table = self.get_service_table(service_table_name)

                    # Fetch the new rows created for the current step
                    new_services = session.query(service_table).filter(service_table.step_count >= current_step).all()

                    for new_service in new_services:
                        step_offset = new_service.step_count - current_step

                        if step_offset < 6:  # We are only interested in the next 6 steps
                            # Initialize availabilities to current availabilities
                            availability_updates = [
                                getattr(new_service, f'availability_{i}', new_service.capacity) for i in range(6)
                            ]

                            for booking in booking_logs:
                                if booking.company_name == new_service.company_name:
                                    for step in range(6):
                                        step_count = current_step + step
                                        if step_count in booking.affected_steps:
                                            availability_updates[step] -= 1
                    

                            # Update the new row with the calculated availabilities for the current step and onwards
                            for i in range(6):
                                if (step_offset + i) < 6:
                                    setattr(new_service, f'availability_{step_offset + i}', availability_updates[step_offset + i])
    

                session.commit()
        except SQLAlchemyError as e:
            print(f"Error updating availability: {e}")

    def dynamic_pricing_share(self):
        """
        Dynamic pricing for shared services that accounts for both direct and MaaS bookings.
        Uses the existing step-based database structure while considering actual utilization.
        """
        try:
            service_categories = {
                'UberLike': ['UberLike1', 'UberLike2'],
                'BikeShare': ['BikeShare1', 'BikeShare2']
            }
            current_step = self.model.get_current_step()

            with self.Session() as session:
                # For each service category and provider
                for category, service_tables in service_categories.items():
                    for service_table_name in service_tables:
                        service_table = self.get_service_table(service_table_name)
                        alpha = self.alpha_values.get(service_table_name, 0.5)
                        base_demand_ratio = 0.5  # DRC baseline

                        # Get the previous step's data
                        previous_services = session.query(service_table).filter(
                            service_table.step_count == current_step - 1
                        ).all()

                        # Get the current step's data
                        new_services = session.query(service_table).filter(
                            service_table.step_count == current_step
                        ).all()

                        for service in new_services:
                            base_price = service.base_price

                            # Update prices for each future time slot
                            for i in range(6):
                                # Get current availability and capacity
                                availability = getattr(service, f'availability_{i}')
                                total_vehicles = service.capacity
                                occupied_vehicles = total_vehicles - availability

                                # Calculate DRC (Demand-Relative-to-Capacity Ratio)
                                DRC = occupied_vehicles / total_vehicles

                                # Get previous price and occupancy
                                prev_price = base_price
                                prev_occupied = 0
                                for prev_service in previous_services:
                                    prev_price = getattr(prev_service, f'current_price_{i}', base_price)
                                    prev_availability = getattr(prev_service, f'availability_{i}', total_vehicles)
                                    prev_occupied = total_vehicles - prev_availability
                                    break

                                # Calculate demand change
                                demand_change = (occupied_vehicles - prev_occupied) / total_vehicles

                                # Calculate price adjustment
                                if DRC < base_demand_ratio:
                                    price_adjustment = -alpha * (base_demand_ratio - DRC) * (1 + demand_change)
                                else:
                                    price_adjustment = alpha * (DRC - base_demand_ratio) * (1 + demand_change)

                                # Calculate new price with bounds
                                dynamic_price = prev_price * (1 + price_adjustment)
                                dynamic_price = max(min(dynamic_price, base_price * 1.8), base_price * 0.2)

                                # Update the price
                                setattr(service, f'current_price_{i}', dynamic_price)

                            session.commit()

        except SQLAlchemyError as e:
            print(f"Error updating pricing: {e}")
            session.rollback()


    def initialize_availability(self, current_step):

        availabilities = self.fetch_availability_from_db(current_step)

        global UberLike1_avail_0, UberLike1_avail_1, UberLike1_avail_2, UberLike1_avail_3, UberLike1_avail_4, UberLike1_avail_5
        global UberLike2_avail_0, UberLike2_avail_1, UberLike2_avail_2, UberLike2_avail_3, UberLike2_avail_4, UberLike2_avail_5
        global BikeShare1_avail_0, BikeShare1_avail_1, BikeShare1_avail_2, BikeShare1_avail_3, BikeShare1_avail_4, BikeShare1_avail_5
        global BikeShare2_avail_0, BikeShare2_avail_1, BikeShare2_avail_2, BikeShare2_avail_3, BikeShare2_avail_4, BikeShare2_avail_5

        UberLike1_avail_0, UberLike1_avail_1, UberLike1_avail_2, UberLike1_avail_3, UberLike1_avail_4, UberLike1_avail_5 = availabilities['UberLike1']
        UberLike2_avail_0, UberLike2_avail_1, UberLike2_avail_2, UberLike2_avail_3, UberLike2_avail_4, UberLike2_avail_5 = availabilities['UberLike2']
        BikeShare1_avail_0, BikeShare1_avail_1, BikeShare1_avail_2, BikeShare1_avail_3, BikeShare1_avail_4, BikeShare1_avail_5 = availabilities['BikeShare1']
        BikeShare2_avail_0, BikeShare2_avail_1, BikeShare2_avail_2, BikeShare2_avail_3, BikeShare2_avail_4, BikeShare2_avail_5 = availabilities['BikeShare2']
        availability_dict = {
            'UberLike1_0': UberLike1_avail_0,
            'UberLike1_1': UberLike1_avail_1,
            'UberLike1_2': UberLike1_avail_2,
            'UberLike1_3': UberLike1_avail_3,
            'UberLike1_4': UberLike1_avail_4,
            'UberLike1_5': UberLike1_avail_5,
            'UberLike2_0': UberLike2_avail_0,
            'UberLike2_1': UberLike2_avail_1,
            'UberLike2_2': UberLike2_avail_2,
            'UberLike2_3': UberLike2_avail_3,
            'UberLike2_4': UberLike2_avail_4,
            'UberLike2_5': UberLike2_avail_5,
            'BikeShare1_0': BikeShare1_avail_0,
            'BikeShare1_1': BikeShare1_avail_1,
            'BikeShare1_2': BikeShare1_avail_2,
            'BikeShare1_3': BikeShare1_avail_3,
            'BikeShare1_4': BikeShare1_avail_4,
            'BikeShare1_5': BikeShare1_avail_5,
            'BikeShare2_0': BikeShare2_avail_0,
            'BikeShare2_1': BikeShare2_avail_1,
            'BikeShare2_2': BikeShare2_avail_2,
            'BikeShare2_3': BikeShare2_avail_3,
            'BikeShare2_4': BikeShare2_avail_4,
            'BikeShare2_5': BikeShare2_avail_5
        }
        return availability_dict
               
    def fetch_availability_from_db(self, current_step):
        old_engine = create_engine(self.db_connection_string)
        OldSession = sessionmaker(bind=old_engine)
        old_session = OldSession()
        # Ensure session uses correct schema (Postgres search_path) if provided
        if self.schema:
            try:
                old_session.execute(text(f"SET search_path TO {self.schema}"))
                old_session.commit()
            except Exception as e:
                print(f"  [DEBUG] Could not set search_path to {self.schema}: {e}")
                try:
                    old_session.rollback()
                except Exception:
                    pass
        if current_step == -1:
            import traceback
            print("🔍 STEP -1 CALLED FROM:")
            traceback.print_stack()
        try:
            # Quick sanity check: how many rows exist in each provider table
            old_tables = {
                'UberLike1': UberLike1,
                'UberLike2': UberLike2,
                'BikeShare1': BikeShare1,
                'BikeShare2': BikeShare2
            }

            # Default to zeros; more robust than None for downstream logic
            availabilities = {
                'UberLike1': [0] * 6,
                'UberLike2': [0] * 6,
                'BikeShare1': [0] * 6,
                'BikeShare2': [0] * 6
            }

            for table_name, table_class in old_tables.items():
                try:
                    count = old_session.query(table_class).count()
                    # Also fetch distinct step_counts for diagnostics
                    try:
                        steps = old_session.query(table_class.step_count).distinct().all()
                        available_steps = [row[0] for row in steps]
                    except Exception:
                        available_steps = []
                    print(f"  [DEBUG] {table_name} total_rows={count}, available_step_counts={available_steps}")
                except Exception as e:
                    print(f"  [DEBUG] Could not count rows for {table_name}: {e}")
                    try:
                        old_session.rollback()
                    except Exception:
                        pass

                try:
                    # Try exact step first, then probe nearby steps to find recent data
                    service = old_session.query(table_class).filter_by(step_count=current_step).first()
                    if not service:
                        # Try nearby small offsets first
                        for offset in [ -1, 1, -2, 2, -3, 3 ]:
                            probe = current_step + offset
                            if probe < 0:
                                continue
                            service = old_session.query(table_class).filter_by(step_count=probe).first()
                            if service:
                                print(f"  [DEBUG] No row at step {current_step} for {table_name}; using nearby step {probe} as fallback")
                                break

                    # If still not found, pick the nearest available step_count from the table (if any)
                    if not service:
                        try:
                            steps = old_session.query(table_class.step_count).distinct().all()
                            available_steps = [row[0] for row in steps]
                        except Exception:
                            available_steps = []

                        if available_steps:
                            # choose the nearest step by absolute difference
                            nearest = min(available_steps, key=lambda s: abs(s - current_step))
                            # Only accept a nearest fallback if it's within the next/previous 5 steps
                            if abs(nearest - current_step) <= 5:
                                service = old_session.query(table_class).filter_by(step_count=nearest).first()
                                if service:
                                    print(f"  [DEBUG] No close row for {table_name} at {current_step}; using nearest available step {nearest} as fallback")
                            else:
                                print(f"  [DEBUG] Nearest available step {nearest} is too far from requested {current_step} (distance {abs(nearest-current_step)}); skipping fallback")

                    if service:
                        # Log which step_count row we're using for transparency
                        try:
                            used_step = getattr(service, 'step_count', None)
                            print(f"  [DEBUG] Using row step_count={used_step} for {table_name} (requested {current_step})")
                        except Exception:
                            used_step = None

                        # If the service row is from a different step_count, shift availability
                        # so that availability index 0 corresponds to the requested current_step.
                        try:
                            row_step = getattr(service, 'step_count', current_step)
                            shift = current_step - row_step
                            # availability_i in returned dict should map to service.availability_{i+shift}
                            for i in range(6):
                                source_idx = i + shift
                                if 0 <= source_idx <= 5:
                                    availabilities[table_name][i] = getattr(service, f'availability_{source_idx}')
                                else:
                                    # out-of-range: assume 0 availability
                                    availabilities[table_name][i] = 0
                        except Exception as e:
                            print(f"  [DEBUG] Error shifting availability for {table_name}: {e}")
                            # Fallback: try direct reads
                            try:
                                availabilities[table_name][0] = service.availability_0
                                availabilities[table_name][1] = service.availability_1
                                availabilities[table_name][2] = service.availability_2
                                availabilities[table_name][3] = service.availability_3
                                availabilities[table_name][4] = service.availability_4
                                availabilities[table_name][5] = service.availability_5
                            except Exception:
                                pass
                    else:
                        # Also print available steps for easier debugging
                        try:
                            steps = old_session.query(table_class.step_count).distinct().all()
                            available_steps = [row[0] for row in steps]
                        except Exception:
                            available_steps = []
                        print(f"  ❌  [DEBUG] No row for {table_name} at step {current_step} (and no nearby fallback). available_steps={available_steps}")
                except Exception as e:
                    print(f" ❌   [DEBUG] Error reading {table_name} at step {current_step}: {e}")
                    try:
                        old_session.rollback()
                    except Exception:
                        pass

        finally:
            old_session.close()

        return availabilities