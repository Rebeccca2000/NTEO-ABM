import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import TextElement
import json

class LiveAgentTracker:
    """Real-time agent movement and route popularity tracker"""
    
    def __init__(self, model):
        self.model = model
        self.network_manager = model.network_manager
        
        # Create edge numbering system
        self.edge_to_number = {}
        self.number_to_edge = {}
        self._create_edge_numbering()
        
        # Track route popularity by mode
        self.route_usage_by_mode = {
            'walk': defaultdict(int),
            'bike': defaultdict(int), 
            'car': defaultdict(int),
            'bus': defaultdict(int),
            'train': defaultdict(int),
            'public': defaultdict(int),
            'MaaS_Bundle': defaultdict(int),
            'all': defaultdict(int)
        }
        
        # Track edge usage by mode (using edge numbers)
        self.edge_usage_by_mode = {
            'walk': defaultdict(int),
            'bike': defaultdict(int),
            'car': defaultdict(int), 
            'bus': defaultdict(int),
            'train': defaultdict(int),
            'public': defaultdict(int),
            'MaaS_Bundle': defaultdict(int),
            'all': defaultdict(int)
        }
        
        # Track mode usage
        self.mode_usage = defaultdict(int)
        
        # Track agent trails
        self.agent_trails = defaultdict(list)  # agent_id -> [(x,y,step), ...]
        self.max_trail_length = 10
        
        # Current step tracking
        self.current_step = 0
        
        print(f"🎯 Live Agent Tracker initialized with {len(self.edge_to_number)} numbered edges!")
    
    def _create_edge_numbering(self):
        """Create numbered mapping for all network edges"""
        edge_count = 1
        
        # Number all network edges
        for u, v in self.network_manager.active_network.edges():
            coord1 = self.network_manager.spatial_mapper.node_to_grid.get(u)
            coord2 = self.network_manager.spatial_mapper.node_to_grid.get(v)
            
            if coord1 and coord2:
                # Create standardized edge key (always smaller node first)
                edge_key = tuple(sorted([u, v]))
                
                if edge_key not in self.edge_to_number:
                    self.edge_to_number[edge_key] = edge_count
                    self.number_to_edge[edge_count] = {
                        'nodes': edge_key,
                        'coords': (coord1, coord2),
                        'description': f"{u}-{v}"
                    }
                    edge_count += 1
    
    def update_step(self, step):
        """Update tracking data for current step"""
        self.current_step = step
        self._track_agent_positions()
        self._track_route_usage()
        self._cleanup_old_trails()
    
    def _track_agent_positions(self):
        """Track current agent positions"""
        for agent in self.model.commuter_agents:
            agent_id = agent.unique_id
            position = agent.location
            
            # Add to trail
            self.agent_trails[agent_id].append((position[0], position[1], self.current_step))
            
            # Limit trail length
            if len(self.agent_trails[agent_id]) > self.max_trail_length:
                self.agent_trails[agent_id].pop(0)
    
    def _track_route_usage(self):
        """Track route usage patterns - FIXED to avoid over-counting and track actual paths"""
        for agent in self.model.commuter_agents:
            # Only track trips that just started (not every step)
            for request_id, request in agent.requests.items():
                if (request['status'] == 'Service Selected' and 
                    request['start_time'] == self.current_step):  # Only count when trip starts
                    
                    origin = request['origin']
                    destination = request['destination']
                    
                    # Get the actual route taken, not just origin-destination
                    selected_route = request.get('selected_route', {})
                    if 'route' in selected_route:
                        route_data = selected_route['route']
                        mode = selected_route.get('mode', 'unknown')
                        
                        # Clean up mode name
                        if 'UberLike' in mode or 'BikeShare' in mode:
                            if 'Uber' in mode:
                                clean_mode = 'car'
                            else:
                                clean_mode = 'bike'
                        elif 'public' in mode:
                            clean_mode = 'public'
                        elif 'walk' in mode:
                            clean_mode = 'walk'
                        elif 'MaaS' in mode:
                            clean_mode = 'MaaS_Bundle'
                        else:
                            clean_mode = mode
                        
                        # Track the actual network path taken
                        self._track_actual_network_path(route_data, clean_mode, origin, destination)
                        self.mode_usage[clean_mode] += 1

    def _track_actual_network_path(self, route_data, mode, origin, destination):
        """Track the actual network path taken by agents"""
        try:
            if isinstance(route_data, list) and len(route_data) >= 2:
                # Case 1: Coordinate-based route (walk, bike, car)
                if (isinstance(route_data[0], (tuple, list)) and 
                    len(route_data[0]) == 2 and 
                    isinstance(route_data[0][0], (int, float))):
                    
                    # Track sequential coordinate pairs as network edges
                    for i in range(len(route_data) - 1):
                        start_coords = route_data[i]
                        end_coords = route_data[i + 1]
                        
                        # Find corresponding network nodes and edge number
                        edge_number = self._get_edge_number_from_coords(start_coords, end_coords)
                        if edge_number:
                            self.edge_usage_by_mode[mode][edge_number] += 1
                            self.edge_usage_by_mode['all'][edge_number] += 1
                            
                            # Also track as route segment
                            route_key = f"Edge_{edge_number}"
                            self.route_usage_by_mode[mode][route_key] += 1
                            self.route_usage_by_mode['all'][route_key] += 1
                
                # Case 2: MaaS/Public transport detailed itinerary  
                elif (isinstance(route_data[0], tuple) and 
                      len(route_data[0]) > 1 and 
                      isinstance(route_data[0][0], str)):
                    
                    self._track_transit_segments(route_data, mode)
            
            # Fallback: if we can't parse the route, track as direct connection
            else:
                edge_number = self._get_edge_number_from_coords(origin, destination)
                if edge_number:
                    self.edge_usage_by_mode[mode][edge_number] += 1
                    self.edge_usage_by_mode['all'][edge_number] += 1
                    
                    route_key = f"Edge_{edge_number}"
                    self.route_usage_by_mode[mode][route_key] += 1
                    self.route_usage_by_mode['all'][route_key] += 1
                    
        except Exception as e:
            # Fallback: try to track as direct edge
            edge_number = self._get_edge_number_from_coords(origin, destination)
            if edge_number:
                self.edge_usage_by_mode[mode][edge_number] += 1
                self.edge_usage_by_mode['all'][edge_number] += 1

    def _get_edge_number_from_coords(self, start_coords, end_coords):
        """Get edge number from coordinate pairs"""
        try:
            # Find nearest network nodes
            start_node = self.network_manager.spatial_mapper.get_nearest_node(start_coords)
            end_node = self.network_manager.spatial_mapper.get_nearest_node(end_coords)
            
            if start_node and end_node and start_node != end_node:
                edge_key = tuple(sorted([start_node, end_node]))
                return self.edge_to_number.get(edge_key)
                
        except Exception as e:
            pass
        
        return None

    def _track_transit_segments(self, detailed_itinerary, mode):
        """Track public transport/MaaS segments properly"""
        for segment in detailed_itinerary:
            if len(segment) < 3:
                continue
                
            segment_type = segment[0]
            
            if segment_type in ['bus', 'train']:
                stations = segment[2] if len(segment) > 2 else []
                
                if len(stations) >= 2:
                    # Track each station-to-station segment
                    for i in range(len(stations) - 1):
                        start_station = stations[i]
                        end_station = stations[i + 1]
                        
                        start_coords = self._station_name_to_coords(start_station)
                        end_coords = self._station_name_to_coords(end_station)
                        
                        if start_coords and end_coords:
                            edge_number = self._get_edge_number_from_coords(start_coords, end_coords)
                            if edge_number:
                                # Track as both specific mode (bus/train) and general mode
                                self.edge_usage_by_mode[segment_type][edge_number] += 1
                                self.edge_usage_by_mode[mode][edge_number] += 1  # MaaS_Bundle or public
                                self.edge_usage_by_mode['all'][edge_number] += 1
                                
                                route_key = f"Edge_{edge_number}"
                                self.route_usage_by_mode[segment_type][route_key] += 1
                                self.route_usage_by_mode[mode][route_key] += 1
                                self.route_usage_by_mode['all'][route_key] += 1

    def _station_name_to_coords(self, station_name):
        """Convert station name to coordinates using the model's method"""
        try:
            # Use the same method that commuters use
            if hasattr(self.model, 'commuter_agents') and self.model.commuter_agents:
                # Borrow the method from any commuter agent
                sample_commuter = self.model.commuter_agents[0]
                coords = sample_commuter.get_station_coordinates(station_name)
                return coords
        except Exception:
            pass
        
        # Fallback: try network manager directly
        try:
            if station_name in self.network_manager.spatial_mapper.node_to_grid:
                return self.network_manager.spatial_mapper.node_to_grid[station_name]
        except Exception:
            pass
        
        return None
    
    def _cleanup_old_trails(self):
        """Remove old trail data"""
        cutoff_step = self.current_step - 50  # Keep last 50 steps
        
        for agent_id in self.agent_trails:
            self.agent_trails[agent_id] = [
                (x, y, step) for x, y, step in self.agent_trails[agent_id]
                if step > cutoff_step
            ]
    
    def get_popular_routes_by_mode(self, mode='all', top_n=5):
        """Get most popular routes for specific mode"""
        if mode in self.route_usage_by_mode:
            sorted_routes = sorted(self.route_usage_by_mode[mode].items(), 
                                 key=lambda x: x[1], reverse=True)
            return sorted_routes[:top_n]
        return []
    
    def get_popular_edges_by_mode(self, mode='all', top_n=10):
        """Get most used network edges for specific mode"""
        if mode in self.edge_usage_by_mode:
            sorted_edges = sorted(self.edge_usage_by_mode[mode].items(),
                                key=lambda x: x[1], reverse=True)
            return sorted_edges[:top_n]
        return []
    
    def get_edge_description(self, edge_number):
        """Get human-readable description of an edge"""
        if edge_number in self.number_to_edge:
            edge_info = self.number_to_edge[edge_number]
            return f"Edge {edge_number}: {edge_info['description']}"
        return f"Edge {edge_number}: Unknown"
    
    def get_available_modes(self):
        """Get list of modes that have been used"""
        used_modes = []
        for mode in self.edge_usage_by_mode:
            if self.edge_usage_by_mode[mode]:
                used_modes.append(mode)
        return used_modes
    
    def get_live_statistics(self):
        """Get current live statistics"""
        active_agents = sum(1 for agent in self.model.commuter_agents
                          if any(r['status'] == 'Service Selected' 
                                for r in agent.requests.values()))
        
        return {
            'step': self.current_step,
            'active_agents': active_agents,
            'total_agents': len(self.model.commuter_agents),
            'popular_routes': len(self.route_usage_by_mode['all']),
            'popular_edges': len(self.edge_usage_by_mode['all']),
            'mode_distribution': dict(self.mode_usage),
            'available_modes': self.get_available_modes()
        }

class LiveVisualizationElement(TextElement):
    """Custom element for live statistics display"""
    
    def __init__(self, tracker):
        self.tracker = tracker
    
    def render(self, model):
        stats = self.tracker.get_live_statistics()
        popular_edges = self.tracker.get_popular_edges_by_mode('all', 5)
        
        html = f"""
        <div style='font-family: monospace; font-size: 12px;'>
        <h3>🎯 Live Agent Tracking - Step {stats['step']}</h3>
        
        <b>📊 Current Status:</b><br>
        Active Agents: {stats['active_agents']}/{stats['total_agents']}<br>
        Routes Tracked: {stats['popular_routes']}<br>
        Network Edges Used: {stats['popular_edges']}<br><br>
        
        <b>🚗 Mode Distribution:</b><br>
        """
        
        for mode, count in stats['mode_distribution'].items():
            html += f"{mode}: {count}<br>"
        
        html += "<br><b>🔥 Popular Edges:</b><br>"
        for i, (edge_num, count) in enumerate(popular_edges, 1):
            edge_desc = self.tracker.get_edge_description(edge_num)
            html += f"{i}. {edge_desc}: {count} uses<br>"
        
        html += "</div>"
        return html

def enhanced_agent_portrayal(agent):
    """Enhanced agent portrayal with trails and route information"""
    portrayal = {}
    
    if hasattr(agent, 'unique_id') and hasattr(agent, 'model'):
        # Get tracker if available
        tracker = getattr(agent.model, 'live_tracker', None)
        
        if hasattr(agent, 'income_level'):  # Commuter agent
            # Base color by income
            if agent.income_level == 'low':
                base_color = "green"
            elif agent.income_level == 'middle':
                base_color = "blue"
            else:  # high income
                base_color = "red"
            
            # Shape and size by current mode
            if agent.current_mode is None:
                portrayal = {
                    "Shape": "circle", 
                    "Color": base_color, 
                    "Filled": "true", 
                    "r": 0.7, 
                    "Layer": 2,
                    "text": str(agent.unique_id)[-2:],  # Show last 2 digits
                    "text_color": "white"
                }
            else:
                # Active agents - show movement mode
                if agent.current_mode == 'walk':
                    portrayal = {
                        "Shape": "circle", 
                        "Color": base_color, 
                        "Filled": "true", 
                        "r": 0.5, 
                        "Layer": 3,
                        "text": "W",
                        "text_color": "white"
                    }
                elif agent.current_mode == 'bike':
                    portrayal = {
                        "Shape": "arrowHead", 
                        "Color": base_color, 
                        "Filled": "true",
                        "scale": 0.6,
                        "heading_x": 0, 
                        "heading_y": 1, 
                        "Layer": 3,
                        "text": "B",
                        "text_color": "white"
                    }
                elif agent.current_mode == 'car':
                    portrayal = {
                        "Shape": "arrowHead", 
                        "Color": base_color, 
                        "Filled": "true",
                        "scale": 0.7,
                        "heading_x": 0, 
                        "heading_y": -1, 
                        "Layer": 3,
                        "text": "C",
                        "text_color": "white"
                    }
                elif agent.current_mode == 'bus':
                    portrayal = {
                        "Shape": "rect", 
                        "Color": base_color, 
                        "Filled": "true", 
                        "w": 0.6, 
                        "h": 0.4, 
                        "Layer": 3,
                        "text": "🚌",
                        "text_color": "white"
                    }
                elif agent.current_mode == 'train':
                    portrayal = {
                        "Shape": "rect", 
                        "Color": base_color, 
                        "Filled": "true", 
                        "w": 0.8, 
                        "h": 0.3, 
                        "Layer": 3,
                        "text": "🚊",
                        "text_color": "white"
                    }
        
        elif hasattr(agent, 'mode'):  # Station agent
            # Station visualization with usage intensity
            usage_intensity = 0
            if tracker:
                # Check how often this station is used
                station_coord = agent.location
                nearby_usage = sum(count for edge_num, count in tracker.edge_usage_by_mode['all'].items()
                                 if edge_num in tracker.number_to_edge)
                usage_intensity = min(nearby_usage / 10.0, 1.0)  # Normalize to 0-1
            
            # Color intensity based on usage
            if agent.mode == 'train':
                base_color = "yellow"
            else:
                base_color = "orange"
            
            # Make popular stations more prominent
            size = 0.2 + (usage_intensity * 0.3)
            alpha = 0.5 + (usage_intensity * 0.5)
            
            portrayal = {
                "Shape": "rect", 
                "Color": base_color, 
                "Filled": "true", 
                "w": size, 
                "h": size * 1.5, 
                "Layer": 1,
                "stroke_color": "black" if usage_intensity > 0.3 else None,
                "stroke_width": 2 if usage_intensity > 0.3 else 0
            }
    
    return portrayal

def create_live_tracking_server(model_class, model_params, grid_width=100, grid_height=80):
    """Create Mesa server with live tracking capabilities"""
    
    # Create enhanced grid with live tracking
    grid = CanvasGrid(enhanced_agent_portrayal, grid_width, grid_height, 600, 480)
    
    # We'll add the live tracker in the model initialization
    def model_creator(**params):
        model = model_class(**params)
        # Add live tracker to model
        model.live_tracker = LiveAgentTracker(model)
        return model
    
    # Create visualization element (will be created when model exists)
    class DynamicLiveElement(TextElement):
        def render(self, model):
            if hasattr(model, 'live_tracker'):
                # Update tracker
                model.live_tracker.update_step(model.current_step)
                # Create and render element
                element = LiveVisualizationElement(model.live_tracker)
                return element.render(model)
            else:
                return "<p>Live tracker not available</p>"
    
    live_element = DynamicLiveElement()
    
    server = ModularServer(
        model_creator,
        [grid, live_element],
        "Live Agent Movement Tracker",
        model_params
    )
    
    return server

# Usage function for your existing model
def add_live_tracking_to_model(model):
    """Add live tracking to an existing model"""
    model.live_tracker = LiveAgentTracker(model)
    
    # Override the step method to include tracking
    original_step = model.step
    
    def enhanced_step():
        original_step()
        model.live_tracker.update_step(model.current_step)
        
        # Print live stats every 10 steps
        if model.current_step % 10 == 0:
            stats = model.live_tracker.get_live_statistics()
            print(f"Step {stats['step']}: {stats['active_agents']} active agents, "
                  f"{len(model.live_tracker.get_popular_edges_by_mode('all'))} popular edges")
    
    model.step = enhanced_step
    return model.live_tracker

if __name__ == "__main__":
    print("🎯 Live Agent Tracker Module")
    print("Use add_live_tracking_to_model(your_model) to add live tracking")
    print("Or integrate with Mesa visualization using create_live_tracking_server()")