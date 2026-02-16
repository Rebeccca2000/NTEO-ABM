import networkx as nx
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class NodeType(Enum):
    MAJOR_HUB = "major_hub"
    TRANSPORT_HUB = "transport_hub"
    LOCAL_STATION = "local_station"
    RESIDENTIAL_ZONE = "residential"
    EMPLOYMENT_ZONE = "employment"

class TransportMode(Enum):
    TRAIN = "train"
    BUS = "bus"
    METRO = "metro"
    FERRY = "ferry"
    BIKE = "bike"
    LIGHT_RAIL = "light_rail"
    WALKING = "walking"

@dataclass
class NetworkNode:
    node_id: str
    node_type: NodeType
    coordinates: Tuple[float, float]
    population_weight: float
    employment_weight: float
    transport_modes: List[TransportMode]
    zone_name: str

@dataclass
class NetworkEdge:
    from_node: str
    to_node: str
    transport_mode: TransportMode
    travel_time: float
    capacity: int
    frequency: float
    distance: float
    route_id: str  # Added for RAPTOR compatibility
    segment_order: int  # Added for route sequencing

class SydneyNetworkTopology:
    """Enhanced Sydney network with realistic structure for transport analysis"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: Dict[str, NetworkEdge] = {}
        self.spatial_bounds = {
            'min_x': 0, 'max_x': 100,
            'min_y': 0, 'max_y': 80
        }
        
    def initialize_base_sydney_network(self):
        """Create realistic Sydney network with 40+ nodes"""
        self._create_comprehensive_sydney_zones()
        self._create_train_network()
        self._create_bus_network()
        self._create_transfers_and_interchanges()
        
        # CRITICAL: Validate base network has required routes
        if not self._validate_base_network():
            print("❌ Base network validation failed. Recreating...")
            self._force_create_critical_routes()
            if not self._validate_base_network():
                raise RuntimeError("Failed to create valid Sydney base network")
        
      
    def _validate_base_network(self) -> bool:
        """Validate that critical Sydney routes exist"""
        required_routes = ['T1_WESTERN', 'T4_ILLAWARRA', 'BUS_380']
        
        for u, v, data in self.graph.edges(data=True):
            route_id = data.get('route_id', '')
            if any(req in route_id for req in required_routes):
                required_routes = [r for r in required_routes if r not in route_id]
        
        return len(required_routes) == 0

    def _force_create_critical_routes(self):
        """Force creation of critical routes if validation fails"""
 
        
        # Ensure T1_WESTERN exists
        if 'CENTRAL' in self.nodes and 'PARRAMATTA' in self.nodes:
            self._add_edge(NetworkEdge(
                from_node='CENTRAL', to_node='PARRAMATTA',
                transport_mode=TransportMode.TRAIN, travel_time=3.0,
                capacity=1800, frequency=1, distance=25.0,
                route_id='T1_WESTERN', segment_order=1
            ))
        
        # Ensure T4_ILLAWARRA exists
        if 'CENTRAL' in self.nodes and 'HURSTVILLE' in self.nodes:
            self._add_edge(NetworkEdge(
                from_node='CENTRAL', to_node='HURSTVILLE',
                transport_mode=TransportMode.TRAIN, travel_time=2.0,
                capacity=1800, frequency=1, distance=15.0,
                route_id='T4_ILLAWARRA', segment_order=1
            ))
        
        # Ensure BUS_380 exists
        if 'CENTRAL' in self.nodes and 'BONDI_JUNCTION' in self.nodes:
            self._add_edge(NetworkEdge(
                from_node='CENTRAL', to_node='BONDI_JUNCTION',
                transport_mode=TransportMode.BUS, travel_time=4.0,
                capacity=600, frequency=1, distance=12.0,
                route_id='BUS_380', segment_order=1
            ))
            
    def _create_comprehensive_sydney_zones(self):
        """Create comprehensive Sydney network with proper coverage"""
        sydney_nodes = [
            # CBD and Inner City
            NetworkNode("CENTRAL", NodeType.MAJOR_HUB, (50, 40), 0.2, 0.8, 
                       [TransportMode.TRAIN, TransportMode.BUS], "Central Station"),
            NetworkNode("TOWN_HALL", NodeType.TRANSPORT_HUB, (52, 42), 0.3, 0.7,
                       [TransportMode.TRAIN, TransportMode.BUS], "Town Hall"),
            NetworkNode("WYNYARD", NodeType.TRANSPORT_HUB, (51, 44), 0.2, 0.6,
                       [TransportMode.TRAIN, TransportMode.BUS], "Wynyard"),
            NetworkNode("CIRCULAR_QUAY", NodeType.TRANSPORT_HUB, (53, 46), 0.1, 0.5,
                       [TransportMode.TRAIN, TransportMode.BUS], "Circular Quay"),
            
            # Fixed coordinates (goes west/northwest as in reality):
            NetworkNode("REDFERN", NodeType.TRANSPORT_HUB, (48, 38), 0.3, 0.4,
                    [TransportMode.TRAIN], "Redfern"),  # Keep as reference point

            # These stations should maintain similar Y-levels or go slightly north, not south
            NetworkNode("STRATHFIELD", NodeType.TRANSPORT_HUB, (40, 40), 0.3, 0.3,
                    [TransportMode.TRAIN, TransportMode.BUS], "Strathfield"),  # Y: 35->40

            NetworkNode("HOMEBUSH", NodeType.LOCAL_STATION, (38, 41), 0.4, 0.2,
                    [TransportMode.TRAIN], "Homebush"),  # Y: 33->41

            NetworkNode("LIDCOMBE", NodeType.LOCAL_STATION, (36, 42), 0.5, 0.2,
                    [TransportMode.TRAIN], "Lidcombe"),  # Y: 31->42

            # Parramatta is Sydney's "second CBD" - major western hub
            NetworkNode("PARRAMATTA", NodeType.MAJOR_HUB, (25, 45), 0.4, 0.6,
                    [TransportMode.TRAIN, TransportMode.BUS], "Parramatta"),  # Y: 30->45

            NetworkNode("WESTMEAD", NodeType.LOCAL_STATION, (23, 46), 0.3, 0.3,
                    [TransportMode.TRAIN], "Westmead"),  # Y: 28->46

            NetworkNode("BLACKTOWN", NodeType.TRANSPORT_HUB, (15, 48), 0.5, 0.3,
                    [TransportMode.TRAIN, TransportMode.BUS], "Blacktown"),  # Y: 25->48

            NetworkNode("PENRITH", NodeType.TRANSPORT_HUB, (8, 50), 0.6, 0.2,
                    [TransportMode.TRAIN, TransportMode.BUS], "Penrith"),  # Y: 22->50

            # CORRECTED T2 Inner West Line - should connect logically to Western Line
            NetworkNode("NEWTOWN", NodeType.LOCAL_STATION, (45, 36), 0.4, 0.3,
                    [TransportMode.TRAIN], "Newtown"),  # Keep similar to original

            NetworkNode("SUMMER_HILL", NodeType.LOCAL_STATION, (42, 38), 0.4, 0.2,
                    [TransportMode.TRAIN], "Summer Hill"),  # Y: 33->38

            NetworkNode("ASHFIELD", NodeType.LOCAL_STATION, (40, 38), 0.4, 0.3,
                    [TransportMode.TRAIN, TransportMode.BUS], "Ashfield"),  # Y: 32->38
            # T3 Bankstown Line
            NetworkNode("SYDENHAM", NodeType.LOCAL_STATION, (43, 30), 0.4, 0.2,
                       [TransportMode.TRAIN], "Sydenham"),
            NetworkNode("WILEY_PARK", NodeType.LOCAL_STATION, (38, 28), 0.5, 0.2,
                       [TransportMode.TRAIN], "Wiley Park"),
            NetworkNode("LAKEMBA", NodeType.LOCAL_STATION, (35, 25), 0.6, 0.2,
                       [TransportMode.TRAIN], "Lakemba"),
            NetworkNode("BANKSTOWN", NodeType.TRANSPORT_HUB, (30, 20), 0.6, 0.3,
                       [TransportMode.TRAIN, TransportMode.BUS], "Bankstown"),
            NetworkNode("LIVERPOOL", NodeType.MAJOR_HUB, (25, 15), 0.7, 0.4,
                       [TransportMode.TRAIN, TransportMode.BUS], "Liverpool"),
            
            # T4 Eastern Suburbs/Illawarra
            NetworkNode("MARTIN_PLACE", NodeType.TRANSPORT_HUB, (51, 41), 0.2, 0.8,
                       [TransportMode.TRAIN], "Martin Place"),
            NetworkNode("KINGS_CROSS", NodeType.LOCAL_STATION, (55, 43), 0.3, 0.4,
                       [TransportMode.TRAIN], "Kings Cross"),
            NetworkNode("EDGECLIFF", NodeType.LOCAL_STATION, (58, 45), 0.3, 0.3,
                       [TransportMode.TRAIN], "Edgecliff"),
            NetworkNode("BONDI_JUNCTION", NodeType.TRANSPORT_HUB, (65, 45), 0.4, 0.4,
                       [TransportMode.TRAIN, TransportMode.BUS], "Bondi Junction"),
            
            # T4 South - Illawarra
            NetworkNode("HURSTVILLE", NodeType.TRANSPORT_HUB, (45, 25), 0.5, 0.3,
                       [TransportMode.TRAIN, TransportMode.BUS], "Hurstville"),
            NetworkNode("KOGARAH", NodeType.LOCAL_STATION, (47, 23), 0.4, 0.2,
                       [TransportMode.TRAIN], "Kogarah"),
            NetworkNode("ROCKDALE", NodeType.LOCAL_STATION, (48, 21), 0.4, 0.2,
                       [TransportMode.TRAIN], "Rockdale"),
            NetworkNode("SUTHERLAND", NodeType.TRANSPORT_HUB, (50, 15), 0.5, 0.3,
                       [TransportMode.TRAIN, TransportMode.BUS], "Sutherland"),
            
            # T1 North Shore Line
            NetworkNode("MILSONS_POINT", NodeType.LOCAL_STATION, (52, 48), 0.2, 0.3,
                       [TransportMode.TRAIN], "Milsons Point"),
            NetworkNode("NORTH_SYDNEY", NodeType.TRANSPORT_HUB, (53, 50), 0.3, 0.5,
                       [TransportMode.TRAIN, TransportMode.BUS], "North Sydney"),
            NetworkNode("CROWS_NEST", NodeType.LOCAL_STATION, (54, 55), 0.4, 0.4,
                       [TransportMode.TRAIN], "Crows Nest"),
            NetworkNode("ST_LEONARDS", NodeType.LOCAL_STATION, (55, 58), 0.4, 0.3,
                       [TransportMode.TRAIN], "St Leonards"),
            NetworkNode("CHATSWOOD", NodeType.MAJOR_HUB, (55, 65), 0.4, 0.5,
                       [TransportMode.TRAIN, TransportMode.BUS, TransportMode.METRO], "Chatswood"),
            NetworkNode("ROSEVILLE", NodeType.LOCAL_STATION, (58, 70), 0.5, 0.2,
                       [TransportMode.TRAIN], "Roseville"),
            NetworkNode("GORDON", NodeType.LOCAL_STATION, (60, 72), 0.5, 0.2,
                       [TransportMode.TRAIN], "Gordon"),
            NetworkNode("HORNSBY", NodeType.TRANSPORT_HUB, (62, 75), 0.6, 0.3,
                       [TransportMode.TRAIN, TransportMode.BUS], "Hornsby"),
            
            # Northern Beaches (Bus Only)
            NetworkNode("MANLY", NodeType.TRANSPORT_HUB, (65, 55), 0.3, 0.2,
                       [TransportMode.BUS], "Manly"),
            NetworkNode("DEE_WHY", NodeType.LOCAL_STATION, (68, 60), 0.4, 0.1,
                       [TransportMode.BUS], "Dee Why"),
            NetworkNode("MONA_VALE", NodeType.LOCAL_STATION, (70, 65), 0.4, 0.1,
                       [TransportMode.BUS], "Mona Vale"),
            
            # Airport Line
            NetworkNode("MASCOT", NodeType.LOCAL_STATION, (48, 32), 0.2, 0.3,
                       [TransportMode.TRAIN], "Mascot"),
            NetworkNode("DOMESTIC_AIRPORT", NodeType.TRANSPORT_HUB, (46, 30), 0.1, 0.3,
                       [TransportMode.TRAIN, TransportMode.BUS], "Domestic Airport"),
            NetworkNode("INTERNATIONAL_AIRPORT", NodeType.TRANSPORT_HUB, (44, 28), 0.1, 0.3,
                       [TransportMode.TRAIN, TransportMode.BUS], "International Airport"),
            
            # Additional Bus Hubs
            NetworkNode("BURWOOD", NodeType.TRANSPORT_HUB, (35, 35), 0.4, 0.3,
                       [TransportMode.BUS], "Burwood"),
            NetworkNode("EASTGARDENS", NodeType.LOCAL_STATION, (60, 35), 0.4, 0.2,
                       [TransportMode.BUS], "Eastgardens"),
            NetworkNode("MAROUBRA", NodeType.LOCAL_STATION, (62, 30), 0.4, 0.1,
                       [TransportMode.BUS], "Maroubra"),
            NetworkNode("RANDWICK", NodeType.TRANSPORT_HUB, (58, 38), 0.4, 0.3,
                       [TransportMode.BUS], "Randwick"),
        ]
        
        for node in sydney_nodes:
            self._add_node(node)
    
    def _create_train_network(self):
        """Create comprehensive train network with proper route IDs"""
        # self._add_edge(NetworkEdge(
        #         from_node='CENTRAL', to_node='BONDI_JUNCTION',
        #         transport_mode=TransportMode.BUS, travel_time=4.0,
        #         capacity=600, frequency=1, distance=12.0,
        #         route_id='BUS_380', segment_order=1
        #     ))
        train_connections = [
            # T1 Western Line (Emu Plains - City - North Shore)
            ("PENRITH", "BLACKTOWN", TransportMode.TRAIN, 0.5, 1800, 1, 15.0, "T1_WESTERN", 1),
            ("BLACKTOWN", "WESTMEAD", TransportMode.TRAIN, 1, 1800, 1.5, 6.0, "T1_WESTERN", 2),
            ("WESTMEAD", "PARRAMATTA", TransportMode.TRAIN, 1.5, 1800, 1, 3.0, "T1_WESTERN", 3),
            ("PARRAMATTA", "LIDCOMBE", TransportMode.TRAIN, 12, 1800, 1, 8.0, "T1_WESTERN", 4),
            ("LIDCOMBE", "HOMEBUSH", TransportMode.TRAIN, 6, 1800, 1, 4.0, "T1_WESTERN", 5),
            ("HOMEBUSH", "STRATHFIELD", TransportMode.TRAIN, 8, 1800, 1, 5.0, "T1_WESTERN", 6),
            ("STRATHFIELD", "REDFERN", TransportMode.TRAIN, 12, 1800, 1, 10.0, "T1_WESTERN", 7),
            ("REDFERN", "CENTRAL", TransportMode.TRAIN, 5, 1800, 1.5, 3.0, "T1_WESTERN", 8),
            
            # T1 North Shore Branch
            ("CENTRAL", "WYNYARD", TransportMode.TRAIN, 0.5, 2000, 1, 5.0, "T1_NORTH_SHORE", 1),
            ("WYNYARD", "MILSONS_POINT", TransportMode.TRAIN, 1, 2000, 1, 4.0, "T1_NORTH_SHORE", 2),
            ("MILSONS_POINT", "NORTH_SYDNEY", TransportMode.TRAIN, 1, 2000, 1, 3.0, "T1_NORTH_SHORE", 3),
            ("NORTH_SYDNEY", "CROWS_NEST", TransportMode.TRAIN, 0.5, 2000, 1, 6.0, "T1_NORTH_SHORE", 4),
            ("CROWS_NEST", "ST_LEONARDS", TransportMode.TRAIN, 1, 2000, 1, 4.0, "T1_NORTH_SHORE", 5),
            ("ST_LEONARDS", "CHATSWOOD", TransportMode.TRAIN, 1.2, 2000, 1, 8.0, "T1_NORTH_SHORE", 6),
            ("CHATSWOOD", "ROSEVILLE", TransportMode.TRAIN, 2, 1600, 1, 6.0, "T1_NORTH_SHORE", 7),
            ("ROSEVILLE", "GORDON", TransportMode.TRAIN, 1, 1600, 1, 4.0, "T1_NORTH_SHORE", 8),
            ("GORDON", "HORNSBY", TransportMode.TRAIN, 1, 1600, 1, 6.0, "T1_NORTH_SHORE", 9),
            
            # T2 Inner West Line
            ("CENTRAL", "NEWTOWN", TransportMode.TRAIN, 1, 1600, 1, 6.0, "T2_INNER_WEST", 1),
            ("NEWTOWN", "SUMMER_HILL", TransportMode.TRAIN, 1.5, 1600, 1, 7.0, "T2_INNER_WEST", 2),
            ("SUMMER_HILL", "ASHFIELD", TransportMode.TRAIN, 0.6, 1600, 1, 4.0, "T2_INNER_WEST", 3),
            ("ASHFIELD", "STRATHFIELD", TransportMode.TRAIN, 0.8, 1600, 1, 5.0, "T2_INNER_WEST", 4),
            
            # T3 Bankstown Line
            ("CENTRAL", "SYDENHAM", TransportMode.TRAIN, 1.5, 1400, 0.8, 12.0, "T3_BANKSTOWN", 1),
            ("SYDENHAM", "WILEY_PARK", TransportMode.TRAIN, 1.2, 1400, 0.8, 9.0, "T3_BANKSTOWN", 2),
            ("WILEY_PARK", "LAKEMBA", TransportMode.TRAIN, 0.8, 1400, 0.8, 6.0, "T3_BANKSTOWN", 3),
            ("LAKEMBA", "BANKSTOWN", TransportMode.TRAIN, 1, 1400, 0.8, 7.0, "T3_BANKSTOWN", 4),
            ("BANKSTOWN", "LIVERPOOL", TransportMode.TRAIN, 1.5, 1400, 0.8, 12.0, "T3_BANKSTOWN", 5),
            
            # T4 Eastern Suburbs Line
            ("CENTRAL", "MARTIN_PLACE", TransportMode.TRAIN, 0.5, 1800, 1.2, 3.0, "T4_EASTERN", 1),
            ("MARTIN_PLACE", "KINGS_CROSS", TransportMode.TRAIN, 0.8, 1800, 1.2, 5.0, "T4_EASTERN", 2),
            ("KINGS_CROSS", "EDGECLIFF", TransportMode.TRAIN, 1.0, 1800, 1.2, 7.0, "T4_EASTERN", 3),
            ("EDGECLIFF", "BONDI_JUNCTION", TransportMode.TRAIN, 1.2, 1800, 1.2, 8.0, "T4_EASTERN", 4),
            
            # T4 Illawarra Line
            ("CENTRAL", "REDFERN", TransportMode.TRAIN, 0.5, 1800, 1.2, 3.0, "T4_ILLAWARRA", 1),
            ("REDFERN", "HURSTVILLE", TransportMode.TRAIN, 2.0, 1800, 1.2, 15.0, "T4_ILLAWARRA", 2),
            ("HURSTVILLE", "KOGARAH", TransportMode.TRAIN, 0.6, 1600, 1.0, 4.0, "T4_ILLAWARRA", 3),
            ("KOGARAH", "ROCKDALE", TransportMode.TRAIN, 0.5, 1600, 1.0, 3.0, "T4_ILLAWARRA", 4),
            ("ROCKDALE", "SUTHERLAND", TransportMode.TRAIN, 0.8, 1600, 1.0, 6.0, "T4_ILLAWARRA", 5),
            
            # Airport Line
            ("CENTRAL", "MASCOT", TransportMode.TRAIN, 1.2, 1200, 0.8, 9.0, "T8_AIRPORT", 1),
            ("MASCOT", "DOMESTIC_AIRPORT", TransportMode.TRAIN, 0.6, 1200, 0.8, 4.0, "T8_AIRPORT", 2),
            ("DOMESTIC_AIRPORT", "INTERNATIONAL_AIRPORT", TransportMode.TRAIN, 0.5, 1200, 0.8, 3.0, "T8_AIRPORT", 3),
        ]
        
        for connection in train_connections:
            self._add_edge(*connection)
    
    def _create_bus_network(self):
        """Create comprehensive bus network"""
        bus_connections = [
            # Cross-city bus routes
            ("CENTRAL", "BONDI_JUNCTION", TransportMode.BUS, 2.5, 600, 1.5, 18.0, "BUS_380", 1),
            ("WYNYARD", "MANLY", TransportMode.BUS, 3.0, 600, 1.2, 22.0, "BUS_E60", 1),
            ("CHATSWOOD", "MANLY", TransportMode.BUS, 2.0, 400, 1.0, 15.0, "BUS_143", 1),
            ("MANLY", "DEE_WHY", TransportMode.BUS, 1.5, 400, 1.2, 10.0, "BUS_136", 1),
            ("DEE_WHY", "MONA_VALE", TransportMode.BUS, 1.2, 400, 1.0, 8.0, "BUS_136", 2),
            
            # Western suburbs connections
            ("PARRAMATTA", "BLACKTOWN", TransportMode.BUS, 2.0, 500, 0.8, 15.0, "BUS_600", 1),
            ("LIVERPOOL", "BANKSTOWN", TransportMode.BUS, 1.8, 500, 1.0, 12.0, "BUS_901", 1),
            ("BURWOOD", "STRATHFIELD", TransportMode.BUS, 1.0, 400, 1.2, 6.0, "BUS_461", 1),
            ("ASHFIELD", "BURWOOD", TransportMode.BUS, 0.8, 400, 1.2, 5.0, "BUS_461", 2),
            
            # Eastern suburbs connections
            ("BONDI_JUNCTION", "RANDWICK", TransportMode.BUS, 1.2, 600, 1.5, 8.0, "BUS_381", 1),
            ("RANDWICK", "EASTGARDENS", TransportMode.BUS, 1.0, 500, 1.2, 6.0, "BUS_394", 1),
            ("EASTGARDENS", "MAROUBRA", TransportMode.BUS, 0.8, 400, 1.0, 5.0, "BUS_394", 2),
            
            # Airport connections
            ("CENTRAL", "DOMESTIC_AIRPORT", TransportMode.BUS, 2.0, 600, 1.2, 15.0, "BUS_400", 1),
            ("DOMESTIC_AIRPORT", "INTERNATIONAL_AIRPORT", TransportMode.BUS, 0.8, 600, 1.2, 5.0, "BUS_400", 2),
            
            # Northern connections
            ("HORNSBY", "CHATSWOOD", TransportMode.BUS, 2.5, 400, 0.8, 18.0, "BUS_575", 1),
            ("NORTH_SYDNEY", "CHATSWOOD", TransportMode.BUS, 2.0, 500, 1.0, 15.0, "BUS_273", 1),
        ]
        
        for connection in bus_connections:
            self._add_edge(*connection)
    
    def _create_transfers_and_interchanges(self):
        """Add transfer connections at major interchanges"""
        # Major interchanges already handled by nodes with multiple transport modes
        # Additional walking connections between nearby stations
        walking_connections = [
            ("CENTRAL", "TOWN_HALL", TransportMode.WALKING, 1.5, 1000, 0, 0.8, "WALK", 1),
            ("TOWN_HALL", "WYNYARD", TransportMode.WALKING, 1, 1000, 0, 0.5, "WALK", 2),
            ("WYNYARD", "CIRCULAR_QUAY", TransportMode.WALKING, 1.5, 1000, 0, 0.6, "WALK", 3),
            ("MARTIN_PLACE", "TOWN_HALL", TransportMode.WALKING, 1, 1000, 0, 0.3, "WALK", 4),
        ]
        
        for connection in walking_connections:
            self._add_edge(*connection)
    
    def _add_node(self, node: NetworkNode):
        """Add a node to the network"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **node.__dict__)
    
    def _add_edge(self, from_node: str, to_node: str, transport_mode: TransportMode, 
              travel_time: float, capacity: int, frequency: float, 
              distance: float, route_id: str, segment_order: int):
        
        edge_id = f"{from_node}_{to_node}_{transport_mode.value}_{route_id}"
        edge = NetworkEdge(from_node, to_node, transport_mode, travel_time, 
                        capacity, frequency, distance, route_id, segment_order)
        
        self.edges[edge_id] = edge
        
        # Add to NetworkX graph with ALL required attributes
        self.graph.add_edge(from_node, to_node,
                        transport_mode=transport_mode,  # CRITICAL
                        travel_time=travel_time,
                        capacity=capacity,
                        frequency=frequency,
                        distance=distance,
                        route_id=route_id,
                        segment_order=segment_order,
                        edge_type="original")  # Mark original edges

