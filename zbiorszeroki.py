from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import numpy as np
import random

# Tworzenie funkcji kosztu trasy, jako odległości euklidesowej między punktami
def euclidean_distance(x1, y1, x2, y2):
    return int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

# Generowanie losowych danych dla problemu marszrutyzacji pojazdów
num_vehicles = 5
depot = 0
locations = [(random.randint(0, 100), random.randint(0, 100)) for i in range(20)]

# Definiowanie funkcji kosztu trasy dla algorytmu OR-Tools
def create_distance_callback(locations):
    distances = {}

    def distance_callback(from_node, to_node):
        if (from_node, to_node) in distances:
            return distances[(from_node, to_node)]
        else:
            x1, y1 = locations[from_node]
            x2, y2 = locations[to_node]
            distance = euclidean_distance(x1, y1, x2, y2)
            distances[(from_node, to_node)] = distance
            return distance

    return distance_callback

# Definiowanie problemu i algorytmu
manager = pywrapcp.RoutingIndexManager(len(locations), num_vehicles, depot)
routing = pywrapcp.RoutingModel(manager)

distance_callback = create_distance_callback(locations)
routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
search_parameters.time_limit.seconds = 30

# Uruchomienie algorytmu dla wielu przypadków testowych
for i in range(10):
    # Generowanie losowych punktów i tworzenie nowego problemu
    num_vehicles = random.randint(3, 7)
    depot = 0
    locations = [(random.randint(0, 100), random.randint(0, 100)) for i in range(20)]

    manager = pywrapcp.RoutingIndexManager(len(locations), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    distance_callback = create_distance_callback(locations)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print(f"Przypadek testowy #{i}: minimalny koszt trasy = {solution.ObjectiveValue()}")
