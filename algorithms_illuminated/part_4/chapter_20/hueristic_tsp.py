"""Two-Opt local search for Traveling Salesman Problem.

This module introduces efficient and approximately correct alternatives to
exhaustive search.
"""


import numpy as np
import itertools


def read_edge_file(filename):
    """Constructs an undirected adjacency matrix from an edge list.
    
    This could be more space-efficient if needed, as half the matrix is
    redundant. 
    
    Args:
        filename (str): The file containing the edge list.
      
    Returns:
        2D-array: An adjacency matrix for the input graph.
    """
    edges = []
    vertices = set()
    with open(filename, 'r') as f:
        next(f) # Skip header metadata
        for line in f:
            v1, v2, weight = (float(e) for e in line.strip().split())
            v1, v2 = int(v1), int(v2)
            vertices.add(v1)
            vertices.add(v2)
            edges.append((v1, v2, weight))
        array = np.ones((len(vertices), len(vertices))) * float('inf')
        for i in range(len(vertices)):
            array[i,i] = 0
        for v1, v2, weight in edges:
            array[v1-1,v2-1] = weight
            array[v2-1,v1-1] = weight
    return array


def generate_graph(n, cost_range=(1,100), seed=None):
    """Constructs an undirected adjacency matrix from an edge list.

    This could be more space-efficient if needed, as half the matrix is
    redundant. 
    
    Args:
        n (int): The number of stops/vertices in the problem.
        cost_range (tuple(int, int)): The minimum and maximum edge costs.
        seed (int): optional, the seed for generating random costs, default 1
    
    Returns:
        2D-array: An adjacency matrix for the input graph.
    """
    rs = np.random.RandomState(seed)
    array = rs.randint(*cost_range, (n,n))
    for i in range(n):
        array[i,i] = 0
    return (array + array.T) / 2 # Make the array symmetric


def exhaustive_tsp(graph):
    """Performs an exhuastive search over all possible tours.
    
    This implementation includes minimal optimizations.
    
    The first stop is fixed, because otherwise, the same tour
    can be shifted into n equivalent representations, n is the
    number of stops. 
    
    Second, for each pair of vertices, of which there are 
    n*(n-1)/2 (O(n^2)), when they make up the second and last
    stops, they can only be in one ordering. I will arbitrarily
    require that the second stop be smaller than the last stop
    index.
    
    Args:
        graph (array): The input graph as an adjacency matrix.
        
    Returns:
        float: The cost of the minimum-cost tour.
        list: The sequence of vertices for the minimum-cost tour,
              converted back to 1-based indexing.
        int: The number of unique cycles that were evaluated.
    """
    # Fix first stop as 0 to avoid redundant shifts in the tour
    n = graph.shape[0]
    min_cost = float('inf')
    min_tour = None
    for tour in itertools.permutations(range(1,n)):
        if tour[0] < tour[-1]: # Only compute one direction per tour
            tour_cost = graph[0,tour[0]] + graph[tour[-1],0]
            for i in range(1,len(tour)):
                tour_cost += graph[tour[i-1],tour[i]]
            if tour_cost < min_cost:
                min_cost = tour_cost
                min_tour = (1, *(s+1 for s in tour))
    return min_cost, min_tour


def nearest_neighbor_tsp(graph):
    """Selects a tour using a greedy nearest-neighbor hueristic.
    
    This approach is simple and very fast, but not correct.
    
    I'm not optimizing with vectorized numpy functions to stick to the
    basic looping implementation.
    
    Args:
        graph (array): The input graph as an adjacency matrix.
        
    Returns:
        float: The cost of the minimum-cost tour.
        list: The sequence of vertices for the minimum-cost tour.
    """
    # Track remaining stops
    remaining = set(range(1, graph.shape[0]+1))
    # Select an arbitrary first vertex
    tour = [1]
    remaining.remove(1)
    # Compute tour and cost
    cost = 0
    while len(remaining) > 0:
        min_distance = float('inf')
        nearest_neighbor = None
        for v in remaining:
            distance = graph[tour[-1]-1, v-1]
            if distance < min_distance:
                min_distance = distance
                nearest_neighbor = v
        remaining.remove(nearest_neighbor)
        cost += min_distance
        tour.append(nearest_neighbor)
    # Close the tour
    cost += graph[tour[-1]-1,0]
    return cost, tour


def two_opt_first_tsp(graph, tour, cost):
    """Attempts to improve a tour using local search.
    
    This variation stops when any improving swap is found. I can
    also implement a variation where each iteration picks the 
    most improving of all improving swaps.
    
    Args:
        graph (array): The input graph as an adjacency matrix.
        tour (list): The initial tour.
        cost (int): The cost of the initial tour.
        
    Returns:
        float: The cost of the final minimum-cost tour.
        list: The sequence of vertices for the minimum-cost tour.
    """
    # Enumerate all candidate pairs of edges
    n = len(tour)
    while True:
        result = False
        for i in range(n-2):
            for j in range(i+2, n-1):
                result = two_change(graph, tour, i, i+1, j, j+1)
                if result:
                    tour = result[0]
                    cost += result[1]
                    break
            if result:
                break
            # Handle edge case
            if i > 0:
                result = two_change(graph, tour, i, i+1, n-1, 0)
                if result:
                    tour = result[0]
                    cost += result[1]
                    break
        # Terminate the algorithm if no further improvement possible
        if not result:
            return cost, tour
                
            
def two_change(graph, tour, x1, x2, y1, y2):
    """Updates the tour, if an improvement, else makes no change.
    
    Args:
        tour (list): The initial tour.
        edge_1 (list): The first edge.
        edge_2 (list): The second edge.
    
    Returns:
        bool: A flag indicating whether the change improved the tour.
    """
    # Only first/first and last/last vertices are valid for connecting
    #breakpoint()
    change = (graph[tour[x1]-1, tour[y1]-1] + graph[tour[x2]-1, tour[y2]-1] -
              graph[tour[x1]-1, tour[x2]-1] - graph[tour[y1]-1, tour[y2]-1]) 
    if change < 0:
        temp_tour = tour[0:x1+1] + tour[y1:x1:-1] 
        if y2 > 0:
            temp_tour += tour[y2:]
        return temp_tour, change
    else:
        return False
    

def two_opt_min_tsp(graph, tour, cost):
    """Attempts to improve a tour using local search.
    
    This variation searches for the most improving swap at
    each iteration.
    
    Args:
        graph (array): The input graph as an adjacency matrix.
        tour (list): The initial tour.
        cost (int): The cost of the initial tour.
        
    Returns:
        float: The cost of the final minimum-cost tour.
        list: The sequence of vertices for the minimum-cost tour.
    """
    # Enumerate all candidate pairs of edges
    n = len(tour)
    while True:
        result = False
        best_change = float('inf')
        best_tour = None
        for i in range(n-2):
            for j in range(i+2, n-1):
                result = two_change(graph, tour, i, i+1, j, j+1)
                if result:
                    if result[1] < best_change:
                        best_change = result[1]
                        best_tour = result[0]
            # Handle edge case
            if i > 0:
                result = two_change(graph, tour, i, i+1, n-1, 0)
                if result:
                    if result[1] < best_change:
                        best_change = result[1]
                        best_tour = result[0]
        # Terminate the algorithm if no further improvement possible
        if best_tour is None:
            return cost, tour
        else:
            tour = best_tour
            cost += best_change
            
            
def get_tsp_tour(graph, func, preproc=False, seed=None, trials=1):
    """Wrapper for testing various TSP approaches."""
    # Get a tour for return
    if func.__name__ in ('nearest_neighbor_tsp', 'exhaustive_tsp'):
        cost, tour = func(graph)
    elif func.__name__ in ('two_opt_first_tsp', 'two_opt_min_tsp'):
        cost = float('inf')
        tour = None
        for _ in range(trials):
            temp_cost, temp_tour = get_input_tour(graph, preproc, seed)
            temp_cost, temp_tour = func(graph, temp_tour, temp_cost)
        if temp_cost < cost:
            cost = temp_cost
            tour = temp_tour
    else:
        raise ValueError(f'Requested function {func.__name__} not recognized.')
    return cost, tour
    
    
def get_input_tour(graph, preproc, seed):
    """Helper for getting input tour for improving algorithms."""
    # Get nearest neighbor or random input tour
    if preproc:
        cost, tour = nearest_neighbor_tsp(graph)
    else:
        tour = list(range(1, graph.shape[0]+1))
        np.random.shuffle(tour)
        cost = 0
        for i in range(len(tour)-1):
            cost += graph[tour[i]-1, tour[i+1]-1]
        cost += graph[tour[-1]-1, tour[0]-1]
    return cost, tour


if __name__ == '__main__':
    import time
    import pandas as pd

    # Define number of repeats per configuration per input size
    NUM_REPEATS = 25

    # Define various configurations to test
    # The configuration tuples are defined as follows:
    # (hueristic, whether to preprocess with nearest-neighbors, number of repeats)
    configs = [(exhaustive_tsp, False, 1),
               (nearest_neighbor_tsp, False, 1),
               (two_opt_first_tsp, False, 1),
               (two_opt_min_tsp, False, 1),
               (two_opt_first_tsp, True, 1),
               (two_opt_min_tsp, True, 1),
               (two_opt_first_tsp, False, 50),
               (two_opt_min_tsp, False, 50)]

    # First, run correctness tests
    test_cases = [('tsptest1.txt', 13),
                  ('tsptest2.txt', 23)]
    
    print('Starting correctness tests...')
    for test_case in test_cases:
        graph = read_edge_file(test_case[0])
        for config in configs:
            cost, tour = get_tsp_tour(graph, config[0], config[1], seed=1)
            print(f'{config[0].__name__} with preproc={config[1]} on {test_case[0]}, {100*cost / test_case[1]:.1f}% of optimal cost.')
    print('Passed all correctness tests.')

    # Second, run performance tests
    print('Starting performance tests...')
    results = []
    for i in range(9, 13):
        result_row = {'size':i}
        df_repeats = []
        for r in range(NUM_REPEATS):
            result_repeat = {}
            graph = generate_graph(i, seed=r)
            for j, config in enumerate(configs):
                start = time.time()
                cost, _ = get_tsp_tour(graph, config[0], config[1], seed=r, trials=config[2])
                result_repeat[f'{config[0].__name__}_{config[1]}_{config[2]}_time'] = time.time() - start
                result_repeat[f'{config[0].__name__}_{config[1]}_{config[2]}_cost'] = cost
            df_repeats.append(result_repeat)
        df_repeats = pd.DataFrame(df_repeats)
        for col in df_repeats.columns:
            result_row[f'{col}_mean'] = df_repeats[col].mean().round(4)
            #result_row[f'{col}_std'] = df_repeats[col].std().round(4)
        results.append(result_row)
    print('Completed all performance tests.')

    # Formatting and display, could improve this later with plotting or 
    # % of exhaustive time/cost representation
    df_results = pd.DataFrame(results).set_index('size')
    df_time = df_results[[col for col in df_results.columns if 'time' in col]]
    df_cost = df_results[[col for col in df_results.columns if 'cost' in col]]
    print('Timing Results:')
    print(df_time.T)
    print('Tour Cost Results:')
    print(df_cost.T)  