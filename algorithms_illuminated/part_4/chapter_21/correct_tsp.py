"""Inefficient, correct approaches for the Traveling Salesman Problem.

This module introduces correct alternatives to exhuastive search for TSP.
First, BellmanHeldKarp which improves complexity from factorial to exponential time.
Second, a standard MIP solver.
These new approaches are compared to exhaustive search and hueristics from prior chapters.
"""


import numpy as np
import itertools
import cvxpy as cp


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
    
                       
def get_tsp_tour(graph, func, preproc=False, seed=None, trials=1):
    """Wrapper for testing various TSP approaches."""
    # Get a tour for return
    if func.__name__ in ('nearest_neighbor_tsp', 'exhaustive_tsp', 'bellman_tsp', 'mip_tsp'):
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


def bellman_tsp(graph):
    """Computes the shortest tour of a fully-connected graph.
    
    This implementation uses the Bellman-Held-Karp dynamic
    programming based algorithm.
    
    Args:
        graph (array): The input graph as an adjacency matrix.
        
    Returns:
        float: The cost of the final minimum-cost tour.
        list: The sequence of vertices for the minimum-cost tour.
    """
    # Initialize the cache
    n = graph.shape[0]
    cache = np.ones((2**(n-1)-1, n-1)) * np.inf
    
    # Base cases, indexed using binary representation of subsets
    for i in range(1, n):
        cache[2**(i-1)-1,i-1] = graph[0,i]
        
    # Systematically solve all subproblems
    for s in range(2,n):
        # Iterate over all combinations of size s
        for combo in itertools.combinations(range(n-1), s):
            idx = sum([2**i for i in combo]) - 1
            for j in combo:
                idx_mj = idx - 2**j # Adjust index to remove j
                min_cost = float('inf')
                min_k = None
                for k in combo:
                    if k != j:
                        cost = cache[idx_mj,k] + graph[k+1,j+1]
                        if cost < min_cost:
                            min_cost = cost
                            min_j = k
                cache[idx,j] = min_cost
    # Compute the final stop and tour cost
    tour = []
    min_cost = float('inf')
    min_k = None
    for k in range(n-1):
        cost = cache[-1,k] + graph[k+1,0]
        if cost < min_cost:
            min_cost = cost
            min_k = k
    # Reconstruct the final tour
    tour.append(1)
    tour.append(min_k + 2)
    idx = cache.shape[0] - 1
    for i in range(n-1):
        temp_cost = cache[idx, min_k]
        temp_idx = idx - 2**min_k
        for k in range(n-1):
            if k + 2 not in tour:
                if cache[temp_idx, k] + graph[min_k+1, k+1] == temp_cost:
                    tour.append(k + 2)
                    min_k = k
                    break
            idx = temp_idx
    return min_cost, tour


def extract_tour(solution):
    """Extract a tour from the MIP solution matrix."""
    tour = [1]
    while True:
        next_stop = np.argmax(solution[tour[-1] - 1,:])
        if next_stop == 0:
            break # Tour complete
        else:
            tour.append(next_stop + 1)
    return tour


def mip_tsp(graph):
    """Computes the shortest tour using an MIP solver."""
    C = graph
    n = graph.shape[0]
    X = cp.Variable(graph.shape, integer=True)
    Y = cp.Variable(graph.shape, integer=True)
    obj = cp.Minimize(cp.sum(cp.multiply(X,C)))
    constraints = [X >= 0,
                   X <= 1,
                   Y >= 0,
                   Y <= n-1,
                   cp.sum(X, axis=0) == 1,
                   cp.sum(X, axis=1) == 1,
                   cp.trace(X) == 0,
                   Y[0,:] == (n-1)*X[0,:],
                   Y <= (n-1)*X,
                   cp.sum(Y, axis=0)[1:] - cp.sum(Y, axis=1)[1:] == 1]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return prob.value, extract_tour(X.value)


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
               (two_opt_first_tsp, True, 1),
               (two_opt_first_tsp, False, 50),
               (bellman_tsp, False, 1),
               (mip_tsp, False, 1)]

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
    init_time = time.time()
    for i in range(9, 13):
        print(f'Starting size {i} at {time.time() - init_time:.0f} seconds.')
        result_row = {'size':i}
        df_repeats = []
        for r in range(NUM_REPEATS):
            result_repeat = {}
            graph = generate_graph(i, seed=r)
            for j, config in enumerate(configs):
                if i < 13 or config[0] is not exhaustive_tsp:
                    start = time.time()
                    cost, _ = get_tsp_tour(graph, config[0], config[1], seed=r, trials=config[2])
                    run_time = time.time() - start
                else:
                    cost = float('inf')
                    run_time = float('inf')
                result_repeat[f'{config[0].__name__}_{config[1]}_{config[2]}_time'] = run_time
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