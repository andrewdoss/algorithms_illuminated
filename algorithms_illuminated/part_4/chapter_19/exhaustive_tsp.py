"""Exhaustive search for Traveling Salesman Problem.

This module provides a baseline for performance on TSP, and highlights
how quickly exhaustive search becomes as the problem size increases.
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


def generate_graph(n, cost_range=(1,100), seed=1):
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
    num_searched = 0
    for tour in itertools.permutations(range(1,n)):
        if tour[0] < tour[-1]: # Only compute one direction per tour
            tour_cost = graph[0,tour[0]] + graph[tour[-1],0]
            for i in range(1,len(tour)):
                tour_cost += graph[tour[i-1],tour[i]]
            if tour_cost < min_cost:
                min_cost = tour_cost
                min_tour = (0, *tour)
            num_searched += 1
    return min_cost, min_tour, num_searched


if __name__ == '__main__':
    import time

    test_cases = [('tsptest1.txt', 13),
                  ('tsptest2.txt', 23)]
    
    # First, run correctness tests
    print('Starting correctness tests...')
    for test_case in test_cases:
        graph = read_edge_file(test_case[0])
        min_cost, _, _ = exhaustive_tsp(graph)
        assert min_cost == test_case[1], f'Failed {test_case[0]}.'
    print('Passed all correctness tests.')

    # Second, run timing tests
    for i in range(3,14):
        graph = generate_graph(i)
        start = time.time()
        min_cost, min_tour, num_searched = exhaustive_tsp(graph)
        print(f'Size: {i}, num_searched: {num_searched}, time: {round(time.time() - start, 3)}')