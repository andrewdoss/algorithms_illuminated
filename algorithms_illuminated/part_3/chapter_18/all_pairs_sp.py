"""Implementations of single-source and all-pairs shortest path algorithms.

These algorithms are correct with negative edge lengths, unlike Dijkstra.

These algorithms provide further examples of dynamic programming.
"""


import numpy as np
from collections import defaultdict


def read_edge_file(filename):
    """Constructs an adjacency list from an edge list.
    
    Args:
        filename (str): The file containing the edge list.
      
    Returns:
        defaultdict(list): An adjacency list mapping vertices to (vertex, weight).
    """
    adj_list = dict()
    with open(filename, 'r') as f:
        next(f) # Skip header metadata
        for line in f:
            v1, v2, weight = (float(e) for e in line.strip().split())
            v1, v2 = int(v1), int(v2)
            if v1 not in adj_list:
                adj_list[v1] = [[],[]] # outbound, inbound edges
            if v2 not in adj_list:
                adj_list[v2] = [[],[]] # outbound, inbound edges
            adj_list[v1][0].append((v2, weight)) # outbound
            adj_list[v2][1].append((v1, weight)) # inbound
        return adj_list
    

def bellman_ford(graph, source):
    """
    Compute the single-source shortest paths using Bellman-Ford.
    
    Args:
        graph (defaultdict(list)): An adjacency list representation of a digraph.
        source (int): The index for the vertex to compute distances from.
        
    Returns:
        array: A mapping of distances from source to each vertex. None if a negative
               cycle occurs.
    """
    # Initialize array for holding solutions
    n = len(graph)
    cache = np.zeros((n + 1, n))
    # Base cases (i=0)
    cache[0,:] = float('inf')
    cache[0,source-1] = 0
    # Solve all subproblems
    for i in range(1,n+1):
        stable = True
        for v in graph.keys():
            min_new = float('inf')
            for v_in, len_in in graph[v][1]:
                min_new = min(min_new, cache[i-1,v_in-1] + len_in)
            if min_new < cache[i-1,v-1]:
                stable = False
                cache[i,v-1] = min_new
            else:
                cache[i,v-1] = cache[i-1,v-1]
        if stable:
            return cache[i,:]
    # Failed to stablize
    return None


def bf_all_paths(graph):
    """
    Compute the all-pairs shortest paths using Bellman-Ford.
    
    Args:
        graph (defaultdict(list)): An adjacency list representation of a digraph.
        
    Returns:
        array: A mapping of distances between all vertices. None if a negative
               cycle occurs.    
    """
    all_pairs = []
    for s in range(1,len(graph)+1): # Assumes consecutive 1-based indices
        ss_sp = bellman_ford(graph, s)
        if ss_sp is not None:
            all_pairs.append(ss_sp)
        else:
            return None
    return np.vstack(all_pairs)


def floyd_warshall(graph):
    """
    Compute all=pairs shortest paths using Floyd-Warshall.
    
    Args:
        graph (defaultdict(list)): An adjacency list representation of a digraph.
        source (int): The index for the vertex to compute distances from.
        
    Returns:
        array: A mapping of distances from source to each vertex. None if a negative
               cycle occurs.
    """
    # Initialize array for holding solutions
    # Array is space-optimized to include only current and previous step
    n = len(graph)
    cache = np.ones((2, n, n)) * np.inf
    # Base cases
    for v1 in range(1,n+1):
        for v2 in range(1,n+1):
            if v1 == v2:
                cache[0,v1-1,v2-1] = 0
            for v_out, e_ln in graph[v1][0]:
                if v_out == v2:
                    cache[0,v1-1,v2-1] = e_ln
    # Systematically solve all sub-problems
    for k in range(1,n+1):
        for v1 in range(1,n+1):
            for v2 in range(1,n+1):
                cache[k % 2,v1-1,v2-1] = min(cache[(k-1) % 2,v1-1,v2-1],
                                             cache[(k-1) % 2,v1-1,k-1] + cache[(k-1) % 2,k-1,v2-1])
    # Check for a negative cycle
    for v in range(1,n+1):
        if cache[k % 2,v-1,v-1] < 0:
            return None
    return cache[k % 2]


if __name__ == '__main__':
    import time
    import pandas as pd

    test_cases = [('problem18.8test1.txt', 5, 8, -2),
                  ('problem18.8test2.txt', 5, 8, None),
                  ('problem18.8file1.txt', 1000, 47978, None),
                  ('problem18.8file2.txt', 1000, 47978, None),
                  ('problem18.8file3.txt', 1000, 47978, -19)]

    algs = [bf_all_paths, floyd_warshall]
                
    # Run correctness and timing tests
    results = []
    print('Starting correctness and timing tests...')
    for test_case in test_cases:
        graph = read_edge_file(test_case[0])
        result = {'test': test_case[0],
                  'vertices': test_case[1],
                  'edges': test_case[2]}
        for alg in algs:
            start = time.time()
            solution = np.min(alg(graph))
            result[f'{alg.__name__}_time'] = round(time.time() - start, 3)
            assert solution == test_case[3], f'Failed correctness for {test_case[0]} with {alg}, solution: {solution}.'
        results.append(result)
        print(f'All algs passed correctness for {test_case[0]}.')
    print('All correctness and timing tests passed.')
    print(pd.DataFrame(results))