"""Implementations of Prim's and Kruskal's Minimum Spanning Tree algorithms.

Each algorithm is implemented in a straightforward way and again with
a more efficient implementation using a heap (Prim) or union-find (Kurskal).
"""


from collections import defaultdict, deque, namedtuple
import heapq


def read_edge_file(filename):
    """Constructs an adjacency list from an edge list.
    
    Args:
        filename (str): The file containing the edge list.
      
    Returns:
        defaultdict(list): An adjacency list mapping vertices to (vertex, weight).
    """
    adj_list = defaultdict(list)
    with open(filename, 'r') as f:
        next(f) # Skip header metadata
        for line in f:
            v1, v2, weight = (int(e) for e in line.strip().split())
            adj_list[v1].append((v2, weight))
            if v2 in adj_list and (v1, weight) not in adj_list[v2]:
                adj_list[v2].append((v1, weight))
            elif v2 not in adj_list:
                adj_list[v2].append((v1, weight))
        return adj_list
    
    
def straightforward_prim(graph):
    """Returns a minimum spanning tree for the provided graph.
    
    Uses a straightforward and inefficient O(n^2) implementation.
    
    Args:
        graph (dict): An undirected graph in adjacency list format.
        
    Returns:
        int: The total cost of the minimum spanning tree.
        list: The set of edges in the minimum spanning tree.
    """
    connected = set() # The vertices that have been connected so far
    mst = [] # The edges in the minimum spanning tree
    cost = 0 # The total edge cost of the minimum spanning tree
    connected.add(next(iter(graph))) # Arbitrarily select starting vertex
    
    while True:
        min_cost = float('inf')
        min_edge = None
        for v1 in connected:
            for edge in graph[v1]:
                if edge[0] not in connected and edge[1] < min_cost:
                    min_edge = v1, edge[0], edge[1]
                    min_cost = edge[1]
        if min_edge is not None:
            connected.add(min_edge[1])
            mst.append(min_edge)
            cost += min_cost
        else:
            break
    return cost, mst


def heap_prim(graph):
    """Returns a minimum spanning tree for the provided graph.
    
    Uses an efficient heap-based implementation.
    I've simplified the approach to not require delete operations,
    at the expense of more memory and a larger heap which is slower.
    
    Args:
        graph (dict): An undirected graph in adjacency list format.
        
    Returns:
        int: The total cost of the minimum spanning tree.
        list: The set of edges in the minimum spanning tree.    
    """
    connected = set() # The vertices that have been connected so far
    mst = [] # The edges in the minimum spanning tree
    cost = 0 # The total edge cost of the minimum spanning tree
    s = next(iter(graph)) # Arbitrarily select starting vertex
    connected.add(s) 
    heap = []
    for edge in graph[s]:
        heapq.heappush(heap, (edge[1], s, edge[0]))
    while len(heap) > 0:
        edge = heapq.heappop(heap) # With this approach, irrelevant edges get popped as well
        if edge[2] not in connected:
            connected.add(edge[2])
            mst.append((edge[1], edge[2], edge[0]))
            cost += edge[0]
            for e in graph[edge[2]]:
                if e[0] not in connected:
                    heapq.heappush(heap, (e[1], edge[2], e[0]))
    return cost, mst


def iterative_dfs(graph, start_vertex):
    '''Depth-first search for all vertices reachable from a vertex.

    Args:
        graph (dict): The graph in adjacency-list format.
        start_vertex (int): The vertex to search from.
 
    Returns:
        set: The vertices reachable from the start vertex.
    '''
    explored = set() 
    explored.add(start_vertex)
    stack = deque()
    stack.append(start_vertex)
    while len(stack) > 0:
        vertex = stack.pop()
        for edge in graph[vertex]:
            if edge[0] not in explored:
                explored.add(edge[0])
                stack.append(edge[0])
    return explored


def straightforward_kruskal(graph):
    """Returns a minimum spanning tree for the provided graph.
    
    Uses a straightforward and inefficient implementation.
    
    Args:
        graph (dict): An undirected graph in adjacency list format.
        
    Returns:
        int: The total cost of the minimum spanning tree.
        list: The set of edges in the minimum spanning tree.    
    """
    # Setup sorted edge list and MST container
    edge_list = []
    included = set() # For deduplicating edges
    for v1 in graph:
        included.add(v1)
        for edge in graph[v1]:
            if edge[0] not in included:
                edge_list.append((edge[1], v1, edge[0]))
    edge_list.sort()
    mst = defaultdict(list)
    
    # Initialize the MST with the lowest-cost edge
    start_edge = edge_list[0]
    mst[start_edge[1]].append((start_edge[2], start_edge[0]))
    mst[start_edge[2]].append((start_edge[1], start_edge[0]))
    cost = start_edge[0]
    for edge in edge_list[1:]:
        # Check for potential cycle
        if edge[2] not in mst or edge[2] not in iterative_dfs(mst, edge[1]):
            mst[edge[1]].append((edge[2], edge[0]))
            mst[edge[2]].append((edge[1], edge[0]))
            cost += edge[0]
    return cost, mst


class UnionFind():
    """A basic union find data structure.
    
    This data structure supports O(log(n)) connected component
    checking and unioning. 
    """
    def __init__(self, vertices):
        # Store (parent, size, vertex) tuples, init all as self-parent
        self._idx_map = {}
        self._contents = []
        for idx, v in enumerate(vertices):
            self._idx_map[v] = idx
            self._contents.append([idx, 1, v])
            
    def find(self, v):
        """Return the root parent for a vertex."""
        idx = self._idx_map[v]
        while idx != self._contents[idx][0]:
            idx = self._contents[idx][0]
        return self._contents[idx][2]
    
    def union(self, v1, v2, root_parents=None):
        """Union the connected components of two vertices."""
        # Find root parents, if not provided in call
        if root_parents is None:
            for idx, (v, rp) in enumerate(zip([v1, v2], root_parents)):
                root_parents[idx] = self.find(v)
        # Union the two connected components
        indices = [self._idx_map[rp] for rp in root_parents]
        sizes = [self._contents[idx][1] for idx in indices]
        if sizes[0] < sizes[1]:
            self._contents[indices[0]][0] = indices[1]
        else:
            self._contents[indices[1]][0] = indices[0]
        return None
        

def unionfind_kruskal(graph):
    """Returns a minimum spanning tree for the provided graph.
    
    Uses a more efficient union-find implementation.
    
    Args:
        graph (dict): An undirected graph in adjacency list format.
        
    Returns:
        int: The total cost of the minimum spanning tree.
        list: The set of edges in the minimum spanning tree.    
    """
    # Setup sorted edge list and MST container
    edge_list = []
    included = set() # For deduplicating edges
    for v1 in graph:
        included.add(v1)
        for edge in graph[v1]:
            if edge[0] not in included:
                edge_list.append((edge[1], v1, edge[0])) # (cost, v1, v2)
    edge_list.sort()
    mst = defaultdict(list)
    
    # Initialize union-find data structure
    uf = UnionFind(graph.keys())
    
    # Add lowest-cost edges that do not create cycles
    num_mst_edges = 0
    cost = 0
    for edge in edge_list:
        # Check for unique connected components
        rp_1, rp_2 = uf.find(edge[1]), uf.find(edge[2])
        if rp_1 != rp_2:
            mst[edge[1]].append((edge[2], edge[0]))
            mst[edge[2]].append((edge[1], edge[0]))
            num_mst_edges += 1
            cost += edge[0]
            uf.union(edge[1], edge[2], [rp_1, rp_2])
        if num_mst_edges == len(graph) - 1: # Stop early if mst complete
            break
    return cost, mst


def get_setup(test):
    '''Get setups for timing tests'''
    s = 'from __main__ import straightforward_prim, heap_prim, straightforward_kruskal, unionfind_kruskal, read_edge_file;'
    s += f" graph = read_edge_file('{test}');" 
    return s


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import timeit

    algs = [straightforward_prim, heap_prim, straightforward_kruskal, unionfind_kruskal]
    test_cases = [('problem15.9test.txt', 14, 6, 10),
                  ('problem15.9.txt', -3612829, 500, 2184)]

    # First, run correctness tests
    print('Starting correctness tests.')
    for test_case in test_cases:
        graph = read_edge_file(test_case[0])
        for alg in algs:
            cost, mst = alg(graph)
            assert cost == test_case[1], f'{alg} failed correctness for {test_case[0]}.'
    print('All correctness tests passed.')

    # Second, run timing tests
    print('Starting timing tests.')
    results = []
    for test_case in test_cases:
        results_row = {'test': test_case[0]}
        results_row['num_vertices'] = test_case[2]
        results_row['num_edges'] = test_case[3]
        for alg in algs:
            runtime = np.round(timeit.timeit(f'{alg.__name__}(graph)', setup=get_setup(test_case[0]), number=1), 5)
            results_row[alg.__name__] = runtime
        results.append(results_row)
    print('All timing tests complete:')
    print(pd.DataFrame(results))