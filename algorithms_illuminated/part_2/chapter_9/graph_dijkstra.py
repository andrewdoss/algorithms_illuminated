'''
This module implements a graph data structure and implementations of Dijkstra's
algorithm.

Implementations are based on the exposition and pseudo-code in Algorithms Illuminated Part 2.
'''
from collections import defaultdict
import timeit
import numpy as np
from test_cases import *


class DigraphDijkstra:
    '''A directed graph data structure with related operations.

    Parameters
    ----------
    filepath: str (optional), default = None
        The filepath, including filename, for an edge list to load from. 
        
    Attributes
    ----------
    num_vertices : int 
        The number of vertices in the graph.
    num_edges : int
        The number of edges in the graph.
    '''
    def __init__(self, filepath=None):
        self.adj_list = defaultdict(list)
        self.edge_count = 0
        if filepath is not None:
            self.read_text(filepath)
            
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_vertices={len(self.adj_list)}, '
                f'num_edges={self.edge_count})')
    
    def reset_graph(self):
        '''Reset current adjacency list and edge count.'''
        self.adj_list = defaultdict(list)
        self.edge_count = 0
    
    def add_edge(self, vertex_a, vertex_b, weight):
        '''Adds an edge to the graph.
        
        Parameters
        ----------
        vertex_a : int
            The id for the first vertex.
        vertex_b : int
            The id for the second vertex.
        weight : int
            The weight/length of the edge.
        
        Returns
        -------
        None
        '''
        self.adj_list[vertex_a].append((vertex_b, weight))
        self.edge_count += 1
        return None

    def read_text(self, filepath, append=False):
        '''Construct or append the adjaceny list using an adjacency list from a text file.
        
        This implementation allows multiple edges between a pair of nodes.
        
        This class can also represent an undirected graph if both directions
        of each edge are provided in the input file. The class does not enforce
        undirectedness, however.
        
        Parameters
        ----------
        filepath : str
            The filepath, including filename, to load edges from.
        append : bool, optional
            Flag for appending vs. overwriting (default) the current graph.
        '''
        if not append:
            self.reset_graph()
        with open(filepath) as f:
            for line in f:
                line = line.replace('\n','').split('\t')
                if line[-1] == '':
                    line.pop()
                vertex_a = int(line[0])
                for e in line[1:]:
                    e = e.split(',')
                    vertex_b, weight = int(e[0]), int(e[1])
                    self.add_edge(vertex_a, vertex_b, weight)
        return self
    
    def dijkstra_baseline(self, start_vertex):
        '''Compute shortest path distances from a starting vertex to all other vertices.
        
        Assumes all edge lengths are non-negative.
        
        This is a very crude baseline implementation. Maintaining the set
        of currently "crossing" edges would likely speed this up significantly.
        
        Paramaters
        ----------
        start_vertex : int
            The index of the starting vertex for the search.
            
        Returns
        -------
        distances : dict
            A mapping from all vertices to their distance from start_vertex.
        '''
        distances = {start_vertex: 0}
        while True:
            min_length = float('inf')
            next_vertex = None
            for vertex_a in distances:
                for vertex_b, weight in self.adj_list[vertex_a]:
                    if vertex_b not in distances:
                        length_a_b = distances[vertex_a] + weight
                        if length_a_b < min_length:
                            min_length = length_a_b
                            next_vertex = vertex_b
            if next_vertex is not None:
                distances[next_vertex] = min_length
            else:
                break
        return distances
    
    def dijkstra_optimized(self, start_vertex):
        '''Compute shortest path distances from a starting vertex to all other vertices.
        
        Assumes all edge lengths are non-negative.
        
        This is a slightly more optimized implementation where the currently crossing
        edges are maintained to reduce edges checked per iteration.
        
        Paramaters
        ----------
        start_vertex : int
            The index of the starting vertex for the search.
            
        Returns
        -------
        distances : dict
            A mapping from all vertices to their distance from start_vertex.
        '''
        distances = {start_vertex: 0}
        crossing = {start_vertex: self.adj_list[start_vertex]}
        i = 0
        while True:
            min_length = float('inf')
            next_vertex = None
            for vertex_a in crossing:
                for vertex_b, weight in crossing[vertex_a]:
                    length_a_b = distances[vertex_a] + weight
                    if length_a_b < min_length:
                        min_length = length_a_b
                        next_vertex = vertex_b
            if next_vertex is not None:
                distances[next_vertex] = min_length
                # Maintain crossing edge dict
                affected_edges = self.adj_list[next_vertex]
                crossing[next_vertex] = [e for e in affected_edges if e[0] not in distances]
                for vertex_b, weight in affected_edges:
                    if vertex_b in crossing:
                        crossing[vertex_b].remove((next_vertex, weight))
                        if len(crossing[vertex_b]) == 0:
                            del crossing[vertex_b]
            else:
                break
        return distances


def check_distances(distances, solution_file, test_vertices=[7,37,59,82,99,115,133,165,188,197]):
    '''Helper for testing output against distances.
    
    Paramaters
    ----------
    distances : dict
        A mapping from vertices to their shortest path distance from vertex 1.
    solution_file : str
        A file containing the solutions for select vertices.
    test_vertices : list, optional
        The vertices to report shortest path distances to. 
    '''
    with open(solution_file) as f:
        solution = f.readline().replace('\n', '').split(',')
    result = [str(distances[v]) for v in test_vertices]
    return result == solution


def get_setup(test_name):
    '''Sets up a test for timing'''
    s = ""
    s += "from __main__ import DigraphDijkstra;"
    s += f"dg = DigraphDijkstra('tests/input_{test_name}.txt');"
    return s


if __name__ == '__main__':
    # First, check correctness and approximate runtime of various test cases
    for test in dijkstra_tests:
        dg = DigraphDijkstra(f"tests/input_{test['name']}.txt")
        baseline_distances = dg.dijkstra_baseline(test['start_vertex'])
        optimized_distances = dg.dijkstra_optimized(test['start_vertex'])
        assert check_distances(baseline_distances, f"tests/output_{test['name']}.txt"), f"Baseline failed {test['name']}"
        assert check_distances(optimized_distances, f"tests/output_{test['name']}.txt"), f"Optimized failed {test['name']}"  
    print('All correctness tests passed.\n')

    # Second, timing tests for problems of increasing size
    results = {}
    for problem in ['random_1_4', 'problem9.8', 'random_25_256']:
        for algorithm in ['dg.dijkstra_baseline(1)', 'dg.dijkstra_optimized(1)']:
            results[algorithm] = np.round(timeit.timeit(algorithm, setup=get_setup(problem), number=1), 4)
        print(f'Results for {problem}:\n', results, '\n')
    print('Timing tests complete.')
