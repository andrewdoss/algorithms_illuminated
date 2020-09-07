'''
This module implements a graph data structures and related operations.

Implementations are based on the exposition and pseudo-code in Algorithms Illuminated Part 2.
'''
from collections import defaultdict, deque

class Digraph:
    '''A directed graph data structure with related operations.
    
    Note: edges are left implicit and therefore cannot have attributes.

    Parameters
    ----------
    filepath: str (optional), default = None
        The filepath, including filename, for an edge list to load from.    
    '''
    def __init__(self, filepath=None):
        if filepath is not None:
            self.read_text(filepath)
            
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_vertices={len(self.adj_list)}, '
                f'num_edges={self.edge_count})')

    def read_text(self, filepath, append=False):
        '''Construct or append a digraph using an edge list from a text file.
        
        Note: this implementation allows multiple edges between a pair of nodes.
        
        Parameters
        ----------
        filepath: str
            The filepath, including filename, to load edges from.
        append: bool, optional
            Flag for appending vs. overwriting (default) the current graph.
            
        Attributes
        ----------
        adj_list: defaultdict of lists
            A mapping from nodes to adjacent (outbound edges) nodes.
        edge_count: int
            The number of edges in the graph.
        '''
        if not append:
            self.adj_list = defaultdict(list)
            self.edge_count = 0
        with open(filepath) as f:
            for edge in f:
                edge = edge.replace('\n', '').split(' ')
                vertex_a, vertex_b = int(edge[0]), int(edge[1])
                self.add_edge(vertex_a, vertex_b)
        return self
    
    def breadth_first_search(self, start_vertex):
        '''Breadth-first search for all vertices reachable from a vertex.
        
        Computes distances assuming unit length for all edges.
        
        Parameters
        ----------
        start_vertex: int
            The vertex to search from.
        Returns
        -------
        dict
            The vertices reachable from the start_vertex as keys with distances if selected.
        '''
        queue = deque()
        explored = dict()
        for vertex in self.adj_list:
            explored[vertex] = float('inf')
        queue.append(start_vertex)
        explored[start_vertex] = 0
        while len(queue) > 0:
            vertex_a = queue.popleft()
            for vertex_b in self.adj_list.get(vertex_a, []):
                if explored[vertex_b] == float('inf'):
                    queue.append(vertex_b)
                    explored[vertex_b] = explored[vertex_a] + 1
        
        to_remove = []
        for vertex in explored:
            if explored[vertex] == float('inf'):
                to_remove.append(vertex)
        for vertex in to_remove:
            del explored[vertex]
        return explored
    
    def depth_first_search(self, start_vertex):
        '''Depth-first search for all vertices reachable from a vertex.
        
        Parameters
        ----------
        start_vertex: int
            The vertex to search from.
        Returns
        -------
        set
            The vertices reachable from the source.
        '''
        stack = deque()
        explored = set()
        stack.append(start_vertex)
        while len(stack) > 0:
            vertex_a = stack.pop()
            explored.add(vertex_a)
            for vertex_b in self.adj_list.get(vertex_a, []):
                if vertex_b not in explored:
                    stack.append(vertex_b)
        return explored
    
    def recursive_dfs(self, start_vertex, explored=None, topo_sort=False):
        '''Recursive depth-first search for all vertices reachable from a vertex.
        
        Parameters
        ----------
        start_vertex: int
            The vertex to search from.
        explored: set, optional
            The vertices that have already been explored.
        topo_sort: bool, optional
            Whether to return a topological ordering (only valid if a DAG). 
        Returns
        -------
        dict
            The vertices reachable from the source with topo rank, if requested.
        ''' 
        if explored is None:
            explored = dict()
        explored[start_vertex] = None
        for vertex_b in self.adj_list[start_vertex]:
            if vertex_b not in explored:
                explored = self.recursive_dfs(vertex_b, explored, topo_sort)
        if topo_sort:
            explored[start_vertex] = explored['current_rank']
            explored['current_rank'] -= 1
        return explored
        
    def topo_sort(self):
        '''Computes a topological ordering of a DAG.
        
        Assumes current graph is a DAG.
        
        Returns
        -------
        dict
            A mapping from vertices to topological ranks.
        '''
        vertex_to_order = dict()
        vertex_to_order['current_rank'] = len(self.adj_list) # Temporarily use to store rank
        for vertex in self.adj_list:
            if vertex not in vertex_to_order:
                vertex_to_order = self.recursive_dfs(vertex, vertex_to_order, topo_sort=True)
        del vertex_to_order['current_rank'] 
        return vertex_to_order
    
    def add_edge(self, vertex_a, vertex_b):
        '''Adds an edge to the graph.
        
        Parameters
        ----------
        vertex_a: int
            The id for the first vertex.
        vertex_b: int
            The id for the second vertex.
        
        Returns
        -------
        None
        '''
        self.adj_list[vertex_a].append(vertex_b)
        if vertex_b not in self.adj_list:
            self.adj_list[vertex_b] = []
        self.edge_count += 1
        return None
    
class Graph(Digraph):
    '''An undirected graph data structure with related operations.
    
    Note: edges are left implicit and therefore cannot have attributes.

    Parameters
    ----------
    filepath: str (optional), default = None
        The filepath, including filename, for an edge list to load from.    
    '''
    
    def add_edge(self, vertex_a, vertex_b):
        '''Adds an edge to the graph.
        
        Parameters
        ----------
        vertex_a: int
            The id for the first vertex.
        vertex_b: int
            The id for the second vertex.
        
        Returns
        -------
        None
        '''
        self.adj_list[vertex_a].append(vertex_b)
        self.adj_list[vertex_b].append(vertex_a)
        self.edge_count += 1
        return None
    
    def undirected_connected_components(self):
        '''Computes the connected components of an undirected graph.
        
        Returns
            dict
                A mapping from vertices to connected components.
        '''
        vertex_to_ucc = dict()
        ucc_num = 0
        for vertex in self.adj_list:
            if vertex not in vertex_to_ucc:
                explored = self.breadth_first_search(vertex)
                for v in explored:
                    vertex_to_ucc[v] = ucc_num
                ucc_num += 1
        return vertex_to_ucc
    
def check_topo_order(adj_list, vertex_to_order):
    '''Checks correctness of a topological ordering.
    
    Note: assumes that the graph is directed-acyclic.
    
    Parameters
    ----------
    adj_list: dict
        Mapping from vertices to adjacent vertices.
    vertex_to_order: dict
        Mapping from vertices to topological orders.
        
    Returns
    -------
    bool
        Flag for correctness of topological ordering.
    '''
    for vertex_a in adj_list:
        for vertex_b in adj_list.get(vertex_a, []):
            if vertex_to_order[vertex_b] <= vertex_to_order[vertex_a]:
                print('Incorrect ordering', vertex_a, vertex_b)
                return False
    return True