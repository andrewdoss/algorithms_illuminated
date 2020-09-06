'''
This module implements a Graph data structures and related operations.
'''
from collections import defaultdict


class Digraph:
    ''' A directed graph data structure with related operations.
    
    Note: edges are left implicit and therefore cannot have attributes.

    Parameters
    ----------
    filepath: str (optional), default = None
        The filepath, including filename, for an edge list to load from.    
        
    Attributes
    ----------
    adj_list_out: defaultdict of lists
        A mapping from nodes to adjacent (outbound edges) nodes.
    self.edge_count: int
        The number of edges in the graph.
    '''
    def __init__(self, filepath=None):
        self.adj_list_out = defaultdict(list)
        self.edge_count = 0
        if filepath is not None:
            self.read_text(filepath)
            
    def __repr__(self):
        return ('Digraph('
                f'num_vertices={len(self.adj_list_out)}, '
                f'num_edges={self.edge_count})')

    def read_text(self, filepath, append=False):
        ''' Construct or append a graph using an edge list from a text file.
        
        Parameters
        ----------
        filepath: str
            The filepath, including filename, to load edges from.
        append: bool (optional), default = False
            Flag for appending vs. overwriting (default) the current graph.
        '''
        if not append:
            self.adj_list_out = defaultdict(list)
        with open(filepath) as f:
            for edge in f:
                edge = edge.replace('\n', '').split(' ')
                source, sink = int(edge[0]), int(edge[1])
                self.adj_list_out[source].append(sink)
                self.edge_count += 1
        return self
    
class Graph(Digraph):
    '''An undirected graph data structure with related operations.
    