'''
This module implements a graph data structures and related operations.

Implementations are based on the exposition and pseudo-code in Algorithms Illuminated Part 2.
'''
from collections import defaultdict, deque
from test_cases import *


class Digraph:
    '''A directed graph data structure with related operations.
    
    Note: edges are left implicit and therefore cannot have attributes.

    Parameters
    ----------
    filepath: str (optional), default = None
        The filepath, including filename, for an edge list to load from. 
        
    Attributes
    ----------
    num_vertices: int 
        The number of vertices in the graph.
    num_edges: int
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

    def read_text(self, filepath, append=False):
        '''Construct or append the adjaceny list using an edge list from a text file.
        
        Note: this implementation allows multiple edges between a pair of nodes.
        
        Parameters
        ----------
        filepath: str
            The filepath, including filename, to load edges from.
        append: bool, optional
            Flag for appending vs. overwriting (default) the current graph.
        '''
        if not append:
            self.reset_graph()
        with open(filepath) as f:
            for edge in f:
                edge = edge.replace('\n', '').split(' ')
                vertex_a, vertex_b = int(edge[0]), int(edge[1])
                self.add_edge(vertex_a, vertex_b)
        return self
    
    def set_adj_list(self, adj_list, append=False):
        '''Set the adjacency list using an existing Python object.
        
        Parameters
        ----------
        adj_list: defaultdict(list)
            The adjaceny list mapping between vertices.
        append: bool, optional
            Flag for appending vs. overwriting (default) the current graph.
        '''
        if not append:
            self.reset_graph()
        for vertex_a in adj_list:
            for vertex_b in adj_list[vertex_a]:
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
    
    def depth_first_search(self, start_vertex, explored=None, topo_sort=False, scc=None, recursive=False):
        '''Depth-first search for all vertices reachable from a vertex.
        
        Parameters
        ----------
        start_vertex: int
            The vertex to search from.
        explored: set, optional
            The vertices that have already been explored.
        topo_sort: bool, optional
            Whether to return a topological ordering (only valid if a DAG).
        scc: int, optional
            The current SCC index if running as part of Kosaraju.
        recursive: bool, optional
            Whether to run the recursive implementation of DFS, defaults to False
        Returns
        -------
        dict
            The vertices reachable from the source with topo rank, if requested.
        '''
        if explored is None:
            explored = dict()
        if recursive:
            method = self.recursive_dfs
        else:
            method = self.iterative_dfs
        return method(start_vertex, explored, topo_sort, scc)
    
    def iterative_dfs(self, start_vertex, explored=None, topo_sort=False, scc=None):
        '''Depth-first search for all vertices reachable from a vertex.
        
        Parameters
        ----------
        start_vertex: int
            The vertex to search from.
        explored: set, optional
            The vertices that have already been explored.
        topo_sort: bool, optional
            Whether to return a topological ordering (only valid if a DAG).
        scc: int, optional
            The current SCC index if running as part of Kosaraju.
            
        Returns
        -------
        dict
            The vertices reachable from the source with topo rank, if requested.
        '''
        explored[start_vertex] = None
        stack = deque()
        stack.append(start_vertex)
        while len(stack) > 0:
            vertex_a = stack[-1]
            for vertex_b in self.adj_list.get(vertex_a, []):
                if vertex_b not in explored:
                    stack.append(vertex_b)
                    explored[vertex_b] = None
            if vertex_a == stack[-1]:
                stack.pop()
                if topo_sort:
                    explored[vertex_a] = explored['current_rank']
                    explored['current_rank'] -= 1
                elif scc is not None:
                    explored[vertex_a] = scc
        return explored
    
    def recursive_dfs(self, start_vertex, explored=None, topo_sort=False, scc=None):
        '''Recursive depth-first search for all vertices reachable from a vertex.
        
        Parameters
        ----------
        start_vertex: int
            The vertex to search from.
        explored: set, optional
            The vertices that have already been explored.
        topo_sort: bool, optional
            Whether to return a topological ordering (only valid if a DAG).
        scc: int, optional
            The current SCC index if running as part of Kosaraju.
            
        Returns
        -------
        dict
            The vertices reachable from the source with topo rank, if requested.
        '''
        explored[start_vertex] = None
        for vertex_b in self.adj_list[start_vertex]:
            if vertex_b not in explored:
                explored = self.recursive_dfs(vertex_b, explored, topo_sort, scc=scc)
        if topo_sort:
            explored[start_vertex] = explored['current_rank']
            explored['current_rank'] -= 1
        else:
            explored[start_vertex] = None
        if scc is not None:
            explored[start_vertex] = scc
        return explored
    
    def topo_sort(self, recursive=False):
        '''Computes a topological ordering of a DAG.
        
        Assumes current graph is a DAG.
        
        Parameters
        ----------
        recursive: bool, optional
            Whether to use the recursive implementation of DFS, defaults to False.
        
        Returns
        -------
        dict
            A mapping from vertices to topological ranks.
        '''
        vertex_to_order = dict()
        vertex_to_order['current_rank'] = len(self.adj_list) # Temporarily use to store rank
        if recursive:
            method = self.recursive_dfs
        else:
            method = self.iterative_dfs
        for vertex in self.adj_list:
            if vertex not in vertex_to_order:
                vertex_to_order = method(vertex, vertex_to_order, topo_sort=True)
        del vertex_to_order['current_rank'] 
        return vertex_to_order        
    
    def kosaraju(self, recursive=False):
        '''Uses depth-first search to find the SCCs of a Digraph.
        
        Parameters
        ----------
        recursive : bool, optional
            Whether to use the iterative or recrusive implementation of DFS, defaults to False.
        
        Returns
        -------
        dict :
            A mapping from vertices to SCCs.
        '''
        # Compute a reversed graph 
        adj_list_rev = self.reverse_graph()
        
        # Get an ordering for the reversed graph
        temp_dg = Digraph()
        temp_dg.set_adj_list(adj_list_rev) 
        search_order = temp_dg.topo_sort(recursive)
        search_order = sorted(list(search_order.items()), key=lambda x: x[1])
        
        # Get DFS implementation
        if recursive:
            method = self.recursive_dfs
        else:
            method = self.iterative_dfs
        
        # Iterate over ordered vertices to discover SCCs
        scc_num = 0
        scc_map = dict()
        for vertex, order in search_order:
            if vertex not in scc_map:
                scc_map = method(vertex, explored=scc_map, scc=scc_num)
                scc_num += 1
        return scc_map
                
    def reverse_graph(self):
        '''A helper for reversing the edges of a Digraph.
        
        Returns
        -------
        dict :
            An adjacency list for the reversed graph.
        '''
        adj_list_rev = defaultdict(list)
        for vertex_a in self.adj_list:
            for vertex_b in self.adj_list[vertex_a]:
                adj_list_rev[vertex_b].append(vertex_a)
                if vertex_a not in adj_list_rev:
                    adj_list_rev[vertex_a] = []
        return adj_list_rev
    
    def top_scc_size(self, n=5, recursive=False):
        '''Uses Kosaraju to get sizes of the top n SCCs.
        
        Parameters
        ----------
        n : int, optional
            The number of largest SCCs to get size for, default 5.
        recursive: bool, optional
            Whether to use the recursive implementation of DFS.
            
        Returns
        -------
        list
            A list of sizes for top n largest SCCs in descending order.
        '''
        scc_sizes = defaultdict(int)
        vertex_to_scc = self.kosaraju(recursive)
        for value in vertex_to_scc.values():
            scc_sizes[value] += 1
        results = sorted(list(scc_sizes.values()), reverse=True)[:n]
        results = results + [0] * max(0, (n - len(results))) # pad to n elements
        return results
    
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


if __name__ == '__main__':
    # First, check correctness of basic search with BFS and DFS
    for test in search_tests:
        graph = Digraph(filepath=test['file'])
        assert set(graph.breadth_first_search(test['start_vertex']).keys()) == test['solution'], 'BFS failed ' + test['name']
        assert set(graph.depth_first_search(test['start_vertex']).keys()) == test['solution'], 'BDFS (iter) failed ' + test['name']
        assert set(graph.depth_first_search(test['start_vertex'], recursive=True).keys()) == test['solution'], 'BDFS (recur) failed ' + test['name']
    print('All search tests passed.')

    # Second, check correctness of BFS (unit) distance search
    for test in bfs_unit_distance_tests:
        graph = Digraph(filepath=test['file'])
        assert graph.breadth_first_search(test['start_vertex']) == test['solution'], 'BFS failed ' + test['name']
    print('All BFS unit distance tests passed.')

    # Third, check correctness of BFS UCC search
    for test in bfs_ucc_tests:
        graph = Graph(filepath=test['file'])
        result = graph.undirected_connected_components()
        pass_flag = True
        for scc in test['solution']:
            scc_num = result[scc[0]]
            for e in scc:
                if result[e] != scc_num:
                    pass_flag = False
        assert pass_flag, 'Failed ' + test['name']
    print('All BFS UCC search tests passed.')

    # Fourth, check correctness when sorting a DAG
    for test in topo_sort_tests:
        graph = Digraph(filepath=test['file'])
        assert check_topo_order(graph.adj_list, graph.topo_sort(recursive=False)), 'Iterative DFS failed ' + test['name']
        assert check_topo_order(graph.adj_list, graph.topo_sort(recursive=True)), 'Recursive DFS failed ' + test['name']
    print('All DFS topological sorting tests passed.')

    # Fifth, check correctness of finding SCCs
    for test in scc_search_tests:
        graph = Digraph(filepath=test['file'])
        assert graph.top_scc_size(recursive=False) == test['solution'], 'Iterative SCC search failed ' + test['name']
        assert graph.top_scc_size(recursive=True) == test['solution'], 'Recursive SCC search failed ' + test['name']
    print('All SCC search tests passed.') 
