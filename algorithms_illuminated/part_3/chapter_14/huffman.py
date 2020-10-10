'''
Varying implementations of Huffman's algorithm for optimal variable-length
encoding.
'''


from collections import deque
import heapq
import timeit
import numpy as np
import pandas as pd


class HuffmanEncoder():
    '''Optimal prefix-free encoding for a set of symbols.
    
    This finds and uses the optimal (minimum average bit length) variable-length
    encoding for a set of symbols, given the frequencies of the symbols.

    Parameters
    ----------
    symbols : list
        A list of (id, frequency) tuples for the symbols to encode.
        
    Attributes
    ----------
    symbols : list
        A list of (id, frequency) tuples for the symbols to encode.   
    '''
    def __init__(self, symbols):
        self.symbols = symbols
            
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def merge_trees(self, left_tree, right_tree):
        '''Merge two binary trees together by their roots.

        Parameters
        ----------
        left_tree : BinaryTree
            A BinaryTree to merge.
        right_tree : BinaryTree
            A BinaryTree to merge.

        Returns
        -------
        BinaryTree
            The merged BinaryTree.
        '''
        self.node_idx += 1
        new_root = Node(0, self.node_idx, left_child=left_tree.root, right_child=right_tree.root)
        return BinaryTree(new_root, total_value=left_tree.total_value + right_tree.total_value)
    
    def straightforward_encoding(self):
        '''Finds an optimal encoding using a straightforward implementation.'''
        # Create a forest of Binary Trees to work from
        self.node_idx = len(self.symbols) - 1
        forest = {i:BinaryTree(Node(symbol[0], symbol[1])) for i, symbol in enumerate(self.symbols)}
        # Incrementing index for new nodes and trees
        while len(forest) > 1:
            min_1, min_2 = float('inf'), float('inf')
            tree_1, tree_2 = None, None
            for i, tree in forest.items():
                if tree.total_value < min_1:
                    min_2 = min_1
                    tree_2 = tree_1
                    min_1 = tree.total_value
                    tree_1 = i
                elif tree.total_value < min_2:
                    min_2 = tree.total_value
                    tree_2 = i
            forest[self.node_idx] = self.merge_trees(forest[tree_1], forest[tree_2])
            del forest[tree_1]
            del forest[tree_2]
        for tree in forest.values():
            self.encoding_tree = tree
        return self
    
    def heap_encoding(self):
        '''Finds an optimal encoding using a heap implementation.'''
        # Create a forest of Binary Trees to work from
        self.node_idx = len(self.symbols) - 1
        forest = [BinaryTree(Node(symbol[0], symbol[1])) for symbol in self.symbols]
        heapq.heapify(forest)
        # Incrementing index for new nodes and trees
        while len(forest) > 1:
            tree_1 = heapq.heappop(forest)
            tree_2 = heapq.heappop(forest)
            heapq.heappush(forest, self.merge_trees(tree_1, tree_2))
        self.encoding_tree = forest[0]
        return self
    
    def sort_encoding(self):
        '''Finds an optimal encoding using a sorting + queue implementation.'''
        # Create a queue for base trees and a queue for merged trees
        self.node_idx = len(self.symbols) - 1
        q1 = deque(sorted([BinaryTree(Node(symbol[0], symbol[1])) for symbol in self.symbols]))
        q2 = deque([])
        # Sequentially merge min from both queues
        while len(q1) + len(q2) > 1:
            min_trees = []
            for _ in range(2):
                if len(q1) > 0 and (len(q2) == 0 or q1[0] < q2[0]):
                    min_trees.append(q1.popleft())
                else:
                    min_trees.append(q2.popleft())
            q2.append(self.merge_trees(min_trees[0], min_trees[1]))
        self.encoding_tree = q2[0]
        return self
    
    def get_enc_stats(self):
        '''Compute the average encoding length and other stats using BFS.'''
        queue = deque()
        node_depth = dict() # Keeps track of depth of each node
        total_length = 0
        min_len = float('inf')
        max_len = 0
        # Start BFS at the root
        queue.append(self.encoding_tree.root)
        node_depth[self.encoding_tree.root.node_id] = 0
        while len(queue) > 0:
            node = queue.popleft()
            children = [node._left_child, node._right_child]
            for child in children:
                if child is not None and child not in node_depth:
                    queue.append(child)
                    child_len = node_depth[node.node_id] + 1
                    if child.value > 0:
                        if child_len < min_len:
                            min_len = child_len
                        if child_len > max_len:
                            max_len = child_len
                    node_depth[child.node_id] = child_len
                    total_length += child_len * child.value # Only counts if node has symbol
        avg_len = round(total_length / self.encoding_tree.total_value, 3)
        return min_len, avg_len, max_len
            
        
class BinaryTree():
    '''A generic binary tree data structure.
    
    Parameters
    ----------
    node : A Node object to set as the root of the tree.
        
    Attributes
    ----------

    '''
    def __init__(self, node, total_value=None):
        self.root = node
        if total_value is None:
            self.total_value = node.value
        else:
            self.total_value = total_value
            
    def __repr__(self):
        return f'{self.__class__.__name__}'

    def __lt__(self, other):
        if self.total_value < other.total_value:
            return True
        else:
            return False

    def __eq__(self, other):
        if self.total_value == other.total_value:
            return True
        else:
            return False

    
    
class Node():
    '''A node for a binary tree.
    
    Parameters
    ----------
    value : object
        The value of the node.
    node_id : int
        Unique ID for search algorithm.
    left_child : object, optional
        The left child object for the node, default None.
    right_child : object, optional
        The right child object for the node, default None.
        
    Attributes
    ----------
    value : object
        The value of the node.
    node_id : int
        Unique ID for search algorithm. 
    left_child : object, optional
        The left child object for the node, default None.
    right_child : object, optional
        The right child object for the node, default None.
    '''
    def __init__(self, value, node_id=None, parent=None, left_child=None, right_child=None):
        self.value = value
        self.node_id = node_id
        self._parent = parent
        self._left_child = left_child
        self._right_child = right_child
            
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"id={self.node_id}, "
                f"value={self.value}, "
                f"left_child=(id:{getattr(self._left_child, 'node_id', None)}, value:{getattr(self._left_child, 'value', None)}), "
                f"right_child=(id:{getattr(self._right_child, 'node_id', None)}, value:{getattr(self._right_child, 'value', None)})")

    
def load_data(filename):
    '''Helper for loading symbol frequencies.'''
    symbols = []
    with open(filename) as f:
        for id, line in enumerate(f):
            if id == 0: # Get problem size
                size = int(line.replace('\n', ''))
            else:
                freq = int(line.replace('\n', ''))
                symbols.append((freq, id))
    return size, symbols


def get_setup(test_name):
    '''Get setups for timing tests'''
    s = 'from __main__ import HuffmanEncoder, load_data;'
    s += f' _, symbols = load_data("{test_name}");'
    s += f' huffman_encoder = HuffmanEncoder(symbols);' 
    return s


if __name__ == '__main__':
    # Declare set of test cases with solutions
    test_cases = [('problem14.6test1.txt', 2, 5, 10),
                  ('problem14.6test2.txt', 3, 6, 15), 
                  ('problem14.6.txt', 9, 19, 1000),
                  ('input_random_37_4000.txt', 11, 22, 4000),
                  ('input_random_45_10000.txt', 12, 24, 10000)]

    # Run correctness tests
    print(f'\nRunning correctness tests...')
    for test_case in test_cases:
        _, symbols = load_data(test_case[0])
        huffman_encoder = HuffmanEncoder(symbols)
        # Straightforward implementation
        stats = huffman_encoder.straightforward_encoding().get_enc_stats()
        assert stats[0] == test_case[1] and stats[2] == test_case[2], f'Straightforward implementation failed {test_case[0]}'
        stats = huffman_encoder.heap_encoding().get_enc_stats()
        assert stats[0] == test_case[1] and stats[2] == test_case[2], f'Heap implementation failed {test_case[0]}'
        stats = huffman_encoder.sort_encoding().get_enc_stats()
        assert stats[0] == test_case[1] and stats[2] == test_case[2], f'Sorting implementation failed {test_case[0]}'
        print(f'All implementations passed {test_case[0]}.')
    print('All correctness tests passed.')

    # Run timing tests
    print(f'\nRunning timing tests...')
    results = pd.DataFrame()
    for test_case in test_cases:
        print(f'Running test case {test_case[0]}...')
        temp_result = {'n': test_case[3]}
        implementations = ['straightforward', 'heap', 'sort']
        for implementation in implementations:
            temp_result[implementation] = np.round(timeit.timeit(f'huffman_encoder.{implementation}_encoding()', setup=get_setup(test_case[0]), number=1), 4)
        results = results.append(temp_result, ignore_index=True)
    print('All timing tests complete.')
    print(results[['n'] + implementations])




    #     # Random correctness tests
    #     mm_classes = [BuiltinHeapMedianMaintainer, SelectMedianMaintainer, BSTMedianMaintainer]
    #     for mm_class in mm_classes:
    #         seq, sol = get_mm_sequence_and_solution(n=n, seed=seed, replacement=replacement)
    #         mm = mm_class()
    #         assert test_median_maintainer(mm, seq, sol), f'{mm_class} failed correctness with replacement = {replacement}'
    #     print(f'\nAll tests passed with replacement = {replacement}.')

    #     # Provided correctness tests
    #     test_cases = [('problem11.3test.txt', '9335'), ('problem11.3.txt', '1213')]
    #     for filename, solution in test_cases:
    #         print(f'\nRunning correctness tests with {filename}...')
    #         seq = read_textfile(filename)
    #         for mm_class in mm_classes:
    #             mm = mm_class()
    #             assert test_km_sum(mm, seq, solution), f'{mm_class} failed correctness with {filename}.'
    #         print(f'\nAll tests passed with {filename}.')

    # # Timing tests
    # results = pd.DataFrame(data={'n':[1e2, 1e3, 1e4]})
    # seed = 1
    # for replacement in [True, False]:
    #     print(f'\nRunning timing tests with replacement = {replacement}...')
    #     for alg in ['builtin_heap', 'linear_selection', 'binary_search_tree']:
    #         temp_results = []
    #         for n in results['n'].values:
    #             temp_results.append(np.round(timeit.timeit('test_median_maintainer(mm, seq, sol)', setup=get_setup(alg, n, seed, replacement), number=1), 4))
    #         results[alg] = temp_results
    #     print(f'\nCompleted timing tests with replacement = {replacement}...')
    #     print(results.head(results.shape[0]))  