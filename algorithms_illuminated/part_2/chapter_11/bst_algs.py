import sys
import os
import pandas as pd
import numpy as np
import timeit
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from chapter_10.heap_algs import BuiltinHeapMedianMaintainer, SelectMedianMaintainer, get_mm_sequence_and_solution, test_median_maintainer

def load_data(filename):
    '''Helper for loading symbol frequencies.'''
    symbols = []
    with open(filename) as f:
        for line in f:
            freq = line.replace('\n', '')
            print(freq)
            break

            
class BinarySearchTree():
    '''A basic binary search tree without balancing considerations.

    Parameters
    ----------
    initial_contents: list (optional), default = None
        A list with the initial contents to load into the Binary Search Tree. 
        
    Attributes
    ----------
    _size : int
        The number of nodes in the tree.
    root : object
        The root object of the tree.
    '''
    def __init__(self, initial_contents=None):
        self._size = 0
        self.root = None
        if initial_contents is not None:
            for e in initial_contents:
                self.insert(e)
            
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_nodes={self._size})')
    
    def insert(self, value):
        '''Insert a new element into the Binary Search Tree.
        
        Parameters
        ----------
        value : object
            The value, with key if needed, to insert into the Binary Search Tree.
        
        Returns
        -------
        None
        '''

        if self._size == 0:
            self.root = Node(value)
        else:
            node = self.root
            while True:
                if value <= node.value:
                    if node._left_child is None:
                        node._left_child = Node(value, parent=node)
                        node = node._left_child
                        break
                    else:
                        node = node._left_child
                else:
                    if node._right_child is None:
                        node._right_child = Node(value, parent=node)
                        node = node._right_child
                        break
                    else:
                        node = node._right_child
            # Update subtree sizes
            while node._parent is not None:
                node = node._parent
                node.subtree_size += 1
        self._size += 1
        return None
       
    def search(self, value):
        '''Searches for a target element.
        
        Parameters
        ----------
        value : object
            The value to search for.
        
        Returns
        -------
        object
            A pointer to the target element if present, else None.
        '''
        node = self.root
        while True:
            if value == node.value:
                return node
            elif value < node.value:
                if node._left_child is None:
                    return None
                else:
                    node = node._left_child
            else:
                if node._right_child is None:
                    return None
                else:
                    node = node._right_child
                                        
    def minimum(self, node=None):
        '''Returns the minimum element.
        
        Parameters
        ----------
        node : object, optional.
            The node at the root of the selected subtree, default None.
        
        Returns
        -------
        object
            A pointer to the minimum element.
        '''
        if node is None:
            node = self.root
        while True:
            if node._left_child is None:
                return node
            else:
                node = node._left_child
                    
    def maximum(self, node=None):
        '''Returns the maximum element.

        Parameters
        ----------
        node : object, optional.
            The node at the root of the selected subtree, default None.

        Returns
        -------
        object
            A pointer to the maximum element.
        '''
        if node is None:
            node = self.root
        while True:
            if node._right_child is None:
                return node
            else:
                node = node._right_child
                    
    def predecessor(self, successor):
        '''Returns the predecessor of a given element.
        
        Parameters
        ----------
        successor : object
            The element to find the predecessor of.
        
        Returns
        -------
        object
            A pointer to the predecessor if it exists, else None.
        '''
        if successor._left_child is not None:
            return self.maximum(successor._left_child)
        else:
            node = successor
            parent = successor._parent
            while parent is not None:
                if parent._right_child is node:
                    return parent
                else:
                    node = parent
                    parent = node._parent
        return None
    
    def successor(self, predecessor):
        '''Returns the sucessor of a given element.
        
        Parameters
        ----------
        predecessor : object
            The element to find the successor of.
        
        Returns
        -------
        object
            A pointer to the successor if it exists, else None.
        '''
        if predecessor._right_child is not None:
            return self.minimum(successor._right_child)
        else:
            node = predecessor
            parent = predecessor._parent
            while parent is not None:
                if parent._left_child is node:
                    return parent
                else:
                    node = parent
                    parent = node._parent
        return None
    
    def output_sorted(self, node=None, result=None):
        '''Output the contents of a search tree in sorted order.
        
        Returns
        -------
        list
            The contents of the search tree in sorted order.
        '''        
        if node is None:
            node = self.root
            result = []
        if node._left_child is not None:
            self.output_sorted(node._left_child, result)
        result.append(node) # Base case
        if node._right_child is not None:
            self.output_sorted(node._right_child, result)
        return result
    
    def delete(self, value):
        '''Delete an element with the specified value, if present.
        
        Parameters
        ----------
        value : object
            The value of the element to delete, if present.
        
        Returns
        -------
        None
        '''        
        # First find the location of the element, if it exists
        node = self.search(value)
        if node is None:
            raise ValueError('The value is not in the tree.')
        else:
            # Scenario 1: No children
            if node._left_child is None and node._right_child is None:
                parent = node._parent
                if parent._left_child is node:
                    parent._left_child = None
                elif parent._right_child is node:
                    parent._right_child = None
            # Scenario 2: One child
            elif node._left_child is None:
                if parent._left_child is node:
                    parent._left_child = node._right_child
                elif parent._right_child is node:
                    parent._right_child = node._right_child
            elif node._right_child is None:
                if parent._left_child is node:
                    parent._left_child = node._left_child
                elif parent._right_child is node:
                    parent._right_child = node._left_child
            # Scenario 3: Two children
            else:
                # Get predecessor
                predecessor = self.predecessor(node)
                # Swap values with predecessor and decrement substree size
                node.value, predecessor.value = predecessor.value, node.value
                # Reassign node to predecessor before "deletion"
                node = predecessor
                # Delete node and move left child up (may be None)
                parent = node._parent
                if parent._left_child is node:
                    parent._left_child = node._left_child
                else:
                    parent._right_child = node._left_child
        self._size -= 1
        # Update subtree sizes
        while node._parent is not None:
            node = node._parent
            node.subtree_size -= 1        
        return None 

    def select(self, i, node=None):
        '''Select the ith order element from the BST.
        
        Parameters
        ----------
        i : int
            The order of element to select (1-based indexing).
            
        Returns
        -------
        object
            The ith order element.
        '''
        if (i <= 0) or (i > self._size):
            raise ValueError('Selected order element is invalid for this tree.')
        if node is None:
            node = self.root
        current_size = self._get_subtree_size(node._left_child) + 1
        if current_size == i:
            return node # Base case
        elif current_size > i:
            node = self.select(i, node=node._left_child)
        else:
            node = self.select(i - current_size, node=node._right_child)
        return node
    
    def _get_subtree_size(self, node):
        '''Getter for subtree size that handles missing nodes.'''
        if node is None:
            return 0
        else:
            return node.subtree_size
    
    
class Node():
    '''A node for a binary search tree.
    
    Parameters
    ----------
    value : object
        An comparable object with a key and value, if value is distinct from the key.
    parent : object, optional
        The parent object for the node, default None.
    left_child : object, optional
        The left child object for the node, default None.
    right_child : object, optional
        The right child object for the node, default None.
        
    Attributes
    ----------
    value : object
        The value for the node as an object with comparable key (value may be the key).
    subtree_size : int
        The size of the substree for which the node is a root.
    parent : object, optional
        The parent object for the node, default None.
    left_child : object, optional
        The left child object for the node, default None.
    right_child : object, optional
        The right child object for the node, default None.
    '''
    def __init__(self, value, parent=None, left_child=None, right_child=None):
        self.value = value
        self.subtree_size = 1
        self._parent = parent
        self._left_child = left_child
        self._right_child = right_child
            
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'value={self.value}, '
                f'subtree_size={self.subtree_size}, '
                f'parent={self._parent.value if self._parent is not None else None}, '
                f'left_child={self._left_child.value if self._left_child is not None else None}, '
                f'right_child={self._right_child.value if self._right_child is not None else None})')


class BSTMedianMaintainer():
    '''A class that maintains the median of a sequence of positive integers.
    
    This implementation uses a BST without balancing.
    
    Attributes
    ----------
    _tree: object
        A binary search tree used to store the integers.
    '''
    def __init__(self):
        self._tree = BinarySearchTree()
        
    def add(self, x):
        '''Add an element to the median maintainer.
        
        Parameters
        ----------
        x : int
            The element to add.
        
        Returns
        -------
        None
        '''
        self._tree.insert(x)
        return None

    @property
    def median(self):
        '''Returns the current median.'''
        size = self._tree._size
        if size % 2 == 1:
            return self._tree.select((size // 2) + 1).value
        else:
            return self._tree.select(size // 2).value


def read_textfile(filename):
    '''Read test dataset and return as a list.'''
    seq = []
    with open(filename) as f:
        for line in f:
           seq.append(int(line.replace('\n','')))
    return seq


def get_setup(alg, n, seed, replacement):
    '''Get setups for timing tests'''
    s = 'from __main__ import BuiltinHeapMedianMaintainer, SelectMedianMaintainer, BSTMedianMaintainer, get_mm_sequence_and_solution, test_median_maintainer;'
    s += f' seq, sol = get_mm_sequence_and_solution(n=int({n}), seed={seed}, replacement={replacement});' 
    if alg == 'builtin_heap':
        s += f' mm = BuiltinHeapMedianMaintainer();'
    if alg == 'linear_selection':
        s += f' mm = SelectMedianMaintainer();'
    if alg == 'binary_search_tree':
        s += f' mm = BSTMedianMaintainer();'
    return s


def test_km_sum(mm, seq, solution):
    '''Helper for testing sum of k-medians'''
    medians = []
    for e in seq:
        mm.add(e)
        medians.append(mm.median)
    return str(sum(medians))[-4:] == solution


if __name__ == '__main__':
    seed = 1
    n = 10000
    for replacement in [True, False]:
        print(f'\nRunning correctness tests with replacement = {replacement}...')
        # Random correctness tests
        mm_classes = [BuiltinHeapMedianMaintainer, SelectMedianMaintainer, BSTMedianMaintainer]
        for mm_class in mm_classes:
            seq, sol = get_mm_sequence_and_solution(n=n, seed=seed, replacement=replacement)
            mm = mm_class()
            assert test_median_maintainer(mm, seq, sol), f'{mm_class} failed correctness with replacement = {replacement}'
        print(f'\nAll tests passed with replacement = {replacement}.')

        # Provided correctness tests
        test_cases = [('problem11.3test.txt', '9335'), ('problem11.3.txt', '1213')]
        for filename, solution in test_cases:
            print(f'\nRunning correctness tests with {filename}...')
            seq = read_textfile(filename)
            for mm_class in mm_classes:
                mm = mm_class()
                assert test_km_sum(mm, seq, solution), f'{mm_class} failed correctness with {filename}.'
            print(f'\nAll tests passed with {filename}.')

    # Timing tests
    results = pd.DataFrame(data={'n':[1e2, 1e3, 1e4]})
    seed = 1
    for replacement in [True, False]:
        print(f'\nRunning timing tests with replacement = {replacement}...')
        for alg in ['builtin_heap', 'linear_selection', 'binary_search_tree']:
            temp_results = []
            for n in results['n'].values:
                temp_results.append(np.round(timeit.timeit('test_median_maintainer(mm, seq, sol)', setup=get_setup(alg, n, seed, replacement), number=1), 4))
            results[alg] = temp_results
        print(f'\nCompleted timing tests with replacement = {replacement}...')
        print(results.head(results.shape[0]))  