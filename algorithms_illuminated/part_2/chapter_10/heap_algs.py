import random
import sys
import os
import heapq
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, 'part_1')))
from chapter_6.selection import rselect
import timeit
import pandas as pd


class Heap:
    '''A heap data structure with related operations.
    
    Parameters
    ----------
    initial_contents: list, optional
        The contents to initialize the heap with, default None.
    approach: str, optional
        The approach to use for constructing the initial heap, default 'optimal'.
    copy_input: bool, optional
        Whether to make a shallow-copy of the initial_contents list for the heap instance.
    
    Attributes
    ----------
    _contents: list
        The contents of the heap maintained in a (dynamic) array. 
    '''
    def __init__(self, initial_contents=None, approach='optimal', copy_input=False):
        if initial_contents is not None:
            if copy_input:
                initial_contents = list(initial_contents)
            self.heapify(initial_contents, approach)
        else:
            self._contents = []
        
    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'contents_size={self.size})')
    
    @property
    def size(self):
        return len(self._contents)
    
    def _get_element(self, idx):
        '''Get an element, given its current index.'''
        if (idx >= 0) and (idx < self.size):
            return self._contents[idx]
        else:
            return None
    
    def _set_element(self, x, idx):
        '''Set an element at a given index.'''
        self._contents[idx] = x
    
    def _get_child_idx(self, idx, left=True):
        '''Given an index, get the left or right child index.
        
        Parameters
        ----------
        idx : int
            The index for the parent.
        left : bool, optional
            Whether to get the index for the left child, default True. If False,
            the right child is retrieved.
            
        Returns
        -------
        int
            The index of the requested child.
        '''
        if left:
            return 2 * (idx + 1) - 1
        else:
            return 2 * (idx + 1)
        
    def _get_child(self, idx, left=True):
        '''Given an index, get the left or right child element.
        
        Parameters
        ----------
        idx : int
            The index for the parent.
        left : bool, optional
            Whether to get the left child, default True. If False,
            the right child is retrieved.
            
        Returns
        -------
        object
            The requested child element.
        '''        
        return self._get_element(self._get_child_idx(idx, left))
    
    def _get_parent_idx(self, idx):
        '''Given an index, get the parent index.
        
        Parameters
        ----------
        idx : int
            The index for the child.
            
        Returns
        -------
        int
            The index of the requested parent.
        '''
        return ((idx + 1) // 2) - 1
    
    def _get_parent(self, idx):
        '''Given a child index, get the parent element.
        
        Parameters
        ----------
        idx : int
            The index for the child.
            
        Returns
        -------
        object
            The requested parent element.
        '''        
        return self._get_element(self._get_parent_idx(idx))
    
    def insert(self, x):
        '''Insert a new element into the heap.
        
        Parameters
        ----------
        x : object
            The element to insert into the heap.
        
        Returns
        -------
        None
        '''
        # First add to the end, then fix violations of heap property
        self._contents.append(x)
        self._bubble_up(self.size - 1)    
        return None
    
    def _bubble_up(self, idx):
        '''Swap an element upwards until the heap property is restored.
        
        Parameters
        ----------
        idx : int
            The initial index for the element to "bubble up".
        
        Returns
        -------
        None
        '''
        x = self._get_element(idx)
        parent_idx = self._get_parent_idx(idx)
        parent = self._get_element(parent_idx)

        while (parent is not None) and (x < parent):
            self._set_element(x, parent_idx)
            self._set_element(parent, idx)
            idx = parent_idx
            parent_idx = self._get_parent_idx(idx)
            parent = self._get_element(parent_idx) 
            
        return None
    
    def extract_min(self):
        '''Extract the minimum element from the heap.
        
        Returns
        -------
        object
            The minimum element from the heap.
        '''
        if self.size > 0:
            self._contents[0], self._contents[-1] = self._contents[-1], self._contents[0]
            min_element = self._contents.pop()
            self._bubble_down(0)
            return min_element
        else:
            return None
        
    @property
    def peek(self):
        '''View the minimum element from the heap.
        
        Returns
        -------
        object
            The minimum element from the heap.
        '''
        if self.size > 0:
            return self._contents[0]
        else:
            return None        
    
    def find_min(self):
        '''Peak at the minimum element without removing it from the heap.'''
        return self._contents[0]
    
    def _bubble_down(self, idx):
        '''Swap an element downwards until the heap property is restored.
        
        Parameters
        ----------
        idx : int
            The initial index for the element to "bubble down".
        
        Returns
        -------
        None
        '''
        x = self._get_element(idx)
        min_child, min_child_idx = self._get_min_child(idx)
        while (min_child is not None) and (x > min_child):
            self._set_element(x, min_child_idx)
            self._set_element(min_child, idx)
            idx = min_child_idx
            min_child, min_child_idx = self._get_min_child(idx)
        return None
            
    def _get_min_child(self, idx):
        '''Given an index, return the minimum child and the child's index.
        
        Parameters
        ----------
        idx : int
            The index to retrieve a minimum child for.
        
        Returns
        -------
        None
        '''
        left_child = self._get_child(idx, left=True)
        right_child = self._get_child(idx, left=False)
        if ((left_child is not None) and (right_child is not None) and (left_child <= right_child)
            or (left_child is not None) and (right_child is None)):
            min_child = left_child
            min_child_idx = self._get_child_idx(idx, left=True)
        elif right_child is not None:
            min_child = right_child
            min_child_idx = self._get_child_idx(idx, left=False)
        else:
            min_child = None
            min_child_idx = None
        return min_child, min_child_idx
    
    def delete(self, idx):
        '''Given an index, delete the corresponding element from the heap.
        
        Parameters
        ----------
        idx : int
            The index to delete an element from.
        
        Returns
        -------
        None
        '''
        if idx < 0 or idx >= self.size:
            raise ValueError('Index not in heap.')
            
        self._contents[idx], self._contents[-1] = self._contents[-1], self._contents[idx]
        self._contents.pop()
        x = self._get_element(idx)
        parent = self._get_parent(idx)
        min_child, _ = self._get_min_child(idx)
        
        if parent is not None and x < parent:
            self._bubble_up(idx)
        elif min_child is not None and x > min_child:
            self._bubble_down(idx)
        return None
    
    def heapify(self, init_contents, approach='optimal'):
        '''Given an array, build a heap using varying methods.
        
        optimal : start from root and work down with repeated calls to bubble-down.
        
        suboptimal : start from leaves and work up with repeated calls to bubble-up.
        
        insert : repeatedly insert elements (O(nlogn))
        
        Credit: https://stackoverflow.com/questions/9755721/how-can-building-a-heap-be-on-time-complexity
        
        Parameters
        ----------
        init_contents : list
            The array of elements to heapify.
        
        Returns
        -------
        None
        '''
        if approach == 'optimal':
            self._contents = init_contents
            for idx in range(len(init_contents) - 1, -1, -1):
                self._bubble_down(idx)
        elif approach == 'suboptimal':
            self._contents = init_contents
            for idx in range(len(init_contents)):
                self._bubble_up(idx)
        elif approach == 'insert':
            self._contents = []
            for e in init_contents:
                self.insert(e)
        else:
            raise ValueError(f'Heapify approach {approach} not recognized.')
    
    def _check_heap_invariant(self):
        '''Debugging helper that checks heap property.'''
        for idx, e in enumerate(self._contents):
            left_child = self._get_child(idx, left=True)
            right_child = self._get_child(idx, left=False)
            assert (left_child is None) or (left_child >= e), f'Heap property violated for left child of index {idx}.'
            assert (right_child is None) or (right_child >= e), f'Heap property violated for right child of index {idx}.'
        return True
    

class HeapMedianMaintainer():
    '''A class that maintains the median of a sequence of positive integers.
    
    This implementation uses heaps.
    
    The max heap is implemented using a min heap with reversed sign elements.
    
    Attributes
    ----------
    _min_heap: Heap
        A min heap for maintaining the upper half of the sequence.
    _max_heap: Heap
        A max heap for maintaining the lower half of the sequence.
    '''
    def __init__(self):
        self._min_heap = Heap()
        self._max_heap = Heap()
        
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
        if self._max_heap.size == 0:
            self._max_heap.insert(-1 * x)
        elif self._min_heap.size == 0:
            if x < -1 * self._max_heap.peek: # Check for rebalancing
                self._min_heap.insert(-1 * self._max_heap.extract_min()) 
                self._max_heap.insert(-1 * x)
            else:
                self._min_heap.insert(x)
        elif x > -1 * self._max_heap.peek:
            self._min_heap.insert(x)
            # Rebalance if needed
            if self._min_heap.size > self._max_heap.size + 1: 
                y = self._min_heap.extract_min()
                self._max_heap.insert(-1 * y)
        else:
            self._max_heap.insert(-1 * x)
            if self._max_heap.size > self._min_heap.size + 1: 
                y = self._max_heap.extract_min()
                self._min_heap.insert(-1 * y)
                
    @property
    def median(self):
        '''Returns the current median.'''
        if self._max_heap.size == self._min_heap.size:
            return sorted((-1 * self._max_heap.peek, self._min_heap.peek))[0]
        elif self._max_heap.size > self._min_heap.size:
            return -1 * self._max_heap.peek
        else: 
            return self._min_heap.peek
        

class BuiltinHeapMedianMaintainer():
    '''A class that maintains the median of a sequence of positive integers.
    
    This implementation uses heaps (built-in implementation).
    
    The max heap is implemented using a min heap with reversed sign elements.
    
    Attributes
    ----------
    _min_heap: list
        A min heap for maintaining the upper half of the sequence.
    _max_heap: list
        A max heap for maintaining the lower half of the sequence.
    '''
    def __init__(self):
        self._min_heap = []
        self._max_heap = []
        
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
        if len(self._max_heap) == 0:
            heapq.heappush(self._max_heap, -1 * x)
        elif len(self._min_heap) == 0:
            if x < -1 * self._max_heap[0]: # Check for rebalancing 
                heapq.heappush(self._min_heap, -1 * heapq.heappop(self._max_heap))
                heapq.heappush(self._max_heap, -1 * x)
            else:
                heapq.heappush(self._min_heap, x)
        elif x > -1 * self._max_heap[0]:
            heapq.heappush(self._min_heap, x)
            # Rebalance if needed
            if len(self._min_heap) > len(self._max_heap) + 1: 
                y = heapq.heappop(self._min_heap)
                heapq.heappush(self._max_heap, -1 * y)
        else:
            heapq.heappush(self._max_heap, -1 * x)
            if len(self._max_heap) > len(self._min_heap) + 1: 
                y = heapq.heappop(self._max_heap)
                heapq.heappush(self._min_heap, -1 * y)
                
    @property
    def median(self):
        '''Returns the current median.'''
        if len(self._max_heap) == len(self._min_heap):
            return sorted((-1 * self._max_heap[0], self._min_heap[0]))[0]
        elif len(self._max_heap) > len(self._min_heap):
            return -1 * self._max_heap[0]
        else: 
            return self._min_heap[0]
                

class SelectMedianMaintainer():
    '''A class that maintains the median of a sequence of positive integers.
    
    This implementation uses linear-time order-statistic selection.
    
    Attributes
    ----------
    _contents: list
        A list of elements added so far.
    '''
    def __init__(self):
        self._contents = []
        
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
        self._contents.append(x)
        return None

    @property
    def median(self):
        '''Returns the current median.'''
        size = len(self._contents)
        if size % 2 == 1:
            return rselect(self._contents, size // 2)
        else:
            return rselect(self._contents, size // 2 - 1)


def get_mm_sequence_and_solution(n, seed=None, replacement=False):
    '''Get a sequence and solution for the median maintenance problem.

    Parameters
    ----------
    n : int
        Length of the sequence to maintain a median for.
    seed : int, optional
        Seed for the random sequence generator.
        
    Returns
    -------
    list
        The randomly generated input sequence.
    list
        The sequence of correct median elements.
    '''
    if seed is not None:
        random.seed(seed)
    seq = [i for i in range(1, n + 1)]
    if replacement:
        seq = random.choices(seq, k=len(seq))
    else:
        seq = random.sample(seq, len(seq))
    compilation = []
    solution = []
    for e in seq:
        compilation.append(e)
        compilation.sort()
        length = len(compilation)
        if length % 2 == 0:
            solution.append(compilation[(length // 2) - 1])
        else:
            solution.append(compilation[length // 2])
    return seq, solution


def test_median_maintainer(mm, seq, sol):
    '''Helper for testing a median maintainer.

    Parameters
    ----------
    mm : object
        The median mainter object to test.
    seq : list
        The sequence to maintain a median for.
    seq : list
        The sequence of correct medians at each iteration. 
        
    Returns
    -------
    bool
        The success of the median maintainer.
    '''    
    for e_in, e_out in zip(seq, sol):
        mm.add(e_in)
        if mm.median != e_out:
            return False
    return True


def heap_sort(x):
    '''Sort an input using a custom heap.
    
    Assumes ascending order and that built-in comparisons are valid.
    
    Parameters
    ----------
    x : list
        The input to be sorted.
    
    Returns
    -------
    list
        The input in sorted order.
    '''
    heap = Heap(x)
    return [heap.extract_min() for i in range(len(x))]


def heap_sort_builtin(x):
    '''Sort an input using a builtin heap.
    
    Assumes ascending order and that built-in comparisons are valid.
    
    Parameters
    ----------
    x : list
        The input to be sorted.
    
    Returns
    -------
    list
        The input in sorted order.
    '''
    heapq.heapify(x)
    return [heapq.heappop(x) for i in range(len(x))]


def get_setup(alg, n, seed, replacement):
    '''Get setups for timing tests'''
    s = 'from __main__ import HeapMedianMaintainer, BuiltinHeapMedianMaintainer, SelectMedianMaintainer, get_mm_sequence_and_solution, test_median_maintainer;'
    s += f' seq, sol = get_mm_sequence_and_solution(n=int({n}), seed={seed}, replacement={replacement});' 
    if alg == 'custom_heap':
        s += f' mm = HeapMedianMaintainer();'
    if alg == 'builtin_heap':
        s += f' mm = BuiltinHeapMedianMaintainer();'
    if alg == 'linear_selection':
        s += f' mm = SelectMedianMaintainer();'
    return s


if __name__ == '__main__':
    seed = 1
    n = 10000
    for replacement in [True, False]:
        print(f'\nRunning correctness tests with replacement = {replacement}...')
        # Correctness tests
        for mm_class in [HeapMedianMaintainer, BuiltinHeapMedianMaintainer, SelectMedianMaintainer]:
            seq, sol = get_mm_sequence_and_solution(n=n, seed=seed, replacement=replacement)
            mm = mm_class()
            assert test_median_maintainer(mm, seq, sol), f'{mm_class} failed correctness with replacement = {replacement}'
        print(f'\nAll tests passed with replacement = {replacement}.')

    # Timing tests
    results = pd.DataFrame(data={'n':[1e2, 1e3, 1e4]})
    seed = 1
    for replacement in [True, False]:
        print(f'\nRunning timing tests with replacement = {replacement}...')
        for alg in ['custom_heap', 'builtin_heap', 'linear_selection']:
            temp_results = []
            for n in results['n'].values:
                temp_results.append(np.round(timeit.timeit('test_median_maintainer(mm, seq, sol)', setup=get_setup(alg, n, seed, replacement), number=1), 4))
            results[alg] = temp_results
        print(f'\nCompleted timing tests with replacement = {replacement}...')
        print(results.head(results.shape[0]))