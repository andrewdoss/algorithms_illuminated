'''
    Quick sort implementations and selection sort comparison baseline. 
'''

import random
import pandas as pd
import numpy as np
import timeit
from collections import defaultdict
import matplotlib.pyplot as plt

def uniform_random_pivot(x, left, right):
    '''Select a pivot point uniformly at random.'''
    return random.randrange(left, right + 1)

def first_pivot(x, left, right):
    '''Select first element as pivot.'''
    return left

def last_pivot(x, left, right):
    '''Select last element as pivot.'''
    return right

def median_of_three_pivot(x, left, right):
    '''Selects median of first, middle, and last element.'''
    indices = [left, (left + right) // 2, right]
    for i in range(3):
        counts = [0] * 3
        for j in range(i+1, 3):
            if x[indices[i]] < x[indices[j]]:
                counts[j] += 1
            else:
                counts[i] += 1
        pivot = None
        for i in range(3):
            if counts[i] == 1:
                pivot = indices[i]
        if pivot is None:
            pivot = indices[0]  # If all 3 elements are equal, return first arbitrarily
        return pivot

def partition(x, left, right, comparison_counter):
    '''Partitions a sub-list around a pivot.
    
    The first element is assumed to be the pivot.
    
    Parameters
    ----------
    x : list
        A list that contains the sub-list with pivot as first element.
    left : int
        The index for the start of the sub-list within x.
    right : int
        The index for the end of the sub-list within x.
    Returns
    -------
    None
    '''
    if comparison_counter is not None:
        comparison_counter[0] += right - left # Count comparisons made against the pivot
    p = x[left]
    i = left + 1 # Pointer for start of the suffix
    for j in range(left + 1, right + 1):
        if x[j] < p:
            x[i], x[j] = x[j], x[i]
            i += 1
    x[left], x[i - 1] = x[i - 1], x[left] # Put pivot where it belongs
    return i - 1 # Report final pivot position

def quick_sort(x, left=None, right=None, select_pivot=uniform_random_pivot, comparison_counter=None):
    '''Sort a list of numbers or strings in place.
    
    Assumes a homogeneous input and that built-in comparison operators apply.
    
    Assumes that ascending order is desired.
    
    Parameters
    ----------
    x : list
        A list to be sorted in place.
    left : int
        The beginning of the current sub-list for sorting.
    right : int
        The end of the current sub-list for sorting.
    select_pivot: func
        A function for choosing pivots with signature (x, left, right).
    comparison_count: list
        An optional list with single int for holding a comparison counter.
        
    Note: this could benefit from OOP given the shared list and counter data.
        
    Returns
    -------
    None
        
    Based off pseudocode from Algorithms Illuminated Part 1
    ''' 
    # Initialize endpoints for outer-most call
    if left is None:
        left = 0
    if right is None:
        right = len(x) - 1
    # Increment comparison counter for non-partition work
    if comparison_counter is not None:
        if select_pivot.__name__ == 'median_of_three_pivot':
            comparison_counter[0] += 4
        else:
            comparison_counter[0] += 1
    # Base case
    if left >= right:
        return None
    # Choose pivot index        
    i = select_pivot(x, left, right)
    # Move pivot to beginning of current sub-array
    x[left], x[i] = x[i], x[left]
    # Partition and return new pivot position
    j = partition(x, left, right, comparison_counter)
    # Recursive calls
    quick_sort(x, left, j - 1, select_pivot, comparison_counter)
    quick_sort(x, j + 1, right, select_pivot, comparison_counter)
    
def selection_sort(x, comparison_counter=None):
    '''Sort a list of numbers or strings.
    
    Assumes a homogeneous input and that built-in comparison operators apply.
    
    Assumes that ascending order is desired.
    
    Parameters
    ----------
    x : list
        A list of numbers or strings to be sorted.
    comparison_count: list
        An optional list with single int for holding a comparison counter.
    Returns
    -------
    list
        A sorted list of the provided numbers or strings.
        
    Implementation: Introduction to Computation and Programming Using Python
    '''
    s = list(x) # Copy list before mutations 
    suffix_start = 0
    while suffix_start < len(x):
        for i in range(suffix_start, len(s)):
            if comparison_counter is not None:
                comparison_counter[0] += 1
            if s[i] < s[suffix_start]:
                s[suffix_start], s[i] = s[i], s[suffix_start]
        suffix_start += 1
    return s

def get_input(input_type, n):
    '''Get inputs for correctness tests'''
    if input_type == 'presorted':
        return np.arange(n)
    if input_type == 'reversed':
        return np.arange(n, 0, -1)
    if input_type == 'random':
        x = np.arange(n)
        return np.random.choice(x, int(n))
    if input_type == 'shuffled':
        x = np.arange(n) 
        np.random.shuffle(x)
        return x
    
def func_wrapper(method, x, comparison_counter=None):
    if method == 'selection_sort':
        return selection_sort(x, comparison_counter=comparison_counter)
    elif method == 'qs_first':
        return quick_sort(x, select_pivot=first_pivot, comparison_counter=comparison_counter)
    elif method == 'qs_last':
        return quick_sort(x, select_pivot=last_pivot, comparison_counter=comparison_counter)
    elif method == 'qs_random':
        return quick_sort(x, select_pivot=uniform_random_pivot, comparison_counter=comparison_counter)
    elif method == 'qs_med_of_3':
        return quick_sort(x, select_pivot=median_of_three_pivot, comparison_counter=comparison_counter)
    else:
        raise ValueError(f"method '{method}' not recognized.")

def check_correctness(input_array, output_array):
    '''Checks correctness against built in sorting routine'''
    output_array = np.array(output_array)
    reference_output = np.array(sorted(input_array))
    differences = np.sum(output_array != reference_output)
    if differences == 0:
        return 'passed'
    else:
        return 'failed'

# Helper to add results for a single problem size to collection
def store_results(n_size, num_trials, trial_results, all_results):
    '''Store comparison operation counts for a single problem size'''
    all_results['n_size'].append(n_size)
    for method, counts in trial_results.items():
        all_results[f'{method}_mean'].append(int(np.mean(trial_results[method])))
        all_results[f'{method}_std'].append(np.round(np.std(trial_results[method]), 1))
    all_results['num_trials'].append(num_trials)
    return None

if __name__ == '__main__':
    np.random.seed(1)

    # First, correctness tests
    print('Running correctness tests...')
    for input_type in ['presorted', 'reversed', 'random', 'shuffled']:
        for method in ['qs_first', 'qs_last', 'qs_random', 'qs_med_of_3']:
            x = get_input(input_type, 10)
            x_c = x.copy()
            func_wrapper(method, x_c)
            result = check_correctness(x, x_c)
            print(input_type, method, result)

    # Second, count comparison operations across different problem sizes and pivot selection strategies
    # Note: in the future, it might be interesting to compare strategies on partially sorted lists
    num_trials = 10
    input_type = 'random'
    n_sizes = np.logspace(1,4,4)
    methods = ['selection_sort', 'qs_first', 'qs_last', 'qs_random', 'qs_med_of_3']
    all_results = defaultdict(list)
    print('Running comparison count tests...')
    for n_size in n_sizes:
        print(f'Starting problem size {int(n_size)}.')
        trial_results = defaultdict(list)
        for trial in range(num_trials):
            x = get_input(input_type, n_size)
            for method in methods:
                if method != 'selection_sort' or trial == 0: # No need to run selection sort repeatedly
                    x_c = x.copy()
                    temp_counter = [0]
                    func_wrapper(method, x_c, temp_counter)
                    trial_results[method].append(temp_counter[0])
                else:
                    continue
        print(f'Ending problem size {n_size}.')
        store_results(n_size, num_trials, trial_results, all_results)
    print(pd.DataFrame(all_results))

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 10/1.6))
    for method in methods:
        ax.errorbar(all_results['n_size'], all_results[f'{method}_mean'], all_results[f'{method}_std'], label='method')
        ax.set(yscale='log')
        ax.legend()
    plt.show()