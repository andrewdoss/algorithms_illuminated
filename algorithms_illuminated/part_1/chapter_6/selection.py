'''
    (Average) linear-time selection of ith order statistic and comparison to sorting. 
'''
import pandas as pd
import numpy as np
import time
from collections import defaultdict
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from chapter_1.sorting import merge_sort, get_input
from chapter_5.quicksort import quicksort, partition, uniform_random_pivot


def rselect(x, n, left=None, right=None):
    '''Select the nth order statistic from a list.
    
    The list is assumed to contain data that works with built-in comparison operators. 
    
    Parameters
    ----------
    x : list
        The list to search for an order statistic.
    n : int 
        The order statistic to select using 0-based indexing.
    left : int
        The starting index for the sub-list for the current call.
    right : int
        The ending index for the sub-list for the current call.
        
    Returns
    -------
    object : 
        The selected order statistic.
        
    Raises
    ------
        ValueError: If `n` is too large for `x`.
        ValueError: If `n` is negative.
    '''
    # Check input validity
    if n < 0:
        raise ValueError('The requested order statistic must be non-negative.')
    if n >= len(x):
        raise ValueError('The requested order statistic is too large for the input.')
    # Initialize endpoints for outer-most call
    if left is None:
        left = 0
    if right is None:
        right = len(x) - 1
    # Base case
    if left == right:
        result = x[left]
    else:
        i = uniform_random_pivot(x, left, right)
        x[left], x[i] = x[i], x[left] # Move pivot to beginning
        j = partition(x, left, right)
        if j == n:
            result = x[j]
        elif j > n:
            result = rselect(x, n, left=left, right= j - 1)
        else:
            result = rselect(x, n, left= j + 1, right=right)
    return result


def sort_select(x, n, sort_func=merge_sort, inplace=False):
    '''Sorting-based selection of nth order statistic.
    
    The list is assumed to contain data that works with built-in comparison operators. 
    
    Parameters
    ----------
    x : list
        A list that contains the sub-list with pivot as first element.
    n : int 
        The order statistic to select.
    sort_func : func
        The sorting function to use.
    inplace : bool
        Flag indicating whether sorting method sorts in-place.
    
    Returns
    -------
    obj
        The selected order statistic.
    '''
    if inplace:
        sort_func(x)
    else:
        x = sort_func(x)
    return x[n]


def func_wrapper(method, x, order_statistic):
    '''Wrapper for calling different selection methods.'''
    if method == 'rselect':
        return rselect(x, order_statistic)
    elif method == 'merge_sort':
        return sort_select(x, order_statistic, merge_sort, inplace=False)
    elif method == 'quicksort':
        return sort_select(x, order_statistic, quicksort, inplace=True)
    elif method == 'sorted':
        return sort_select(x, order_statistic, sorted, inplace=False)
    else:
        raise ValueError(f"method '{method}' not recognized.")

def store_results(n_size, num_trials, trial_results, all_results):
    '''Store comparison operation counts for a single problem size.'''
    all_results['n_size'].append(n_size)
    for method, counts in trial_results.items():
        all_results[f'{method}_mean'].append(int(np.mean(counts)))
        all_results[f'{method}_std'].append(np.round(np.std(counts), 1))
    all_results['num_trials'].append(num_trials)
    return None

if __name__ == '__main__':
    np.random.seed(1)

    # First, correctness tests using provided test cases and built-in sorting as the reference solution
    print('Running correctness tests...')
    print('Running provided test cases...')
    test_cases = {'test_case_1.txt': (5, 5469), 'test_case_2.txt': (50, 4715)}
    for file_name, (order_statistic, solution) in test_cases.items():
        x = pd.read_csv(file_name, header=None).iloc[:,0].to_list()
        if rselect(x, order_statistic - 1) == solution:
            print(f'{file_name} passed.')
        else:
            print(f'{file_name} failed.')

    print('Running random inputs...')
    for input_type in ['presorted', 'reversed', 'random', 'shuffled']:
        x = get_input(input_type, 10000)
        order_statistic = np.random.randint(0, len(x))
        if rselect(x.copy(), order_statistic) == sort_select(x.copy(), order_statistic):
            result = 'passed'
        else:
            result = 'failed'

    # Second, timing tests using random inputs with different methods
    print('\nRunning timing tests...')
    num_trials = 10
    input_type = 'random'
    n_sizes = list(np.logspace(5,7,3))
    n_sizes.append(2 * n_sizes[-1]) 
    methods = ['rselect', 'merge_sort', 'sorted']
    all_results = defaultdict(list)
    for n_size in n_sizes:
        trial_results = defaultdict(list)
        for trial in range(num_trials):
            x = get_input(input_type, n_size)
            order_statistic = np.random.randint(0, len(x))
            for method in methods:
                x_c = x.copy()
                start = time.time()
                func_wrapper(method, x_c, order_statistic)
                trial_results[method].append(time.time() - start) # using time is not ideal, but works for first cut
        print(f'Ending problem size {n_size}.')
        store_results(n_size, num_trials, trial_results, all_results)
    print('Timing tests complete.')
    print(pd.DataFrame(all_results))