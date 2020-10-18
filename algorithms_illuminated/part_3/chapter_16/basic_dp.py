'''
This module covers two basic dynamic programming applications:
1. Weighted Independent Set (WIS) from a path
2. Knapsack with binary items and integer sizes

The key dynamic programming feature is that a recurrence can be set up
using a limited number of scenarios involving optimal sub-problems. 
'''


import numpy as np
import pandas as pd


def WIS(w):
    '''Returns the optimal WIS for a given path.
    
    Parameters
    ----------
    w : list
        A list of weights in order of the path.
        
    Returns
    -------
    int
        The optimal total weight.
    list
        The optimal set of indices (1-based indexing).
    '''
    # Declare list to hold sub-problem solutions
    solutions = [0] * (len(w) + 1)
    # Base case
    solutions[0], solutions[1] = 0, w[0]
    # Solve all sub-problems
    for i in range(2, len(solutions)):
        solutions[i] = max(w[i-1] + solutions[i-2], solutions[i-1])
    # Reconstruct set of indices
    i = len(solutions) - 1
    selected = []
    while (i-2) >= 0:
        if solutions[i] == (w[i-1] + solutions[i-2]):
            selected.append(i)
            i -= 2
        else:
            i -= 1
    return solutions[-1], selected


def load_wis_data(filename):
    '''Helper for loading WIS data.'''
    path = []
    with open(filename) as f:
        next(f)
        for line in f:
            path.append(int(line.replace('\n', '')))
    return path


def iterative_knapsack(capacity, items):
    '''Returns the optimal items for the knapsack.
    
    Note: I will limit myself to non-vectorized operations.
    
    Parameters
    ----------
    capacity : int
        The maximum capacity of the knapsack.
    items : list
        A list of (value, weight) tuples for the items.
    
    Returns
    -------
    int
        The optimal weight.
    list
        The indices for the optimal set of items.
    '''
    # Initialize cache for sub-problem solutions
    cache = np.zeros((len(items) + 1, capacity + 1))
    # Initialize base case with 0 items
    for c in range(capacity + 1):
        cache[0,c] = 0
    # Solve all sub-problems
    for i in range(1, len(items) + 1):
        for c in range(capacity + 1):
            value, weight = items[i-1]
            if weight > c:
                cache[i,c] = cache[i-1,c]
            else:
                cache[i,c] = max(value + cache[i-1, c-weight],
                                 cache[i-1,c])
    # Reconstruct optimal knapsack contents
    contents = []
    i, c = len(items), capacity
    while i > 0:
        value, weight = items[i-1]
        if cache[i,c] == value + cache[i-1, c-weight]:
            contents.append(i) # Storing using 1-based indexing
            i -= 1
            c -= weight
        else:
            i -= 1
    return cache[len(items), capacity], contents


def recursive_knapsack(capacity, items, item_idx=None, cache=None):
    '''Returns the optimal items for the knapsack.
    
    Parameters
    ----------
    capacity : int
        The maximum capacity of the knapsack.
    items : list
        A list of (value, weight) tuples for the items.
    item_idx : int
        The index of the last item being considered in a subproblem.
    cache : dict, optional
        A cache of previously solved sub-problems.
        
    Returns
    -------
    int
        The optimal weight.
    dict
        The cached sub-problem solutions.
    '''
    # Initialize cache, if needed
    if cache is None:
        cache = dict()
    if item_idx is None:
        item_idx = len(items) - 1

    # Base case
    if item_idx == -1:
        solution = 0
    
    # Recursive cases
    else: 
        # Get value, weight for last item
        value, weight = items[item_idx]
        # First, scenario where the last item is not kept
        try:
            # Check for existing solution to sub-problem
            s1 = cache[(item_idx - 1, capacity)]
        except KeyError:
            # Compute and cache solution to sub-problem
            cache[(item_idx - 1, capacity)] = recursive_knapsack(capacity, items, item_idx - 1, cache)[0]
            s1 = cache[(item_idx - 1, capacity)]
        # Consider scenario where item is kept
        if weight > capacity:
            s2 = 0
        else:
            try:
                s2 = cache[(item_idx - 1, capacity - weight)]
                s2 += value
            except KeyError:
                cache[(item_idx - 1, capacity - weight)] = recursive_knapsack(capacity - weight, items, item_idx - 1, cache)[0]
                s2 = cache[(item_idx - 1, capacity - weight)]
                s2 += value
        # Check for optimal sub-problem
        if s2 > s1:
            solution = s2
        else:
            solution = s1
    
    return solution, cache


def recursive_contents(capacity, items, cache, solution):
    '''Helper for reconstructing knapsack contents from recursive implementation.
    
    Parameters
    ----------
    capacity : int
        The maximum knapsack capacity.
    items : list
        A list of (value, weight) tuples for the items.
    cache : dict
        The cache of sub-problem solutions.
    solution:
        The optimal value that fits in the knapsack.
        
    Returns
    -------
    list
        The optimal knapsack contents.
    '''
    # Initialize
    i = len(items) - 1
    cache[(i, capacity)] = solution
    contents = []
    while i >= 0:
        value, weight = items[i]
        if capacity >= weight and cache[(i, capacity)] == value + cache[(i-1, capacity - weight)]:
            contents.append(i+1) # Convert to 1-based indexing
            i -= 1
            capacity -= weight
        else:
            i -= 1
    return contents

            
def load_knapsack_data(filename):
    '''Helper for loading knapsack data.'''
    items = []
    with open(filename) as f:
        capacity, _ = next(f).replace('\n', '').split(' ')
        capacity = int(capacity)
        for line in f:
            value, weight = line.replace('\n', '').split(' ')
            items.append((int(value), int(weight)))
    return capacity, items


if __name__ == '__main__':
    import sys
    import time # Going to use the rough time approach

    # Bump up the recursion limit
    sys.setrecursionlimit(3000)

    # First, test WIS implementation for correctness
    wis_test_cases = [('problem16.6test.txt', 2617),
                      ('problem16.6.txt', 2955353732)]

    print(f'Starting WIS test cases...')
    for test_case in wis_test_cases:
        path = load_wis_data(test_case[0])
        assert WIS(path)[0] == test_case[1], f'WIS test {test_case[0]} failed.'
    print(f'All WIS test cases passed.')

    # Second, test knapsack implementations for correctness and operation count
    test_cases = [('problem16.7test.txt', 10000, 100, 2493893),
                  ('problem16.7.txt', 2000000, 2000, 11475230)]
    # I have a tower machine w/ 32 GB of RAM, and I was able to compare
    # the very costly iterative implementation on the larger problem.
    # If you want to run it, comment out the line below but be warned you
    # might run out of memory and crash your machine if it's not as beefy.
    # The moral of the story is that enumerating all potential subproblems in a
    # cache still becomes intractable at a certain problem size, at that point
    # it's better to use the recursive implementation with dictionary caching of
    # only the subset of relevent unique subproblems.  
    test_cases = [('problem16.7test.txt', 10000, 100, 2493893)]

    print(f'Starting knapsack test cases...this will take a while.')
    df_ops = []
    for test_case in test_cases:
        capacity, items = load_knapsack_data(test_case[0])
        results = {'test': test_case[0]}
        # Iterative run
        start = time.time()
        solution, _ = iterative_knapsack(capacity, items)
        assert solution == test_case[3], f'Test case {test_case[0]} failed with iterative implementation.'
        results['iter_time'] = round(time.time() - start,3)
        results['iter_counts'] = capacity * len(items) # ignoring base case rows 
        # Recursive run
        start = time.time()
        solution, cache = recursive_knapsack(capacity, items)
        _ = recursive_contents(capacity, items, cache, solution)
        assert solution == test_case[3], f'Test case {test_case[0]} failed with recursive implementation.'
        results['recursive_time'] = round(time.time() - start, 3)
        results['recursive_counts'] = len(cache)         
        df_ops.append(results)
        print(f'Test case {test_case[0]} complete.')
    print('All knapsack correctness and operation count cases complete.')
    df = pd.DataFrame(df_ops)
    df['iter_rec_count_ratio'] = (df['iter_counts'] / df['recursive_counts']).round(3)
    print(df)