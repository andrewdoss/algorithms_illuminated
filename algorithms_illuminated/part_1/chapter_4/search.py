'''
    Algorithms for searching a list for a target element.
'''

import pandas as pd
import numpy as np
import timeit

def linear_search(x, target):
    '''Returns the index of a target element in a list, if present.
    
    Note: the list is searched for an equivalent element, not a reference
    to the same object as target.
    
    If the list contains multiple instances of the target element, the 
    index returned is arbitrary.
    
    Parameters
    ----------
    x : list
        A list of elements that may contain the target element.
    target : primitive object
        The target element to search for.
    Returns
    -------
    int
        An index of the target element if found, else returns None.
        
    Based off pseudocode from Algorithms Illuminated Part 1.
    '''
    index = None
    for i, x_i in enumerate(x):
        if x_i == target:
            index = i
            break
    return index

def binary_search(x, target, lower=None, upper=None):
    '''Returns the index of a target element from a sorted list, if present.
    
    Note: the list is searched for an equivalent element, not a reference
    to the same object as target.
    
    If the list contains multiple instances of the target element, the 
    index returned is arbitrary.
    
    Parameters
    ----------
    x : list
        A sorted (ascending) list of elements that may contain the target element.
    target : primitive object
        The target element to search for.
    Returns
    -------
    int
        An index of the target element if found, else returns None.
        
    Based off pseudocode from Algorithms Illuminated Part 1.
    '''
    # Initialize current slice
    if lower is None:
        lower = 0
    if upper is None:
        upper = len(x) - 1
    # Check for target if down to a single element left
    if lower == upper:
        if x[lower] == target:
            index = lower
        else:
            index = None
    # Check for target and recurse if needed with > single element left
    else:
        midpoint = (lower + upper) // 2
        if x[midpoint] == target:
            index = midpoint
        elif x[midpoint] > target:
            index = binary_search(x, target, lower, midpoint - 1)
        else:
            index = binary_search(x, target, midpoint + 1, upper)
    return index

def built_in_search(x, target):
    '''Wrapper for built-in index operator
    
    Note: This should be slower than binary because it can't exploit sorted input.
    '''
    try:
        idx = x.index(target)
    except ValueError:
        idx = None
    return idx

def get_setup(algorithm_name, n, target):
    s = f"from __main__ import {algorithm_name}; import numpy as np;"
    s += f"x = list(range({n}));"
    s += f"target = {target}"
    return s 

if __name__ == '__main__':
    # First, correctness tests involving varying length combinations
    algorithms = {'linear_search': linear_search,
                  'binary_search': binary_search,
                  #'built_in_search': built_in_search
                  }
    
    # Define tests with answers
    small_test_no_repeats = list(range(20))
    small_test_w_repeats = sorted(list(range(10)) * 2)
    large_test_no_repeats = list(range(10000000))
    large_test_w_repeats = sorted(list(range(5000000)) * 2)
    test_cases = []
    # test syntax: <description>, <correct index/indices>, <target>, <sorted list>
    test_cases.append(('small test, no repeats, target present', {19}, 19, small_test_no_repeats))
    test_cases.append(('small test, with repeats, target present', {18, 19}, 9, small_test_w_repeats))
    test_cases.append(('small test, no repeats, target not present', None, 20, small_test_no_repeats))
    test_cases.append(('small test, with repeats, target not present', None, 10, small_test_w_repeats))    
    test_cases.append(('large test, no repeats, target present', {9999999}, 9999999, large_test_no_repeats))
    test_cases.append(('large test, with repeats, target present', {9999998, 9999999}, 4999999, large_test_w_repeats))
    test_cases.append(('large test, no repeats, target not present', None, 10000000, large_test_no_repeats))
    test_cases.append(('large test, with repeats, target not present', None, 5000000, large_test_w_repeats))    

    for algorithm_name, algorithm in algorithms.items():
        print(f'\nRunning correctness tests for {algorithm_name}...')
        num_tested = 0
        num_passed = 0
        failed_tests = []
        # Iterate over test cases
        for test_name, correct_idx, target, input_list in test_cases:
            result = algorithm(input_list, target)
            if (correct_idx is None and result is None or 
                correct_idx is not None and result in correct_idx):
                num_passed += 1
                print(f'{test_name} passed.')
            else:
                failed_tests.append((test_name, correct_idx, result))
                print(f'{test_name} failed.')
            num_tested += 1
        print(f'Results: {num_passed} passed out of {num_tested} tests.')
        if num_tested > num_passed:
            print(f'Failed input sets:')
            for inputs in failed_tests:
                print(inputs)

    # Second, timing tests for various input sizes
    # I will check both worst-case (not found) and uniform random case.
    # Note: the expectation for uniform random is the equivalent to halving input size for linear scans.
    # The more interesting part is the variation in run times.
    results = pd.DataFrame(data={'n': [int(n) for n in np.logspace(6,8,3)]})
    for algorithm_name, algorithm in algorithms.items():
        print(f'\nRunning worst-case timing tests for {algorithm_name}...')
        temp_results = []
        for n in results['n'].values:
            temp_results.append(np.round(timeit.timeit(f'{algorithm_name}(x, target)', setup=get_setup(algorithm_name, n, n), number=1), 4))
        results[algorithm_name] = temp_results
    print('Timing tests complete.')
    print(results.head(results.shape[0]))

    results = pd.DataFrame(data={'n': [int(n) for n in np.logspace(6,8,3)]})
    num_trials = 20
    for algorithm_name, algorithm in algorithms.items():
        print(f'\nRunning average of {num_trials} random timing tests for {algorithm_name}...')
        mean_results = []
        std_results = []
        for n in results['n'].values:
            trial_results = []
            for trial in range(num_trials):
                target = np.random.randint(0, n, 1)[0]
                trial_results.append(timeit.timeit(f'{algorithm_name}(x, target)', setup=get_setup(algorithm_name, n, target), number=1))
            mean_results.append(np.round(np.mean(trial_results), 4))
            std_results.append(np.round(np.std(trial_results), 4))
        results[f'{algorithm_name}_mean'] = mean_results
        results[f'{algorithm_name}_std'] = std_results
    print('Timing tests complete.')
    print(results.head(results.shape[0]))