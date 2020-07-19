'''
    Basic sorting algorithms
'''

import pandas as pd
import numpy as np
import timeit

def selection_sort(x):
    '''Sort a list of numbers or strings.
    
    Assumes a homogeneous input and that built-in comparison operators apply.
    
    Assumes that ascending order is desired.
    
    Parameters
    ----------
    x : list
        A list of numbers or strings to be sorted.
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
            if s[i] < s[suffix_start]:
                s[suffix_start], s[i] = s[i], s[suffix_start]
        suffix_start += 1
    return s

def merge_(left, right):
    '''Merges two sorted lists into a single sorted list.

    Assumes each input is individually sorted.
    
    Assumes that ascending order is desired.
    
    Parameters
    ----------
    left : list
        A sorted list of numbers or strings.
    right: list
        A sorted list of numbers or strings.
    Returns
    -------
    list
        A list containing the sorted, merged contents of both inputs.
        
    Based off pseudocode from Algorithms Illuminated Part 1    
    '''
    result = []
    i, j = 0, 0
    while True:
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
            if i == len(left):
                result.extend(right[j:])
                break
        else:
            result.append(right[j])
            j += 1
            if j == len(right):
                result.extend(left[i:])
                break
    return result

def merge_sort(x):
    '''Sort a list of numbers or strings.
    
    Assumes a homogeneous input and that built-in comparison operators apply.
    
    Assumes that ascending order is desired.
    
    Parameters
    ----------
    x : list
        A list of numbers or strings to be sorted.
    Returns
    -------
    list
        A sorted list of the provided numbers or strings.
        
    Based off pseudocode from Algorithms Illuminated Part 1
    ''' 
    # Recursively split and sort the list
    split_point = len(x) // 2
    left, right = x[:split_point], x[split_point:]
    if len(left) > 1:
        left = merge_sort(left)
    if len(right) > 1:
        right = merge_sort(right)
    # Merge and return results
    return merge_(left, right)

def get_setup(input_type, n):
    '''Get setups for timing tests'''
    s = 'from __main__ import selection_sort, merge_sort; import numpy as np;' 
    if input_type == 'presorted':
        s += f' x=np.arange({n});'
    if input_type == 'reversed':
        s += f' x=np.arange({n}, 0, -1);'
    if input_type == 'random':
        s += f' x=np.arange({n}); x = np.random.choice(x, int({n}));'
    if input_type == 'shuffled':
        s += f' x=np.arange({n}); np.random.shuffle(x);'
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
        return np.random.shuffle(x)

def check_correctness(input, output):
    '''Checks correctness against built in sorting routine'''
    output = np.array(output)
    reference_output = np.array(sorted(input))
    differences = np.sum(output != reference_output)
    if differences == 0:
        return 'passed'
    else:
        return 'failed'

if __name__ == '__main__':
    # First, correctness tests
    print('Running correctness tests...')
    for input_type in ['presorted', 'reversed', 'random']:
        for algorithm in [selection_sort, merge_sort]:
            x = get_input(input_type, 1000)
            output = algorithm(x)
            result = check_correctness(x, output)
            print(input_type, algorithm, result)

    # Second, timing tests for various input types and sizes
    results = pd.DataFrame(data={'n':[1e2, 1e3, 1e4, 2e4]})
    print('\nRunning timing tests...')
    for input_type in ['presorted', 'reversed', 'random']:
        print(input_type)
        for algorithm in ['selection_sort(x)', 'merge_sort(x)', 'sorted(x)']:
            temp_results = []
            for n in results['n'].values:
                temp_results.append(np.round(timeit.timeit(algorithm, setup=get_setup(input_type, n), number=1), 4))
            results[f'{input_type}_{algorithm[:-3]}'] = temp_results
    print('Timing tests complete.')
    print(results.head(results.shape[0]))