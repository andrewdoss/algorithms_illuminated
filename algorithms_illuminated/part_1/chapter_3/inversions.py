'''
    Algorithms for counting inversions.
'''

import pandas as pd
import numpy as np
import timeit


def brute_count_inversions(x):
    '''Count number of inversions in a sequence using brute force search.
    
    Parameters
    ----------
    x : iterable
        An ordered iterable with elements that have a valid order comparison.
    Returns
    -------
    int
        The number of inversions in the input.
        
    Based off pseudocode from Algorithms Illuminated Part 1.
    '''
    num_inversions = 0
    for i in range(len(x) - 1):
        for j in range(i, len(x)):
            if x[i] > x[j]:
                num_inversions += 1
    return num_inversions


def sort_count_inversions(x):
    '''Sort and count inversions in a list.
    
    Assumes a homogeneous input and that built-in comparison operators apply.

    Assumes that ascending order is desired.
    
    Parameters
    ----------
    x : list
        A list of numbers or strings.
    Returns
    -------
    list
        A sorted copy of the input list.
    int
        The number of inversions in the list.
        
    Based off pseudocode from Algorithms Illuminated Part 1
    ''' 
    # Recursively split and sort the list
    split_point = len(x) // 2
    left, right = x[:split_point], x[split_point:]
    left_num_inv, right_num_inv = 0, 0
    if len(left) > 1:
        left, left_num_inv = sort_count_inversions(left)
    if len(right) > 1:
        right, right_num_inv = sort_count_inversions(right)
    # Merge and return results
    return merge_(left, right, left_num_inv + right_num_inv)


def merge_(left, right, num_inv):
    '''Merges two sorted lists and counts inversions.

    Assumes each input is individually sorted.
    
    Assumes that ascending order is desired.
    
    Parameters
    ----------
    left : list
        A sorted list of numbers or strings.
    right: list
        A sorted list of numbers or strings.
    num_inv:
        The number of inversions already counted within left and right.
    Returns
    -------
    list
        A list containing the sorted, merged contents of both inputs.
    int
        The number of inversions counted within the returned list.

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
            num_inv += len(left) - i # Count one inversion per remaining in left
            j += 1
            if j == len(right):
                result.extend(left[i:])
                break
    return result, num_inv   



def get_setup(algorithm_name, n):
    s = f"from __main__ import {algorithm_name}; import numpy as np;"
    s += f"x = [digit for digit in np.random.randint(0,{n},{n})];"
    return s 



if __name__ == '__main__':
    # First, correctness tests involving varying length combinations
    algorithms = {'brute_count_inversions': brute_count_inversions,
                  'sort_count_inversions': sort_count_inversions}
    
    # Define test lists with answers
    test_cases = []
    test_cases.append(('sorted ascending', 0, list(range(20))))
    test_cases.append(('sorted descending', 20*19/2, list(range(19,-1,-1))))
    test_cases.append(('small test', 28, pd.read_csv('problem3.5_small.txt', header=None).iloc[:,0].to_list()))
    test_cases.append(('challenge test', 2407905288, pd.read_csv('problem3.5_challenge.txt', header=None).iloc[:,0].to_list()))

    for algorithm_name, algorithm in algorithms.items():
        print(f'\nRunning correctness tests for {algorithm_name}...')
        num_tested = 0
        num_passed = 0
        failed_tests = []
        # Iterate over test cases
        for test_name, correct_count, input_list in test_cases:
            result = algorithm(input_list)
            try:
                result = result[1] # Unpack inversion count, if applicable 
            except:
                pass
            if result == correct_count:
                num_passed += 1
                print(f'{test_name} passed.')
            else:
                failed_tests.append((test_name, correct_count, result))
                print(f'{test_name} failed.')
            num_tested += 1
        print(f'Results: {num_passed} passed out of {num_tested} tests.')
        if num_tested > num_passed:
            print(f'Failed input sets:')
            for inputs in failed_tests:
                print(inputs)

    # Second, timing tests for various input sizes
    results = pd.DataFrame(data={'n':[40, 80, 160, 320]})
    for algorithm_name, algorithm in algorithms.items():
        print(f'\nRunning timing tests for {algorithm_name}...')
        temp_results = []
        for n in results['n'].values:
            n = int(n)
            temp_results.append(np.round(timeit.timeit(f'{algorithm_name}(x)', setup=get_setup(algorithm_name, n), number=1), 4))
        results[algorithm_name] = temp_results
    print('Timing tests complete.')
    print(results.head(results.shape[0]))