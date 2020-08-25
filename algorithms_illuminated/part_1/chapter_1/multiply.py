'''
    Recursive multiplication algorithms.
'''

import pandas as pd
import numpy as np
import timeit


def pad_inputs_(x, y):
    '''Left 0-pad inputs to have equal number of digits.'''
    if len(x) > len(y):
        y = '0' * (len(x) - len(y)) + y
    elif len(y) > len(x):
        x = '0' * (len(y) - len(x)) + x
    return x, y


def split_input_(s):
    '''Splits input digits in half.'''
    split_point = len(s) // 2
    return s[:split_point], s[split_point:] 


def rec_int_mult_(x, y):
    '''Multiply two integers using a recrusive approach.'''
    # Pad inputs to be the same length
    x, y = pad_inputs_(x, y)
    
    # Check for base case
    n = len(x)
    if n == 1:
        return int(x) * int(y)
    
    # Split inputs
    a, b = split_input_(x)
    c, d = split_input_(y)
    
    # Recursively compute pair products
    ac = rec_int_mult_(a, c)
    ad = rec_int_mult_(a, d)
    bc = rec_int_mult_(b, c)
    bd = rec_int_mult_(b, d)
    
    # Combine results
    if n % 2 == 1: # Handle odd-length strings
        n += 1
    return ac*10**(n) + (ad + bc)*10**(n//2) + bd


def rec_int_mult(x, y):
    '''Multiply two integers using a recursive approach.
    
    Parameters
    ----------
    x : int
        The left operand for integer multiplication.
    y : int
        The right operand for integer multiplication.
    Returns
    -------
    int
        The result of multiplying x and y.
        
    Based off pseudocode from Algorithms Illuminated Part 1.
    '''
    # This is a wrapper to cast the inputs to strings for the first call
    x = str(x)
    y = str(y)
    return rec_int_mult_(x, y)


def karatsuba_(x, y):
    '''Multiply two integers using Karatsuba multiplication.'''
    # Pad inputs to be the same length
    x, y = pad_inputs_(x, y)
    
    # Check for base case
    n = len(x)
    if n == 1:
        return int(x) * int(y)
    
    # Split inputs
    a, b = split_input_(x)
    c, d = split_input_(y)
    
    # Compute p and q
    p = str(int(a) + int(b))
    q = str(int(c) + int(d))
    
    # Recursively compute pair products
    ac = karatsuba_(a, c)
    bd = karatsuba_(b, d)
    pq = karatsuba_(p, q)
    
    # Combine results
    adbc = pq - ac - bd
    if n % 2 == 1: # Handle odd-length strings
        n += 1
    return ac*10**(n) + (adbc)*10**(n//2) + bd    


def karatsuba(x, y):
    '''Multiply two integers using Karatsuba multiplication.
    
    Parameters
    ----------
    x : int
        The left operand for integer multiplication.
    y : int
        The right operand for integer multiplication.
    Returns
    -------
    int
        The result of multiplying x and y.
        
    Based off pseudocode from Algorithms Illuminated Part 1.
    '''
    # This is a wrapper to cast the inputs to strings for the first call
    x = str(x)
    y = str(y)
    return karatsuba_(x, y)


def built_in_multiply(x, y):
    '''Applies the built-in multiplication operator '''
    return x * y


def get_random_input(n):
    '''Returns random integer with specified number of digits.'''
    return int(''.join([str(digit) for digit in np.random.randint(0,10,n)]))


def get_setup(algorithm_name, n):
    s = f"from __main__ import {algorithm_name}; import numpy as np;"
    s += f"x = int(''.join([str(digit) for digit in np.random.randint(0,10,{n})]));"
    s += f"y = int(''.join([str(digit) for digit in np.random.randint(0,10,{n})]));"
    return s 


if __name__ == '__main__':
    # First, correctness tests involving varying length combinations
    algorithms = {'rec_int_mult': rec_int_mult, 'karatsuba': karatsuba} 
    digit_range = (1,20) # Range of number of digits to test
    for algorithm_name, algorithm in algorithms.items():
        print(f'\nRunning correctness tests for {algorithm_name} with number of digits from {digit_range[0]} to {digit_range[1]}...')
        num_tested = 0
        num_passed = 0
        failed_inputs = []
        # Tests all permutations of digit ranges with built-in operator as reference
        for x_digits in range(min_digits, max_digits):
            for y_digits in range(min_digits, max_digits):
                x, y = get_random_input(x_digits), get_random_input(y_digits)
                result = algorithm(x, y)
                if result == x * y:
                    num_passed += 1
                else:
                    failed_inputs.append((x, y))
                num_tested += 1
        print(f'Results: {num_passed} passed out of {num_tested} tests.')
        if num_tested > num_passed:
            print(f'Failed input sets:')
            for inputs in failed_inputs:
                print(inputs)

    # Second, timing tests for various input sizes
    algorithms['built_in_multiply'] = built_in_multiply
    results = pd.DataFrame(data={'n':[40, 80, 160, 320]})
    for algorithm_name, algorithm in algorithms.items():
        print(f'\nRunning timing tests for {algorithm_name}...')
        temp_results = []
        for n in results['n'].values:
            n = int(n)
            temp_results.append(np.round(timeit.timeit(f'{algorithm_name}(x, y)', setup=get_setup(algorithm_name, n), number=1), 4))
        results[algorithm_name] = temp_results
    print('Timing tests complete.')
    print(results.head(results.shape[0]))