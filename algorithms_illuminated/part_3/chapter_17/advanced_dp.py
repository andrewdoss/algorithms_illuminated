"""Implementations of sequence alignment and optimal BST algorithms.

These algorithms provide examples of dynamic programming.

For now, I did not implement the reconstruction functions because the idea
is so similar to the algorithms in Chapter 16.
"""


import numpy as np


def read_seq_file(filename):
    """Reads data from sequence alignment test file.
    
    Args:
        filename (str): The file containing the edge list.
      
    Returns:
        str: The first sequence of characters.
        str: The second sequence of characters.
        int: The cost per gap in a sequence.
        int: The cost per mismatch in a sequence.
    """
    with open(filename, 'r') as f:
        next(f) # Skip first line
        cost_gap, cost_mismatch = next(f).strip().split()
        cost_gap, cost_mismatch = int(cost_gap), int(cost_mismatch)
        seq_x = next(f).strip()
        seq_y = next(f).strip()
        return seq_x, seq_y, cost_gap, cost_mismatch
    
    
def nw_score(seq_x, seq_y, cost_gap, cost_mismatch):
    """Returns the Needleman-Wunsch score for two sequences.
    
    The NW score is based on an optimal alignment, allowing gaps.
    
    I'm using a numpy array but avoiding vectorization to keep things minimal.
    
    Args:
        seq_x (str): The first sequence.
        seq_y (str): The second sequence.
        cost_gap (int): Cost per gap.
        cost_mismatch (int): Cost per mismatch.
        
    Returns:
        int: The NW score.
    """
    # Initialize array for holding solutions
    cache = np.zeros((len(seq_x)+1, len(seq_y)+1))
    # Base case #1 - seq_y is empty
    for i in range(len(seq_x) + 1):
        cache[i,0] = i * cost_gap
    # Base case #2 - seq_x is empty
    for j in range(len(seq_y) + 1):
        cache[0,j] = j * cost_gap
    # Systematically solve all sub-problems
    for i in range(1, len(seq_x) + 1):
        for j in range(1, len(seq_y) + 1):
            if seq_x[i-1] == seq_y[j-1]:
                match_cost = 0
            else:
                match_cost = cost_mismatch
            cost_1 = cache[i-1,j-1] + match_cost
            cost_2 = cache[i-1, j] + cost_gap
            cost_3 = cache[i, j-1] + cost_gap
            cache[i,j] = min(cost_1, cost_2, cost_3)
    return cache[len(seq_x), len(seq_y)]


def read_bst_file(filename):
    """Reads data from sequence alignment test file.
    
    Args:
        filename (str): The file containing the edge list.
      
    Returns:
        list: The frequencies for each key.
    """
    with open(filename, 'r') as f:
        next(f) # Skip first line
        freq = [int(x) for x in next(f).strip().split(',')]
        return freq
    
    
def opt_bst(freq):
    """Returns the search cost for an optimal BST.
    
    This algorithm finds an optimal BST, provided key frequencies.
    
    The costs are not normalized, so search cost is not average length.
    
    Args:
        freq (list): The frequencies for each key.
        
    Returns:
        int: The search cost for the optimal BST.
    """
    # Initialize array for holding solutions
    n = len(freq)
    cache = np.zeros((n+1, n+1))
    # Base case 
    for i in range(n + 1):
        cache[i,i] = 0
    # Systematically solve all subproblems
    for s in range(n):
        for i in range(1, n + 1 - s):
            pk = sum(freq[i-1:i+s])
            min_cost = float('inf')
            for r in range(i, i + s + 1):
                min_cost = min(min_cost, cache[i-1,r-1] + cache[r,i+s])
            cache[i-1][i+s] = pk + min_cost
    return cache[0][n]


if __name__ == '__main__':
    import time

    nw_tests = [('problem17.8nw.txt', 224)]
    bst_opt_tests = [('problem17.8optbst.txt', 2780)]
                

    # Run NW correctness and timing tests
    print('Starting NW correctness tests...')
    for test_case in nw_tests:
        nw_inputs = read_seq_file(test_case[0])
        start = time.time()
        score = nw_score(*nw_inputs)
        end = round(time.time() - start, 3)
        assert score == test_case[1], f'Failed correctness for {test_case[0]}.'
        print(f'Passed correctness for {test_case[0]} in {end} seconds.')
    print('All NW correctness tests passed.')

    # Run NW correctness and timing tests
    print('Starting optimal BST correctness tests...')
    for test_case in bst_opt_tests:
        freq = read_bst_file(test_case[0])
        start = time.time()
        cost = opt_bst(freq)
        end = round(time.time() - start, 3)
        assert cost == test_case[1], f'Failed correctness for {test_case[0]}.'
        print(f'Passed correctness for {test_case[0]} in {end} seconds.')
    print('All optimal BST correctness tests passed.')