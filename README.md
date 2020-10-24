### Algorithms Practice

This repository contains a collection of Python implementations of classic algorithms covered in the Algorithms Illuminated book series (better known as the Stanford Algorithms MOOC). So far, I am building correctness and (where applicable for comparisons) efficiency test cases for each algorithm or pulling test cases from the following repository: https://github.com/beaunus/stanford-algs

Most of these implementations are derived directly from pseudo-code. 

### Contents

#### Part 1 - The Basics

Key concepts:
* algorithmic complexity analysis through bounding asymptotic growth
* recursive divide-and-conquer approaches with Master Method time complexity analysis of recurrence relations - can we reduce work per-problem fast enough relative to proliferation of sub-problems to reduce total computational complexity?
* relaxation from worst-case time analysis, using randomization for efficient average time complexity and robustness against pathological inputs  

Practice implementations:
* Chapter 1 - selection sort, merge sort, recursive integer multiplication, Karatsuba multiplication (optimized recursive integer multiplication)
* Chapter 2 - no programming exercises (asymptotic notation/analysis theory)
* Chapter 3 - merge-sort based O(nlogn) inversion counting
* Chapter 4 - linear search, binary search
* Chapter 5 - quicksort
* Chapter 6 - linear-time selection (of order statistics)

#### Part 2 - Graph Algorithms and Data Structures

Key concepts:
* graph representations and basic search and topological ordering
* intro to basic data sctructures - stacks, queues, sorted arrays (linked-lists assumed as prior knowledge)
* intro to more advanced data structures - heaps, binary search trees, hash tables, and bloom filters
* applications of advanced data structures and comparisons of supported operations and efficiency per operation on various test problems

Practice implementations:
* Chapter 7 - no programming exercises (intro to graphs)
* Chapter 8 - breadth and depth first search, various extensions including finding connected components in a graph, strongly connected components in a digraph and shortest path lengths in a graph with unit lengths (all linear time complexity)
* Chapter 9 - straighforward Dijkstra's algorithm with O(nm) complexity (n=number of vertices, m=number of edges), both baseline and optimized implementations
* Chapter 10 - custom heap implementation; comparison of custom heap, built-in heap, and linear-time selection for median maintenance; heap-sort
* Chapter 11 - custom binary search tree implementation, comparison with heaps for median maintenance
* Chapter 12 - basic two-sum solution using dictionary to demonstrate hashing application

#### Part 3 - Greedy Algorithms and Dynamic Programming

Key concepts:
* development of and correctness proofs for greedy algorithms, often relying on exchange argument - i.e. assume greedy solution is not optimal then show that this leads to a logical contradiction
* example applications of greedy algorithms - scheduling, Huffman encoding, minimum spanning trees
* dynamic programming - useful when we can enumerate all relevant sub-problems and compute them with a lower computational complexity than a direct recursive solution, caching of computations that are needed multiple times to solve a problem, works when problem has overlapping sub-problems and correct solution is limited to a limited number of optimal sub-problems

Practice implementations:
* Chapter 13 - greedy scheduling, reduces to sorting jobs by the correct key, correct for some problem definitions
* Chapter 14 - Huffman encoding using a straightforward, heap-based, and sorting + queue based implementation
* Chapter 15 - Prim (straightforward and heap-based) and Kruskal's (straight-forward and union-find based) MST algorithms, custom union-find data structure implementation
* Chapter 16 - basic dynamic programming examples; weighted independent set problem; simple knapsack problem with iterative and recursive implementations, demonstration of intractibility of iterative enumeration for a larger problem vs. tractable approach with recursive implementation that solves and caches a smaller sub-set of relevant sub-problems
* Chapter 17 - advanced dynamic programming examples; Needleman-Wunsch similarity score for string sequences; optimal binary search trees when search frequencies are known
* Chapter 18 - shortest paths (including negative edge lengths) with dynamic programming; Bellman-Ford for single-source shortest paths; Floyd-Warshall for all-pairs shortest paths problems

#### Part 4 - Algorithms for NP-Hard Problems

