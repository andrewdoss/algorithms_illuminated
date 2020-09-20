### Algorithms Practice

This repository contains a collection of Python implementations of classic algorithms covered in the Algorithms Illuminated book series (better known as the Stanford Algorithms MOOC). So far, I am building correctness and (where applicable for comparisons) efficiency test cases for each algorithm or pulling test cases from the following repository: https://github.com/beaunus/stanford-algs

Most of these implementations are derived directly from pseudo-code. 

### Contents

#### Part 1 - The Basics

Key concepts:
* algorithmic complexity analysis through bounding asymptotic growth
* recursive divide-and-conquer approaches with Master Method time complexity analysis of recurrence relations
* using randomization for efficient average time complexity  

Practice implementations:
* Chapter 1 - selection sort, merge sort, recursive integer multiplication, Karatsuba multiplication (optimized recursive integer multiplication)
* Chapter 2 - no programming exercises (asymptotic notation/analysis theory)
* Chapter 3 - merge-sort based O(nlogn) inversion counting
* Chapter 4 - linear search, binary search
* Chapter 5 - quicksort
* Chapter 6 - linear-time selection (of order statistics)

#### Part 2 - Graph Algorithms and Data Structures

Practice implementations:
* Chapter 7 - no programming exercises (intro to graphs)
* Chapter 8 - breadth and depth first search, various extensions including finding connected components in a graph, strongly connected components in a digraph and shortest path lengths in a graph with unit lengths (all linear time complexity)
* Chapter 9 - straighforward Dijkstra's algorithm with O(nm) complexity (n=number of vertices, m=number of edges), both baseline and optimized implementations

#### Part 3 - Greedy Algorithms and Dynamic Programming