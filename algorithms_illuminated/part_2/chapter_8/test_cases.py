search_tests = [
    {'name': 'search_1',
        'file': 'problem8.10test1.txt',
        'start_vertex': 6,
        'solution': {1, 3, 4, 6, 7, 9}},
    {'name': 'search_2',
        'file': 'problem8.10test1.txt',
        'start_vertex': 5,
        'solution': {1, 2, 3, 4, 5, 6, 7, 8, 9}},
    {'name': 'search_3',
        'file': 'problem8.10test3.txt',
        'start_vertex': 4,
        'solution': {4}},
        {'name': 'search_4',
        'file': 'problem8.10test4.txt',
        'start_vertex': 4,
        'solution': {1, 2, 3, 4, 6, 7, 8}},
        {'name': 'search_5',
        'file': 'problem8.10test5.txt',
        'start_vertex': 1,
        'solution': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}
]


bfs_unit_distance_tests = [
    {'name': 'bfs_unit_dist_1',
        'file': 'problem8.10test1.txt',
        'start_vertex': 6,
        'solution': {1:3, 3:2, 4:4, 6:0, 7:2, 9:1}},
    {'name': 'bfs_unit_dist_2',
        'file': 'problem8.10test1.txt',
        'start_vertex': 5,
        'solution': {1:6, 2:1, 3:5, 4:7, 5:0, 6:3, 7:5, 8:2, 9:4}},
    {'name': 'bfs_unit_dist_3',
        'file': 'problem8.10test3.txt',
        'start_vertex': 4,
        'solution': {4:0}},
    {'name': 'bfs_unit_dist_4',
        'file': 'problem8.10test4.txt',
        'start_vertex': 4,
        'solution': {1:2, 2:3, 3:1, 4:0, 6:1, 7:2, 8:3}},
    {'name': 'bfs_unit_dist_5',
        'file': 'problem8.10test5.txt',
        'start_vertex': 1,
        'solution': {1:0, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:4, 9:5, 10:4, 11:5, 12:6}}
]


bfs_ucc_tests = [
    {'name': 'bfs_ucc_1',
        'file': 'problem8.10test1.txt',
        'solution': [[1, 2, 3, 4, 5, 6, 7, 8, 9]]},
    {'name': 'bfs_ucc_2',
        'file': 'problem8.10test1mod1.txt',
        'solution': [[1, 4, 7], [2, 3, 5, 6, 8, 9]]},
    {'name': 'bfs_ucc_3',
        'file': 'problem8.10test1mod2.txt',
        'solution': [[1, 4, 7], [2, 5, 8], [3, 6, 9]]}
]


topo_sort_tests = [
    {'name': 'topo_sort_1',
        'file': 'problem8.10test1mod3.txt'},
    {'name': 'bfs_ucc_2',
        'file': 'problem8.10test1mod4.txt'},
    {'name': 'bfs_ucc_3',
        'file': 'problem8.10test3mod1.txt'}    
]


scc_search_tests = [
    {'name': 'scc_search_1',
     'file': 'problem8.10test1.txt',
     'solution': [3, 3, 3, 0, 0]},
    {'name': 'scc_search_2',
     'file': 'problem8.10test2.txt',
     'solution': [3, 3, 2, 0, 0]},
    {'name': 'scc_search_3',
     'file': 'problem8.10test3.txt',
     'solution': [3, 3, 1, 1, 0]},
    {'name': 'scc_search_4',
     'file': 'problem8.10test4.txt',
     'solution': [7, 1, 0, 0, 0]},
    {'name': 'scc_search_5',
     'file': 'problem8.10test5.txt',
     'solution': [6, 3, 2, 1, 0]},
    {'name': 'challenge_data',
     'file': 'problem8.10.txt',
     'solution': [434821, 968, 459, 313, 211]}            
]
