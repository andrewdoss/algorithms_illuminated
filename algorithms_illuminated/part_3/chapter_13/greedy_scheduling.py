'''
Demo of greedy scheduling strategies.

This module is trivial and shows that greedy scheduling amounts to
computing sorting keys and sorting.
'''


def read_input(filepath):
    '''Helper for loading test data.'''
    jobs = []
    with open(filepath) as f:
        for line in f:
            job = line.replace('\n', '').split(' ')
            jobs.append((int(job[0]), int(job[1])))
    return jobs


def greedy_diff(jobs):
    '''Returns schedule using greedy diff criteria'''
    jobs_sorted = sorted(jobs, reverse=True) # Pre-sort by weight to break ties
    return sorted(jobs_sorted, key=lambda x: x[0] - x[1], reverse=True)


def greedy_ratio(jobs):
    '''Returns schedule using greedy ratio criteria'''
    return sorted(jobs, key=lambda x: x[0]/x[1], reverse=True)


def weighted_sum(schedule):
    '''Scores schedule by weighted sum of completion times.'''
    score = 0
    completion_time = 0
    for job in schedule:
        completion_time += job[1]
        score += job[0] * completion_time
    return score


if __name__ == '__main__':
    test_jobs = read_input('problem13.4test.txt')
    greedy_diff_sched_test = greedy_diff(test_jobs)
    greedy_ratio_sched_test = greedy_ratio(test_jobs)
    greedy_diff_score_test = weighted_sum(greedy_diff_sched_test)
    greedy_ratio_score_test = weighted_sum(greedy_ratio_sched_test)

    assert greedy_diff_score_test == 68615, 'Greedy Diff Test Failed'
    assert greedy_ratio_score_test == 67247, 'Greedy Ratio Test Failed'

    test_jobs = read_input('problem13.4.txt')
    greedy_diff_sched = greedy_diff(test_jobs)
    greedy_ratio_sched = greedy_ratio(test_jobs)
    greedy_diff_score = weighted_sum(greedy_diff_sched)
    greedy_ratio_score = weighted_sum(greedy_ratio_sched)

    assert greedy_diff_score == 69119377652, 'Greedy Diff Challenge Failed'
    assert greedy_ratio_score == 67311454237, 'Greedy Ratio Challenge Failed'

    print('All tests passed for both Greedy Diff and Greedy Ratio strategies.')