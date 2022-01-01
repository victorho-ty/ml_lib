import itertools


def power_set(iterable, skip_empty_set=False):
    """
    Power set is a set which includes all the subsets including the empty set and the original set itself.
    e.g set A = {x, y, z}:
        Power set of A, P(A) = { (x,), (y,), (z,), (x, y), (y, z), (x, z), (x, y, z), () }
    :param iterable:
    :param skip_empty_set:
    :return: set
    """
    in_list = list(iterable)
    start_idx = 1 if skip_empty_set else 0
    len_combos = [itertools.combinations(in_list, length) for length in range(start_idx, len(in_list)+1)]
    return itertools.chain.from_iterable(len_combos)


if __name__ == '__main__':
    tmp = (1, 2, 3)
    ret = power_set(tmp, skip_empty_set=True)
    print(set(ret))



