from itertools import permutations
from typing import List
import numpy as np


def permutation_test(l: List[List[int]], means=True, medians=False):
    # calculate t_obs
    if means:
        obs = [np.mean(i) for i in l]
        t_obs = abs(obs[0] - obs[1])
    else:
        obs = [np.median(i) for i in l]
        t_obs = abs(obs[0] - obs[1])

    l_count = len(l)
    list_lengths = [len(i) for i in l]
    list_idx = []
    for i in range(0, len(list_lengths)):
        if i == 0:
            list_idx.append((0, list_lengths[i]))
        else:
            list_idx.append((list_idx[i - 1][1], list_lengths[i] + list_idx[i - 1][1]))

    comb_lists = [x for i in l for x in i]
    list_perm = list(permutations(comb_lists))

    new_lists = [[list(lp[s:e]) for s, e in list_idx] for lp in list_perm]

    if means is True:
        x = [[np.mean(li) for li in n] for n in new_lists]
    else:
        x = [[np.median(li) for li in n] for n in new_lists]

    ab = [abs(i[0] - i[1]) for i in x]

    p_value = sum([1 if a > t_obs else 0 for a in ab]) / len(ab)
    print(p_value)
    return p_value
