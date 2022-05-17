from itertools import permutations
from math import ceil
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as ss


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


def t_test(data: np.array, pop_mean: float) -> Tuple[float, float]:
    """Calculate (X_bar - mu)/(s/√n). Return p-value."""
    x_bar = np.mean(data)
    s_stdev = np.std(data)
    n_count = len(data)

    test_statistic = (x_bar - pop_mean) / (s_stdev / np.sqrt(n_count))
    test_statistic = abs(test_statistic)

    # return test statistic and p-value
    return test_statistic, ss.t.cdf(test_statistic)


def walds_test(data: np.array, null_hypo_value):
    """(theta_hat - theta_null) / standard_error(theta_hat)"""
    # return test statistic and p-value
    return data, null_hypo_value


def z_test(data: np.array, pop_mean: float, pop_stdev: float):
    """Calculate (X_bar - mu)/sigma. Return p-value."""
    if len(data) < 30:
        print("Too few observations to properly run Z test")
        return
    else:
        x_bar = data.mean()
        test_statistic = abs((x_bar - pop_mean)) / pop_stdev

        # return test statistic and p-value
        return test_statistic, 2 * (1 - ss.norm.cdf(test_statistic))


# manually calculate outliers
def tukey(df: pd.DataFrame, col: str):
    data = df[col].sort_values()
    data.reset_index(drop=True, inplace=True)

    # calculate IQR
    q1 = ceil(25 / 100 * len(data))
    q3 = ceil(75 / 100 * len(data))
    iqr = data[q3] - data[q1]
    lo = data[q1] - 1.5 * iqr
    hi = data[q3] + 1.5 * iqr

    print("Min\tQ1\tQ3\tMax")
    print("%d\t%d\t%d\t%d" % (min(data), data[q1], data[q3], max(data)))
    print("Discarding non-zero values outside of range [%d, %d]" % (lo, hi))
    return df[(df[col] >= lo) & (df[col] <= hi) | (df[col] == 0)]
