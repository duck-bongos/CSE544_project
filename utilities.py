from itertools import permutations
from logging import critical
from statistics import stdev
from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy
import scipy.stats as ss
import random
from scipy.stats import poisson, kstwo, geom, binom # just used for PMF/PDF/CDF calculation
from datetime import datetime
from math import exp, log, sqrt, ceil
from typing import List
import matplotlib.pyplot as plt

ALPHA = 0.025


def scale_data(data: np.array):
	max_val = data.max()
	return data / max_val

def walds_test(data: np.array, null_hypothesis: float) -> Tuple[float, float, float]:
	"""(theta_hat - theta_null) / standard_error(theta_hat)

	Since we're using the MLE of the data for a poisson distribution
	the sample mean and sample standard deviation are the same value."""
	mle_mean = sum(data) / len(data)
	mle_stderr = sqrt(mle_mean / len(data))
	# use scipy.stats.norm ppf to find the critical value for alpha instead of T Table lookup
	critical_value = -ss.norm.ppf(ALPHA / 2)

	test_statistic = (mle_mean - null_hypothesis) / mle_stderr

	# wald's test follows chi squared distribution
	return test_statistic, 2 * ss.norm.cdf(-abs(test_statistic)), critical_value


def z_test(data: np.array, pop_mean: float, pop_sigma: float) -> Tuple[float, float, float]:
	"""Calculate (X_bar - mu)/sigma. Return p-value."""
	# data = scale_data(data)
	x_bar = sum(data) / len(data)
	# use scipy.stats.norm ppf to find the critical value for alpha instead of T Table lookup
	critical_value = -ss.norm.ppf(ALPHA / 2)

	if len(data) < 30:
		print("Too few observations to properly run Z test, but we'll try anyways.")

	test_statistic = (x_bar - pop_mean) / (pop_sigma / sqrt(len(data)))

	# return test statistic and p-value
	return test_statistic, 2 * ss.norm.cdf(-abs(test_statistic)), critical_value


def t_test(data: np.array, pop_mean: float) -> Tuple[float, float, float]:
	"""Calculate (X_bar - mu)/(s/√n). Return p-value."""
	# data = scale_data(data)
	x_bar = sum(data) / len(data)
	s_stdev = stddev(data, corrected=True)
	n_count = len(data)
	# use scipy.stats.norm ppf to find the critical value for alpha instead of T Table lookup
	critical_value = -ss.t.ppf(ALPHA / 2, df=len(data) - 1)

	if len(data) < 30:
		print("Data size less than 30, using T distribution for T test.")
		# return test statistic and p-value
		test_statistic = (x_bar - pop_mean) / (s_stdev / np.sqrt(n_count))
		return test_statistic, 2 * ss.t.cdf(-abs(test_statistic), n_count - 1), critical_value

	else:
		print(
			"Data size greater than or equal to 30, using Normal distribution for T test."
		)
		test_statistic = (x_bar - pop_mean) / (s_stdev / np.sqrt(n_count))
		# return test statistic and p-value
		return test_statistic, 2 * ss.norm.cdf(-abs(test_statistic)), critical_value


def two_sample_t_test(data_1: np.array, data_2: np.array) -> Tuple[float, float, float]:
	x_bar_1 = sum(data_1) / len(data_1)
	x_bar_2 = sum(data_2) / len(data_2)

	var_1 = stddev(data_1, corrected=True) ** 2
	var_2 = stddev(data_2, corrected=True) ** 2

	# use scipy.stats.norm ppf to find the critical value for alpha instead of T Table lookup
	critical_value = -ss.t.ppf(ALPHA/2, df=len(data_1) + len(data_2) - 2)

	denom = np.sqrt(var_1 / len(data_1)) + (var_2 / len(data_2))
	# denom = paired_s * np.sqrt((1 / len(data_1)) + (1 / len(data_1)))

	test_statistic = (x_bar_1 - x_bar_2) / denom

	if len(data_1) + len(data_2) < 30:
		return test_statistic, 2 * ss.t.cdf(-abs(test_statistic)), critical_value
	else:
		return test_statistic, 2 * ss.norm.cdf(-abs(test_statistic)), critical_value


def two_sample_walds_test(
	data_1: np.array, data_2: np.array
) -> Tuple[float, float, float]:
	x_bar_1 = sum(data_1) / len(data_1)
	x_bar_2 = sum(data_2) / len(data_2)

	# plug in estimate for variance of MLE estimate for mean of Poisson distribution
	var_1 = x_bar_1 / len(data_1)
	var_2 = x_bar_2 / len(data_2)

	# use scipy.stats.norm ppf to find the critical value for alpha instead of T Table lookup
	critical_value = ss.norm.ppf(1 - ALPHA)

	num = x_bar_1 - x_bar_2
	# standard error
	denom = np.sqrt(var_1 + var_2)

	test_statistic = num / denom

	return test_statistic, 2 * ss.norm.cdf(-abs(test_statistic)), critical_value

def gen_exp_logpdf(lamb):
	def logpdf(x):
		return log(lamb) - lamb * x

	return logpdf


def gen_logpdf(prior, data):
	def logpdf(x):
		loglikelihood = prior(x)
		for d in data:
			loglikelihood += poisson.logpmf(d, x)
		return loglikelihood

	return logpdf

def stddev(s, corrected=False):
	if type(s) == pd.Series:
		s = s.to_numpy()
	assert type(s) == np.ndarray
	xbar = sum(s) / len(s)
	sqr_err = (s - xbar) ** 2
	if corrected:
		sqr_err = sum(sqr_err) / (len(s) - 1)
	else:
		sqr_err = sum(sqr_err) / len(s)
	return sqrt(sqr_err)


def ks_test1(cdf, data):
	data = data.sort_values()
	maxi = 0
	n = len(data)
	for i, d in enumerate(data):
		lo = i / n
		hi = (i + 1) / n
		x = cdf(d)
		maxi = max(maxi, abs(lo - x))
		maxi = max(maxi, abs(hi - x))
	p = 1 - kstwo.cdf(maxi, n)
	return maxi, p

def ks_test2(d1, d2):
	d1 = d1.sort_values().to_numpy()
	d2 = d2.sort_values().to_numpy()
	maxi = 0
	if len(d1) > len(d2):
		d1, d2 = d2, d1
	n = len(d1)
	i2 = 0
	for i, d in enumerate(d1):
		lo = i / n
		hi = (i + 1) / n
		while d2[i2] < d and i2 < len(d2) - 1:
			i2 += 1
		if d2[i2] >= d:
			lo2 = i / len(d2)
			maxi = max(maxi, abs(lo - lo2))
			maxi = max(maxi, abs(hi - lo2))
		if d2[i2] <= d:
			hi2 = (i + 1) / len(d2)
			maxi = max(maxi, abs(lo - hi2))
			maxi = max(maxi, abs(hi - hi2))
	# two sample KS test has a special formula for the value of n to use for KS distribution
	# https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_stats_py.py#L7275-L7449
	n = len(d1)
	m = len(d2)
	ks_n = round(n * m / (n + m))
	p = 1 - kstwo.cdf(maxi, ks_n)
	return maxi, p

def mean(d1):
	return sum(d1) / len(d1)

def permutation_test(d1, d2):
	obs = abs(mean(d1) - mean(d2))
	data = pd.concat([d1, d2])
	data.reset_index(inplace=True, drop=True)
	samples = 1000
	n = len(d1)
	more_extreme = 0
	for _ in range(samples):
		random.shuffle(data)
		diff = abs(mean(data[:n]) - mean(data[n:]))
		if diff > obs:
			more_extreme += 1
	return more_extreme / samples


# data is 1D numpy array
# p is integer
def AR(data, p):
	train = np.ones((len(data) - p, p + 1))
	y = data[p:]
	for r in range(len(data) - p):
		train[r, 1:] = data[r : r + p]
	# beta = (X^T X)^-1 X^T Y
	beta = train.transpose() @ train
	beta = np.linalg.inv(beta)
	beta = beta @ train.transpose() @ y
	return beta


def predict(x, weights):
	x = [1] + x[len(x) - len(weights) + 1 :].tolist()
	yhat = []
	for i in range(7):
		yhat.append(np.array(x) @ weights)
		x = [1] + x[2:] + yhat[-1:]
	return yhat


def error(yhat, truth):
	ape = 0
	se = 0
	for hat, t in zip(yhat, truth):
		ape += abs(hat - t) / t * 100
		se += (hat - t) ** 2
	mape = ape / len(yhat)
	mse = se / len(yhat)
	return mape, mse


def ewma(x, alpha):
	weights = [alpha * (1 - alpha) ** i for i in range(len(x) - 1)]
	weights.append((1 - alpha) ** (len(x) - 1))
	weights.reverse()
	return np.array(x) @ np.array(weights)

def paired_ttest(x1, x2):
	diff = x1 - x2
	xbar = sum(diff) / len(diff)
	sqr_err = (diff - xbar) ** 2
	sqr_err = sum(sqr_err) / len(sqr_err)
	std_dev = sqrt(sqr_err)
	t = xbar / (std_dev / sqrt(len(diff)))
	return t

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
