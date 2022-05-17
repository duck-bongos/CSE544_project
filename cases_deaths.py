import pandas as pd
import random
import numpy as np
from math import exp, log
from scipy.stats import poisson, kstwo, geom, binom # just used for PMF/PDF/CDF calculation
import matplotlib.pyplot as plt

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

def part_b(dfs):
	print('PART B')
	states = list(dfs.keys())
	start = pd.to_datetime('2021-9-1')
	stop = pd.to_datetime('2022-1-1')
	for col in dfs[states[0]]:
		print('Analyzing %s data' % col)
		s1 = dfs[states[0]][col]
		s2 = dfs[states[1]][col]
		s1 = s1[(s1['date'] >= start) & (s1['date'] < stop)]
		s2 = s2[(s2['date'] >= start) & (s2['date'] < stop)]
		# 1 sample K-S tests
		# poisson
		lamb_mme = s1[col].sum() / s1.shape[0]
		def cdf(x):
			return poisson.cdf(x, lamb_mme)
		ks, p = ks_test1(cdf, s2[col])
		print('1 Sample KS-test with Poisson distribution')
		print('K-S Statistic: %.4f' % ks)
		print('p-val: %.4f' % p)

		# geometric
		p_mme = 1 / (s1[col].sum() / s1.shape[0])
		def cdf(x):
			return geom.cdf(x, p_mme)
		ks, p = ks_test1(cdf, s2[col])
		print('1 Sample KS-test with Geometric distribution')
		print('K-S Statistic: %.4f' % ks)
		print('p-val: %.4f' % p)

		# binomial
		ex = s1[col].sum() / s1.shape[0]
		ex2 = sum(s1[col].to_numpy() ** 2) / s1.shape[0]
		s = ex2 - ex ** 2
		n_mme = ex ** 2 / (ex - s)
		p_mme = (ex - s) / ex
		if (n_mme < 0 or p_mme < 0):
			print('MME for binomial distribution returned negative values')
			# do not proceed with test
			# https://piazza.com/class/kxyzhcwa3ie43m?cid=157
		else:
			def cdf(x):
				return binom.cdf(x, n_mme, p_mme)
			ks, p = ks_test1(cdf, s2[col])
			print('1 Sample KS-test with Binomial distribution')
			print('K-S Statistic: %.4f' % ks)
			print('p-val: %.4f' % p)

		# 2 sample K-S test
		ks, p = ks_test2(s1[col], s2[col])
		print('2 Sample KS-test')
		print('K-S Statistic: %.4f' % ks)
		print('p-val: %.4f' % p)
		#plt.figure()
		#x = s1[col].sort_values()
		#y = np.linspace(0, 1, num=len(s1[col]))
		#plt.plot(x, y)
		#x = s2[col].sort_values()
		#y = np.linspace(0, 1, num=len(s2[col]))
		#plt.plot(x, y)
		#plt.savefig('test-%s.pdf' % col)

		# Permutation test
		p = permutation_test(s1[col], s2[col])
		print('Permutation Test')
		print('p-val: %.6f' % p)
	print()

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

def part_c(dfs):
	print('PART C')
	dates = set()
	first = True
	start = pd.to_datetime('2020-6-1')
	stop = start + pd.Timedelta(days=7*8)
	# remove any dates with missing data
	for state in dfs:
		for col in dfs[state]:
			df = dfs[state][col]
			df = df[(df['date'] >= start) & (df['date'] < stop)]
			dfs[state][col] = df
			if first:
				dates.update(dfs[state][col]['date'])
				first = False
			else:
				dates.intersection_update(dfs[state][col]['date'])
	combined = pd.DataFrame(columns=['date', 'count'])
	dates = sorted(list(dates))
	combined['date'] = dates
	combined['count'] = np.zeros(len(dates))
	for state in dfs:
		for col in dfs[state]:
			combined['count'] = combined['count'].values + dfs[state][col][col].values
	stop = start + pd.Timedelta(days=7*4)
	first_four = combined[combined['date'] < stop]
	lamb_mme = first_four['count'].sum() / first_four.shape[0]
	logpdfs = [gen_exp_logpdf(1 / lamb_mme)]
	for week in range(4):
		start = stop
		stop += pd.Timedelta(days=7)
		data = combined[(combined['date'] >= start) & (combined['date'] < stop)]
		logpdfs.append(gen_logpdf(logpdfs[-1], data['count']))
	num = 3000
	stop = 3000
	dx = stop / num
	x = np.linspace(0, stop, num=num)
	maps = []
	for logpdf in logpdfs:
		logs = logpdf(x)
		# adding a constant is equivalent to linear scaling of pdf
		# scale it up for numerical stability
		logs -= max(logs)
		y = np.exp(logs)
		total = sum(y) * dx
		y /= total
		maps.append(max(range(len(y)), key=lambda i: y[i]))
		plt.plot(x, y)
	plt.ylim((0, plt.ylim()[1]))
	legend = ['Original Prior']
	for i in range(5, 9):
		legend.append('Prior after %dth week' % i)
		print('MAP for posterior after %dth week: %d' % (i, maps[i - 4]))
	plt.legend(legend)
	plt.ylabel('Likelihood')
	plt.xlabel('λ')
	plt.title('Bayesian Probability Distributions for λ (2c)')
	plt.savefig('2c.pdf')
	print()
