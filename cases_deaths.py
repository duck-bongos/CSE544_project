import pandas as pd
import random
import numpy as np
from scipy.stats import poisson, kstwo, geom, binom # just used for PMF/PDF/CDF calculation
from datetime import datetime
from math import exp, log
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities import *

def part_a(st: pd.DataFrame, state: str, col: str) -> List[str]:
	#### PART A ####
	print('PART A for %s data in %s' % (col, state))
	feb_21 = st[
		(st["date"] >= datetime(2021, 2, 1)) & (st["date"] <= datetime(2021, 2, 28))
	]

	# get H0 from february
	h0= sum(feb_21[col]) / feb_21.shape[0]
	# get population standard deviation
	pop_sigma = stddev(st[col])

	# get sample data
	mar_21 = st[
		(st["date"] >= datetime(2021, 3, 1)) & (st["date"] <= datetime(2021, 3, 31))
	]

	# run hypothesis tests assuming D ~ (X1, ... Xn) ~ Poisson(lambda)
	print("""ONE SAMPLE TESTS""")
	for t in (t_test, walds_test, z_test):
		if t.__name__ == "t_test":
			t_stat, p_value, cv = t(data=mar_21[col].values, pop_mean=h0)

		if t.__name__ == "walds_test":
			t_stat, p_value, cv = t(data=mar_21[col].values, null_hypothesis=h0)

		if t.__name__ == "z_test":
			t_stat, p_value, cv = t(data=mar_21[col].values, pop_mean=h0, pop_sigma=pop_sigma)

		should = "SHOULD" if abs(t_stat) > cv else "SHOULD NOT"
		res = f"State: {state} | For {' '.join(t.__name__.split('_'))}: Null Hypothesis mean: {h0}\nT statistic: {t_stat}, p-value {p_value}, critical value: {cv}\nThis indicates we {should} reject the null hypothesis that the mean #{col} from March is the same as in February.\n"
		print(res)

	# two sample tests
	print("""TWO SAMPLE TESTS""")
	for ts in (two_sample_t_test, two_sample_walds_test):
		if ts.__name__ == "two_sample_t_test":
			t_stat, p_value, cv = ts(
				data_1=feb_21[col].values, data_2=mar_21[col].values
			)

		if ts.__name__ == "two_sample_walds_test":
			t_stat, p_value, cv = ts(
				data_1=feb_21[col].values, data_2=mar_21[col].values
			)
		should = "SHOULD" if abs(t_stat) > cv else "SHOULD NOT"
		tsres = f"State: {state} | For {' '.join(ts.__name__.split('_'))}: T statistic: {t_stat}, p-value {p_value}, critical value: {cv}\nThis indicates we {should} reject the null hypothesis that the mean #{col} from March is the same as in February.\n"

		print(tsres)
	print()

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


def part_c(dfs):
	print("PART C")
	dates = set()
	first = True
	start = pd.to_datetime("2020-6-1")
	stop = start + pd.Timedelta(days=7 * 8)
	# remove any dates with missing data
	for state in dfs:
		for col in dfs[state]:
			df = dfs[state][col]
			df = df[(df["date"] >= start) & (df["date"] < stop)]
			dfs[state][col] = df
			if first:
				dates.update(dfs[state][col]["date"])
				first = False
			else:
				dates.intersection_update(dfs[state][col]["date"])
	combined = pd.DataFrame(columns=["date", "count"])
	dates = sorted(list(dates))
	combined["date"] = dates
	combined["count"] = np.zeros(len(dates))
	for state in dfs:
		for col in dfs[state]:
			combined["count"] = combined["count"].values + dfs[state][col][col].values
	stop = start + pd.Timedelta(days=7 * 4)
	first_four = combined[combined["date"] < stop]
	lamb_mme = first_four["count"].sum() / first_four.shape[0]
	logpdfs = [gen_exp_logpdf(1 / lamb_mme)]
	for week in range(4):
		start = stop
		stop += pd.Timedelta(days=7)
		data = combined[(combined["date"] >= start) & (combined["date"] < stop)]
		logpdfs.append(gen_logpdf(logpdfs[-1], data["count"]))
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
	legend = ["Original Prior"]
	for i in range(5, 9):
		legend.append("Prior after %dth week" % i)
		print("MAP for posterior after %dth week: %d" % (i, maps[i - 4]))
	plt.legend(legend)
	plt.ylabel("Likelihood")
	plt.xlabel("λ")
	plt.title("Bayesian Probability Distributions for λ (2c)")
	plt.savefig("2c.pdf")
	print()
