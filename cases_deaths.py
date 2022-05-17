import pandas as pd
import numpy as np
from math import exp, log
from scipy.stats import poisson # just used for PMF calculation
import matplotlib.pyplot as plt

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

def part_a(dfs):
	#### PART A ####
            feb_21 = st[
                (st["date"] >= datetime(2021, 2, 1))
                & (st["date"] <= datetime(2021, 2, 28))
            ]

            # get population means and standard deviations, which we get from February
            pop_mu = feb_21[col].mean()
            pop_sigma = feb_21[col].std()

            # get sample data
            mar_21 = st[
                (st["date"] >= datetime(2021, 3, 1))
                & (st["date"] <= datetime(2021, 3, 31))
            ]

            # run hypothesis tests assuming D ~ (X1, ... Xn) ~ Poisson(lambda)
            print("PART A")
            for t in (t_test, walds_test, z_test):
                if t.__name__ == "t_test":
                    t_stat, p_value, cv = t(data=mar_21[col].values, pop_mean=pop_mu)

                if t.__name__ == "walds_test":
                    t_stat, p_value, cv = t(
                        data=mar_21[col].values, null_hypothesis=pop_mu
                    )

                if t.__name__ == "z_test":
                    t_stat, p_value, cv = t(data=mar_21[col].values, pop_mean=pop_mu)

                should = "SHOULD" if abs(t_stat) > cv else "SHOULD NOT"
                res = f"\nState: {state} | For {' '.join(t.__name__.split('_'))}: Population mean: {pop_mu}\nT statistic: {t_stat}, p-value {p_value}, critical value: {cv}\nThis indicates we {should} reject the sample #{col} from March as being part of the same distribution.\n\n"
                out_to_file.append(res)
                print(res)

                """
				We are told the data is distributed with a Poisson distribution. 
				Therefore, Wald's test and Z test should not be applied because 
				they are both asymptotically normally when valid.
				"""

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
                tsres = f"\nState: {state} | For {' '.join(ts.__name__.split('_'))}: T statistic: {t_stat}, p-value {p_value}, critical value: {cv}\nThis indicates we {should} reject the sample #{col} from March as being part of the same distribution.\n\n"

                out_to_file.append(tsres)
                print(tsres)

        with open("hypothesis_tests.txt", "w") as ht:
            for line in out_to_file:
                ht.write(f"{line}")

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
