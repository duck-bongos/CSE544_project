import pandas as pd
import numpy as np
from math import sqrt
from scipy import stats # only used for t distribution cdf

# data is 1D numpy array
# p is integer
def AR(data, p):
	train = np.ones((len(data) - p, p + 1))
	y = data[p:]
	for r in range(len(data) - p):
		train[r,1:] = data[r:r+p]
	# beta = (X^T X)^-1 X^T Y
	beta = train.transpose() @ train
	beta = np.linalg.inv(beta)
	beta = beta @ train.transpose() @ y
	return beta

def predict(x, weights):
   x = [1] + x[len(x) - len(weights) + 1:].tolist()
   yhat = []
   for i in range(7):
       yhat.append(np.array(x) @ weights)
       x = [1] + x[2:] + yhat[-1:]
   return yhat

def error(yhat, truth):
    ape = 0
    se = 0
    for hat, t in zip(yhat, truth):
        ape += abs(hat - t)/t * 100
        se += (hat - t) ** 2
    mape = ape / len(yhat)
    mse = se / len(yhat)
    return mape, mse

def ewma(x, alpha):
    weights = [alpha * (1 - alpha) ** i for i in range(len(x) - 1)]
    weights.append((1 - alpha) ** (len(x) - 1))
    weights.reverse()
    return np.array(x) @ np.array(weights)

def part_d(vax, name):
    # extract the training and testing data
    print('PART D for %s' % name)
    first = pd.to_datetime('2021-5-1')
    last = first + pd.Timedelta(days=20)
    train = vax[(vax['date'] >= first) & (vax['date'] <= last)]
    first = last + pd.Timedelta(days=1)
    last += pd.Timedelta(days=7)
    test = vax[(vax['date'] >= first) & (vax['date'] <= last)]

    # part i
    x = train['admin'].to_numpy()
    print('Actual #vaccines administered during fourth week of May 2021:')
    print('\t'.join(['%d' % round(x) for x in test['admin']]))

    p = 3
    beta = AR(x, p)
    yhat = predict(x, beta)
    print('Predicted #vaccines administered during fourth week of May 2021 using AR(%d):' % p)
    print('\t'.join(['%d' % round(x) for x in yhat]))
    print('MAPE: %.2f%%\nMSE: %.2f' % error(yhat, test['admin']))

    p = 5
    beta = AR(x, p)
    yhat = predict(x, beta)
    print('Predicted #vaccines administered during fourth week of May 2021 using AR(%d):' % p)
    print('\t'.join(['%d' % round(x) for x in yhat]))
    print('MAPE: %.2f%%\nMSE: %.2f' % error(yhat, test['admin']))

    alpha = 0.5
    # regress down to plain lists
    # dan loves to see it
    xlist = x.tolist()
    for i in range(7):
    	xlist.append(ewma(xlist, alpha))
    yhat = xlist[-7:]
    print('Predicted #vaccines administered during fourth week of May 2021 using EWMA(%.1f):' % alpha)
    print('\t'.join(['%d' % round(x) for x in yhat]))
    print('MAPE: %.2f%%\nMSE: %.2f' % error(yhat, test['admin']))

    alpha = 0.8
    xlist = x.tolist()
    for i in range(7):
    	xlist.append(ewma(xlist, alpha))
    yhat = xlist[-7:]
    print('Predicted #vaccines administered during fourth week of May 2021 using EWMA(%.1f):' % alpha)
    print('\t'.join(['%d' % round(x) for x in yhat]))
    print('MAPE: %.2f%%\nMSE: %.2f' % error(yhat, test['admin']))
    print()

def paired_ttest(x1, x2):
	diff = x1 - x2
	xbar = sum(diff) / len(diff)
	sqr_err = (diff - xbar) ** 2
	sqr_err = sum(sqr_err) / len(sqr_err)
	std_dev = sqrt(sqr_err)
	t = xbar / (std_dev / sqrt(len(diff)))
	return t
    
def part_e(s1, s2, name1, name2):
	print('PART E')
	for month in [9, 11]:
		start = pd.to_datetime('2021-%d-1' % month)
		stop = pd.to_datetime('2021-%d-1' % (month + 1))
		x1 = s1[(s1['date'] >= start) & (s1['date'] < stop)]
		x2 = s2[(s2['date'] >= start) & (s2['date'] < stop)]
		# remove missing days
		x1 = x1[x1['date'].apply(lambda x: x in x2['date'].values)]
		x2 = x2[x2['date'].apply(lambda x: x in x1['date'].values)]
		x1 = x1['admin'].to_numpy()
		x2 = x2['admin'].to_numpy()
		t = paired_ttest(x1, x2)
		print('Paired T-test for comparing the number of vaccines administered each day in %s and %s during 2022-%d:' % (name1, name2, month))
		days = (stop - start).days
		print('Using %d out of %d days (missing days are outliers)' % (len(x1), days))
		# we subtract MA from MS so positive t-value means MA has higher mean
		print('t = %.4f' % t)
		alpha = 2 * (1 - stats.t.cdf(t, len(x1) - 1))
		print('alpha = %.6f' % alpha)
	print()
