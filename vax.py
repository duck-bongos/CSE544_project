import pandas as pd
import numpy as np

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

def part_d(vax):
    # extract the training and testing data
    print('PART D')
    first = pd.to_datetime('2021-5-1')
    last = first + pd.Timedelta(days=20)
    train = vax[(vax['date'] >= first) & (vax['date'] <= last)]
    first = last + pd.Timedelta(days=1)
    last += pd.Timedelta(days=7)
    test = vax[(vax['date'] >= first) & (vax['date'] <= last)]

    # part i
    x = train['Administered'].to_numpy()
    print('Actual #vaccines administered during fourth week of May 2021:')
    print('\t'.join(['%d' % round(x) for x in test['Administered']]))

    p = 3
    beta = AR(x, p)
    yhat = predict(x, beta)
    print('Predicted #vaccines administered during fourth week of May 2021 using AR(%d):' % p)
    print('\t'.join(['%d' % round(x) for x in yhat]))
    print('MAPE: %.2f%%\nMSE: %.2f' % error(yhat, test['Administered']))

    p = 5
    beta = AR(x, p)
    yhat = predict(x, beta)
    print('Predicted #vaccines administered during fourth week of May 2021 using AR(%d):' % p)
    print('\t'.join(['%d' % round(x) for x in yhat]))
    print('MAPE: %.2f%%\nMSE: %.2f' % error(yhat, test['Administered']))

    alpha = 0.5
    # regress down to plain lists
    # dan loves to see it
    xlist = x.tolist()
    for i in range(7):
    	xlist.append(ewma(xlist, alpha))
    yhat = xlist[-7:]
    print('Predicted #vaccines administered during fourth week of May 2021 using EWMA(%.1f):' % alpha)
    print('\t'.join(['%d' % round(x) for x in yhat]))
    print('MAPE: %.2f%%\nMSE: %.2f' % error(yhat, test['Administered']))

    alpha = 0.8
    xlist = x.tolist()
    for i in range(7):
    	xlist.append(ewma(xlist, alpha))
    yhat = xlist[-7:]
    print('Predicted #vaccines administered during fourth week of May 2021 using EWMA(%.1f):' % alpha)
    print('\t'.join(['%d' % round(x) for x in yhat]))
    print('MAPE: %.2f%%\nMSE: %.2f' % error(yhat, test['Administered']))

    print()
    
def part_e(vax):
    pass
