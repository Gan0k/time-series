import matplotlib.pyplot as plt
import numpy
from TimeSeriesNnet import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# read csv and separate data
csv = numpy.genfromtxt('../datasets/2007.txt', delimiter=";", names=True, dtype=None, missing_values='?')
time_series = map(lambda row : 0 if numpy.isnan(row[2]) else row[2], filter(lambda row: row[0].rpartition('/')[2] == '2007', csv))

def collapse (ts, ahead):
    r = []
    for i in xrange(0,len(ts),ahead):
        r.append(max(ts[i:i+ahead-1]))
    return r

nahead = 24
time_series = collapse(time_series,nahead)

time_series = np.array(time_series,dtype='int32')
trainset, testset = time_series[0:-nahead], time_series[-nahead:]

neural_net = TimeSeriesNnet(hidden_layers = [20, 15, 5], activation_functions = ['sigmoid', 'sigmoid', 'sigmoid'])
neural_net.fit(trainset, lag = 1, epochs = 1000)
neural_net.predict_ahead(n_ahead = nahead)

# RMSE Training error
mse = mean_squared_error(testset, neural_net.timeseries[-nahead:])
print numpy.sqrt(mse)

plt.plot(range(len(neural_net.timeseries)), neural_net.timeseries, '-r', label='Predictions', linewidth=1)
plt.plot(range(len(time_series)), time_series, '-g',  label='Original series')
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.show()

