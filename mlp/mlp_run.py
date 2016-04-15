import matplotlib.pyplot as plt
import numpy
from TimeSeriesNnet import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# read csv and separate data
csv = numpy.genfromtxt('../datasets/2007.txt', delimiter=";", names=True, dtype=None, missing_values='?')
time_series = map(lambda row : 0 if numpy.isnan(row[2]) else row[2]*1000, filter(lambda row: row[0].rpartition('/')[2] == '2007', csv))

csv = numpy.genfromtxt('../datasets/2008.txt', delimiter=";", names=True, dtype=None, missing_values='?')
test_series = map(lambda row : 0 if numpy.isnan(row[2]) else row[2]*1000, filter(lambda row: row[0].rpartition('/')[2] == '2008', csv))

def collapse (ts, ahead):
    r = []
    for i in xrange(0,len(ts),ahead):
        r.append(max(ts[i:i+ahead-1]))
    return r

hours = 60
time_series = collapse(time_series,hours)
test_series = collapse(test_series,hours)

time_series = np.array(time_series,dtype='int32')

neural_net = TimeSeriesNnet(hidden_layers = [20, 15, 5], activation_functions = ['sigmoid', 'sigmoid', 'sigmoid'])
neural_net.fit(time_series, lag=4, epochs=500)
neural_net.predict_year(test_series)

# RMSE Training error
rmse = mean_squared_error(test_series, neural_net.predictions)**0.5
print rmse

plt.plot(range(len(neural_net.predictions)), neural_net.predictions, '-r', label='Predictions', linewidth=1)
plt.plot(range(len(test_series)), test_series, '-g',  label='Original series')
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.show()

