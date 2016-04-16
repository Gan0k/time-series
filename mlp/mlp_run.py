import matplotlib.pyplot as plt
import numpy
from TimeSeriesNnet import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# read csv and separate data
csv = numpy.genfromtxt('../datasets/2007.txt', delimiter=";", names=True, dtype=None, missing_values='?')
time_series = map(lambda row : 0 if numpy.isnan(row[2]) else row[2], filter(lambda row: row[0].rpartition('/')[2] == '2007', csv))

csv = numpy.genfromtxt('../datasets/2008.txt', delimiter=";", names=True, dtype=None, missing_values='?')
test_series = map(lambda row : 0 if numpy.isnan(row[2]) else row[2], filter(lambda row: row[0].rpartition('/')[2] == '2008', csv))

csv = numpy.genfromtxt('../datasets/2009.txt', delimiter=";", names=True, dtype=None, missing_values='?')
test_series2 = map(lambda row : 0 if numpy.isnan(row[2]) else row[2], filter(lambda row: row[0].rpartition('/')[2] == '2009', csv))

def collapse (ts, ahead):
    r = []
    for i in xrange(0,len(ts),ahead):
        r.append(max(ts[i:i+ahead-1]))
    return r

hours = 60
time_series = collapse(time_series,hours)
test_series = collapse(test_series,hours)
test_series2 = collapse(test_series2,hours)

time_series = np.array(time_series,dtype='float64')

neural_net = TimeSeriesNnet(hidden_layers = [2], activation_functions = ['sigmoid'])
neural_net.fit(time_series, lag=1, epochs=500)

neural_net.predict_year(test_series)
# RMSE Training error
rmse = mean_squared_error(test_series, neural_net.predictions)**0.5
print rmse

neural_net.predict_year(test_series2)
# RMSE Training error
rmse = mean_squared_error(test_series2, neural_net.predictions)**0.5
print rmse

plt.plot(range(len(neural_net.predictions)), neural_net.predictions, '-r', label='Predictions', linewidth=1)
plt.plot(range(len(test_series2)), test_series2, '-g',  label='Original series')
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.show()

