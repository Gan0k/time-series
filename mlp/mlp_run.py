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

def collapse (ts, ahead):
    r = []
    for i in xrange(0,len(ts),ahead):
        r.append(max(ts[i:i+ahead-1]))
    return r

hours = 60
time_series = collapse(time_series,hours)
test_series = collapse(test_series,hours)

time_series = np.array(time_series,dtype='float64')
time_series = (time_series - time_series.mean()) / time_series.std()

test_series = np.array(test_series,dtype='float64')
test_series = (test_series - test_series.mean()) / test_series.std()

neural_net = TimeSeriesNnet(hidden_layers = [20,15,5], activation_functions = ['sigmoid','sigmoid','sigmoid'])
neural_net.fit(time_series, lag=1, epochs=500)

neural_net.predict_ahead(len(test_series))
# RMSE Training error
#lag = 1
#test_series = test_series[lag:]
rmse = mean_squared_error(test_series, neural_net.predictions)**0.5
print rmse

window = 24*7;
plt.plot(range(len(neural_net.predictions[:window])), neural_net.predictions[:window], '-r', label='Predictions', linewidth=1)
plt.plot(range(len(test_series[:window])), test_series[:window], '-g',  label='Original series')
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.show()

