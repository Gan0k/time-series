import matplotlib.pyplot as plt
import numpy
from TimeSeriesNnet import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import mean_squared_error

# read csv and separate data
csv = numpy.genfromtxt('../datasets/2007.txt', delimiter=";", names=True, dtype=None, missing_values='?')
time_series = map(lambda row : 0 if numpy.isnan(row[2]) else row[2], filter(lambda row: row[0].rpartition('/')[2] == '2007', csv))

time_series = np.array(time_series,dtype='int32')
neural_net = TimeSeriesNnet(hidden_layers = [20, 15, 5], activation_functions = ['sigmoid', 'sigmoid', 'sigmoid'])
neural_net.fit(time_series, lag = 40, epochs = 10000)
neural_net.predict_ahead(n_ahead = 30)

# MSE Computation
mse = mean_squared_error(neural_net.time_series - time_series)
print numpy.sqrt(mse)

plt.plot(range(len(neural_net.timeseries)), neural_net.timeseries, '-r', label='Predictions', linewidth=1)
plt.plot(range(len(time_series)), time_series, '-g',  label='Original series')
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.show()

