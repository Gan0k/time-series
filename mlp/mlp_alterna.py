import matplotlib.pyplot as plt
import numpy
from TimeSeriesNnet import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sknn.mlp import Regressor, Layer

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
test_series = np.array(test_series,dtype='float64')

#normalise series
time_series = (time_series-time_series.mean())/time_series.std()
test_series = (test_series-test_series.mean())/test_series.std()

#initialize values
n_hidden = 100
lag = 3
n_input = lag
n_output = 1

layer = Layer('Sigmoid', units=3)
layer2 = Layer('Sigmoid', units=1)
nn = Regressor([layer,layer2], n_iter=100)

#build x-train y train
x_train = np.zeros((len(time_series)-lag,lag))
y_train = np.zeros(len(time_series)-lag)

for i in range(len(time_series)-lag):
    x_train[i,:] = time_series[i:i+lag]
    y_train[i] = np.cos(i+lag)

#training
nn.fit(x_train, y_train)

#testing
x_test = np.zeros((len(test_series)-lag,lag))
y_test = np.zeros(len(test_series)-lag)
predictions = np.zeros(len(test_series)-lag)

for i in range(len(test_series)-lag):
    x_test[i,:] = test_series[i:i+lag]
    y_test[i] = test_series[i+lag]

predictions = nn.predict( x_test )

# RMSE Training error
rmse = mean_squared_error(y_test, predictions)**0.5
print rmse

window = 24*7;
plt.plot(range(len(predictions[:window])), predictions[:window], '-r', label='Predictions', linewidth=1)
plt.plot(range(len(y_test[:window])), y_test[:window], '-g',  label='Original series')
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.show()

