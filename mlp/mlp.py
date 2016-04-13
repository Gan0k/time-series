import os
import sys
import time
from os import path

import numpy

import theano
import theano.tensor as T

from datetime import datetime
import time


def load_data(datapath):
    ''' Loads the dataset
    We transform the date into unix time without taking
    the year into account.

    :type datapath: string
    :param datapath: the path to the dataset (here MNIST)
    '''
    # read csv and separate data
    csv = numpy.genfromtxt(datapath, delimiter=";", names=True, dtype=None, missing_values='?', filling_values=0.0)
    trainSet = filter(lambda row: row[0].rpartition('/')[2] == '2007', csv)
    validationSet = filter(lambda row: row[0].rpartition('/')[2] == '2008', csv)
    testSet = filter(lambda row: row[0].rpartition('/')[2] == '2009', csv)

    # change date to unix time
    def changedate (row):
        date = datetime.strptime(';'.join([row[0],row[1]]), "%d/%m/%Y;%H:%M:%S")
        date = date.replace(year=2000)
        return time.mktime(date.timetuple())

    trainDates = map(changedate,trainSet)
    validationDates = map(changedate,validationSet)
    testDates = map(changedate,testSet)

    def normalise (a):
        return a*1000;

    # prepare train, validation and tests sets by sampling randomly
    train_set = tuple([numpy.asmatrix([[d] for d in trainDates]), [normalise(row[2]) for row in trainSet]])

    powervalid = [normalise(row[2]) for row in validationSet]
    valid_set = tuple([numpy.asmatrix([[validationDates[c]] for c in choices]), powervalid[c]])

    powertest = [normalise(row[2]) for row in testSet]
    test_set = tuple([[numpy.asmatrix([testDates[c]]) for c in choices], powertest])

    return _make_array(train_set), _make_array(valid_set), _make_array(test_set)


def _make_array(xy):
    data_x, data_y = xy
    return zip(
        numpy.asarray(data_x, dtype=theano.config.floatX),
        numpy.asarray(data_y, dtype='int32'))


def _init_logreg_weights(n_hidden, n_out):
    weights = numpy.zeros((n_hidden, n_out), dtype=theano.config.floatX)
    bias = numpy.zeros((n_out,), dtype=theano.config.floatX)
    return (
        theano.shared(name='W', borrow=True, value=weights),
        theano.shared(name='b', borrow=True, value=bias)
    )


def _init_hidden_weights(n_in, n_out):
    rng = numpy.random.RandomState(1234)
    weights = numpy.asarray(
        rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )
    bias = numpy.zeros((n_out,), dtype=theano.config.floatX)
    return (
        theano.shared(value=weights, name='W', borrow=True),
        theano.shared(value=bias, name='b', borrow=True)
    )


# Define how an input is fed through a layer of the network, and how a step of
# the stochastic gradient descent is computed.
    
# Note that these are *symbolic expressions* --- we are just compiling code here.
# These functions are only called during compile_model.  The *actual* feed-forward
# and SGD update procedures, which happen iteratively on each example, are 
# Theano-internal.
def feed_forward(activation, weights, bias, input_):
    return activation(T.dot(input_, weights) + bias)

def sgd_step(param, cost, learning_rate):
    return param - (learning_rate * T.grad(cost, param))

# These are also symbolic.
def L1(L1_reg, w1, w2):
    return L1_reg * (abs(w1).sum() + abs(w2).sum())


def L2(L2_reg, w1, w2):
    return L2_reg * ((w1 ** 2).sum() + (w2 ** 2).sum())
 

def compile_model(n_in, n_classes, n_hidden, learning_rate, L1_reg, L2_reg):
    '''Compile train and evaluation functions, which we'll then call iteratively
    to train the parameters.  This function is called exactly once --- think of
    it like a compiler.  We declare variables, allocate memory, and define some
    computation.
    '''
    # allocate symbolic variables for the data
    x = T.vector('x')  # Features
    y = T.iscalar('y') # (Gold) Label

    # Allocate and initialize weights.  These are stored internally, and updated.
    hidden_W, hidden_b = _init_hidden_weights(n_in, n_hidden)
    logreg_W, logreg_b = _init_logreg_weights(n_hidden, n_classes)

    # Estimate P(y | x) given the current weights
    p_y_given_x = feed_forward(
                      T.nnet.softmax,
                      logreg_W,
                      logreg_b,
                      feed_forward(
                          T.tanh,
                          hidden_W,
                          hidden_b,
                          x)) # <--- Our input variable (the features)

    cost = (
        -T.log(p_y_given_x[0, y]) # <-- Negative log likelihood of gold label
        + L1(L1_reg, logreg_W, hidden_W)
        + L2(L2_reg, logreg_W, hidden_W)
    )

    # Compile the training function.  Successive calls to this update the weights.
    # Internal state is maintained.
    # The output is "cost", which requires the computation of p_y_given_x.  We
    # also define how to update the weights based on the input label.
    train_model = theano.function(
        inputs=[x, y],
        outputs=cost, # <-- Output depends on cost, which depends on P(y | x)
        updates=[
            (logreg_W, sgd_step(logreg_W, cost, learning_rate)),
            (logreg_b, sgd_step(logreg_b, cost, learning_rate)),
            (hidden_W, sgd_step(hidden_W, cost, learning_rate)),
            (hidden_b, sgd_step(hidden_b, cost, learning_rate)),
        ]
    )

    # Compile the evaluation function, which returns a 0/1 loss wrt the true
    # label.  Note that the output depends on p_y_given_x, so the program must
    # compute it.
    evaluate_model = theano.function(
        inputs=[x, y],
        outputs=T.neq(y, T.argmax(p_y_given_x[0])),
    )
    return train_model, evaluate_model


def main(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='../datasets/household_power_consumption.txt', n_hidden=500):
    train_examples, dev_examples, test_examples = load_data(dataset)
    print '... building the model'
    train_model, evaluate_model = compile_model(28*28, 10, n_hidden, learning_rate, L1_reg, L2_reg)
    print '... training'
    for epoch in range(1, n_epochs+1):
        for x, y in train_examples:
            train_model(x, y)
        # compute zero-one loss on validation set
        error = numpy.mean([evaluate_model(x, y) for x, y in dev_examples])
        print('epoch %i, validation error %f %%' % (epoch, error * 100))


if __name__ == '__main__':
    main()
