#!/usr/bin/env python
# coding=utf-8
# Neural network
# usage (XOR problem example):
#    x = np.array([[0, 0],
#                  [0, 1],
#                  [1, 0],
#                  [1, 1]])
#    y = np.array([[0],
#                  [1],
#                  [1],
#                  [0]])
#    nn = NN([2, 4, 1])    # add  'True'(boolean) arg for batch algorithm
#    opts = [100000,  50]  # opts means [loop count, stochastic num]
#    nn.train(x, y, opts)
#    nn.test(x)

import numpy


class NN(object):
    netstruct = []
    batch = False

    def __init__(self, ns, batch=False):
        if type(ns) is list:
            self.netstruct = ns
        else:
            print("error! please set up netstruct")
            exit(1)

        numpy.random.seed(1)
        self.batch = batch
        self.syn = [(2 * numpy.random.random((ns[i], ns[i + 1])) - 1)
                    for i in range(len(ns) - 1)]
        self.l = [0 for i in range(len(ns))]
        self.l_err = [0 for i in range(len(ns))]
        self.l_delta = [0 for i in range(len(ns))]

    def nonlin(self, x, deriv=False):
        if deriv is True:
            return x * (1 - x)
        return 1 / (1 + numpy.exp(-x))

    def train(self, x, y, opts):
        train_index = 0
        for iter in range(opts[0]):
            if (iter % opts[1]) == 0:
                self.rand = int(numpy.random.random() * len(x))
            # forward
            # random or batch
            if(self.batch):
                self.l[0] = x
            else:
                self.l[0] = numpy.array(x[train_index:train_index + 1])
            for i in range(len(self.l) - 1):
                self.l[i + 1] = self.nonlin(numpy.dot(self.l[i],
                                                      self.syn[i]))
            # backward
            if(self.batch):
                self.l_err[len(self.l) - 1] = y - self.l[len(self.l) - 1]
            else:
                self.l_err[len(self.l_err)-1] = \
                    numpy.array(y[train_index:train_index + 1]) -\
                    self.l[len(self.l_err) - 1]
            self.l_delta[len(self.l) - 1] = \
                self.l_err[len(self.l) - 1] * \
                self.nonlin(self.l[len(self.l) - 1], deriv=True)
            for i in reversed(range(len(self.l) - 1)):
                self.l_err[i] = self.l_delta[i + 1].dot(self.syn[i].T)
                self.l_delta[i] = self.l_err[i] * \
                    self.nonlin(self.l[i], deriv=True)
            for i in reversed(range(len(self.syn))):
                self.syn[i] += self.l[i].T.dot(self.l_delta[i+1])
            train_index += self.rand
            train_index %= len(x)
            # if (iter % 10000) == 0:
            #     print("Error:" + str(numpy.mean(
            #           numpy.abs(self.l_err[len(self.netstruct) - 1]))))

    def test(self, x):
        output = x
        for i in range(len(self.netstruct) - 1):
            output = self.nonlin(numpy.dot(output, self.syn[i]))
        return output
