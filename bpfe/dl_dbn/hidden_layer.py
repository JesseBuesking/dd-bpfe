

import numpy as np
import theano
import theano.tensor as T
from bpfe.dl_dbn.constants import DTYPES


class HiddenLayer(object):
    def __init__(self, numpy_rng, layer_input, n_in, n_out, W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: a random number generator used to initialize weights

        :type layer_input: theano.tensor.dmatrix
        :param layer_input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units
        """

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # DTYPES.FLOATX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        if b is None:
            b_values = np.zeros((n_out,), dtype=DTYPES.FLOATX)
            # noinspection PyUnresolvedReferences
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.layer_input = layer_input

    def output(self):
        raise NotImplementedError()


class TanhLayer(HiddenLayer):

    def __init__(self, numpy_rng, layer_input, n_in, n_out, W=None, b=None):
        super(TanhLayer, self).__init__(
            numpy_rng, layer_input, n_in, n_out, W, b)

        if W is None:
            W_values = np.asarray(
                numpy_rng.uniform(
                    # TODO figure out initial weights
                    # low=-1./np.sqrt(n_in),
                    # high=-1./np.sqrt(n_in),
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=DTYPES.FLOATX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        # parameters of the model
        self.params = [self.W, self.b]

    def output(self):
        return T.tanh(T.dot(self.layer_input, self.W) + self.b)


class SigmoidLayer(HiddenLayer):

    def __init__(self, numpy_rng, layer_input, n_in, n_out, W=None, b=None):
        super(SigmoidLayer, self).__init__(
            numpy_rng, layer_input, n_in, n_out, W, b)

        if W is None:
            W_values = np.asarray(
                numpy_rng.uniform(
                    # TODO figure out initial weights
                    # low=-1./np.sqrt(n_in),
                    # high=-1./np.sqrt(n_in),
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=DTYPES.FLOATX
            )

            # since it's sigmoid
            W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        # parameters of the model
        self.params = [self.W, self.b]

    def output(self):
        return T.nnet.sigmoid(T.dot(self.layer_input, self.W) + self.b)
