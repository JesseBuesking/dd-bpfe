"""
"""


import os
import gzip
import math

import numpy as np
from scipy.sparse import csr_matrix

import theano
import theano.tensor as T

from bpfe.dl_dbn.constants import DTYPES
from bpfe.dl_dbn.hidden_layer import SigmoidLayer
from logistic_sgd import LogisticRegression
from rbm import RBM


# noinspection PyBroadException
try:
    import cPickle as pickle
except:
    import pickle


# noinspection PyCallingNonCallable
class DBN(object):
    """
    Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    # noinspection PyDefaultArgument
    def __init__(
            self,
            name,
            n_ins=784,
            hidden_layers_sizes=[500, 500],
            n_outs=10
    ):
        """
        This class is made to support a variable number of layers.

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN
        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
         at least one value
        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.name = name
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.hidden_layer_sizes = hidden_layers_sizes
        self.number_of_inputs = n_ins
        self.number_of_outputs = n_outs

        assert self.n_layers > 0

        # allocate symbolic variables for the data
        # the data is presented as rasterized images
        self.x = T.matrix('x')
        # the labels are a matrix
        self.y = T.matrix('y')

        # track momentums
        self.Ms = []

        # track rmsprop
        self.MSs = []

        self.logLayer = None

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to changing the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

    def save(self):
        fname = 'data/pretrain-layer/dbn-{}.pkl'.format(self.name)
        with gzip.open(fname, 'wb') as ifile:
            objs = [
                self.sigmoid_layers,
                self.rbm_layers,
                self.n_layers,
                self.hidden_layer_sizes,
                self.number_of_inputs,
                self.number_of_outputs,
                self.Ms,
                self.MSs,
                self.logLayer,
                self.x,
                self.y
            ]
            for obj in objs:
                pickle.dump(obj, ifile, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        fname = 'data/pretrain-layer/dbn-{}.pkl'.format(self.name)
        # noinspection PyUnresolvedReferences
        if not os.path.exists(fname):
            return None

        with gzip.open(fname, 'rb') as ifile:
            self.sigmoid_layers = pickle.load(ifile)
            self.rbm_layers = pickle.load(ifile)
            self.n_layers = pickle.load(ifile)
            self.hidden_layer_sizes = pickle.load(ifile)
            self.number_of_inputs = pickle.load(ifile)
            self.number_of_outputs = pickle.load(ifile)
            self.Ms = pickle.load(ifile)
            self.MSs = pickle.load(ifile)
            self.logLayer = pickle.load(ifile)
            self.x = pickle.load(ifile)
            self.y = pickle.load(ifile)

            self.params = []
            for sigmoid_layer in self.sigmoid_layers:
                self.params.extend(sigmoid_layer.params)

            if self.logLayer is not None:
                self.params.extend(self.logLayer.params)

            return self

    def create_hidden_layer(
            self, layer_num, numpy_rng, train_size):
        # construct the sigmoidal layer
        # the size of the input is either the number of hidden
        # units of the layer below or the input size if we are on
        # the first layer
        input_size = self.number_of_inputs
        if layer_num != 0:
            input_size = self.hidden_layer_sizes[layer_num - 1]

        # the input to this layer is either the activation of the
        # hidden layer below or the input of the DBN if you are on
        # the first layer
        layer_input = self.x
        if layer_num != 0:
            layer_input = self.sigmoid_layers[-1].output()

        sigmoid_layer = SigmoidLayer(
            numpy_rng=numpy_rng,
            layer_input=layer_input,
            n_in=input_size,
            n_out=self.hidden_layer_sizes[layer_num],
        )

        # add the layer to our list of layers
        self.sigmoid_layers.append(sigmoid_layer)

        # its arguably a philosophical question...  but we are
        # going to only declare that the parameters of the
        # sigmoid_layers are parameters of the DBN. The visible
        # biases in the RBM are parameters of those RBMs, but not
        # of the DBN.
        self.params.extend(sigmoid_layer.params)

        # Construct an RBM that shared weights with this layer
        rbm_layer = RBM(
            numpy_seed=numpy_rng.randint(1e9),
            input_vector=layer_input,
            n_in=input_size,
            n_hidden=self.hidden_layer_sizes[layer_num],
            hidden_layer=sigmoid_layer,
            # seems reasonable
            lmbda=train_size / 1000.
        )
        self.rbm_layers.append(rbm_layer)

        return rbm_layer

    def create_output_layer(self, train_size):
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input_vector=self.sigmoid_layers[-1].output(),
            n_in=self.hidden_layer_sizes[-1],
            n_out=self.number_of_outputs,
            lmbda=train_size / 1000.
        )
        self.params.extend(self.logLayer.params)

    def pretraining_function(
            self, train_set_x, batch_size, k, layer_num, numpy_rng, theano_rng,
            train_size, load=False):
        """
        Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k
        """

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # beginning of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        if load:
            rbm = self.rbm_layers[layer_num]
        else:
            rbm = self.create_hidden_layer(
                layer_num,
                numpy_rng,
                train_size
            )

            # get the cost and the updates list
            # using CD-k here (persistent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            rbm.get_cost_updates(learning_rate, k=k)

        # compile the theano function
        fn = theano.function(
            inputs=[index],
            outputs=rbm.monitoring_cost,
            updates=rbm.updates,
            givens={
                self.x: train_set_x[batch_begin:batch_end]
            }
        )

        ##################################
        # computes the current free energy
        ##################################

        def recursive(X, total_layers, current_layer):
            # recursively propogate input data through each layer of the DBN
            if current_layer != 0:
                X = recursive(
                    X,
                    total_layers,
                    current_layer-1
                )

            if total_layers == current_layer:
                return X
            else:
                return self.rbm_layers[current_layer].sample_h_given_v(X)[2]

        # function to compute the actual free energy
        free_energy = theano.function(
            inputs=[],
            outputs=self.rbm_layers[-1].free_energy(
                recursive(self.x, layer_num, layer_num)
            ),
            givens={
                self.x: train_set_x
            }
        )

        return fn, free_energy

    def predict_proba(self, data, batch_size):
        X = theano.shared(
            np.asarray([[]], dtype=DTYPES.FLOATX),
            borrow=True
        )

        # index to a [mini]batch
        index = T.lscalar('index')

        func = theano.function(
            [index],
            self.logLayer.predict_proba(),
            givens={
                self.x: X[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        predict_probas = None
        to_gpu_size = 1000
        to_gpu_batches = int(
            math.ceil(data.shape[0] / float(to_gpu_size))
        )
        for to_gpu_batch in range(to_gpu_batches):
            start = (to_gpu_batch * to_gpu_size)
            end = min(start + to_gpu_size, data.shape[0])
            subset_data = csr_matrix(
                data[start:end],
                dtype=DTYPES.FLOATX
            )
            subset_data = subset_data.todense()
            X.set_value(subset_data, borrow=True)

            n_batches = int(
                math.ceil(subset_data.shape[0] / float(10))
            )
            for batch_index in xrange(n_batches):
                if predict_probas is None:
                    predict_probas = func(batch_index)
                else:
                    predict_probas = np.concatenate(
                        (predict_probas, func(batch_index)),
                        axis=0
                    )
            del subset_data
        return predict_probas

    def finetune(self, dataX, dataY, batch_size, learning_rate, momentum):
        X = theano.shared(
            np.asarray([[]], dtype=DTYPES.FLOATX),
            borrow=True
        )
        Y = theano.shared(
            np.asarray([[]], dtype=DTYPES.FLOATX),
            borrow=True
        )

        # index to a [mini]batch
        index = T.lscalar('index')
        learning_rate = T.cast(learning_rate, dtype=DTYPES.FLOATX)
        momentum = T.cast(momentum, dtype=DTYPES.FLOATX)

        def momentumed(pidx, param, momentum):
            # We must not compute the gradient through the gibbs sampling

            if len(self.Ms) < pidx + 1:
                # initialize momentum for this element to zeros
                self.Ms.append(theano.shared(
                    param.get_value() * 0.,
                    broadcastable=param.broadcastable
                ))

            M_update = self.Ms[pidx]

            grad = T.grad(self.logLayer.bpfe_log_loss(self.y), param)

            momentum = T.cast(momentum, dtype=DTYPES.FLOATX)

            v_prime = momentum * M_update - learning_rate * grad
            w_prime = param + v_prime
            updates.append((M_update, v_prime))
            updates.append((param, w_prime))

        # compute list of fine-tuning updates
        updates = []
        for pidx, param in enumerate(self.params):
            momentumed(pidx, param, momentum)

        train_fn = theano.function(
            inputs=[index],
            outputs=self.logLayer.bpfe_log_loss(self.y),
            updates=updates,
            givens={
                self.x: X[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: Y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        to_gpu_size = 1000
        to_gpu_batches = int(
            math.ceil(dataX.shape[0] / float(to_gpu_size))
        )
        for to_gpu_batch in range(to_gpu_batches):
            start = (to_gpu_batch * to_gpu_size)
            end = min(start + to_gpu_size, dataX.shape[0])
            subset_data = csr_matrix(
                dataX[start:end],
                dtype=DTYPES.FLOATX
            )
            subset_data = subset_data.todense()
            subset_labels = np.array(
                dataY[start:end],
                dtype=DTYPES.FLOATX
            )
            X.set_value(subset_data, borrow=True)
            Y.set_value(subset_labels, borrow=True)

            n_train_batches = int(
                math.ceil(subset_data.shape[0] / float(batch_size))
            )
            for batch_index in xrange(n_train_batches):
                train_fn(batch_index)
            del subset_data
