"""
"""
import math

import numpy
from scipy.sparse import csr_matrix

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression
from rbm import RBM


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, numpy_rng, layer_input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
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

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            # noinspection PyUnresolvedReferences
            W_values = numpy.asarray(
                numpy_rng.uniform(
                    # TODO figure out initial weights
                    # low=-1./numpy.sqrt(n_in),
                    # high=-1./numpy.sqrt(n_in),
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            # noinspection PyUnresolvedReferences
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(layer_input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-1
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
    def __init__(self, n_ins=784, hidden_layers_sizes=[500, 500], n_outs=10):
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

        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to changing the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

    def create_hidden_layer(self, layer_num, numpy_rng, theano_rng,
                            train_size):
        # construct the sigmoidal layer
        # the size of the input is either the number of hidden
        # units of the layer below or the input size if we are on
        # the first layer
        if layer_num == 0:
            input_size = self.number_of_inputs
        else:
            input_size = self.hidden_layer_sizes[layer_num - 1]

        # the input to this layer is either the activation of the
        # hidden layer below or the input of the DBN if you are on
        # the first layer
        if layer_num == 0:
            layer_input = self.x
        else:
            layer_input = self.sigmoid_layers[-1].output

        sigmoid_layer = HiddenLayer(
            numpy_rng=numpy_rng,
            layer_input=layer_input,
            n_in=input_size,
            n_out=self.hidden_layer_sizes[layer_num],
            activation=T.nnet.sigmoid
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
            numpy_rng=numpy_rng,
            theano_rng=theano_rng,
            input=layer_input,
            n_in=input_size,
            n_hidden=self.hidden_layer_sizes[layer_num],
            W=sigmoid_layer.W,
            hbias=sigmoid_layer.b,
            # seems reasonable
            lmbda=train_size / 1000.
        )
        self.rbm_layers.append(rbm_layer)

        return rbm_layer

    def create_output_layer(self, train_size, numpy_rng):
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input_vector=self.sigmoid_layers[-1].output,
            n_in=self.hidden_layer_sizes[-1],
            n_out=self.number_of_outputs,
            lmbda=train_size / 1000.,
            numpy_rng=numpy_rng
        )
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.bpfe_log_loss(self.y)
        #
        # # compute the gradients with respect to the model parameters
        # # symbolic variable that points to the number of errors made on the
        # # minibatch given by self.x and self.y
        # self.errors = self.logLayer.errors(self.y)

    def pretraining_function(self, train_set_x, batch_size, k, layer_num,
                             numpy_rng, theano_rng, train_size, load=False):
        """Generates a list of functions, for performing one step of
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
                layer_num, numpy_rng, theano_rng, train_size)

            # get the cost and the updates list
            # using CD-k here (persistent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            rbm.get_cost_updates(learning_rate, persistent=None, k=k)

        # compile the theano function
        fn = theano.function(
            inputs=[index, rbm.learning_rate],
            outputs=rbm.monitoring_cost,
            updates=rbm.updates,
            givens={
                self.x: train_set_x[batch_begin:batch_end]
            }
        )

        def recursive(X, total_layers, current_layer):
            if current_layer != 0:
                X = recursive(
                    X,
                    total_layers,
                    current_layer-1
                )

            if total_layers == current_layer:
                return X
            else:
                v = self.rbm_layers[current_layer].sample_h_given_v(X)[2]
                return v

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

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        """

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        (submission_set_x, submission_set_y) = datasets[3]

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        train_predict_proba_i = theano.function(
            [index],
            self.logLayer.predict_proba(),
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_predict_proba_i = theano.function(
            [index],
            self.logLayer.predict_proba(),
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_predict_proba_i = theano.function(
            [index],
            self.logLayer.predict_proba(),
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        submission_predict_proba_i = theano.function(
            [index],
            self.logLayer.predict_proba(),
            givens={
                self.x: submission_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        #### PREDICTION PROBABILITY FUNCTIONS

        def train_preds(gen):
            return pred(gen, train_set_x, train_predict_proba_i)

        def valid_preds(gen):
            return pred(gen, valid_set_x, valid_predict_proba_i)

        def test_preds(gen):
            return pred(gen, test_set_x, test_predict_proba_i)

        def submission_preds(gen):
            return pred(gen, submission_set_x, submission_predict_proba_i)

        def pred(data, X, predict_proba_i):
            predict_probas = None
            to_gpu_size = 5000
            to_gpu_batches = int(
                math.ceil(data.shape[0] / float(to_gpu_size))
            )
            for to_gpu_batch in range(to_gpu_batches):
                start = (to_gpu_batch * to_gpu_size)
                end = min(start + to_gpu_size, data.shape[0])
                # noinspection PyUnresolvedReferences
                subset_data = csr_matrix(
                    data[start:end],
                    dtype=theano.config.floatX
                )
                subset_data = subset_data.todense()
                X.set_value(subset_data, borrow=True)

                n_batches = int(
                    math.ceil(subset_data.shape[0] / float(10))
                )
                for batch_index in xrange(n_batches):
                    if predict_probas is None:
                        predict_probas = predict_proba_i(batch_index)
                    else:
                        predict_probas = numpy.concatenate(
                            (predict_probas, predict_proba_i(batch_index)),
                            axis=0
                        )
            return predict_probas

        return train_fn, train_preds, valid_preds, test_preds, submission_preds
