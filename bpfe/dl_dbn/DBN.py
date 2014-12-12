"""
"""
import math

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression
from rbm import RBM


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, layer_input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

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
                rng.uniform(
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
        # the labels are presented as 1D vector of [int] labels
        self.y = T.ivector('y')

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

    def create_hidden_layer(self, layer_num, numpy_rng, theano_rng):
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
            rng=numpy_rng,
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
            n_visible=input_size,
            n_hidden=self.hidden_layer_sizes[layer_num],
            W=sigmoid_layer.W,
            hbias=sigmoid_layer.b
        )
        self.rbm_layers.append(rbm_layer)

        return rbm_layer

    def create_output_layer(self):
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input_vector=self.sigmoid_layers[-1].output,
            n_in=self.hidden_layer_sizes[-1],
            n_out=self.number_of_outputs
        )
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_function(self, train_set_x, batch_size, k, layer_num,
                             numpy_rng, theano_rng):
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

        rbm = self.create_hidden_layer(layer_num, numpy_rng, theano_rng)

        # get the cost and the updates list
        # using CD-k here (persistent=None) for training each RBM.
        # TODO: change cost function to reconstruction error
        cost, updates = rbm.get_cost_updates(
            learning_rate, persistent=None, k=k)

        # compile the theano function
        fn = theano.function(
            inputs=[index, theano.Param(learning_rate, default=0.1)],
            outputs=cost,
            updates=updates,
            givens={
                self.x: train_set_x[batch_begin:batch_end]
            }
        )
        return fn

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
                self.y: T.cast(train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ], 'int32')
            }
        )

        train_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: T.cast(train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ], 'int32')
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: T.cast(test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ], 'int32')
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: T.cast(valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ], 'int32')
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

        #### SCORING FUNCTIONS

        # Create a function that scans the entire train set
        def train_score(gen):
            return score(gen, train_set_x, train_score_i)

        # Create a function that scans the entire validation set
        def valid_score(gen):
            return score(gen, valid_set_x, valid_score_i)

        # Create a function that scans the entire test set
        def test_score(gen):
            return score(gen, test_set_x, test_score_i)

        def score(gen, X, score_i):
            scores = []
            for _ in gen:
                batches = X.get_value(borrow=True).shape[0]
                batches = int(math.ceil(batches / float(batch_size)))
                scores += [score_i(i) for i in xrange(batches)]
            return scores

        #### PREDICTION PROBABILITY FUNCTIONS

        def train_preds(gen):
            return pred(gen, train_set_x, train_predict_proba_i)

        def test_preds(gen):
            return pred(gen, test_set_x, test_predict_proba_i)

        def submission_preds(gen):
            return pred(gen, submission_set_x, submission_predict_proba_i)

        def pred(gen, X, predict_proba_i):
            predict_probas = None
            for _ in gen:
                batches = X.get_value(borrow=True).shape[0]
                batches = int(math.ceil(batches / float(batch_size)))
                for i in xrange(batches):
                    if predict_probas is None:
                        predict_probas = predict_proba_i(i)
                    else:
                        predict_probas = numpy.concatenate(
                            (predict_probas, predict_proba_i(i)),
                            axis=0
                        )
            return predict_probas

        return train_fn, train_score, valid_score, test_score, \
            train_preds, test_preds, submission_preds
