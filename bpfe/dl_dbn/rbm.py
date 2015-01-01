"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""

import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams


# start-snippet-1
# noinspection PyCallingNonCallable
class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        n_in=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        lmbda=0.1,
        momentum=0.9,
        weight_decay_cost=0.001,
        numpy_rng=None,
        theano_rng=None
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_in: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_in
        self.n_hidden = n_hidden
        self.lmbda = lmbda
        self.weight_decay_cost = weight_decay_cost

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.monitoring_cost = None
        self.updates = None

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                self.numpy_rng.uniform(
                    # TODO figure out initial weights
                    # low=-.01,
                    # high=.01,
                    # low=-1./numpy.sqrt(n_in),
                    # high=1./numpy.sqrt(n_in),
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_in)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_in)),
                    size=(n_in, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        # track momentums
        self.Ms = []

        # track mean squareds
        self.MSs = []

        self.momentum = momentum

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_in,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]
        # end-snippet-1

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    # start-snippet-2
    def get_cost_updates(self, lr=0.1, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM
        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        chain_start = ph_sample
        # end-snippet-2
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))

        # noinspection PyUnresolvedReferences
        learning_rate = T.cast(0.01, dtype=theano.config.floatX)

        def momentumed(pidx, param):
            if len(self.Ms) < pidx + 1:
                # initialize momentum for this element to zeros
                self.Ms.append(theano.shared(
                    param.get_value() * 0.,
                    broadcastable=param.broadcastable
                ))

            M_update = self.Ms[pidx]

            grad = T.grad(cost, param, consider_constant=[chain_end])

            # noinspection PyUnresolvedReferences
            momentum = T.cast(self.momentum, dtype=theano.config.floatX)

            v_prime = momentum * M_update - learning_rate * grad
            w_prime = param + v_prime
            updates[M_update] = v_prime
            updates[param] = w_prime

        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for pidx, param in enumerate(self.params):
            # # We must not compute the gradient through the gibbs sampling
            # grad = T.grad(cost, param, consider_constant=[chain_end])
            # current_cost = learning_rate * grad
            # updates[param] = param - current_cost

            momentumed(pidx, param)

            # if len(self.Ms) < pidx + 1:
            #     # initialize momentum for this element to zeros
            #     self.Ms.append(theano.shared(
            #         param.get_value() * 0.,
            #         broadcastable=param.broadcastable
            #     ))
            #
            # # if len(self.MSs) < pidx + 1:
            # #     # initialize momentum for this element to zeros
            # #     # noinspection PyUnresolvedReferences
            # #     # self.MSs.append(theano.shared(1.))
            # #     self.MSs.append(theano.shared(
            # #         (param.get_value() * 0.) + 1,
            # #         broadcastable=param.broadcastable
            # #     ))
            #
            # M_update = self.Ms[pidx]
            # # MS_update = self.MSs[pidx]
            #
            # grad = T.grad(cost, param, consider_constant=[chain_end])
            #
            # # current_rmsprop = (0.9 * MS_update) + (0.1 * T.sqr(grad))
            #
            # # current_rmsprop = (0.9 * MS_update) + \
            # #                   (0.1 * T.mean(T.sqr(grad)))
            #
            # # updates[MS_update] = current_rmsprop
            #
            # # noinspection PyUnresolvedReferences
            # momentum = T.cast(self.momentum, dtype=theano.config.floatX)
            #
            # # # noinspection PyUnresolvedReferences
            # # learning_rate = T.sqrt(
            # #     T.cast(current_rmsprop, dtype=theano.config.floatX)
            # # )
            # learning_rate = 0.1
            # # learning_rate = theano.printing.Print('lr')(learning_rate)
            #
            # current_cost = learning_rate * grad
            # updates[param] = param - current_cost
            # # updates[param] = param - (learning_rate * grad)
            #
            # updates[M_update] = momentum * M_update + current_cost

        # reconstruction cross-entropy is a better proxy for CD
        monitoring_cost = self.get_reconstruction_cost(
            updates,
            pre_sigmoid_nvs[-1]
        )

        self.monitoring_cost = monitoring_cost
        self.updates = updates
        self.learning_rate = lr
        # end-snippet-4

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """
        y = self.input
        a = T.nnet.sigmoid(pre_sigmoid_nv)

        cross_entropy = T.mean(
            T.sum(y * T.log(a) + (1 - y) * T.log(1 - a), axis=1)
        )

        n = y.shape[0]

        regularization = \
            (self.lmbda / (2. * n)) * \
            (self.weight_decay_cost * T.sum(T.sqr(self.W)))

        regularized_cross_entropy = cross_entropy + regularization

        return regularized_cross_entropy
