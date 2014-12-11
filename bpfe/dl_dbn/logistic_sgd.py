"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""

import numpy

import theano
import theano.tensor as T


debug = False


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input_vector, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input_vector: theano.tensor.TensorType
        :param input_vector: symbolic variable that describes the input of the
         architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
         which the data points lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
         which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        # noinspection PyUnresolvedReferences
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        # noinspection PyUnresolvedReferences
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        if debug:
            self.p_y_given_x_raw = \
                T.nnet.softmax(T.dot(input_vector, self.W) + self.b)
            self.p_y_given_x = theano.printing.Print('p_y_given_x')(
                self.p_y_given_x_raw
            )
        else:
            self.p_y_given_x = \
                T.nnet.softmax(T.dot(input_vector, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        if False:
            pygx = theano.printing.Print('pygx')(self.p_y_given_x)
            am = theano.printing.Print('am')(T.argmax(pygx, axis=1))
            self.y_pred = am
        else:
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        # noinspection PyUnresolvedReferences
        l = T.log(self.p_y_given_x)
        t_arange = T.arange(y.shape[0])
        if debug:
            l_printed = theano.printing.Print('l')(l)
            t_arange_printed = theano.printing.Print('t_arange')(t_arange)
            y_printed = theano.printing.Print('y')(y)
            mn_printed = -T.mean(l_printed[t_arange_printed, y_printed])
            val_printed = theano.printing.Print('mean')(mn_printed)
            return val_printed
        else:
            return -T.mean(l[t_arange, y])

        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
         correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
        # noinspection PyUnresolvedReferences
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # check if y is of the correct data type
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            if False:
                y_pred_printed = theano.printing.Print('y_pred')(self.y_pred)
                y_printed = theano.printing.Print('y')(y)
                neq_printed = theano.printing.Print('neq')(T.neq(
                    y_pred_printed, y_printed
                ))
                mean_printed = theano.printing.Print('mean')(T.mean(
                    neq_printed))
                return mean_printed
            else:
                return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def predict_proba(self):
        return self.p_y_given_x
