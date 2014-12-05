

import os
from os.path import dirname
import sys
import math
import time
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from bpfe import scoring, feature_engineering
from bpfe.config import FLAT_LABELS
from bpfe.dl_dbn.logistic_sgd import load_data
from bpfe.dl_dbn.DBN import DBN
import numpy
import bpfe.load as load
from bpfe.models.perceptron_model import PerceptronModel
import numpy as np
# noinspection PyBroadException
try:
    import cPickle as pickle
except:
    import pickle


# SEED = random.randint(1, sys.maxint)
SEED = 1
# AMT = 20000
AMT = 400277
ITERATIONS = 60
loc = dirname(__file__) + '/results'
if not os.path.exists(loc):
    os.makedirs(loc)
filename = '{}-{}'.format(AMT, ITERATIONS)
loc += '/' + filename


label_tracker = dict()
label_tracker_idx = 0


def vectorizers(vectzers, dtype=theano.config.floatX):
    global label_tracker_idx

    attributes = [
        'object_description', 'program_description',
        'subfund_description', 'job_title_description',
        'facility_or_department', 'sub_object_description',
        'location_description', 'fte', 'function_description', 'position_extra',
        'text_4', 'total', 'text_2', 'text_3', 'fund_description', 'text_1'
    ]

    def vectorize(generator, X, Y, num_chunks, batch_size=1000, skip_labels=False):
        global label_tracker_idx
        for data in generator(num_chunks):
            total_size = len(data)
            batches = int(math.ceil(total_size / batch_size))
            for i in range(batches):
                sub_data = data[int(i*batch_size): int((i+1)*batch_size)]
                vecs = []
                for d, _ in sub_data:
                    avecs = []
                    for attr in attributes:
                        v, settings, method_name = vectzers[attr]
                        m = getattr(feature_engineering, method_name)
                        avec = m(v, getattr(d, attr), settings)
                        avecs.append(avec)

                    vecs.append(np.concatenate(avecs, axis=1)[0])

                try:
                    vecs = np.array(vecs, dtype=dtype)
                except:
                    print(vecs.shape, i)
                    raise

                if not skip_labels:
                    labels = []
                    for _, label in sub_data:
                        if label is None:
                            break

                        val = ''.join([str(i) for i in label.to_vec()])
                        if val not in label_tracker:
                            idx = label_tracker_idx
                            label_tracker[val] = idx
                            label_tracker_idx += 1
                        else:
                            idx = label_tracker[val]

                        labels.append(idx)

                    if len(labels) > 0:
                        labels = np.array(labels, dtype=dtype)
                else:
                    labels = None

                X.set_value(vecs, borrow=True)
                Y.set_value(labels, borrow=True)

                yield vecs, labels

    return vectorize


def run():
    # facility_or_department.info()
    # program_description.info()
    # job_title_description.info()
    # sub_object_description.info()
    # location_description.info()
    # function_description.info()
    # position_extra.info()
    # subfund_description.info()
    # fund_description.info()
    # object_description.info()
    # text_1.info()

    # text_2.info()
    # text_3.info()
    # text_4.info()
    # fte.info()
    # total.info()

    load.store_raw()

    # for key, value in v.iteritems():
    #     vec = value[0]
    #     print(key)
    #     print(vec.labels)

    # train, validate, test, vec = vectorizers(1)
    # for t, l in vec(train):
    #     print('t: {}, l: {}'.format(t.shape, l.shape))

    # train, validate, test = load_vectorized('simple-1')
    # stop = 'here'

    # pm = PerceptronModel()
    # # amt = sys.maxint
    #
    # pm.train(train, test, AMT, nr_iter=ITERATIONS, seed=SEED, save_loc=loc)
    # plotting.plot_train_vs_validation(pm.scores, AMT)


def predict():
    with open('results/predictions-{}'.format(filename), 'w') as outfile:
        def write(x):
            outfile.write(x + '\n')
        _predict(write)


def _predict(output_method, num_rows=None):
    if num_rows is None:
        num_rows = sys.maxint

    pm = PerceptronModel()
    pm.load(loc)

    header = ['__'.join(i) for i in FLAT_LABELS]
    headers = []
    for i in header:
        if ' ' in i:
            i = '"{}"'.format(i)
        headers.append(i)

    output_line = ',' + ','.join(headers)
    output_method(output_line)

    i = 0
    for data in load.generate_submission_rows(SEED, sys.maxint):
        v = pm.predict_proba(data)
        pred_tups = [data.id]
        for raw_key in FLAT_LABELS:
            pred_tups.append(v[raw_key])

        output_line = ','.join([str(i) for i in pred_tups])
        output_method(output_line)

        i += 1
        if i > num_rows:
            break


def predict_train():
    pm = PerceptronModel()
    pm.load(loc)

    actuals = []
    predictions = []
    for label, data in load.generate_training_rows(pm.seed, pm.amt):
        # prediction = pm.predict(data)
        actuals.append(
            PerceptronModel.label_output(label)[1:]
        )
        v = pm.predict_proba(data)
        pred_tups = []
        for raw_key in FLAT_LABELS:
            pred_tups.append(v[raw_key])
        predictions.append(pred_tups)
        # predictions.append(
        #     PerceptronModel.prediction_output(data, prediction)[1:]
        # )

    print('')
    print('mc log loss: {}'.format(
        round(scoring.multi_multi(actuals, predictions), 5)
    ))


def stats():
    pm = PerceptronModel()
    pm.load(loc)

    print('')
    header = 'Model Information'
    print(header)
    print('-'*len(header))
    print(pm)
    pm.print_scores()










def test_DBN(finetune_lr=0.1, pretraining_epochs=100,
             pretrain_lr=0.01, k=1, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=5):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """
    batch_size = 5
    pretraining_epochs = 1
    training_epochs = 1

    total_size = 0

    datasets = load_data(dataset)

    num_chunks = 1
    v = load.load_vectorizers(num_chunks)

    # def print_sizes(gen, name):
    #     total_size = 0
    #     for i in gen():
    #         total_size += len(i)
    #     print('{} size: {}'.format(name, total_size))
    #
    # print_sizes(load.gen_train, 'train')
    # print_sizes(load.gen_validate, 'validate')
    # print_sizes(load.gen_test, 'test')
    # print_sizes(load.gen_submission, 'submission')

    gen = vectorizers(v)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    for _ in gen(load.gen_train, train_set_x, train_set_y, num_chunks):
        break
    input_size = train_set_x.get_value(borrow=True).shape[1]
    print('input size: {}'.format(input_size))

    # numpy random generator
    # noinspection PyUnresolvedReferences
    numpy_rng = numpy.random.RandomState(123)
    # theano random generator
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(
        n_ins=input_size,
        hidden_layers_sizes=[1000, 1000, 1000],
        n_outs=104
    )

    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):

        print '... getting the pre-training function'
        dbn.hidden_layer_sizes[i] = 1000
        pretraining_fn = dbn.pretraining_function(
            train_set_x,
            batch_size,
            k,
            i,
            numpy_rng,
            theano_rng
        )

        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            start = time.clock()
            # go through the training set
            c = []

            for data, labels in gen(load.gen_train, train_set_x, train_set_y, num_chunks):
                row_len = train_set_x.get_value(borrow=True).shape[0]
                if epoch == 0:
                    total_size += row_len

                # compute number of minibatches for training, validation and testing
                n_train_batches = row_len / batch_size
                for batch_index in xrange(n_train_batches):
                    c.append(
                        pretraining_fn(index=batch_index, lr=pretrain_lr)
                    )

            print(
                'Pre-training layer {}, epoch {}, cost {}, elapsed {}'.format(
                    i, epoch, numpy.mean(c), time.clock() - start
                )
            )

    total_size /= dbn.n_layers

    end_time = time.clock()
    # end-snippet-2
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ########################
    # FINETUNING THE MODEL #
    ########################

    dbn.create_output_layer()

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    n_train_batches = total_size / batch_size
    patience = 4 * n_train_batches  # look as this many examples regardless
    # go through this many
    # minibatches before checking the network
    # on the validation set; in this case we
    # check every epoch
    validation_frequency = min(n_train_batches, patience / 2)
    patience_increase = 2.    # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    itr = 0
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        idx = 0
        while idx < n_train_batches:
            for _ in gen(load.gen_train, train_set_x, train_set_y, num_chunks):
                row_len = train_set_x.get_value(borrow=True).shape[0]
                sub_train_batches = row_len / batch_size

                for minibatch_index in xrange(sub_train_batches):
                    train_fn(minibatch_index)
                    itr = (epoch - 1) * n_train_batches + idx
                    idx += 1

                if (itr + 1) % validation_frequency == 0:

                    validation_losses = validate_model(
                        gen(load.gen_validate, valid_set_x, valid_set_y, num_chunks)
                    )
                    this_validation_loss = numpy.mean(validation_losses)
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%'
                        % (
                            epoch,
                            idx,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                        ):
                            patience = max(patience, itr * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = itr

                        # test it on the test set
                        test_losses = test_model(gen(load.gen_test, test_set_x, test_set_y, num_chunks))
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, idx, n_train_batches, test_score * 100.))

                if patience <= itr:
                    done_looping = True
                    break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))





if __name__ == '__main__':
    # predict()
    # predict_train()
    test_DBN()
    # run()
    # stats()



