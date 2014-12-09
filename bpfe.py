

import os
from os.path import dirname
import sys
import time
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from bpfe import scoring, save
from bpfe.config import FLAT_LABELS, KLASS_LABEL_INFO
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
# from memory_profiler import profile


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

NP_NDARRAY = None
NP_ARRAY = None


def _get_np_ndarray(shape, dtype):
    global NP_NDARRAY
    if not NP_NDARRAY is None and \
       NP_NDARRAY.shape == shape and \
       NP_NDARRAY.dtype == dtype:
        return NP_NDARRAY

    NP_NDARRAY = np.empty(shape, dtype, order='C')
    return NP_NDARRAY


def _get_np_array(shape, dtype):
    global NP_ARRAY
    if not NP_ARRAY is None and \
       NP_ARRAY.shape == shape and \
       NP_ARRAY.dtype == dtype:
        return NP_ARRAY

    NP_ARRAY = np.empty(shape, dtype, order='C')
    return NP_ARRAY


def vectorize(generator, X, Y, num_chunks, klass_num):
    batch_size = 5000
    data_len, index = 0, 0
    full_data, full_labels = None, None
    for data, labels in generator(num_chunks):
        if full_data is None:
            # noinspection PyUnresolvedReferences
            full_data = _get_np_ndarray(
                shape=(batch_size, data.shape[1]),
                dtype=theano.config.floatX
            )
            # noinspection PyUnresolvedReferences
            full_labels = _get_np_array(
                shape=(batch_size,),
                dtype=theano.config.floatX
            )
        data_len += data.shape[0]

        start = index * data.shape[0]
        end = start + data.shape[0]
        full_data[start:end, :] = data
        if labels is not None:
            full_labels[start:end] = labels[:, klass_num]

        done = False
        # 5000 == batch_size in save.XXX
        if data.shape[0] < 5000:
            full_data = full_data[:data_len, :]
            if full_labels is not None:
                full_labels = full_labels[:data_len]
            done = True

        if not done and data_len < batch_size:
            continue

        X.set_value(full_data, borrow=True)
        if labels is not None:
            Y.set_value(full_labels, borrow=True)

        yield data_len

        full_data, full_labels = None, None
        data_len = 0

    if full_data is not None:
        X.set_value(full_data, borrow=True)
        if full_labels is not None:
            Y.set_value(full_labels, borrow=True)

        yield data_len


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


def test_DBN():
    """ """
    # k_opts = [5]
    # batch_size_opts = [10]
    # hidden_layer_size_opts = [500]
    # pretrain_lr_opts = [0.01, 0.001]
    # finetune_lr_opts = [0.1, 0.01]
    # hidden_layer_depth_opts = [3, 4]

    k_opts = [1]
    hidden_layer_depth_opts = [3]
    batch_size_opts = [5]
    hidden_layer_size_opts = [500]
    pretrain_lr_opts = [0.01]
    finetune_lr_opts = [0.1]
    combos = []
    for ko in k_opts:
        for bso in batch_size_opts:
            for hlso in hidden_layer_size_opts:
                for plo in pretrain_lr_opts:
                    for flo in finetune_lr_opts:
                        for hldo in hidden_layer_depth_opts:
                            combos.append((ko, bso, hlso, hldo, flo, plo))
    for ko, bso, hlso, hldo, flo, plo in combos:
        hls = [hlso for _ in range(hldo)]
        start = time.clock()
        _run_with_params(
            finetune_lr=flo,
            pretraining_epochs=100,
            pretrain_lr=plo,
            k=ko,
            training_epochs=100,
            batch_size=bso,
            hidden_layer_sizes=hls
        )
        print('DURATION: {}'.format(time.clock() - start))

    settings_stats_fname = 'data/settings_stats.pkl'
    if os.path.exists(settings_stats_fname):
        with open(settings_stats_fname, 'rb') as ifile:
            settings_stats = pickle.load(ifile)
            for klass, all_stats in settings_stats.items():
                all_stats.sort(key=lambda x: x[3], reverse=True)

                print('for {}:'.format(klass))
                for ind, stats in enumerate(all_stats):
                    settings_key = ', '.join([
                        str(k) + ': ' + str(v) for k, v in stats[0].items()
                    ])
                    print('\t{}: {}, {} for [{}]'.format(
                        ind, stats[3], stats[1], settings_key
                    ))


def _run_with_params(finetune_lr, pretraining_epochs, pretrain_lr, k,
                     training_epochs, batch_size, hidden_layer_sizes):
    """
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
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    total_size = 0

    def shared_dataset():
        # noinspection PyUnresolvedReferences
        shared_x = theano.shared(
            numpy.asarray([[]], dtype=theano.config.floatX),
            borrow=True
        )
        # noinspection PyUnresolvedReferences
        shared_y = theano.shared(
            numpy.asarray([], dtype=theano.config.floatX),
            borrow=True
        )
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset()
    valid_set_x, valid_set_y = shared_dataset()
    train_set_x, train_set_y = shared_dataset()
    datasets = [
        (train_set_x, train_set_y),
        (valid_set_x, valid_set_y),
        (test_set_x, test_set_y)
    ]

    train_chunks = save.TRAIN_CHUNKS
    validate_chunks = 1
    test_chunks = 1

    # train_len = 0
    # for data, _ in save.train(train_chunks):
    #     train_len += len(data)
    #
    # validate_len = 0
    # for data, _ in save.validate(validate_chunks):
    #     validate_len += len(data)
    #
    # test_len = 0
    # for data, _ in save.test(test_chunks):
    #     test_len += len(data)
    #
    # print('train size: {}'.format(train_len))
    # print('validate size: {}'.format(validate_len))
    # print('test size: {}'.format(test_len))

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

    for _ in vectorize(save.train, train_set_x, train_set_y, train_chunks, 0):
        break
    input_size = train_set_x.get_value(borrow=True).shape[1]

    settings_stats_fname = 'data/settings_stats.pkl'
    if os.path.exists(settings_stats_fname):
        with open(settings_stats_fname, 'rb') as ifile:
            settings_stats = pickle.load(ifile)
    else:
        settings_stats = dict()

    settings = {
        'k': k,
        'pretraining_epochs': pretraining_epochs,
        'batch_size': batch_size,
        'pretrain_lr': pretrain_lr,
        'finetune_lr': finetune_lr,
        'train_chunks': train_chunks,
        'validate_chunks': validate_chunks,
        'test_chunks': test_chunks,
        'hidden_layer_sizes': '_'.join([str(z) for z in hidden_layer_sizes]),
        'training_epochs': training_epochs
    }

    def settings_name(s):
        tups = [(k, v) for k, v in s.items()]
        tups.sort(key=lambda x: x[0])
        return ', '.join([str(k) + ': ' + str(v) for k, v in tups])

    current_settings_key = settings_name(settings)

    already_ran = False
    a_class = KLASS_LABEL_INFO.keys()[0]
    if a_class in settings_stats:
        for data in settings_stats[a_class]:
            data_settings_key = settings_name(data[0])
            if data_settings_key == current_settings_key:
                already_ran = True

    if already_ran:
        return

    file_key = {
        'k': k,
        'batch_size': batch_size,
        'pretrain_lr': pretrain_lr,
        'hidden_layer_sizes': str(hidden_layer_sizes[0])
    }
    settings_values = sorted(file_key.values(), key=lambda x: x)

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
        hidden_layers_sizes=hidden_layer_sizes,
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
        rbm_info = [
            i,
            dbn.hidden_layer_sizes[i]
        ] + settings_values
        filename = 'data/hidden_layers/{}.pkl'.format(
            '-'.join([str(s) for s in rbm_info])
        )
        if os.path.exists(filename):
            print('layer {} already exists on disk, loading ...'.format(i))
            with open(filename, 'rb') as ifile:
                data = pickle.load(ifile)
                dbn = data[0]
                dbn.hidden_layer_sizes = hidden_layer_sizes
                total_size = data[1]
                numpy_rng = data[2]
                theano_rng = data[3]
                continue

        print('getting the pre-training function for layer {} ...'.format(i))
        pretraining_fn = dbn.pretraining_function(
            train_set_x,
            settings['batch_size'],
            settings['k'],
            i,
            numpy_rng,
            theano_rng
        )

        # go through pretraining epochs
        pepochs = settings['pretraining_epochs']
        if i > 0:
            pepochs /= 2
        for epoch in xrange(pepochs):
            start = time.clock()
            # go through the training set
            c = []

            for row_len in vectorize(
                    save.train, train_set_x, train_set_y, train_chunks, 0):
                if epoch == 0:
                    total_size += row_len

                # compute number of minibatches for training, validation and testing
                n_train_batches = row_len / settings['batch_size']
                for batch_index in xrange(n_train_batches):
                    c.append(
                        pretraining_fn(index=batch_index,
                                       lr=settings['pretrain_lr'])
                    )

            print(
                'Pre-training layer {}, epoch {}, cost {}, elapsed {}'.format(
                    i, epoch, numpy.mean(c), time.clock() - start
                )
            )

        with open(filename, 'wb') as ifile:
            data = (dbn, total_size, numpy_rng, theano_rng)
            pickle.dump(data, ifile)

    total_size /= dbn.n_layers
    # print('ts', total_size)

    end_time = time.clock()
    # end-snippet-2
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ########################
    # FINETUNING THE MODEL #
    ########################

    # TODO since we're running a single class....
    # with open('data/current-dbn.pkl', 'wb') as ifile:
    #     pickle.dump(dbn, ifile)
    for klass, (klass_num, count) in KLASS_LABEL_INFO.items():
        if klass != 'Function':
            continue

        # TODO since we're running a single class....
        # with open('data/current-dbn.pkl', 'rb') as ifile:
        #     dbn = pickle.load(ifile)

        dbn.number_of_outputs = count
        dbn.create_output_layer()

        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = dbn.build_finetune_functions(
            datasets=datasets,
            batch_size=settings['batch_size'],
            learning_rate=settings['finetune_lr']
        )

        print '... finetuning the model'
        # early-stopping parameters
        n_train_batches = total_size / settings['batch_size']
        # print('tb', n_train_batches)
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
                for row_len in vectorize(
                        save.train, train_set_x, train_set_y, train_chunks,
                        klass_num):
                    sub_train_batches = row_len / settings['batch_size']

                    for minibatch_index in xrange(sub_train_batches):
                        train_fn(minibatch_index)
                        itr = (epoch - 1) * n_train_batches + idx
                        idx += 1

                    if (itr + 1) % validation_frequency == 0:

                        validation_losses = validate_model(
                            vectorize(save.validate, valid_set_x, valid_set_y,
                                      validate_chunks, klass_num)
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
                            test_losses = test_model(
                                vectorize(save.test, test_set_x, test_set_y,
                                          test_chunks, klass_num)
                            )
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
                'Optimization complete with best validation score of {}%, '
                'obtained at iteration {}, '
                'with test performance {}% '
                'for class {}'
            ).format(
                best_validation_loss * 100.,
                best_iter + 1,
                test_score * 100.,
                klass
            )
        )
        print >> sys.stderr, (
            'The fine tuning code for file ' +
            os.path.split(__file__)[1] +
            ' ran for %.2fm' % ((end_time - start_time) / 60.)
        )

    # with open('data/final-model.pkl', 'wb') as ifile:
    #     pickle.dump(dbn, ifile)

        # stats = [
        #     settings,
        #     best_validation_loss,
        #     best_iter,
        #     test_score
        # ]
        #
        # if klass not in settings_stats:
        #     settings_stats[klass] = []
        #
        # for data in settings_stats[klass]:
        #     data_settings_key = settings_name(data[0])
        #     if data_settings_key == current_settings_key:
        #         break
        #
        # settings_stats[klass].append(stats)
        #
        # with open(settings_stats_fname, 'wb') as ifile:
        #     pickle.dump(settings_stats, ifile)


if __name__ == '__main__':
    # predict()
    # predict_train()
    test_DBN()
    # run()
    # stats()
