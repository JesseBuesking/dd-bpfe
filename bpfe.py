

import os
from os.path import dirname
import sys
import time
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from bpfe import scoring, save, cache
from bpfe.cache import load_stats
from bpfe.config import FLAT_LABELS, KLASS_LABEL_INFO, Settings, \
    HiddenLayerSettings, FinetuningSettings, ChunkSettings
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

    settings = Settings()
    settings.version = 1.0
    settings.k = 1
    settings.hidden_layers = [
        HiddenLayerSettings(50, 1, 0.01),
        HiddenLayerSettings(50, 1, 0.01),
        HiddenLayerSettings(50, 1, 0.01)
    ]
    settings.batch_size = 5
    settings.finetuning = FinetuningSettings(1, 0.1)
    settings.chunks = ChunkSettings(save.TRAIN_CHUNKS, 1, 1)

    # combos = []
    # for ko in k_opts:
    #     for bso in batch_size_opts:
    #         for hlso in hidden_layer_size_opts:
    #             for plo in pretrain_lr_opts:
    #                 for flo in finetune_lr_opts:
    #                     for hldo in hidden_layer_depth_opts:
    #                         combos.append((ko, bso, hlso, hldo, flo, plo))
    # for ko, bso, hlso, hldo, flo, plo in combos:
    # hls = [hlso for _ in range(hldo)]
    start = time.clock()
    _run_with_params(settings)
    print('DURATION: {}'.format(time.clock() - start))

    # settings_stats_fname = 'data/settings_stats.pkl'
    # if os.path.exists(settings_stats_fname):
    #     with open(settings_stats_fname, 'rb') as ifile:
    #         settings_stats = pickle.load(ifile)
    #         for klass, all_stats in settings_stats.items():
    #             all_stats.sort(key=lambda x: x[3], reverse=True)
    #
    #             print('for {}:'.format(klass))
    #             for ind, stats in enumerate(all_stats):
    #                 settings_key = ', '.join([
    #                     str(k) + ': ' + str(v) for k, v in stats[0].items()
    #                 ])
    #                 print('\t{}: {}, {} for [{}]'.format(
    #                     ind, stats[3], stats[1], settings_key
    #                 ))


def _run_with_params(settings):
    """
    :param Settings settings:
    """

    # deal_gen = load.gen_train(
    #     num_chunks=None, batch_size=15000
    # )
    #
    # d = dict()
    # for idx, data in enumerate([dg for dg in deal_gen][0]):
    #     if data[1].function not in d:
    #         d[data[1].function] = 0
    #     d[data[1].function] += 1
    # tups = [(a, b) for (a, b) in d.items()]
    # tups.sort(key=lambda x: x[0])
    # print(tups)
    # raise

    settings.train_size = 0

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
    submission_set_x, submission_set_y = shared_dataset()
    datasets = [
        (train_set_x, train_set_y),
        (valid_set_x, valid_set_y),
        (test_set_x, test_set_y),
        (submission_set_x, submission_set_y)
    ]

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

    settings.train_size = 0
    settings.num_cols = 0
    for _ in vectorize(
            save.train,
            train_set_x,
            train_set_y,
            settings.chunks.train,
            0
    ):
        settings.train_size += train_set_x.get_value(borrow=True).shape[0]
    settings.num_cols = train_set_x.get_value(borrow=True).shape[1]

    settings_stats = load_stats()

    already_ran = False
    a_class = KLASS_LABEL_INFO.keys()[0]
    if a_class in settings_stats:
        for data in settings_stats[a_class]:
            if data[0] == settings:
                already_ran = True

    if already_ran:
        return

    print('input size: {}'.format(settings.num_cols))

    # numpy random generator
    # noinspection PyUnresolvedReferences
    settings.numpy_rng = numpy.random.RandomState(123)
    # theano random generator
    settings.theano_rng = RandomStreams(
        settings.numpy_rng.randint(2 ** 30)
    )

    print('... building the model')

    # construct the Deep Belief Network
    dbn = DBN(
        n_ins=settings.num_cols,
        hidden_layers_sizes=[hl.num_nodes for hl in settings.hidden_layers],
        n_outs=104
    )

    # start-snippet-2
    #########################
    # PRETRAINING THE MODEL #
    #########################

    print('... pre-training the model')

    start_time = time.clock()
    ## Pre-train layer-wise
    for layer_idx in xrange(dbn.n_layers):
        cache_pretrain = cache.load_pretrain_layer(layer_idx, settings)
        if cache_pretrain is not None:
            print('layer {} already exists on disk, loading ...'.format(
                layer_idx
            ))
            dbn, settings = cache_pretrain
            continue

        print('getting the pre-training function for layer {} ...'.format(
            layer_idx
        ))
        pretraining_fn = dbn.pretraining_function(
            train_set_x,
            settings.batch_size,
            settings.k,
            layer_idx,
            settings.numpy_rng,
            settings.theano_rng
        )

        # go through pretraining epochs
        for epoch in xrange(settings.hidden_layers[layer_idx].epochs):
            start = time.clock()
            # go through the training set
            c = []

            for row_len in vectorize(
                    save.train,
                    train_set_x,
                    train_set_y,
                    settings.chunks.train,
                    0
            ):

                # compute number of mini-batches for training, validation and
                # testing
                n_train_batches = row_len / settings.batch_size
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fn(
                        index=batch_index,
                        lr=settings.hidden_layers[layer_idx].learning_rate
                    ))

            print(
                'Pre-training layer {}, epoch {}, cost {}, elapsed {}'.format(
                    layer_idx, epoch, numpy.mean(c), time.clock() - start
                )
            )

        cache.save_pretrain_layer(dbn, layer_idx, settings)

    # print('ts', settings.train_size)

    end_time = time.clock()
    # end-snippet-2
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ########################
    # FINETUNING THE MODEL #
    ########################

    finetune(dbn, datasets, settings)


def finetune(dbn, datasets, settings):

    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x, test_set_y) = datasets[2]
    (submission_set_x, submission_set_y) = datasets[3]

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

        val = cache.load_finetuning(settings, klass_num)
        if val is not None:
            dbn, settings = val
            train_fn, train_model, validate_model, test_model, \
                train_pred, submission_pred = \
                    dbn.build_finetune_functions(
                        datasets=datasets,
                        batch_size=settings.batch_size,
                        learning_rate=settings.finetuning.learning_rate
                    )
        else:
            print('... finetuning the model')
            dbn.create_output_layer()

            # get the training, validation and testing function for the model
            print('... getting the finetuning functions')
            train_fn, train_model, validate_model, test_model, \
                train_pred, submission_pred = \
                dbn.build_finetune_functions(
                    datasets=datasets,
                    batch_size=settings.batch_size,
                    learning_rate=settings.finetuning.learning_rate
                )

            n_train_batches = settings.train_size / settings.batch_size
            patience = 4 * n_train_batches
            validation_frequency = min(n_train_batches, patience / 2)
            patience_increase = 2.
            improvement_threshold = 0.995

            best_validation_loss = numpy.inf
            test_score = 0.
            start_time = time.clock()

            done_looping = False
            epoch = 0

            itr = 0
            while (epoch < settings.finetuning.epochs) and (not done_looping):
                epoch += 1
                idx = 0
                while idx < n_train_batches:
                    for row_len in vectorize(
                            save.train,
                            train_set_x,
                            train_set_y,
                            settings.chunks.train,
                            klass_num
                    ):
                        sub_train_batches = row_len / settings.batch_size

                        for minibatch_index in xrange(sub_train_batches):
                            train_fn(minibatch_index)
                            itr = (epoch - 1) * n_train_batches + idx
                            idx += 1

                        if (itr + 1) % validation_frequency == 0:

                            validation_losses = validate_model(vectorize(
                                save.validate,
                                valid_set_x,
                                valid_set_y,
                                settings.chunks.validate,
                                klass_num
                            ))
                            this_validation_loss = numpy.mean(validation_losses)

                            train_losses = train_model(vectorize(
                                save.train,
                                train_set_x,
                                train_set_y,
                                settings.chunks.train,
                                klass_num
                            ))
                            this_train_loss = numpy.mean(train_losses)
                            print(
                                'epoch {}, validation error {:.3f}% {:.3f}%'.format(
                                    epoch,
                                    this_validation_loss * 100.,
                                    this_train_loss * 100.
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

                                cache.save_finetuning(dbn, settings, klass_num)

                                # test it on the test set
                                test_losses = test_model(vectorize(
                                    save.test,
                                    test_set_x,
                                    test_set_y,
                                    settings.chunks.test,
                                    klass_num
                                ))
                                test_score = numpy.mean(test_losses)
                                print('  epoch {}, test error {}%'.format(
                                    epoch, test_score * 100.
                                ))

                        if patience <= itr:
                            done_looping = True
                            break

            end_time = time.clock()
            print(
                'optimization complete. validation {}% @ iter {}, '
                'with test performance {}% for class {}'.format(
                best_validation_loss * 100.,
                best_iter + 1,
                test_score * 100.,
                klass
            ))

            sys.stderr.write('fine tuning for {} ran for {:.2f}\n'.format(
                os.path.split(__file__)[1], ((end_time - start_time) / 60.)
            ))

        test_losses = train_pred(vectorize(
            save.train,
            train_set_x,
            train_set_y,
            settings.chunks.test,
            klass_num
        ))

        num_predictions = len(test_losses)
        print(num_predictions)
        prec = np.get_printoptions()['precision']
        np.set_printoptions(suppress=True, precision=4)
        for vec in test_losses[:3]:
            print(vec)

        deal_gen = load.gen_train(
            num_chunks=None, batch_size=num_predictions
        )

        matches = 0
        score = 0
        d = dict()
        for idx, data in enumerate([dg for dg in deal_gen][0]):
            am = numpy.argmax(test_losses[idx])
            if am not in d:
                d[am] = 0
            d[am] += 1
            matches += data[1].to_klass_num(klass_num) == am
            # noinspection PyUnresolvedReferences
            score += ((
                numpy.array(data[1].to_vec(klass_num)) - test_losses[idx]
            ) ** 2).sum()
        print('score: {:.4f}/{} = {:.4f}'.format(
            score, num_predictions, score/num_predictions
        ))

        deal_gen = load.gen_train(
            num_chunks=None, batch_size=num_predictions
        )

        for idx, data in enumerate([dg for dg in deal_gen][0]):
            preds = ['%.10f' % i for i in numpy.nditer(test_losses[idx])]
            print(
                data[0].id,
                data[1].function,
                data[1].to_klass_num(klass_num),
                ' '.join(preds)
            )
            if idx > 10:
                break
        np.set_printoptions(suppress=False, precision=prec)
        print(d)
        print(matches, num_predictions)

    # with open(final_fname, 'wb') as ifile:
    #     pickle.dump(dbn, ifile)
    # return final_fname

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
