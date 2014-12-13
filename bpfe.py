import gzip
from itertools import izip
import os
from os.path import dirname
import sys
import time
import math
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from bpfe import scoring, save, cache
from bpfe.cache import load_stats
from bpfe.config import FLAT_LABELS, KLASS_LABEL_INFO, Settings, \
    HiddenLayerSettings, FinetuningSettings, ChunkSettings, LABELS
from bpfe.dl_dbn.DBN import DBN
import numpy
from bpfe.features import all_text_rows
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


def create_datasets():
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
    return datasets


def vectorize(generator, X, Y, settings, klass_num):
    batch_size = 5000
    data_len, index = 0, 0
    full_data, full_labels = None, None
    for data, labels in generator(settings):
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
    all_text_rows.info()

    # text_2.info()
    # text_3.info()
    # text_4.info()
    # fte.info()
    # total.info()

    # load.store_raw()

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
    settings.version = 4.0
    settings.k = 1
    settings.hidden_layers = [
        HiddenLayerSettings(500, 50, 0.01),
        HiddenLayerSettings(500, 50, 0.01),
        HiddenLayerSettings(500, 50, 0.01)
    ]
    settings.batch_size = 5
    settings.finetuning = FinetuningSettings(100, 0.1)
    settings.chunks = ChunkSettings(9, 4, 4, 11)

    train_len = 0
    for data, _ in save.train(settings):
        train_len += len(data)

    validate_len = 0
    for data, _ in save.validate(settings):
        validate_len += len(data)

    test_len = 0
    for data, _ in save.test(settings):
        test_len += len(data)

    submission_len = 0
    for data, _ in save.submission(settings):
        submission_len += len(data)

    print('train size: {}'.format(train_len))
    print('validate size: {}'.format(validate_len))
    print('test size: {}'.format(test_len))
    print('submission size: {}'.format(submission_len))

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

    create_submission_file(settings)

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

    datasets = create_datasets()
    train_set_x, train_set_y = datasets[0]

    settings.train_size = 0
    settings.num_cols = 0
    for _ in vectorize(save.train, train_set_x, train_set_y, settings, 0):
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
        resume_epoch = None
        if cache_pretrain is not None:
            print('layer {} already exists on disk, loading ...'.format(
                layer_idx
            ))
            dbn, settings, resume_epoch = cache_pretrain
            if resume_epoch >= settings.hidden_layers[layer_idx].epochs - 1:
                continue
            print('resuming layer {} at epoch {} ...'.format(
                layer_idx, resume_epoch + 1
            ))

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
            if resume_epoch is not None and resume_epoch >= epoch:
                continue

            start = time.clock()
            # go through the training set
            c = []

            for row_len in vectorize(
                    save.full_train,
                    train_set_x,
                    train_set_y,
                    settings,
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

            cache.save_pretrain_layer(dbn, layer_idx, settings, epoch)

    # print('ts', settings.train_size)

    end_time = time.clock()
    # end-snippet-2
    sys.stderr.write(
        'The pretraining code for file {} ran for {:.2f}m\n'.format(
            os.path.split(__file__)[1],
            ((end_time - start_time) / 60.)
        )
    )

    ########################
    # FINETUNING THE MODEL #
    ########################
    finetune(dbn, datasets, settings)


def finetune(dbn, datasets, settings):
    # TODO since we're running a single class....
    with open('data/current-dbn.pkl', 'wb') as ifile:
        pickle.dump(dbn, ifile)

    for klass, (klass_num, count) in KLASS_LABEL_INFO.items():
        if klass in {'Function', 'Use', 'Reporting'}:
            continue

        dbn.number_of_outputs = count
        finetune_class(dbn, datasets, settings, klass, klass_num, count)

        try_predict_test(datasets, settings, klass, klass_num)

        # TODO since we're running a single class....
        with open('data/current-dbn.pkl', 'rb') as ifile:
            dbn = pickle.load(ifile)


def finetune_class(dbn, datasets, settings, klass, klass_num, count):
    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]
    (test_set_x, test_set_y) = datasets[2]

    epoch = 0
    val = cache.load_finetuning(settings, klass_num)
    # val = None
    if val is not None:
        dbn, settings, epoch = val
        train_fn, train_model, validate_model, test_model, \
            train_pred, test_pred, submission_pred = \
            dbn.build_finetune_functions(
                datasets=datasets,
                batch_size=settings.batch_size,
                learning_rate=settings.finetuning.learning_rate
            )
    else:
        dbn.create_output_layer()

        # get the training, validation and testing function for the model
        print('... getting the finetuning functions')
        train_fn, train_model, validate_model, test_model, \
            train_pred, test_pred, submission_pred = \
            dbn.build_finetune_functions(
                datasets=datasets,
                batch_size=settings.batch_size,
                learning_rate=settings.finetuning.learning_rate
            )

    if epoch >= settings.finetuning.epochs:
        print('... finetuning for "{}" is already complete'.format(klass))
        return

    print('... finetuning for "{}" ({} classes), starting at epoch {}'.format(
        klass,
        count,
        epoch + 1
    ))

    settings.finetuning[klass_num].patience = \
        4. * (settings.train_size / settings.batch_size)

    start_time = time.clock()
    done_looping = False
    itr = epoch
    while (epoch < settings.finetuning.epochs) and (not done_looping):
        epoch += 1
        idx = 0
        start_epoch = time.clock()
        while idx < settings.train_batches:
            print('idx: {} < train_batches: {}'.format(
                idx,
                settings.train_batches
            ))

            for row_len in vectorize(
                save.train,
                train_set_x,
                train_set_y,
                settings,
                klass_num
            ):
                sub_train_batches = row_len / settings.batch_size
                # print('{} {} {}'.format(
                #     sub_train_batches,
                #     row_len,
                #     settings.batch_size
                # ))
                # print(settings.validation_frequency)

                for minibatch_index in xrange(sub_train_batches):
                    train_fn(minibatch_index)
                    itr = (epoch - 1) * settings.train_batches + idx
                    idx += 1

                if (itr + 1) % settings.validation_frequency == 0:

                    validation_losses = validate_model(vectorize(
                        save.validate,
                        valid_set_x,
                        valid_set_y,
                        settings,
                        klass_num
                    ))
                    val_loss = numpy.mean(validation_losses)

                    print('epoch {}, validation error {:.4f}%'.format(
                        epoch,
                        val_loss * 100.
                    ))

                    # if we got the best validation score until now
                    if val_loss < settings.finetuning[klass_num]\
                        .best_validation_loss:

                        # improve patience if loss improvement is
                        # good enough
                        if val_loss < settings.finetuning[klass_num]\
                            .minimum_improvement:
                            # print('updating patience', itr,
                            #       settings.finetuning[klass_num]
                            # .patience_increase)
                            settings.finetuning[klass_num].patience = itr * \
                                settings.finetuning[klass_num].patience_increase
                            # print('patience', settings.finetuning[klass_num]
                            # .patience)

                        # save best validation score and iteration
                        # number
                        settings.finetuning[klass_num].best_validation_loss = \
                            val_loss
                        settings.finetuning[klass_num].best_iteration = itr

                        cache.save_finetuning(dbn, settings, klass_num, epoch)

                        # test it on the test set
                        test_losses = test_model(vectorize(
                            save.test,
                            test_set_x,
                            test_set_y,
                            settings,
                            klass_num
                        ))
                        test_loss = numpy.mean(test_losses)

                        if test_loss < settings.finetuning[klass_num]\
                            .best_test_loss:
                            print('... saving current best model for {}'.format(
                                klass
                            ))
                            cache.save_best_finetuning(dbn, settings, klass_num)
                            settings.finetuning[klass_num].best_test_loss = \
                                test_loss
                            settings.finetuning[klass_num].patience = itr * \
                                settings.finetuning[klass_num].patience_increase

                        if False:
                            train_losses = train_model(vectorize(
                                save.train,
                                train_set_x,
                                train_set_y,
                                settings,
                                klass_num
                            ))
                            train_loss = numpy.mean(train_losses)
                            print('  epoch {}, test error {:.4f}% {:.4f}%'
                                  .format(
                                      epoch,
                                      test_loss * 100.,
                                      train_loss * 100.
                                  ))
                        else:
                            print('  epoch {}, test error {:.4f}%'.format(
                                epoch,
                                test_loss * 100.
                            ))

                if settings.finetuning[klass_num].patience <= itr:
                    print('done', settings.finetuning[klass_num].patience, itr)
                    done_looping = True
                    break

        print('epoch {}: {}'.format(epoch, (time.clock() - start_epoch)))

    end_time = time.clock()

    print('optimization complete. validation {}% @ iter {}, '
          'with test performance {}% for class {}'.format(
              settings.finetuning[klass_num].best_validation_loss * 100.,
              settings.finetuning[klass_num].best_iteration,
              settings.finetuning[klass_num].best_test_loss * 100.,
              klass
          ))

    sys.stderr.write('fine tuning for {} ran for {:.2f}\n'.format(
        os.path.split(__file__)[1],
        ((end_time - start_time) / 60.)
    ))


def try_predict_test(datasets, settings, klass, klass_num):
    (test_set_x, test_set_y) = datasets[2]

    print('... loading best model for class "{}"'.format(klass))
    val = cache.load_best_finetuning(settings, klass_num)
    if val is None:
        print('no "best" model for class "{}"'.format(klass))
        return

    dbn, settings = val
    train_fn, train_model, validate_model, test_model, \
        train_pred, test_pred, submission_pred = \
        dbn.build_finetune_functions(
            datasets=datasets,
            batch_size=settings.batch_size,
            learning_rate=settings.finetuning.learning_rate
        )

    test_predictions = test_pred(vectorize(
        save.test,
        test_set_x,
        test_set_y,
        settings,
        klass_num
    ))

    num_predictions = len(test_predictions)
    print(num_predictions)

    deal_gen = load.gen_test(settings, batch_size=num_predictions)

    total_predictions, matches, misses, score, idx = 0, 0, 0, 0, 0
    red = reduce(
        lambda x, y: x + y,
        [dg for dg in deal_gen],
        []
    )
    for data in red:
        am = numpy.argmax(test_predictions[idx])
        actual_num = data[1].to_klass_num(klass_num)
        miss = actual_num != am
        matches += not miss
        misses += miss
        total_predictions += 1

        # noinspection PyUnresolvedReferences
        score += ((
            numpy.array(data[1].to_vec(klass_num)) - test_predictions[idx]
        ) ** 2).sum()

        # print first 3 misses
        if misses <= 3 and miss:
            print('id {}: {}'.format(data[0].id, data[1].function))
            print('  expected {} -> {}'.format(
                data[1].to_klass_num(klass_num),
                am
            ))
            preds = '  '
            len_preds = len(test_predictions[idx])
            per_row = 8
            for col in range(int(math.ceil(len_preds / float(per_row)))):
                cidx = col * per_row
                preds += '{:>2}: '.format(cidx)
                for j in range(per_row):
                    if j + cidx >= len_preds:
                        break
                    preds += '{:.8f} '.format(test_predictions[idx][j + cidx])
                preds += '\n  '
            print(preds)

        idx += 1
    score /= 2.0
    print('sum of squares error: {:.4f}'.format(score/total_predictions))
    print('matches {}/{} = {:.4f}% error rate'.format(
        matches,
        total_predictions,
        (1 - matches / float(total_predictions)) * 100.0
    ))

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


def create_submission_file(settings):
    datasets = create_datasets()

    for klass, (klass_num, count) in KLASS_LABEL_INFO.items():
        predict_submission(datasets, settings, klass, klass_num)

    header = ['__'.join(i) for i in FLAT_LABELS]
    headers = []
    for i in header:
        if ' ' in i:
            i = '"{}"'.format(i)
        headers.append(i)

    header_line = ',' + ','.join(headers)

    class_files = []

    ordered_classes = sorted(LABELS.keys())
    for klass in ordered_classes:
        klass_num = KLASS_LABEL_INFO[klass][0]
        class_files.append(
            gzip.open('data/submission/{}-{}.csv.gz'.format(
                settings.version,
                int(klass_num)
            ), 'r')
        )

    fname = 'data/submission/{}-final.csv.gz'.format(settings.version)
    with gzip.open(fname, 'w') as ifile:
        ifile.write(header_line + '\n')
        for all_row in izip(*class_files):
            current_id = None
            full_row = []
            for filerow in all_row:
                filerow = filerow.strip()
                values = filerow.split(',')
                if current_id is None:
                    current_id = values[0]
                else:
                    assert current_id == values[0]
                full_row += values[1:]
            row_string = '{},{}'.format(
                current_id,
                ','.join(full_row)
            )

            ifile.write(row_string + '\n')


def predict_submission(datasets, settings, klass, klass_num):
    (submission_set_x, submission_set_y) = datasets[3]

    print('... loading best model for class "{}"'.format(klass))
    val = cache.load_best_finetuning(settings, klass_num)
    if val is None:
        print('no "best" model for class "{}"'.format(klass))
        return

    dbn, settings = val
    train_fn, train_model, validate_model, test_model, \
        train_pred, test_pred, submission_pred = \
        dbn.build_finetune_functions(
            datasets=datasets,
            batch_size=settings.batch_size,
            learning_rate=settings.finetuning.learning_rate
        )

    predictions = submission_pred(vectorize(
        save.submission,
        submission_set_x,
        submission_set_y,
        settings,
        klass_num
    ))

    num_predictions = len(predictions)
    print(num_predictions)

    deal_gen = load.gen_submission(settings, batch_size=num_predictions)

    total_predictions, matches, misses, score, idx = 0, 0, 0, 0, 0
    red = reduce(
        lambda x, y: x + y,
        [dg for dg in deal_gen],
        []
    )
    print('submission rows: {}'.format(len(red)))
    fname = 'data/submission/{}-{}.csv.gz'.format(settings.version, klass_num)
    with gzip.open(fname, 'w') as ifile:
        for data in red:
            row = '{},{}'.format(
                data[0].id,
                ','.join(
                    ['{:.12f}'.format(sub) for sub in predictions[idx]]
                )
            )
            ifile.write(row + '\n')
            idx += 1
        print('final idx: {}'.format(idx))

if __name__ == '__main__':
    # predict()
    # predict_train()
    test_DBN()
    run()
    # stats()
