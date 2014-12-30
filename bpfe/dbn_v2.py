# coding=utf-8
import gzip

import os
from os.path import dirname
import sys
import time
import math
from scipy.sparse import vstack, csr_matrix
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from bpfe import save, cache, scoring
from bpfe.cache import load_stats
from bpfe.config import KLASS_LABEL_INFO, Settings, \
    HiddenLayerSettings, FinetuningSettings, ChunkSettings, \
    REVERSE_LABEL_MAPPING, LABELS, LABEL_MAPPING
from bpfe.dl_dbn.DBN import DBN
import numpy
import bpfe.load as load
import numpy as np
from bpfe.vectorizer.BPFEVectorizer import BPFEVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import theano.tensor as T


# noinspection PyBroadException
try:
    import cPickle as pickle
except:
    import pickle


data_dir = dirname(dirname(__file__)) + '/data'
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
            numpy.asarray([[]], dtype=theano.config.floatX),
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


# noinspection PyDefaultArgument
def get_vects(name, indices=None, percent=1., unique=True):
    def _get_vects_internal(_name, _unique):
        vectorizer = BPFEVectorizer(
            ngrams=2,
            use_stemmer=True,
            use_grades=True,
            use_titles=True,
            filter_stopwords=False,
            binarize=True,
            min_df=None
        )

        if _unique:
            vname = data_dir + '/models/{}-dbn-vects.pkl'.format(_name)
        else:
            vname = data_dir + '/models/final-{}-dbn-vects.pkl'.format(_name)

        # load the vectorized data from disk
        if os.path.exists(vname):
            with open(vname, 'rb') as ifile:
                data = pickle.load(ifile)
                return data

        # create the vectorizer + vectorized data since it doesn't exist
        if _name == 'train':
            g = load.ugen_train()
        elif _name == 'test':
            g = load.ugen_test()
        elif _name == 'validate':
            g = load.ugen_validate()
        elif _name == 'submission':
            g = load.ugen_submission(_unique)
        else:
            raise

        # save the vectorizers to disk
        vectorizer_file = data_dir + '/models/vectorizer-dbn.pkl'
        if not os.path.exists(vectorizer_file):
            with open(vectorizer_file, 'wb') as ifile:
                # load all data
                all_data = [d for d, _ in load.ugen_all()]
                # create the vectorizer using all the data
                vectorizer.fit_transform(all_data)
                # SAVE
                pickle.dump(vectorizer, ifile, -1)
        else:
            with open(vectorizer_file, 'rb') as ifile:
                vectorizer = pickle.load(ifile)

        # save the vectorized data to disk
        with open(vname, 'wb') as ifile:
            # vectorize the data
            data = [vectorizer.transform([d]) for d, _ in g]
            # turn into a big numpy sparse matrix
            data = vstack(data)

            # SAVE
            pickle.dump(data, ifile, -1)

            # RETURN
            return data

    if isinstance(name, basestring):
        data = _get_vects_internal(name, unique)
        num_rows = percent * data.shape[0]
        data = data[:num_rows]
        if indices is not None:
            data = data[indices]

        return data

    elif isinstance(name, list):
        all_data = []
        for n in name:
            data = _get_vects_internal(n, unique)
            num_rows = percent * data.shape[0]
            data = data[:num_rows]
            all_data.append(data)

        data = vstack(all_data)
        if indices is not None:
            data = data[indices]

        return data


def get_labels(name, klass, indices=None, percent=1., unique=True,
               argmax=True):
    def _get_labels_internal(_name, _unique):
        if _unique:
            lname = data_dir + '/models/{}-{}-dbn-labels.pkl'.format(
                _name,
                klass
            )
        else:
            lname = data_dir + '/models/final-{}-{}-dbn-labels.pkl'.format(
                _name,
                klass
            )

        # load the vectorized data from disk
        if os.path.exists(lname):
            with open(lname, 'rb') as ifile:
                data = pickle.load(ifile)
                return data

        # create the vectorizer + vectorized data since it doesn't exist
        if _name == 'train':
            g = load.ugen_train()
        elif _name == 'test':
            g = load.ugen_test()
        elif _name == 'validate':
            g = load.ugen_validate()
        elif _name == 'submission':
            g = load.ugen_submission(_unique)
        else:
            raise

        # save the labels to disk
        with open(lname, 'wb') as ifile:
            ty = [getattr(l, klass) for _, l in g]
            ty = np.array(ty)

            klass_raw = REVERSE_LABEL_MAPPING[klass]
            labels = LABELS[klass_raw]

            actuals = []
            for cidx in range(ty.shape[0]):
                label = ty[cidx]
                tmp = np.zeros(len(labels))
                tmp[labels.index(label)] = 1.0
                label = tmp
                actuals.append(label)
            actuals = np.array(actuals)

            # SAVE
            pickle.dump(actuals, ifile, -1)

            # RETURN
            return actuals

    if isinstance(name, basestring):
        data = _get_labels_internal(name, unique)
        num_rows = percent * data.shape[0]
        data = data[:num_rows]
        if indices is not None:
            data = data[indices]

        if argmax:
            data = np.argmax(data, axis=1)

        return data

    elif isinstance(name, list):
        all_data = []
        for n in name:
            data = _get_labels_internal(n, unique)
            num_rows = percent * data.shape[0]
            data = data[:num_rows]
            all_data.append(data)

        data = vstack(all_data)
        if indices is not None:
            data = data[indices]

        if argmax:
            data = np.argmax(data, axis=1)

        return data


def DBN_tuning(percent):
    """ """

    ptlr = 0.01
    ftlr = 0.1
    settings = Settings()
    settings.version = 10.
    settings.k = 1
    settings.hidden_layers = [
        HiddenLayerSettings(
            1000,
            300,
            ptlr
        ),
        HiddenLayerSettings(
            500,
            300,
            ptlr
        ),
        HiddenLayerSettings(
            250,
            300,
            ptlr
        ),
        HiddenLayerSettings(
            100,
            300,
            ptlr
        )
    ]
    settings.batch_size = 10
    settings.finetuning = FinetuningSettings(
        2,
        ftlr
    )
    settings.chunks = ChunkSettings(1, 1, 1, None)

    start = time.clock()
    _run_with_params(settings, percent)
    print('DURATION: {}'.format(time.clock() - start))

    if not os.path.exists('data/stats/stats.pkl'):
        return

    with open('data/stats/stats.pkl', 'rb') as ifile:
        stats = pickle.load(ifile)
        for row in stats:
            print(' '.join([str(i) for i in row]))


def DBN_run(percent):
    settings = Settings()
    settings.version = 10.0
    settings.k = 1
    settings.hidden_layers = [
        HiddenLayerSettings(1000, 50, 0.01),
        HiddenLayerSettings(1000, 50, 0.01),
        HiddenLayerSettings(1000, 50, 0.01)
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

    start = time.clock()
    _run_with_params(settings, percent)
    print('DURATION: {}'.format(time.clock() - start))


def _run_with_params(settings, percent):
    dbn, datasets = pretrain(settings, percent)
    finetune(dbn, datasets, settings, percent)


def pretrain(settings, percent):
    datasets = create_datasets()
    train_set_x, train_set_y = datasets[0]

    train_vectors = get_vects('train', percent=percent)
    settings.train_size = train_vectors.shape[0]
    settings.num_cols = train_vectors.shape[1]
    del train_vectors

    settings_stats = load_stats()

    already_ran = False
    a_class = KLASS_LABEL_INFO.keys()[0]
    if a_class in settings_stats:
        for data in settings_stats[a_class]:
            if data[0] == settings:
                already_ran = True

    if already_ran:
        return

    print('num inputs: {}'.format(settings.num_cols))
    print('train size: {}'.format(settings.train_size))

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

    #########################
    # PRETRAINING THE MODEL #
    #########################

    print('... pre-training the model')

    start_time = time.clock()
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
        pretraining_fn, free_energy = dbn.pretraining_function(
            train_set_x,
            settings.batch_size,
            settings.k,
            layer_idx,
            settings.numpy_rng,
            settings.theano_rng,
            settings.train_size,
            resume_epoch is not None
        )

        # go through pretraining epochs
        for epoch in xrange(settings.hidden_layers[layer_idx].epochs):
            if resume_epoch is not None and resume_epoch >= epoch:
                continue

            tstart = time.clock()
            # go through the training set
            c = []

            training_dataset_names = ['train', 'test', 'submission']

            all_data = get_vects(name=training_dataset_names, percent=percent)
            shuffler = np.arange(all_data.shape[0])
            # noinspection PyUnresolvedReferences
            np.random.shuffle(shuffler)

            all_data = all_data[shuffler]

            to_gpu_size = 5000
            to_gpu_batches = int(
                math.ceil(all_data.shape[0] / float(to_gpu_size))
            )
            for to_gpu_batch in range(to_gpu_batches):
                start = (to_gpu_batch * to_gpu_size)
                end = min(start + to_gpu_size, all_data.shape[0])
                # noinspection PyUnresolvedReferences
                subset = csr_matrix(
                    all_data[start:end],
                    dtype=theano.config.floatX
                )
                subset = subset.todense()
                train_set_x.set_value(subset, borrow=True)

                n_train_batches = int(
                    math.ceil(subset.shape[0] / float(settings.batch_size))
                )
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fn(
                        index=batch_index,
                        lr=settings.hidden_layers[layer_idx].learning_rate
                    ))
                del subset

            # find the free energy for the same subset of the training data
            # and also for a validation subset (use for overfitting)
            # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf#6.1
            all_data = get_vects(name=training_dataset_names, percent=percent)
            # noinspection PyUnresolvedReferences
            all_data = csr_matrix(
                all_data[:min(2500, all_data.shape[0])],
                dtype=theano.config.floatX
            )
            train_set_x.set_value(all_data.todense(), borrow=True)
            fe_all = free_energy()
            del all_data

            validate_data = get_vects(
                name='validate',
                percent=percent
            )
            # noinspection PyUnresolvedReferences
            validate_data = csr_matrix(
                validate_data[:min(2500, validate_data.shape[0])],
                dtype=theano.config.floatX
            )
            train_set_x.set_value(validate_data.todense(), borrow=True)
            fe_validate = free_energy()
            del validate_data

            # noinspection PyUnresolvedReferences
            print(
                'Pre-training layer {}, epoch {}, cost {}, elapsed {}\n '
                '\tfree energy mean: train {:.6f}, validate {:.6f}, '
                'gap {:.4f}'.format(
                    layer_idx,
                    epoch,
                    numpy.mean(c),
                    _td(time.clock() - tstart),
                    np.mean(fe_all),
                    np.mean(fe_validate),
                    np.absolute(np.mean(fe_all) - np.mean(fe_validate))
                )
            )

            errname = data_dir + '/{}-{}-pretrain-errors.pkl.gz'.format(
                layer_idx, percent
            )
            errors = []
            if os.path.exists(errname):
                with gzip.open(errname, 'rb') as ifile:
                    errors = pickle.load(ifile)

            errors.append([epoch, layer_idx, np.mean(c), 'cost {}'.format(
                layer_idx
            )])
            errors.append([epoch, layer_idx, np.mean(fe_all), 'all'])
            errors.append([epoch, layer_idx, np.mean(fe_validate), 'validate'])
            errors.append([epoch, layer_idx, time.clock() - tstart, 'time'])

            with gzip.open(errname, 'wb') as ifile:
                pickle.dump(errors, ifile, -1)

            if (epoch % 5 == 0 and epoch != 0) or \
               epoch == settings.hidden_layers[layer_idx].epochs - 1:
                cache.save_pretrain_layer(dbn, layer_idx, settings, epoch)

    end_time = time.clock()
    # end-snippet-2
    sys.stderr.write(
        'The pretraining code for file {} ran for {:.2f}m\n'.format(
            os.path.split(__file__)[1],
            ((end_time - start_time) / 60.)
        )
    )

    return dbn, datasets


def finetune(dbn, datasets, settings, percent):
    # TODO since we're running a single class....
    with open('data/current-dbn.pkl', 'wb') as ifile:
        pickle.dump(dbn, ifile)

    for klass, (klass_num, count) in KLASS_LABEL_INFO.items():
        # if klass in {'Function'}:
        #     continue

        dbn.number_of_outputs = count
        finetune_class(
            dbn, datasets, settings, klass, klass_num, count, percent)

        # try_predict_test(datasets, settings, klass, klass_num)

        # TODO since we're running a single class....
        with open('data/current-dbn.pkl', 'rb') as ifile:
            dbn = pickle.load(ifile)


def finetune_class(dbn, datasets, settings, klass, klass_num, count, percent):
    scores = []
    (train_set_x, train_set_y) = datasets[0]

    epoch = 0
    val = cache.load_finetuning(settings, klass_num)
    if val is not None:
        dbn, settings, epoch = val
        train_fn, train_pred, valid_pred, test_pred, submission_pred = \
            dbn.build_finetune_functions(
                datasets=datasets,
                batch_size=settings.batch_size,
                learning_rate=settings.finetuning.learning_rate
            )
    else:
        dbn.create_output_layer(settings.train_size, settings.numpy_rng)

        # get the training, validation and testing function for the model
        print('... getting the finetuning functions')
        train_fn, train_pred, valid_pred, test_pred, submission_pred = \
            dbn.build_finetune_functions(
                datasets=datasets,
                batch_size=settings.batch_size,
                learning_rate=settings.finetuning.learning_rate
            )

    def get_mmll(name):
        _data = get_vects(name, percent=percent)
        _labels = get_labels(
            name,
            LABEL_MAPPING[klass],
            percent=percent,
            argmax=False
        )
        if name == 'train':
            preds = train_pred(_data)
        elif name == 'validate':
            preds = valid_pred(_data)
        elif name == 'test':
            preds = test_pred(_data)
        else:
            raise
        mmll = scoring.multi_multi_log_loss(
            preds,
            _labels,
            np.array([range(_labels.shape[1])])
        )
        return mmll

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
    while (epoch < settings.finetuning.epochs) and (not done_looping):
        epoch += 1
        start_epoch = time.clock()

        train_data = get_vects('train', percent=percent)
        train_labels = get_labels(
            'train',
            LABEL_MAPPING[klass],
            percent=percent,
            argmax=False
        )
        shuffler = np.arange(train_data.shape[0])
        # noinspection PyUnresolvedReferences
        np.random.shuffle(shuffler)

        train_data = train_data[shuffler]
        train_labels = train_labels[shuffler]

        to_gpu_size = 5000
        to_gpu_batches = int(
            math.ceil(train_data.shape[0] / float(to_gpu_size))
        )
        for to_gpu_batch in range(to_gpu_batches):
            start = (to_gpu_batch * to_gpu_size)
            end = min(start + to_gpu_size, train_data.shape[0])
            # noinspection PyUnresolvedReferences
            subset_data = csr_matrix(
                train_data[start:end],
                dtype=theano.config.floatX
            )
            subset_data = subset_data.todense()
            # noinspection PyUnresolvedReferences
            subset_labels = np.array(
                train_labels[start:end],
                dtype=theano.config.floatX
            )
            train_set_x.set_value(subset_data, borrow=True)
            train_set_y.set_value(subset_labels, borrow=True)

            n_train_batches = int(
                math.ceil(subset_data.shape[0] / float(settings.batch_size))
            )
            for batch_index in xrange(n_train_batches):
                train_fn(batch_index)

        val_mmll = get_mmll('validate')

        if epoch % 5 == 0:
            cache.save_finetuning(dbn, settings, klass_num, epoch)

        # if we got the best validation score until now
        if val_mmll < settings.finetuning[klass_num].best_validation_loss:
            # save best validation score and iteration number
            settings.finetuning[klass_num].best_validation_loss = val_mmll
            settings.finetuning[klass_num].best_iteration = epoch

        train_mmll = get_mmll('train')
        scores.append([epoch, train_mmll, 'train'])
        scores.append([epoch, val_mmll, 'validate'])

        print(
            'epoch {}: train mmll {:.4f}, validate mmll {:.4f} after {}'
            .format(
                epoch,
                train_mmll,
                val_mmll,
                _td(time.clock() - start_epoch)
            )
        )

    scorename = data_dir + '/ft-{}-{:.2f}.pkl.gz'.format(
        LABEL_MAPPING[klass], percent
    )
    with gzip.open(scorename, 'wb') as ifile:
        pickle.dump(scores, ifile, -1)
    df = pd.DataFrame(scores, columns=['epoch', 'error', 'dataset'])
    sns.pointplot('epoch', 'error', data=df, hue='dataset')
    plt.title('error for hl epochs {}, ft epochs {}, {:.1f}%'.format(
        settings.hidden_layers[0].epochs,
        settings.finetuning.epochs,
        percent * 100.
    ))
    plt.xlabel('error for {}'.format(klass))
    # plt.show()

    print(
        '{} optimization complete.\n'
        '\tvalidation mmll {:.4f} @ iter {}\n'
        '\ttest mmll {:.4f}'
        .format(
            klass,
            settings.finetuning[klass_num].best_validation_loss,
            settings.finetuning[klass_num].best_iteration,
            get_mmll('test')
        )
    )

    sys.stderr.write('fine tuning for {} ran for {}\n'.format(
        klass,
        _td(time.clock() - start_time)
    ))

    # if os.path.exists('data/stats/stats.pkl'):
    #     with open('data/stats/stats.pkl', 'rb') as ifile:
    #         stats = pickle.load(ifile)
    # else:
    #     stats = []
    #
    # with open('data/stats/stats.pkl', 'wb') as ifile:
    #     stats.append(settings.stats_info())
    #     pickle.dump(stats, ifile)


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


def _td(value):
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)

    return '%02d:%02d:%02d' % (hours, minutes, seconds)

if __name__ == '__main__':
    DBN_tuning(.05)
    # DBN_run()
    # stats()

