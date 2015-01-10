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
    REVERSE_LABEL_MAPPING, LABELS, LABEL_MAPPING, FLAT_LABELS
from bpfe.dl_dbn.DBN import DBN
import bpfe.load as load
import numpy as np
from bpfe.vectorizer.BPFEVectorizer import BPFEVectorizer
from bpfe.dl_dbn.constants import DTYPES
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


def create_datasets():
    def shared_dataset():
        shared_x = theano.shared(
            np.asarray([[]], dtype=DTYPES.FLOATX),
            borrow=True
        )
        shared_y = theano.shared(
            np.asarray([[]], dtype=DTYPES.FLOATX),
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
                pickle.dump(vectorizer, ifile, protocol=pickle.HIGHEST_PROTOCOL)
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
            pickle.dump(data, ifile, protocol=pickle.HIGHEST_PROTOCOL)

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
            pickle.dump(actuals, ifile, protocol=pickle.HIGHEST_PROTOCOL)

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
    settings.version = 14.
    settings.k = 1
    settings.hidden_layers = [
        HiddenLayerSettings(
            1250,
            8,
            ptlr
        ),
        HiddenLayerSettings(
            1250,
            10,
            ptlr
        ),
        HiddenLayerSettings(
            1250,
            10,
            ptlr
        ),
        # HiddenLayerSettings(
        #     2000,
        #     8,
        #     ptlr
        # )
    ]
    settings.batch_size = 10
    settings.finetuning = FinetuningSettings(
        13,
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
    for klass, (klass_num, count) in KLASS_LABEL_INFO.items():
        dbn, settings = pretrain(settings, percent)
        dbn, settings = finetune(
            dbn, settings, percent, klass, klass_num, count)
        mmll('validate', dbn, settings, klass, percent, verbose=True)
        mmll('test', dbn, settings, klass, percent, verbose=True)

    probas = predict_submission(percent, settings)
    print('predicted submission')
    print(probas.shape)
    print(probas[:10])
    save_submission(probas, settings, percent)


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
    settings.numpy_rng = np.random.RandomState(123)
    # theano random generator
    settings.theano_rng = RandomStreams(
        settings.numpy_rng.randint(2 ** 30)
    )

    print('... building the model')

    # construct the Deep Belief Network
    dbn = DBN(
        name='{}-{}'.format(settings.version, percent),
        n_ins=settings.num_cols,
        hidden_layers_sizes=[hl.num_nodes for hl in settings.hidden_layers],
        n_outs=104
    )

    extra = '_'.join([
        '{}-{}'.format(
            hl.num_nodes,
            hl.epochs
        ) for hl in settings.hidden_layers
    ])

    a, b = cache.load_full(percent, settings.version, extra)
    if a is not None:
        b.finetuning.epochs = settings.finetuning.epochs
        return a, b

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
            settings.numpy_rng.shuffle(shuffler)

            all_data = all_data[shuffler]

            to_gpu_size = 5000
            to_gpu_batches = int(
                math.ceil(all_data.shape[0] / float(to_gpu_size))
            )
            for to_gpu_batch in range(to_gpu_batches):
                start = (to_gpu_batch * to_gpu_size)
                end = min(start + to_gpu_size, all_data.shape[0])
                subset = csr_matrix(
                    all_data[start:end],
                    dtype=DTYPES.FLOATX
                )
                subset = subset.todense()
                train_set_x.set_value(subset, borrow=True)

                n_train_batches = int(
                    math.ceil(subset.shape[0] / float(settings.batch_size))
                )
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fn(
                        index=batch_index
                    ))
                del subset

            # find the free energy for the same subset of the training data
            # and also for a validation subset (use for overfitting)
            # https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf#6.1
            all_data = get_vects(name=training_dataset_names, percent=percent)
            all_data = csr_matrix(
                all_data[:min(2500, all_data.shape[0])],
                dtype=DTYPES.FLOATX
            )
            train_set_x.set_value(all_data.todense(), borrow=True)
            fe_all = free_energy()
            del all_data

            validate_data = get_vects(
                name='validate',
                percent=percent
            )
            validate_data = csr_matrix(
                validate_data[:min(2500, validate_data.shape[0])],
                dtype=DTYPES.FLOATX
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
                    np.mean(c),
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
                pickle.dump(errors, ifile, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = time.clock()
    sys.stderr.write(
        'The pretraining code for file {} ran for {:.2f}m\n'.format(
            os.path.split(__file__)[1],
            ((end_time - start_time) / 60.)
        )
    )

    cache.save_full(dbn, settings, percent, settings.version, extra)

    return dbn, settings


def finetune(dbn, settings, percent, klass, klass_num, count):
    dbn.number_of_outputs = count

    extra = 'class-{}'.format(
        '-'.join(klass.lower().split())
    )
    a = cache.load_dbn(percent, settings.version, extra)
    if a is not None:
        b = cache.load_settings(percent, settings.version, extra)
        print('loading {} from disk'.format(klass))
        dbn, settings = a, b
    else:
        dbn, settings = finetune_class(
            dbn, settings, klass, klass_num, count, percent)

    return dbn, settings


def finetune_class(dbn, settings, klass, klass_num, count, percent):
    scores = []

    epoch = 0
    val = cache.load_finetuning(settings, klass_num)
    if val[0] is not None:
        dbn, settings, epoch = val
    else:
        dbn.create_output_layer(settings.train_size)

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
        settings.numpy_rng.shuffle(shuffler)

        train_data = train_data[shuffler]
        train_labels = train_labels[shuffler]

        dbn.finetune(
            train_data,
            train_labels,
            settings.batch_size,
            settings.finetuning.learning_rate,
            # momentum
            0.9
        )

        val_mmll = mmll('validate', dbn, settings, klass, percent)

        # if we got the best validation score until now
        if val_mmll < settings.finetuning[klass_num].best_validation_loss:
            # save best validation score and iteration number
            settings.finetuning[klass_num].best_validation_loss = val_mmll
            settings.finetuning[klass_num].best_iteration = epoch

            import gc
            gc.collect()
            extra = 'class-{}'.format(
                '-'.join(klass.lower().split())
            )
            cache.save_dbn(dbn, percent, settings.version, extra)
            cache.save_settings(settings, percent, settings.version, extra)

        train_mmll = mmll('train', dbn, settings, klass, percent)

        scorename = data_dir + '/ft-{}-{:.2f}.pkl.gz'.format(
            LABEL_MAPPING[klass], percent
        )
        scores = []
        if os.path.exists(scorename):
            with gzip.open(scorename, 'rb') as ifile:
                scores = pickle.load(ifile)
        scores.append([epoch, train_mmll, 'train'])
        scores.append([epoch, val_mmll, 'validate'])
        with gzip.open(scorename, 'wb') as ifile:
            pickle.dump(scores, ifile, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            'epoch {}: train mmll {:.4f}, validate mmll {:.4f} after {}'
            .format(
                epoch,
                train_mmll,
                val_mmll,
                _td(time.clock() - start_epoch)
            )
        )

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
            mmll('test', dbn, settings, klass, percent)
        )
    )

    sys.stderr.write('fine tuning for {} ran for {}\n'.format(
        klass,
        _td(time.clock() - start_time)
    ))

    return dbn, settings

    # if os.path.exists('data/stats/stats.pkl'):
    #     with open('data/stats/stats.pkl', 'rb') as ifile:
    #         stats = pickle.load(ifile)
    # else:
    #     stats = []
    #
    # with open('data/stats/stats.pkl', 'wb') as ifile:
    #     stats.append(settings.stats_info())
    #     pickle.dump(stats, ifile)


def predict_probas(dbn, settings, name, percent):
    unique = True
    if name == 'submission':
        unique = False
    _data = get_vects(name, percent=percent, unique=unique)
    preds = dbn.predict_proba(_data, settings.batch_size)
    return preds


def mmll(name, dbn, settings, klass, percent, verbose=False):
    _labels = get_labels(
        name,
        LABEL_MAPPING[klass],
        percent=percent,
        argmax=False
    )
    preds = predict_probas(dbn, settings, name, percent)
    mmll = scoring.multi_multi_log_loss(
        preds,
        _labels,
        np.array([range(_labels.shape[1])])
    )
    if verbose:
        sys.stderr.write('mmll "{}" {}: {:.4f}\n'.format(
            klass,
            name,
            mmll
        ))
    return mmll


def save_submission(data, settings, percent):
    print('saving submission file')

    header = ['__'.join(i) for i in FLAT_LABELS]
    headers = []
    for i in header:
        if ' ' in i:
            i = '"{}"'.format(i)
        headers.append(i)

    header_line = ',' + ','.join(headers)

    ids = [dg.id for dg, _ in load.ugen_submission(False)]
    print('submission rows: {}'.format(len(ids)))
    fname = 'data/submission/{}-{}.csv.gz'.format(settings.version, percent)
    with gzip.open(fname, 'w') as ifile:
        ifile.write(header_line + '\n')
        for idx, cid in enumerate(ids):
            row = '{},{}'.format(
                cid,
                ','.join(
                    ['{:.12f}'.format(sub) for sub in data[idx]]
                )
            )
            ifile.write(row + '\n')


def predict_submission(percent, settings):
    print('getting submission predictions')
    probas = None
    for klass in sorted(LABELS.keys()):
        print('\tgetting predictions for {}'.format(klass))
        extra = 'class-{}'.format(
            '-'.join(klass.lower().split())
        )
        dbn = cache.load_dbn(percent, settings.version, extra)
        if dbn is not None:
            settings = cache.load_settings(percent, settings.version, extra)
        else:
            raise Exception('missing klass {}'.format(klass))

        klass_probas = predict_probas(dbn, settings, 'submission', 1.)
        if probas is None:
            probas = klass_probas
        else:
            probas = np.concatenate((probas, klass_probas), axis=1)
    return probas


def _td(value):
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)

    return '%02d:%02d:%02d' % (hours, minutes, seconds)

if __name__ == '__main__':
    DBN_tuning(.9)
    # DBN_run()
    # stats()

