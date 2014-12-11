

import csv
import os
import random
import sys
from bpfe.clean import clean_value
from bpfe.config import INPUT_MAPPING, LABEL_MAPPING
from bpfe.entities import Data, Label
from bpfe.feature_engineering import get_vectorizers
from bpfe.reservoir import reservoir
import math

# noinspection PyBroadException
try:
    import cPickle as pickle
except:
    import pickle


def generate_submission_rows(seed=1, amt=None):
    if amt is None:
        amt = sys.maxint
    ret = []
    for data, _ in generate_rows('data/TestData.csv', seed, amt):
        ret.append((data, None))
    return ret


def generate_training_rows(seed=1, amt=None):
    if amt is None:
        amt = sys.maxint
    ret = []
    for data, label in generate_rows('data/TrainingData.csv', seed, amt):
        ret.append((data, label))
    random.seed(seed)
    random.shuffle(ret)
    return ret


def generate_rows(file_path, seed, amt):
    with open(file_path) as infile:
        reader = csv.reader(infile)
        header = next(reader)
        input_index_map = dict()
        label_index_map = dict()
        for idx, col in enumerate(header):
            if col == '':
                continue

            if col in INPUT_MAPPING:
                input_index_map[col] = idx
            else:
                label_index_map[col] = idx

        for line in reservoir(reader, seed, amt):
            d = Data()
            d.id = line[0]
            l = Label()
            for key, idx in input_index_map.iteritems():
                setattr(d, INPUT_MAPPING[key], clean_value(line[idx]))
            for key, idx in label_index_map.iteritems():
                setattr(l, LABEL_MAPPING[key], line[idx])

            yield d, l


def split_test_train(data):
    batch_size = 20000
    validate_data = data[:batch_size]
    data = data[batch_size:]
    test_data = data[:batch_size]
    train_data = data[batch_size:]
    return validate_data, test_data, train_data


def store_raw(seed=1, verbose=False):
    random_data = generate_training_rows(seed)
    validate, test, train = split_test_train(random_data)
    submission = generate_submission_rows(seed)
    chunk_size = 5000
    if verbose:
        print('{}, {}, {}, {}'.format(
            len(train),
            len(validate),
            len(test),
            len(submission)
        ))

    def store_in_chunks(data, name):
        with open('data/raw-{}.pkl'.format(name), 'wb') as datafile:
            chunks = len(data) / float(chunk_size)
            chunks = int(math.ceil(chunks))
            pickle.dump(chunks, datafile, -1)
            # noinspection PyArgumentList
            for i in xrange(0, len(data), chunk_size):
                pickle.dump(data[i: i + chunk_size], datafile, -1)

    store_in_chunks(validate, 'validate')
    store_in_chunks(test, 'test')
    store_in_chunks(submission, 'submission')
    store_in_chunks(train, 'train')


def gen_validate(settings, batch_size=None):
    for data in _gen_name('validate', settings.chunks.validate, batch_size):
        yield data


def gen_test(settings, batch_size=None):
    for data in _gen_name('test', settings.chunks.test, batch_size):
        yield data


def gen_train(settings, batch_size=None):
    for data in _gen_name('train', settings.chunks.train, batch_size):
            yield data


def gen_submission(settings, batch_size=None):
    for data in _gen_name('submission', settings.chunks.submission, batch_size):
        yield data


def _gen_name(name, num_chunks, batch_size=None):
    if batch_size is None:
        batch_size = sys.maxint

    with open('data/raw-{}.pkl'.format(name), 'rb') as datafile:
        chunks = pickle.load(datafile)
        data = []
        for i in range(chunks):
            if i >= num_chunks:
                break

            data += pickle.load(datafile)
            total_size = len(data)

            if batch_size > total_size:
                continue

            batches = int(math.ceil(total_size / float(batch_size)))
            if batches > 1:
                for j in range(batches):
                    sub_data = data[int(j*batch_size): int((j+1)*batch_size)]
                    yield sub_data
            else:
                yield data

            data = []

        if len(data) > 0:
            yield data


def load_vectorizers(settings):
    print('loading vectorizers')
    fname = 'data/vectorizers-{}.pkl'.format(
        settings.chunks
    )
    if os.path.exists(fname):
        with open(fname, 'rb') as ifile:
            v = pickle.load(ifile)
    else:
        print('creating vectorizers for {} chunks'.format(settings.chunks))
        v = get_vectorizers(settings)
        with open(fname, 'wb') as ifile:
            pickle.dump(v, ifile, -1)

    return v
