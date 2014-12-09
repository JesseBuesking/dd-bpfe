

import gzip
import os
from bpfe import feature_engineering, load
import numpy as np
from bpfe.config import KLASS_LABEL_INFO


# noinspection PyBroadException
try:
    import cPickle as pickle
except:
    import pickle


TRAIN_CHUNKS = 3


def to_np_array(vectzers, data):
    attributes = [
        'object_description', 'program_description',
        'subfund_description', 'job_title_description',
        'facility_or_department', 'sub_object_description',
        'location_description', 'fte', 'function_description', 'position_extra',
        'text_4', 'total', 'text_2', 'text_3', 'fund_description', 'text_1'
    ]

    vecs = []
    for d, _ in data:
        avecs = []
        for attr in attributes:
            v, settings, method_name = vectzers[attr]
            m = getattr(feature_engineering, method_name)
            avec = m(v, getattr(d, attr), settings)
            avecs.append(avec)

        vecs.append(np.concatenate(avecs, axis=1)[0])

    try:
        vecs = np.array(vecs, dtype=np.int32)
    except:
        if isinstance(vecs, list):
            print(len(vecs))
        else:
            print(vecs.shape)
        raise

    labels = []
    for _, label in data:
        if label is None:
            break

        l = []
        for klass_num in range(len(KLASS_LABEL_INFO)):
            l.append(label.to_klass_num(klass_num))
        labels.append(l)

    if len(labels) > 0:
        # noinspection PyUnresolvedReferences
        labels = np.array(labels)

    return vecs, labels


def vectorize(generator, num_chunks, batch_size):
    vectorizers = load.load_vectorizers(TRAIN_CHUNKS)
    data_len, index = 0, 0
    full_data, full_labels = None, None
    load_size = 1000
    assert load_size <= batch_size

    for data in generator(num_chunks, load_size):
        v, l = to_np_array(vectorizers, data)
        if full_data is None:
            full_data = np.ndarray(
                shape=(batch_size, v.shape[1]),
                dtype=np.int32
            )
            full_labels = np.ndarray(
                shape=(batch_size, len(KLASS_LABEL_INFO)),
                dtype=np.int32
            )
        data_len += v.shape[0]

        start = index * v.shape[0]
        end = start + v.shape[0]
        full_data[start:end, :] = v
        if l is not None:
            full_labels[start:end] = l

        done = False
        if v.shape[0] < load_size:
            full_data = full_data[:data_len, :]
            if full_labels is not None:
                full_labels = full_labels[:data_len]
            done = True

        if not done and data_len < batch_size:
            continue

        yield full_data, full_labels

        full_data, full_labels = None, None
        data_len, index = 0, 0


def train(num_chunks):
    return _load_vectors('train', num_chunks)


def validate(num_chunks):
    return _load_vectors('validate', num_chunks)


def test(num_chunks):
    return _load_vectors('test', num_chunks)


def submission(num_chunks):
    return _load_vectors('submission', num_chunks)


def _save_vectors(name, num_chunks):
    batch_size = 5000

    if name == 'train':
        loader = load.gen_train
    elif name == 'validate':
        loader = load.gen_validate
    elif name == 'test':
        loader = load.gen_test
    elif name == 'submission':
        loader = load.gen_submission
    else:
        raise Exception('unexpected name {}'.format(name))

    fname = 'data/{}-vec-{}-{}-{}.pkl.gz'.format(
        name, batch_size, num_chunks, TRAIN_CHUNKS
    )
    with gzip.open(fname, 'wb') as ifile:
        for data, labels in vectorize(loader, num_chunks, batch_size):
            extra_bits = 8 - (data.shape[1] % 8)
            data = np.packbits(data, axis=1)
            pickle.dump((extra_bits, data, labels), ifile)


def _load_vectors(name, num_chunks):
    batch_size = 5000
    fname = 'data/{}-vec-{}-{}-{}.pkl.gz'.format(
        name, batch_size, num_chunks, TRAIN_CHUNKS
    )
    if not os.path.exists(fname):
        _save_vectors(name, num_chunks)

    with gzip.open(fname, 'rb') as ifile:
        while True:
            try:
                (extra_bits, data, labels) = pickle.load(ifile)
                data = np.unpackbits(data, axis=1)
                data = data[:, :-extra_bits]
                yield data, labels
            except EOFError:
                break
