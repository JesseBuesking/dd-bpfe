

from dircache import listdir
from genericpath import isfile
from os.path import join
import re
from bpfe.config import StatsSettings


import gzip
import os
import gc
import cPickle as pickle


def _pickle_save(name, value):
    if name.endswith('gz'):
        with gzip.open(name, 'wb') as ifile:
            pickle.dump(
                value,
                ifile,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        gc.collect()
    else:
        with open(name, 'wb') as ifile:
            pickle.dump(
                value,
                ifile,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        gc.collect()


def _pickle_load(name):
    if not os.path.exists(name):
        return None

    if name.endswith('gz'):
        with gzip.open(name, 'rb') as ifile:
            value = pickle.load(ifile)
            gc.collect()
            return value
    else:
        with open(name, 'rb') as ifile:
            value = pickle.load(ifile)
            gc.collect()
            return value


def save_full(dbn, settings, percent, version, extra=''):
    fname = 'data/pretrain-layer/dbn-{}-{}-{}'.format(
        version, percent, extra
    )

    _pickle_save(fname + '-1.pkl', dbn)
    _pickle_save(fname + '-2.pkl', settings)


def load_full(percent, version, extra=''):
    fname = 'data/pretrain-layer/dbn-{}-{}-{}'.format(
        version, percent, extra
    )
    dbn = _pickle_load(fname + '-1.pkl')
    settings = _pickle_load(fname + '-2.pkl')

    return dbn, settings


def save_dbn(dbn, percent, version, extra=''):
    fname = 'data/pretrain-layer/dbn-{}-{}-{}.pkl'.format(
        version, percent, extra
    )

    _pickle_save(fname, dbn)


def load_dbn(percent, version, extra=''):
    fname = 'data/pretrain-layer/dbn-{}-{}-{}.pkl'.format(
        version, percent, extra
    )
    if not os.path.exists(fname):
        return None

    dbn = _pickle_load(fname)

    return dbn


def save_settings(settings, percent, version, extra=''):
    fname = 'data/pretrain-layer/settings-{}-{}-{}.pkl'.format(
        version, percent, extra
    )

    _pickle_save(fname, settings)


def load_settings(percent, version, extra=''):
    fname = 'data/pretrain-layer/settings-{}-{}-{}.pkl'.format(
        version, percent, extra
    )
    if not os.path.exists(fname):
        return None

    settings = _pickle_load(fname)

    return settings


def save_pretrain_layer(dbn, layer_num, settings, epoch):
    fname = 'data/pretrain-layer/{}-{}'.format(
        settings.pretrain_fname(layer_num), str(epoch).zfill(5)
    )
    with gzip.open(fname, 'wb') as ifile:
        pickle.dump((dbn, settings), ifile, protocol=pickle.HIGHEST_PROTOCOL)


def save_finetuning(dbn, settings, klass_num, epoch):
    fname = 'data/finetuning/{}-{}-{}.pkl.gz'.format(
        settings.finetuning_fname(), klass_num, str(epoch).zfill(5)
    )
    with gzip.open(fname, 'wb') as ifile:
        pickle.dump((dbn, settings), ifile, protocol=pickle.HIGHEST_PROTOCOL)


def save_best_finetuning(dbn, settings, klass_num):
    fname = 'data/best-finetuning/{}-{}.pkl.gz'.format(
        settings.finetuning_fname(), klass_num
    )
    with gzip.open(fname, 'wb') as ifile:
        pickle.dump((dbn, settings), ifile, protocol=pickle.HIGHEST_PROTOCOL)


def load_pretrain_layer(layer_num, settings):
    pth = 'data/pretrain-layer'
    files = [f for f in listdir(pth) if isfile(join(pth, f))]
    if len(files) <= 0:
        return None

    this_run_epochs = settings.hidden_layers[layer_num].epochs - 1

    files.sort(reverse=True)
    fname, epoch = None, None
    for f in files:
        epoch = int([m.group() for m in re.finditer('\d+', f)][-1])
        if epoch > this_run_epochs:
            continue

        fname = '{}-{}.pkl.gz'.format(
            settings.pretrain_fname(layer_num),
            str(epoch).zfill(5)
        )
        if settings.pretrain_fname(layer_num) in f:
            fname = '{}/{}'.format(pth, fname)
            break

    if not os.path.exists(fname):
        return None

    with gzip.open(fname, 'rb') as ifile:
        data = pickle.load(ifile)
        dbn = data[0]
        dbn.hidden_layer_sizes = \
            [hl.num_nodes for hl in settings.hidden_layers]
        settings_bak = settings
        settings = data[1]
        settings.load_pretrain(settings_bak, layer_num)
        return dbn, settings, epoch


def load_finetuning(settings, klass_num):
    pth = 'data/finetuning'
    files = [f for f in listdir(pth) if isfile(join(pth, f))]
    if len(files) <= 0:
        return None, None, None

    this_run_epochs = settings.finetuning.epochs

    files.sort(reverse=True)
    fname, epoch = None, None
    for f in files:
        epoch = int([m.group() for m in re.finditer('\d+', f)][-1])
        if epoch > this_run_epochs:
            continue

        fname = '{}-{}-{}.pkl.gz'.format(
            settings.finetuning_fname(),
            klass_num,
            str(epoch).zfill(5)
        )
        if settings.finetuning_fname() in f:
            fname = '{}/{}'.format(pth, fname)
            break

    if not os.path.exists(fname):
        return None, None, None

    with gzip.open(fname, 'rb') as ifile:
        data = pickle.load(ifile)
        dbn = data[0]
        settings_bak = settings
        settings = data[1]
        settings.load_finetuning(settings_bak)
        return dbn, settings, epoch


def load_best_finetuning(settings, klass_num):
    fname = 'data/best-finetuning/{}-{}.pkl.gz'.format(
        settings.finetuning_fname(), klass_num
    )
    if os.path.exists(fname):
        with gzip.open(fname, 'rb') as ifile:
            data = pickle.load(ifile)
            dbn = data[0]
            dbn.hidden_layer_sizes = \
                [hl.num_nodes for hl in settings.hidden_layers]
            settings = data[1]
            return dbn, settings
    else:
        return None


def load_stats():
    if os.path.exists(StatsSettings.fname):
        with open(StatsSettings.fname, 'rb') as ifile:
            return pickle.load(ifile)
    else:
        return dict()
