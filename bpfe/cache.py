from dircache import listdir
from genericpath import isfile
from os.path import join
import re
from bpfe.config import StatsSettings


# noinspection PyBroadException
try:
    import cPickle as pickle
except:
    import pickle


import gzip
import os


def save_pretrain_layer(dbn, layer_num, settings, epoch):
    fname = 'data/pretrain-layer/{}-{}.pkl.gz'.format(
        settings.pretrain_fname(layer_num), epoch
    )
    with gzip.open(fname, 'wb') as ifile:
        pickle.dump((dbn, settings), ifile, -1)


def save_finetuning(dbn, settings, klass_num, epoch):
    fname = 'data/finetuning/{}-{}-{}.pkl.gz'.format(
        settings.finetuning_fname(), klass_num, epoch
    )
    with gzip.open(fname, 'wb') as ifile:
        pickle.dump((dbn, settings), ifile, -1)


def save_best_finetuning(dbn, settings, klass_num):
    fname = 'data/best-finetuning/{}-{}.pkl.gz'.format(
        settings.finetuning_fname(), klass_num
    )
    with gzip.open(fname, 'wb') as ifile:
        pickle.dump((dbn, settings), ifile, -1)


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
            epoch
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
        settings.numpy_rng = data[1].numpy_rng
        settings.theano_rng = data[1].theano_rng
        settings.train_size = data[1].train_size
        return dbn, settings, epoch


def load_finetuning(settings, klass_num):
    pth = 'data/finetuning'
    files = [f for f in listdir(pth) if isfile(join(pth, f))]
    if len(files) <= 0:
        return None

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
            epoch
        )
        if settings.finetuning_fname() in f:
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
        settings.finetuning.epochs = settings_bak.finetuning.epochs
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
