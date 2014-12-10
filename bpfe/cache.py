

from bpfe.config import StatsSettings


# noinspection PyBroadException
try:
    import cPickle as pickle
except:
    import pickle


import gzip
import os


def save_pretrain_layer(dbn, layer_num, settings):
    fname = 'data/pretrain-layer/{}.pkl.gz'.format(
        settings.pretrain_fname(layer_num)
    )
    with gzip.open(fname, 'wb') as ifile:
        pickle.dump((dbn, settings), ifile, -1)


def save_finetuning(dbn, settings, klass_num):
    fname = 'data/finetuning/{}-{}.pkl.gz'.format(
        settings.finetuning_fname(), klass_num
    )
    with gzip.open(fname, 'wb') as ifile:
        pickle.dump((dbn, settings), ifile, -1)


def load_pretrain_layer(layer_num, settings):
    fname = 'data/pretrain-layer/{}.pkl.gz'.format(
        settings.pretrain_fname(layer_num)
    )
    if os.path.exists(fname):
        with gzip.open(fname, 'rb') as ifile:
            data = pickle.load(ifile)
            dbn = data[0]
            dbn.hidden_layer_sizes = \
                [hl.num_nodes for hl in settings.hidden_layers]
            settings.numpy_rng = data[1].numpy_rng
            settings.theano_rng = data[1].theano_rng
            settings.train_size = data[1].train_size
            return dbn, settings
    else:
        return None


def load_finetuning(settings, klass_num):
    fname = 'data/finetuning/{}-{}.pkl.gz'.format(
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
