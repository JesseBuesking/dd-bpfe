

import csv
import sys
from bpfe.clean import clean_value
from bpfe.config import INPUT_MAPPING, LABEL_MAPPING
from bpfe.entities import Data, Label
from bpfe.reservoir import reservoir


def generate_test_rows(seed=1, amt=None):
    if amt is None:
        amt = sys.maxint
    ret = []
    for _, data in generate_rows('data/TestData.csv', seed, amt):
        ret.append(data)
    return ret


def generate_training_rows(seed=1, amt=None):
    if amt is None:
        amt = sys.maxint
    ret = []
    for label, data in generate_rows('data/TrainingData.csv', seed, amt):
        ret.append((label, data))
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

            yield l, d
