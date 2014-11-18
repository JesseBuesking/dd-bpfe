

import csv
from bpfe.clean import clean_value
from bpfe.config import INPUT_MAPPING, LABEL_MAPPING
from bpfe.entities import Data, Label


def generate_test_rows():
    for _, data in generate_rows('data/TestData.csv'):
        yield data


def generate_training_rows():
    for label, data in generate_rows('data/TrainingData.csv'):
        yield label, data


def generate_rows(file_path):
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

        for line in reader:
            d = Data()
            l = Label()
            for key, idx in input_index_map.iteritems():
                setattr(d, INPUT_MAPPING[key], clean_value(line[idx]))
            for key, idx in label_index_map.iteritems():
                setattr(l, LABEL_MAPPING[key], clean_value(line[idx]))

            yield l, d
