

import csv
from bpfe.clean import clean_value
from bpfe.config import INPUT_MAPPING, LABEL_MAPPING


# noinspection PyUnresolvedReferences
class Data(object):
    __slots__ = (
        'object_description', 'program_description', 'subfund_description',
        'job_title_description', 'facility_or_department',
        'sub_object_description', 'location_description', 'fte',
        'function_description', 'position_extra', 'text_4', 'total', 'text_2',
        'text_3', 'fund_description', 'text_1'
    )

    def __str__(self):
        val = '{}:\n'.format(self.__class__.__name__)
        for key in self.__slots__:
            val += '  {:>25}: "{}"\n'.format(key, getattr(self, key))
        return val


# noinspection PyUnresolvedReferences
class Label(object):
    __slots__ = (
        'function', 'use', 'sharing', 'reporting', 'student_type',
        'position_type', 'object_type', 'pre_k', 'operating_status'
    )

    def __str__(self):
        val = '{}:\n'.format(self.__class__.__name__)
        for key in self.__slots__:
            val += '  {:>25}: "{}"\n'.format(key, getattr(self, key))
        return val


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
