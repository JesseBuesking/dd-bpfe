

import re


def isfloat(value):
    # noinspection PyBroadException
    try:
        float(value)
        return True
    except:
        return False


def sentence_splitter(sentence):
    # for word in sentence.split():
    #     yield word
    for word in re.findall('\w+|\d+|\.', sentence):
        yield word
