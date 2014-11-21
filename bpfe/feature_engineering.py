

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
remove_punc_keep_hyph_tokenizer = RegexpTokenizer(r'\w+(-\w+)*')
sw = stopwords.words('english')


def update_features(datas):
    for label, data in datas:
        for idx, t in enumerate(data.attribute_types):
            if t != str:
                continue

            att = data.attributes[idx]
            value = getattr(data, att)
            values = remove_punctuation(value)
            values = remove_stopwords(values)

            # add all ngrams <= length 5
            values = all_ngrams(values, 5)

            # only add the raw value if it's not there
            if value not in values:
                values.append(value)

            setattr(data, att, values)

    return datas


def parse_words_raw(value):
    values = word_tokenize(value)
    return values


def remove_stopwords(values):
    values = [i for i in values if i not in sw]
    return values


def remove_punctuation(values):
    values = remove_punc_keep_hyph_tokenizer.tokenize(values)
    return values


def all_ngrams(values, n):
    l = len(values)
    il = [values[i:j+1] for i in xrange(l) for j in xrange(i, l)]
    il = [' '.join(i) for i in il if len(i) <= n]
    return il


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))
