

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from bpfe.text_to_vec import TextVectorizer, BinaryVectorizer


remove_punc_keep_hyph_tokenizer = RegexpTokenizer(r'\w+(-\w+)*')
sw = stopwords.words('english')


def update_features(datas):
    for data, label in datas:
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
    # noinspection PyArgumentList
    il = [values[i:j+1] for i in xrange(l) for j in xrange(i, l)]
    il = [' '.join(i) for i in il if len(i) <= n]
    return il


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def get_vectorizers(num_chunks):
    text_attributes = [
        'object_description', 'program_description',
        'subfund_description', 'job_title_description',
        'facility_or_department', 'sub_object_description',
        'location_description', 'function_description', 'position_extra',
        'text_4', 'text_2', 'text_3', 'fund_description', 'text_1'
    ]

    v = dict()

    for attr in text_attributes:
        v[attr] = text_vectorizer(num_chunks, attr)

    v['fte'] = bucket_vectorizer(num_chunks, 'fte', [
        (None, 0.0),
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
        (20.0, 25.0),
        (25.0, 30.0),
        (30.0, 35.0),
        (35.0, 40.0),
        (40.0, 45.0),
        (45.0, 50.0),
        (50.0, 55.0),
        (55.0, 60.0),
        (60.0, 65.0),
        (65.0, 70.0),
        (70.0, 75.0),
        (75.0, 80.0),
        (80.0, 85.0),
        (85.0, 90.0),
        (90.0, 95.0),
        # i want 100% to be included, otherwise it gets lumped with the
        # garbage above 100
        (95.0, 100.001),
        (100.001, None),
    ])

    v['total'] = bucket_vectorizer(num_chunks, 'total', [
        (None, 0),
        (0, 5000),
        (5000, 10000),
        (10000, 15000),
        (15000, 20000),
        (20000, 25000),
        (25000, 30000),
        (30000, 35000),
        (35000, 40000),
        (40000, 45000),
        (45000, 50000),
        (50000, 55000),
        (55000, 60000),
        (60000, 65000),
        (65000, 70000),
        (70000, 75000),
        (75000, 80000),
        (80000, 85000),
        (85000, 90000),
        (90000, 95000),
        (95000, 100000),
        (100000, 105000),
        (105000, 110000),
        (110000, 115000),
        (115000, 120000),
        (120000, None),
    ])

    return v


def _text_vectorizer_prep(value):
    return value


def text_vectorizer_transform(v, values, _):
    return v.transform(_text_vectorizer_prep(values))


def text_vectorizer(num_chunks, attr):
    import bpfe.load as load
    v = TextVectorizer()

    def get_values(generator):
        ret = []
        for data in generator(num_chunks):
            for row, label in data:
                value = getattr(row, attr)
                ret.append(_text_vectorizer_prep(value))
        return ret

    values = \
        get_values(load.gen_train) + \
        get_values(load.gen_validate) + \
        get_values(load.gen_test) + \
        get_values(load.gen_submission)

    v.fit(values)
    return v, None, 'text_vectorizer_transform'


def _bucket_vectorizer_prep(value, buckets):
    # noinspection PyBroadException
    try:
        value = float(value)
    except:
        return 0

    value *= 100

    idx = 1
    for mini, maxi in buckets:
        if mini is None and value < maxi:
            return idx
        elif maxi is None and mini <= value:
            return idx
        elif mini <= value < maxi:
            return idx
        idx += 1


def bucket_vectorizer_transform(v, values, buckets):
    return v.transform(_bucket_vectorizer_prep(values, buckets))


def bucket_vectorizer(num_chunks, attr, buckets):
    import bpfe.load as load
    v = BinaryVectorizer()

    def get_values(generator):
        ret = []
        for data in generator(num_chunks):
            for row, label in data:
                value = getattr(row, attr)
                ret.append(_bucket_vectorizer_prep(value, buckets))
        return ret

    values = \
        get_values(load.gen_train) + \
        get_values(load.gen_validate) + \
        get_values(load.gen_test) + \
        get_values(load.gen_submission)

    v.fit(values)
    return v, buckets, 'bucket_vectorizer_transform'
