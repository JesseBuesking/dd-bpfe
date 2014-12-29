import gzip
import os
from os.path import dirname
import pickle
import re
import math
from nltk import NaiveBayesClassifier, SklearnClassifier, accuracy
from scipy.sparse import hstack, vstack
from sklearn import cross_validation
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import sys
import time
from bpfe import load, util, feature_engineering, scoring
from bpfe.config import Settings, ChunkSettings, KLASS_LABEL_INFO, LABEL_MAPPING, LABELS, FLAT_LABELS
from bpfe.entities import Data, Label
import numpy as np
from unidecode import unidecode
from bpfe.load import ugen_train, ugen_validate, ugen_test, ugen_submission
from bpfe.models.perceptron_model import PerceptronModel
from bpfe.text_transform import transform
from bpfe.transformer.dense_transformer import DenseTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import SnowballStemmer
from bpfe.vectorizer.BPFEVectorizer import BPFEVectorizer


sbs = SnowballStemmer('english')


def bow(string):
    # return util.sentence_splitter(string)
    for word in re.findall(
            r'GRADE=k\|k|GRADE=k\|\d+|GRADE=\d+\|\d+|\w+|\d+|\.', string):
        yield word


def get_x_y(name, generator):
    bv = BPFEVectorizer()
    vect_data = bv.fit_transform([i for i, _ in generator])
    labels = [i.function for _, i in generator if i is not None]

    return vect_data, labels


# def get_x_y(name, generator):
#     x, y = [], []
#     uq = set()
#     uqsbs = set()
#     for data, label in generator:
#         row = []
#         for attr in Data.text_attributes:
#             value = data.cleaned[attr + '-mapped']
#             # value = getattr(data, attr)
#             b_o_w = []
#             for i in bow(value):
#                 uq.add(i)
#                 i = sbs.stem(i)
#                 uqsbs.add(i)
#                 b_o_w.append(i)
#             # need 1 before and 1 after to support 3-grams
#             # e.g. board ENDHERE BEGHERE something
#             #     |---------------------|
#             row += ['BEGHERE'] + b_o_w + ['ENDHERE']
#
#         x.append(' '.join(row))
#         if label is not None:
#             y.append(label.function)
#         else:
#             y = None
#     print('uq {}: {}'.format(name, len(uq)))
#     print('uqsbs {}: {}'.format(name, len(uqsbs)))
#     return x, y


def ll_score(classifier, klasses, tX, tY):
    preds = classifier.predict_proba(tX)
    actuals = []
    for expected in tY:
        tmp = np.zeros(len(klasses))
        tmp[klasses.index(expected)] = 1
        actuals.append(tmp)

    mmll = scoring.multi_multi_log_loss(
        np.array(preds),
        np.array(actuals),
        np.array([range(actuals[0].shape[0])])
    )
    return mmll


def kf(pipeline, tx, ty, name, k=5, idx=None):
    scores = []
    for train_ix, test_ix in cross_validation.KFold(len(tx), k):
        train_x2, train_y2 = [], []
        test_x2, test_y2 = [], []
        for tix in train_ix:
            train_x2.append(tx[tix])
            train_y2.append(ty[tix])
        for tix in test_ix:
            test_x2.append(tx[tix])
            test_y2.append(ty[tix])
        pipeline.fit(train_x2, train_y2)
        score = ll_score(
            pipeline,
            list(pipeline.named_steps['classifier'].classes_),
            test_x2,
            test_y2
        )
        scores.append([idx, score, name])
    return scores


def run():
    # TODO stemming / lemmatization

    # REALLY good
    # pipeline = Pipeline([
    #     ('vect',
    #      CountVectorizer(
    #          stop_words='english',
    #          ngram_range=(1, 3),
    #          binary=True
    #      )),
    #     ('tfidf', TfidfTransformer(norm=None)),
    #     ('classifier',
    #      LogisticRegression(
    #          C=10
    #      ))
    # ])

    train = [i for i in ugen_train()]
    validate = [i for i in ugen_validate()]
    test = [i for i in ugen_test()]
    submission = [i for i in ugen_submission()]

    train_x, train_y = get_x_y('train', train)
    test_x, test_y = get_x_y('test', test)
    validate_x, validate_y = get_x_y('validate', validate)
    submission_x, submission_y = get_x_y('submission', submission)

    create_file = False
    plot_stuff = True
    # create_file = True
    # plot_stuff = False
    fname = dirname(dirname(__file__)) + '/data/rf-unique.pkl'
    if plot_stuff:
        with open(fname, 'rb') as ifile:
            # scores = pickle.load(ifile)
            # df = pd.DataFrame(scores, columns=['amount', 'score', 'dataset'])
            # sns.pointplot('amount', 'score', data=df, hue='dataset')
            # plt.title('score by size of dataset (1-3 gram count vectorizer, '
            #           'tfidf, chi2 2k, RF depth-3 1k feat)')
            # plt.xlabel('amount (5k increments)')
            # plt.show()

            scores = pickle.load(ifile)
            df = pd.DataFrame(scores, columns=['amount', 'score', 'dataset'])
            sns.pointplot('amount', 'score', data=df, hue='dataset')
            plt.title('score by size of dataset (1-3 gram count vectorizer, '
                      'tfidf, logreg C=10)')
            plt.xlabel('amount (5k increments)')
            plt.show()
            return

    # pipeline = Pipeline([
    #     ('vect',
    #      CountVectorizer(
    #          stop_words='english',
    #          # min_df=3,
    #          ngram_range=(1, 3),
    #      )),
    #     ('tfidf', TfidfTransformer()),
    #     ('chi2',
    #      SelectKBest(
    #          chi2,
    #          k=2000
    #      )),
    #     ('todense', DenseTransformer()),
    #     ('classifier',
    #      RandomForestClassifier(
    #          max_depth=30,
    #          max_features=100,
    #          bootstrap=True,
    #          criterion='entropy',
    #          random_state=1,
    #          oob_score=True
    #      )),
    #     # ('classifier',
    #     #  GradientBoostingClassifier(
    #     #      n_estimators=200,
    #     #      max_depth=3
    #     #  )),
    #     # ('classifier',
    #     #  MultinomialNB(
    #     #      alpha=0.001,
    #     #      fit_prior=True
    #     #  )),
    #     # ('classifier',
    #     #  LogisticRegression(
    #     #      C=10
    #     #  ))
    # ])

    pipeline = Pipeline([
        ('vect',
         CountVectorizer(
             stop_words='english',
             ngram_range=(1, 3),
         )),
        ('tfidf', TfidfTransformer()),
        ('classifier',
         LogisticRegression(
             C=10
         ))
    ])

    lr_grid = dict(
        vect__min_df=[1, 5],
        vect__binary=[True, False],
        tfidf__norm=[None, 'l1', 'l2'],
        classifier__C=[.01, .1, 1, 10, 100],
        classifier__penalty=[None, 'l1', 'l2'],
        classifier__fit_intercept=[False, True]
    )

    num = 20000
    tx, ty = train_x[:num], train_y[:num]

    grid_search = GridSearchCV(pipeline, param_grid=lr_grid, verbose=10)
    grid_search.fit(tx, ty)

    print('\nbest estimator:')
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return
    #
    # scores = sorted(
    #     grid_search.grid_scores_, key=lambda x: x.mean_validation_score,
    #     reverse=True
    # )
    # for score in scores[:100]:
    #     print(score)

    scores = []
    for i in range(5, 41, 5):
        i_s = i
        print('i_s: {}'.format(i_s))

        # load the data
        tx, ty = train_x[:i*1000], train_y[:i*1000]
        for train_ix, test_ix in cross_validation.KFold(len(tx), 5):
            train_ix.sort()
            test_ix.sort()
            train_x2, train_y2 = [], []
            test_x2, test_y2 = [], []
            for tix in train_ix:
                train_x2.append(tx[tix])
                train_y2.append(ty[tix])
            for tix in test_ix:
                test_x2.append(tx[tix])
                test_y2.append(ty[tix])
            pipeline.fit(train_x2, train_y2)
            score = ll_score(
                pipeline,
                list(pipeline.named_steps['classifier'].classes_),
                test_x2,
                test_y2
            )
            scores.append([i_s, score, 'cv'])

        pipeline.fit(train_x, train_y)

        score = ll_score(
            pipeline,
            list(pipeline.named_steps['classifier'].classes_),
            validate_x,
            validate_y
        )
        for z in range(5):
            scores.append([i_s, score, 'validate'])

        score = ll_score(
            pipeline,
            list(pipeline.named_steps['classifier'].classes_),
            train_x,
            train_y,
        )
        for z in range(5):
            scores.append([i_s, score, 'train'])

    # if create_file:
    #     with open(fname, 'wb') as ifile:
    #         pickle.dump(scores, ifile, -1)


def rf_tuning(plot_me=False):
    fname = 'data/grid-search-rf-snowball.pkl'
    if plot_me:
        with open(fname, 'rb') as ifile:
            scores = pickle.load(ifile)
            df = pd.DataFrame(scores, columns=['amount', 'score', 'name'])
            df.sort(columns=['score'])
            print(df.head(sys.maxint))
            return

    num = 10000
    tx, ty = [], []
    for idx, (d, l) in enumerate(ugen_train()):
        if idx >= num:
            break

        tx.append(d)
        ty.append(l.function)

    best = (sys.maxint, 0, '')
    scores = []
    for vect__ngrams in [1]:
        for vect__use_grades in [True]:
            for vect__use_titles in [True]:
                for vect__filter_stopwords in [True, False]:
                    for vect__binarize in [True, False]:
                        for vect__min_df in [1, 3]:
                            for classifier__max_depth in [9, 12, 15]:
                                for classifier__max_features in [200, 400]:
                                    for classifier__criterion in ['gini', 'entropy']:
                                        pstring = ', '.join([
                                            'vect__ngrams: {}'.format(
                                                vect__ngrams),
                                            'vect__use_grades: {}'.format(
                                                vect__use_grades),
                                            'vect__use_titles: {}'.format(
                                                vect__use_titles),
                                            'vect__filter_stopwords: {}'.format(
                                                vect__filter_stopwords),
                                            'vect__binarize: {}'.format(
                                                vect__binarize),
                                            'vect__min_df: {}'.format(
                                                vect__min_df),
                                            'classifier__max_depth: {}'.format(
                                                classifier__max_depth),
                                            'classifier__max_features: {}'.format(
                                                classifier__max_features),
                                            'classifier__criterion: {}'.format(
                                                classifier__criterion)
                                        ])
                                        print(pstring)
                                        start = time.clock()
                                        pipeline = Pipeline([
                                            ('vect',
                                             BPFEVectorizer(
                                                 ngrams=vect__ngrams,
                                                 stemmer=sbs,
                                                 use_grades=vect__use_grades,
                                                 use_titles=vect__use_titles,
                                                 filter_stopwords=vect__filter_stopwords,
                                                 binarize=vect__binarize,
                                                 min_df=vect__min_df
                                             )),
                                            # ('chi2',
                                            #  SelectKBest(
                                            #      chi2,
                                            #      k=1000
                                            #  )),
                                            ('todense', DenseTransformer()),
                                            ('classifier',
                                             RandomForestClassifier(
                                                 max_depth=classifier__max_depth,
                                                 max_features=classifier__max_features,
                                                 bootstrap=True,
                                                 criterion=classifier__criterion,
                                                 random_state=1,
                                                 oob_score=True
                                             )),
                                        ])

                                        s = kf(pipeline, tx, ty, pstring, k=3)
                                        scores += s
                                        ss = [i[1] for i in s]
                                        mn = np.mean(ss)
                                        std = np.std(ss)
                                        print('{:.4f} +/- {:.4f} after {}'.format(
                                            mn,
                                            std,
                                            time.clock() - start
                                        ))
                                        if mn < best[0]:
                                            best = (mn, std, pstring)

    print('best:')
    print(best)

    # with open(fname, 'wb') as ifile:
    #     pickle.dump(scores, ifile, -1)


def ada_rf_tuning(plot_me=False):
    fname = 'data/grid-search-ada-rf-snowball.pkl'
    if plot_me:
        with open(fname, 'rb') as ifile:
            scores = pickle.load(ifile)
            df = pd.DataFrame(scores, columns=['amount', 'score', 'name'])
            df.sort(columns=['score'])
            print(df.head(sys.maxint))
            return

    train = [i for i in ugen_train()]
    train_x, train_y = get_x_y('train', train)

    num = 3000
    tx, ty = train_x[:num], train_y[:num]

    best = (sys.maxint, 0, '')
    scores = []
    for vect__ngrams in [1, 2, 3]:
        for vect__use_grades in [True]:
            for vect__use_titles in [True]:
                for vect__filter_stopwords in [True, False]:
                    for vect__binarize in [True, False]:
                        for vect__min_df in [1, 3]:
                            for classifier__n_estimators in [50, 100]:
                                for classifier__learning_rate in [.1, 1, 10]:
                                    pstring = ', '.join([
                                        'vect__ngrams: {}'.format(
                                            vect__ngrams),
                                        'vect__use_grades: {}'.format(
                                            vect__use_grades),
                                        'vect__use_titles: {}'.format(
                                            vect__use_titles),
                                        'vect__filter_stopwords: {}'.format(
                                            vect__filter_stopwords),
                                        'vect__binarize: {}'.format(
                                            vect__binarize),
                                        'vect__min_df: {}'.format(
                                            vect__min_df),
                                        'classifier__n_estimators: {}'.format(
                                            classifier__n_estimators),
                                        'classifier__learning_rate: {}'.format(
                                            classifier__learning_rate)
                                    ])
                                    print(pstring)
                                    start = time.clock()
                                    pipeline = Pipeline([
                                        ('vect',
                                         BPFEVectorizer(
                                             ngrams=vect__ngrams,
                                             stemmer=sbs,
                                             use_grades=vect__use_grades,
                                             use_titles=vect__use_titles,
                                             filter_stopwords=vect__filter_stopwords,
                                             binarize=vect__binarize,
                                             min_df=vect__min_df
                                         )),
                                        ('chi2',
                                         SelectKBest(
                                             chi2,
                                             k=1000
                                         )),
                                        ('todense', DenseTransformer()),
                                        ('classifier',
                                         AdaBoostClassifier(
                                             # base_estimator=ExtraTreesClassifier(),
                                             n_estimators=classifier__n_estimators,
                                             learning_rate=classifier__learning_rate
                                         )),
                                    ])

                                    s = kf(pipeline, tx, ty, pstring, k=3)
                                    scores += s
                                    ss = [i[1] for i in s]
                                    mn = np.mean(ss)
                                    std = np.std(ss)
                                    print('{:.4f} +/- {:.4f} after {}'.format(
                                        mn,
                                        std,
                                        time.clock() - start
                                    ))
                                    if mn < best[0]:
                                        best = (mn, std, pstring)

    print('best:')
    print(best)

    with open(fname, 'wb') as ifile:
        pickle.dump(scores, ifile, -1)


def gradient_boosting_tuning(plot_me=False):
    fname = 'data/grid-search-gradient-boosting-snowball.pkl'
    if plot_me:
        with open(fname, 'rb') as ifile:
            scores = pickle.load(ifile)
            df = pd.DataFrame(scores, columns=['amount', 'score', 'name'])
            df.sort(columns=['score'])
            print(df.head(sys.maxint))
            return

    train = [i for i in ugen_train()]
    train_x, train_y = get_x_y('train', train)

    num = 5000
    tx, ty = train_x[:num], train_y[:num]

    best = (sys.maxint, 0, '')
    scores = []
    for vect__min_df in [3]:
        for vect__binary in [True, False]:
            for tfidf__norm in [None, 'l1', 'l2']:
                for chi2__k in [1000, 2000]:
                    for classifier__n_estimators in [250]:
                        for classifier__max_depth in [5]:
                            pstring = ', '.join([
                                'vect__min_df: {}'.format(vect__min_df),
                                'vect__binary: {}'.format(vect__binary),
                                'tfidf__norm: {}'.format(tfidf__norm),
                                'classifier__n_estimators: {}'.format(
                                    classifier__n_estimators),
                                'classifier__max_depth: {}'.format(
                                    classifier__max_depth)
                            ])
                            print(pstring)
                            start = time.clock()
                            pipeline = Pipeline([
                                ('vect',
                                 CountVectorizer(
                                     stop_words='english',
                                     ngram_range=(1, 3),
                                     min_df=vect__min_df,
                                     binary=vect__binary
                                 )),
                                ('tfidf',
                                 TfidfTransformer(
                                     norm=tfidf__norm
                                 )),
                                ('chi2',
                                 SelectKBest(
                                     chi2,
                                     k=chi2__k
                                 )),
                                ('todense', DenseTransformer()),
                                ('classifier',
                                 GradientBoostingClassifier(
                                     n_estimators=classifier__n_estimators,
                                     max_depth=classifier__max_depth
                                 )),
                            ])

                            s = kf(pipeline, tx, ty, pstring, k=3)
                            scores += s
                            ss = [i[1] for i in s]
                            mn = np.mean(ss)
                            std = np.std(ss)
                            print('{:.4f} +/- {:.4f} after {}'.format(
                                mn,
                                std,
                                time.clock() - start
                            ))
                            if mn < best[0]:
                                best = (mn, std, pstring)

    print('best:')
    print(best)

    with open(fname, 'wb') as ifile:
        pickle.dump(scores, ifile, -1)


def naive_bayes_tuning(plot_me=False):
    fname = 'data/grid-search-naive_bayes-snowball.pkl'
    if plot_me:
        with open(fname, 'rb') as ifile:
            scores = pickle.load(ifile)
            df = pd.DataFrame(scores, columns=['amount', 'score', 'name'])
            df.sort(columns=['score'])
            print(df.head(sys.maxint))
            return

    num = 10000
    tx, ty = [], []
    for idx, (d, l) in enumerate(ugen_train()):
        if idx >= num:
            break

        tx.append(d)
        ty.append(l.function)

    best = (sys.maxint, 0, '')
    scores = []
    for vect__ngrams in [1, 2, 3]:
        for vect__use_grades in [True]:
            for vect__use_titles in [True]:
                for vect__filter_stopwords in [True, False]:
                    for vect__binarize in [True, False]:
                        for vect__min_df in [1, 3]:
                            for classifier__alpha in [.1, .5, 1]:
                                for classifier__fit_prior in [False]:
                                    pstring = ', '.join([
                                        'vect__ngrams: {}'.format(
                                            vect__ngrams),
                                        'vect__use_grades: {}'.format(
                                            vect__use_grades),
                                        'vect__use_titles: {}'.format(
                                            vect__use_titles),
                                        'vect__filter_stopwords: {}'.format(
                                            vect__filter_stopwords),
                                        'vect__binarize: {}'.format(
                                            vect__binarize),
                                        'vect__min_df: {}'.format(
                                            vect__min_df),
                                        'classifier__alpha: {}'.format(
                                            classifier__alpha),
                                        'classifier__fit_prior: {}'.format(
                                            classifier__fit_prior)
                                    ])
                                    print(pstring)
                                    start = time.clock()
                                    pipeline = Pipeline([
                                        ('vect',
                                         BPFEVectorizer(
                                             ngrams=vect__ngrams,
                                             stemmer=sbs,
                                             use_grades=vect__use_grades,
                                             use_titles=vect__use_titles,
                                             filter_stopwords=vect__filter_stopwords,
                                             binarize=vect__binarize,
                                             min_df=vect__min_df
                                         )),
                                        ('classifier',
                                         MultinomialNB(
                                             alpha=classifier__alpha,
                                             fit_prior=classifier__fit_prior
                                         )),
                                    ])

                                    s = kf(pipeline, tx, ty, pstring, k=3)
                                    scores += s
                                    ss = [i[1] for i in s]
                                    mn = np.mean(ss)
                                    std = np.std(ss)
                                    print('{:.4f} +/- {:.4f} after {}'.format(
                                        mn,
                                        std,
                                        time.clock() - start
                                    ))
                                    if mn < best[0]:
                                        best = (mn, std, pstring)

    print('best:')
    print(best)

    with open(fname, 'wb') as ifile:
        pickle.dump(scores, ifile, -1)


def log_reg_tuning(plot_me=False):
    fname = 'data/grid-search-logreg-snowball.pkl'
    if plot_me:
        with open(fname, 'rb') as ifile:
            scores = pickle.load(ifile)
            df = pd.DataFrame(scores, columns=['amount', 'score', 'name'])
            df.sort(columns=['score'])
            print(df.head(sys.maxint))
            return

    num = 10000
    tx, ty = [], []
    for idx, (d, l) in enumerate(ugen_train()):
        if idx >= num:
            break

        tx.append(d)
        ty.append(l.function)

    best = (sys.maxint, 0, '')
    scores = []
    for vect__ngrams in [1, 2, 3]:
        for vect__use_grades in [True, False]:
            for vect__use_titles in [True, False]:
                for vect__filter_stopwords in [True, False]:
                    for vect__binarize in [True, False]:
                        for vect__min_df in [1, 3, 5]:
                            for tfidf__norm in ['l2']:
                                for classifier__C in [10]:
                                    for classifier__penalty in ['l1']:
                                        for classifier__fit_intercept in [True]:
                                            pstring = ', '.join([
                                                'vect__ngrams: {}'.format(
                                                    vect__ngrams),
                                                'vect__use_grades: {}'.format(
                                                    vect__use_grades),
                                                'vect__use_titles: {}'.format(
                                                    vect__use_titles),
                                                'vect__filter_stopwords: {}'.format(
                                                    vect__filter_stopwords),
                                                'vect__binarize: {}'.format(
                                                    vect__binarize),
                                                'vect__min_df: {}'.format(
                                                    vect__min_df),
                                                'tfidf__norm: {}'.format(
                                                    tfidf__norm),
                                                'classifier__C: {}'.format(
                                                    classifier__C),
                                                'classifier__penalty: {}'.format(
                                                    classifier__penalty),
                                                'classifier__fit_intercept: {}'.format(
                                                    classifier__fit_intercept),
                                            ])
                                            print(pstring)
                                            start = time.clock()
                                            pipeline = Pipeline([
                                                ('vect',
                                                 BPFEVectorizer(
                                                     ngrams=vect__ngrams,
                                                     stemmer=sbs,
                                                     use_grades=vect__use_grades,
                                                     use_titles=vect__use_titles,
                                                     filter_stopwords=vect__filter_stopwords,
                                                     binarize=vect__binarize,
                                                     min_df=vect__min_df
                                                 )),
                                                ('tfidf',
                                                 TfidfTransformer(
                                                     norm=tfidf__norm
                                                 )),
                                                ('classifier',
                                                 LogisticRegression(
                                                     C=classifier__C,
                                                     penalty=classifier__penalty,
                                                     fit_intercept=classifier__fit_intercept
                                                 ))
                                            ])

                                            s = kf(pipeline, tx, ty, pstring, k=3)
                                            scores += s
                                            ss = [i[1] for i in s]
                                            mn = np.mean(ss)
                                            std = np.std(ss)
                                            print('{:.4f} +/- {:.4f} after {}'.format(
                                                mn,
                                                std,
                                                time.clock() - start
                                            ))
                                            if mn < best[0]:
                                                best = (mn, std, pstring)

    print('best:')
    print(best)

    with open(fname, 'wb') as ifile:
        pickle.dump(scores, ifile, -1)


def avg_perc_tuning(plot_me=False):
    fname = 'data/grid-search-avg-perc-snowball.pkl'
    if plot_me:
        with open(fname, 'rb') as ifile:
            scores = pickle.load(ifile)
            df = pd.DataFrame(scores, columns=['amount', 'score', 'name'])
            df.sort(columns=['score'])
            print(df.head(sys.maxint))
            return

    start = time.clock()
    num = 20000
    train = [[data, label.function] for data, label in ugen_train()][:num]

    scores = []
    for train_ix, test_ix in cross_validation.KFold(len(train), 5):
        train_x2, test_x2 = [], []
        for tix in train_ix:
            train_x2.append(train[tix])
        for tix in test_ix:
            test_x2.append(train[tix])

        pm = PerceptronModel()
        pm.train(
            train_x2, test_x2, sys.maxint, nr_iter=75, seed=1, save_loc=None)

        preds = []
        actuals = []
        for data, label in test_x2:
            probs = pm.predict_proba(data)
            tmp = np.zeros(len(pm.classes))
            tmp[pm.classes.index(label)] = 1
            preds.append(probs)
            actuals.append(tmp)

        mmll = scoring.multi_multi_log_loss(
            np.array(preds),
            np.array(actuals),
            np.array([range(actuals[0].shape[0])])
        )
        print('mmll: {:.4f}'.format(mmll))

        scores.append([0, mmll, 'avg-perc'])

    ss = [i[1] for i in scores]
    mn = np.mean(ss)
    std = np.std(ss)
    print('')
    print('{:.4f} +/- {:.4f} after {}'.format(
        mn,
        std,
        time.clock() - start
    ))

    with open(fname, 'wb') as ifile:
        pickle.dump(scores, ifile, -1)


# noinspection PyDefaultArgument
def get_vects(name, classifier_indices=[0, 1], indices=None, num=sys.maxint,
              unique=False):
    vects = [
        BPFEVectorizer(
            ngrams=2,
            use_stemmer=True,
            use_grades=True,
            use_titles=True,
            filter_stopwords=False,
            binarize=True,
            min_df=None
        ),
        BPFEVectorizer(
            ngrams=1,
            use_stemmer=True,
            use_grades=True,
            use_titles=True,
            filter_stopwords=True,
            binarize=True,
            min_df=1
        )
    ]
    if unique:
        vname = dirname(dirname(__file__)) + \
            '/data/models/{}-vects.pkl'.format(name)
    else:
        vname = dirname(dirname(__file__)) + \
            '/data/models/final-{}-vects.pkl'.format(name)
    if not os.path.exists(vname):
        if name == 'train':
            g = ugen_train()
        elif name == 'test':
            g = ugen_test()
        elif name == 'validate':
            g = ugen_validate()
        elif name == 'submission':
            g = ugen_submission(unique)
        else:
            raise

        # save the vectorizers to disk
        vzrs = dirname(dirname(__file__)) + '/data/models/vectorizers.pkl'
        if not os.path.exists(vzrs):
            with open(vzrs, 'wb') as ifile:
                tx = []
                for idx, (d, l) in enumerate(ugen_train()):
                    if num < idx:
                        break
                    tx.append(d)

                for v in vects:
                    v.fit_transform(tx)
                pickle.dump(vects, ifile, -1)
        else:
            with open(vzrs, 'rb') as ifile:
                vects = pickle.load(ifile)

        tx = []
        for idx, (d, l) in enumerate(g):
            if num < idx:
                break
            tx.append(d)

        # save the vectorized data to disk
        with open(vname, 'wb') as ifile:
            data = []
            for i in range(len(vects)):
                data.append([])

            for ridx, row in enumerate(tx):
                for vidx, v in enumerate(vects):
                    data[vidx].append(v.transform([row]))

            data = [vstack(i) for i in data]
            pickle.dump(data, ifile, -1)

    # load the vectorized data from disk
    with open(vname, 'rb') as ifile:
        data = pickle.load(ifile)
        tmp = []
        for ridx in classifier_indices:
            t = data[ridx]
            if num < t.shape[0]:
                t = t[:num]
            if indices is not None:
                t = t[indices]
            tmp.append(t)
        if len(tmp) == 1:
            return tmp[0]
        else:
            return tmp


# @profile
def weighted_average(create_file=False):
    num = sys.maxint
    # num = 1000
    kfolds = 5
    num_classifiers = 2

    total_rows = get_vects('train', [0], num=num).shape[0]

    ifile = None
    for attr in Label.__slots__:
        if attr == 'id':
            continue

        ty = []
        for idx, (d, l) in enumerate(ugen_train()):
            if num < idx:
                break
            ty.append(
                getattr(l, attr)
            )
        ty = np.array(ty)

        fname = dirname(dirname(__file__)) + \
            '/data/models/weighted-average-{}.pkl'.format(attr)

        if create_file:
            if ifile is not None:
                ifile.close()
            ifile = gzip.open(fname + '.gz', 'wb')

        for kidx, (train_ix, test_ix) in enumerate(
                cross_validation.KFold(total_rows, kfolds)
        ):
            classifiers = [
                Pipeline([
                    ('tfidf',
                     TfidfTransformer(
                         norm='l2'
                     )),
                    ('classifier',
                     LogisticRegression(
                         C=10,
                         penalty='l1',
                         fit_intercept=True
                     ))
                ]),
                Pipeline([
                    ('chi2',
                     SelectKBest(
                         chi2,
                         k=1000
                     )),
                    ('todense', DenseTransformer()),
                    ('classifier',
                     RandomForestClassifier(
                         max_depth=12,
                         max_features=200,
                         bootstrap=True,
                         criterion='entropy',
                         random_state=1,
                         oob_score=True
                     )),
                ])
            ]

            train_y2 = ty[train_ix]

            print('training classifiers')
            for idx in range(len(classifiers)):
                print('training classifier {}'.format(idx))
                vect = get_vects('train', [idx], train_ix, num)
                classifiers[idx].fit(
                    vect,
                    train_y2
                )

            classes = [
                list(i.named_steps['classifier'].classes_) for i in
                classifiers
            ]

            print('vectorizing labels')
            actuals = []
            for cidx in range(ty.shape[0]):
                label = ty[cidx]
                tmp = np.zeros(len(classes[0]))
                tmp[classes[0].index(label)] = 1.0
                actuals.append(tmp)

            test_y2 = np.array(actuals)[test_ix]

            print('making predictions')
            pp = []
            for nidx in range(num_classifiers):
                vect = get_vects('train', [nidx], test_ix, num=num)
                pr = classifiers[nidx].predict_proba(vect)
                pp.append(pr)

            print('finding deltas')
            deltas = []
            for nidx in range(num_classifiers):
                deltas.append(1 - (test_y2 - pp[nidx]))

            print('computing adjusted weights')
            adj_weights = []
            for d in deltas:
                adj_weights.append(
                    (d - np.min(deltas, axis=0)) /
                    (np.max(deltas, axis=0) - np.min(deltas, axis=0))
                )

            print('cleaning adjusted weights')
            for i in range(len(adj_weights)):
                nw = adj_weights[i]
                # noinspection PyUnresolvedReferences
                nw[np.isnan(nw)] = 1
                eps = 1e-3
                nw = np.clip(nw, eps, 1 - eps)
                adj_weights[i] = nw

            print('finding final weights')
            weights = np.zeros((num_classifiers, len(classes[0]))) + \
                (1 / float(num_classifiers))
            for nidx, nw in enumerate(adj_weights):
                weights[nidx] = np.mean(nw, axis=0)

            print('making weighted predicts for mmll')
            pp2 = []
            for nidx in range(num_classifiers):
                pr = pp[nidx] * weights[nidx]
                pp2.append(pr)

            preds = None
            for p in pp2:
                if preds is None:
                    preds = p
                else:
                    preds += p
            preds /= float(num_classifiers)

            mmll = scoring.multi_multi_log_loss(
                np.array(preds),
                np.array(test_y2),
                np.array([range(test_y2.shape[1])])
            )
            print('mmll: {:.4f}'.format(mmll))

            print('weights', weights)

            if create_file:
                pickle.dump((weights, classifiers), ifile, -1)


def blah():
    validate_x, validate_y = get_x_y('validate', load.gen_validate(s))
    test_x, test_y = get_x_y('test', load.gen_test(s))

    # create train list of dicts
    train = []
    for row, label in zip(train_x, train_y):
        curr = []
        for word in row.split():
            curr.append((word, label))
        if len(curr) <= 0:
            continue
        train.append(dict(curr))

    # run the pipeline
    pipeline.fit(train_x, train_y)

    if False:
        if True:
            features = np.asarray(
                pipeline.named_steps['vect'].get_feature_names()
            )[pipeline.named_steps['chi2'].get_support()]
            feats = []
            for idx, name in enumerate(features):
                feats.append((
                    pipeline.named_steps['classifier'].feature_importances_[idx],
                    name
                ))
        else:
            feats = []
            for idx, name in enumerate(
                    pipeline.named_steps['vect'].get_feature_names()):
                feats.append((
                    pipeline.named_steps['classifier'].feature_importances_[idx],
                    name
                ))

        feats.sort(key=lambda x: x[0], reverse=True)
        t = 500
        b = 25
        header = 'top {} feats:'.format(t)
        print(header)
        print('-' * len(header))
        for f in feats[:t]:
            print(f)

        print('')

        header = 'bottom {} feats:'.format(b)
        print(header)
        print('-' * len(header))
        for f in feats[-b:]:
            print(f)

    # zip(
    #     pipeline.named_steps['vect'].vocabulary_,
    #     pipeline.named_steps['rf'].feature_importances_
    # )
    #
    # print(pipeline.named_steps['rf'].feature_importances_)

    # param_grid = dict(
    #     # chi2__k=[1000, 1500, 'all'],
    #     # more is better
    #     chi2__k=['all'],
    #     # vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
    #     # least variance
    #     vect__ngram_range=[(1, 1)],
    #     vect__stop_words=['english'],
    #     # vect__max_features=[None, 100, 500, 1000],
    #     # nb__alpha=[0.001],
    #     # nb__fit_prior=[True],
    #     log__C=[10]
    # )
    #
    # grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
    # grid_search.fit(train_x, train_y)
    #
    # print('\nbest estimator:')
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)
    #
    # scores = sorted(
    #     grid_search.grid_scores_, key=lambda x: x.mean_validation_score,
    #     reverse=True
    # )
    # for score in scores[:100]:
    #     print(score)

    def stats(name, X, Y):
        predictions = pipeline.predict_proba(X)
        matches = 0
        total = 0
        for actual, expected in zip(predictions, Y):
            actual_name = pipeline.steps[-1][1].classes_[np.argmax(actual)]
            matches += actual_name == expected
            total += 1

        print('{}: {} / {} = {:.2f}%'.format(
            name,
            matches,
            total,
            matches / float(total)
        ))
    stats('train', train_x, train_y)
    stats('validate', validate_x, validate_y)


def score_by_size_of_dataset(create_file=False, plot_stuff=True):
    fname = dirname(dirname(__file__)) + '/data/score-by-size.pkl'
    if plot_stuff:
        with open(fname, 'rb') as ifile:
            scores = pickle.load(ifile)
            df = pd.DataFrame(scores, columns=['amount', 'score', 'dataset'])
            sns.pointplot('amount', 'score', data=df, hue='dataset')
            plt.title('score by size of dataset (1-3 gram count vectorizer, '
                      'tfidf, logreg C=10)')
            plt.xlabel('amount (5k increments)')
            plt.show()
            return

    pipeline = Pipeline([
        ('vect',
         CountVectorizer(
             stop_words='english',
             ngram_range=(1, 3),
         )),
        ('tfidf', TfidfTransformer()),
        ('classifier',
         LogisticRegression(
             C=10
         ))
    ])

    scores = []
    for i in range(1, 20, 1):
        i_s = i
        s = Settings()
        s.chunks = ChunkSettings(i, sys.maxint, sys.maxint)

        # load the data
        train_x, train_y = get_x_y('train', load.gen_train(s))
        validate_x, validate_y = get_x_y('validate', load.gen_validate(s))
        s = cross_validation.cross_val_score(
            pipeline,
            train_x,
            train_y,
            cv=5
        )
        for si in s:
            scores.append([i_s, si, 'cv'])

        pipeline.fit(train_x, train_y)

        def stats(X, Y):
            predictions = pipeline.predict_proba(X)
            matches = 0
            total = 0
            for actual, expected in zip(predictions, Y):
                actual_name = pipeline.steps[-1][1].classes_[np.argmax(actual)]
                matches += actual_name == expected
                total += 1

            return matches, total

        matches, total = stats(validate_x, validate_y)
        for z in range(5):
            scores.append([i_s, matches / float(total), 'validate'])

        matches, total = stats(train_x, train_y)
        for z in range(5):
            scores.append([i_s, matches / float(total), 'train'])

    if create_file:
        with open(fname, 'wb') as ifile:
            pickle.dump(scores, ifile, -1)


def weighted_average_pred(name='train', unique=True):
    kfolds = 5

    if name == 'train':
        g = ugen_train
    elif name == 'test':
        g = ugen_test
    elif name == 'validate':
        g = ugen_validate
    elif name == 'submission':
        g = ugen_submission
    else:
        raise

    ordered_classes = sorted(LABELS.keys())
    all_preds = None
    for klass in ordered_classes:
        print('getting predictions for {}'.format(klass))
        labels = LABELS[klass]
        attr = LABEL_MAPPING[klass]

        if name != 'submission':
            ty = []
            for idx, (d, l) in enumerate(g(unique=unique)):
                ty.append(
                    getattr(l, attr)
                )
            ty = np.array(ty)

        fname = dirname(dirname(__file__)) + \
            '/data/models/weighted-average-{}.pkl'.format(attr)

        with gzip.open(fname + '.gz', 'rb') as ifile:
            preds = []
            for i in range(kfolds):
                preds.append(None)

            for i in range(kfolds):
                weights, classifiers = pickle.load(ifile)

                classes = [
                    list(z.named_steps['classifier'].classes_) for z in
                    classifiers
                ]
                assert classes[0] == labels
                num_classifiers = len(classifiers)
                # to get just the logistic regression score
                # num_classifiers = 1

                pp = []
                for nidx in range(num_classifiers):
                    vect = get_vects(name, [nidx], unique=unique)
                    pr = classifiers[nidx].predict_proba(vect)
                    pr2 = pr * weights[nidx]
                    pp.append(pr2)
                pred = preds[i]
                for p in pp:
                    if pred is None:
                        pred = p
                    else:
                        pred += p
                pred /= float(num_classifiers)
                preds[i] = pred

            preds = np.mean(preds, axis=0)

            if name == 'submission':
                if all_preds is None:
                    all_preds = preds
                else:
                    all_preds = np.concatenate([all_preds, preds], axis=1)
            else:
                actuals = []
                for cidx in range(preds.shape[0]):
                    label = ty[cidx]
                    tmp = np.zeros(len(classes[0]))
                    tmp[classes[0].index(label)] = 1.0
                    label = tmp
                    actuals.append(label)

                mmll = scoring.multi_multi_log_loss(
                    np.array(preds),
                    np.array(actuals),
                    np.array([range(actuals[0].shape[0])])
                )
                print('mmll: {:.4f}'.format(mmll))

    if all_preds is not None:
        header = ['__'.join(i) for i in FLAT_LABELS]
        headers = []
        for i in header:
            if ' ' in i:
                i = '"{}"'.format(i)
            headers.append(i)

        header_line = ',' + ','.join(headers)
        print(header_line)
        fname = dirname(dirname(__file__)) + '/data/submission/blend.csv'
        with open(fname, 'w') as ifile:
            ifile.write(header_line + '\n')
            for idx, (data, label) in enumerate(ugen_submission(unique)):
                row = '{},{}'.format(
                    data.id,
                    ','.join(['{:.12f}'.format(n) for n in all_preds[idx]])
                )
                if idx < 10:
                    print(row)
                ifile.write(row + '\n')


if __name__ == '__main__':
    # run()
    # log_reg_tuning(False)
    # naive_bayes_tuning(False)
    # rf_tuning(False)
    # avg_perc_tuning(False)

    # ada_rf_tuning(False)
    # gradient_boosting_tuning(False)

    # weighted_average(create_file=True)
    weighted_average_pred('submission', False)
