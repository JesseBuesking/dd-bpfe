from os.path import dirname
import pickle
import re
import math
from nltk import NaiveBayesClassifier, SklearnClassifier, accuracy
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
from bpfe.config import Settings, ChunkSettings
from bpfe.entities import Data
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


sbs = SnowballStemmer('english')


def bow(string):
    # return util.sentence_splitter(string)
    for word in re.findall(
            r'GRADE=k\|k|GRADE=k\|\d+|GRADE=\d+\|\d+|\w+|\d+|\.', string):
        yield word


def get_x_y(name, generator):
    x, y = [], []
    uq = set()
    uqsbs = set()
    for data, label in generator:
        row = []
        for attr in Data.text_attributes:
            value = data.cleaned[attr + '-mapped']
            # value = getattr(data, attr)
            b_o_w = []
            for i in bow(value):
                uq.add(i)
                i = sbs.stem(i)
                uqsbs.add(i)
                b_o_w.append(i)
            # need 1 before and 1 after to support 3-grams
            # e.g. board ENDHERE BEGHERE something
            #     |---------------------|
            row += ['BEGHERE'] + b_o_w + ['ENDHERE']

        x.append(' '.join(row))
        if label is not None:
            y.append(label.function)
        else:
            y = None
    print('uq {}: {}'.format(name, len(uq)))
    print('uqsbs {}: {}'.format(name, len(uqsbs)))
    return x, y


def ll_score(classifier, klasses, tX, tY):
    predictions = classifier.predict_proba(tX)
    lls = 0
    for actual, expected in zip(predictions, tY):
        tmp = np.zeros(len(klasses))
        tmp[klasses.index(expected)] = 1
        ll = log_loss(tmp, actual)
        lls += ll

    return lls / float(len(predictions))


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

    train = [i for i in ugen_train()]
    train_x, train_y = get_x_y('train', train)

    num = 20000
    tx, ty = train_x[:num], train_y[:num]

    best = (sys.maxint, 0, '')
    scores = []
    for vect__min_df in [3]:
        for vect__binary in [False]:
            for classifier__max_depth in [9, 12, 15]:
                for classifier__max_features in [200, 400]:
                    for classifier__criterion in ['gini', 'entropy']:
                        pstring = ', '.join([
                            'vect__min_df: {}'.format(vect__min_df),
                            'vect__binary: {}'.format(vect__binary),
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
                             CountVectorizer(
                                 stop_words='english',
                                 ngram_range=(1, 3),
                                 min_df=vect__min_df,
                                 binary=vect__binary
                             )),
                            ('chi2',
                             SelectKBest(
                                 chi2,
                                 k=1000
                             )),
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

    with open(fname, 'wb') as ifile:
        pickle.dump(scores, ifile, -1)


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
    for vect__min_df in [3]:
        for vect__binary in [False]:
            for classifier__n_estimators in [50, 100]:
                for classifier__learning_rate in [.1, 1, 10]:
                    pstring = ', '.join([
                        'vect__min_df: {}'.format(vect__min_df),
                        'vect__binary: {}'.format(vect__binary),
                        'classifier__n_estimators: {}'.format(
                            classifier__n_estimators),
                        'classifier__learning_rate: {}'.format(
                            classifier__learning_rate)
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
                        ('chi2',
                         SelectKBest(
                             chi2,
                             k=2000
                         )),
                        ('todense', DenseTransformer()),
                        ('classifier',
                         AdaBoostClassifier(
                             base_estimator=ExtraTreesClassifier(),
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

    train = [i for i in ugen_train()]
    train_x, train_y = get_x_y('train', train)

    num = 20000
    tx, ty = train_x[:num], train_y[:num]

    best = (sys.maxint, 0, '')
    scores = []
    for vect__min_df in [3]:
        for vect__binary in [True, False]:
            for classifier__alpha in [.1, .5, 1]:
                for classifier__fit_prior in [False]:
                    pstring = ', '.join([
                        'vect__min_df: {}'.format(vect__min_df),
                        'vect__binary: {}'.format(vect__binary),
                        'classifier__alpha: {}'.format(
                            classifier__alpha),
                        'classifier__fit_prior: {}'.format(
                            classifier__fit_prior)
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

    train = [i for i in ugen_train()]
    train_x, train_y = get_x_y('train', train)

    num = 20000
    tx, ty = train_x[:num], train_y[:num]

    best = (sys.maxint, 0, '')
    scores = []
    for vect__min_df in [3]:
        for vect__binary in [False]:
            for tfidf__norm in [None, 'l2']:
                for classifier__C in [.1, 1]:
                    for classifier__penalty in ['l1']:
                        for classifier__fit_intercept in [False, True]:
                            pstring = ', '.join([
                                'vect__min_df: {}'.format(vect__min_df),
                                'vect__binary: {}'.format(vect__binary),
                                'tfidf__norm: {}'.format(tfidf__norm),
                                'classifier__C: {}'.format(classifier__C),
                                'classifier__penalty: {}'.format(
                                    classifier__penalty),
                                'classifier__fit_intercept: {}'.format(
                                    classifier__fit_intercept),
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
            train_x2, test_x2, sys.maxint, nr_iter=50, seed=1, save_loc=None)

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


def weighted_average():
    log_reg_classifier = Pipeline([
        ('vect',
         CountVectorizer(
             stop_words='english',
             ngram_range=(1, 3),
             min_df=3,
             binary=False
         )),
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
    ])

    nb_classifier = Pipeline([
        ('vect',
         CountVectorizer(
             stop_words='english',
             ngram_range=(1, 3),
             min_df=3,
             binary=True
         )),
        ('classifier',
         MultinomialNB(
             alpha=.5,
             fit_prior=False
         )),
    ])

    rf_classifier = Pipeline([
        ('vect',
         CountVectorizer(
             stop_words='english',
             ngram_range=(1, 3),
             min_df=3,
             binary=False
         )),
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

    num = 20000
    train = [i for i in ugen_train()][:num]
    tx, ty = get_x_y('train', train)

    for train_ix, test_ix in cross_validation.KFold(len(tx), 5):
        train_x2, train_y2 = [], []
        test_x2, test_y2 = [], []
        for tix in train_ix:
            train_x2.append(tx[tix])
            train_y2.append(ty[tix])
        for tix in test_ix:
            test_x2.append(tx[tix])
            test_y2.append(ty[tix])
        log_reg_classifier.fit(train_x2, train_y2)
        nb_classifier.fit(train_x2, train_y2)
        rf_classifier.fit(train_x2, train_y2)

        lr_c = list(log_reg_classifier.named_steps['classifier'].classes_)
        nb_c = list(nb_classifier.named_steps['classifier'].classes_)
        rf_c = list(rf_classifier.named_steps['classifier'].classes_)

        lr_w = np.zeros(len(lr_c)) + .33
        nb_w = np.zeros(len(lr_c)) + .33
        rf_w = np.zeros(len(lr_c)) + .33

        preds = []
        actuals = []
        for epoch in range(20):
            lls = 0
            for data, label in zip(test_x2, test_y2):
                data = [data]
                tmp = np.zeros(len(lr_c))
                tmp[lr_c.index(label)] = 1.0
                label = tmp

                lr_p = log_reg_classifier.predict_proba(data)[0]
                nb_p = nb_classifier.predict_proba(data)[0]
                rf_p = rf_classifier.predict_proba(data)[0]

                def rearrange(klasses, preds, pred_klasses):
                    tmp = np.zeros(len(klasses))
                    for c in klasses:
                        pidx = pred_klasses.index(c)
                        tmp[pidx] = preds[pidx]
                    return tmp

                pred = ((lr_w * lr_p) + (nb_w * nb_p) + (rf_w * rf_p)) / 3
                preds.append(pred)
                actuals.append(label)

                nb_p = rearrange(lr_c, nb_p, nb_c)
                rf_p = rearrange(lr_c, rf_p, rf_c)

                lr_close = 1 - np.absolute(label - (lr_w * lr_p))
                nb_close = 1 - np.absolute(label - (nb_w * nb_p))
                rf_close = 1 - np.absolute(label - (rf_w * rf_p))

                conc = np.array([lr_close, nb_close, rf_close])
                conc = (conc - conc.min(axis=0)) / \
                       (conc.max(axis=0) - conc.min(axis=0))
                # conc = conc / conc.sum(axis=0)
                conc[conc == 0] = 1e-6

                lr_change = conc[0]
                nb_change = conc[1]
                rf_change = conc[2]

                ep = epoch + 1
                lr_w = ((ep * lr_w) + lr_change) / (ep + 1)
                nb_w = ((ep * nb_w) + nb_change) / (ep + 1)
                rf_w = ((ep * rf_w) + rf_change) / (ep + 1)

            mmll = scoring.multi_multi_log_loss(
                np.array(preds),
                np.array(actuals),
                np.array([range(actuals[0].shape[0])])
            )
            print('mmll: {:.4f}'.format(mmll))

        print('lr_w', lr_w)
        print('nb_w', nb_w)
        print('rf_w', rf_w)


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


if __name__ == '__main__':
    # run()
    # log_reg_tuning(False)
    # naive_bayes_tuning(False)
    # rf_tuning(False)
    avg_perc_tuning(False)

    # ada_rf_tuning(False)
    # gradient_boosting_tuning(False)

    # weighted_average()
