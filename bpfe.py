import os
from os.path import dirname
import random
import sys
from bpfe import load, scoring
from bpfe.config import FLAT_LABELS
from bpfe.models.perceptron_model import PerceptronModel


SEED = random.randint(1, sys.maxint)
# AMT = 1000
AMT = 400277
ITERATIONS = 20
loc = dirname(__file__) + '/results'
if not os.path.exists(loc):
    os.makedirs(loc)
filename = '{}-{}'.format(AMT, ITERATIONS)
loc += '/' + filename


def split_test_train(data, max_rows, train_percent):
    max_rows = min(len(data), max_rows)
    data = data[:max_rows]
    train_amt = int(max_rows*float(train_percent))
    train_data = data[:train_amt]
    test_data = data[train_amt:]
    return train_data, test_data


def run():
    # facility_or_department.info()
    # program_description.info()
    # job_title_description.info()
    # sub_object_description.info()
    # location_description.info()
    # function_description.info()
    # position_extra.info()
    # subfund_description.info()
    # fund_description.info()
    # object_description.info()
    # text_1.info()
    # text_2.info()
    # text_3.info()
    # text_4.info()
    # fte.info()
    # total.info()
    pm = PerceptronModel()
    # amt = sys.maxint
    random_data = load.generate_training_rows(SEED, AMT)
    train, test = split_test_train(random_data, AMT, 0.8)

    pm.train(train, test, AMT, nr_iter=ITERATIONS, seed=SEED, save_loc=loc)
    # plotting.plot_train_vs_validation(pm.scores, AMT)


def predict():
    with open('results/predictions-{}'.format(filename), 'w') as outfile:
        def write(x):
            outfile.write(x + '\n')
        _predict(write)


def _predict(output_method, num_rows=None):
    if num_rows is None:
        num_rows = sys.maxint

    pm = PerceptronModel()
    pm.load(loc)

    header = ['__'.join(i) for i in FLAT_LABELS]
    headers = []
    for i in header:
        if ' ' in i:
            i = '"{}"'.format(i)
        headers.append(i)

    output_line = ',' + ','.join(headers)
    output_method(output_line)

    i = 0
    for data in load.generate_test_rows(SEED, sys.maxint):
        v = pm.predict_proba(data)
        pred_tups = [data.id]
        for raw_key in FLAT_LABELS:
            pred_tups.append(v[raw_key])

        output_line = ','.join([str(i) for i in pred_tups])
        output_method(output_line)

        i += 1
        if i > num_rows:
            break


def predict_train():
    pm = PerceptronModel()
    pm.load(loc)

    actuals = []
    predictions = []
    for label, data in load.generate_training_rows(pm.seed, pm.amt):
        # prediction = pm.predict(data)
        actuals.append(
            PerceptronModel.label_output(label)[1:]
        )
        v = pm.predict_proba(data)
        pred_tups = []
        for raw_key in FLAT_LABELS:
            pred_tups.append(v[raw_key])
        predictions.append(pred_tups)
        # predictions.append(
        #     PerceptronModel.prediction_output(data, prediction)[1:]
        # )

    print('')
    print('mc log loss: {}'.format(
        round(scoring.multi_multi(actuals, predictions), 5)
    ))


def stats():
    pm = PerceptronModel()
    pm.load(loc)

    print('')
    header = 'Model Information'
    print(header)
    print('-'*len(header))
    print(pm)
    pm.print_scores()


if __name__ == '__main__':
    # predict()
    predict_train()
    # run()
    stats()
