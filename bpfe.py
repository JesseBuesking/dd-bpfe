

from datetime import datetime
import os
from os.path import dirname
import sys
from bpfe import load, plotting
from bpfe.config import FLAT_LABELS
from bpfe.models.perceptron_model import PerceptronModel


SEED = 1
# AMT = 4000
AMT = 400277
ITERATIONS = 40
DT = datetime.utcnow().strftime('%Y%m%d%H%M%S')
loc = dirname(__file__) + '/results'
if not os.path.exists(loc):
    os.makedirs(loc)


def split_test_train(data, max_rows, train_percent):
    max_rows = min(len(data), max_rows)
    data = data[:max_rows]
    train_amt = int(max_rows*float(train_percent))
    train_data = data[:train_amt]
    test_data = data[train_amt:]
    return train_data, test_data


def run():
    global loc
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

    loc += '/' + '{}-{}-{}-{}'.format(DT, AMT, SEED, ITERATIONS)
    pm.train(train, test, nr_iter=ITERATIONS, seed=SEED, save_loc=loc)
    plotting.plot_train_vs_validation(pm.scores, AMT)


def predict():
    with open('results/predictions-20141119091224-400277-1-40', 'w') as outfile:
        def write(x):
            outfile.write(x + '\n')
        _predict(write)


def _predict(output_method, num_rows=None):
    global loc

    if num_rows is None:
        num_rows = sys.maxint

    loc += '/' + '20141119091224-400277-1-40'
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
        prediction = pm.predict(data)
        predicted = PerceptronModel.prediction_output(data, prediction)
        output_line = ','.join([str(i) for i in predicted])
        output_method(output_line)

        i += 1
        if i > num_rows:
            break

if __name__ == '__main__':
    predict()
    # run()
