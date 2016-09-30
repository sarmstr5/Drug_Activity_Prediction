# read in the data
# partition/startify the data
# do i need to do amything about the different columns?
# test the releveance of the columns
from pylab import *
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
import sklearn as sk
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import scipy.sparse as sps
import matplotlib.pylab as plt
import scipy.io
import sys
from datetime import datetime as dt

def feature_selection():
    pass


def evaluate_model():
    pass


def graph_ROC():
    pass


def f1_scoring():
    pass


def naive_bayes(train_csc, train_result, test_csr):
    nb_model = BernoulliNB()
    nb_model.fit(train_csc.toarray(), train_result.binding)
    prediction = nb_model.predict(test_csc.toarray())
    return prediction

def kNN():
    pass

def print_results_to_csv(predictions):
    time = dt.now()
    hour, minute = str(time.hour), str(time.minute)
    if (len(minute) == 1):
        minute = '0' + minute
    if (len(hour) == 1):
        hour = '0' + hour
    test_output = "test_results" + hour + minute + '.csv'
    with open(test_output, 'w') as results:
        for y in predictions:
            results.write('{0}\n'.format(y.test_sentiment))

def classify():
    pass


def read_in_data(fn_t, fn_y, fn_x):
    print("Reading in Data")
    npy_file = np.load(fn_t)
    csc_train = csc_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])
    npy_file = np.load(fn_x)
    csc_test = csc_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])

    y_df = pd.read_csv(fn_y, header=None, names=['binding'])
    return y_df, csc_train, csc_test

def plot_coo(csc):
    if not isinstance(csc, coo_matrix):
        m = coo_matrix(csc)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.show()

#####################################needs work
def plot_csc(csr):
    plt.spy(csc, aspect='auto')
    plt.show()


def data_fn(full_run):
    dir = 'data/'
    if(full_run == True):
        fn_pos = dir+'train_drugs_pos.npz'
        fn_neg = dir+'train_drugs_neg.npz'
        fn_full = dir+'train_drugs_full.npz'
        fn_y_pos = dir+'train_drugs_bindings_pos'
        fn_y_neg = dir+'train_drugs_bindings_neg'
        fn_y_full = dir+'train_drugs_bindings_full'
        fn_test = dir+'test_drugs.npz'
    else:
        fn_pos = dir+'train_drugs_short_pos.npz'
        fn_neg = dir+'train_drugs_short_neg.npz'
        fn_full = dir+'train_drugs_short_full.npz'
        fn_y_pos = dir+'train_drugs_bindings_short_pos'
        fn_y_neg = dir+'train_drugs_bindings_short_neg'
        fn_y_full = dir+'train_drugs_bindings_short_full'
        fn_test = dir+'test_drugs_short.npz'
    return fn_pos, fn_neg, fn_full, fn_y_pos, fn_y_neg, fn_y_full, fn_test

if __name__ == '__main__':
    # print("the number of parameters is {0}, length {1}, second entry is : \n{2}".format(largest_num_params, len(param_d[0][1]), param_d[0]))
    # need to clean data first
    # can i visualize data?
    print('to sparse')
    full_data_run = True
    # Reading in data that has previously been formatted
    fn_pos, fn_neg, fn_full, fn_y_pos, fn_y_neg, fn_y_full, fn_test = data_fn(full_data_run)
    binding_df, data_csc, test_csc = read_in_data(fn_full, fn_y_full, fn_test)
    # print(binding_df.binding)
    # print(data_csc)
    # print(test_csc)
    print('Running Model')
    class_predictions = naive_bayes(data_csc, binding_df, test_csc)
    print_results_to_csv(class_predictions)
    # print(model)
    # first step is to figure out what the data is like
    # i should first check columns first
    # can i see what the columns represent?
    # how do i do the bayes? do i have to break them up into ranges? i can use //
    #
