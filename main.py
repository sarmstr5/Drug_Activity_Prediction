# read in the data
# partition/startify the data
# do i need to do amything about the different columns?
# test the releveance of the columns
from pylab import *
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import sklearn as sk
from sklearn.naive_bayes import GaussianNB
import scipy.sparse as sps
import matplotlib.pylab as plt
import scipy.io
import sys

def feature_selection():
    pass


def evaluate_model():
    pass


def graph_ROC():
    pass


def f1_scoring():
    pass


def naive_bayes(train_csr, train_result, test_csr):
    nb_model = GaussianNB()
    nb_model.fit(train_csr.toarray(), train_result.binding)
    prediction = nb_model.predict(test_csr.toarray())
    return prediction

def kNN():
    pass


def classify():
    pass


def read_in_data(fn_t, fn_y, fn_x):
    print("Reading in Data")
    npy_file = np.load(fn_t)
    csr_train = csr_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])
    npy_file = np.load(fn_x)
    csr_test = csr_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])

    y_df = pd.read_csv(fn_y, header=None, names=['binding'])
    return y_df, csr_train, csr_test

def plot_coo(csr):
    if not isinstance(csr, coo_matrix):
        m = coo_matrix(csr)
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
def plot_csr(csr):
    plt.spy(csr, aspect='auto')
    plt.show()


def data_fn(full_run):
    if(full_run == True):
        t_fn = 'train_drugs.npz'
        y_fn = 'train_drugs_bindings'
        x_fn = 'test_drugs.npz'
    else:
        t_fn = 'train_drugs_short.npz'
        y_fn = 'train_drugs_bindings_short'
        x_fn = 'test_drugs_short.npz'
    return t_fn, y_fn, x_fn

if __name__ == '__main__':
    # print("the number of parameters is {0}, length {1}, second entry is : \n{2}".format(largest_num_params, len(param_d[0][1]), param_d[0]))
    # need to clean data first
    # can i visualize data?
    print('to sparse')
    full_data_run = True
    # Reading in data that has previously been formatted
    fn_t, fn_y, fn_x = data_fn(full_data_run)    #train, bindings, test filenames
    binding_df, data_csr, test_csr = read_in_data(fn_t, fn_y, fn_x)
    print(binding_df.binding)
    print(data_csr)
    print(test_csr)

    print('Running Model')
    model = naive_bayes(data_csr, binding_df, test_csr)
    print(model)
    # first step is to figure out what the data is like
    # i should first check columns first
    # can i see what the columns represent?
    # how do i do the bayes? do i have to break them up into ranges? i can use //
    #
