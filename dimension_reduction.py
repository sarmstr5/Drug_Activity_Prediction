from pylab import *
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import itertools


def read_in_npy(fn):
    print("Reading in {0} Data".format(fn))
    npy_file = np.load(fn)
    csc = csc_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])
    return csc

def read_in_csv(fn):
    print("Reading in Data")
    y_df = pd.read_csv(fn_y, header=None, names=['binding'])
    return y_df

def data_fn(full_run, dir):
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

def PCA_run(csc):

def main():
    # print("the number of parameters is {0}, length {1}, second entry is : \n{2}".format(largest_num_params, len(param_d[0][1]), param_d[0]))
    # need to clean data first
    # can i visualize data?
    print('to sparse')
    dir = 'data/'
    full_data_run = True
    # Reading in data that has previously been formatted
    fn_pos, fn_neg, fn_full, fn_y_pos, fn_y_neg, fn_y_full, fn_test = data_fn(full_data_run,dir)
    pos_csc = read_in_npy(fn_pos)
    neg_csc = read_in_npy(fn_neg)
    full_csc = read_in_npy(fn_full)
    test_csc = read_in_npy(fn_test)
    y_pos = read_in_csv(fn_y_pos)
    y_neg = read_in_csv(fn_y_neg)
    y_full = read_in_csv(fn_y_full)

    print('Running Model')
    # first step is to figure out what the data is like
    # i should first check columns first
    # can i see what the columns represent?
    # how do i do the bayes? do i have to break them up into ranges? i can use //
    #

if __name__ == '__main__':
    main()
