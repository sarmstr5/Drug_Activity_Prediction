from pylab import *
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import itertools
from sklearn import preprocessing
NUMBER_OF_PARAMETERS = 100000


def training_to_list(file):
    print("converting training data to list")
    file = 'data/'+file
    max_param_count = 0
    pos_param_list = []
    neg_param_list = []
    full_param_list = []
    pos_bind_list = []
    neg_bind_list = []
    full_bind_list = []
    with open(file, 'r') as csv:
        param_num = 0
        for row in csv:
            row_list = row.split('\t')
            if row_list[0] == 0:
                neg_bind_list.append(int(row_list[0]))
                neg_param_list.append(row_list[1])      #list of strings that represent column location of bools
            else:
                pos_bind_list.append(int(row_list[0]))
                pos_param_list.append(row_list[1])

            full_bind_list.append(int(row_list[0]))
            full_param_list.append(row_list[1])
    return neg_bind_list, neg_param_list, pos_bind_list, pos_param_list, full_bind_list, full_param_list

def test_to_list(file):
    print("converting training data to list")
    test_list = []
    file = 'data/'+file
    with open(file, 'r') as csv:
        for row in csv:
            test_list.append(row)
    return test_list

def jagged_list_to_csc(jagged_lists):
    print("in jagged method")
    i = 0
    param_lists = []
    row_lists = []
    value_lists = []
    for list in jagged_lists:
        list = list.strip()
        params = [int(n) for n in list.split(' ')] # list with string to list of nums, index is +1
        row = [i] * len(params)
        value = [True] * len(params)
        param_lists.append(params)
        row_lists.append(row)
        value_lists.append(value)
        i += 1
    csr = create_csr(param_lists, row_lists, value_lists, i)
    return csc_matrix(csr)

def create_csr(param_lists, row_lists, value_lists, num_rows):
    # in create CSR
    flattened_params = np.array(list(itertools.chain.from_iterable(param_lists)))
    flattened_rows = np.array(list(itertools.chain.from_iterable(row_lists)))
    flattened_values = np.array(list(itertools.chain.from_iterable(value_lists)))
    sparse_csr = csr_matrix((flattened_values, (flattened_rows, flattened_params)), #three 1D lists
                            shape=(num_rows, NUMBER_OF_PARAMETERS+1), #size of matrix, +1 bc of indexing,
                            dtype=np.bool)  # creates a boolean compressed sparse row matrix

    return sparse_csr
# t = training data, x = test data
def write_csc_to_disk(csc, fn,):
    print('file {0} to disk'.format(fn))
    fn = 'data/'+fn
    np.savez(fn, data=csc.data, indices=csc.indices, indptr=csc.indptr, shape=csc.shape)

def write_list_to_disk(list, fn):
    fn = 'data/'+fn
    with open(fn, 'w') as csv:
        for row in list:
            csv.write("{0}\n".format(row))

def main():
    print('Starting File')
    full_run = True
    if(full_run == True):
        filename_pos = 'train_drugs_pos'
        filename_neg = 'train_drugs_neg'
        filename_full = 'train_drugs_full'
        training_file = 'train_drugs.txt'
        filename_y_pos = 'train_drugs_bindings_pos'
        filename_y_neg = 'train_drugs_bindings_neg'
        filename_y_full = 'train_drugs_bindings_full'
        filename_test = 'test_drugs'
        test_file = 'test.txt'
    else:
        filename_pos = 'train_drugs_short_pos'
        filename_neg = 'train_drugs_short_neg'
        filename_full = 'train_drugs_short_full'
        training_file = 'train_drugs_short.txt'
        filename_y_pos = 'train_drugs_bindings_short_pos'
        filename_y_neg = 'train_drugs_bindings_short_neg'
        filename_y_full = 'train_drugs_bindings_short_full'
        filename_test = 'test_drugs_short'
        test_file = 'test_short.txt'

    # convert training and test data to lists
    neg_binding_vector, neg_jagged_list, pos_binding_vector, pos_jagged_list, full_binding_vector, full_jagged_list \
        = training_to_list(training_file)
    jagged_test_list = test_to_list(test_file)
    # convert lists to cscs
    neg_training_csc = jagged_list_to_csc(neg_jagged_list)
    pos_training_csc = jagged_list_to_csc(pos_jagged_list)
    full_training_csc = jagged_list_to_csc(full_jagged_list)
    test_csc = jagged_list_to_csc(jagged_test_list)

    # write cleaned data to file
    write_csc_to_disk(neg_training_csc, filename_neg)
    write_csc_to_disk(pos_training_csc, filename_pos)
    write_csc_to_disk(full_training_csc, filename_full)
    write_csc_to_disk(test_csc, filename_test)
    write_list_to_disk(neg_binding_vector, filename_y_neg )
    write_list_to_disk(pos_binding_vector, filename_y_pos )
    write_list_to_disk(full_binding_vector, filename_y_full )

if __name__ == '__main__':
    main()
