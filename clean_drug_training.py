from pylab import *
import numpy as np
from scipy.sparse import csr_matrix
import itertools

NUMBER_OF_PARAMETERS = 100000


def training_to_list(file):
    print("converting training data to list")
    max_param_count = 0
    pos_param_list = []
    neg_param_list = []
    pos_bind_list = []
    neg_bind_list = []
    with open(file, 'r') as csv:
        param_num = 0
        for row in csv:
            row_list = row.split('\t')
            if row_list[0] == 0:
                neg_bind_list.append(int(row_list[0]))
                neg_param_list.append(row_list[1])      #list of strings that represent column location of bools
            else:
                pos_bind_list.append(int(row_list[0]))
                pos_param_list.append(row_list[1])      #list of strings that represent column location of bools
    return neg_bind_list, neg_param_list, pos_bind_list, pos_param_list

def test_to_list(file):
    print("converting training data to list")
    test_list = []
    with open(file, 'r') as csv:
        for row in csv:
            test_list.append(row)
    return test_list

def jagged_list_to_csr(jagged_lists):
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
    return csr

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
def write_csr_to_disk(csr, fn,):
    print('file {0} to disk'.format(fn))
    np.savez(fn, data=csr.data, indices=csr.indices, indptr=csr.indptr, shape=csr.shape)

def write_list_to_disk(list, fn):
    with open(fn, 'w') as csv:
        for row in list:
            csv.write("{0}\n".format(row))

def split_classes(train_csr, y):
    train_df =

def main():
    print('Starting File')
    full_run = False
    if(full_run == True):
        filename_pos = 'train_drugs_pos'
        filename_neg = 'train_drugs_neg'
        training_file = 'train_drugs.txt'
        filename_y_pos = 'train_drugs_bindings_pos'
        filename_y_neg = 'train_drugs_bindings_neg'
        filename_test = 'test_drugs'
        test_file = 'test.txt'
    else:
        filename_pos = 'train_drugs_short_pos'
        filename_neg = 'train_drugs_short_neg'
        training_file = 'train_drugs_short.txt'
        filename_y_pos = 'train_drugs_bindings_short_pos'
        filename_y_neg = 'train_drugs_bindings_short_neg'
        filename_test = 'test_drugs_short'
        test_file = 'test_short.txt'

    # convert training and test data to lists
    neg_binding_vector, neg_jagged_data_list, pos_binding_vector, pos_jagged_data_list = training_to_list(training_file)
    jagged_test_list = test_to_list(test_file)

    # convert lists to csrs
    neg_training_csr = jagged_list_to_csr(neg_jagged_data_list)
    pos_training_csr = jagged_list_to_csr(pos_jagged_data_list)
    test_csr = jagged_list_to_csr(jagged_test_list)

    # write cleaned data to file
    write_csr_to_disk(neg_training_csr, filename_neg)
    write_csr_to_disk(pos_training_csr, filename_pos)
    write_csr_to_disk(test_csr, filename_test)
    write_list_to_disk(neg_binding_vector, filename_y_neg )
    write_list_to_disk(pos_binding_vector, filename_y_pos )

if __name__ == '__main__':
    main()
