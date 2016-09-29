from pylab import *
import numpy as np
from scipy.sparse import csr_matrix
import itertools

NUMBER_OF_PARAMETERS = 100000


def training_to_list(file):
    print("converting training data to list")
    max_param_count = 0
    param_list = []
    bind_list = []
    with open(file, 'r') as csv:
        param_num = 0
        for row in csv:
            row_list = row.split('\t')
            # for column in row_list[1]:
            bind_list.append(int(row_list[0]))
            param_list.append(row_list[1])      #list of strings that represent column location of bools
    return bind_list, param_list

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
def write_file_to_disk(csr_t, y, csr_x,fn_t, fn_y, fn_x):
    print('file to disk')
    np.savez(fn_t, data=csr_t.data, indices=csr_t.indices, indptr=csr_t.indptr, shape=csr_t.shape)
    np.savez(fn_x, data=csr_x.data, indices=csr_x.indices, indptr=csr_x.indptr, shape=csr_x.shape)
    with open(fn_y, 'w') as csv:
        for binding in y:
            csv.write("{0}\n".format(binding))

def main():
    print('Starting File')
    full_run = False
    if(full_run == True):
        filename = 'train_drugs'
        training_file = 'train_drugs.txt'
        filename_y = 'train_drugs_bindings'
        filename_test = 'test_drugs'
        test_file = 'test.txt'
    else:
        filename = 'train_drugs_short'
        training_file = 'train_drugs_short.txt'
        filename_y = 'train_drugs_bindings_short'
        filename_test = 'test_drugs_short'
        test_file = 'test_short.txt'

    # convert training and test data to lists
    binding_vector, jagged_data_list = training_to_list(training_file)
    jagged_test_list = test_to_list(test_file)

    # convert lists to csrs
    training_csr = jagged_list_to_csr(jagged_data_list)
    test_csr = jagged_list_to_csr(jagged_test_list)

    # write cleaned data to file
    write_file_to_disk(training_csr, binding_vector, test_csr, filename, filename_y, filename_test)

if __name__ == '__main__':
    main()
