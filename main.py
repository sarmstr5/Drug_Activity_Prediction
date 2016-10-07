from pylab import *
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from datetime import datetime as dt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import svm
from sklearn import decomposition
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

N_PARAMETERS = 100000


# F1 = 2 * (precision * recall) / (precision + recall), harmonic mean of precision and recall
def f1_scoring(y, y_predicted):
    return f1_score(y, y_predicted, average='None')  # returns list score [pos neg], can use weighted


def print_results_to_csv(predictions):
    print('Printing Results')
    time = dt.now()
    hour, minute = str(time.hour), str(time.minute)
    if (len(minute) == 1):
        minute = '0' + minute
    if (len(hour) == 1):
        hour = '0' + hour
    test_output = 'test_output/' + "test_results" + hour + minute + '.csv'

    with open(test_output, 'w') as results:
        for y in predictions:
            results.write('{0}\n'.format(y))
    test_output = 'test_output/' + "test_results_with_probabilities" + hour + minute + '.csv'

def classify():
    pass


def read_in_data(fn_t, fn_y, fn_x):
    npy_file = np.load(fn_t)
    csc_train = csc_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])

    npy_file = np.load(fn_x)
    csc_test = csc_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])

    y_df = pd.read_csv(fn_y, header=None, names=['binding'])
    return y_df, csc_train, csc_test


def data_fn(get_truncated_data):
    dir = 'data/'
    if (get_truncated_data):
        fn_pos = dir + 'train_drugs_short_pos.npz'
        fn_neg = dir + 'train_drugs_short_neg.npz'
        fn_full = dir + 'train_drugs_short_full.npz'
        fn_y_pos = dir + 'train_drugs_bindings_short_pos'
        fn_y_neg = dir + 'train_drugs_bindings_short_neg'
        fn_y_full = dir + 'train_drugs_bindings_short_full'
        fn_test = dir + 'test_drugs_short.npz'
    else:
        fn_pos = dir + 'train_drugs_pos.npz'
        fn_neg = dir + 'train_drugs_neg.npz'
        fn_full = dir + 'train_drugs_full.npz'
        fn_y_pos = dir + 'train_drugs_bindings_pos'
        fn_y_neg = dir + 'train_drugs_bindings_neg'
        fn_y_full = dir + 'train_drugs_bindings_full'
        fn_test = dir + 'test_drugs.npz'
    return fn_pos, fn_neg, fn_full, fn_y_pos, fn_y_neg, fn_y_full, fn_test


def get_processed_data(get_neg_data, get_trunc_data):
    print('Processing Data')
    fn_pos, fn_neg, fn_full, fn_y_pos, fn_y_neg, fn_y_full, fn_test = data_fn(
        get_trunc_data)  # filenames, data trunc or not
    y_df, x_csc, t_csc = read_in_data(fn_full, fn_y_full, fn_test)

    if get_neg_data:
        partial_y_df, partial_x_csc, partial_t_csc = read_in_data(fn_neg, fn_y_neg, fn_test)
    else:
        partial_y_df, partial_x_csc, partial_t_csc = read_in_data(fn_pos, fn_y_pos, fn_test)

    return partial_y_df, partial_x_csc, y_df, x_csc, t_csc  # the partial test csc is redundant


def fit_svd(n_params, x, y):
    svd_m = decomposition.TruncatedSVD(algorithm='randomized', n_components=n_params, n_iter=7)
    svd_model = svd_m.fit(x, y)
    return svd_model


def predict_test_set_svd(n_params, test, x, y, x_pos, y_pos, error):
    print('Performing Dimension Reduction with SVD and classifying with SVM')
    # Create Models
    svd_model = fit_svd(n_params, x, y)
    svm_m = svm.SVC(kernel='linear', C=error, probability=True)  # C is penalty of error
    # run_combined_validation_runs(data_csc, y, chi_n_params, svd_n_params, folds, error, runs)
    # SVM/SVD with positive set
    # first get fit truncated SVD on positive data set, trim full and test parameters
    x_svd = svd_model.transform(x)
    test_svd = svd_model.transform(test)
    # now train svm on tranformed data sets
    svm_model = svm_m.fit(x_svd, y)
    # predict test set
    svd_svm_predictions = svm_model.predict(test_svd)
    svd_svm_test_prob = svm_model.predict_proba(test_svd)
    svd_svm_train_prob = svm_model.predict_proba(x_svd)
    return svd_svm_predictions, svd_svm_test_prob, svd_svm_train_prob


def predict_test_set_chi2(n_params, test, x, y):
    print('Performing Feature Selection with Chi2 and classifying with Naive Bayes')
    # use NB/CHI2 with full set
    # first get fit chi2 on  data set, trim test parameters
    ch2_m = SelectKBest(chi2, k=n_params)
    nb_m = BernoulliNB()
    ch2_model = ch2_m.fit(x, y)
    x_chi2 = ch2_model.transform(x)
    test_chi2 = ch2_model.transform(test)

    # now train svm on tranformed data sets
    nb_model = nb_m.fit(x_chi2, y)

    # predict test set
    chi2_nb_predictions = nb_model.predict(test_chi2)
    chi2_nb_train_prob = nb_model.predict_proba(x_chi2)
    chi2_nb_test_prob = nb_model.predict_proba(test_chi2)

    return chi2_nb_predictions, chi2_nb_test_prob, chi2_nb_train_prob


def merge_probabilities(svm_test_prob, nb_test_prob, svm_x_prob, nb_x_prob):
    pred_test_prob_list = list(zip(svm_test_prob[:, 0], svm_test_prob[:, 1], nb_test_prob[:, 0], nb_test_prob[:, 1]))
    pred_train_prob_list = list(zip(svm_x_prob[:, 0], svm_x_prob[:, 1], nb_x_prob[:, 0], nb_x_prob[:, 1]))
    test_df = pd.DataFrame(data=pred_test_prob_list)
    train_df = pd.DataFrame(data=pred_train_prob_list)
    knn_df = pd.DataFrame(pred_test_prob_list, columns=['svm pos', 'svm neg', 'nb neg', 'nb pos'])
    return knn_df, train_df, test_df


def kNN(svm_test_prob, nb_test_prob, svm_x_prob, nb_x_prob, x_pos, y_pos, x, y, test):
    print('Performing Dimension Reduction with SVD and classifying with KNN')
    neigh = KNeighborsClassifier(n_neighbors=3)
    svd_model = fit_svd(23, x_pos, y_pos)
    x_svd = svd_model.transform(x)
    test_svd = svd_model.transform(test)
    neigh.fit(x_svd, y)
    knn_df, train_prob_df, test_prob_df = merge_probabilities(svm_test_prob, nb_test_prob, svm_x_prob, nb_x_prob)
    val_list = cross_val_score(neigh, x, y, cv=10, scoring='f1')
    print(val_list)
    y_knn = neigh.predict(test_svd)
    return y_knn, knn_df


def merge_predictions(y_svm, y_nb, y_knn):
    print("Merging Predictions")
    i = 0
    joined_predictions = []
    non_binary_predictions = []
    while i < len(y_svm):
        y1 = y_svm[i]
        y2 = y_nb[i]
        y3 = y_knn[i]
        sum = y1 + y2 + y3
        if sum > 1:
            joined_predictions.append(1)
        else:
            joined_predictions.append(0)
        non_binary_predictions.append(sum)
        i += 1
    return joined_predictions, non_binary_predictions


def main():
    # print("the number of parameters is {0}, length {1}, second entry is : \n{2}".format(largest_num_params, len(param_d[0][1]), param_d[0]))
    # need to clean data first
    # can i visualize data?
    time = dt.now()
    print('Starting Test Prediction Run at {}'.format(time))
    print('Reading In Data')
    # data file with choice of negative or positive dataset
    # Reading in data that has previously been formatted
    get_neg_data = False  # get neg | pos data
    get_truncated_data = False
    partial_y_df, partial_data_csc, y_df, data_csc, test_csc = get_processed_data(get_neg_data, get_truncated_data)
    y, y_pos = y_df.binding, partial_y_df.binding

    print(y_pos)
    # PCA feature selection
    # SVM performs best with SVD reduction, at error rate = .93 with 22 params
    # BN performs best with chi2, at 347-355
    chi_n_params = 348
    svd_n_params = 22
    error = 0.93
    # Get predictions
    svd_svm_predictions, svm_test_proba, svm_x_proba = predict_test_set_svd(svd_n_params, test_csc, data_csc, y,
                                                                            partial_data_csc, y_pos, error)
    chi2_nb_predictions, nb_test_proba, nb_x_proba = predict_test_set_chi2(chi_n_params, test_csc, data_csc, y)
    knn_predictions, knn_df = kNN(svm_test_proba, nb_test_proba, svm_x_proba, nb_x_proba, partial_data_csc, y_pos,
                                  data_csc, y, test_csc)
    # Combine Predictions
    merged_p_list, sum_vote_list = merge_predictions(svd_svm_predictions, chi2_nb_predictions, knn_predictions)
    # Print Results
    print_results_to_csv(sum_vote_list)
    time = dt.now()
    print('Ending Test Run at {}'.format(time))


if __name__ == '__main__':
    main()
