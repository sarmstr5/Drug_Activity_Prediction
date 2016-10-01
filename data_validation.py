from pylab import *
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse import csc_matrix, coo_matrix
import scipy.io
import sys
from datetime import datetime as dt
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import svm
from sklearn import decomposition
from sklearn.metrics import f1_score

N_PARAMETERS = 100000
def feature_selection(data):
    print('in feature selection')
    n_comps = np.arange(0, N_PARAMETERS, 25000)
    # n_components = N_PARAMETERS * 800 #0 to n by x
    pca = decomposition.SparsePCA()
    pca_scores = []
    for n in n_comps:
        pca.n_components = n
        pca_scores.append(cross_val_score(pca, data, score='f1').mean())
    optimal_pca_components = n_comps[np.argmax(pca_scores)]    # returns component indexed the max pca score
    plot_PCA(n_comps, pca_scores)
    return optimal_pca_components   #an integer

def plot_PCA(components, scores):
    plt.figure()
    plt.plot(components, scores)

def compute_feature_scores(X):
    pass

def evaluate_model():
    pass

def graph_ROC():
    pass

def f1_scoring(y, y_predicted):
    # F1 = 2 * (precision * recall) / (precision + recall), harmonic mean of precision and recall
    return f1_score(y, y_predicted, average='None') #returns list score [pos neg], can use weighted

def naive_bayes_model(nb):
    # I should change weights of bayes.  I want to weight the params to increase f1 score
    # May want to +multinomial+ had test_csr and gave a different output
    # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
    if(nb == 'binomial'):
        model = BernoulliNB()
    else:
        model = MultinomialNB()
        # prediction = bnb_model.predict_log_proba(test_csc.toarray()) #prediction using log probability
    return model

def support_vector_machine_model(error_penalty):
    # http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
    # http: // scikit - learn.org / stable / modules / generated / sklearn.svm.SVC.html
    svm_model = svm.SVC(kernel='linear', C=error_penalty) # C is penalty of error
    return svm_model

def kNN():
    pass

def print_results_to_csv(predictions, model):
    time = dt.now()
    hour, minute = str(time.hour), str(time.minute)
    if (len(minute) == 1):
        minute = '0' + minute
    if (len(hour) == 1):
        hour = '0' + hour
    test_output = 'test_output/'+model+"test_results" + hour + minute + '.csv'
    with open(test_output, 'w') as results:
        for y in predictions:
            results.write('{0}\n'.format(y))

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

def data_validation_score(train_csc, train_y, model, folds):
    # http: // scikit - learn.org / stable / auto_examples / plot_compare_reduction.html
    # sphx-glr-auto-examples-plot-compare-reduction-py
    # cross val score automatically runs a stratified k fold cross validation
    mean_f1 = cross_val_score(model, train_csc, train_y, cv=folds, scoring='f1_None').mean()

    # svm_model.fit(train_csc, train_classes.binding)
    # predictions = svm_model.predict(test_csc)
    return mean_f1

def bnb_validation_run(train_csc, train_y):
    folds = 10
    model = BernoulliNB()
    # for i in range(attr_start, attr_begin):


if __name__ == '__main__':
    # print("the number of parameters is {0}, length {1}, second entry is : \n{2}".format(largest_num_params, len(param_d[0][1]), param_d[0]))
    # need to clean data first
    # can i visualize data?
    print('to sparse')
    full_data_run = True
    # Reading in data that has previously been formatted
    fn_pos, fn_neg, fn_full, fn_y_pos, fn_y_neg, fn_y_full, fn_test = data_fn(full_data_run)
    binding_df, data_csc, test_csc = read_in_data(fn_full, fn_y_full, fn_test)
    train_y = binding_df.binding
    #PCA feature selection
    feature_scores = feature_selection(data_csc)

    # folds = 5
    # svm_model = support_vector_machine_model(0.5) #error penalty
    # bn_model = naive_bayes_model('binomial')
    #
    # print('Running Model')
    # mean_f1_svm = cross_val_score(svm_model, data_csc, train_y, cv=folds, scoring='f1') #.mean()
    # mean_f1_bn = cross_val_score(bn_model, data_csc, train_y, cv=folds, scoring='f1')
    # print(mean_f1_svm)
    # print(mean_f1_bn)
    #
    # log_prediction = False

    #printing off csv
    # print_results_to_csv(svm_predictions, 'svm')
    # print_results_to_csv(nb_predictions,'nb')
