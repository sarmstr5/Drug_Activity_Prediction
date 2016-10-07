from pylab import *
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse import csc_matrix, coo_matrix
import scipy.io
import sys
from datetime import datetime as dt
from itertools import cycle
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import svm
from sklearn import decomposition
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

N_PARAMETERS = 100000


def plot_PCA(components, scores):
    plt.figure()
    plt.plot(components, scores)

def compute_feature_scores(X):
    pass
def get_time():
    time = dt.now()
    hour, minute = str(time.hour), str(time.minute)
    if (len(minute) == 1):
        minute = '0' + minute
    if (len(hour) == 1):
        hour = '0' + hour
    time = hour + minute
    return time

def evaluate_knn_csv(scores, neighbors, folds, chi2_n_params, runs):
    time = get_time()
    test_output = 'cross_validation/' + "knn_results"+'.txt'
    i = 0
    with open(test_output, 'a') as csv:
        for score in scores:
            print(score)
            csv.write('{:0.4f}\t'.format(float(score)))
            csv.write("\t{0}\t{1}\t{2}\t{3}\t{4}\n".format(runs[i], neighbors, folds, "SVM","full set"))
            i += 1

def evaluate_models(svd_data, chi2_data, y, folds, error_penalty, chi_size, svd_size,runtype):
    svm_model = support_vector_machine_model(error_penalty)  # error penalty of margin break
    bn_model = naive_bayes_model('binomial')
    scores = []
    scores.append(cross_val_score(svm_model, svd_data, y, cv=folds, scoring='f1').mean())
    scores.append(cross_val_score(bn_model, svd_data, y, cv=folds, scoring='f1').mean())
    scores.append(cross_val_score(svm_model, chi2_data, y, cv=folds, scoring='f1').mean())
    scores.append(cross_val_score(bn_model, chi2_data, y, cv=folds, scoring='f1').mean())

    time = dt.now()
    hour, minute = str(time.hour), str(time.minute)
    if (len(minute) == 1):
        minute = '0' + minute
    if (len(hour) == 1):
        hour = '0' + hour
    time = hour + minute
    test_output = 'test_output/' + "cross_validation_results"+'.txt'

    i = 0
    with open(test_output, 'a') as csv:
        for score in scores:
            if(i==1):
                csv.write('{:0.4f}\t\t\t'.format(float(score)))
            csv.write('{:0.4f}\t\t'.format(float(score)))
        csv.write("\t{0}\t\t{1}\t\t{2}\t\t{3}\t\t{4}\t\t{5}\n".format(time,error_penalty,folds,chi_size, svd_size, runtype))

def graph_ROC(x, y):
    # Run classifier with cross-validation and plot ROC curves
    # using code from scikit library
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    cv = StratifiedKFold(n_splits=10)
    random_state = np.random.RandomState(0)
    svm_m = svm.SVC(kernel='linear', C=error, probability=True, random_state=random_state)  # C is penalty of error
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    i = 0
    for (train_indices, test_indices), color in zip(cv.split(x, y), colors): 
        probas = svm_m.fit(x[train_indices], y[train_indices]).predict_proba(x[test_indices])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test_indices], probas[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(x, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)     # compute area
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(bbox_to_anchor=(1.05,1), loc=2)
    plt.show()

def f1_scoring(y, y_predicted):
    # F1 = 2 * (precision * recall) / (precision + recall), harmonic mean of precision and recall
    return f1_score(y, y_predicted, average='None')  # returns list score [pos neg], can use weighted

def naive_bayes_model(nb):
    # I should change weights of bayes.  I want to weight the params to increase f1 score
    # May want to +multinomial+ had test_csr and gave a different output
    # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
    if (nb == 'binomial'):
        model = BernoulliNB()
    else:
        model = MultinomialNB()
        # prediction = bnb_model.predict_log_proba(test_csc.toarray()) #prediction using log probability
    return model

def support_vector_machine_model(error_penalty):
    # http://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
    # http: // scikit - learn.org / stable / modules / generated / sklearn.svm.SVC.html
    svm_model = svm.SVC(kernel='linear', C=error_penalty)  # C is penalty of error
    return svm_model

def print_results_to_csv(predictions, df):
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
    df.to_csv(path=test_output)
    # with open(test_output, 'w') as results:
    #     results.write('pred\tsvm_p\tnb_p\n')
    #     i = 0
    #     for y in predictions:
    #         results.write('{}\t'.format(y))
    #         results.write('{:0.4f}\t{:0.4f}\t'.format(float(svm_p[i, 0]), float(svm_p[i, 1])))
    #         results.write('{:0.4f}\t{:0.4f}\n'.format(float(nb_p[i, 0]), float(nb_p[i, 1])))
    #         i += 1

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
def plot_csc(csc):
    plt.spy(csc, aspect='auto')
    plt.show()

def data_fn(get_truncated_data):
    dir = 'data/'
    if(get_truncated_data):
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

def get_processed_data(get_neg_data, get_trunc_data):
    fn_pos, fn_neg, fn_full, fn_y_pos, fn_y_neg, fn_y_full, fn_test = data_fn(
        get_trunc_data)  # filenames, data trunc or not
    y_df, x_csc, t_csc = read_in_data(fn_full, fn_y_full, fn_test)

    if get_neg_data:
        partial_y_df, partial_x_csc, partial_t_csc = read_in_data(fn_neg, fn_y_neg, fn_test)
    else:
        partial_y_df, partial_x_csc, partial_t_csc = read_in_data(fn_pos, fn_y_pos, fn_test)

    return partial_y_df, partial_x_csc, y_df, x_csc, t_csc  # the partial test csc is redundant

def feature_selection(x, y, svd_n, chi2_n):
    print('in feature selection')
    # By doing tests on the pos and full set,
    # ~100% of variance is explained by 80 variables - positives
    # ~100% of variance is explained by 800 variables - full

    print('Finding reduced SVD matrix')
    svd = decomposition.TruncatedSVD(algorithm='randomized', n_components=svd_n, n_iter=7)
    svd.fit(x, y)

    print('Finding reduced Chi^2 Matrix')
    ch2 = SelectKBest(chi2, k=chi2_n).fit(x, y)
    return svd, ch2

def evaluate_kNN(x_pos, y_pos, x, y, folds, n_params, runs, steps, k_neighbors):
    print("in evaluate kNN")
    neigh = KNeighborsClassifier(n_neighbors=k_neighbors)
    svd_m = decomposition.TruncatedSVD(algorithm='randomized', n_components=n_params, n_iter=7)
    scores = []
    run = []
    for i in np.arange(n_params, runs, steps):
        # svd_model = svd_m.fit(x_pos, y_pos)
        # x_svd = svd_model.transform(x)
        # test_svd = svd_model.transform(test)
        # neigh.fit(x_svd, y)
        # val_list = cross_val_score(neigh, x_ch2, y, cv=folds, scoring='f1').mean()
        neigh.n_neighbors=i
        ch2_model = SelectKBest(chi2, k=i).fit(x, y)
        x_ch2 = ch2_model.transform(x)
        neigh.fit(x_ch2, y)
        val_list = cross_val_score(neigh, x_ch2, y, cv=folds, scoring='f1').mean()
        scores.append(val_list)
        run.append(i)
    evaluate_knn_csv(scores, k_neighbors, folds, chi2_n_params, run)
    print(scores)

def run_combined_validation_runs(x, y, chi_n_params, svd_n_params, folds, error, runs):
    n = 0
    while n < runs:
        # error += .005
        svd_n_params += 1
        chi_n_params += 1
        svd_model, chi2_model = feature_selection(x, y, svd_n_params, chi_n_params)  # is this sparse?
        svd_model, chi2_model = feature_selection(x, y, svd_n_params, chi_n_params)  # is this sparse?
        kNN(chi2_model, )
        x_svd = svd_model.transform(data_csc)
        x_ch2 = chi2_model.transform(data_csc)
        x_df = pd.DataFrame(x_)
        # Model runs
        print('Running Model')
        evaluate_models(x_svd, x_ch2, y, folds, error, chi_n_params, svd_n_params, 'full')
        n += 1
    
def predict_test_set_svd(n_params, test, x, y, x_pos, y_pos, error):
    random_state = np.random.RandomState(0)
    svd_m = decomposition.TruncatedSVD(algorithm='randomized', n_components=n_params, n_iter=7)
    svm_m = svm.SVC(kernel='linear', C=error, probability=True, random_state=random_state)  # C is penalty of error
    # run_combined_validation_runs(data_csc, y, chi_n_params, svd_n_params, folds, error, runs)
    # SVM/SVD with positive set
    # first get fit truncated SVD on positive data set, trim full and test parameters
    svd_model = svd_m.fit(x_pos, y_pos)
    x_svd = svd_model.transform(x)
    test_svd = svd_model.transform(test)
    graph_ROC(x_svd, y)
    # now train svm on tranformed data sets
    svm_model = svm_m.fit(x_svd, y)
    # predict test set
    svd_svm_predictions = svm_model.predict(test_svd)
    svd_svm_test_prob = svm_model.predict_proba(test_svd)
    svd_svm_train_prob = svm_model.predict_proba(x_svd)


    return svd_svm_predictions, svd_svm_test_prob, svd_svm_train_prob

def predict_test_set_chi2(n_params,test,x,y):
    # use NB/CHI2 with full set
    # first get fit chi2 on  data set, trim test parameters
    ch2_m = SelectKBest(chi2, k=n_params)
    nb_m = BernoulliNB()
    ch2_model = ch2_m.fit(data_csc, y)
    x_chi2 = ch2_model.transform(x)
    test_chi2 = ch2_model.transform(test)

    # now train svm on tranformed data sets
    nb_model = nb_m.fit(x_chi2, y)

    # predict test set
    chi2_nb_predictions = nb_model.predict(test_chi2)
    chi2_nb_train_prob = nb_model.predict_proba(x_chi2)
    chi2_nb_test_prob = nb_model.predict_proba(test_chi2)

    return chi2_nb_predictions, chi2_nb_test_prob, chi2_nb_train_prob

def kNN(svm_test_prob, nb_test_prob, svm_x_prob, nb_x_prob, x_pos, y_pos, x, y, test, n_params):
    print('Performing Dimension Reduction with SVD and classifying with KNN')
    neigh = kneighborsclassifier(n_neighbors=3)
    svd_m = decomposition.truncatedsvd(algorithm='randomized', n_components=n_params, n_iter=7)
    svd_model = svd_m.fit(x_pos, y_pos)
    x_svd = svd_model.transform(x)
    test_svd = svd_model.transform(test)
    neigh.fit(x_svd, y)
    knn_df, train_prob_df, test_prob_df = merge_probabilities(svm_test_prob, nb_test_prob, svm_x_prob, nb_x_prob)
    val_list = cross_val_score(neigh, x, y, cv=10, scoring='f1')
    y_knn = neigh.predict(test_svd)
    print(val_list)
    # plt.figure()
    # print(y_knn)
    # plt.plot(x=range(0, len(y_knn)), y=y_knn, type='scatter')
    return y_knn, knn_df

def merge_probabilities(svm_test_prob, nb_test_prob, svm_x_prob, nb_x_prob):
    pred_test_prob_list = list(zip(svm_test_prob[:, 0], svm_test_prob[:, 1], nb_test_prob[:, 0], nb_test_prob[:, 1]))
    pred_train_prob_list = list(zip(svm_x_prob[:, 0], svm_x_prob[:, 1], nb_x_prob[:, 0], nb_x_prob[:, 1]))
    test_df = pd.DataFrame(data=pred_test_prob_list)
    train_df = pd.DataFrame(data=pred_train_prob_list)
    knn_df = pd.DataFrame(pred_test_prob_list, columns=['svm pos', 'svm neg', 'nb neg', 'nb pos'])
    return knn_df, train_df, test_df

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

if __name__ == '__main__':
    # print("the number of parameters is {0}, length {1}, second entry is : \n{2}".format(largest_num_params, len(param_d[0][1]), param_d[0]))
    # need to clean data first
    # can i visualize data?
    print('to sparse')
    get_neg_data = False  # get neg | pos data
    get_truncated_data = False
    # data file with choice of negative or positive dataset
    partial_binding_df, partial_data_csc, binding_df, data_csc, test_csc = get_processed_data(get_neg_data,
                                                                                              get_truncated_data)

    # Reading in data that has previously been formatted
    y, y_pos = binding_df.binding, partial_binding_df.binding

    # PCA feature selection
    # SVM performs best with SVD reduction, at error rate = .93 with 22 params
    # BN performs best with chi2, at 347-355
    chi_n_params = 348
    svd_n_params = 22
    error = 0.93
    # Get predictions
    # Testing kNN
    x = data_csc
    x_pos = partial_data_csc
    folds = 5
    chi2_n_params = 1
    max_params = 50
    steps = 2
    k_n = 5
    evaluate_kNN(x_pos, y_pos, x, y, folds, chi2_n_params, max_params, steps, k_n)

    #
    # svd_svm_predictions, svm_test_proba, svm_x_proba = predict_test_set_svd(svd_n_params, test_csc, data_csc, y,
    #                                                                         partial_data_csc, y_pos, error)
    #
    # chi2_nb_predictions, nb_test_proba, nb_x_proba = predict_test_set_chi2(chi_n_params, test_csc, data_csc, y)
    # knn_predictions, knn_df = kNN(svm_test_proba, nb_test_proba, svm_x_proba, nb_x_proba, partial_data_csc, y_pos,
    #                               data_csc, y, test_csc, svd_n_params)
    # # Combine Predictions
    # merged_p_list, sum_vote_list = merge_predictions(svd_svm_predictions, chi2_nb_predictions, knn_predictions)
    # # Print Results
    # print_results_to_csv(merged_p_list, knn_df)
    # time = dt.now()
    # print('Ending Test Run at {}'.format(time))

