"""
Compare different classifiers for booking classification
"""
from pprint import pprint
from time import time

import numpy as np
from sklearn import tree, svm, metrics
from sklearn.datasets import fetch_20newsgroups
#from sklearn_evaluation import plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, cross_val_predict, KFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from categories import Categories as cat
from comparison.plotter import Plotter
from feature_extraction import FeatureExtractor
import matplotlib.pyplot as plt
from sklearn.decomposition import pca, TruncatedSVD
from sklearn.preprocessing import Normalizer

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]


def classify(plot=False, multinomial_nb=False, bernoulli_nb=False, knn=False, support_vm=False,
             decision_tree=False, random_forest=False, persist=False, logistic_regression=False):
    """
    Validate the classifier against unseen data using k-fold cross validation
    """

    parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    counts, targets = FeatureExtractor().extract_features_from_csv()

    # Dimension reduction
    #svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    #normalizer = Normalizer(copy=False)
    #lsa = make_pipeline(svd, normalizer)
    #counts = svd.fit_transform(counts)

    # split data into test and training set - hold 20% out for testing
    X_train, X_test, y_train, y_test = train_test_split(counts, targets, test_size=0.2, random_state=0)

    #X_train.shape, y_train.shape
    #X_test.shape, y_test.shape
    #sc = StandardScaler(with_mean=False)
    #sc.fit(X_train)
    #X_train_std = sc.transform(X_train)
    #X_test_std = sc.transform(X_test)


    if multinomial_nb:
        clf = MultinomialNB(fit_prior=False).fit(X_train, y_train)
        clf_title = 'Multinomial NB'
    elif bernoulli_nb:
        clf = BernoulliNB().fit(X_train, y_train)
        clf_title = 'Bernoulli NB'
    elif knn:
        clf = KNeighborsClassifier().fit(X_train, y_train)
        clf_title = 'K-Nearest-Neighbour'
    elif decision_tree:
        clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
        clf_title = 'Decision Tree'
    elif support_vm:
        #clf = svm.LinearSVC(penalty='l2', dual=False)
        #clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=100) # Linear SVC
        #clf = SGDClassifier(loss='squared_hinge', alpha=0.001, max_iter=100)
        #clf = SGDClassifier(loss='modified_huber', alpha=0.001, max_iter=100)

        # best parameters
        #clf = SVC(kernel='rbf', gamma=0.001, C=1000)
        clf = SVC(kernel='linear', C=10)

        """
        Optimal Parameters
        {'clf__C': 1000, 'clf__gamma': 0.001, 'clf__kernel': 'rbf',
         'tfidf__max_df': 0.5, 'tfidf__ngram_range': (1, 1),
         'tfidf__sublinear_tf': False, 'tfidf__use_idf': True}
        """

        #clf.fit(X_train, y_train)
        clf.fit(counts, targets)
        clf_title = 'Support Vector Machine'
    elif logistic_regression:
        #clf = SGDClassifier(loss='log', alpha=0.001, max_iter=100)
        #clf = SGDClassifier(loss='log', max_iter=100, tol=None, shuffle=True)
        clf = SGDClassifier(loss='log')
        #clf.fit(X_train, y_train)
        clf.fit(counts, targets)
        clf_title = 'Logistic Regression'
    elif random_forest:
        clf = RandomForestClassifier().fit(X_train, y_train)
        clf_title = 'Random Forest'
    else:
        print('Please provide a classifer algorithm')
        return
    predictions = clf.predict(X_test)

    if persist:
        joblib.dump(clf, clf_title+'.pkl')

    # scores
    print(clf.score(X_test, y_test))
    print("Accuracy Score: %0.2f" % accuracy_score(y_test, predictions))


    # K-folds cross validation
    #text, targets = FeatureExtractor().fetch_data()
    kc_scores = []
    kf = KFold(n_splits=6)
    for train_indices, test_indices in kf.split(counts):
        train_text = counts[train_indices]
        train_y = targets[train_indices]

        test_text = counts[test_indices]
        test_y = targets[test_indices]

        clf.fit(train_text, train_y)
        predictions_k = clf.predict(test_text)

        k_score = accuracy_score(test_y, predictions_k)
        kc_scores.append(k_score)

    print("K-Folds score: ", sum(kc_scores)/len(kc_scores))

    #scores = cross_val_score(clf, counts, targets, cv=6, scoring='accuracy')

    #predicted = cross_val_predict(clf, counts, targets, cv=10)
    #print(metrics.accuracy_score(targets, predicted))
    #scores = cross_val_score(clf, counts, targets, cv=6, scoring='f1_macro')
    #print(scores)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    """
    confusion = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])

    confusion += confusion_matrix(y_test, predictions)
    print(confusion)
    """

    if plot:
        Plotter.plot_and_show_confusion_matrix(confusion,
                                              category_names,
                                              normalize=True,
                                              title=clf_title,
                                              save=True)


def estimate_parameters(multinomial_nb=False, bernoulli_nb=False, k_nearest=False, support_vm=False):
    booking_data, booking_targets = FeatureExtractor().fetch_data()

    # test bag-of-words and tf-idf with SVM
    """
    pipeline = Pipeline([
        #('vect', CountVectorizer()),
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC())
    ])

    SVC_KERNELS = ['linear', 'sigmoid', 'rbf', 'poly']
    C_OPTIONS = [1, 10, 100, 1000]
    GAMMAS = [1e-3, 1e-4]
    """
    MAX_DF = [0.5, 0.75, 1.0]
    N_GRAMS = [(1, 1), (1, 2), (1, 3)]

    if multinomial_nb:
        CLF = MultinomialNB()
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': N_GRAMS,
            #'tfidf__max_df': MAX_DF,
            #'tfidf__ngram_range': N_GRAMS,
            #'tfidf__sublinear_tf': (True, False),
            #'tfidf__use_idf': (True, False),
            'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
        }
    elif bernoulli_nb:
        CLF = BernoulliNB()
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': N_GRAMS,
            #'tfidf__max_df': MAX_DF,
            #'tfidf__ngram_range': N_GRAMS,
            #'tfidf__sublinear_tf': (True, False),
            #'tfidf__use_idf': (True, False),
            'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001),
            'clf__binarize': (0.0, 0.1, 0.2, 0.5)
        }
    elif k_nearest:
        CLF = KNeighborsClassifier()
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': N_GRAMS,
            #'tfidf__max_df': MAX_DF,
            #'tfidf__ngram_range': N_GRAMS,
            #'tfidf__sublinear_tf': (True, False),
            #'tfidf__use_idf': (True, False),
            #'clf__n_neighbours': (2, 3, 4, 5, 6, 7, 8, 9, 10),
            'clf__weights': ('uniform', 'distance'),
            'clf__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
            'clf__leaf_size': (20, 30, 40)
        }
    elif support_vm:
        CLF = SVC()
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__ngram_range': N_GRAMS,
            #'tfidf__max_df': MAX_DF,
            #'tfidf__ngram_range': N_GRAMS,
            #'tfidf__sublinear_tf': (True, False),
            #'tfidf__use_idf': (True, False),
            'clf__kernel': ('linear', 'sigmoid', 'rbf', 'poly'),
            'clf__C': (1, 10, 100, 1000),
            'clf__gamma': (1e-3, 1e-4)
        }
    else:
        print('Please provide one which algorithm to use')
        return

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        #('tfidf', TfidfVectorizer()),
        ('clf', CLF)
    ])


    #kernel_labels=['linear', 'sigmoid', 'rbf', 'poly']
    #plot.grid_search(clf.grid_scores_, change='n_estimators', kind='bar')

    # perform grid search on pipeline
    grid_search = GridSearchCV(pipeline, param_grid = parameters, n_jobs=1, verbose=10)

    print("parameters:")
    pprint(parameters)
    grid_search.fit(booking_data, booking_targets)

    print()
    print("best_param: " + str(grid_search.best_params_))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print((param_name, best_parameters[param_name]))


    #plot.grid_search(grid_search.grid_scores_, change='n_estimators', kind='bar')
    # Plot results
    """
    mean_scores = np.array(grid_search.cv_results_['mean_test_score'])
    mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(SVC_KERNELS))

    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    #bar_offsets = (np.arange(len(SVC_KERNELS)) * (len(kernel_labels) + 1) + .5)
    bar_offsets = (np.arange(len(C_OPTIONS)) * (len(C_OPTIONS) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(kernel_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

    plt.title("Comparing SVC kernel techniques")
    plt.xlabel('Kernels')
    plt.xticks(bar_offsets + len(kernel_labels) / 2, SVC_KERNELS)
    plt.ylabel('Accuracy')
    plt.ylim((0, 1))
    plt.legend(loc='upper left')
    plt.show()

    """

    #scores = [x[1] for x in grid_search.cv_results_['mean_test_score']]
    #scores = np.array(scores).reshape(len(C_OPTIONS), len(GAMMAS))
    """
    scores = np.array(grid_search.cv_results_['mean_test_score'])
    scores = scores.reshape(len(C_OPTIONS), -1, len(GAMMAS))

    for ind, i in enumerate(C_OPTIONS):
        plt.plot(GAMMAS, scores[ind], label='C: ' + str(i))
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Mean score')
    plt.show()
    """

classify(support_vm=True)
#classify(hyperparam_estim=True)
#classify(logistic_regression=True)
#classify(gridsearch=True)
#estimate_parameters(support_vm=True)

#booking_data, booking_targets = FeatureExtractor().fetch_data()
#print(type(booking_data))
#print(booking_data)
#print(type(booking_targets))
#print(booking_targets)
