"""
Compare different classifiers for booking classification
Model selection and hyperparameter estimation for prototype system
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
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score, \
    f1_score, precision_score, recall_score, average_precision_score, \
    precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, \
    cross_validate, cross_val_predict, LeaveOneOut
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from categories import Categories as cat
from comparison.plotter import Plotter
from feature_extraction import FeatureExtractor
import matplotlib.pyplot as plt
from sklearn.decomposition import pca, TruncatedSVD
from sklearn.preprocessing import Normalizer, label_binarize
from collections import Counter

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

category_names_reverse = ['barentnahme', 'finanzen', 'freizeitlifestyle',
                         'lebenshaltung', 'mobilitaetverkehrsmittel',
                          'versicherungen','wohnenhaushalt']

def get_classweight(classes, smoth_factor=0.02):
    counter = Counter(classes)

    if smoth_factor > 0:
        p = max(counter.values()) * smoth_factor
        for k in counter.keys():
            counter[k] += p

        majority = max(counter.values())

        return {cls: float(majority / count) for cls, count in counter.items()}


def classify(plot=False, multinomial_nb=False, bernoulli_nb=False, knn=False, support_vm=False,
             svm_sgd=False,
             decision_tree=False, random_forest=False, persist=False, logistic_regression=False):
    """
    Validate the classifier against unseen data using k-fold cross validation
    """
    counts, targets = FeatureExtractor().extract_features_from_csv()
    target_ints = []
    for target in targets:
        target_ints.append(category_names_reverse.index(target))

    class_weights = get_classweight(targets)


    # split data into test and training set - hold 20% out for testing
    X_train, X_test, y_train, y_test = train_test_split(counts, targets, test_size=0.2, random_state=0)


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
        #clf = SVC(kernel='linear', C=10)
        clf = SVC(kernel='linear', C=10, decision_function_shape='ovr', probability=True)
        """
        Optimal Parameters
        {'clf__C': 1000, 'clf__gamma': 0.001, 'clf__kernel': 'rbf',
         'tfidf__max_df': 0.5, 'tfidf__ngram_range': (1, 1),
         'tfidf__sublinear_tf': False, 'tfidf__use_idf': True}
        """

        #clf.fit(X_train, y_train)
        clf.fit(counts, targets)
        clf_title = 'Support Vector Machine'
    elif svm_sgd:
        clf = SGDClassifier(loss='hinge', penalty='elasticnet', max_iter=100, tol=None, class_weight=class_weights)
        clf.fit(counts, targets)
        clf_title = 'SVM (SGD)'
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
    #predictions = clf.predict(X_test)

    if persist:
        joblib.dump(clf, clf_title+'.pkl')

    # scores
    #print(clf.score(X_test, y_test))
    #print("Accuracy Score: %0.2f" % accuracy_score(y_test, predictions))
    # K-folds cross validation
    #text, targets = FeatureExtractor().fetch_data()
    #kc_scores = []
    ac_scores = []
    f1_scores = []
    prec_scores = []
    rec_scores = []

    # Use label_binarize to be multi-label like settings
    #Y = label_binarize(targets, classes=[0, 1, 2])
    #n_classes = Y.shape[1]


    #kf = KFold(n_splits=15)
    #kf = KFold(n=1062, n_folds=10)
    kf = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
    #loo = LeaveOneOut()
    #loo.get_n_splits(counts)

    confusion = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])

    for k, (train_indices, test_indices) in enumerate(kf):
        train_text = counts[train_indices]
        train_y = targets[train_indices]

        test_text = counts[test_indices]
        test_y = targets[test_indices]

        clf.fit(train_text, train_y)
        predictions = clf.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)

        ac_scores.append(accuracy_score(test_y, predictions))
        f1_scores.append(f1_score(test_y, predictions, average="micro"))
        prec_scores.append(precision_score(test_y, predictions, average="micro"))
        rec_scores.append(recall_score(test_y, predictions, average="micro"))

        #print(classification_report(test_y, predictions, target_names=targets))

    """
    for train_indices, test_indices in loo.split(counts):
        train_text = counts[train_indices]
        train_y = targets[train_indices]

        test_text = counts[test_indices]
        test_y = targets[test_indices]

        clf.fit(train_text, train_y)
        predictions_k = clf.predict(test_text)

        confusion += confusion_matrix(test_y, predictions_k)
        k_score = accuracy_score(test_y, predictions_k)
        kc_scores.append(k_score)
    """
    print("K-Folds Accuracy-score: ", sum(ac_scores)/len(ac_scores))
    print("K-Folds F1-score: ", sum(f1_scores)/len(f1_scores))
    print("K-Folds Precision-score: ", sum(prec_scores)/len(prec_scores))
    print("K-Folds Recall-score: ", sum(rec_scores)/len(rec_scores))

    print("CV accuracy : %.3f +/- %.3f" % (np.mean(ac_scores), np.std(ac_scores)))

    #scores = cross_val_score(clf, counts, targets, cv=10, scoring='accuracy')
    #predicted = cross_val_predict(clf, counts, targets, cv=10)
    #print(metrics.accuracy_score(targets, predicted))
    #scores = cross_val_score(clf, counts, targets, cv=10, scoring='f1_macro')
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    pipeline = Pipeline([
        ('clf', SVC(kernel='linear'))
    ])
    #param_range = [0.6, 0.8, 1.0, 1.2]
    param_range = [1, 10, 100, 1000]
    #param_range = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    train_scores, test_scores = validation_curve(estimator=pipeline, X=X_train,
                                                 y=y_train,
                                                 param_name='clf__C',
                                                 param_range=param_range,
                                                 cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5,
             label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')
    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')


    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter gamma')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.show()



    # confusion matrix train test split
    #confusion += confusion_matrix(y_test, predictions)
    #print(confusion)

    labels = ['Barent.', 'Finanzen', 'Freizeit&L.', 'Lebensh.',
              'Mobil.&V.', 'Versich.', 'Wohn.&Haus.']
    if plot:
        Plotter.plot_and_show_confusion_matrix(confusion,
                                              labels,
                                              normalize=True,
                                              title=clf_title,
                                              save=True)


def estimate_parameters(multinomial_nb=False, bernoulli_nb=False,
                        k_nearest=False, support_vm=False, support_vmsgd=False):
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
            'vect__analyer':'word',
            #'tfidf__max_df': MAX_DF,
            #'tfidf__ngram_range': N_GRAMS,
            #'tfidf__sublinear_tf': (True, False),
            #'tfidf__use_idf': (True, False),
            'tfidf_norm': ('l1', 'l2'),
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
            'clf__n_neighbours': range(2, 31),
            'clf__weights': ('uniform', 'distance'),
            'clf__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
            'clf__leaf_size': (20, 30, 40)
        }
    elif support_vm:
        CLF = SVC()
        parameters = {
            #'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__ngram_range': N_GRAMS,
            'tfidf__max_df': MAX_DF,
            'tfidf__ngram_range': N_GRAMS,
            'tfidf__sublinear_tf': (True, False),
            'tfidf__use_idf': (True, False),
            'clf__kernel': ('linear', 'sigmoid', 'rbf', 'poly'),
            'clv__decision_function_shape': ('ovo', 'ovr'),
            'clf__C': (1, 10, 100, 1000),
            'clf__gamma': (0.6, 0.8, 1.0, 1.2)
        }
    elif support_vmsgd:
        CLF = SGDClassifier()
        parameters = {
            #'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__ngram_range': N_GRAMS,
            'tfidf__max_df': MAX_DF,
            #'tfidf__ngram_range': N_GRAMS,
            'tfidf__analyzer': ('word', 'char'),
            #'tfidf__sublinear_tf': (True, False),
            #'tfidf__use_idf': (True, False),
            #'clf__loss': ('hinge', 'modified_huber', 'squared_hinge'),
            'clf__loss': ('hinge'),
            'clf__penalty': ('l1', 'l2', 'elasticnet'),
            #'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001),
            #'clf__max_iter': (0.4, 0.5, 0.6, 100),

        }
    else:
        print('Please provide one which algorithm to use')
        return

    pipeline = Pipeline([
        #('vect', CountVectorizer()),
        ('tfidf', TfidfVectorizer()),
        ('clf', CLF)
    ])

    # perform grid search on pipeline
    grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1,
                               verbose=10, cv=12)

    print("parameters:")
    pprint(parameters)

    # learn vocabulary
    grid_search.fit(booking_data, booking_targets)

    print()
    print("Best parameters: " + str(grid_search.best_params_))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print((param_name, best_parameters[param_name]))



classify(svm_sgd=True)
#classify(hyperparam_estim=True)
#classify(logistic_regression=True)
#classify(gridsearch=True)
#estimate_parameters(support_vmsgd=True)

#booking_data, booking_targets = FeatureExtractor().fetch_data()
#print(type(booking_data))
#print(booking_data)
#print(type(booking_targets))
#print(booking_targets)
