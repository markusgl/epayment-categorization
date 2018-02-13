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
    precision_recall_curve, jaccard_similarity_score
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
from sklearn.decomposition import pca, TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer, label_binarize, StandardScaler
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
    # Feature extraction using tf-idf
    #counts, targets = FeatureExtractor(ngramrange=(1,3), maxdf=0.5, useidf=True, sublinear=False).extract_features_from_csv()

    if multinomial_nb:
        counts, targets = FeatureExtractor(maxdf=0.5, useidf=True,
                                           sublinear=False).extract_features_from_csv()
        clf = MultinomialNB(alpha=0.1, fit_prior=False)
        clf_title = 'Multinomial NB'
    elif bernoulli_nb:
        counts, targets = FeatureExtractor(maxdf=0.5, useidf=True,
                                           sublinear=True).extract_features_from_csv()
        clf = BernoulliNB(alpha=1e-05)
        clf_title = 'Bernoulli NB'
    elif knn:
        counts, targets = FeatureExtractor(ngramrange=(1, 1), maxdf=0.5, useidf=True,
                                           sublinear=True).extract_features_from_csv()
        clf = KNeighborsClassifier(weights='distance', n_neighbors=2, leaf_size=20, algorithm='auto')
        clf_title = 'K-Nearest-Neighbour'
    elif support_vm:
        counts, targets = FeatureExtractor(ngramrange=(1, 2), maxdf=0.5, useidf=False,
                                           sublinear=True).extract_features_from_csv()
        clf = SVC(kernel='sigmoid', C=10, gamma=1.4, decision_function_shape='ovr', probability=True)
        clf_title = 'Support Vector Machine'
    elif svm_sgd:
        counts, targets = FeatureExtractor(ngramrange=(1, 1), maxdf=0.5, useidf=True,
                                           sublinear=True).extract_features_from_csv()
        clf = SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.001, max_iter=1000, tol=0.2)
        clf_title = 'SVM (SGD)'

    elif decision_tree:
        clf = tree.DecisionTreeClassifier()
        clf_title = 'Decision Tree'
    elif logistic_regression:
        clf = SGDClassifier(loss='log')
        clf_title = 'Logistic Regression'
    elif random_forest:
        clf = RandomForestClassifier()
        clf_title = 'Random Forest'
    else:
        print('Please provide a classifer algorithm')
        return

    target_ints = []
    for target in targets:
        target_ints.append(category_names_reverse.index(target))
    class_weights = get_classweight(targets)

    # split data into test and training set - hold 20% out for testing
    X_train, X_test, y_train, y_test = train_test_split(counts, targets, test_size=0.2, random_state=0)

    clf.fit(counts, targets)
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
        f1_scores.append(f1_score(test_y, predictions, average="macro"))
        prec_scores.append(precision_score(test_y, predictions, average="macro"))
        rec_scores.append(recall_score(test_y, predictions, average="macro"))

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


def estimate_jaccard_similarity():
    counts, targets = FeatureExtractor(ngramrange=(1, 1), maxdf=0.5, useidf=True,
                                       sublinear=True).extract_features_from_csv()
    clf = SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.001, max_iter=1000, tol=0.2)
    X_train, X_test, y_train, y_test = train_test_split(counts, targets, test_size=0.2, random_state=0)

    #clf.fit(counts, targets)
    clf.fit(X_train, y_train)

    print("Jaccard %.3f" % jaccard_similarity_score(y_test, clf.predict(X_test)))


classify(support_vm=True)
#estimate_jaccard_similarity()