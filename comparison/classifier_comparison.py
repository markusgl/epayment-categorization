"""
Compare different classifiers for booking classification
"""

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from feature_extraction import FeatureExtractor
from categories import Categories as cat
from comparison.plotter import Plotter

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]


def classify(plot=False, multinomial_nb=False, bernoulli_nb=False, knn=False, svm=False,
             decision_tree=False, random_forest=False, tfidf=False, persist=False):
    """
    Validate the classifier against unseen data using k-fold cross validation
    """
    counts, target = FeatureExtractor().extract_features_from_csv(tfidf)

    # hold 20% out for testing
    X_train, X_test, y_train, y_test = train_test_split(counts, target, test_size=0.2, random_state=0)

    #X_train.shape, y_train.shape
    #X_test.shape, y_test.shape
    sc = StandardScaler(with_mean=False)
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    if multinomial_nb:
        clf = MultinomialNB(fit_prior=False).fit(X_train_std, y_train)
        clf_title = 'Multinomial NB'
    elif bernoulli_nb:
        clf = BernoulliNB().fit(X_train_std, y_train)
        clf_title = 'Bernoulli NB'
    elif knn:
        clf = KNeighborsClassifier().fit(X_train_std, y_train)
        clf_title = 'K-Nearest-Neighbour'
    elif decision_tree:
        clf = tree.DecisionTreeClassifier().fit(X_train_std, y_train)
        clf_title = 'Decision Tree'
    elif svm:
        clf = SGDClassifier(loss='hinge', alpha=0.001, max_iter=100).fit(X_train, y_train)
        clf_title = 'Support Vector Machine'
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
    # print(clf.score(X_test_std, y_test))

    scores = cross_val_score(clf, counts, target, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    confusion = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])

    confusion += confusion_matrix(y_test, predictions)
    print(confusion)

    if plot:
        Plotter.plot_and_show_confusion_matrix(confusion,
                                              category_names,
                                              normalize=True,
                                              title=clf_title,
                                              save=True)

classify(svm=True, tfidf=True)
