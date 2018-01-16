"""
Compare different classifiers for booking classification
"""

import numpy as np
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from categories import Categories as cat
from comparison.plotter import Plotter
from feature_extraction import FeatureExtractor

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]


def classify(plot=False, multinomial_nb=False, bernoulli_nb=False, knn=False, support_vm=False,
             decision_tree=False, random_forest=False, persist=False, logistic_regression=False,
             hyperparam_estim=False):
    """
    Validate the classifier against unseen data using k-fold cross validation
    """
    counts, target = FeatureExtractor().extract_features_from_csv()

    # hold 20% out for testing
    X_train, X_test, y_train, y_test = train_test_split(counts, target, test_size=0.4, random_state=0)

    #X_train.shape, y_train.shape
    #X_test.shape, y_test.shape
    #sc = StandardScaler(with_mean=False)
    #sc.fit(X_train)
    #X_train_std = sc.transform(X_train)
    #X_test_std = sc.transform(X_test)

    tuned_parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

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

        clf.fit(X_train, y_train)
        clf_title = 'Support Vector Machine'
    elif logistic_regression:
        #clf = SGDClassifier(loss='log', alpha=0.001, max_iter=100)
        clf = SGDClassifier(loss='log', max_iter=100, tol=None, shuffle=True)
        clf.fit(X_train, y_train)
        clf_title = 'Logistic Regression'
    elif random_forest:
        clf = RandomForestClassifier().fit(X_train, y_train)
        clf_title = 'Random Forest'
    elif hyperparam_estim:
        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
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

classify(hyperparam_estim=True)
#classify(support_vm=True)