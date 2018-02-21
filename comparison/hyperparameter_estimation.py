"""
This module is used for estimating the best hyperparemter values
for bank transactions
"""

from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
from feature_extraction import FeatureExtractor
import matplotlib.pyplot as plt


def plot_validation_curve():
    """
    Plots the validation curve for a given range of parameters (param_range)
    This is intedet for limiting the values used in grid search
    (method estimate_parameters)
    """
    counts, targets = FeatureExtractor.tfidf(ngram_range=(1,4), max_df=0.5,
                                             use_idf=False, sublinear_tf=True)\
                                             .extract_features_from_csv

    # split data into test and training set - hold 20% out for testing
    X_train, X_test, y_train, y_test = train_test_split(counts, targets,
                                                        test_size=0.2,
                                                        random_state=1)

    # example usage validating param range for 'alpha' of SGDClassifier
    pipeline = Pipeline([
        ('clf', SGDClassifier())
    ])
    param_range = [10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1]

    train_scores, test_scores = validation_curve(estimator=pipeline, X=X_train,
                                                 y=y_train,
                                                 param_name='clf__alpha',
                                                 param_range=param_range,
                                                 cv=10)
    print(train_scores)
    print(test_scores)
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
    plt.xlabel('Parameter alpha')
    plt.ylabel('Accuracy')
    plt.ylim([0.2, 1.0])
    plt.show()


def estimate_parameters(multinomial_nb=False, bernoulli_nb=False,
                        k_nearest=False, support_vm=False, support_vmsgd=False,
                        bow=False, tfidf=False):
    """
    This method performs a grid search on the given algorithm using a fixed
    set of parameter ranges.
    The values with highest score are printed to stdout after evaluation
    :param multinomial_nb: MultinomialNB
    :param bernoulli_nb: BernoulliNB
    :param k_nearest: KNearestClassifier
    :param support_vm: Linear SVM aka SVC
    :param support_vmsgd: SGDClassifier
    :param bow: CountVectorizer aka Bag-of-words
    :param tfidf: TfidfVectorizer
    """
    fe = FeatureExtractor()
    counts, targets = fe.fetch_data()

    MAX_DF = [0.25, 0.5, 0.75, 1.0]
    N_GRAMS = [(1, 1), (1, 2), (1, 3), (1, 4)]

    if multinomial_nb:
        CLF = MultinomialNB()
        parameters = {
            'clf__alpha': 10.0 ** -np.arange(5, 11)
        }
    elif bernoulli_nb:
        CLF = BernoulliNB()
        parameters = {
            'clf__alpha': 10.0 ** -np.arange(5, 11)
        }
    elif k_nearest:
        CLF = KNeighborsClassifier()
        parameters = {
            'clf__n_neighbors': range(2, 10),
            'clf__weights': ('uniform', 'distance'),
            'clf__algorithm': ('auto', 'brute'),
            'clf__leaf_size': (20, 30, 40)
        }
    elif support_vm:
        CLF = SVC()
        parameters = {
            'clf__kernel': ('linear', 'sigmoid', 'rbf', 'poly'),
            'clf__decision_function_shape': ('ovo', 'ovr'),
            'clf__C': (100, 1000, 10000, 100000, 1000000),
            'clf__gamma': (0.001, 0.01, 0.1, 1)
        }
    elif support_vmsgd:
        CLF = SGDClassifier(max_iter=50)
        parameters = {
            'clf__loss': ('hinge', 'modified_huber', 'squared_hinge'),
            'clf__penalty': ('l1', 'l2', 'elasticnet'),
            'clf__alpha': 10.0**-np.arange(1,8),
            'clf__tol': (0.3, 0.2, 1e-2, 1e-3, 1e-4),
            'clf__n_iter': np.ceil(10**6 / 1062),
            'clf__eta0': (0.0, 0.2, 0.5, 0.7),
            'clf__learning_rage': ('constant', 'optimal', 'invscaling'),
            'clf__average': (True, False)
        }
    else:
        print('Please provide one which algorithm to use')
        return

    # add feature extraction params and classifier to pipeline
    if bow:
        parameters.update({
            'vect__max_df': MAX_DF,
            'vect__ngram_range': N_GRAMS
            })

        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', CLF)
        ])
    elif tfidf:
        parameters.update({
            'tfidf__max_df': MAX_DF,
            'tfidf__ngram_range': N_GRAMS,
            'tfidf__analyzer': ('word', 'char'),
            'tfidf__sublinear_tf': (True, False),
            'tfidf__smooth_idf': (True, False),
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2', None)
            })

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', CLF)
        ])
    else:
        print('Please provide one which algorithm to use')
        return

    # perform grid search on pipeline
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters,
                               cv=15, scoring='accuracy')
    print("parameters:")
    pprint(parameters)
    print("Starting grid search. This may take some time...")

    # learn vocabulary
    grid_search.fit(counts, targets)

    print("Best parameters: " + str(grid_search.best_params_))
    print("Best score: %0.3f" % grid_search.best_score_)

    filename = '/var/booking_categorizer/'
    with open(filename, 'a') as file:
        file.write("Best parameters: " + str(grid_search.best_params_) + "\n" +
                   "Best score: %0.3f" % grid_search.best_score_)


#estimate_parameters(support_vm=True, tfidf=True)
#plot_validation_curve()
