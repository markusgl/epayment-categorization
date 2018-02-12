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
    counts, targets = FeatureExtractor(ngramrange=(1,4), maxdf=0.5, useidf=False, sublinear=True).extract_features_from_csv()
    # split data into test and training set - hold 20% out for testing
    X_train, X_test, y_train, y_test = train_test_split(counts, targets, test_size=0.2, random_state=1)

    pipeline = Pipeline([
        #('clf', MultinomialNB())
        ('clf', SGDClassifier())
    ])
    #param_range = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    #param_range = [1.0, 1.2, 1.4, 1.6]
    param_range = [0.01, 0.001, 0.0001]

    train_scores, test_scores = validation_curve(estimator=pipeline, X=X_train,
                                                 y=y_train,
                                                 param_name='clf__alpha',
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


def estimate_parameters(multinomial_nb=False, bernoulli_nb=False, k_nearest=False, support_vm=False, support_vmsgd=False):
    fe = FeatureExtractor()
    counts, targets = fe.fetch_data()
    #counts, targets = FeatureExtractor().extract_features_from_csv()
    #X_train, X_test, y_train, y_test = train_test_split(counts, targets, test_size=0.2, random_state=1)

    MAX_DF = [0.5, 0.75, 1.0]
    N_GRAMS = [(1, 1), (1, 2), (1, 3)]

    if multinomial_nb:
        CLF = MultinomialNB()
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__ngram_range': N_GRAMS,
            #'tfidf__max_df': MAX_DF,
            #'tfidf__ngram_range': N_GRAMS,
            #'tfidf__sublinear_tf': (True, False),
            #'tfidf__use_idf': (True, False),
            'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
        }
    elif bernoulli_nb:
        CLF = BernoulliNB()
        parameters = {
            #'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__ngram_range': N_GRAMS,
            #'vect__analyer': ('word', 'char', 'char_wb'),
            'tfidf__max_df': MAX_DF,
            'tfidf__ngram_range': N_GRAMS,
            'tfidf__sublinear_tf': (True, False),
            'tfidf__use_idf': (True, False),
            #'tfidf_norm': ('l1', 'l2'),
            'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)
            #'clf__binarize': (0.0, 0.1, 0.2, 0.5)
        }
    elif k_nearest:
        CLF = KNeighborsClassifier()
        parameters = {
            #'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__ngram_range': N_GRAMS,
            'tfidf__max_df': MAX_DF,
            'tfidf__ngram_range': N_GRAMS,
            'tfidf__sublinear_tf': (True, False),
            'tfidf__use_idf': (True, False),
            'clf__n_neighbors': range(2, 10),
            'clf__weights': ('uniform', 'distance'),
            'clf__algorithm': ('auto', 'brute'),
            'clf__leaf_size': (20, 30, 40)
        }
    elif support_vm:
        CLF = SVC()
        parameters = {
            #'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__ngram_range': N_GRAMS,
            'tfidf__max_features': (1, 2, 3),
            'tfidf__max_df': MAX_DF,
            'tfidf__ngram_range': N_GRAMS,
            'tfidf__sublinear_tf': (True, False),
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2', None),
            'tfidf__smooth_idf': (True, False),
            'clf__kernel': ('linear', 'sigmoid', 'rbf', 'poly'),
            'clf__decision_function_shape': ('ovo', 'ovr'),
            'clf__C': (1, 10, 100),
            'clf__gamma': (1.0, 1.2, 1.4, 1.6)
        }
    elif support_vmsgd:
        CLF = SGDClassifier(max_iter=1000)
        parameters = {
            #'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__ngram_range': N_GRAMS,
            'tfidf__max_df': MAX_DF,
            'tfidf__ngram_range': N_GRAMS,
            'tfidf__analyzer': ('word', 'char'),
            'tfidf__sublinear_tf': (True, False),
            'tfidf__use_idf': (True, False),
            'clf__loss': ('hinge', 'modified_huber', 'squared_hinge'),
            'clf__penalty': ('l1', 'l2', 'elasticnet'),
            'clf__alpha': (0.01, 0.001, 0.0001),
            'clf__tol': (0.2, 1e-2, 1e-3, 1e-4)
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
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=10,
                               cv=15, scoring='accuracy')
    print("parameters:")
    pprint(parameters)

    # learn vocabulary
    grid_search.fit(counts, targets)

    print("Best parameters: " + str(grid_search.best_params_))
    print("Best score: %0.3f" % grid_search.best_score_)

    """
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print((param_name, best_parameters[param_name]))
    """


#estimate_parameters(support_vm=True)
plot_validation_curve()