import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas
import re

nltk.download('stopwords')
disturb_chars = '([\/+]|\s{3,})' #regex

class StemTokenizer(object):
    def __init__(self):
        self.sbs = nltk.SnowballStemmer('german', ignore_stopwords=True)

    def __call__(self, doc):
        # TreeBankTokenizer
        return [self.sbs.stem(t) for t in nltk.word_tokenize(doc)]


def estimate_parameters(multinomial_nb=False, bernoulli_nb=False,
                        k_nearest=False, support_vm=False, support_vmsgd=False,
                        bow=False, tfidf=False):

    df = pandas.read_csv(filepath_or_buffer='/var/booking_categorizer/Labeled_transactions_sorted_same_class_amount.csv', delimiter=',')
    #df['values'] = ' '.join((df.bookingtext, df.usage, df.owner)).encode('latin-1').decode('latin-1').lower().replace(disturb_chars, ' ')
    df['values'] =  df['values'] = df.bookingtext.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.usage.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.owner.str.replace(disturb_chars, ' ').str.lower()

    targets = df['category'].values
    counts = df['values'].values.astype(str)

    MAX_DF = [0.25, 0.5, 0.75, 1.0]
    N_GRAMS = [(1, 1), (1, 2), (1, 3), (1, 4)]

    if multinomial_nb:
        CLF = MultinomialNB()
        clf_name = "MultinomialNB"
        parameters = {
            'clf__alpha': 10.0 ** -np.arange(5, 11)
        }
    elif bernoulli_nb:
        CLF = BernoulliNB()
        clf_name = "BernoulliNB"
        parameters = {
            'clf__alpha': 10.0 ** -np.arange(5, 11)
        }
    elif k_nearest:
        CLF = KNeighborsClassifier()
        clf_name = "KNN"
        parameters = {
            'clf__n_neighbors': range(2, 10),
            'clf__weights': ('uniform', 'distance'),
            'clf__algorithm': ('auto', 'brute'),
            'clf__leaf_size': (20, 30, 40)
        }
    elif support_vm:
        CLF = SVC()
        clf_name = "SVC"
        parameters = {
            'clf__kernel': ('linear', 'sigmoid', 'rbf', 'poly'),
            'clf__decision_function_shape': ('ovo', 'ovr'),
            'clf__C': (1, 10, 100),
            'clf__gamma': (1.0, 1.2, 1.4, 1.6)
        }
    elif support_vmsgd:
        CLF = SGDClassifier(max_iter=50)
        clf_name = "SGDClassifier"
        parameters = {
            'clf__loss': ('hinge', 'modified_huber', 'squared_hinge'),
            'clf__penalty': ('l1', 'l2', 'elasticnet'),
            'clf__alpha': 10.0**-np.arange(1,7),
            'clf__tol': (0.2, 1e-2, 1e-3, 1e-4)
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

    with open('/var/booking_categorizer/gridsearch_log', 'a') as file:
        file.write("Starting gridsearch\n")

    # learn vocabulary
    grid_search.fit(counts, targets)

    print("Best parameters: " + str(grid_search.best_params_))
    print("Best score: %0.3f" % grid_search.best_score_)

    filename = '/var/booking_categorizer/gridsearch_result'
    with open(filename, 'a') as file:
        file.write("------------------------------" + "\n" +
                    clf_name + "\n" +
                    "Best parameters: " + str(grid_search.best_params_) + "\n" +
                   "Best score: %0.3f" % grid_search.best_score_  + "\n")

estimate_parameters(k_nearest=True, tfidf=True)
#estimate_parameters(support_vm=True, tfidf=True)
#estimate_parameters(support_vmsgd=True, bow=True)
#estimate_parameters(support_vmsgd=True, tfidf=True)