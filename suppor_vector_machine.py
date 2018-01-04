"""
SVM with stochastic gradient descend (SGD) learning
"""

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import feature_extraction
from sklearn import metrics
from sklearn.cross_validation import KFold
import numpy
from sklearn.metrics import confusion_matrix, accuracy_score
from plotter import Plotter
from categories import Categories as cat
import numpy as np
import matplotlib.pyplot as plt
import scipy

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

def classify_examples(tfidf=False, plot=False, log=False):
    """
    Classify examples and print prediction result
    :param bernoulliNB: use Bernoulli Model - default is Multinomial NB
    :param tfidf: use TF-IDF - default is bag-of-words (word count)
    """

    classifier = SGDClassifier(loss='hinge', alpha=0.001, max_iter=100)
    if log:
        classifier = SGDClassifier(loss='log')

    # retrieve feature vector and target vector
    counts, targets = feature_extraction.extract_features()
    if tfidf:
        counts, targets = feature_extraction.extract_features_tfidf()

    example_counts, examples = feature_extraction.extract_example_features()

    classifier.fit(counts, targets) #train the classifier
    predictions = classifier.predict(example_counts)

    if plot:
        #TODO
        print("not implemented yet")

    if log:
        predict_probabilities = classifier.predict_proba(example_counts)
        for i in range(len(predict_probabilities)):
            print(examples[i])
            val = predict_probabilities[i]
            for j in range(len(category_names)):
                print(category_names[j] + ": " + str(round(val[j] * 100, 2)) + "%")
            print(" ")

    print(predictions)

    #print(metrics.classification_report())


def classify_w_cross_validation(plot=False):
    """
    Validate the classifier against unseen data using k-fold cross validation
    :param plot: choose whether to plot the confusion matrix with matplotlib
    """
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer()),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', SGDClassifier(loss='log'))
        #('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))
    ])

    data = feature_extraction.append_data_frames()
    k_fold = KFold(n=len(data), n_folds=6)
    scores = []
    confusion = numpy.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])
    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['class'].values.astype(str)

        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['class'].values.astype(str)

        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        #score = f1_score(test_y, predictions, average='samples')
        score = accuracy_score(test_y, predictions)
        scores.append(score)

    print('Total transactions classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)

    if plot:
        Plotter.plot_and_show_confusion_matrix(confusion,
                                               category_names,
                                               normalize=True,
                                               title='SVM Classifier',
                                               save=True)

classify_examples(log=True)
#classify_w_cross_validation(True)