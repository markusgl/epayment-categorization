"""
Multi-Class categorization vor e-payments using Naive Bayes classifier
"""

import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from plot_confusion_matrix import Ploter
import feature_extraction
from categories import Categories as cat
from sklearn.externals import joblib

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]


def classify_examples(examples, bernoulliNB=False, tfidf=False, persist=False):
    """
    Classify examples and print prediction result
    :param bernoulliNB: use Bernoulli Model - default is Multinomial NB
    :param tfidf: use TF-IDF - default is bag-of-words (word count)
    """
    data = feature_extraction.append_data_frames()
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(data['text'].values)

    classifier = MultinomialNB(fit_prior=False)
    if bernoulliNB:
        classifier = BernoulliNB()
    #classifier = joblib.load('nb_classifier.pkl')

    # retrieve feature vector and target vector
    counts, targets = feature_extraction.extract_features()
    if tfidf:
        counts, targets = feature_extraction.extract_features_tfidf()

    example_counts = count_vectorizer.transform(examples)

    classifier.fit(counts, targets)  # train the classifier
    if persist:
        joblib.dump(classifier, 'nb_classifier.pkl')


    predict_probabilities = classifier.predict_proba(example_counts)
    # print predictions for each class
    classify_result = []
    for i in range(len(predict_probabilities)):
        print(examples[i])
        prob = predict_probabilities[i]

        # print probabilites for individual classes
        for j in range(7):
            print(category_names[j] + ": " + str(round(prob[j] * 100, 2)) + "%")

        # classify_result
        if min(prob) == max(prob):
            print("Resultat: Sonstiges")
            classify_result.append('Sonstiges')
        else:
            m = max(prob)
            result = [i for i, j in enumerate(prob) if j == m]
            print("Resultat: " + str(category_names[result[0]]))

        print(" ")

    predictions = classifier.predict(example_counts)
    print(predictions)


def classify_examples_pipeline():
    """
    Classify examples and print prediction result
    ###### USE PIPELINING - DRAFT #######
    - Feature extraction and classification task are merged into one operation
    """
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])

    ''' Generate training data '''
    count_vectorizer = CountVectorizer()
    examples = ['advocard', 'versicherungen', 'dauerauftrag miete spenglerstr',
                'norma',
                'adac', 'nuernberger']
    # document to document-term matrix
    example_counts = count_vectorizer.transform(examples)

    # retrieve feature vector and target vector
    tfidf, targets = feature_extraction.extract_features_tfidf()
    pipeline.fit(tfidf, targets)  # train the classifier
    predictions = pipeline.predict(example_counts)

    print(predictions)


def classify_w_cross_validation(plot=False):
    """
    Validate the classifier against unseen data using k-fold cross validation
    :param plot: choose whether to plot the confusion matrix with matplotlib
    """
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', MultinomialNB(fit_prior=False))
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
        # score = f1_score(test_y, predictions, average='samples')
        score = accuracy_score(test_y, predictions)
        scores.append(score)

    print('Total transactions classified:', len(data))
    print('Score:', sum(scores) / len(scores))
    print('Confusion matrix:')
    print(confusion)

    if plot:
        Ploter.plot_and_show_confusion_matrix(confusion,
                                              category_names,
                                              normalize=True,
                                              title='NB Classifier normalized',
                                              save=True)


classify_examples(['advocard', 'versicherung', 'xsadadf', 'miete'])
#classify_w_cross_validation(True)
