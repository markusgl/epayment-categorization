from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import feature_extraction
from sklearn import metrics
from sklearn.cross_validation import KFold
import numpy
from sklearn.metrics import confusion_matrix, accuracy_score
from plot_confusion_matrix import Ploter
from categories import Categories as cat

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

def classify_examples(tfidf=False):
    """
    Classify examples and print prediction result
    :param bernoulliNB: use Bernoulli Model - default is Multinomial NB
    :param tfidf: use TF-IDF - default is bag-of-words (word count)
    """
    data = feature_extraction.append_data_frames()
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(data['text'].values)

    classifier = SGDClassifier()
    # retrieve feature vector and target vector
    counts, targets = feature_extraction.extract_features()
    if tfidf:
        counts, targets = feature_extraction.extract_features_tfidf()

    examples = ['versicherungen', 'dauerauftrag miete spenglerstr', 'norma', 'adac', 'nuernberger']
    example_counts = count_vectorizer.transform(examples)

    classifier.fit(counts, targets) #train the classifier
    predictions = classifier.predict(example_counts)

    print(predictions)

    print(metrics.classification_report())


def classify_w_cross_validation(plot=False):
    """
    Validate the classifier against unseen data using k-fold cross validation
    :param plot: choose whether to plot the confusion matrix with matplotlib
    """
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer()),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))
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
        Ploter.plot_and_show_confusion_matrix(confusion,
                                              category_names,
                                              normalize=True,
                                              title='NB Classifier normalized',
                                              save=True)

classify_w_cross_validation(False)