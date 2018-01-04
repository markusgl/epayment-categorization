import feature_extraction
from sklearn.cross_validation import KFold
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from plotter import Plotter
from categories import Categories as cat

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

def classify_w_cross_validation(plot=False):
    """
    Validate the classifier against unseen data using k-fold cross validation
    :param plot: choose whether to plot the confusion matrix with matplotlib
    """
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf_transformer', TfidfTransformer()),
        ('classifier', KNeighborsClassifier())
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
        Plotter.plot_and_show_confusion_matrix(confusion,
                                               category_names,
                                               normalize=True,
                                               title='K-Nearest-Neighbours Classifier',
                                               save=True)
classify_w_cross_validation(plot=True)