from sklearn import tree
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from plotter import Ploter
import feature_extraction
from categories import Categories as ctg


# TODO - work in progess...

def test_with_examples():
    count_vectorizer = CountVectorizer()
    counts, targets = feature_extraction.extract_features()
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(counts, targets)

    examples = ['versicherungen', 'dauerauftrag miete spenglerstr', 'norma', 'adac', 'nuernberger']
    example_counts = count_vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print(predictions)


'''
    ###### USE PIPELINING ####### 
'''
pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',         tree.DecisionTreeClassifier())
])

''' ###### CROSS VALIDATION ####### 
Validate the classifier against unseen data using k-fold cross validation
'''
data = feature_extraction.append_data_frames()
k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
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

class_names = [ctg.BARENTNAHME.name, ctg.FINANZEN.name,
               ctg.FREIZEITLIFESTYLE.name, ctg.LEBENSHALTUNG.name,
               ctg.MOBILITAETVERKEHR.name, ctg.VERSICHERUNGEN.name,
               ctg.WOHNENHAUSHALT.name]

Ploter.plot_and_show_confusion_matrix(confusion, class_names, normalize=True, title='Decision Tree normalized', save=False)