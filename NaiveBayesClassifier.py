''' Multi-Class categorization vor e-payments using Naive Bayes classifier'''

import os
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer

NEWLINE = '\n'

BARENTNAHME = 'barentnahme'
FINANZEN = 'finanzen'
FREIZEITLIFESTYLE = 'freizeitlifestyle'
LEBENSHALTUNG = 'lebenshaltung'
MOBILITAETVERKEHR = 'mobilitaetverkehrsmittel'
VERSICHERUNGEN = 'versicherungen'
WOHNENHAUSHALT = 'wohnenhaushalt'


SOURCES = [
    ('F:\\Datasets\\Transaction-Dataset\\barentnahme',      BARENTNAHME),
    ('F:\\Datasets\\Transaction-Dataset\\finanzen',    FINANZEN),
    ('F:\\Datasets\\Transaction-Dataset\\freizeitlifestyle',  FREIZEITLIFESTYLE),
    ('F:\\Datasets\\Transaction-Dataset\\lebenshaltung',   LEBENSHALTUNG),
    ('F:\\Datasets\\Transaction-Dataset\\mobilitaetverkehrsmittel',     MOBILITAETVERKEHR),
    ('F:\\Datasets\\Transaction-Dataset\\versicherungen', VERSICHERUNGEN),
    ('F:\\Datasets\\Transaction-Dataset\\wohnenhaushalt',          WOHNENHAUSHALT)
]

SKIP_FILES = {'cmds'}

''' iterate through all files an yield the text body '''
def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    #past_header, lines = False, []
                    lines = []
                    f = open(file_path, encoding="iso-8859-1")
                    for line in f:
                        lines.append(line)
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


''' build a dataset from transaction purpose, reciver and booking type '''
def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

''' concatenate DataFrames using pandas append method '''
data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = data.reindex(numpy.random.permutation(data.index))
'''
    ##### WITHOUT PIPELINING  #####
'''
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(data['text'].values)

classifier = MultinomialNB()
#classifier = BernoulliNB()
#classifier = GaussianNB()
targets = data['class'].values
classifier.fit(counts, targets)

examples = ['versicherungen', 'dauerauftrag miete spenglerstr', 'norma', 'adac', 'nuernberger']
example_counts = count_vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
#for prediction in predictions:
#    for example in examples:
#        print(prediction + ": " + example)
print(predictions)


'''
    ###### USE PIPELINING ####### 
'''
pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',         MultinomialNB())
])

# Cross validation
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

print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
