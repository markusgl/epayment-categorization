import os
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from categories import Categories as cat

NEWLINE = '\n'

#root_path='F:\\Datasets\\Transaction-Dataset\\'
root_path='/Users/mgl/Training_Data/Transaction-Dataset/'

SOURCES = [
    (root_path +'barentnahme', cat.BARENTNAHME),
    (root_path+'finanzen', cat.FINANZEN),
    (root_path+'freizeitlifestyle', cat.FREIZEITLIFESTYLE),
    (root_path+'lebenshaltung', cat.LEBENSHALTUNG),
    (root_path+'mobilitaetverkehrsmittel', cat.MOBILITAETVERKEHR),
    (root_path+'versicherungen', cat.VERSICHERUNGEN),
    (root_path+'wohnenhaushalt', cat.WOHNENHAUSHALT)
]

SKIP_FILES = {'cmds'}

def read_files(path):
    """
    iterate through all files an yield the text body
    :param path:
    :return: file path, content
    """
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


def build_data_frame(path, classification):
    """
    build a dataset from transaction bodies containing purpose, receiver and
    booking type
    :param path:
    :param classification:
    :return: pandas data frame
    """
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

def append_data_frames():
    """
    concatenate DataFrames using pandas append method
    :return: pandas data frame
    """
    data = DataFrame({'text': [], 'class': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))

    return data.reindex(np.random.permutation(data.index))

def extract_features():
    """
    Learn vocabulary and extract features using bag-of-words (word count)
    :return: term-document matrix
    """
    data = append_data_frames()
    count_vectorizer = CountVectorizer() # bag-of-words

    targets = data['class'].values
    word_counts = count_vectorizer.fit_transform(data['text'].values)

    return word_counts, targets


def extract_features_tfidf():
    """
    Learn vocabulary and extract features using tf-idf
    (term frequency - inverse document frequency)
    :return: term-document matrix
    """
    data = append_data_frames()
    tfidf_transformer = TfidfTransformer()

    targets = data['class'].values
    tfidf = tfidf_transformer.fit_transform(data['text'].values)

    return tfidf, targets
