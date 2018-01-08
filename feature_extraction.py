import os
import numpy as np
import pandas
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from categories import Categories as cat
from preprocessing.nltk_preprocessor import NLTKPreprocessor
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
import nltk
import re

#nltk.download('punkt')
NEWLINE = '\n'
nastygrammer = '([,\/+]|\s{3,})' #regex

#root_path='F:\\Datasets\\Transaction-Dataset\\'
root_path='/Users/mgl/Training_Data/Transaction-Dataset/'

SOURCES = [
    (root_path+'barentnahme', cat.BARENTNAHME.name),
    (root_path+'finanzen', cat.FINANZEN.name),
    (root_path+'freizeitlifestyle', cat.FREIZEITLIFESTYLE.name),
    (root_path+'lebenshaltung', cat.LEBENSHALTUNG.name),
    (root_path+'mobilitaetverkehrsmittel', cat.MOBILITAETVERKEHR.name),
    (root_path+'versicherungen', cat.VERSICHERUNGEN.name),
    (root_path+'wohnenhaushalt', cat.WOHNENHAUSHALT.name)
]

SKIP_FILES = {'cmds'}

filepath = '/Users/mgl/Datasets/transactions_and_categories_new_cats.csv'

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


def extract_features_from_csv(tfidf=False):
    """
    builds a pandas data frame from csv file (semicolon separated)
    class name has to be in column 0
    columns 3, 4 and 8 need to be text, usage and owner
    :param filepath: path to csv file
    :return: word counts, targets
    """
    df = pandas.read_csv(filepath_or_buffer=filepath, encoding = "ISO-8859-1", delimiter=';', usecols=[0, 3, 4, 8])
    df['text'] = df.Buchungstext.str.replace(nastygrammer, ' ').str.lower() + \
                 ' ' + df.Verwendungszweck.str.replace(nastygrammer, ' ').str.lower() + \
                 ' ' + df.Beguenstigter.str.replace(nastygrammer, ' ').str.lower()
    #TODO normalize dataframe
    if tfidf:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, 2))

    targets = df['Kategorie'].values
    word_counts = vectorizer.fit_transform(df['text'].values.astype(str)).astype(float)

    return word_counts, targets


def extract_features():
    """
    Learn vocabulary and extract features using bag-of-words (word count)
    :return: term-document matrix
    """
    data = append_data_frames()
    #print(data['class'].values)
    count_vectorizer = CountVectorizer(ngram_range=(1, 2)) # bag-of-words

    #print(data['text'].values)
    targets = data['class'].values
    word_counts = count_vectorizer.fit_transform(data['text'].values).astype(float)

    return word_counts, targets


def extract_features_tfidf():
    """
    Learn vocabulary and extract features using tf-idf
    (term frequency - inverse document frequency)
    :return: term-document matrix, array of class labels
    """
    data = append_data_frames()
    tfidf_vectorizer = TfidfVectorizer()

    targets = data['class'].values
    tfidf = tfidf_vectorizer.fit_transform(data['text'].values)

    return tfidf, targets

def extract_example_features():
    data = append_data_frames()
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(data['text'].values)

    examples = ['advocard', 'xdfsd', 'versicherungen',
                'dauerauftrag miete spenglerstr', 'norma', 'adac',
                'nuernberger']
    example_counts = count_vectorizer.transform(examples)

    return example_counts, examples


#data = extract_features_from_csv('F:/Datasets/transactions_and_categories_full.csv')
data = extract_features_from_csv()
#print(data)
#counts, targets = extract_features()