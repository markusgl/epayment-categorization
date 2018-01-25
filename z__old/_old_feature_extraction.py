import os
import numpy as np
import pandas
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from categories import Categories as cat
from preprocessing.nltk_preprocessor import NLTKPreprocessor
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
import re
from booking import Booking
import scipy as sp

#nltk.download('wordnet')
#nltk.download('punkt')
NEWLINE = '\n'
nastygrammer = '([\/+]|\s{3,})' #regex

#root_path='F:\\Datasets\\Transaction-Dataset\\'
root_path='/Users/mgl/Datasets/Transaction-Dataset/'

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

#filepath = '/Users/mgl/Datasets/transactions_and_categories_new_cats.csv'
#filepath = 'F:/Datasets/transactions_and_categories_new_cats.csv'
#filepath = 'F:/Datasets/Labeled_transactions.csv'


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class OldFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), sublinear_tf=True, max_df=0.5)

    def extract_features_from_csv(self):
        """
        builds a pandas data frame from csv file (semicolon separated)
        class name has to be in column 0
        columns 3, 4 and 8 need to be text, usage and owner
        :param filepath: path to csv file
        :return: word counts, targets
        """
        df = pandas.read_csv(filepath_or_buffer=filepath, encoding = "ISO-8859-1", delimiter=';', usecols=[0, 3, 4, 8])
        df['text'] = df.text.str.replace(nastygrammer, ' ').str.lower() + \
                     ' ' + df.usage.str.replace(nastygrammer, ' ').str.lower() + \
                     ' ' + df.owner.str.replace(nastygrammer, ' ').str.lower()

        #TODO normalize dataframe with ntlk

        targets = df['Kategorie'].values
        word_counts = self.vectorizer.fit_transform(df['text'].values.astype(str)).astype(float)
        #word_counts = sp.hstack(text.apply(lambda col: self.vectorizer.fit_transform(col.values.astype(str)).astype(float)))

        return word_counts, targets

    def extract_example_features(self, examples):
        # df = pandas.DataFrame({'text': [' '.join(examples[0:3])], 'class': []})
        #df = DataFrame({'text': []})
        #df['text'] = [' '.join(examples[0:3])]

        example_counts = self.vectorizer.transform([' '.join(examples[0:3])])

        return example_counts

    def extract_new_features(self, booking):
        """
        class name has to be in column 0
        columns 3, 4 and 8 need to be text, usage and owner
        :param booking
        :return: word counts, targets
        """
        # Load base data set
        df = pandas.read_csv(filepath_or_buffer=filepath, encoding = "ISO-8859-1", delimiter=';', usecols=[0, 3, 4, 8])
        df['text'] = df.Buchungstext.str.replace(nastygrammer, ' ').str.lower() + \
                     ' ' + df.Verwendungszweck.str.replace(nastygrammer, ' ').str.lower() + \
                     ' ' + df.Beguenstigter.str.replace(nastygrammer, ' ').str.lower()

        # add new booking to data set
        #df['text'].append(booking.text + ' ' + booking.usage + ' ' + booking.owner)

        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), sublinear_tf=True, max_df=0.5)
        targets = df['Kategorie'].values
        word_counts = vectorizer.fit_transform(df['text'].values.astype(str)).astype(float)

        return word_counts, targets


### DEPRECATED ###

def read_files(path):
    """
    iterate through all files an yield the text body
    :param path:
    :return: file path, content
    """
    for root_path, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root_path, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root_path, file_name)
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



#examples_entry = ['KARTENZAHLUNG', '2017-09-03T08:41:04 Karte1 2018-12', 'SUPOL NURNBERG AUSSERE BAYREUTHER STR']
#extract_example_features(examples_entry)
#data = extract_features_from_csv('F:/Datasets/transactions_and_categories_full.csv')

#data = extract_features_from_csv()
#print(data)
#counts, targets = extract_features()