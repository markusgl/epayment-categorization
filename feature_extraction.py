import os
import numpy as np
import pandas
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from categories import Categories as cat
from preprocessing.nltk_preprocessor import NLTKPreprocessor
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import string
import nltk
import re
from booking import Booking
import scipy as sp

#nltk.download('wordnet')
#nltk.download('punkt')

nastygrammer = '([\/+]|\s{3,})' #regex

#filepath = '/Users/mgl/Datasets/transactions_and_categories_new_cats.csv'
filepath = 'C:/Users/MG/OneDrive/Datasets/Labeled_transactions.csv'


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.sbs = SnowballStemmer('german')

    def __call__(self, doc):
        # TreeBankTokenizer
        return [self.sbs.stem(t) for t in word_tokenize(doc)]
        #return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class FeatureExtractor:
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
        df = pandas.read_csv(filepath_or_buffer=filepath, encoding = "ISO-8859-1", delimiter=',')
        df['value'] = df.bookingtext.str.replace(nastygrammer, ' ').str.lower() + \
                     ' ' + df.usage.str.replace(nastygrammer, ' ').str.lower() + \
                     ' ' + df.owner.str.replace(nastygrammer, ' ').str.lower()

        targets = df['category'].values
        word_counts = self.vectorizer.fit_transform(df['value'].values.astype(str)).astype(float)
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

#fex = FeatureExtractor()
#w,c = fex.extract_features_from_csv()
wln_test = WordNetLemmatizer()
sbs = SnowballStemmer('german')
print(wln_test.lemmatize('Statistik'))
print(sbs.stem('Statistik'))

