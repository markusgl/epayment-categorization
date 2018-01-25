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
from file_handling.file_handler import FileHandler

#nltk.download('wordnet')
#nltk.download('punkt')

disturb_chars = '([\/+]|\s{3,})' #regex

#filepath = '/Users/mgl/Documents/OneDrive/Datasets/Labeled_transactions.csv'
#filepath = 'C:/Users/MG/OneDrive/Datasets/Labeled_transactions.csv'


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
        self.file_handler = FileHandler()
        self.vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                          ngram_range=(1,1),
                                          sublinear_tf=True,
                                          use_idf=True,
                                          max_df=0.5)

    def extract_features_from_csv(self):
        """
        builds a pandas data frame from csv file (semicolon separated)
        only columns category, bookingtext, usage and owner are necessary
        :return: word counts, targets
        """
        #df = pandas.read_csv(filepath_or_buffer=filepath, encoding = "ISO-8859-1", delimiter=',')
        df = self.file_handler.read_csv()
        df['values'] = df.bookingtext.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.usage.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.owner.str.replace(disturb_chars, ' ').str.lower()

        targets = df['category'].values
        # create term-document matrix
        word_counts = self.vectorizer.fit_transform(df['values'].values.astype(str)).astype(float)
        #word_counts = sp.hstack(text.apply(lambda col: self.vectorizer.fit_transform(col.values.astype(str)).astype(float)))

        return word_counts, targets

    def extract_example_features(self, examples):
        # df = pandas.DataFrame({'text': [' '.join(examples[0:3])], 'class': []})
        #df = DataFrame({'text': []})
        #df['text'] = [' '.join(examples[0:3])]

        example_counts = self.vectorizer.transform([' '.join(examples[0:3])])

        return example_counts

    def fetch_data(self):
        #df = pandas.read_csv(filepath_or_buffer=filepath, encoding = "UTF-8", delimiter=',')
        df = self.file_handler.read_csv()
        df['values'] = df.bookingtext.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.usage.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.owner.str.replace(disturb_chars, ' ').str.lower()

        targets = df['category'].values

        return df['values'].values.astype(str), targets

    def get_dataframes(self):
        df = self.file_handler.read_csv()
        df['values'] = df.bookingtext.str.replace(disturb_chars,
                                                  ' ').str.lower() + \
                       ' ' + df.usage.str.replace(disturb_chars,
                                                  ' ').str.lower() + \
                       ' ' + df.owner.str.replace(disturb_chars,
                                                  ' ').str.lower()


        return df['values'], df['category'].values

#fex = FeatureExtractor()
#w,c = fex.extract_features_from_csv()
#wln_test = WordNetLemmatizer()
#sbs = SnowballStemmer('german')
#print(wln_test.lemmatize('Statistik'))
#print(sbs.stem('Statistik'))

