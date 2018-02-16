
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from categories import Categories as cat
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import string
import nltk
import re
from booking import Booking
import scipy as sp
from file_handling.file_handler import FileHandler
import editdistance
import pandas as pd

#nltk.download('wordnet')
#nltk.download('punkt')
nltk.download('stopwords')
disturb_chars = '([\/+]|\s{3,})' #regex


class StemTokenizer(object):
    def __init__(self):
        self.sbs = SnowballStemmer('german', ignore_stopwords=True)

    def __call__(self, doc):
        # TreeBankTokenizer
        return [self.sbs.stem(t) for t in word_tokenize(doc)]


class FeatureExtractor:
    def __init__(self, vectorizer):
        self.file_handler = FileHandler()
        self.vectorizer = vectorizer

    @classmethod
    def bow(cls, **kwargs):
        vectorizer = CountVectorizer(**kwargs)
        return cls(vectorizer)

    @classmethod
    def tfidf(cls, **kwargs):
        vectorizer = TfidfVectorizer(**kwargs)
        return cls(vectorizer)

    @property
    def extract_features_from_csv(self):
        """
        builds a pandas data frame from csv file (semicolon separated)
        only columns category, bookingtext, usage and owner are necessary
        :return: word counts, targets
        """
        #df = self.file_handler.read_csv('C:/Users/MG/OneDrive/Datasets/Labeled_transactions.csv')
        df = self.file_handler.read_csv('/Users/mgl/Documents/OneDrive/Datasets/Labeled_transactions.csv')
        #df = self.file_handler.read_csv('C:/Users/MG/OneDrive/Datasets/Labeled_transactions_mobilitaet.csv')

        #df['values'] = df.bookingtext.str.replace(disturb_chars, ' ').str.lower() + \
        #             ' ' + df.usage.str.replace(disturb_chars, ' ').str.lower() + \
        #             ' ' + df.owner.str.replace(disturb_chars, ' ').str.lower()
        df['values'] = df[['bookingtext', 'usage', 'owner']].astype(str)\
                                                            .sum(axis=1)\
                                                            .replace(disturb_chars, ' ')\
                                                            .str.lower()

        targets = df['category'].values

        # create term-document matrix
        word_counts = self.vectorizer.fit_transform(df['values'].values.astype(str)).astype(float)
        #word_counts = sp.hstack(text.apply(lambda col: self.vectorizer.fit_transform(col.values.astype(str)).astype(float)))

        return word_counts, targets

    def extract_termlist_features(self, term_list):
        example_counts = self.vectorizer.transform([' '.join(term_list[0:3])])

        return example_counts

    def fetch_data(self):
        #df = self.file_handler.read_csv('C:/tmp/Labeled_transactions_sorted_same_class_amount.csv')
        df = self.file_handler.read_csv('/Users/mgl/Documents/OneDrive/Datasets/Labeled_transactions_sorted_same_class_amount.csv')

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

    def get_jaccard(self):
        df = self.file_handler.read_csv('C:/Users/MG/OneDrive/Datasets/Labeled_transactions_mobilitaet.csv')
        #df = self.file_handler.read_csv('C:/Users/MG/OneDrive/Datasets/Labeled_transactions_barentnahme.csv')
        #df = self.file_handler.read_csv('C:/Users/MG/OneDrive/Datasets/Labeled_transactions_versicherungen.csv')

        sum = 0
        count = 0
        for index, row in df.iterrows():
            str1 = row.bookingtext.replace(disturb_chars, ' ').lower() + \
                           ' ' + row.usage.replace(disturb_chars, ' ').lower() + \
                           ' ' + row.owner.replace(disturb_chars, ' ').lower()

            for index, row2 in df.iterrows():
                str2 = row2.bookingtext.replace(disturb_chars, ' ').lower() + \
                               ' ' + row2.usage.replace(disturb_chars, ' ').lower() + \
                               ' ' + row2.owner.replace(disturb_chars, ' ').lower()

                #a = set(str1.split())
                #b = set(str2.split())

                a = set(str1)
                b = set(str2)

                c = a.intersection(b)
                count += 1
                sum += float(len(c)) / (len(a) + len(b) - len(c))

        print(sum / count)

    def get_levenshtein(self):
        #df = self.file_handler.read_csv('C:/Users/MG/OneDrive/Datasets/Labeled_transactions_mobilitaet.csv')
        df = self.file_handler.read_csv('C:/Users/MG/OneDrive/Datasets/Labeled_transactions_barentnahme.csv')
        #df = self.file_handler.read_csv('C:/Users/MG/OneDrive/Datasets/Labeled_transactions_versicherungen.csv')

        sum = 0
        count = 0
        for index, row in df.iterrows():
            str1 = row.bookingtext.replace(disturb_chars, ' ').lower() + \
                           ' ' + row.usage.replace(disturb_chars, ' ').lower() + \
                           ' ' + row.owner.replace(disturb_chars, ' ').lower()

            for index, row2 in df.iterrows():
                str2 = row2.bookingtext.replace(disturb_chars, ' ').lower() + \
                               ' ' + row2.usage.replace(disturb_chars, ' ').lower() + \
                               ' ' + row2.owner.replace(disturb_chars, ' ').lower()
                count += 1
                sum += editdistance.eval(row, row2)

        print(sum / count)

#fe = FeatureExtractor(ngramrange=(1, 1), maxdf=0.5, useidf=True, sublinear=True)
#fe.get_jaccard()
#fe.get_levenshtein()

#fex = FeatureExtractor()
#w,c = fex.extract_features_from_csv()
#wln_test = WordNetLemmatizer()
#sbs = SnowballStemmer('german')
#print(wln_test.lemmatize('Statistik'))
#print(sbs.stem('Statistik'))
