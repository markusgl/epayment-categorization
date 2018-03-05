
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from file_handling.file_handler import FileHandler
import editdistance
from pathlib import Path
import os

nltk.download('punkt')
nltk.download('stopwords')
disturb_chars = '(["\/+]|\s{3,})' #regex


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
        df = self.file_handler.read_csv('data/Labeled_transactions.csv')

        df['values'] = df.bookingtext.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.usage.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.owner.str.replace(disturb_chars, ' ').str.lower()

        targets = df['category'].values
        # create term-document matrix
        word_counts = self.vectorizer.fit_transform(df['values'].values.astype(str)).astype(float)

        return word_counts, targets

    def extract_features_from_csv_flask(self):
        """
        builds a pandas data frame from csv file (semicolon separated)
        only columns category, bookingtext, usage and owner are necessary
        :return: word counts, targets
        """

        df = self.file_handler.read_csv()

        df['values'] = df.bookingtext.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.usage.str.replace(disturb_chars, ' ').str.lower() + \
                     ' ' + df.owner.str.replace(disturb_chars, ' ').str.lower()

        targets = df['category'].values
        # create term-document matrix
        word_counts = self.vectorizer.fit_transform(df['values'].values.astype(str)).astype(float)

        return word_counts, targets


    def extract_termlist_features(self, term_list):
        term_list = term_list.replace(disturb_chars, ' ').lower()
        word_counts = self.vectorizer.transform([term_list]).astype(float)

        return word_counts

    def fetch_data(self):
        df = self.file_handler.read_csv('data/Labeled_transactions.csv')

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

    def get_levenshtein(self):
        # TODO choose only transactions with same category
        df = self.file_handler.read_csv('data/Labeled_transactions.csv')

        df['values'] = df.bookingtext.str.replace(disturb_chars,
                                                  ' ').str.lower() + \
                       ' ' + df.usage.str.replace(disturb_chars,
                                                  ' ').str.lower() + \
                       ' ' + df.owner.str.replace(disturb_chars,
                                               ' ').str.lower()
        sum = 0
        count = 0
        for index, row in df['values'].iteritems():
            for index, row2 in df['values'].iteritems():
                count += 1
                sum += editdistance.eval(row, row2)

        print(sum / count)

#fn = os.path.join(os.path.dirname(__file__), '/data/Labeled_transactions.csv')
