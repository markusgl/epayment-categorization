import numpy as np
from nltk import WhitespaceTokenizer
from pymongo import MongoClient
from sklearn.externals import joblib
from sklearn.svm import SVC
from pathlib import Path
from categories import Categories as cat
from categories import FallbackCategorie as fbcat
from feature_extraction import FeatureExtractor
import re
import os

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class BookingClassifier:
    def __init__(self, flaskr=None):
        client = MongoClient('mongodb://localhost:27017/')
        self.db = client.companyset

        # Load model and features from disk
        # TODO use pipelining
        self.resource_path = str(ROOT_DIR + '/resources/')
        #if flaskr:
        #    self.pickle_path = '../resources/'
        #else:
        #    self.pickle_path = 'resources/'
        if Path(self.resource_path + 'booking_classifier.pkl').is_file() and Path(self.resource_path + 'booking_features.pkl').is_file():
            self.load_model()
        else:
            print('No model found. Start training classifier...')
            self.train_classifier()

    def match_creditor_id(self, booking):
        """
        Compares creditor id with entries in mongodb
        :param booking: booking following BookingSchema in booking.py
        :return: category if creditor exists in mongodb
                 -1 if no entry was found
        """
        try:
            regex = re.compile(booking.creditor_id, re.IGNORECASE)
            db_entry = self.db.companies.find_one({"creditorid": regex})
            return db_entry['category']
        except:
            return -1

    def classify(self, booking):
        """
        Classify booking and return prediction result
        :param booking: booking following BookingSchema in booking.py
        :return: category as string
        """
        # check if creditor_id is already known
        category = self.match_creditor_id(booking)
        if category != -1:
            return str(category), "0"

        # check if creditor_id is in purpose code
        wst = WhitespaceTokenizer()
        tokens = wst.tokenize(booking.usage)
        try:
            print(tokens[tokens.index("Einreicher-ID") + 1])
            booking.creditor_id = tokens[tokens.index("Einreicher-ID") + 1]
        except ValueError:
            print("No SEPA purpose code found")

        # start text analysis
        term_list = booking.text + ' ' + booking.usage + ' ' + booking.owner
        word_counts = self.feature_extractor.extract_termlist_features(term_list)
        predict_probabilities = self.clf.predict_proba(word_counts)
        #category = self.clf.predict(example_counts)

        # if max prediction probability is less than 70% assume that the booking category is unknown
        prob = str(max(max(predict_probabilities)))
        #print("P:" + str(prob))
        #print("Highest ranked category: " + str(category_names[np.argmax(predict_probabilities)]))

        if max(max(predict_probabilities)) < 0.7:
            category = str(fbcat.SONSTIGES.name)  # fallback category
        else:
            category = str(category_names[np.argmax(predict_probabilities)])

        #print(category)
        return str(category), predict_probabilities

    def add_new_booking(self, booking):
        self.train_classifier()

    def train_classifier(self):
        """
        Train classifier and save to disk
        :return:
        """
        feature_extractor = FeatureExtractor.tfidf(ngram_range=(1, 2), max_df=0.5, use_idf=False,
                                                   sublinear_tf=True)
        clf = SVC(kernel='linear', C=100, gamma=0.01, decision_function_shape='ovo', probability=True)
        counts, targets = feature_extractor.extract_features_from_csv

        print('start training...')
        clf.fit(counts, targets)  # train the classifier
        print('training finished. start dumping model...')

        # save model and classifier to disk
        joblib.dump(clf, self.resource_path + 'booking_classifier.pkl')
        joblib.dump(feature_extractor, self.resource_path + 'booking_features.pkl')
        self.load_model()

    def load_model(self):
        print('loading model...')
        self.clf = joblib.load(Path(self.resource_path + 'booking_classifier.pkl'))
        self.feature_extractor = joblib.load(Path(self.resource_path + 'booking_features.pkl'))

