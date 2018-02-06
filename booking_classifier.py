
import numpy as np
from pymongo import MongoClient
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from pathlib import Path
from categories import Categories as cat
from categories import FallbackCategorie as fbcat
from feature_extraction import FeatureExtractor
import re
import pprint
from booking import Booking

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

class BookingClassifier:
    def __init__(self):
        # Load model and features from disk
        client = MongoClient('mongodb://localhost:27017/')
        self.db = client.companyset

        # TODO use pipelining
        if Path('booking_classifier.pkl').is_file() and Path('booking_features.pkl'):
            print('loading model...')
            self.clf = joblib.load('booking_classifier.pkl')
            self.feature_extractor = joblib.load('booking_features.pkl')
        else:
            print('No model found. Start training classifier...')
            self._train_classifier()

    def match_creditor_id(self, booking):
        if booking.creditor_id:
            regex = re.compile(booking.creditor_id, re.IGNORECASE)
            db_entry = self.db.companies.find_one({"creditorid": regex})
            return db_entry['category']
        return -1

    def classify(self, term_list):
        """
        Classify examples and print prediction result
        :param: booking as list of owner, text and usage
        """
        # check if creditor_id is already known
        category = self.match_creditor_id()
        if category != -1:
            return category

        # TODO SEPA Purpose Code
        # check if creditor_id is in purpose code

        # othwerwise start text analysis
        word_counts = self.feature_extractor.extract_termlist_features(term_list)
        predict_probabilities = self.clf.predict_proba(word_counts)
        #category = self.clf.predict(example_counts)


        print(max(max(predict_probabilities)))
        if max(max(predict_probabilities)) < 0.7:
            category = str(fbcat.SONSTIGES.name) # fallback category
        else:
            category = str(category_names[np.argmax(predict_probabilities)])

        print(category)
        return str(category)

    def add_new_booking(self, booking):
        self._train_classifier()

    def _train_classifier(self):
        """
        Train classifier and save to disk
        :return:
        """
        # clf = MultinomialNB(fit_prior=False)
        #clf = SGDClassifier(loss='hinge', alpha=0.001, max_iter=100)
        clf = SVC(kernel='linear', C=10, decision_function_shape='ovr', probability=True)
        #clf = SGDClassifier(loss='log', max_iter=100, tol=None, shuffle=True)
        feature_extractor = FeatureExtractor()

        counts, targets = feature_extractor.extract_features_from_csv()
        print('start training...')
        clf.fit(counts, targets) # train the classifier
        print('training finished. start dumping model...')

        # save model and classifier to disk
        joblib.dump(clf, 'booking_classifier.pkl')
        joblib.dump(feature_extractor, 'booking_features.pkl')


#clf = BookingClassifier()
#clf._train_classifier()
#examples = ['KARTENZAHLUNG', '2017-09-03T08:41:04 Karte1 2018-12', 'SUPOL NURNBERG AUSSERE BAYREUTHER STR']
#examples = ['bli', 'bla', 'blub']
#clf.classify(examples)
