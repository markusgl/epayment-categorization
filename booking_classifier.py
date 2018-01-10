
from categories import Categories as cat
from feature_extraction import FeatureExtractor
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pickle
from sklearn.externals import joblib
from booking import Booking


category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

class BookingClassifier:
    def __init__(self):
        # Load model and features from disk
        # TODO use pipelining
        self.clf = joblib.load('../../booking_classifier.pkl')
        self.feature_extractor = joblib.load('../../booking_features.pkl')

    def classify(self, term_list):
        """
        Classify examples and print prediction result
        :param: booking as list of owner, text and usage
        """
        example_counts = self.feature_extractor.extract_example_features(term_list)
        predict_probabilities = self.clf.predict_proba(example_counts)

        if max(max(predict_probabilities)) < 0.5:
            category = 'Sonstiges (unbekannt)'
        else:
            category = str(category_names[np.argmax(predict_probabilities)])

        print(category)
        return category

    def add_new_booking(self, booking):
        self._train_classifier()


    def _train_classifier(self, booking):
        """
        Train classifier and save to disk
        :return:
        """
        # self.clf = MultinomialNB(fit_prior=False)
        #self.clf = SGDClassifier(loss='hinge', alpha=0.001, max_iter=100)
        clf = SGDClassifier(loss='log', max_iter=100, tol=None)
        feature_extractor = FeatureExtractor()

        counts, targets = feature_extractor.extract_new_features(booking)
        clf.fit(counts, targets) # train the classifier

        # save model and classifier to disk
        joblib.dump(clf, 'booking_classifier.pkl')
        joblib.dump(feature_extractor, 'booking_features.pkl')


#classify = BookingClassifier()
#examples = ['KARTENZAHLUNG', '2017-09-03T08:41:04 Karte1 2018-12', 'SUPOL NURNBERG AUSSERE BAYREUTHER STR']
#examples = ['blub', 'bla', 'bli']
#classify.classify(examples)