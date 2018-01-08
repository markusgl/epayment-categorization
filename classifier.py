from matplotlib.colors import ListedColormap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from plotter import Plotter
import feature_extraction
from categories import Categories as cat
from sklearn.externals import joblib
from preprocessing.normalization import Normalizer


category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

class Classifier:
    def classify(self, term_list, tfidf=False):
        """
        Classify examples and print prediction result
        :param bernoulliNB: use Bernoulli Model - default is Multinomial NB
        :param tfidf: use TF-IDF - default is bag-of-words (word count)
        """
        normalizer = Normalizer()
        fields = normalizer.normalize_text_fields(term_list)
        data = feature_extraction.append_data_frames()
        count_vectorizer = CountVectorizer()
        count_vectorizer.fit_transform(data['text'].values)

        classifier = joblib.load('svm_classifier.pkl')

        # feature vector and target vector
        counts, targets = feature_extraction.extract_features()
        if tfidf:
            counts, targets = feature_extraction.extract_features_tfidf()

        example_counts = count_vectorizer.transform(fields)
        classifier.fit(counts, targets)  # train the classifier

        predict_probabilities = classifier.predict_proba(example_counts)
        category = " "
        # print predictions for each class
        for i in range(len(predict_probabilities)):
            prob = predict_probabilities[i]

            # classify_result
            if min(prob) == max(prob):
                category ='Sonstiges'
            else:
                m = max(prob)
                result = [i for i, j in enumerate(prob) if j == m]
                category = str(category_names[result[0]])
        #print(category)
        return category