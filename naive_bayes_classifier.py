"""
Multi-Class categorization for e-payments using Naive Bayes classifier
"""

import numpy as np
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
from sklearn.preprocessing import StandardScaler


category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

class NBClassifier:
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

        classifier = MultinomialNB(fit_prior=False)
        #if bernoulliNB:
        #    classifier = BernoulliNB()
        #classifier = joblib.load('nb_classifier.pkl')

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


    def classify_examples(self, examples, bernoulliNB=False,
                          tfidf=False, persist=False,
                          probabilites=False):
        """
        Classify examples and print prediction result
        :param bernoulliNB: use Bernoulli Model - default is Multinomial NB
        :param tfidf: use TF-IDF - default is bag-of-words (word count)
        """
        data = feature_extraction.append_data_frames()
        count_vectorizer = CountVectorizer()
        count_vectorizer.fit_transform(data['text'].values)

        classifier = MultinomialNB(fit_prior=False)
        if bernoulliNB:
            classifier = BernoulliNB()
        #classifier = joblib.load('nb_classifier.pkl')

        # retrieve feature vector and target vector
        counts, targets = feature_extraction.extract_features()
        if tfidf:
            counts, targets = feature_extraction.extract_features_tfidf()

        example_counts = count_vectorizer.transform(examples)

        classifier.fit(counts, targets)  # train the classifier
        if persist:
            joblib.dump(classifier, 'nb_classifier.pkl')


        predict_probabilities = classifier.predict_proba(example_counts)
        # print predictions for each class
        classify_result = []
        for i in range(len(predict_probabilities)):
            print(examples[i])
            prob = predict_probabilities[i]

            # print probabilites for individual classes
            if probabilites:
                for j in range(7):
                    print(category_names[j] + ": " + str(round(prob[j] * 100, 2)) + "%")

            # classify_result
            if min(prob) == max(prob):
                print("Resultat: Sonstiges")
                classify_result.append('Sonstiges')
            else:
                m = max(prob)
                result = [i for i, j in enumerate(prob) if j == m]
                print("Resultat: " + str(category_names[result[0]]))

            print(" ")

        predictions = classifier.predict(example_counts)
        print(predictions)



    def classify_w_cross_validation(self, bernoulliNB=False, plot=False):
        """
        Validate the classifier against unseen data using k-fold cross validation
        - Uses pipelining: feature extraction and classification task are merged into one operation
        :param plot: choose whether to plot the confusion matrix with matplotlib
        """
        pipeline = Pipeline([
            ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf_transformer', TfidfTransformer()),
            ('classifier', MultinomialNB(fit_prior=False))
        ])

        data = feature_extraction.append_data_frames()
        #k_fold = KFold(n=len(data), n_folds=6)
        k_fold = KFold(n_splits=6)

        scores = []
        confusion = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0]])
        for train_indices, test_indices in k_fold.split(data):
            train_text = data.iloc[train_indices]['text'].values
            train_y = data.iloc[train_indices]['class'].values.astype(str)

            test_text = data.iloc[test_indices]['text'].values
            test_y = data.iloc[test_indices]['class'].values.astype(str)

            pipeline.fit(train_text, train_y)
            predictions = pipeline.predict(test_text)

            confusion += confusion_matrix(test_y, predictions)
            # score = f1_score(test_y, predictions, average='samples')
            score = accuracy_score(test_y, predictions)
            scores.append(score)

        print('Total transactions classified:', len(data))
        print('Score:', sum(scores) / len(scores))
        print('Confusion matrix:')
        print(confusion)

        if plot:
            Plotter.plot_and_show_confusion_matrix(confusion,
                                                   category_names,
                                                   normalize=True,
                                                   title='NB Classifier normalized',
                                                   save=True)


    def classify_new_cross_validation(self, plot=False, fit=False):
        """
        Validate the classifier against unseen data using k-fold cross validation
        """
        counts, target = feature_extraction.extract_features_from_csv()

        # hold 20% out for testing
        X_train, X_test, y_train, y_test = train_test_split(counts, target, test_size=0.2, random_state=0)

        #X_train.shape, y_train.shape
        #X_test.shape, y_test.shape

        clf = MultinomialNB(fit_prior=False).fit(X_train, y_train)
        if fit:
            sc = StandardScaler(with_mean=False)
            sc.fit(X_train)
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            clf = MultinomialNB(fit_prior=False).fit(X_train_std, y_train)

        predictions = clf.predict(X_test)

        # scores
        #print(clf.score(X_test_std, y_test))
        scores = cross_val_score(clf, counts, target, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


        confusion = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0]])

        confusion += confusion_matrix(y_test, predictions)
        print(confusion)

        if plot:
            Plotter.plot_and_show_confusion_matrix(confusion,
                                                   category_names,
                                                   normalize=True,
                                                   title='NB Classifier normalized',
                                                   save=True)

        # plot decision boundaries
        """
        df = feature_extraction.append_data_frames()
        y = df.iloc[0:100, 0].values
        X = df.iloc[0:100, 1].values
        
        plt.scatter(X[:50], y[:],
                    color='red', marker='o', label='Barentnahme')
       #plt.scatter(X[0:50], X[50:100],
       #             color='blue', marker='x', label='Lebenshaltung')

        plt.xlabel(' length ')
        plt.ylabel('petal length ')
        plt.legend(loc='upper left')
        plt.show()
        """
        '''
        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))
        self.plot_decision_regions(X=X_combined_std, y=y_combined, classifier=clf)
        plt.xlabel('test')
        plt.ylabel('test1')
        plt.show()
        '''



#if __name__ == 'main':
clf = NBClassifier()
#clf.classify_examples(['advocard', 'versicherung', 'xsadadf', 'miete'])
#clf.classify_w_cross_validation()
clf.classify_new_cross_validation()

