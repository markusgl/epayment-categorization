"""
- Compares different classifiers for booking classification
- Model selection for prototype system
"""

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix,  accuracy_score, \
    f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from categories import Categories as cat
from comparison.plotter import Plotter
from feature_extraction import FeatureExtractor
from collections import Counter

category_names = [cat.BARENTNAHME.name, cat.FINANZEN.name,
                  cat.FREIZEITLIFESTYLE.name, cat.LEBENSHALTUNG.name,
                  cat.MOBILITAETVERKEHR.name, cat.VERSICHERUNGEN.name,
                  cat.WOHNENHAUSHALT.name]

category_names_reverse = ['barentnahme', 'finanzen', 'freizeitlifestyle',
                         'lebenshaltung', 'mobilitaetverkehrsmittel',
                          'versicherungen','wohnenhaushalt']


def get_classweight(classes, smoth_factor=0.02):
    counter = Counter(classes)

    if smoth_factor > 0:
        p = max(counter.values()) * smoth_factor
        for k in counter.keys():
            counter[k] += p

        majority = max(counter.values())

        return {cls: float(majority / count) for cls, count in counter.items()}


def classify(bow=False, plot=False, multinomial_nb=False, bernoulli_nb=False, knn=False, support_vm=False,
             svm_sgd=False,
             decision_tree=False, random_forest=False, persist=False, logistic_regression=False):
    """
    Validate the classifier against unseen data using k-fold cross validation
    """

    if multinomial_nb:
        clf_title = 'Multinomial NB'
        if bow:
            vectorizer_title = 'Bag-of-Words'
            counts, targets = FeatureExtractor.bow(max_df=0.25,
                                                     ngram_range=(1, 3)).extract_features_from_csv
            clf = MultinomialNB(alpha=1e-05)
        else:
            vectorizer_title = 'TF-IDF'
            counts, targets = FeatureExtractor.tfidf(analyzer='char',
                                                     max_df=0.5,
                                                     ngram_range=(1,4),
                                                     norm='l1',
                                                     sublinear_tf=False,
                                                     use_idf=True).extract_features_from_csv
            clf = MultinomialNB(alpha=1e-07)
    elif bernoulli_nb:
        clf_title = 'Bernoulli NB'
        if bow:
            vectorizer_title = 'Bag-of-Words'
            counts, targets = FeatureExtractor.bow(max_df=0.25,
                                                     ngram_range=(1, 3)).extract_features_from_csv
            clf = BernoulliNB(alpha=1e-05)
        else:
            vectorizer_title = 'TF-IDF'
            counts, targets = FeatureExtractor.tfidf(analyzer='word',
                                                     max_df=0.25,
                                                     ngram_range=(1,3),
                                                     norm='l1',
                                                     sublinear_tf=True,
                                                     use_idf=True).extract_features_from_csv
            clf = BernoulliNB(alpha=1e-05)
    elif knn:
        clf_title = 'K-Nearest-Neighbour'
        if bow:
            vectorizer_title = 'Bag-of-Words'
            counts, targets = FeatureExtractor.bow(max_df=0.5,
                                                   ngram_range=(1, 1)).extract_features_from_csv
            clf = KNeighborsClassifier(weights='distance', n_neighbors=2,
                                       leaf_size=20, algorithm='auto')
        else:
            vectorizer_title = 'TF-IDF'
            counts, targets = FeatureExtractor.tfidf(analyzer='word',
                                                     max_df=0.5,
                                                     ngram_range=(1,1),
                                                     norm='l1',
                                                     sublinear_tf=True,
                                                     use_idf=True).extract_features_from_csv
            clf = KNeighborsClassifier(weights='distance', n_neighbors=2, leaf_size=20, algorithm='auto')

    elif support_vm:
        clf_title = 'Support Vector Machine'
        if bow:
            vectorizer_title = 'Bag-of-Words'
            counts, targets = FeatureExtractor.tfidf(max_df=0.5,
                                                     ngram_range=(1, 2)).extract_features_from_csv
            clf = SVC(kernel='sigmoid', C=100, gamma=0.01,
                      decision_function_shape='ovo', probability=True)
        else:
            vectorizer_title = 'TF-IDF'
            counts, targets = FeatureExtractor.tfidf(analyzer='char',
                                                     max_df=1.0,
                                                     ngram_range=(1, 4),
                                                     norm='l2',
                                                     use_idf=False,
                                                     sublinear_tf=True).extract_features_from_csv

            clf = SVC(kernel='sigmoid', C=10, gamma=1.4, decision_function_shape='ovo', probability=True)
    elif svm_sgd:
        clf_title = 'SVM (SGD)'
        if bow:
            vectorizer_title = 'Bag-of-Words'
            counts, targets = FeatureExtractor.bow(max_df=0.25,
                                                   ngram_range=(1, 4)).extract_features_from_csv
            target_ints = []
            for target in targets:
                target_ints.append(category_names_reverse.index(target))
            class_weights = get_classweight(targets)
            clf = SGDClassifier(loss='squared_hinge', penalty='l1', alpha=1e-05, max_iter=100, tol=0.2, class_weight=class_weights)
        else:
            vectorizer_title = 'TF-IDF'
            counts, targets = FeatureExtractor.tfidf(ngram_range=(1, 4), max_df=0.25, use_idf=False,
                                                     sublinear_tf=True).extract_features_from_csv
            target_ints = []
            for target in targets:
                target_ints.append(category_names_reverse.index(target))
            class_weights = get_classweight(targets)
            clf = SGDClassifier(loss='hinge', penalty='l1', alpha=1e-05, max_iter=100, tol=0.2, class_weight=class_weights)
    else:
        print('Please provide a classifer algorithm')
        return

    clf.fit(counts, targets)

    if persist:
        joblib.dump(clf, clf_title+'.pkl')

    # scores
    ac_scores = []
    f1_scores = []
    prec_scores = []
    rec_scores = []

    confusion = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])
    cv = list(StratifiedKFold(n_splits=15, random_state=1).split(counts, targets))

    for k, (train_indices, test_indices) in enumerate(cv):
        train_text = counts[train_indices]
        train_y = targets[train_indices]

        test_text = counts[test_indices]
        test_y = targets[test_indices]

        clf.fit(train_text, train_y)
        predictions = clf.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)

        ac_scores.append(accuracy_score(test_y, predictions))
        f1_scores.append(f1_score(test_y, predictions, average="macro"))
        prec_scores.append(precision_score(test_y, predictions, average="macro"))
        rec_scores.append(recall_score(test_y, predictions, average="macro"))

    print("---------------------- \nResults for ", clf_title, " with ", vectorizer_title, ":")
    print("K-Folds Accuracy-score: ", sum(ac_scores)/len(ac_scores))
    print("K-Folds F1-score: ", sum(f1_scores)/len(f1_scores))
    print("K-Folds Precision-score: ", sum(prec_scores)/len(prec_scores))
    print("K-Folds Recall-score: ", sum(rec_scores)/len(rec_scores))

    print("CV accuracy : %.3f +/- %.3f" % (np.mean(ac_scores), np.std(ac_scores)))

    labels = ['Barent.', 'Finanzen', 'Freiz.&Lifes.', 'Lebensh.',
              'Mob.&Verk.', 'Versich.', 'Wohn.&Haus.']
    if plot:
        Plotter.plot_and_show_confusion_matrix(confusion,
                                              labels,
                                              normalize=True,
                                              title=clf_title,
                                              save=True)

#classify(support_vm=True)
#estimate_jaccard_similarity()
