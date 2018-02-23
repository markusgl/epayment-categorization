import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import word_tokenize
from nltk import WordNetLemmatizer, StanfordSegmenter, PorterStemmer, RegexpStemmer, LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, WhitespaceTokenizer, \
     RegexpTokenizer, SpaceTokenizer

from nltk import sent_tokenize
from nltk import pos_tag
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
import random
import re
import spacy
from preprocessing.classifier_based_german_tagger import ClassifierBasedGermanTagger
nastygrammer = '([/+]|\s{3,})' #regex


class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        """
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('german'))
        self.punct = punct or set(string.punctuation)
        """
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

    """
    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)
    """

#examples = ['AUGENOPTIK SCHMIDT 210710001234567890987654321 ELV12345678 01.01 10.00 ME1 SEPA-ELV-Lastschrift',
#            'DANKE, IHR LIDL//Nuernberg/DE 2017-01-01T10:00:00 Karte1 2017-12 Kartenzahlung',
#            'MUSTER LEBENSVERSICHERUNG AG 123456789098 MLV LEBENSVERS. / 01.01.2017 50, 00 Lastschrift',
#            'Gesund BKK 1234567 AKV KRANKENVERS. R/ 01.01.2017 6,00 Lastschrift']

#examples = ['DANKE, IHR LIDL//Nuernberg/DE 2017-01-01T10:00:00 Karte1 2017-12 Kartenzahlung']

#tokenized_examples = (['DANKE', 'IHR', 'LIDL', 'Nuernberg', 'DE'])
#tagger.tag(tokenized_examples)

#lemma_examples = ['die welt']

"""
for example in lemma_examples:
    nlp = spacy.load('de')
    doc = nlp(example)

    for token in doc:
        print(token.lemma_)


for example in examples:
    clean_example = re.sub(nastygrammer, ' ', example.lower())
    #print('cleaned example: ' + clean_example)

    print('TreebankWordTokenizer: ' + str(TreebankWordTokenizer().tokenize(example)))
    print('WordPunctTokenizer: ' + str(WordPunctTokenizer().tokenize(example)))
    #print('PunktWordTokenizer: ' + str(.tokenize(example)))
    print('SpaceTokenizer: ' + str(SpaceTokenizer().tokenize(example)))
    #print('WhitespaceTokenizer: ' + str(WhitespaceTokenizer().tokenize(example)))
    #print('RegexpTokenizer' + str(RegexpTokenizer(r'\w+|[,\-.]').tokenize(example)))


    # Stemmer 
    # print('SnowBallStemmer: ' + str(SnowballStemmer('german').stem(clean_example)))
    #print('StanfordSegmenter: ' + str(StanfordSegmenter()))
    #print('PorterStemmer: ' + str(PorterStemmer().stem(clean_example)))
    #print('LancasterStemmer: ' + str(LancasterStemmer().stem(clean_example)))

    # Lemmatizer
    #print('WordNetLemmatizer: ' + str(WordNetLemmatizer().lemmatize(clean_example)))
    #nlp = spacy.load('de')
    #doc = nlp(clean_example)

    #for token in doc:
    #    print(token.lemma_)
"""