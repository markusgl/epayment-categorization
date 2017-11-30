import os
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

NEWLINE = '\n'

BARENTNAHME = 'barentnahme'
FINANZEN = 'finanzen'
FREIZEITLIFESTYLE = 'freizeitlifestyle'
LEBENSHALTUNG = 'lebenshaltung'
MOBILITAETVERKEHR = 'mobilitaetverkehrsmittel'
VERSICHERUNGEN = 'versicherungen'
WOHNENHAUSHALT = 'wohnenhaushalt'


SOURCES = [
    ('F:\\Datasets\\Transaction-Dataset\\barentnahme',      BARENTNAHME),
    ('F:\\Datasets\\Transaction-Dataset\\finanzen',    FINANZEN),
    ('F:\\Datasets\\Transaction-Dataset\\freizeitlifestyle',  FREIZEITLIFESTYLE),
    ('F:\\Datasets\\Transaction-Dataset\\lebenshaltung',   LEBENSHALTUNG),
    ('F:\\Datasets\\Transaction-Dataset\\mobilitaetverkehrsmittel',     MOBILITAETVERKEHR),
    ('F:\\Datasets\\Transaction-Dataset\\versicherungen', VERSICHERUNGEN),
    ('F:\\Datasets\\Transaction-Dataset\\wohnenhaushalt',          WOHNENHAUSHALT)
]

SKIP_FILES = {'cmds'}

''' iterate through all files an yield the text body '''
def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    #past_header, lines = False, []
                    lines = []
                    f = open(file_path, encoding="iso-8859-1")
                    for line in f:
                        lines.append(line)
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


''' build a dataset from transaction bodies containing purpose, receiver and booking type '''
def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

def append_data_frames():
    ''' concatenate DataFrames using pandas append method '''
    data = DataFrame({'text': [], 'class': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))

    return data.reindex(np.random.permutation(data.index))


''' ##### WITHOUT PIPELINING  ##### '''
def extract_featrues():
    data = append_data_frames()

    count_vectorizer = CountVectorizer()
    # learn vocabulary and extract word count features
    targets = data['class'].values
    counts = count_vectorizer.fit_transform(data['text'].values)

    return counts, targets

