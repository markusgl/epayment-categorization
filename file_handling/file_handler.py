""" class for reading from and writing to CSV file used for training of the classifier """

import pandas
import booking_classifier


class FileHandler():
    def __init__(self):
        self.filepath = str(booking_classifier.ROOT_DIR + '/resources/Labeled_transactions.csv')
        self.delimiter = ";"

    def read_csv(self, file):
        if file:
            return pandas.read_csv(filepath_or_buffer=file, encoding='ISO-8859-1', delimiter=self.delimiter)
        else:
            return pandas.read_csv(self.filepath, encoding='ISO-8859-1', delimiter=self.delimiter)

    def write_csv(self, booking):
        with open(self.filepath, 'a') as file:
            file.write('\n'+booking.category+self.delimiter+booking.text+self.delimiter+booking.usage+self.delimiter+booking.owner)
