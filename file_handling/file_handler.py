""" class for reading from and writing to CSV file used for training of the classifier """

import pandas
import csv

class FileHandler():
    def __init__(self):
        self.filepath = '../resources/Labeled_transactions.csv'

    def read_csv(self, file):
        if file:
            return pandas.read_csv(filepath_or_buffer=file, encoding='ISO-8859-1', delimiter=',')
        else:
            return pandas.read_csv(self.filepath, encoding='ISO-8859-1', delimiter=',')

    def write_csv(self, booking):
        #booking_props = booking.to_array()
        booking_props = booking.to_small_array()
        with open(self.filepath, 'a') as file:
            writer = csv.writer(file)
            # TODO check if linebreak already exists before adding one
            writer.writerow(booking_props)
