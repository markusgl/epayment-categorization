""" class for reading from and writing to CSV file used for training of the classifier """

import pandas
import csv

class FileHandler():
    def __init__(self):
        self.filepath = '/Users/mgl/Documents/OneDrive/Datasets/Labeled_transactions.csv'

    def read_csv(self, file):
        return pandas.read_csv(filepath_or_buffer=file, encoding="UTF-8", delimiter=',')

    def write_csv(self, booking):
        booking_props = booking.to_array()
        with open(self.filepath, 'a') as file:
            writer = csv.writer(file)
            # TODO check if linebreak already exists before adding one
            writer.writerow(booking_props)
