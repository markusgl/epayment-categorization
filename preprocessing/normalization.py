import csv
import re
from categories import Categories
import os.path

class Normalizer:

    nastygrammer = '([,\/+]|\s{3,})' #regex
    input_file = 'F:\\Datasets\\Transaction-Dataset\\transactions_and_categories_full.csv'
    output_path = 'F:\\Datasets\\Transaction-Dataset\\'

    def normalize_text_fields(self, fields):
        normalized_fields = [str(fields[0]).lower(),
                             re.sub(self.nastygrammer, ' ', fields[1].lower()),
                             re.sub(self.nastygrammer, ' ', fields[2].lower())]

        return normalized_fields

    def read_and_normalize_all_columns(self, path):
        with open(self.path) as csvfile:
            # replace nasty grammar with whitespace
            #cleaner = re.sub(nastygrammer, '', purpose.lower())
            #booking_list = []
            bookings = []
            filereader = csv.reader(csvfile, delimiter=';')
            for row in filereader:
                if filereader.line_num > 1: #Kommmas werden ersetzt
                    category = str(row[0].lower()).replace('&','')
                    bookingtype = str(row[3]).lower()
                    purpose = str(row[4])
                    purpose = re.sub(self.nastygrammer, ' ', purpose.lower())
                    creditor_id = str(row[5])  # Glaeubiger Id
                    receiver = str(row[8])
                    receiver = re.sub(self.nastygrammer, ' ', receiver.lower())
                    account_id = str(row[9])
                    bank_id = str(row[10])
                    '''
                    booking_list.append(category + ';' + bookingtype + ';'
                                        + purpose + ';' + creditor_id + ';'
                                        + receiver + ';' + account_id + ';' + bank_id)
                    '''
                    # multidimensional booking array
                    bookings.append([category, bookingtype,
                                     purpose, creditor_id,
                                     receiver, account_id, bank_id])
        return bookings

    def build_trainingset(self):
        bookings = self.read_and_normalize_all_columns(input_file)
        count_barentnahme = 0
        count_finanzen = 0
        count_freizeitlifestyle = 0
        count_lebenshaltung = 0
        count_mobilitaetverkehr = 0
        count_sonstiges = 0
        count_versicherungen = 0
        count_wohnenhaushalt = 0
        for row in bookings:
            data = row[1] + " " + row[2] + " " + row[4]
            filepath = self.output_path + 'sonstiges' + "\\" + str(count_sonstiges)
            if row[0] == 'barentnahme':
                filepath = self.output_path + row[0] + "\\" + str(count_barentnahme)
                count_barentnahme += 1
            elif row[0] == 'finanzen':
                filepath = self.output_path + row[0] + "\\" + str(count_finanzen)
                count_finanzen += 1
            elif row[0] == 'freizeitlifestyle':
                filepath = self.output_path + row[0] + "\\" + str(count_freizeitlifestyle)
                count_freizeitlifestyle += 1
            elif row[0] == 'lebenshaltung':
                filepath = self.output_path + row[0] + "\\" + str(count_lebenshaltung)
                count_lebenshaltung += 1
            elif row[0] == 'mobilitaetverkehrsmittel':
                filepath = self.output_path + row[0] + "\\" + str(count_mobilitaetverkehr)
                count_mobilitaetverkehr += 1
            elif row[0] == 'versicherungen':
                filepath = self.output_path + row[0] + "\\" + str(count_versicherungen)
                count_versicherungen += 1
            elif row[0] == 'wohnenhaushalt':
                filepath = self.output_path + row[0] + "\\" + str(count_wohnenhaushalt)
                count_wohnenhaushalt += 1
            else:
                filepath = self.output_path + 'sonstiges' + "\\" + str(count_sonstiges)
                count_sonstiges += 1

            if not os.path.isfile(filepath, encoding="iso-8859-1"):
                file = open(filepath, 'w+')
                file.write(data)
                file.close()
            else:
                print("File already exists: " + filepath)


if __name__ == '__main__':
    normalizer = Normalizer()
    # build_trainingset()
    fields = ['FOLGELASTSCHRIFT', '140787061022 NLV ALT-VORSORGE / 01.09.2017 10,00', 'NUERNBERGER LEBENSVERSICHERUNG AG']
    print(normalizer.normalize_text_fields(fields))