import csv
import pprint
import random
import re
import os
import json

#nastygrammer = '([,\/+]|\s{3,})' #regex
import string

from pymongo import MongoClient

nastygrammar = '([,"\/+])'
input_file = '/Users/mgl/Training_Data/smartanalytics_set.csv'
output_file = '/Users/mgl/Training_Data/generated_testdata.csv'

def read_and_format_csv(filepath):
    with open(filepath) as csvfile:
        filereader = csv.reader(csvfile, delimiter=';')
        record = []
        for row in filereader:
            if filereader.line_num > 1: #ignore header line
                receiver = str(row[0])
                receiver = re.sub(nastygrammar, ' ', receiver.lower())
                rule = str(row[1])
                category = str(row[2])

                record.append([category, receiver, rule])
    return record

def build_data():
    records = read_and_format_csv(input_file)
    extended_records = []
    for record in records:
        category = record[0]
        receiver = record[1]
        rule = record[2]
        gid = ""
        vwz = ""
        if "GID=" in rule:
            #print(rule)
            gid = rule[rule.index("GID=\'"):re.search("[(0-9)|(A-Z)]\'",rule).end()]
            gid = gid.replace('GID=', '').replace('\'', '').lower()
        if "VWZ LIKE" in rule:
            vwz = rule[rule.index("VWZ LIKE"):re.search("[\s(0-9)|(A-Z|a-z)]\%\'",rule).end()]
            vwz = vwz.replace("VWZ LIKE '%", '').replace("%", ' ').lower()
        extended_records.append([category, receiver, vwz, gid])

    if not os.path.isfile(output_file):
        file = open(output_file, 'w+')
        for entry in extended_records:
            file.write(entry[0] + ";" + entry[1] + ";" + entry[2] + ";" + entry[3] + "\n")
        file.close()
    else:
        print("File already exists: " + output_file)

def random_char_generator(size=1, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def save_generated_data_to_mongodb():
    with open('/Users/mgl/Datasets/generated_testdata.csv') as csvfile:
        filereader = csv.reader(csvfile, delimiter=';')

        for row in filereader:
            category = str(row[0]).replace('&', '').lower()
            receiver = str(row[1])
            creditor_id = str(row[3])

            company = {"category": category, "receiver": receiver,
                       "creditorid": creditor_id}
            client = MongoClient('mongodb://localhost:27017/')
            db = client.companyset # dbname
            companies = db.companies # collection

            id = companies.insert_one(company).inserted_id
            print("successfully added " + str(id))


#save_generated_data_to_mongodb()
"""
client = MongoClient('mongodb://localhost:27017/')
db = client.companyset # dbname

regex = re.compile("de74zzz00000045294", re.IGNORECASE)
#pprint.pprint(db.companies.find_one({"creditorid": regex}))
db_entry = db.companies.find_one({"creditorid": regex})
print(db_entry['category'])
"""

