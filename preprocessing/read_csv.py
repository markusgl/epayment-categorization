import csv
import random
import re
import os

#nastygrammer = '([,\/+]|\s{3,})' #regex
import string

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


build_data()
