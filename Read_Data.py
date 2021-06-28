'''
    ** This program checks if we can read the input and if each Text and Class 
    labels can be read via csv reader without error.
'''

import os
import csv
from os import listdir
from os.path import isfile, join


def read_csv(file_name, separator):
    '''This function reads the csv file given by the file_name parameter'''
    try:
        file = open(file_name, 'r', newline='', encoding='utf-8')
    except IOError:
        print('Cannot open the file <{}>'.format(file_name))
        raise SystemExit

    # csv_read will be a dataframe
    csv_read = csv.reader(file, delimiter=separator)
    
    return csv_read


def process_data(data):
    '''This function check if we are getting the same number of comments and the right text/labels
     and also separates the reports and labels into two different lists'''

    # reports and class labels will be stored here
    reports = []
    labels = []

    data = list(data)

    #print(len(data))
    #print(data[25])

    count = 0
    for line in data[1:]:

        # each line in the categorical 'data' is of 2 elements -> UM, REPORT
        try:
            label, report = line
            count += 1
        except:
            print('Data Parsing Error at line = {}'.format(count))
            raise SystemExit
        #checking:
        # making multiple line Report to a single line
        # report = report.replace('\n', ' ').replace('\r', ' ')

        # # if the following 2 validation is right than the parsing is right
        # if label not in ['machinery and equipment', 'working at height', ....]:
        #     # if any of the class label is not right
        #     print('Parsing Error of Class Label at {}'.format(line_no))
        #     raise SystemExit

        # append to the corresponding lists

        reports.append(report)
        labels.append(label)


    return reports, labels






