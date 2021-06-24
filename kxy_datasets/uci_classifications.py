#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCI classification datasets
"""
import os
import sys
import inspect

import pandas as pd
from unlzw import unlzw
from urllib import request

from kxy_datasets.base import BaseRealClassification
from kxy_datasets.utils import extract_from_url


class Adult(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/adult
    """
    def __init__(self):
        url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        url_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        df_train = pd.read_csv(url_train, names=['Age', 'Workclass', 'Fnlwgt', 'Education',\
            'Education Num', 'Marital Status',\
            'Occupation', 'Relationship', 'Race', 'Sex', \
            'Capital Gain', 'Capital Loss', 'Hours Per Week', \
            'Native Country', 'Income'])
        df_test = pd.read_csv(url_test, names=['Age', 'Workclass', 'Fnlwgt', 'Education',\
            'Education Num', 'Marital Status',\
            'Occupation', 'Relationship', 'Race', 'Sex', \
            'Capital Gain', 'Capital Loss', 'Hours Per Week', \
            'Native Country', 'Income'])
        df_test['Income'] = df_test['Income'].str.rstrip('.')
        df_test.index += len(df_train)
        df = pd.concat([df_train, df_test])
        self._df = df

        y_columns = ['Income']
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x_train = df_train[x_columns].values
        self.x_test  = df_test[x_columns].values
        self.x = df[x_columns].values

        self.y_train = df_train[y_columns].values
        self.y_test  = df_test[y_columns].values 
        self.y = df[y_columns].values

        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]



class APSFailure(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks
    """
    def __init__(self):
        url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv'
        url_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_test_set.csv'
        df_train = pd.read_csv(url_train, skiprows=20, na_values='na')
        df_test = pd.read_csv(url_test, skiprows=20, na_values='na')
        df = pd.concat([df_train, df_test])
        self._df = df

        y_columns = ['class']
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x_train = df_train[x_columns].values
        self.x_test  = df_test[x_columns].values
        self.x = df[x_columns].values

        self.y_train = df_train[y_columns].values
        self.y_test  = df_test[y_columns].values 
        self.y = df[y_columns].values

        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]



class Avila(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/Avila
    """
    def __init__(self, root_dir='./'):
        dataset_path = os.path.join(root_dir, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip'
        extract_from_url(url, dataset_path)
        file_path_train = os.path.join(dataset_path, 'avila', 'avila-tr.txt')
        file_path_test = os.path.join(dataset_path, 'avila', 'avila-ts.txt')

        columns = ['F%d' % i for i in range(1, 11)] + ['Class']
        df_train = pd.read_csv(file_path_train, names=columns)
        df_test = pd.read_csv(file_path_test, names=columns)
        df = pd.concat([df_train, df_test])
        self._df = df
        y_columns = ['Class']
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x_train = df_train[x_columns].values
        self.x_test  = df_test[x_columns].values
        self.x = df[x_columns].values

        self.y_train = df_train[y_columns].values
        self.y_test  = df_test[y_columns].values 
        self.y = df[y_columns].values

        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]


class BankMarketing(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    """
    def __init__(self, root_dir='./'):
        dataset_path = os.path.join(root_dir, self.name)
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
        extract_from_url(url, dataset_path)
        file_path = os.path.join(dataset_path, 'bank-additional', 'bank-additional-full.csv')
        df = pd.read_csv(file_path, sep=';')
        self._df = df
        y_columns = ['y']
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x = df[x_columns].values
        self.y = df[y_columns].values
        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]



class BankNote(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    """
    def __init__(self):
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/' \
            'data_banknote_authentication.txt'
        df = pd.read_csv(url, names=['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Is Fake'])
        self._df = df
        y_columns = ['Is Fake']
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x = df[x_columns].values
        self.y = df[y_columns].values
        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]



class CardDefault(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    """
    def __init__(self, root_dir='./'):
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00350/' \
              'default%20of%20credit%20card%20clients.xls'
        df = pd.read_excel(url, skiprows=1, index_col='ID')
        self._df = df
        y_columns = ['default payment next month']
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x = df[x_columns].values
        self.y = df[y_columns].values
        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]



class Landsat(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)
    """
    def __init__(self, root_dir='./'):
        url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn'
        url_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst'
        df_train = pd.read_csv(url_train, sep=' ', header=None)
        df_test = pd.read_csv(url_test, sep=' ', header=None)
        df_test.index += len(df_train)
        df = pd.concat([df_train, df_test])
        y_columns = [36]
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x_train = df_train[x_columns].values
        self.x_test  = df_test[x_columns].values
        self.x = df[x_columns].values

        self.y_train = df_train[y_columns].values
        self.y_test  = df_test[y_columns].values 
        self.y = df[y_columns].values

        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]



class LetterRecognition(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/letter+recognition
    """
    def __init__(self):
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        df = pd.read_csv(url, header=None)
        y_columns = [0]
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x = df[x_columns].values
        self.y = df[y_columns].values

        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]



class MagicGamma(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope
    """
    def __init__(self):
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data'
        df = pd.read_csv(url, header=None)
        y_columns = [10]
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x = df[x_columns].values
        self.y = df[y_columns].values

        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]



class SensorLessDrive(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis
    """
    def __init__(self):
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt'
        df = pd.read_csv(url, header=None, sep=' ')
        y_columns = [48]
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x = df[x_columns].values
        self.y = df[y_columns].values

        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]



class Shuttle(BaseRealClassification):
    """
    Reference: https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
    """
    def __init__(self):
        url_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z'
        url_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst'
        os.makedirs(os.path.join('./', self.name), exist_ok=True)
        file_path_z = os.path.join(os.path.join('./', self.name), 'train.z')
        request.urlretrieve(url_train, file_path_z)
        file_path_csv = os.path.join(os.path.join('./', self.name), 'train.csv')
        with open(file_path_z, 'rb') as f_in:
            with open(file_path_csv, 'wb') as f_out:
                f_out.write(unlzw(f_in.read()))
        df_train = pd.read_csv(file_path_csv, header=None, sep=' ')
        df_test  = pd.read_csv(url_test, header=None, sep=' ')
        df = pd.concat([df_train, df_test])

        y_columns = [9]
        x_columns = [_ for _ in df.columns if _ not in y_columns]

        self.x_train = df_train[x_columns].values
        self.x_test  = df_test[x_columns].values
        self.x = df[x_columns].values

        self.y_train = df_train[y_columns].values
        self.y_test  = df_test[y_columns].values 
        self.y = df[y_columns].values

        self.classes = list(set(list(self.y.flatten())))
        self.x_columns = x_columns
        self.y_column = y_columns[0]


all_uci_classification_datasets = [\
    Adult, APSFailure, Avila, BankMarketing, BankNote, CardDefault, Landsat, LetterRecognition, \
    MagicGamma, SensorLessDrive, Shuttle
]



