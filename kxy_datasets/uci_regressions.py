#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCI regression datasets
"""
import os
import sys
import inspect
from copy import deepcopy

import numpy as np
import pandas as pd
from urllib import request

from kxy_datasets.base import UCIBaseRegression
from kxy_datasets.utils import extract_from_url



class Abalone(UCIBaseRegression):
	"""
	Reference: https://archive.ics.uci.edu/ml/datasets/Abalone
	"""
	def __init__(self):
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
		df = pd.read_csv(url, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', \
			'Shucked weight', 'Viscera weight', 'Shell weight', 'Age'])
		df['Age'] += 1.5
		self._df = df
		y_columns = ['Age']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]



class AirFoil(UCIBaseRegression):
	"""
	Reference: https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
	"""
	def __init__(self):
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
		df = pd.read_csv(url, sep='\t', names=['Frequency', 'Angle of Attack', 'Chord Length', 'Velocity', 'Displacement Thickness', 'Sound Pressure'])
		self._df = df
		y_columns = ['Sound Pressure']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]



class AirQuality(UCIBaseRegression):
	"""
	Reference: https://archive.ics.uci.edu/ml/datasets/Air+Quality
	"""
	def __init__(self, root_dir='./'):
		dataset_path = os.path.join(root_dir, self.name)
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'
		extract_from_url(url, dataset_path)
		file_path = os.path.join(dataset_path, 'AirQualityUCI.csv')
		df = pd.read_csv(file_path, sep=';', parse_dates=[0, 1])
		df.dropna(axis=0, how='all', inplace=True)
		df.dropna(axis=1, how='all', inplace=True)

		df.Date = (df.Date - df.Date.min()).astype('timedelta64[D]')  # Days as int
		df.Time = df.Time.apply(lambda x: int(x.split('.')[0]))  # Hours as int
		df['C6H6(GT)'] = df['C6H6(GT)'].apply(lambda x: float(x.replace(',', '.')))  # Target as float

		# Some floats are given with ',' instead of '.'
		df = df.applymap(lambda x: float(x.replace(',', '.')) if type(x) is str else x)  # Target as float

		df = df[df['C6H6(GT)'] != -200]  # Drop all rows with missing target values
		df.loc[df['CO(GT)'] == -200, 'CO(GT)'] = np.nan  # -200 means missing value, change to np.nan

		y_columns = ['C6H6(GT)']
		x_columns = [_ for _ in df.columns if _ not in y_columns]

		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self._df = df
		self.x_columns = x_columns
		self.y_column = y_columns[0]



class BlogFeedback(UCIBaseRegression):
	"""
	Reference: https://archive.ics.uci.edu/ml/datasets/BlogFeedback
	"""
	def __init__(self, root_dir='./'):
		file_name = 'blogData_train.csv'
		dataset_path = os.path.join(root_dir, self.name)
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00304/BlogFeedback.zip'
		extract_from_url(url, dataset_path)

		test_dfs = []
		for fn in os.listdir(dataset_path):
			if 'blogData_test' not in fn:
				continue
			file_path = os.path.join(dataset_path, fn)
			test_dfs.append(pd.read_csv(file_path, header=None))
		df_test = pd.concat(test_dfs)

		file_path = os.path.join(dataset_path, file_name)
		df_train = pd.read_csv(file_path, header=None)
		y_columns = [280]
		x_columns = [_ for _ in df_train.columns if _ not in y_columns]
		df_train[y_columns[0]] = np.log(df_train[y_columns[0]] + 0.01)
		df_test[y_columns[0]]  = np.log(df_test[y_columns[0]] + 0.01)
		df = pd.concat([df_train, df_test])

		self.x_train = df_train[x_columns].values
		self.x_test  = df_test[x_columns].values
		self.x = df[x_columns].values

		self.y_train = df_train[y_columns].values
		self.y_test  = df_test[y_columns].values 
		self.y = df[y_columns].values

		self.x_columns = x_columns
		self.y_column = y_columns[0]



class CTSlices(UCIBaseRegression):
	"""
	Reference: https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis
	"""
	def __init__(self, root_dir='./'):
		dataset_path = os.path.join(root_dir, self.name)
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip'
		extract_from_url(url, dataset_path)
		file_name = 'slice_localization_data.csv'
		file_path = os.path.join(dataset_path, file_name)
		df = pd.read_csv(file_path)
		self._df = df
		y_columns = ['reference']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]



class FacebookComments(UCIBaseRegression):
	"""
	Reference: https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset
	"""
	def __init__(self, root_dir='./'):
		dataset_path = os.path.join(root_dir, self.name)
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip'
		extract_from_url(url, dataset_path)
		dataset_path = os.path.join(dataset_path, 'Dataset')
		train_path = os.path.join(dataset_path, 'Training', 'Features_Variant_5.csv')
		test_path  = os.path.join(dataset_path, 'Testing', 'Features_TestSet.csv')
		df_train   = pd.read_csv(train_path, header=None)
		df_test    = pd.read_csv(test_path, header=None)
		df = pd.concat([df_train, df_test])

		y_columns = df_train.columns[-1:]
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x_train = df_train[x_columns].values
		self.x_test  = df_test[x_columns].values
		self.x = df[x_columns].values

		self.y_train = df_train[y_columns].values
		self.y_test  = df_test[y_columns].values 
		self.y = df[y_columns].values

		self.x_columns = x_columns
		self.y_column = y_columns[0]



class OnlineNews(UCIBaseRegression):
	"""
	Reference: https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
	"""
	def __init__(self, root_dir='./'):
		dataset_path = os.path.join(root_dir, self.name)
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip'
		extract_from_url(url, dataset_path)
		file_path = os.path.join(dataset_path, 'OnlineNewsPopularity', 'OnlineNewsPopularity.csv')
		df = pd.read_csv(file_path)
		df.drop(columns=['url', ' timedelta'], inplace=True)
		self._df = df
		y_columns = [' shares']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]



class Parkinson(UCIBaseRegression):
	"""
	Reference: http://archive.ics.uci.edu/ml/datasets/parkinsons
	"""
	def __init__(self):
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' \
			  'parkinsons/telemonitoring/parkinsons_updrs.data'
		df = pd.read_csv(url)
		self._df = df
		y_columns = ['motor_UPDRS', 'total_UPDRS']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_columns = y_columns



class PowerPlant(UCIBaseRegression):
	"""
	Reference: https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant
	"""
	def __init__(self, root_dir='./'):
		dataset_path = os.path.join(root_dir, self.name)
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip'
		extract_from_url(url, dataset_path)
		file_path = os.path.join(dataset_path, 'CCPP', 'Folds5x2_pp.xlsx')
		df = pd.read_excel(file_path)
		self._df = df
		y_columns = ['PE']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]



class RealEstate(UCIBaseRegression):
	"""
	Reference: http://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
	"""
	def __init__(self):
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx'
		df = pd.read_excel(url, index_col='No')
		self._df = df
		y_columns = ['Y house price of unit area']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]



class Superconductivity(UCIBaseRegression):
	"""
	Reference: https://archive.ics.uci.edu/ml/datasets/superconductivty+data
	"""
	def __init__(self, root_dir='./'):
		dataset_path = os.path.join(root_dir, self.name)
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip'
		extract_from_url(url, dataset_path)
		file_path = os.path.join(dataset_path, 'train.csv')
		df = pd.read_csv(file_path)
		self._df = df
		y_columns = ['critical_temp']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]



class YachtHydrodynamics(UCIBaseRegression):
	"""
	Reference: http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics
	"""
	def __init__(self):
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
		df = pd.read_csv(url, sep='[ ]{1,2}', engine='python', \
						 names=['Longitudinal Position', 'Prismatic Coeefficient',\
								'Length-Displacement', 'Beam-Draught Ratio',\
								'Length-Beam Ratio', 'Froude Number',\
								'Residuary Resistance'])
		df.rename(columns={col: col.title() for col in df.columns}, inplace=True)
		self._df = df
		y_columns = ['Residuary Resistance']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]



class WhiteWineQuality(UCIBaseRegression):
	"""
	Reference: http://archive.ics.uci.edu/ml/datasets/Wine+Quality
	"""
	def __init__(self):
		url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
		df = pd.read_csv(url, sep=';')
		self._df = df
		y_columns = ['quality']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]



all_uci_regression_datasets = [\
	Abalone, AirFoil, AirQuality, BlogFeedback, CTSlices, FacebookComments, OnlineNews, PowerPlant, \
	Parkinson, RealEstate, Superconductivity, YachtHydrodynamics, WhiteWineQuality
]





