#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kaggle classification datasets
"""
import os
import sys
import inspect

import pandas as pd
from unlzw import unlzw
from urllib import request

from kxy_datasets.base import BaseRealClassification
from kxy_datasets.utils import extract_from_url



class HeartAttack(BaseRealClassification):
	"""
	Reference: https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
	"""
	def __init__(self):
		url = 's3://datasets.kxy.ai/classification/kaggle_heart_attack.csv'
		df = pd.read_csv(url)
		self._df = df
		y_columns = ['output']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]
		self.classes = list(set(list(self.y.flatten())))


class HeartDisease(BaseRealClassification):
	"""
	Reference: https://www.kaggle.com/ronitf/heart-disease-uci
	"""
	def __init__(self):
		url = 's3://datasets.kxy.ai/classification/kaggle_heart_disease.csv'
		df = pd.read_csv(url)
		self._df = df
		y_columns = ['target']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]
		self.classes = list(set(list(self.y.flatten())))



class Titanic(BaseRealClassification):
	"""
	Reference: https://www.kaggle.com/c/titanic/
	"""
	def __init__(self):
		url = 's3://datasets.kxy.ai/classification/kaggle_titanic_train.csv'
		df = pd.read_csv(url)
		self._df = df
		y_columns = ['Survived']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]
		self.classes = list(set(list(self.y.flatten())))


class WaterQuality(BaseRealClassification):
	"""
	Reference: https://www.kaggle.com/adityakadiwal/water-potability
	"""
	def __init__(self):
		url = 's3://datasets.kxy.ai/classification/kaggle_water_potability.csv'
		df = pd.read_csv(url)
		self._df = df
		y_columns = ['Potability']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]
		self.classes = list(set(list(self.y.flatten())))





all_kaggle_classification_datasets = [
	HeartAttack, HeartDisease, Titanic, WaterQuality
]

