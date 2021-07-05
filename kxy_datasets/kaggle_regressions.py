#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kaggle regression datasets
"""
import os
import sys
import inspect

import pandas as pd
from unlzw import unlzw
from urllib import request

from kxy_datasets.base import KaggleBaseRegression
from kxy_datasets.utils import extract_from_url


class HousePricesAdvanced(KaggleBaseRegression):
	"""
	Reference: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/
	"""
	def __init__(self):
		url = 's3://datasets.kxy.ai/regression/kaggle_house_prices_advanced_regression.csv'
		df = pd.read_csv(url)
		df.set_index('Id', inplace=True)
		self._df = df
		y_columns = ['SalePrice']
		x_columns = [_ for _ in df.columns if _ not in y_columns]
		self.x = df[x_columns].values
		self.y = df[y_columns].values
		self.x_columns = x_columns
		self.y_column = y_columns[0]


all_kaggle_regression_datasets = [cls for _, cls in inspect.getmembers(sys.modules[__name__]) \
	if inspect.isclass(cls) and issubclass(cls, KaggleBaseRegression) and cls != KaggleBaseRegression]