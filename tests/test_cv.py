#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kxy_datasets.regressions import HousePricesAdvanced


def test_cv():
	dataset = HousePricesAdvanced()
	d = dataset.x.shape[1]
	n = dataset.x.shape[0]
	for expected_count in [5, 10, 20]:
		count = 0
		for x_train, y_train, x_test, y_test in dataset.cv_split(0.8, expected_count):
			assert x_train.shape[1] == d, 'The number of features should be the same'
			assert x_test.shape[1] == d, 'The number of features should be the same'
			assert x_test.shape[0]+x_train.shape[0] == n, 'No row should be left out'
			assert y_test.shape[0]+y_train.shape[0] == n, 'No row should be left out'
			count += 1
		assert count == expected_count, 'The number of splits should be as expected'