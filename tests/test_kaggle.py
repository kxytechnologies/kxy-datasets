#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kxy_datasets.kaggle_classifications import *
from kxy_datasets.kaggle_regressions import *


expected_results = [
	# Classification datasets
	(Titanic, (891, 11), (891, 1), 'classification', 2),
	(HeartAttack, (303, 13), (303, 1), 'classification', 2),
	(HeartDisease, (303, 13), (303, 1), 'classification', 2),
	(WaterQuality, (3276, 9), (3276, 1), 'classification', 2),

	# Regression datasets
	(HousePricesAdvanced, (1460, 79), (1460, 1), 'regression', None),
]


def test_shapes():
	for dataset, x_shape, y_shape, problem_type, num_classes in expected_results:
		z = dataset()
		assert z.x.shape == x_shape, 'The shape of x should be as expected for %s' % z.name
		assert z.y.shape == y_shape, 'The shape of y should be as expected for %s' % z.name
		assert z.num_features == x_shape[1], 'The number of features should be consistent with the shape of x for %s' % z.name
		assert z.num_outputs == y_shape[1], 'The number of outputs should be consistent with the shape of y for %s' % z.name
		assert z.x.shape[0] == z.y.shape[0], 'x and y should always have the same dimension for %s' % z.name
		assert len(z) == z.x.shape[0], 'len() should be available for convenience for %s' % z.name


def test_df():
	for dataset, x_shape, y_shape, problem_type, num_classes in expected_results:
		df_shape = (x_shape[0], x_shape[1]+y_shape[1])
		z = dataset()
		assert z.df.shape == df_shape, 'The shape of .df should be as expected for %s' % z.name


def test_problem_type():
	for dataset, x_shape, y_shape, problem_type, num_classes in expected_results:
		z = dataset()
		assert z.problem_type == problem_type, 'The problem type should be available and as expected for %s' % z.name


def test_n_classes():
	for dataset, x_shape, y_shape, problem_type, num_classes in expected_results:
		if problem_type == 'classification':
			z = dataset()
			assert z.num_classes == num_classes, 'The number of classes should be as expected for %s' % z.name


def test_num_datasets():
	n_datasets = len(all_kaggle_regression_datasets + all_kaggle_classification_datasets)
	assert n_datasets > 4, 'There should be more than 4 datasets'