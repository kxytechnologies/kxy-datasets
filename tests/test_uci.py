#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kxy_datasets.uci_classifications import *
from kxy_datasets.uci_regressions import *


expected_results = [
	# Classification datasets
	(Adult, (48843, 14), (48843, 1), 'classification', 3),
	(APSFailure, (76000, 170), (76000, 1), 'classification', 2),
	(Avila, (20867, 10), (20867, 1), 'classification', 12),
	(BankMarketing, (41188, 20), (41188, 1), 'classification', 2),
	(BankNote, (1372, 4), (1372, 1), 'classification', 2),
	(CardDefault, (30000, 23), (30000, 1), 'classification', 2),
	(Landsat, (6435, 36), (6435, 1), 'classification', 6),
	(LetterRecognition, (20000, 16), (20000, 1), 'classification', 26),
	(MagicGamma, (19020, 10), (19020, 1), 'classification', 2),
	(SensorLessDrive, (58509, 48), (58509, 1), 'classification', 11),
	(Shuttle, (58000, 9), (58000, 1), 'classification', 7),

	# Regression datasets
	(Abalone, (4177, 8), (4177, 1), 'regression', None),
	(AirFoil, (1503, 5), (1503, 1), 'regression', None),
	(AirQuality, (8991, 14), (8991, 1), 'regression', None),
	(BlogFeedback, (60021, 280), (60021, 1), 'regression', None),
	(CTSlices, (53500, 385), (53500, 1), 'regression', None),
	(FacebookComments, (209074, 53), (209074, 1), 'regression', None),
	(OnlineNews, (39644, 58), (39644, 1), 'regression', None),
	(PowerPlant, (9568, 4), (9568, 1), 'regression', None),
	(Parkinson, (5875, 20), (5875, 2), 'regression', None),
	(RealEstate, (414, 6), (414, 1), 'regression', None),
	(Superconductivity, (21263, 81), (21263, 1), 'regression', None),
	(YachtHydrodynamics, (308, 6), (308, 1), 'regression', None),
	(WhiteWineQuality, (4898, 11), (4898, 1), 'regression', None),
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
	n_datasets = len(all_uci_regression_datasets + all_uci_classification_datasets)
	assert n_datasets > 23, 'There should be more than 20 datasets'










