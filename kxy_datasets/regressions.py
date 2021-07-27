#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression datasets
"""
from kxy_datasets.kaggle_regressions import *
from kxy_datasets.uci_regressions import *

# All datasets
all_regression_datasets = all_uci_regression_datasets + all_kaggle_regression_datasets

# Categorization by the number of explanatory variables
# Up to 10 explanatory variables
small_d_regression_datasets = [Abalone, AirFoil, Concrete, EnergyEfficiency, PowerPlant, RealEstate, \
	YachtHydrodynamics]
# Between 11 and 20 explanatory variables
medium_d_regression_datasets = [AirQuality, BikeSharing, NavalPropulsion, Parkinson, WhiteWineQuality]
# More than 20 explanatory variables
large_d_regression_datasets = [BlogFeedback, CTSlices, FacebookComments, HousePricesAdvanced, OnlineNews, \
	SocialMediaBuzz, Superconductivity, YearPredictionMSD]

# Categorization by the number of rows
# Less than 100 rows per explanatory variable
small_n_regression_datasets = [EnergyEfficiency, HousePricesAdvanced, RealEstate, YachtHydrodynamics]
# Between 101 and 1000 rows per explanatory variable
medium_n_regression_datasets = [Abalone, AirFoil, AirQuality, BikeSharing, BlogFeedback, Concrete, \
	CTSlices, NavalPropulsion, OnlineNews, Parkinson, Superconductivity, WhiteWineQuality]
# More than 1000 rows per explanatory variable
large_n_regression_datasets = [FacebookComments, PowerPlant, SocialMediaBuzz, YearPredictionMSD]

