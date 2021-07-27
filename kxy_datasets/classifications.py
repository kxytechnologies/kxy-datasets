#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classification datasets
"""
from kxy_datasets.kaggle_classifications import *
from kxy_datasets.uci_classifications import *

# All datasets
all_classification_datasets = all_uci_classification_datasets + all_kaggle_classification_datasets

# Categorization by the number of explanatory variables
# Up to 10 explanatory variables
small_d_classification_datasets = [Avila, BankNote, MagicGamma, Shuttle, SkinSegmentation, WaterQuality]
# Between 11 and 20 explanatory variables
medium_d_classification_datasets = [Adult, BankMarketing, DiabeticRetinopathy, EEGEyeState, HeartAttack, \
	HeartDisease, LetterRecognition, Titanic]
# More than 20 explanatory variables
large_d_classification_datasets = [APSFailure, CardDefault, Landsat, SensorLessDrive]

# Categorization by the number of rows
# Less than 100 rows per explanatory variable
small_n_classification_datasets = [HeartAttack, HeartDisease, Titanic]
# Between 101 and 1000 rows per explanatory variable
medium_n_classification_datasets = [APSFailure, BankNote, DiabeticRetinopathy, Landsat, WaterQuality]
# More than 1000 rows per explanatory variable
large_n_classification_datasets = [Adult, Avila, BankMarketing, CardDefault, EEGEyeState, LetterRecognition, \
	MagicGamma, SensorLessDrive, Shuttle, SkinSegmentation]

# Categorization by the number of classes
binary_classification_datasets = [APSFailure, BankMarketing, BankNote, CardDefault, DiabeticRetinopathy, \
	EEGEyeState, HeartAttack, HeartDisease, MagicGamma, SkinSegmentation, Titanic, WaterQuality]
multiclass_classification_datasets = [Adult, Avila, Landsat, LetterRecognition, SensorLessDrive, \
	Shuttle]
