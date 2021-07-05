#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kxy_datasets.kaggle_classifications import *
from kxy_datasets.kaggle_regressions import *
from kxy_datasets.uci_classifications import *
from kxy_datasets.uci_regressions import *

def test_data_valuation():
	titanic = Titanic()
	try:
		titanic.data_valuation()
		assert True
	except:
		assert False, 'kxy data valuation should succeed'


def test_variable_selection():
	titanic = Titanic()
	try:
		titanic.variable_selection()
		assert True
	except:
		assert False, 'kxy variable selection should succeed'


