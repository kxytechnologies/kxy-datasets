#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kxy_datasets.kaggle_classifications import *
from kxy_datasets.kaggle_regressions import *
from kxy_datasets.uci_classifications import *
from kxy_datasets.uci_regressions import *
from kxy_datasets.synthetic_regressions import *

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


if __name__ == '__main__':
	# titanic = Titanic()
	# titanic.data_valuation()
	# titanic.variable_selection()
	import kxy
	df = LINRegCorr(.9, 50, 100000).df
	df.kxy.variable_selection('y', problem_type='regression')



