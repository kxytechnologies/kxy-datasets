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


if __name__ == '__main__':
	# titanic = Titanic()
	# print(titanic.data_valuation())
	# print(titanic.variable_selection())

	# house = HousePricesAdvanced()
	# print(house.data_valuation())
	# print(house.variable_selection())

	# heart = HeartAttack()
	# print(heart.data_valuation())
	# print(heart.variable_selection())

	# heart = HeartDisease()
	# print(heart.data_valuation())
	# print(heart.variable_selection())

	# water = WaterQuality()
	# print(water.data_valuation())
	# print(water.variable_selection())

	card = Avila()
	print(card.data_valuation())
	print(card.variable_selection())
