
import json
import pickle as pkl
import pandas as pd
import re

from kxy_datasets.regressions import all_regression_datasets
from kxy_datasets.classifications import all_classification_datasets


def prettify_name(name):
	if name.startswith('UCI'):
		pref = re.sub(r"(\w)([A-Z])([a-z])", r"\1 \2\3", name[3:]) 
		pref = re.sub(r"(\w)([a-z])([A-Z])", r"\1\2 \3", pref) 
		return pref + ' (UCI)'

	if name.startswith('Kaggle'):
		pref = re.sub(r"(\w)([A-Z])([a-z])", r"\1 \2\3", name[6:]) 
		pref = re.sub(r"(\w)([a-z])([A-Z])", r"\1\2 \3", pref) 
		return pref + ' (Kaggle)'

	return re.sub(r"(\w)([A-Z])([a-z])", r"\1 \2\3", name)


def run_experiments():
	try:
		with open('valuation_results.pkl', 'rb') as f:
			results = pkl.load(f)
	except:
		results = []
		# Name, Problem Type, n, d, RMSE, R^2, Accuracy, # Classes
		for model in all_regression_datasets:
			dataset = model()
			name = dataset.name
			problem_type = dataset.problem_type
			n = str(len(dataset))
			d = str(dataset.num_features)
			dv = dataset.data_valuation()
			r2 = dv['Achievable R-Squared'][0]
			rmse = dv['Achievable RMSE'][0]
			accuracy = '-'
			n_classes = '-'
			line = [name, problem_type, n, d, n_classes, rmse, r2, accuracy]
			print(line)
			results += [line]

		for model in all_classification_datasets:
			dataset = model()
			name = dataset.name
			problem_type = dataset.problem_type
			n = str(len(dataset))
			d = str(dataset.num_features)
			rmse = '-'
			dv = dataset.data_valuation()
			r2 = dv['Achievable R-Squared'][0]
			accuracy = dv['Achievable Accuracy'][0]
			n_classes = dataset.num_classes

			line = [name, problem_type, n, d, n_classes, rmse, r2, accuracy]
			print(line)
			results += [line]

		with open('valuation_results.pkl', 'wb') as f:
			pkl.dump(results, f)

	df = pd.DataFrame(results, columns=['Dataset', 'Problem Type', 'n', 'd', 'Number of Classes', 'RMSE', 'R-Squared', 'Classification Accuracy'])
	df[['n', 'd']] = df[['n', 'd']].astype(int)
	df = df.sort_values(by=['Problem Type', 'd', 'n'], ignore_index=True)
	df['Dataset'] = df['Dataset'].apply(prettify_name)

	return df


def print_results():
	df = run_experiments()
	print(df)


def print_latex_table():
	df = run_experiments()
	print(df.to_latex(index=False))




if __name__ == '__main__':
	def prettify_name_reg(name, problem_type, n, d):
		if name.startswith('UCI'):
			pref = re.sub(r"(\w)([A-Z])([a-z])", r"\1 \2\3", name[3:]) 
			pref = re.sub(r"(\w)([a-z])([A-Z])", r"\1\2 \3", pref) 
			return pref + ' (UCI, %s, n=%d, d=%d)' % (problem_type, n, d)

		if name.startswith('Kaggle'):
			pref = re.sub(r"(\w)([A-Z])([a-z])", r"\1 \2\3", name[6:]) 
			pref = re.sub(r"(\w)([a-z])([A-Z])", r"\1\2 \3", pref) 
			return pref + ' (Kaggle, %s, n=%d, d=%d)' % (problem_type, n, d)

		return re.sub(r"(\w)([A-Z])([a-z])", r"\1 \2\3", name)

	def prettify_name_cls(name, problem_type, n, d, q):
		if name.startswith('UCI'):
			pref = re.sub(r"(\w)([A-Z])([a-z])", r"\1 \2\3", name[3:]) 
			pref = re.sub(r"(\w)([a-z])([A-Z])", r"\1\2 \3", pref) 
			return pref + ' (UCI, %s, n=%d, d=%d, %d classes)' % (problem_type, n, d, q)

		if name.startswith('Kaggle'):
			pref = re.sub(r"(\w)([A-Z])([a-z])", r"\1 \2\3", name[6:]) 
			pref = re.sub(r"(\w)([a-z])([A-Z])", r"\1\2 \3", pref) 
			return pref + ' (Kaggle, %s, n=%d, d=%d, %d classes)' % (problem_type, n, d, q)

		return re.sub(r"(\w)([A-Z])([a-z])", r"\1 \2\3", name)

	exps = []
	for model in all_regression_datasets:
		dataset = model()
		name = dataset.name
		problem_type = dataset.problem_type
		n = len(dataset)
		d = dataset.num_features
		exps += [prettify_name_reg(name, problem_type.title(), n, d)]

	for model in all_classification_datasets:
		dataset = model()
		name = dataset.name
		problem_type = dataset.problem_type
		n = len(dataset)
		d = dataset.num_features
		q = dataset.num_classes
		exps += [prettify_name_cls(name, problem_type.title(), n, d, q)]

	exps = sorted(exps)

	for exp in exps:
		print('* :ref:`%s`' % exp)










