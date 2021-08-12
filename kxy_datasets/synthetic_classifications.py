#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synthetic classification datasets with known theoretical-best performance achievable.
"""
import logging
import numpy as np
from kxy_datasets.base import BaseSyntheticClassification


class GenericSyntheticClassification(BaseSyntheticClassification):
	'''
	Generic synthetic classification dataset with independent features and known theoretical-best accuracy.
	'''
	def __init__(self, achievable_accuracy, d, n, input_distribution=None, seed=None):
		super(GenericSyntheticClassification, self).__init__()
		if seed:
			np.random.seed(seed)
			
		if input_distribution is None:
			input_distribution = np.random.rand

		self.x = input_distribution(n, d)
		fx = self.latent_function(self.x)
		assert np.can_cast(fx, int), 'Classes should be integer by convention'
		self.classes = sorted([_ for _ in set(fx)])
		self.q = len(self.classes) # Number of classes

		# 1. Create the random label to use in case we need to change the label returned by the latent function.
		np.random.shuffle(self.classes)
		self.class_map = {self.classes[i]: i for i in range(len(self.classes))}
		non_fx = np.array([self.class_map[(self.class_map[l]+1)%self.q] for l in fx])

		# 2. Change the label returned by the latent function with probability (1-achievable_accuracy)
		s = (np.random.rand(n) > achievable_accuracy).astype(int)
		# self.y == fx iff s == 0 (i.e. with probability 'achievable_accuracy')
		self.y = (1-s)*fx.astype(int) + s*non_fx
		if len(self.y.shape)==1:
			self.y = self.y[:, None]

		self._achievable_accuracy = achievable_accuracy



	def plot(self):
		''' '''
		all_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', \
			'#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', \
			'#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff']
		if self.q <= len(all_colors) and self.num_features <= 2:
			import pylab as plt
			fig = plt.figure(figsize=(10, 10))
			ax = fig.add_subplot(111)
			ax.set_aspect('equal')
			ax.grid()
			ax.set_title(self.name + ' (Achievable Accuracy: %.2f)' % self._achievable_accuracy)

			if self.num_features == 2:
				x0 = self.x[:, 0]
				x1 = self.x[:, 1]
				for l in self.classes:
					selector = self.y.flatten() == l
					ax.scatter(x0[selector], x1[selector], color=all_colors[self.class_map[l]], label='y=%d' % l)
				ax.set_xticks(())
				ax.set_yticks(())
				ax.legend(loc='upper right')
				plt.xlabel('x0')
				plt.ylabel('x1')		
				plt.show()

			elif self.num_features == 1:
				x = self.x[:, 0]
				for l in self.classes:
					selector = self.y.flatten() == l
					ax.plot(x[selector], np.zeros_like(x[selector]), '|', color=all_colors[self.class_map[l]], linewidth=2, markersize=10,\
						label='y=%d' % l)
				ax.set_xticks(())
				ax.set_yticks(())
				ax.legend(loc='upper right')
				plt.xlabel('x')
				plt.show()

		else:
			logging.warning('Plotting is only available up to 2D and up to %d classes' % self.q)



# Binary classification datasets with independent features
class LinearBoundaryBin(GenericSyntheticClassification):
	def latent_function(self, x):
		d = x.shape[1]
		fx = np.dot(x, 1./np.arange(1., d+1.))
		fx = 2.*fx/(x.max()-fx.min())
		fx = fx-fx.min()-1.
		fx = fx/fx.std()
		fx = (fx>fx.mean()).astype(int)
		return fx


class BandBoundaryBin(GenericSyntheticClassification):
	def latent_function(self, x):
		d = x.shape[1]
		fx = np.dot(x, 1./np.arange(1., d+1.))
		fx = 2.*fx/(x.max()-fx.min())
		fx = fx-fx.min()-1.
		fx = fx/fx.std()
		fx = -np.sqrt(np.abs(fx))
		fx = (fx>fx.mean()).astype(int)
		return fx


class EllipticalBoundaryBin(GenericSyntheticClassification):
	def latent_function(self, x):
		d = x.shape[1]
		fx = np.dot(np.abs(x-0.5), 1./np.arange(1., d+1.))
		fx = 2.*fx/(fx.max()-fx.min())
		fx = fx-fx.min()-1.
		fx = fx/fx.std()
		fx = (-fx)**3
		fx = (fx>fx.mean()).astype(int)
		return fx


class HexagonalBoundaryBin(GenericSyntheticClassification):
	def latent_function(self, x):
		d = x.shape[1]
		fx = np.dot((x-0.5)**2, 1./np.arange(1., d+1.))
		fx = 2.*fx/(fx.max()-fx.min())
		fx = fx-fx.min()-1.
		fx = fx/fx.std()
		fx = np.tanh(2.5*fx)
		fx = (fx>fx.mean()).astype(int)
		return fx


# Binary classification datasets with correlated features
class LinearBoundaryBinCorr(LinearBoundaryBin):
	def __init__(self, achievable_accuracy, d, n, input_distribution=np.random.rand, w=None, seed=None):
		super(LinearBoundaryBinCorr, self).__init__(achievable_accuracy, d, n, input_distribution=input_distribution, \
			seed=seed)
		if w is None:
			# If the mixing matrix is not provided, use a random mixing matrix.
			w = np.random.rand(self.num_features, self.num_features)
		self.x = np.dot(self.x, w)


class BandBoundaryBinCorr(BandBoundaryBin):
	def __init__(self, achievable_accuracy, d, n, input_distribution=np.random.rand, w=None, seed=None):
		super(BandBoundaryBinCorr, self).__init__(achievable_accuracy, d, n, input_distribution=input_distribution, \
			seed=seed)
		if w is None:
			# If the mixing matrix is not provided, use a random mixing matrix.
			w = np.random.rand(self.num_features, self.num_features)
		self.x = np.dot(self.x, w)


class EllipticalBoundaryBinCorr(EllipticalBoundaryBin):
	def __init__(self, achievable_accuracy, d, n, input_distribution=np.random.rand, w=None, seed=None):
		super(EllipticalBoundaryBinCorr, self).__init__(achievable_accuracy, d, n, input_distribution=input_distribution, \
			seed=seed)
		if w is None:
			# If the mixing matrix is not provided, use a random mixing matrix.
			w = np.random.rand(self.num_features, self.num_features)
		self.x = np.dot(self.x, w)


class HexagonalBoundaryBinCorr(HexagonalBoundaryBin):
	def __init__(self, achievable_accuracy, d, n, input_distribution=np.random.rand, w=None, seed=None):
		super(HexagonalBoundaryBinCorr, self).__init__(achievable_accuracy, d, n, input_distribution=input_distribution, \
			seed=seed)
		if w is None:
			# If the mixing matrix is not provided, use a random rmixing matrix.
			w = np.random.rand(self.num_features, self.num_features)
		self.x = np.dot(self.x, w)



# TODO: Binary classification datasets with categorical features


# TODO: Binary classification datasets with categorical and continuous features


# TODO: Multiclass classification datasets



all_synthetic_classification_datasets = [\
	LinearBoundaryBin, BandBoundaryBin, EllipticalBoundaryBin, HexagonalBoundaryBin, \
	LinearBoundaryBinCorr, BandBoundaryBinCorr, EllipticalBoundaryBinCorr, HexagonalBoundaryBinCorr \
]



if __name__ == '__main__':
	z = LinearBoundaryBin(.99, 1, 10000)
	print(len(z), z.num_features, z.name)
	z.plot()
	z_corr = LinearBoundaryBinCorr(.99, 1, 10000)
	print(len(z_corr), z_corr.num_features, z_corr.name)
	z_corr.plot()

	z = BandBoundaryBin(.9, 2, 10000)
	print(len(z), z.num_features, z.name)
	z.plot()
	z_corr = BandBoundaryBinCorr(.9, 2, 10000)
	print(len(z_corr), z_corr.num_features, z_corr.name)
	z_corr.plot()

	z = EllipticalBoundaryBin(.9, 2, 10000)
	print(len(z), z.num_features, z.name)
	z.plot()
	z_corr = EllipticalBoundaryBinCorr(.9, 2, 10000)
	print(len(z_corr), z_corr.num_features, z_corr.name)
	z_corr.plot()

	z = HexagonalBoundaryBin(.9, 2, 10000)
	print(len(z), z.num_features, z.name)
	z.plot()
	z_corr = HexagonalBoundaryBinCorr(.9, 2, 10000)
	print(len(z_corr), z_corr.num_features, z_corr.name)
	z_corr.plot()
	print(z.df)










