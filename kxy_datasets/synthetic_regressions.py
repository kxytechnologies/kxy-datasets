#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synthetic regression datasets with known theoretical-best performance achievable.
"""
import logging
import numpy as np
from kxy_datasets.base import BaseSyntheticRegression


class GenericSyntheticRegression(BaseSyntheticRegression):
	'''
	Generic synthetic regression dataset with independent features and known theoretical-best r-squared.
	'''
	def __init__(self, achievable_r_squared, d, n, input_distribution=None, \
			noise_distribution=np.random.randn, seed=None):
		super(GenericSyntheticRegression, self).__init__()
		if seed:
			np.random.seed(seed)

		if input_distribution is None:
			input_distribution = np.random.rand

		self.x = input_distribution(n, d)
		self.fx = self.latent_function(self.x)
		noise_std = np.sqrt((np.var(self.fx)/achievable_r_squared)-np.var(self.fx))
		self.y = self.fx + noise_std*noise_distribution(n)
		if len(self.y.shape)==1:
			self.y = self.y[:, None]
		self._achievable_rmse = noise_std
		self._achievable_r_squared = achievable_r_squared


	def plot(self):
		''' '''
		if self.num_features==1:
			import pylab as plt
			fig = plt.figure(figsize=(10, 10))
			ax = fig.add_subplot(111)
			ax.grid()
			ax.set_title(self.name + ' (Achievable R-Squared: %.2f)' % self._achievable_r_squared)

			if self.num_features == 1:
				z = sorted(list(zip(self.x.flatten(), self.fx.flatten())), key=lambda x: x[0])
				x, y = zip(*z)
				plt.plot(x, y, '--b')
				plt.plot(self.x.flatten(), self.y.flatten(), '*r')
				plt.show()

		else:
			logging.warning('Plotting is only available up to 2D.')


# Regression datasets with independent continuous features.
class LINReg(GenericSyntheticRegression):
	def latent_function(self, x):
		"""
		:math:`f(x) \\propto \\sum_{i=1}^d\frac{x_i}{i}`
		"""
		d = x.shape[1]
		fx = np.dot(x, 1./np.arange(1., d+1.))
		fx = 2.*fx/(fx.max()-fx.min())
		fx = fx-fx.min()-1.
		fx = fx/fx.std()
		return fx


class SQRTABSReg(GenericSyntheticRegression):
	def latent_function(self, x):
		"""
		:math:`f(x) \\propto \\sqrt{|\\sum_{i=1}^d\\frac{x_i}{i}|}`
		"""
		d = x.shape[1]
		fx = np.dot(x, 1./np.arange(1., d+1.))
		fx = 2.*fx/(fx.max()-fx.min())
		fx = fx-fx.min()-1.
		fx = -np.sqrt(np.abs(fx))
		fx = fx/fx.std()
		return fx


class CUBABSReg(GenericSyntheticRegression):
	def latent_function(self, x):
		"""
		:math:`f(x) \\propto -(\\sum_{i=1}^d\\frac{|x_i-0.5|}{i})^3`
		"""
		d = x.shape[1]
		fx = np.dot(np.abs(x-0.5), 1./np.arange(1., d+1.))
		fx = 2.*fx/(fx.max()-fx.min())
		fx = fx-fx.min()-1.
		fx = (-fx)**3
		fx = fx/fx.std()
		return fx


class TANHABSReg(GenericSyntheticRegression):
	def latent_function(self, x):
		"""
		:math:`f(x) \\propto \tanh(\\frac{5}{2}\\sum_{i=1}^d\frac{(x_i-0.5)^2}{i})`
		"""
		d = x.shape[1]
		fx = np.dot((x-0.5)**2, 1./np.arange(1., d+1.))
		fx = 2.*fx/(fx.max()-fx.min())
		fx = fx-fx.min()-1.
		fx = np.tanh(2.5*fx)
		fx = fx/fx.std()
		return fx




# Regression datasets with correlated continuous features.
class LINRegCorr(LINReg):
	def __init__(self, achievable_r_squared, d, n, input_distribution=np.random.rand, \
			noise_distribution=np.random.randn, w=None, seed=None):
		super(LINRegCorr, self).__init__(achievable_r_squared, d, n, input_distribution=input_distribution, \
			noise_distribution=noise_distribution, seed=seed)
		if w is None:
			# If the mixing matrix is not provided, use a random rmixing matrix.
			w = np.random.rand(self.num_features, self.num_features)
		self.x = np.dot(self.x, w)


class SQRTABSRegCorr(SQRTABSReg):
	def __init__(self, achievable_r_squared, d, n, input_distribution=np.random.rand, \
			noise_distribution=np.random.randn, w=None, seed=None):
		super(SQRTABSRegCorr, self).__init__(achievable_r_squared, d, n, input_distribution=input_distribution, \
			noise_distribution=noise_distribution, seed=seed)
		if w is None:
			# If the mixing matrix is not provided, use a random rmixing matrix.
			w = np.random.rand(self.num_features, self.num_features)
		self.x = np.dot(self.x, w)


class CUBABSRegCorr(CUBABSReg):
	def __init__(self, achievable_r_squared, d, n, input_distribution=np.random.rand, \
			noise_distribution=np.random.randn, w=None, seed=None):
		super(CUBABSRegCorr, self).__init__(achievable_r_squared, d, n, input_distribution=input_distribution, \
			noise_distribution=noise_distribution, seed=seed)
		if w is None:
			# If the mixing matrix is not provided, use a random rmixing matrix.
			w = np.random.rand(self.num_features, self.num_features)
		self.x = np.dot(self.x, w)


class TANHABSRegCorr(TANHABSReg):
	def __init__(self, achievable_r_squared, d, n, input_distribution=np.random.rand, \
			noise_distribution=np.random.randn, w=None, seed=None):
		super(TANHABSRegCorr, self).__init__(achievable_r_squared, d, n, input_distribution=input_distribution, \
			noise_distribution=noise_distribution, seed=seed)
		if w is None:
			# If the mixing matrix is not provided, use a random rmixing matrix.
			w = np.random.rand(self.num_features, self.num_features)
		self.x = np.dot(self.x, w)


# TODO: Regression datasets with categorical features only.



# TODO: Regression datasets with categorical and continuous features.





all_synthetic_regression_datasets = [\
	LINReg, SQRTABSReg, CUBABSReg, TANHABSReg, \
	LINRegCorr, SQRTABSRegCorr, CUBABSRegCorr, TANHABSRegCorr, \
]


if __name__ == '__main__':
	z = LINReg(.9, 1, 100)
	print(len(z), z.num_features, z.name)
	z.plot()
	z_corr = LINRegCorr(.9, 1, 100)
	print(len(z_corr), z_corr.num_features, z_corr.name)
	z_corr.plot()

	z = SQRTABSReg(.9, 1, 100)
	print(len(z), z.num_features, z.name)
	z.plot()
	z_corr = SQRTABSRegCorr(.9, 1, 100)
	print(len(z_corr), z_corr.num_features, z_corr.name)
	z_corr.plot()

	z = CUBABSReg(.9, 1, 100)
	print(len(z), z.num_features, z.name)
	z.plot()
	z_corr = CUBABSRegCorr(.9, 1, 100)
	print(len(z_corr), z_corr.num_features, z_corr.name)
	z_corr.plot()

	z = TANHABSReg(.9, 1, 100)
	print(len(z), z.num_features, z.name)
	z.plot()
	z_corr = TANHABSRegCorr(.9, 1, 100)
	print(len(z_corr), z_corr.num_features, z_corr.name)
	z_corr.plot()







