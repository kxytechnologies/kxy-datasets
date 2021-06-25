#!/usr/bin/env python
# -*- coding: utf-8 -*-
import abc

import numpy as np
import pandas as pd


class BaseDataset(abc.ABC):
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    @property
    def name(self):
        return getattr(self, '__name__', self.__class__.__name__)

    @property
    def num_features(self):
        return self.x.shape[1]

    @property
    def num_outputs(self):
        return self.y.shape[1] if len(self.y.shape) > 1 else 1

    @property
    def df(self):
        if getattr(self, '_df', None) is None:
            y_columns = ['y'] if self.num_outputs == 1 else ['y_%d' % i for i in range(self.num_outputs)]
            x_columns = ['x_%d' % i for i in range(self.num_features)]
            columns = y_columns + x_columns
            return pd.DataFrame(np.concatenate([self.y, self.x], axis=1), columns=columns)

        else:
            return self._df



class BaseSyntheticClassification(BaseDataset):
    '''
    A classification dataset where the theoretical best performance achievable is known.
    '''
    @property
    def problem_type(self):
        return 'classification'

    @property
    def achievable_performance(self):
        return {'accuracy': self._achievable_accuracy}



class BaseRealClassification(BaseDataset):
    '''
    A classification dataset where the theoretical best performance achievable is not known.
    '''
    @property
    def problem_type(self):
        return 'classification'

    @property
    def num_classes(self):
        return len(self.classes)



class BaseSyntheticRegression(BaseDataset):
    '''
    A regression dataset where the theoretical best performance achievable is known.
    '''
    @property
    def problem_type(self):
        return 'regression'

    @property
    def achievable_performance(self):
        return {'rmse': self._achievable_rmse, 'r-squared': self._achievable_r_squared}



class BaseRealRegression(BaseDataset):
    '''
    A regression dataset where the theoretical best performance achievable is not known.
    '''
    @property
    def problem_type(self):
        return 'regression'


class KaggleBaseClassification(BaseRealClassification):
    @property
    def name(self):
        return getattr(self, '__name__', 'Kaggle' + self.__class__.__name__)


class KaggleBaseRegression(BaseRealRegression):
    @property
    def name(self):
        return getattr(self, '__name__', 'Kaggle' + self.__class__.__name__)


class UCIBaseClassification(BaseRealClassification):
    @property
    def name(self):
        return getattr(self, '__name__', 'UCI' + self.__class__.__name__)


class UCIBaseRegression(BaseRealRegression):
    @property
    def name(self):
        return getattr(self, '__name__', 'UCI' + self.__class__.__name__)




