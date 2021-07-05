<div align="center">
  <img src="https://www.kxy.ai/theme/images/logos/logo.svg"><br>
</div>

-----------------

# A Python package to access ML datasets (UCI, Kaggle, synthetic, etc.) in a normalized format.
[![License](https://img.shields.io/badge/license-GPLv3%2B-blue)](https://github.com/kxytechnologies/kxy-datasets/blob/master/LICENSE)
[![PyPI Latest Release](https://img.shields.io/pypi/v/kxy-datasets.svg)](https://www.kxy.ai/)
[![Downloads](https://pepy.tech/badge/kxy-datasets)](https://www.kxy.ai/)



## Example real-life datasets

Loading the data
```
>>> from kxy_datasets.uci_regressions import AirQuality
>>> air_quality = AirQuality()
>>> print(air_quality.name)
UCIAirQuality
```

Retrieving target and explanatory variables as numpy arrays
```
>>> y, x = air_quality.x, air_quality.y
>>> print(air_quality.x.shape)
(8991, 14)
>>> print(air_quality.y.shape)
(8991, 1)
>>> print(len(air_quality))
8991
```

Reading the problem type (classification/regression)
```
>>> print(air_quality.problem_type)
regression
```

Retrieving the data as a dataframe
```
>>> air_quality.df
       Date  Time  CO(GT)  PT08.S1(CO)  NMHC(GT)  C6H6(GT)  PT08.S2(NMHC)  NOx(GT)  PT08.S3(NOx)  NO2(GT)  PT08.S4(NO2)  PT08.S5(O3)     T    RH      AH
0     273.0    18     2.6       1360.0     150.0      11.9         1046.0    166.0        1056.0    113.0        1692.0       1268.0  13.6  48.9  0.7578
1     273.0    19     2.0       1292.0     112.0       9.4          955.0    103.0        1174.0     92.0        1559.0        972.0  13.3  47.7  0.7255
2     273.0    20     2.2       1402.0      88.0       9.0          939.0    131.0        1140.0    114.0        1555.0       1074.0  11.9  54.0  0.7502
3     273.0    21     2.2       1376.0      80.0       9.2          948.0    172.0        1092.0    122.0        1584.0       1203.0  11.0  60.0  0.7867
4     273.0    22     1.6       1272.0      51.0       6.5          836.0    131.0        1205.0    116.0        1490.0       1110.0  11.2  59.6  0.7888
...     ...   ...     ...          ...       ...       ...            ...      ...           ...      ...           ...          ...   ...   ...     ...
9352  456.0    10     3.1       1314.0    -200.0      13.5         1101.0    472.0         539.0    190.0        1374.0       1729.0  21.9  29.3  0.7568
9353  456.0    11     2.4       1163.0    -200.0      11.4         1027.0    353.0         604.0    179.0        1264.0       1269.0  24.3  23.7  0.7119
9354  456.0    12     2.4       1142.0    -200.0      12.4         1063.0    293.0         603.0    175.0        1241.0       1092.0  26.9  18.3  0.6406
9355  456.0    13     2.1       1003.0    -200.0       9.5          961.0    235.0         702.0    156.0        1041.0        770.0  28.3  13.5  0.5139
9356  456.0    14     2.2       1071.0    -200.0      11.9         1047.0    265.0         654.0    168.0        1129.0        816.0  28.5  13.1  0.5028

[8991 rows x 15 columns]
>>> air_quality.y_column
'C6H6(GT)'
>>> air_quality.x_columns
['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
```

UCI classification datasets
```
>>> from kxy_datasets.uci_classifications import BankNote
```

Kaggle regression datasets
```
>>> from kxy_datasets.kaggle_regressions import HousePricesAdvanced
```

Kaggle classification datasets
```
>>> from kxy_datasets.kaggle_classifications import Titanic
```

## Example synthetic datasets

Synthetic regression datasets (with known theoretical-best performance achievable)
```
>>> from kxy_datasets.synthetic_regressions import SQRTABSReg
```

Synthetic classification datasets (with known theoretical-best performance achievable)
```
>>> from kxy_datasets.synthetic_classifications import EllipticalBoundaryBin
```

## Data valuation and model-free variable selection with the kxy package
Data valuation
```
>>> from kxy_datasets.kaggle_classifications import Titanic
>>> titanic = Titanic()
>>> titanic.data_valuation()
[====================================================================================================] 100% ETA: 0s   
  Achievable R-Squared Achievable Log-Likelihood Per Sample Achievable Accuracy
0                 0.53                            -2.89e-01                0.92
```
Model-free variable selection
```
>>> titanic.variable_selection()
[====================================================================================================] 100% ETA: 0s   
                    Variable Running Achievable R-Squared Running Achievable Accuracy
Selection Order                                                                      
0                No Variable                         0.00                        0.62
1                        Sex                         0.26                        0.79
2                PassengerId                         0.27                        0.79
3                     Pclass                         0.37                        0.84
4                      Parch                         0.37                        0.84
5                        Age                         0.48                        0.90
6                   Embarked                         0.48                        0.90
7                      SibSp                         0.53                        0.92
8                       Fare                         0.53                        0.92
```



