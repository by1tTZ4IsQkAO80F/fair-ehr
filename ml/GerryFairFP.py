from .SklearnGerryFair import SklearnGerryFairClassifier
from xgboost import XGBRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from protected_groups import protected_attribute_names, group_levels

clf = SklearnGerryFairClassifier(
                protected_attribute_names = protected_attribute_names,
                protected_attribute_levels = [group_levels['SEX'],
                                              group_levels['RACE'],
                                              group_levels['ETHNICITY']
                                              ],
                printflag=True,
                predictor = XGBRegressor(n_estimators = 100,
                                         objective='reg:squarederror'),
                fairness_def = 'FP',
                gamma=0.001
                )

name = 'GerryFairFP'
