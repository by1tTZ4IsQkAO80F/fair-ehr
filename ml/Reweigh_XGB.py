from aif360.sklearn.preprocessing import Reweighing
from .RWM import RWM
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from protected_groups import protected_attribute_names

reweigher = Reweighing(prot_attr = protected_attribute_names)

clf = RWM(estimator = XGBClassifier(n_estimators=500), 
                reweigher = reweigher)

name = 'Reweigh_XGB'
