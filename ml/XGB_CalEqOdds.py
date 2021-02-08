from .XGB import name as clf_name
from .XGB import clf as estimator
from .CalEqOdds import CalWrapper

clf = CalWrapper(estimator=estimator)

name = clf_name + '_CalEqOdds'

