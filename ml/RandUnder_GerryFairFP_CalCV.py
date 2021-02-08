from .RandUnder_GerryFairFP import clf
from .RandUnder_GerryFairFP import name as clf_name
from .CalCV import CalWrapper

clf = CalWrapper(clf)

name = clf_name + '_CalCV'

