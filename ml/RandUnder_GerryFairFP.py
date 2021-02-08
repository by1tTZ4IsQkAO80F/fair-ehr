from imblearn.pipeline import make_pipeline
from .RandUnder import prep
from .GerryFairFP import clf as estimator

clf = make_pipeline(prep, estimator)

name = 'RandUnder_GerryFairFP'
