from .XGB import clf as estimator
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from .RandUnder import prep
from protected_groups import protected_attribute_names


clf = Pipeline([
                ('RmDem', ColumnTransformer(
                    [('rm_dem','drop', protected_attribute_names)],
                    remainder='passthrough'
                    )
                ),
                ('estimator', estimator)
              ])


name = 'RmDem_XGB'
