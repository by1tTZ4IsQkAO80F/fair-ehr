protected_attribute_names = ['SEX','RACE','ETHNICITY']
protected_attribute_idx = [1, 2, 3]

old_mapping = {
            0: 'AI',
            1: 'Asian',
            2: 'Black',
            3: 'NHPI',
            4: 'Other',
            5: 'Unknown',
            6: 'White'
            }

# Note: mapping AI, NHPI, Other, and Unkown as Other
new_mapping = {
            0: 'Other',
            1: 'Asian',
            2: 'Black',
            3: 'Other',
            4: 'Other',
            5: 'Other',
            6: 'White'
}

# Note: mapping AI, NHPI, Other, and Unkown as Other
group_mapping = {
        'RACE': {
            0: 0,
            1: 1,
            2: 2,
            3: 0,
            4: 0,
            5: 0,
            6: 3,
        }
}

group_levels = {
        'SEX': [0,1],
        'RACE': [0,1,2,3],
        'ETHNICITY':[0,1,2]
        }

privileged_group_levels = {
        'SEX':[1],
        'RACE': [3],
        'ETHNICITY':[1]
        }
        
unprivileged_group_levels = {
        'SEX': [0],
        'RACE': [0,1,2],
        'ETHNICITY':[0,2]
        }

# define a binary version of privelege
## white non-hispanic men:
# single_privileged = {'SEX':1, 'RACE':3, 'ETHNICITY':1}
#white non-hispanics
single_privileged = [{'RACE':3, 'ETHNICITY':1}]
single_unprivileged = [
                       {'RACE':0, 'ETHNICITY':0},
                       {'RACE':0, 'ETHNICITY':2},
                       {'RACE':1, 'ETHNICITY':0},
                       {'RACE':1, 'ETHNICITY':2},
                       {'RACE':2, 'ETHNICITY':0},
                       {'RACE':2, 'ETHNICITY':2},
                      ]
