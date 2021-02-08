import pandas as pd
import numpy as np
import pdb
from protected_groups import group_mapping

def read_file(filename, longitudinal=False, rare=True):
    """read in EHR data."""
    xd_name = filename + '_demographics.csv'
    if not longitudinal:
        xc_name = filename + '_common_median_imputed.csv'
    xr_name = filename + '_rare.csv'
    label_name = filename + '_class.csv'

    xd = pd.read_csv(xd_name,index_col='PT_ID')
    # remap protected groups
    for grp, grp_map in group_mapping.items():
        print('re-mapping',grp)
        xd[grp] = xd[grp].apply(lambda x: grp_map[x])
    print('loaded',xd.shape[1],'demographic measures')

    if not longitudinal: 
        xc = pd.read_csv(xc_name,index_col='PT_ID')
        print('loaded',xc.shape[1],'common measures')
    if rare:
        xr = pd.read_csv(xr_name,index_col='PT_ID')
        print('loaded',xr.shape[1],'rare measures')
    
    label = pd.read_csv(label_name,index_col='PT_ID')
    
    print('longitudinal =',longitudinal,'rare =',rare)


    if not longitudinal and rare:   # demographics, common, and rare labs
        df_X = pd.concat([xd, xc, xr],axis=1)
        print('loading demographics, common, and rare labs')
        print('result:',df_X.shape[1],'features')
    elif not longitudinal:  # keep common labs in there, remove rare
        df_X = pd.concat([xd, xc],axis=1)
        print('loading demographics and common labs (rare = ',rare,')')
    elif not rare:  
        # if longitudinal AND don't include rare, use only demographics
        df_X = xd
        print('loading demographics only (longitudinal = ',longitudinal,')')
    else:   
        # for longitudinal case with rare, remove common labs, include 
        # everything else
        df_X = pd.concat([xd, xr],axis=1)
        print('loading demographics and rare labs (longitudinal = ',
                longitudinal,')')
    
    assert(all(df_X.index==label.index))
    ###
    # # Drop total cholesterol (sorry for the hack)
    # if '2093-3' in df_X.columns:
    #     print('dropping total cholesterol')
    #     df_X = df_X.drop('2093-3',axis=1)

    feature_names = np.array([x for x in df_X.columns.values if x != 'class'])

    X = df_X #.values #astype(float)
    y = np.array(label.values, dtype=bool).ravel()
    pt_ids = df_X.index.values

    assert(X.shape[1] == feature_names.shape[0])
    # pdb.set_trace()
    return X, y, pt_ids, feature_names, filename + '_long_imputed.csv'
