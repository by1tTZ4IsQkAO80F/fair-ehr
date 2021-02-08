import os
import itertools
import json
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, 
                                     GridSearchCV, ParameterGrid, train_test_split)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from metrics import balanced_accuracy
from quartile_exact_match import QuartileExactMatch
import time
from read_file import read_file
from feature_importance import feature_importance 
import pdb
import numpy as np
import pickle
from fairness import fairness, fair_classification_rates
from utils import sanitize_json


def evaluate_dataset(dataset, results_dir, random_state, longitudinal=False,
                     rare=True):

    print('reading data...',end='')
    features, labels, pt_ids, feature_names, zfile = read_file(dataset,
                                                               longitudinal,
                                                               rare)
    print('done.',len(labels),'samples,',np.sum(labels==1),'cases,',
          features.shape[1],'features')

    #### 
    ## use random undersampling to sample the controls
    ####
    #sampler = RandomUnderSampler(random_state=random_state)

    #print('sampling data...',end='')
    #X,y= sampler.fit_sample(features,labels)
    #sidx = sampler.sample_indices_
    #print('sampled data contains',
    #       np.sum(y==1), 'cases',
    #       np.sum(y==0),'controls')

    df= pd.DataFrame(features, columns = feature_names)
    df['label'] = labels

    ##########
    # metrics: test the best classifier on the held-out test set 
    print('getting train and test predictions...')
    
    ##########
    # save results to file
    # what to save:
    #   - basic stats; dataset
    #   - fairness scores 

    print('saving results...')
    
    dataset_name = dataset.split('/')[-1]

    out_data = {
            'dataset':dataset.split('/')[-1],
            'feature_names':feature_names,
            'random_state':random_state, 
            'total_samples':len(labels),
            'prevalence':np.sum(labels==1)/len(labels),
            }

    print('results_dir:',results_dir)
    print('dataset_name:',dataset_name)
    save_name = (
            results_dir + '/' 
            + '_'.join([
                        dataset_name, 
                        'baseline',
                        str(random_state)
                        ])
            )
    if longitudinal:
        save_name += '_long_'
    if not rare:
        save_name += '_no-rare'

    print('save_name:',save_name)

    if not os.path.exists(results_dir + '/' + dataset_name + '/' ):
        os.makedirs(results_dir + '/' + dataset_name + '/')

    # fairness assessment
    print('assessing fairness...')
    fair_data = fairness(df) #{'classification_rates':fair_classification_rates(df)}
    # fair_test = fairness(df)
    out_data['fairness'] = fair_data

    print('saving stats to {}...'.format(save_name+'.json'))
    # reformat any ndarrays
    out_data = sanitize_json(out_data)
    # for k in out_data:
    #     if isinstance(out_data[k],np.ndarray):
    #         out_data[k] = out_data[k].tolist()
    #     elif out_data[k].__class__.__name__ == 'int64'):
    #         print('casting int')
    #         out_data[k] = int(out_data[k])
    # save stats
    json.dump(out_data, open(save_name+'.json', 'w'))

###############################################################################
# main entry point                                                              
###############################################################################
import argparse                                                                 
import importlib                                                                

if __name__ == '__main__':                                                      

    # parse command line arguments                                              
    parser = argparse.ArgumentParser(                                           
        description="Evaluate a dataset's fairness.", add_help=True)       
    # parser.add_argument('-h', '--help', action='help',                        
    #                     help='Show this help message and exit.')              
    parser.add_argument('-ml', action='store', dest='ALG', default=None,        
            type=str,help='Name of estimator (with matching filename)')
    parser.add_argument('-rdir', action='store', dest='RDIR',default=None,      
                        type=str, help='Root Directory ') 
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
                        default=None, type=int, help='Seed / trial')
    parser.add_argument('-dataset', action='store', dest='DATASET',               
                        default=None, type=str, help='dataset')  
    parser.add_argument('-long', action='store_true', dest='LONGITUDINAL',
                        default=False, help='dataset')        
    parser.add_argument('-no_rare', action='store_false', dest='RARE',  
                        default=True,  
                        help='exclude rare variables')                       

    args = parser.parse_args()                                                  

    evaluate_dataset(args.DATASET, args.RDIR, args.RANDOM_STATE, 
                     args.LONGITUDINAL,
                     args.RARE)
