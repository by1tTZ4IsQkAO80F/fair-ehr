import os
import itertools
import json
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, 
                                     GridSearchCV, ParameterGrid, train_test_split)
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, make_scorer,
                             average_precision_score, mean_squared_error)
from sklearn.calibration import calibration_curve
from quartile_exact_match import QuartileExactMatch
import time
from read_file import read_file
from feature_importance import feature_importance 
import pdb
import numpy as np
import pickle
from fairness import fairness
from utils import sanitize_json, total_size
from aif360.sklearn.datasets.utils import standardize_dataset

def evaluate_model(dataset, results_dir, random_state, clf, clf_name, 
                   longitudinal=False,rare=True):

    print('reading data...',end='')
    features, labels, pt_ids, feature_names, zfile = read_file(dataset,
                                                               longitudinal,
                                                               rare)
    print('done.',len(labels),'samples,',np.count_nonzero(labels),'cases,',
          features.shape[1],'features')


    ### 
    # split into train/test 
    ###
    # sidx = np.arange(len(labels))
    X_train, X_test, y_train, y_test = \
        train_test_split(features, labels, 
                         train_size=0.5,
                         test_size=0.5,
                         # train_size=0.01,
                         # test_size=0.01,
                         random_state=random_state,
                         stratify = labels
                         )
    print('training on',np.count_nonzero(y_train),'cases and',
            len(X_train)-np.count_nonzero(y_train),'controls')
    print('testing on',np.count_nonzero(y_test),'cases and',
            len(X_test)-np.count_nonzero(y_test),'controls')
    assert(X_train.shape[1] == X_test.shape[1])

    random_states = [k for k in clf.get_params().keys() if 'random_state' in k]
    for rs in random_states:
        print('setting',rs,'to',random_state)
        clf.set_params(**{rs:random_state}) 


    t0 = time.process_time()


    print('X_train columns:',X_train.columns)
    print('X_train index:',X_train.index.names)
    print('fitting model...')
    if longitudinal:
        clf.fit(X_train, y_train, zfile, pt_ids[sidx_train])
    else:
        clf.fit(X_train,y_train)

    # if len(hyper_params)== 0: 
    runtime = time.process_time() - t0

    ##########
    # metrics: test the best classifier on the held-out test set 
    print('getting train and test predictions...')
    
    # args = [X_test]
    # if longitudinal:
    #     args += [zfile, pt_ids[sidx_test]]
    print('X_test cols:',X_test.columns)
    y_pred = np.array(clf.predict(X_test), dtype=bool)
    print('X_test cols:',X_test.columns)
    # get probabilities
    if getattr(clf, "predict_proba", None):
        y_pred_proba = clf.predict_proba(X_test)[:,1]
    elif getattr(clf, "decision_function", None):
        y_pred_proba = clf.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # if AD is used, the TF session has to be closed
    if 'AD' in clf_name:
        def close_sess(nested_est):
            # close the AD session in a pipeline or estimator
            if hasattr(nested_est, 'sess_'):
                print('closing sess_')
                nested_est.sess_.close()
            elif hasattr(nested_est, 'named_steps'):
                print('closing sess_')
                nested_est.named_steps['ad'].sess_.close()
            elif hasattr(nested_est, 'estimator'):
                close_sess(nested_est.estimator)
            elif hasattr(nested_est, 'base_estimator'):
                close_sess(nested_est.base_estimator)
            else:
                print('failed to close AD session')

        close_sess(clf)

    # check memory usage
    for n,x in [('clf',clf), 
                ('X_train',X_train), 
                ('y_test',y_test),
                ('y_pred_proba',y_pred_proba),
                ('y_test & y_pred', np.logical_and(y_test, y_pred))]:
        print('size of {}: {}'.format(n, total_size(x)))
    ##########
    # save results to file
    # what to save:
    #   - basic stats; dataset, clf_name, test set performance
    #   - pickled model
    #   - feature importance (shap values, maybe shap object?)
    #   - fairness scores on train and test

    print('saving results...')
    
    dataset_name = dataset.split('/')[-1]


    print('y_test:',y_test.shape,
          'y_pred:',y_pred.shape,
          'y_pred_proba:',y_pred_proba.shape)

    # P = (y_test==1)
    # N = (y_test==0)
    # PHAT = (y_pred==1)
    # NHAT = (y_pred==0)

    TP = np.count_nonzero(np.logical_and(y_test, y_pred))
    P = np.count_nonzero(y_test)
    Pguesses = np.count_nonzero(y_pred)

    TN = np.count_nonzero(~y_test & ~y_pred)
    N = np.count_nonzero(~y_pred)
    Nguesses = np.count_nonzero(~y_pred)

    out_data = {
            'dataset':dataset.split('/')[-1],
            # 'pt_ids_train':pt_ids[sidx_train],
            # 'pt_ids_test':pt_ids[sidx_test],
            'feature_names':feature_names,
            'clf_name':clf_name,
            'clf_params':str(clf.get_params()),
            'random_state':random_state, 
            'accuracy':accuracy,
            'macro_f1':macro_f1,
            'roc_auc':roc_auc,
            'runtime':runtime,
            'positive_predictions':Pguesses,
            'mean_risk_prediction':y_pred_proba.mean(),
            'sum_risk_prediction':y_pred_proba.sum(),
            # sensitivity
            'TPR': TP/P if P != 0 else np.NAN,
            # specificity
            'TNR': TN/N if N != 0 else np.NAN,
            # positive predictive value
            'PPV': TP/Pguesses if Pguesses != 0 else np.NAN,
            # negative predictive value
            'NPV': TN/Nguesses if Nguesses != 0 else np.NAN,
            # average precision (approximates area)
            'AUPRC': average_precision_score(y_test, y_pred_proba),
            # area under roc curve
            'AUROC': roc_auc_score(y_test, y_pred_proba) \
                        if len(np.unique(y_test)) == 2 else np.NAN,
            # calibration loss
            'calibration_loss': mean_squared_error(
                    *calibration_curve(y_test, y_pred_proba))
            }

    print('results_dir:',results_dir)
    print('dataset_name:',dataset_name)
    save_name = (
            results_dir + '/' 
            # + dataset_name + '/' 
            + '_'.join([
                        dataset_name, 
                        clf_name,
                        str(random_state)
                        ])
            )
    if longitudinal:
        save_name += '_long_'
    if not rare:
        save_name += '_no-rare'

    print('save_name:',save_name)

    # if not os.path.exists(results_dir + '/' + dataset_name + '/' ):
    #     os.makedirs(results_dir + '/' + dataset_name + '/')

    # TODO: this needs updating
    # print('saving feature importance') 
    # write feature importances
    # if not longitudinal:
    #     FI_data = feature_importance(clf, 
    #                                  clf_name, 
    #                                  feature_names, 
    #                                  X_train, 
    #                                  X_test)
    #     out_data.update(FI_data)

    # fairness assessment
    # make dataframes
    # df_train = X_train.copy()
    # df_train['label'] = y_train

    # df_test = pd.DataFrame(X_test, columns = feature_names)
    df_test = X_test #.copy()
    df_test['label'] = y_test
    print('assessing fairness...')
    # fair_train = fairness(best_clf, clf_name, df_train, df_test, protected)
    fair_test = fairness(df_test, y_pred, y_pred_proba)
    out_data['fairness'] = fair_test

    print('saving pickled model to {}...'.format(save_name+'.pkl'))
    # save model
    try:
        pickle.dump(clf, open(save_name+'.pkl','wb'))
    except Exception as e:
        print('Pickling failed:',e)

    print('saving stats to {}...'.format(save_name+'.json'))
    # reformat any ndarrays
    out_data = sanitize_json(out_data)

    print('model metrics:',
            json.dumps({k:v for k,v in out_data.items() if 'pt_ids' not in k},
                       sort_keys=True,
                       indent=4))
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
        description="Evaluate a method on a dataset.", add_help=True)       
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

    # import algorithm                                                          
    print('import from',args.ALG)                                     
    algorithm = importlib.__import__('ml.'+args.ALG,globals(),locals(),     
                                     ['clf','name'])                              

    print('algorithm:',algorithm.name,algorithm.clf)                            

    evaluate_model(args.DATASET, args.RDIR, args.RANDOM_STATE, 
                   algorithm.clf, algorithm.name, 
                   args.LONGITUDINAL,
                   args.RARE)
