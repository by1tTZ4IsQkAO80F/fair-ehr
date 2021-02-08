"""Evaluation metrics for fairness of the models. """

from aif360.datasets import BinaryLabelDataset 
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.sklearn.metrics import difference
from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple
import copy
import itertools as it
import json
import numpy as np
from utils import sanitize_json
# (disparate_impact_ratio, average_odds_error,
#                                     generalized_fpr)

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, mean_squared_error)
from sklearn.calibration import calibration_curve
from protected_groups import (single_privileged,
                              single_unprivileged,
                              protected_attribute_names, 
                              privileged_group_levels, 
                              unprivileged_group_levels)
import pdb



def get_metric(data, y_pred, 
                # y_pred_proba, 
                priv=None, unpriv=None):
    # if no predictions are passed, just get metrics on the dataset itself
    if len(y_pred)==0:
        return BinaryLabelDatasetMetric(data, 
                                        privileged_groups = priv,
                                        unprivileged_groups = unpriv
                                        )
    else:
        classified_data = data.copy()
        classified_data.labels = y_pred
        # classified_data.scores = y_pred_proba
        return ClassificationMetric(dataset = data, 
                                    classified_dataset = classified_data,
                                    # privileged_groups = priv,
                                    privileged_groups = single_privileged,
                                    unprivileged_groups = single_unprivileged
                                    )

def fair_classification_rates(df, y_pred=[], y_pred_proba=[]):

    print('fair_classification_rates')

    # record simple classification rates across intersections of groups
    df2 = df
    if len(y_pred)>0:
        df2['prediction'] = y_pred
        df2['prediction_proba'] = y_pred_proba
    frames = []
    total_grp_prev = 0
    population_prevalence = np.sum(df['label'] == 1)/len(df)

    print('getting groupwise metrics')
    for grp, df_sub in df2.groupby(protected_attribute_names):
        sex,race,ethnicity = grp
        # print('sex:',sex,'race:',race,'ethn',ethnicity)

        y = df_sub['label'].values

        stats = {
            'SEX':sex,
            'RACE':race,
            'ETHNICITY': ethnicity,
            'case_count': np.sum(y==1),
            'group_case_prevalence': np.sum(y==1)/len(y),
            'group_size': len(df_sub),
            'group_prevalence': len(df_sub)/len(df2),
            'case_group_prevalence': np.sum(y==1)/np.sum(df2['label']==1),
            }
        # how prevalent is this group in the outcome cases, 
        # relative to the prevalence of the group in the population?
        stats['delta_case-group-prevalence_group-prevalence'] = \
                stats['case_group_prevalence'] - stats['group_prevalence']
        stats['ratio_case-group-prevalence_group-prevalence'] = \
                stats['case_group_prevalence'] / stats['group_prevalence']

        total_grp_prev += stats['group_prevalence']

        # additional stats for models
        if len(y_pred)>0:
            yhat = df_sub['prediction']
            yhat_proba=df_sub['prediction_proba']
            stats.update({
                'positive_predictions':np.sum(yhat==1),
                'mean_risk_prediction':yhat_proba.mean(),
                'sum_risk_prediction':yhat_proba.sum(),
                # sensitivity (recall)
                'TPR': np.sum((y==1) & (yhat == 1))/np.sum(y==1),
                # specificity
                'TNR': np.sum(( y==0) & (yhat == 0 ))/np.sum(y==0),
                # positive predictive value (precision)
                'PPV': np.sum(( yhat==1) & (y== 1 ))/np.sum(yhat==1),
                # negative predictive value
                'NPV': np.sum(( yhat==0) & (y== 0 ))/np.sum(yhat==0),
                # average precision (approximates area)
                'AUPRC': average_precision_score(y, yhat_proba),
                # area under roc curve
                'AUROC': roc_auc_score(y, yhat_proba) \
                            if len(np.unique(y)) == 2 else np.NAN
                })
            stats['FNR'] = 1-stats['TPR'] 
            stats['FPR'] = 1-stats['TNR'] 

            # calibration
            stats['calibration_loss'] = mean_squared_error(
                    *calibration_curve(y, yhat_proba))

        frames.append(stats)

    # sanity check
    assert(round(total_grp_prev,3) == 1.0)

    return frames

def fairness(df, y_pred=[], y_pred_proba=[]):
    """Returns a dictionary summarizing fairness measures of the model or
    dataset.
    """
    # what to measure:
    # - false omission rate parity (NPV Parity, relevant for rare disease)
    # - false negative rate parity (TPR parity, aka equality of opportunity, 
    #                               relevant for higher prevalence disease)
    print('making BinaryLabelDataset')
    data = BinaryLabelDataset(
                favorable_label=1.0, 
                unfavorable_label=0.0,
                df=df, 
                label_names = ['label'],
                protected_attribute_names = protected_attribute_names,
                privileged_protected_attributes = \
                        list(privileged_group_levels.values()),
                unprivileged_protected_attributes = \
                    list(unprivileged_group_levels.values())
                #max values are the privileged ones by default 
                # privileged_classes = [1, 6, 1],             
                #categorical_features (optional list to one hot encode)
                )
            
    print('getting metric')
    metric = get_metric(data, y_pred)

    # metrics for all groups, will not change wrt privileged/unprivileged def
    print('differential fairness')
    fairness_metrics = dict(
        smoothed_empirical_differential_fairness = \
                metric.smoothed_empirical_differential_fairness(),
        )

    if isinstance(metric, ClassificationMetric):
        fairness_metrics['between_all_groups_theil_index'] = \
                metric.between_all_groups_theil_index()
        fairness_metrics['differential_fairness_bias_amplification'] = \
                metric.differential_fairness_bias_amplification()
        #TODO: capture the group index with highest rich subgroup violation.
        fairness_metrics['rich_subgroup_FN'] = \
                metric.rich_subgroup(array_to_tuple(y_pred.reshape(-1,1)),'FN')
        fairness_metrics['rich_subgroup_FP'] = \
                metric.rich_subgroup(array_to_tuple(y_pred.reshape(-1,1)),'FP')
        fairness_metrics['average_abs_odds_difference'] = \
                metric.average_abs_odds_difference()
        fairness_metrics['error_rate_difference'] = \
                metric.error_rate_difference()
    # elif isinstance(metric, BinaryLabelDatasetMetric):
    #     print('consistency')
    #     fairness_metrics['consistency'] = metric.consistency()[0]

    print('all group metrics:',fairness_metrics)

    fairness_metrics['classification_rates'] = \
            fair_classification_rates(df, y_pred, y_pred_proba)
       
    return fairness_metrics
