Code used in the experiments for "Navigating intersectional biases in predictive health models". 

# Installation


See `environment.yml` for the conda environment configuration. 

# Code structure

`ccs_analyze.py`: responsible for running the experiment. 

`evaluate_dataset.py`: evaluates a dataset for fairness. 
 
`evaluate_model.py`: evaluates a model for fairness. 

`fairness.py`: methods for evaluating fairness over subgroups. 

`ml/*.py`: model definitions. 

`protected_groups.py`: definitions of protected attributes in the datasets. 


# Running the experiment

The following line will re-run the experiment in the paper: 

```
python ccs_analyze.py -rdir results_ccs -n_trials 50 -data_dir ~/data/geis-ehr/ -cutoffs 365 -ml XGB,Reweigh_XGB,RmDem_XGB,XGB_CalEqOdds,AD,RandUnder_GerryFairFP_CalCV
```

See `python ccs_analyze.py -h` for other experiment options. 



# Contact

William La Cava, lacava@upenn.edu
