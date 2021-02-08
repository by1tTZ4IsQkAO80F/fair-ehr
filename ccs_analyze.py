import pandas as pd
import numpy as np
import argparse
import os
import errno
import sys
import itertools as it
from seeds import SEEDS
from joblib import Parallel, delayed

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(
            description="An analyst for quick ML applications.",
            add_help=False)
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-data_dir', action='store', dest='DIR', type=str,
                        default='/project/geis-ehr/phased/')
    parser.add_argument('-ccs', action='store', dest='CODES', 
            default=','.join([
                'Chronic-kidney-disease', 
                'Diabetes-mellitus-with-complications', 
                'Disorders-of-lipid-metabolism', 
                'Esophageal-disorders', 
                'Mood-disorders', 
                'Osteoporosis', 
                ]
                ), type=str,
                    help='Comma-separated list of ccs codes to run')
    parser.add_argument('-cutoffs',action='store',dest='CUTOFFS', type=str, 
            default='1,182,365,547,730')
    parser.add_argument('--norare', action='store_true', dest='NORARE', 
            default=False, help='do not include rare labs')
    parser.add_argument('--long', action='store_true', dest='LONG', 
            default=False, help='pass longitudinal data')
    parser.add_argument('--data-only', action='store_true', dest='DATAONLY', 
            default=False, help='Just evaluate the dataset')
    parser.add_argument('-ml', action='store', dest='LEARNERS', 
                        default=','.join([
                            'XGB',
                            'AD',
                            'GerryFair',
                            'RmDem_XGB',
                            'Reweigh_XGB',
                            'XGB_CalEqOdds'
                            'RandUnder_GerryFairFP_CalCV',
                            ]),
                        type=str,
                        help='Comma-separated list of ML methods to use '
                        '(should correspond to a py file name in ml/)')
    parser.add_argument('--local', action='store_true', dest='LOCAL', 
            default=False, help='Run locally instead of on HPC')
    parser.add_argument('-n_jobs', action='store', dest='N_JOBS', default=1, 
            type=int, help='Number of parallel jobs')
    parser.add_argument('-n_trials', action='store', dest='NTRIALS', 
            default=1, type=int, help='Number of parallel jobs')
    parser.add_argument('-seeds', action='store', dest='SEEDS', 
           default='', 
            type=str, help='specific trial seeds to run')
    parser.add_argument('-rdir', action='store', dest='RDIR', type=str, 
            help='Results directory',
            default='results/')
    parser.add_argument('-q', action='store', dest='QUEUE',
                        default='epistasis_normal', type=str, help='LSF queue')
    parser.add_argument('-m', action='store', dest='M', default=12000,
                        type=int, help='LSF memory request and limit (MB)')

    args = parser.parse_args()

    if args.SEEDS == '':
        seeds = SEEDS 
    else:
        seeds = [int(t) for t,s in args.SEEDS.split(',')]

    assert(args.NTRIALS <= len(seeds))
    seeds = seeds[:args.NTRIALS]

    codes = args.CODES.split(',')
    cutoffs = args.CUTOFFS.split(',')

    q = args.QUEUE

    if args.LOCAL:
        lpc_options = ''
    else:
        lpc_options = '--lsf -q {Q} -m {M} -n_jobs 1'.format(Q=q, M=args.M)

    learners = [ml for ml in args.LEARNERS.split(',')]  # learners
    model_dir = 'ml'

    script = 'evaluate_dataset' if args.DATAONLY else 'evaluate_model'
    if args.DATAONLY:
        learners = ['dummy']

    # write run commands
    all_commands = []
    job_info = []
    # submit per dataset,cutoff,trial,learner
    for seed, ml, cutoff, code in it.product(seeds, learners, cutoffs, codes):
        task_name = 'ccs_' + code
        if cutoff != '1': 
            task_name += '_cutoff' + cutoff

        rarity = '_noRare' if args.NORARE else ''
        
        dataset_prefix = 'geis_' + task_name.split('ccs_')[-1]
        dataset_path = '/'.join([args.DIR, task_name, dataset_prefix])
        print('dataset:', dataset_prefix)
        print('dataset path:', dataset_path)

        if cutoff == 1:
            task_name += '_cutoff' + cutoff

        results_path = '/'.join([args.RDIR, task_name])+'/'

        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        
        print('seed:', seed)

        all_commands.append(
            'python -u {script}.py -dataset {DATASET} -ml {ML} '
            '-rdir {RDIR} -seed {RS} '
            '{LONG} {NORARE}'.format(
                script = script,
                DATASET=dataset_path,
                # ML=model_dir + '.' + ml,
                ML=ml,
                RDIR=results_path,
                RS=seed,
                NORARE='-no_rare' if args.NORARE else '',
                N_JOBS=args.N_JOBS,
                LONG='-long' if args.LONG else ''
            )
        )

        job_info.append({
            'ml': ml, 
            'dataset': task_name,
            'results_path': results_path
            })

    print(all_commands)
    if args.LOCAL:   # run locally
        Parallel(n_jobs=args.N_JOBS)(
            delayed(os.system)(run_cmd) for run_cmd in all_commands)
            # delayed(print)(run_cmd) for run_cmd in all_commands)
    else:
        for i, run_cmd in enumerate(all_commands):
            job_name = job_info[i]['ml'] + '_' + job_info[i]['dataset']
            out_file = job_info[i]['results_path'] + job_name + '_%J.out'
            # error_file = out_file[:-4] + '.err'

            # GerryFair is a memory hog, give it more RAM!
            if 'GerryFair' in run_cmd:
                mem = args.M*2
            else:
                mem = args.M

            bsub_cmd = ('bsub -o {OUT_FILE} -n {N_CORES} -J {JOB_NAME} '
                        '-q {QUEUE} -R "span[hosts=1] rusage[mem={M}]" '
                        '-M {M} ').format(
                OUT_FILE=out_file,
                JOB_NAME=job_name,
                QUEUE=args.QUEUE,
                N_CORES=args.N_JOBS,
                M=mem
            )

            bsub_cmd += '"' + run_cmd + '"'
            print(bsub_cmd)
            os.system(bsub_cmd)     # submit jobs
