
'''
The results of PREDICTD are improved by averaging the imputed data from multiple models to produce a consensus imputed data track. This script will loop through a list of validation sets and train a model for each one so that the imputation results from multiple models can be averaged for a particular test set.
'''

import argparse
import copy
import numpy
import os
import subprocess
import sys

sys.path.append(os.path.dirname(__file__))
import s3_library
#import azure_library
#import impute_roadmap_consolidated_data as spark_model
import train_model as spark_model
spark_model.pl.s3_library = s3_library
#spark_model.pl.azure_library = azure_library

num_folds = 8

if __name__ == "__main__":
    parser = spark_model.parser
    args = parser.parse_args()
    out_root = args.out_root

    rs = numpy.random.RandomState(seed=args.factor_init_seed)

    cmd_line_path = os.path.join(out_root, 'command_line.txt')
    cmd_line_txt = ' '.join(sys.argv) + '\n'
    if args.data_url.startswith('s3'):
        bucket = s3_library.S3.get_bucket(args.run_bucket)
        key = bucket.new_key(cmd_line_path)
        key.set_contents_from_string(cmd_line_txt)
        spark_model.pl.STORAGE = 'S3'
#    elif args.data_url.startswith('wasb'):
#        azure_library.load_blob_from_text(args.run_bucket, cmd_line_path, cmd_line_txt)
#        spark_model.pl.STORAGE = 'BLOB'
    else:
        raise Exception('Unrecognized URL prefix on data url: {!s}'.format(args.data_url))

    active_fold_path = os.path.join(out_root, 'active_fold_idx.pickle')
    try:
        if spark_model.pl.STORAGE == 'S3':
#            range_start = s3_library.get_pickle_s3(args.run_bucket, active_fold_path)
##        elif spark_model.pl.STORAGE == 'BLOB':
##            range_start = azure_library.get_blob_pickle(args.run_bucket, active_fold_path)
            range_start = len(s3_library.glob_keys(args.run_bucket,
                                                   os.path.join(out_root, 'valid_fold*/ct_factors.pickle')))
    except:
        range_start = 0

    rs2 = numpy.random.RandomState(seed=args.factor_init_seed)
    for iter_idx in range(range_start, num_folds):
        sc = spark_model.SparkContext(appName='avg_valid_folds',
                                      pyFiles=[os.path.join(os.path.dirname(__file__), 's3_library.py'),
                                               #os.path.join(os.path.dirname(__file__), 'impute_roadmap_consolidated_data.py'),
                                               os.path.join(os.path.dirname(__file__), 'train_model.py'),
                                               os.path.join(os.path.dirname(__file__), 'predictd_lib.py')])
        spark_model.pl.sc = sc
        if spark_model.pl.STORAGE == 'S3':
            s3_library.set_pickle_s3(args.run_bucket, active_fold_path, iter_idx)
#        elif spark_model.pl.STORAGE == 'BLOB':
#            azure_library.load_blob_pickle(args.run_bucket, active_fold_path, iter_idx)

        idx_args = copy.deepcopy(args)
        valid_cmd_line_path = ''
        while not valid_cmd_line_path or s3_library.S3.get_bucket(args.run_bucket).get_key(valid_cmd_line_path):
            valid_idx = rs2.randint(8)
            idx_args.out_root = os.path.join(out_root, 'valid_fold{!s}'.format(valid_idx))
            valid_cmd_line_path = os.path.join(idx_args.out_root, 'command_line.txt')
            
        valid_cmd_line_txt = cmd_line_txt.strip().replace(os.path.basename(__file__).replace('.pyc', '.py'), os.path.basename(spark_model.__file__).replace('.pyc', '.py')).replace(out_root, idx_args.out_root)

        idx_args.valid_fold_idx = valid_idx
        valid_cmd_line_txt += ' --valid_fold_idx={!s}'.format(idx_args.valid_fold_idx)

        idx_args.factor_init_seed = rs.randint(int(1e6))
        valid_cmd_line_txt += ' --factor_init_seed={!s}'.format(idx_args.factor_init_seed)

#        idx_args.data_iteration_seed = rs.randint(int(1e6))
#        valid_cmd_line_txt += ' --data_iteration_seed={!s}'.format(idx_args.data_iteration_seed)

#        idx_args.random_loci_fraction_seed = rs.randint(int(1e6))
#        valid_cmd_line_txt += ' --random_loci_fraction_seed={!s}'.format(idx_args.random_loci_fraction_seed)

#        idx_args.train_on_subset_seed = rs.randint(int(1e6))
#        valid_cmd_line_txt += ' --train_on_subset_seed={!s}'.format(idx_args.train_on_subset_seed)

        if iter_idx < range_start:
            continue

        if idx_args.data_url.startswith('s3'):
            bucket = s3_library.S3.get_bucket(idx_args.run_bucket)
            key = bucket.new_key(valid_cmd_line_path)
            key.set_contents_from_string(valid_cmd_line_txt)
#        elif idx_args.data_url.startswith('wasb'):
#            azure_library.load_blob_from_text(idx_args.run_bucket, valid_cmd_line_path, valid_cmd_line_txt)
        else:
            raise Exception('Unrecognized URL prefix on data url: {!s}'.format(args.data_url))

        #if this is a restart, make sure we don't just restart again in the next iteration
        if iter_idx == range_start and args.restart is True:
            args.restart = False
            args.checkpoint = None
#        #don't use the same pctl_res for the next fold because the training set changes.
#        if args.pctl_res is not None:
#            args.pctl_res = None
        spark_model.train_consolidated(idx_args)
        sc.stop()

    imp_result_glob = os.path.join(out_root, '*/hyperparameters.pickle')
    if spark_model.pl.STORAGE == 'S3':
        glob_result1 = s3_library.glob_keys(args.run_bucket, imp_result_glob)
#        glob_result2 = s3_library.glob_keys(args.run_bucket, os.path.join(os.path.dirname(imp_result_glob), 'num_parts.pickle'))
#    elif spark_model.pl.STORAGE == 'BLOB':
#        glob_result1 = azure_library.glob_blobs(args.run_bucket, imp_result_glob)
#        glob_result2 = azure_library.glob_blobs(args.run_bucket, os.path.join(os.path.dirname(imp_result_glob), 'num_parts.pickle'))
    imp_result_paths = ['s3://{!s}/{!s}'.format(args.run_bucket, key_path) 
                        for key_path in set([os.path.dirname(elt.name) for elt in glob_result1])]
#    num_folds = 4
    if len(imp_result_paths) != num_folds:
        raise Exception('Only {!s} of {!s} validation folds calculated.'.format(len(imp_result_paths), num_folds))
    sc = spark_model.SparkContext(appName='avg_valid_folds',
                                  pyFiles=[os.path.join(os.path.dirname(__file__), 's3_library.py'),
#                                              os.path.join(os.path.dirname(__file__), 'azure_library.py'),
#                                           os.path.join(os.path.dirname(__file__), 'impute_roadmap_consolidated_data.py'),
                                           os.path.join(os.path.dirname(__file__), 'train_model.py'),
                                           os.path.join(os.path.dirname(__file__), 'predictd_lib.py')])
    spark_model.pl.sc = sc
    if spark_model.pl.STORAGE == 'S3':
#        storage_url_fmt = 's3n://{!s}/{!s}'
        storage_url_fmt = 's3://{!s}/{!s}'
#    elif spark_model.pl.STORAGE == 'BLOB':
#        storage_url_fmt = 'wasbs://{!s}@imputationstoretim.blob.core.windows.net/{!s}'
#    to_join = [sc.pickleFile(storage_url_fmt.format(args.run_bucket, elt)) for elt in imp_result_paths]
    avg_imp = spark_model.pl.impute_and_avg(imp_result_paths, coords='test').persist()
    avg_imp.count()
    out_url = storage_url_fmt.format(args.run_bucket, os.path.join(out_root, '3D_svd_imputed.avg.test_set.rdd.pickle'))
    spark_model.pl.save_rdd_as_pickle(avg_imp, out_url)
    sc.stop()

    #transform averaged models
#    cmd = ['spark-submit', os.path.join(os.path.dirname(__file__), 'transform_imputed_data.py'), '--data_url={!s}'.format(args.data_url), '--imputed_rdd={!s}'.format(out_url), '--fold_idx={!s}'.format(args.fold_idx), '--valid_fold_idx=-1', '--num_percentiles=100']
#    subprocess.check_call(cmd)
