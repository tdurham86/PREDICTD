
'''Once models have been trained on all desired data subsets, run this script to average the
results and produce bigwigs for the consensus imputation for missing data files.
'''

import argparse
import itertools
import numpy
import os
from pyspark import SparkContext, StorageLevel
import smart_open
import sys

sys.path.append(os.path.dirname(__file__))
import s3_library
import predictd_lib as pl
import assembled_data_to_rdd as adtr

def avg_impute(gtotal_elt, second_order_params):
    final_impute = numpy.zeros(gtotal_elt[1].shape, dtype=float)
    impute_sum = final_impute.copy()
    for (tidx, vidx), (z_mat, z_h, ac_bias, ct_assay, ct_assay_bias, gmean, subsets_flat) in second_order_params:
        gidx, data, genome, genome_bias = pl.second_order_genome_updates(gtotal_elt, numpy.where(~subsets_flat[pl.SUBSET_TRAIN]), z_mat, z_h, ac_bias)[:4]
        imputed = pl._compute_imputed2_helper((gidx, data, genome, genome_bias), ct_assay, ct_assay_bias, gmean)[1]
        non_train_coords = numpy.where(~numpy.logical_or(~subsets_flat[pl.SUBSET_TRAIN].reshape(data.shape),
                                                         ~subsets_flat[pl.SUBSET_VALID].reshape(data.shape)))
        impute_sum[non_train_coords] += 1.0 
        final_impute[non_train_coords] += imputed[non_train_coords]
    imputed_coords = numpy.where(impute_sum)
    final_impute[imputed_coords] /= impute_sum[imputed_coords]
    return (gidx, final_impute)

#def avg_impute(gtotal_elt, second_order_params, folds, data_coords):
#    genome_params = [(tidx, pl.second_order_genome_updates(gtotal_elt, numpy.where(~subsets_flat[pl.SUBSET_TRAIN]), z_mat, z_h, ac_bias)[-2:]) for (tidx, vidx), (z_mat, z_h, ac_bias, ct_assay, ct_assay_bias, gmean, subsets_flat) in second_order_params]
#    final_impute = numpy.zeros(gtotal_elt[1].shape)
#    impute_sum = final_impute.copy()
#    for tidx, test_fold in enumerate([elt['test'] for elt in folds]):
#        imputed = [pl._compute_imputed2_helper(gtotal_elt[:2] + elt[1], second_order_params[idx][1][3], second_order_params[idx][1][4], second_order_params[idx][1][5])[-1] for idx, elt in enumerate(genome_params) if elt[0] == tidx]
#        imputed_sum = numpy.sum(imputed, axis=0)
#        impute_sum += imputed_sum
#        test_coords = numpy.where(~test_fold)
#        final_impute[test_coords] = imputed_sum[test_coords]/len(imputed)
#    final_impute[~data_coords] = impute_sum[~data_coords]/len(genome_params)
#    return (gtotal_elt[0], final_impute)

def prep_ctassays(hyperparams_path, subsets, valid_ct_model_coords, valid_assay_model_coords, ri2=2.9):
    bucket_txt, key_txt = s3_library.parse_s3_url(hyperparams_path)
    ct_key_txt = os.path.join(os.path.dirname(key_txt), 'ct_factors.pickle')
    ct, ct_bias = s3_library.get_pickle_s3(bucket_txt, ct_key_txt)
    print(ct.shape)
    print(ct_bias.shape)
    ct, ct_bias = ct[list(valid_ct_model_coords),:], ct_bias[list(valid_ct_model_coords)]

    assay_key_txt = os.path.join(os.path.dirname(key_txt), 'assay_factors.pickle')
    assay, assay_bias = s3_library.get_pickle_s3(bucket_txt, assay_key_txt)
    assay, assay_bias = assay[list(valid_assay_model_coords),:], assay_bias[list(valid_assay_model_coords)]

    gmean = s3_library.get_pickle_s3(bucket_txt, os.path.join(os.path.dirname(key_txt), 'gmean.pickle'))

    all_valid_coords = list(zip(*itertools.product(valid_ct_model_coords, valid_assay_model_coords)))
    subsets_flat = [subsets[i][all_valid_coords].flatten() for i in range(len(subsets))]
    subsets_flat_coords = numpy.where(~subsets_flat[pl.SUBSET_TRAIN])
    ac_bias = numpy.add.outer(ct_bias, assay_bias).flatten()[subsets_flat_coords]
    ct_z = numpy.hstack([ct, numpy.ones((ct.shape[0], 1))])
    assay_z = numpy.hstack([assay, numpy.ones((assay.shape[0], 1))])
    z_mat = numpy.vstack([numpy.outer(ct_z[:,idx], assay_z[:,idx]).flatten()[subsets_flat_coords] for idx in xrange(ct_z.shape[1])])
    reg_coef_add = numpy.ones(z_mat.shape[0])
    reg_coef_add[-1] = 0
    reg_coef_add = numpy.diag(reg_coef_add * ri2)
    z_h = numpy.linalg.inv(numpy.dot(z_mat, z_mat.T) + reg_coef_add)

    ct_assay = numpy.vstack([numpy.outer(ct[:,idx], assay[:,idx]).flatten() for idx in xrange(ct.shape[1])])
    ct_assay_bias = numpy.add.outer(ct_bias, assay_bias).flatten()
    return (z_mat, z_h, ac_bias, ct_assay, ct_assay_bias, gmean, subsets_flat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_url')
    parser.add_argument('--model_data_idx')

    parser.add_argument('--agg_data_idx')
    parser.add_argument('--agg_coord_order')
    parser.add_argument('--agg_parts')

    parser.add_argument('--addl_agg_data_idx')
    parser.add_argument('--addl_agg_coord_order')

    parser.add_argument('--cts_to_impute')
    parser.add_argument('--assays_to_impute')

    parser.add_argument('--out_root_url')
    parser.add_argument('--tmpdir', default='/data/tmp')
    args = parser.parse_args()

    #read in metadata and locations of data files
    print('Loading Data')
    agg_data_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.agg_data_idx))
    agg_coord_order = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.agg_coord_order))
    if args.agg_coord_order == 's3://encodeimputation-alldata/25bp/alldata.column_coords.pickle':
        correction = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/alldata.data_idx_coord_to_alldata_coord_map.pickle')
        correction = {v:k for k,v in correction.items()}
        agg_coord_order = [correction[elt] for elt in agg_coord_order]
#    agg_ctet_coord_map = {tuple(elt[:2]):elt[-1] for elt in agg_data_idx}
    agg_ct_coord_map = {elt[0]:elt[-1][0] for elt in agg_data_idx.values()}
    agg_assay_coord_map = {elt[1]:elt[-1][1] for elt in agg_data_idx.values()}

    if args.agg_parts:
        bucket_txt, key_txt = s3_library.parse_s3_url(args.agg_parts)
#        parts_keys = s3_library.glob_keys(bucket_txt, os.path.join(key_txt, '*.part*.txt.gz'))
        parts_keys = s3_library.glob_keys(bucket_txt, os.path.join(key_txt, '*.part*.txt'))
    else:
        bucket_txt, key_txt = s3_library.parse_s3_url(args.agg_coord_order)
#        parts_keys = s3_library.glob_keys(bucket_txt, os.path.join(os.path.dirname(key_txt), '*.part*.txt.gz'))
        parts_keys = s3_library.glob_keys(bucket_txt, os.path.join(os.path.dirname(key_txt), '*.part*.txt'))
        
    agg_data_parts = sorted(['s3://{!s}/{!s}'.format(bucket_txt, elt.name) for elt in parts_keys],
                            key=lambda x: int(os.path.basename(x).split('.')[1][4:]))

    if args.addl_agg_coord_order:
        addl_agg_data_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.addl_agg_data_idx))
        bucket_txt, key_txt = s3_library.parse_s3_url(args.addl_agg_coord_order)
        addl_agg_coord_order = s3_library.get_pickle_s3(bucket_txt, key_txt)
#        parts_keys = s3_library.glob_keys(bucket_txt, os.path.join(os.path.dirname(key_txt), '*.part*.txt.gz'))
        parts_keys = s3_library.glob_keys(bucket_txt, os.path.join(os.path.dirname(key_txt), '*.part*.txt'))
        addl_agg_data_parts = sorted(['s3://{!s}/{!s}'.format(bucket_txt, elt.name) for elt in parts_keys],
                                     key=lambda x: int(os.path.basename(x).split('.')[1][4:]))
        
        if len(agg_data_parts) != len(addl_agg_data_parts):
            raise Exception('agg_data and addl_agg_data directories must contain the same number of part files.')
        agg_data_parts = list(zip(agg_data_parts, addl_agg_data_parts))

    #read in models to use
    print('Loading Models')
    model_data_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.model_data_idx))
    model_ctet_coord_map = {tuple(elt[:2]):elt[-1] for elt in model_data_idx}
    model_ct_coord_map = {elt[0]:elt[-1][0] for elt in model_data_idx.values()}
    model_assay_coord_map = {elt[1]:elt[-1][1] for elt in model_data_idx.values()}

#    #get ct assays in common between the model and agg_data
#    valid_ct = set(model_ct_coord_map) & set(agg_ct_coord_map)
#    valid_assay = set(model_assay_coord_map) & set(agg_assay_coord_map)
#    if not valid_ct or not valid_assay:
#        raise Exception('No ct or assay overlap between the input model and input data. Exiting.')

    #only take requested ct/assays that are present in the model/data overlap set
    if args.cts_to_impute and args.assays_to_impute:
        requested_ct = set(args.cts_to_impute.split(',')) & set(model_ct_coord_map)
        requested_assay = set(args.assays_to_impute.split(',')) & set(model_assay_coord_map)
        if not requested_ct or not requested_assay:
            raise Exception('No requested ct or assay present in model.')
        requested_ctassay = list(itertools.product(sorted(requested_ct), sorted(requested_assay)))
    elif args.cts_to_impute:
        requested_ct = set(args.cts_to_impute.split(',')) & set(model_ct_coord_map)
        if not requested_ct:
            raise Exception('No requested ct present in data and model overlap.')
        requested_ctassay = list(itertools.product(sorted(requested_ct), sorted(model_assay_coord_map.keys())))
    elif args.assays_to_impute:
        requested_assay = set(args.assays_to_impute.split(',')) & set(model_assay_coord_map)
        if not requested_assay:
            raise Exception('No requested assay present in data and model overlap.')
        requested_ctassay = list(itertools.product(sorted(model_ct_coord_map.keys()), sorted(requested_assay)))
    requested_coords = list(zip(*[(model_ct_coord_map[elt[0]], model_assay_coord_map[elt[1]]) 
                                  for elt in requested_ctassay]))

#    #ensure that there is training data for the overlapping ct/assays
    valid_ct, valid_ct_model_coords = zip(*sorted(model_ct_coord_map.items(), key=lambda x:x[1]))
    print(len(valid_ct_model_coords))
    print(valid_ct_model_coords[:5])
    valid_assay, valid_assay_model_coords = zip(*sorted(model_assay_coord_map.items(), key=lambda x:x[1]))
    print(len(valid_assay_model_coords))
    print(valid_assay_model_coords[:5])
    valid_ctassay = list(itertools.product(valid_ct, valid_assay))
#    valid_ctassay_w_data = set(valid_ctassay) - set(agg_data_idx)
#    if not valid_ctassay_w_data:
#        raise Exception('No valid ct/assay coords with data. Cannot train genome factors.')

#    #translate the ct/assays to model and agg data coords
#    print('Proceeding to impute genome-wide data for {!s} ct/assay pairs.'.format(len(valid_ctassay)))
    #coords to filter model parameters
#    valid_ct_model_coords = [model_ct_coord_map[elt] for elt in valid_ct]
#    valid_assay_model_coords = [model_assay_coord_map[elt] for elt in valid_assay]
#    merged_coord_map = {tuple(elt):(valid_ct.index(elt[0]), valid_assay.index(elt[1])) for elt in valid_ctassay}
#    requested_coords = list(zip(*[merged_coord_map[elt] for elt in requested_ctassay])) 
    #coords to assemble data from agg columns
#    agg_coord_to_ctassay = {elt[-1]:tuple(elt[:2]) for elt in agg_data_idx.values()}
#    agg_line_coord = [i for i in range(len(agg_coord_order)) if agg_coord_to_ctassay[agg_coord_order[i]] in valid_ctassay_w_data]
#    agg_mat_coord = [merged_coord_map[agg_coord_to_ctassay[agg_coord_order[elt]]] for elt in agg_line_coord]
#    model_coords = [tuple(coord) for coord in zip(*[model_ctet_coord_map[elt] for elt in valid_ctet])]
#    agg_coords = [tuple(coord) for coord in zip(*[agg_ctet_coord_map[elt] for elt in valid_ctet])]

    bucket_txt, key_txt = s3_library.parse_s3_url(args.model_url)
    key_glob = os.path.join(key_txt, '*hyperparameters.pickle')
    hyperparam_paths = sorted([elt.name for elt in s3_library.glob_keys(bucket_txt, key_glob)])
    second_order_params = []
    for path in hyperparam_paths:
        print(path)
        hyperparams = s3_library.get_pickle_s3(bucket_txt, path)
        subsets = hyperparams['subsets']
        path_url = 's3://{!s}/{!s}'.format(bucket_txt, path)
        second_order_params.append(((hyperparams['args'].fold_idx, hyperparams['args'].valid_fold_idx), prep_ctassays(path_url, subsets, valid_ct_model_coords, valid_assay_model_coords, ri2=hyperparams['args'].ri2)))

    #impute data for all models and average
    print('Imputing and generating bedgraphs')
    #assay_list = [e2[0] for e2 in sorted(set([(e1[1], e1[-1][1]) for e1 in data_idx.values()]), key=lambda x: x[1])]
    #ct_list = [e2[0] for e2 in sorted(set([(e1[0], e1[-1][0]) for e1 in data_idx.values()]), key=lambda x: x[1])]
    assay_list = valid_assay
    ct_list = valid_ct
    tmpdir = args.tmpdir
    parts_at_once = 6
    agg_data_shape = tuple(numpy.max(numpy.array([elt[-1] for elt in agg_data_idx.values()], dtype=int), axis=0) + 1)
    print('agg_data_shape: {!s}'.format(agg_data_shape))
    addl_agg_data_shape = None
    if args.addl_agg_coord_order:
        addl_agg_data_shape = tuple(numpy.max(numpy.array([elt[-1] for elt in addl_agg_data_idx.values()], dtype=int), axis=0) + 1)
        print('addl_agg_data_shape: {!s}'.format(addl_agg_data_shape))
    print(agg_data_parts)
    for idx in xrange(0, len(agg_data_parts), parts_at_once):
        print(agg_data_parts[idx:idx+parts_at_once])
        #start SparkContext
        sc = SparkContext(appName='impute_with_models_gen_bedgraph',
                          pyFiles=[s3_library.__file__.replace('.pyc', '.py'),
                                   pl.__file__.replace('.pyc', '.py'),
                                   adtr.__file__.replace('.pyc', '.py')])
        pl.sc = sc
        adtr.pl = pl

        bdg_path = os.path.join(tmpdir, '{{0!s}}_{{1!s}}/{{0!s}}_{{1!s}}.{:05d}.{{2!s}}.txt'.format(idx))
#    imputed_part = sc.parallelize(agg_data_parts[idx:idx+parts_at_once], numSlices=parts_at_once).flatMap(lambda x: adtr.read_in_part(x, data_shape, agg_mat_coord)).repartition(500).map(lambda (x,y): avg_impute((x,y,None,None), second_order_params, folds, data_coords)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
        agg_mat_coords = list(zip(*agg_coord_order))
        if addl_agg_data_shape is None:
            imputed_part = sc.parallelize(agg_data_parts[idx:idx+parts_at_once], numSlices=parts_at_once)\
                             .flatMap(lambda x: adtr.read_in_part(x, agg_data_shape, agg_mat_coords))\
                             .repartition(500)\
                             .map(lambda (x,y): avg_impute((x,y,None,None), second_order_params))\
                             .persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
        else:
            addl_agg_mat_coords = list(zip(*addl_agg_coord_order))
            addl_agg_to_agg_idx = pl._make_rdd2_to_rdd1_idx(agg_data_idx, addl_agg_data_idx)
            imputed_part = sc.parallelize(agg_data_parts[idx:idx+parts_at_once], numSlices=parts_at_once)\
                             .flatMap(lambda (x,y): zip(adtr.read_in_part(x, agg_data_shape, agg_mat_coords),
                                                      adtr.read_in_part(y,addl_agg_data_shape,addl_agg_mat_coords)))\
                             .map(lambda ((idx1,x),(idx2,y)): (idx1,pl._merge_rdds_helper(x,y,addl_agg_to_agg_idx)))\
                             .repartition(500)\
                             .map(lambda (x,y): avg_impute((x,y,None,None), second_order_params))\
                             .persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
        imputed_part.count()

        sorted_w_idx = imputed_part.repartition(200)\
                                   .sortByKey()\
                                   .mapPartitionsWithIndex(lambda x,y: pl._construct_bdg_parts(x, y, bdg_path, ct_list, assay_list, None, None, None, None, None, winsize=25, sinh=False, coords=requested_coords, tmpdir=tmpdir)).count()
        imputed_part.unpersist()
        del(imputed_part)
        sc.stop()

#####
#    sys.exit()
#####

    #start SparkContext
    sc = SparkContext(appName='impute_with_models_gen_bigwig',
                      pyFiles=[s3_library.__file__.replace('.pyc', '.py'),
                               pl.__file__.replace('.pyc', '.py'),
                               adtr.__file__.replace('.pyc', '.py')])
    pl.sc = sc
    adtr.pl = pl
    print('Generating bigwig')
    bdg_coord_glob = os.path.join(tmpdir, 'bdg_coords.*.txt')
    sc.parallelize([bdg_coord_glob], numSlices=1).foreach(pl._combine_bdg_coords)

    bdg_path = os.path.join(tmpdir, '{0!s}_{1!s}/{0!s}_{1!s}.*.{2!s}.txt')
    out_bucket, out_root = s3_library.parse_s3_url(args.out_root_url)
#    out_bucket = 'encodeimputation-alldata'
#    out_root='predictd_demo/all-imputed'
    track_lines = sc.parallelize(requested_ctassay, numSlices=len(requested_ctassay)/2).mapPartitions(lambda x: pl._compile_bdg_and_upload(x, out_bucket, out_root, bdg_path, tmpdir=tmpdir)).collect()

    out_url = 's3://{!s}/{!s}'.format(out_bucket, os.path.join(out_root, 'track_lines.txt'))
    with smart_open.smart_open(out_url, 'w') as out:
        out.write('\n'.join(track_lines))

    sc.stop()
