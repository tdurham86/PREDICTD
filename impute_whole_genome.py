
import numpy
import os
from pyspark import SparkContext

import s3_library
import predictd_lib as pl
import assembled_data_to_rdd as adtr

def avg_impute(gtotal_elt, second_order_params, folds, data_coords):
    genome_params = [(tidx, pl.second_order_genome_updates(gtotal_elt, folds[tidx]['train'][vidx]['train'], z_mat, z_h, ac_bias)[-2:]) for (tidx, vidx), (z_mat, z_h, ac_bias, ct_assay, ct_assay_bias, gmean) in second_order_params]
    final_impute = numpy.zeros(gtotal_elt[1].shape)
    impute_sum = final_impute.copy()
    for tidx, elt in enumerate([elt['test'] for elt in folds]):
        imputed = [pl._compute_imputed2_helper(gtotal_elt[:2] + elt[1:], second_order_params[idx][-3], second_order_params[idx][-2], second_order_params[idx][-1])[-1] for idx, elt in enumerate(genome_params) if elt[0] == tidx]
        imputed_sum = numpy.sum(imputed, axis=0)
        impute_sum += imputed_sum
        test_coords = numpy.where(~elt)
        final_impute[test_coords] = imputed_sum[test_coords]/len(imputed)
    final_impute[~data_coords] = impute_sum[~data_coords]/len(genome_params)
    return (gtotal_elt[0], final_impute)

def prep_ctassays(ct_params_path, train_subset, ri=2.9):
    ct, ct_bias = s3_library.get_pickle_s3('encodeimputation2', ct_params_path)
    assay, assay_bias = s3_library.get_pickle_s3('encodeimputation2', os.path.join(os.path.dirname(ct_params_path), '3D_model_assay_params.pickle'))
    gmean = s3_library.get_pickle_s3('encodeimputation2', os.path.join(os.path.dirname(ct_params_path), 'gmean.pickle'))
    subset_flat_coords = numpy.where(~train_subset.flatten())
    ac_bias = numpy.add.outer(ct_bias, assay_bias).flatten()[subset_flat_coords]
    ct_z = numpy.hstack([ct, numpy.ones((ct.shape[0], 1))])
    assay_z = numpy.hstack([assay, numpy.ones((assay.shape[0], 1))])
    z_mat = numpy.vstack([numpy.outer(ct_z[:,idx], assay_z[:,idx]).flatten()[subset_flat_coords] for idx in xrange(ct_z.shape[1])])
    reg_coef_add = numpy.ones(z_mat.shape[0])
    reg_coef_add[-1] = 0
    reg_coef_add = numpy.diag(reg_coef_add * ri)
    z_h = numpy.linalg.inv(numpy.dot(z_mat, z_mat.T) + reg_coef_add)

    ct_assay = numpy.vstack([numpy.outer(ct[:,idx], assay[:,idx]).flatten() for idx in xrange(ct.shape[1])])
    ct_assay_bias = numpy.add.outer(ct_bias, assay_bias).flatten()
    return (z_mat, z_h, ac_bias, ct_assay, ct_assay_bias, gmean)

#read in metadata and locations of data files
data_idx = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/data_idx.pickle')
parts_idx_map = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/alldata.data_idx_coord_to_alldata_coord_map.pickle')
all_parts = sorted(['s3://encodeimputation-alldata/{!s}'.format(elt.name) for elt in s3_library.glob_keys('encodeimputation-alldata', '25bp/alldata-parts/alldata.part*.txt.gz')], key=lambda x: int(os.path.basename(x).split('.')[1][4:]))
data_shape = (127, 24)

#read in models to use
folds = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/folds.5.8.pickle')
data_coords = numpy.zeros(data_shape, dtype=bool)
for elt in folds:
    data_coords = numpy.logical_or(data_coords, ~elt['test'])
ct_param_paths = sorted([elt.name for elt in s3_library.glob_keys('encodeimputation2', 'NADAM-25bp/imputed_for_paper/test_fold*/valid_fold*/3D_model_ct_params.pickle')])
second_order_params = []
for path in ct_param_paths:
    test_idx = int(os.path.dirname(os.path.dirname(path))[-1])
    valid_idx = int(os.path.dirname(path)[-1])
    second_order_params.append(((test_idx, valid_idx), prep_ctassays(path, folds[test_idx]['train'][valid_idx]['train'])))

#start SparkContext
sc = SparkContext(appName='impute_whole_genome',
                  pyFiles=[s3_library.__file__.replace('.pyc', '.py'),
                           pl.__file__.replace('.pyc', '.py'),
                           adtr.__file__.replace('.pyc', '.py')])
pl.sc = sc
adtr.pl = pl

#impute data for all models and average
assay_list = [e2[0] for e2 in sorted(set([(e1[1], e1[-1][1]) for e1 in data_idx.values()]), key=lambda x: x[1])]
ct_list = [e2[0] for e2 in sorted(set([(e1[0], e1[-1][0]) for e1 in data_idx.values()]), key=lambda x: x[1])]
for idx in xrange(0, len(all_parts), 2):
    imputed_part = sc.parallelize(all_parts[idx:idx+2], numSlices=2).flatMap(lambda x: adtr.read_in_part(x, data_shape, col_coords)).repartition(500).map(lambda (x,y): avg_impute((x,y,None,None), second_order_params, folds, data_coords)).persist()
    imputed_part.count()
    
    tmpdir = '/data/tmp'
    bdg_path = os.path.join(tmpdir, '{{!s}}_{{!s}}.{!s}.{{!s}}.txt'.format(idx))
    sorted_w_idx = imputed_part.sortByKey().map(lambda (x,y): y).mapPartitionsWithIndex(lambda x,y: pl._construct_bdg_parts(x, y, bdg_path, ct_list, assay_list, None, None, None, None, None, winsize=25, sinh=False, coords=None)).count()
    break
sc.stop()
