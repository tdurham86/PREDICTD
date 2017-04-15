
import itertools
import numpy
import os
from pyspark import SparkContext, StorageLevel
import smart_open

import s3_library
import predictd_lib as pl
import assembled_data_to_rdd as adtr

def avg_impute(gtotal_elt, second_order_params, folds, data_coords):
    genome_params = [(tidx, pl.second_order_genome_updates(gtotal_elt, numpy.where(~folds[tidx]['train'][vidx]['train'].flatten()), z_mat, z_h, ac_bias)[-2:]) for (tidx, vidx), (z_mat, z_h, ac_bias, ct_assay, ct_assay_bias, gmean) in second_order_params]
    final_impute = numpy.zeros(gtotal_elt[1].shape)
    impute_sum = final_impute.copy()
    for tidx, test_fold in enumerate([elt['test'] for elt in folds]):
        imputed = [pl._compute_imputed2_helper(gtotal_elt[:2] + elt[1], second_order_params[idx][1][-3], second_order_params[idx][1][-2], second_order_params[idx][1][-1])[-1] for idx, elt in enumerate(genome_params) if elt[0] == tidx]
        imputed_sum = numpy.sum(imputed, axis=0)
        impute_sum += imputed_sum
        test_coords = numpy.where(~test_fold)
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
print('Loading Data')
data_idx = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/data_idx.pickle')
parts_idx = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/alldata.column_coords.pickle')
parts_idx_map = {v:k for k,v in s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/alldata.data_idx_coord_to_alldata_coord_map.pickle').items()}
parts_idx = list(zip(*[parts_idx_map[elt] for elt in parts_idx]))
all_parts = sorted(['s3://encodeimputation-alldata/{!s}'.format(elt.name) for elt in s3_library.glob_keys('encodeimputation-alldata', '25bp/alldata-parts/alldata.part*.txt.gz')], key=lambda x: int(os.path.basename(x).split('.')[1][4:]))
data_shape = (127, 24)

#read in models to use
print('Loading Models')
folds = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/folds.5.8.pickle')
data_coords = numpy.zeros(data_shape, dtype=bool)
for elt in folds:
    data_coords = numpy.logical_or(data_coords, ~elt['test'])
ct_param_paths = sorted([elt.name for elt in s3_library.glob_keys('encodeimputation2', 'NADAM-25bp/imputed_for_paper/test_fold*/valid_fold*/3D_model_ct_params.pickle')])
print(len(ct_param_paths))
second_order_params = []
for path in ct_param_paths:
    test_idx = int(os.path.dirname(os.path.dirname(path))[-1])
    valid_idx = int(os.path.dirname(path)[-1])
    second_order_params.append(((test_idx, valid_idx), prep_ctassays(path, folds[test_idx]['train'][valid_idx]['train'])))

#impute data for all models and average
print('Imputing and generating bedgraphs')
assay_list = [e2[0] for e2 in sorted(set([(e1[1], e1[-1][1]) for e1 in data_idx.values()]), key=lambda x: x[1])]
ct_list = [e2[0] for e2 in sorted(set([(e1[0], e1[-1][0]) for e1 in data_idx.values()]), key=lambda x: x[1])]
tmpdir = '/data2/tmp'
parts_at_once = 6
for idx in xrange(0, len(all_parts), parts_at_once):
    #start SparkContext
    sc = SparkContext(appName='impute_whole_genome',
                      pyFiles=[s3_library.__file__.replace('.pyc', '.py'),
                               pl.__file__.replace('.pyc', '.py'),
                               adtr.__file__.replace('.pyc', '.py')])
    pl.sc = sc
    adtr.pl = pl

    bdg_path = os.path.join(tmpdir, '{{0!s}}_{{1!s}}/{{0!s}}_{{1!s}}.{:05d}.{{2!s}}.txt'.format(idx))
    if idx >= 120:
        imputed_part = sc.parallelize(all_parts[idx:idx+parts_at_once], numSlices=parts_at_once).flatMap(lambda x: adtr.read_in_part(x, data_shape, parts_idx)).repartition(500).map(lambda (x,y): avg_impute((x,y,None,None), second_order_params, folds, data_coords)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
        imputed_part.count()

        sorted_w_idx = imputed_part.repartition(200).sortByKey().mapPartitionsWithIndex(lambda x,y: pl._construct_bdg_parts(x, y, bdg_path, ct_list, assay_list, None, None, None, None, None, winsize=25, sinh=False, coords=None, tmpdir=tmpdir)).count()
        imputed_part.unpersist()
        del(imputed_part)
    else:
        sorted_w_idx = sc.parallelize(all_parts[idx:idx+parts_at_once], numSlices=parts_at_once).flatMap(lambda x: adtr.read_in_part(x, data_shape, parts_idx)).repartition(200).sortByKey().mapPartitionsWithIndex(lambda x,y: pl._just_write_bdg_coords(x, y, bdg_path, winsize=25, tmpdir=tmpdir)).count()
    sc.stop()

#start SparkContext
sc = SparkContext(appName='impute_whole_genome',
                  pyFiles=[s3_library.__file__.replace('.pyc', '.py'),
                           pl.__file__.replace('.pyc', '.py'),
                           adtr.__file__.replace('.pyc', '.py')])
pl.sc = sc
adtr.pl = pl

print('Generating bigwig')
bdg_coord_glob = os.path.join(tmpdir, 'bdg_coords.*.txt')
sc.parallelize([bdg_coord_glob], numSlices=1).foreach(pl._combine_bdg_coords)

bdg_path = os.path.join(tmpdir, '{0!s}_{1!s}/{0!s}_{1!s}.*.{2!s}.txt')
out_bucket = 'encodeimputation-alldata'
out_root='predictd_demo/all-imputed'
#coords_to_output = list(zip(*itertools.product((ct_list.index('H1_Cell_Line'),), numpy.arange(len(assay_list)))))
coords_to_output = list(zip(*itertools.product(numpy.arange(len(ct_list)), numpy.arange(len(assay_list)))))
ct_assay_list = [(ct_list[c], assay_list[a]) for c, a in zip(*coords_to_output)]
track_lines = sc.parallelize(ct_assay_list, numSlices=len(ct_assay_list)/2).mapPartitions(lambda x: pl._compile_bdg_and_upload(x, out_bucket, out_root, bdg_path, tmpdir=tmpdir)).collect()

out_url = 's3://{!s}/{!s}'.format(out_bucket, os.path.join(out_root, 'track_lines.txt'))
with smart_open.smart_open(out_url, 'w') as out:
    out.write('\n'.join(track_lines))
    
sc.stop()
