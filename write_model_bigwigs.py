import itertools
import numpy
import os
from pyspark import SparkContext
import sys

sys.path.append(os.path.dirname(__file__))
import s3_library
import predictd_lib as pl

sc = SparkContext(appName='write_bigwigs',
                  pyFiles=[s3_library.__file__.replace('.pyc', '.py'),
                           pl.__file__.replace('.pyc', '.py')])

pl.sc = sc

(gmean, ct, ct_bias, assay, assay_bias, genome_params, hyperparams, data) = pl.load_model('s3://encodeimputation-alldata/predictd_demo/train_roadmap_test2', load_data_too=False)

data, subsets = pl.load_data('s3://encodeimputation-alldata/25bp/alldata.25bp.hars_plus_encodePilots.2500.coords_adjusted.rdd.pickle', win_per_slice=10000, fold_idx=0, valid_fold_idx=0, folds_fname='folds.5.8.pickle')
gtotal = data.join(genome_params).map(lambda (gidx, (d, (g, gb))): (gidx, d, g, gb)).repartition(120).persist()
gtotal.count()

data_idx = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/data_idx.pickle')
assay_list = [e2[0] for e2 in sorted(set([(e1[1], e1[-1][1]) for e1 in data_idx.values()]), key=lambda x: x[1])]
ct_list = [e2[0] for e2 in sorted(set([(e1[0], e1[-1][0]) for e1 in data_idx.values()]), key=lambda x: x[1])]
coords_to_output = list(zip(*itertools.product((ct_list.index('H1_Cell_Line'),), numpy.arange(len(assay_list)))))

pl.write_bigwigs2(gtotal, ct, ct_bias, assay, assay_bias, gmean, ct_list, assay_list, 'encodeimputation-alldata', 'predictd_demo/train_roadmap_test2', sinh=False, coords=coords_to_output)
