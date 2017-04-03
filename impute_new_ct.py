
import argparse
import itertools
import numpy
import os
from pyspark import SparkContext, StorageLevel
import scipy.sparse as sps
import sys

sys.path.append(os.path.dirname(__file__))
import predictd_lib as pl
import s3_library

def _make_gtotal_plus_noise(gtotal_elt, rseed, noise_spread=0.01, plus=True):
    rs = numpy.random.RandomState(rseed)
    gidx, data, genome, genome_bias = gtotal_elt[:4]
    gtotal_rest = (numpy.zeros(genome.shape), #genome_m1
                   numpy.zeros(genome.shape), #genome_m2
                   0,                         #genome_t
                   0,                         #genome_bias_m1
                   0,                         #genome_bias_m2
                   None)                      #position-wise subsets
    if plus is True:
        return (gidx, data, genome + rs.normal(0, noise_spread, gtotal_elt[2].shape),
                genome_bias + rs.normal(0, noise_spread)) + gtotal_rest
    else:
        return (gidx, data, genome/noise_spread, genome_bias/noise_spread) + gtotal_rest

def train_new_ct(args):
    #read in data
    print('Read in data for new cell type.')
    ct_data = pl.load_data(args.data_url, win_per_slice=args.win_per_slice)

    #read in model params (assay & genome)
    print('Read in parameters from previously-trained model.')
    (gmean, ct, ct_bias, assay, assay_bias, genome_params,
     rc, ra, ri, rbc, rba, rbi, learning_rate, model_data, model_subsets) = pl.load_model(args.model_url, load_data_too=True)
    rc = args.rc
    rbc = args.rbc
    learning_rate = learning_rate if args.learning_rate is None else args.learning_rate
    #set new cell type parameters
    rs = numpy.random.RandomState(args.init_seed)
    ct = numpy.vstack([ct, rs.uniform(-0.33, 0.33, size=(ct_data.first()[1].shape[0], ct.shape[1]))])
    ct_bias = numpy.concatenate([ct_bias, rs.uniform(0, 0.1, size=(ct_data.first()[1].shape[0],))])

    #add a new row to the subset matrices
    subsets = [numpy.vstack([elt, numpy.zeros((ct_data.first()[1].shape[0], elt.shape[1])).astype(bool)]) for elt in model_subsets]
    sample_frac = 50.0/ct_data.count()
    ct_data_coords = numpy.any(numpy.array([elt[1].toarray() for elt in ct_data.sample(False, sample_frac, rs.randint(0,10000)).collect()], dtype=bool), axis=0)
    subsets[pl.SUBSET_TRAIN][(0-ct_data_coords.shape[0]):,:] = ~ct_data_coords

    for elt in subsets:
        print(elt.shape, numpy.sum(~elt))

    #assemble initial factors and gtotal
    noise_seed = rs.randint(0, int(1e8))
    noise_spread = 0.001
#    assay = assay + rs.normal(0, noise_spread, assay.shape)
#    assay_bias = assay_bias + rs.normal(0, noise_spread, assay_bias.shape)
#    ct = ct + rs.normal(0, noise_spread, ct.shape)
#    ct_bias = ct_bias + rs.normal(0, noise_spread, ct_bias.shape)
    gp_type = len(genome_params.first())
    to_join = genome_params if gp_type == 2 else genome_params.map(lambda (idx, g, gb): (idx, (g, gb)))
    gtotal_all_tmp = model_data.join(ct_data).mapValues(lambda (x,y): pl.vstack_csr_matrices(x,y)).join(to_join).map(lambda (gidx, (d, (g, gb))): (gidx, d, g, gb, None)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    gmean = pl.calc_gmean(gtotal_all_tmp, subset=subsets[pl.SUBSET_TRAIN])
    gtotal_all = gtotal_all_tmp.map(lambda x: (x[0], pl.subtract_from_csr(x[1], gmean)) + x[2:]).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    gtotal_all.count()
    gtotal_all_tmp.unpersist()
    del(gtotal_all_tmp)
    gtotal_init = gtotal_all.sample(False, args.training_fraction, args.training_fraction_seed).repartition(120).zipWithIndex().map(lambda ((gidx, d, g, gb, s), zidx): _make_gtotal_plus_noise((gidx, d, g, gb), noise_seed + zidx, noise_spread=1, plus=False)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    gtotal_init.count()
    genome_params.unpersist()
    del(genome_params)
    model_data.unpersist()
    del(model_data)

    #train factors with parallel SGD until convergence on validation gtotal
    print('Train new parameters.')
    iseed = rs.randint(0, int(1e8))
    gtotal, ct, ct_bias, assay, assay_bias, iter_errs = pl.train_predictd(gtotal_init, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, learning_rate, args.run_bucket, args.out_root, args.iters_per_mse, args.batch_size, args.win_size, args.win_spacing, args.win2_shift, args.pval, args.lrate_search_num, init_seed=iseed, min_iters=args.min_iters, max_iters=args.max_iters, subsets=subsets)

    #train genome factors across whole genome
    print('Apply new cell type parameters across genome.')
    genome_total = pl.train_genome_dimension(gtotal_all, 
                                             ct, ct_bias,
                                             assay, assay_bias, args.ri_2nd_order, subsets=subsets)

    final_mse = pl.calc_mse(genome_total, ct, rc, ct_bias, rbc, assay, ra,
                            assay_bias, rba, ri, rbi, subsets=subsets)
    iter_errs.append(final_mse)

    #save model to S3
    print('Save newly-trained model to S3.')
#    hyperparameters = {'rc':rc, 'rbc':rbc, 'ra':ra, 'rba':rba, 'ri':ri, 'rbi':rbi, 'ri_2nd_order':args.ri_2nd_order,
#                       'learning_rate':learning_rate, 'iters_per_mse':args.iters_per_mse,
#                       'batch_size':args.batch_size, 'win_size':args.win_size, 'win_spacing':args.win_spacing,
#                       'win2_shift':args.win2_shift, 'pval':args.pval, 'lrate_search_num':args.lrate_search_num,
#                       'training_fraction':args.training_fraction, 
#                       'training_fraction_seed':args.training_fraction_seed, 'init_seed':args.init_seed,
#                       'max_iters':args.max_iters, 'min_iters':args.min_iters, 'subsets':None, 'beta1':0.9,
#                       'beta2':0.999, 'epsilon':1e-8, 'run_bucket':args.run_bucket, 'out_root':args.out_root,
#                       'data_url':args.data_url, 'data_idx':args.data_idx_url, 'model_url':args.model_url, 
#                       'model_data_idx':args.model_data_idx_url}
    hyperparameters = {'args':args, subsets=subsets}
    pl.save_model_to_s3(gmean, ct, ct_bias, assay, assay_bias, genome_total, iter_errs, hyperparameters, 
                        args.run_bucket, args.out_root, iter_errs_header_line='Iter\tObjective\tTrain\tValid\n')

    #make browser view
    print('Generate UCSC Genome Browser tracks.')
    model_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.model_data_idx_url))
    assay_list = [e2[0] for e2 in sorted(set([(e1[1], e1[-1][1]) for e1 in model_idx.values()]), key=lambda x: x[1])]
    ct_list = [e2[0] for e2 in sorted(set([(e1[0], e1[-1][0]) for e1 in model_idx.values()]), key=lambda x: x[1])]
    data_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(args.data_idx_url))
    ct_list += [e2[0] for e2 in sorted(set([(e1[0], e1[-1][0]) for e1 in data_idx.values()]), key=lambda x: x[1])]
    coords_to_output = list(zip(*itertools.product((0 - (numpy.arange(ct_data.first()[1].shape[0]) + 1)), numpy.arange(len(assay_list)))))
    pl.write_bigwigs2(genome_total.repartition(120), ct, ct_bias, assay, assay_bias, gmean, ct_list, assay_list, args.run_bucket, args.out_root, sinh=not args.no_bw_sinh, coords=coords_to_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_url', help='The S3 url to the trained SVD3D model to provide parameters.')
    parser.add_argument('--model_data_idx_url')
    parser.add_argument('--data_url', help='The S3 url to the cell type data to run the model on.')
    parser.add_argument('--data_idx_url')
    parser.add_argument('--run_bucket')
    parser.add_argument('--out_root')
    parser.add_argument('--init_seed', type=int, default=1)
    parser.add_argument('--iters_per_mse', type=int, default=3)
    parser.add_argument('--lrate_search_num', type=int, default=3)
#default is to pick a smaller subset of genomic positions to train on (default: 0.1% of the whole genome)
    parser.add_argument('--training_fraction', type=float, default=0.002)
    parser.add_argument('--training_fraction_seed', type=int)
#automatically determine batch size based on the number of cell types we are imputing
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--win_per_slice', type=int)
    parser.add_argument('--rc', type=float, default=0.000288)
    parser.add_argument('--rbc', type=float, default=0.0)
    parser.add_argument('--ri_2nd_order', type=float, default=2.9)
    parser.add_argument('--learning_rate', type=float)

    parser.add_argument('--win_size', type=int, default=15)
    parser.add_argument('--win_spacing', type=int, default=5)
    parser.add_argument('--win2_shift', type=float, default=1e-05)
    parser.add_argument('--pval', type=float, default=0.05)

    parser.add_argument('--min_iters', type=int, default=None)
    parser.add_argument('--max_iters', type=int, default=None)

    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--checkpoint')

    parser.add_argument('--no_bw_sinh', default=False, action='store_true')

    args = parser.parse_args()

    sc = SparkContext(appName='train_new_ct',
                      pyFiles=[os.path.join(os.path.dirname(__file__), 'predictd_lib.py'),
                               os.path.join(os.path.dirname(__file__), 's3_library.py')])
    pl.sc = sc
    if args.data_url.startswith('s3'):
        if not s3_library.glob_keys(args.run_bucket, os.path.join(args.out_root, 'command_line.txt')):
            bucket = s3_library.S3.get_bucket(args.run_bucket)
            key = bucket.new_key(os.path.join(args.out_root, 'command_line.txt'))
            key.set_contents_from_string(' '.join(sys.argv) + '\n')
        elif not args.restart:
            raise Exception('Error: Output directory already exists.')
    train_new_ct(args)
    sc.stop()
