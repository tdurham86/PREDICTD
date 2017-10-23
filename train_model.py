
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

def train_consolidated(args):
    #read in data
    print('Read in data.')
    data, subsets = pl.load_data(args.data_url, win_per_slice=args.win_per_slice, fold_idx=args.fold_idx,
                                 valid_fold_idx=args.valid_fold_idx, folds_fname=args.folds_fname)
    data_idx_url = os.path.splitext(args.data_url)[0] + '.data_idx.pickle'
    data_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(data_idx_url))
    if args.addl_data_url:
        orig_data, orig_subsets, orig_data_idx = data, subsets, data_idx
        user_data = pl.load_data(args.addl_data_url, win_per_slice=args.win_per_slice)
        user_data_idx_url = os.path.splitext(args.addl_data_url)[0] + '.data_idx.pickle'
        user_data_idx = s3_library.get_pickle_s3(*s3_library.parse_s3_url(user_data_idx_url))
        data, data_idx, subsets = pl.merge_rdds(orig_data, orig_data_idx, user_data, user_data_idx, orig_subsets)
    s3_library.set_pickle_s3(args.run_bucket, os.path.join(args.out_root, 'data_idx.pickle'), data_idx)

    rc = args.rc/(args.latent_factors * data.first()[1].shape[0])
    ra = args.ra/(args.latent_factors * data.first()[1].shape[1])
    ri = args.ri/(args.latent_factors * data.count())
    rbc = args.rbc
    rba = args.rba
    rbi = args.rbi
    learning_rate = args.learning_rate
    #initialize factor matrices
    rs = numpy.random.RandomState(args.factor_init_seed)
    gmean = pl.calc_gmean(data, subset=subsets[pl.SUBSET_TRAIN])
    data2 = data.mapValues(lambda x: pl.subtract_from_csr(x, gmean)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    ct, ct_bias, assay, assay_bias, genome, genome_bias = pl.init_factors(data2, subsets[pl.SUBSET_TRAIN], args.latent_factors, init_seed=rs.randint(0,int(1e8)), uniform_bounds=(-0.33, 0.33))
    gtotal_all = data2.join(genome.join(genome_bias)).map(lambda (idx, (d, (g, gb))): (idx, d, g, gb, None)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    gtotal_all.count()
    data.unpersist()
    del(data)
    genome.unpersist()
    del(genome)
    genome_bias.unpersist()
    del(genome_bias)

#    gtotal_init = gtotal_all.sample(False, args.training_fraction, args.training_fraction_seed).repartition(20).zipWithIndex().map(lambda ((gidx, d, g, gb, s), zidx): _make_gtotal_plus_noise((gidx, d, g, gb), zidx, noise_spread=1, plus=False)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    training_fraction_seed = rs.randint(0, int(1e8))
    gtotal_init = gtotal_all.sample(False, args.training_fraction, training_fraction_seed).repartition(20).zipWithIndex().map(lambda ((gidx, d, g, gb, s), zidx): _make_gtotal_plus_noise((gidx, d, g, gb), zidx, noise_spread=1, plus=False)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    gtotal_init.count()
    
    #train factors with parallel SGD 
    print('Train PREDICTD')
    iseed = rs.randint(0, int(1e8))
    gtotal, ct, ct_bias, assay, assay_bias, iter_errs = pl.train_predictd(gtotal_init, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, learning_rate, args.run_bucket, args.out_root, args.iters_per_mse, args.batch_size, args.stop_winsize, args.stop_winspacing, args.stop_win2shift, args.stop_pval, args.lrate_search_num, init_seed=iseed, min_iters=args.min_iters, max_iters=args.max_iters, subsets=subsets, burn_in_epochs=args.burn_in_epochs)

    #train genome factors across whole genome
    print('Apply new cell type parameters across genome.')
    genome_total = pl.train_genome_dimension(gtotal_all, 
                                             ct, ct_bias,
                                             assay, assay_bias, args.ri2, subsets=subsets)

    final_mse = pl.calc_mse(genome_total, ct, rc, ct_bias, rbc, assay, ra,
                            assay_bias, rba, ri, rbi, subsets=subsets)
    iter_errs.append(final_mse)

    #save model to S3
    print('Save newly-trained model to S3.')
#    hyperparameters = {'rc':rc, 'rbc':rbc, 'ra':ra, 'rba':rba, 'ri':ri, 'rbi':rbi, 'ri2':args.ri_2nd_order,
#                       'learning_rate':learning_rate, 'iters_per_mse':args.iters_per_mse,
#                       'batch_size':args.batch_size, 'win_size':args.win_size, 'win_spacing':args.win_spacing,
#                       'win2_shift':args.win2_shift, 'pval':args.pval, 'lrate_search_num':args.lrate_search_num,
#                       'training_fraction':args.training_fraction, 
#                       'training_fraction_seed':args.training_fraction_seed, 'factor_init_seed':args.factor_init_see#d,
#                       'max_iters':args.max_iters, 'min_iters':args.min_iters, 'subsets':None, 'beta1':0.9,
#                       'beta2':0.999, 'epsilon':1e-8, 'run_bucket':args.run_bucket, 'out_root':args.out_root,
#                       'data_url':args.data_url, 'args':args}
    hyperparameters = {'args':args, 'subsets':subsets}
    pl.save_model_to_s3(gmean, ct, ct_bias, assay, assay_bias, genome_total, iter_errs, hyperparameters, 
                        args.run_bucket, args.out_root, iter_errs_header_line='Iter\tObjective\tTrain\tValid\n')

    #remove checkpoint dir
    checkpoint_keys = s3_library.glob_keys(args.run_bucket, os.path.join(args.out_root, 'checkpoints/*'))
    for key in checkpoint_keys:
        if key:
            key.delete()

    #make browser view of H1 cell line tracks
    if args.no_browser_tracks is False:
        print('Generate UCSC Genome Browser tracks.')
        assay_list = [e2[0] for e2 in sorted(set([(e1[1], e1[-1][1]) for e1 in data_idx.values()]), key=lambda x: x[1])]
        ct_list = [e2[0] for e2 in sorted(set([(e1[0], e1[-1][0]) for e1 in data_idx.values()]), key=lambda x: x[1])]
        
        coords_to_output = list(zip(*itertools.product((ct_list.index('H1_Cell_Line'),), numpy.arange(len(assay_list)))))
        pl.write_bigwigs2(genome_total.repartition(120), ct, ct_bias, assay, assay_bias, gmean, ct_list, assay_list, args.run_bucket, args.out_root, sinh=not args.no_bw_sinh, coords=coords_to_output)
    return final_mse[pl.MSE_VALID]


parser = argparse.ArgumentParser()
parser.add_argument('--data_url', help='S3 url to a data RDD.')
parser.add_argument('--addl_data_url', help='User data RDD to be added to the RDD provided in --data_url before training the model.')
parser.add_argument('--run_bucket', help='S3 bucket to which all '
                    'output files associated with this run should be '
                    'written.')
parser.add_argument('-o', '--out_root', help='The root key path that all output files for this PREDICTD run should share.')
parser.add_argument('--fold_idx', type=int, default=-1, help='The index of the test set that should be used for this run.')
parser.add_argument('--valid_fold_idx', type=int, default=-1, help='The index of the validation set that should be used for the given test set.')
parser.add_argument('--factor_init_seed', type=int, default=5, help='The seed that should be used for the random number generator that initializes the factor matrices. [default: %(default)s]')
#parser.add_argument('--data_iteration_seed', type=int, default=1)

parser.add_argument('-f', '--latent_factors', type=int, default=100, help='The number of latent factors to use in the model. This hyperparameter controls the number of parameters in the model and the size of the factor matrices. [default: %(default)s]')
parser.add_argument('-l', '--learning_rate', type=float, default=0.005, help='The learning rate for parallel stochastic gradient descent. This hyperparameter controls how far the model steps down the gradient at each training iteration. [default: %(default)s]')
parser.add_argument('--beta1', type=float, default=0.9, help='The decay rate of the moving average of the first moment of the gradient in the Adam optimizer. [default: %(default)s]')
parser.add_argument('--beta2', type=float, default=0.999, help='The decay rate of the moving average of the second moment of the gradient in the Adam optimizer. [default: %(default)s]')
parser.add_argument('--epsilon', type=float, default=1e-8, help='A small value used for numerical stability in computing the parameter updates in Adam. [default: %(default)s]')
parser.add_argument('-b', '--batch_size', type=int, default=5000, help='The number of stochastic gradient descent iterations that should occur during each parallel SGD iteration before the cell type and assay parameters are averaged for the next parallel iteration. [default: %(default)s]')
parser.add_argument('-w', '--win_per_slice', type=int, default=1000, help='The number of genomic windows that should be batched together for stochastic gradient descent updates during each iteration of parallel SGD. This parameter controls the degree of parallelization across the tensor. [default: %(default)s]')
parser.add_argument('--rc', type=float, default=3.66, help='The L2 regularization coefficient for the cell type factor matrix. [default: %(default)s]')
parser.add_argument('--ra', type=float, default=1.23e-7, help='The L2 regularization coefficient for the assay factor matrix. [default: %(default)s]')
parser.add_argument('--ri', type=float, default=1.23e-5, help='The L2 regularization coefficient for the genome factor matrix during the parallel SGD iterations. [default: %(default)s]')
parser.add_argument('--ri2', type=float, default=2.9, help='The L2 regularization coefficient for the genome factor matrix when the cell type and assay factors are fixed and applied across all genome windows using second order updates. [default: %(default)s]')
parser.add_argument('--rbc', type=float, default=0, help='The L2 regularization coefficient for the cell type bias vector. [default: %(default)s]')
parser.add_argument('--rba', type=float, default=0, help='The L2 regularization coefficient for the assay bias vector. [default: %(default)s]')
parser.add_argument('--rbi', type=float, default=0, help='The L2 regularization coefficient for the genome bias vector. [default: %(default)s]')

parser.add_argument('--stop_winsize', type=int, default=15, help='The number of parallel SGD iterations that should be taken into account when assessing whether the model has converged. A larger window will result in less noise in the convergence status estimate, but might cause the model to train longer than necessary before stopping. [default: %(default)s]')
parser.add_argument('--stop_winspacing', type=int, default=5, help='The number of parallel SGD iterations that should be left between the two windows testing for model convergence. A higher value here provides more spacing between the windows and will require the model to train longer before it detects model convergence. [default: %(default)s]')
parser.add_argument('--stop_win2shift', type=float, default=1e-05, help='The value by which the second window of parallel SGD results should be shifted in order to detect model convergence. This value is added to the second window of parallel SGD results before testing to see if the second window is shifted up relative to the first. [default: %(default)s]')
parser.add_argument('--stop_pval', type=float, default=0.05, help='The p-value of the Wilcoxon rank sum test between window 1 and window 2 at which the model is considered to have converged. [default: %(default)s]')

parser.add_argument('--restart', action='store_true', default=False, help='The training of this model is being restarted from an existing checkpoint.')
parser.add_argument('--checkpoint', help='The S3 URI of the checkpoint the training should be restarted from.')
parser.add_argument('--min_iters', type=int, default=None, help='The minimum number of parallel SGD iterations this model should train for. Model convergence detection will be skipped until after this number of iterations. [default: %(default)s]')
parser.add_argument('--max_iters', type=int, default=None, help='The maximum number of parallel SGD iterations this model should train for. Model training will stop at convergence or if it reaches this number of training iterations, whichever comes first. [default: %(default)s]')
parser.add_argument('--iters_per_mse', type=int, default=3, help='The number of parallel SGD iterations that should occur between iterations of testing the validation mean squared error for convergence detection. Testing the validation MSE too often wastes time because convergence is a fairly gradual process. Setting this greater than zero causes the model to complete that number of parallel SGD iterations in between validation MSE checks. [default %(default)s]')
#parser.add_argument('--mse_sample_size', type=float, default=None)
#parser.add_argument('--trainall', action='store_true', default=False)
#parser.add_argument('--genome_catchup', action='store_true', default=False)
#parser.add_argument('--genome_catchup_threshold', type=int, help='Genome catch up seems '
#                    'to be most helpful in the early part of training, so this parameter '
#                    'provides the number of iterations to allow genome catch up before '
#                    'just running normal parallel SGD.')
parser.add_argument('--lrate_search_num', type=int, default=3, help='When model convergence is detected, if this parameter is greater than zero, training will continue with --learning_rate = 0.5 * --learning_rate and --beta1 = --beta1 - (1.0 - --beta1), and --lrate_search_num will be decremented by 1. [default: %(default)s]')
parser.add_argument('--burn_in_epochs', type=float, default=0.5, help='After the model chooses a random subset of 8000 genomic locations for burning in the model, this parameter controls how many times through the available training data in that subset the model will pass during the burn in stochastic gradient descent updates. [default: %(default)s]')
#parser.add_argument('--weighted_err', action='store_true', default=False)
#parser.add_argument('--pctl_res', help='S3 url to data percentiles that have already been calculated.')
#parser.add_argument('--no_arcsinh_transform', action='store_true', default=False)
#parser.add_argument('--random_loci_fraction', type=float)
#parser.add_argument('--random_loci_fraction_seed', type=int, default=20)
#parser.add_argument('--no_sort_by_genomic_position', action='store_true', default=False)
#parser.add_argument('--final_genome_training', action='store_true', default=False)
#parser.add_argument('--init_genome_w_data', action='store_true', default=False)
parser.add_argument('--folds_fname', help='Basename for the file containing the definitions of the training, test, and validation sets as created by the generate_folds.py script.')
#parser.add_argument('--data_frac', type=float)
#parser.add_argument('--train_on_subset', type=float)
#parser.add_argument('--train_on_subset_seed', type=int, default=33)
#parser.add_argument('--final_genome_training_on_all_loci', action='store_true', default=False)
parser.add_argument('--no_bw_sinh', default=False, action='store_true', help='If generating bigwig files after model training, do not reverse the inverse hyperbolic sine on these files and simply output the raw imputed values.')
parser.add_argument('--training_fraction', type=float, default=0.01, help='The fraction of genomic positions in the RDD tensor to use for model training. Smaller values will result in fewer slices of the tensor to train at each parallel SGD iteration, and will speed up model training as long as the number of slices exceeds the number of available cores. [default: %(default)s]')
#parser.add_argument('--training_fraction_seed', type=int, default=55)
parser.add_argument('--no_browser_tracks', action='store_true', default=False, help='If set, do not output any tracks of imputed data after training. Useful if one is training multiple models with the goal of averaging their results or if one is only interested in whole genome imputed tracks and plans to run the impute_data.py script after model training.')

if __name__ == "__main__":
    args = parser.parse_args()

    sc = SparkContext(appName='train_model',
                      pyFiles=[os.path.join(os.path.dirname(__file__), 'predictd_lib.py'),
                               os.path.join(os.path.dirname(__file__), 's3_library.py')])
    pl.sc = sc
    if args.data_url.startswith('s3'):
        if not s3_library.glob_keys(args.run_bucket, os.path.join(args.out_root, 'command_line.txt')):
            bucket = s3_library.S3.get_bucket(args.run_bucket)
            key = bucket.new_key(os.path.join(args.out_root, 'command_line.txt'))
            key.set_contents_from_string(' '.join(sys.argv) + '\n', headers={'x-amz-request-payer':'requester'})
        elif not args.restart:
            raise Exception('Error: Output directory already exists.')
    train_consolidated(args)
    sc.stop()
