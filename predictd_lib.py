'''Library of functions for running PREDICTD
'''

import copy
import glob
import gzip
import itertools
import numpy
from operator import concat
import os
import pickle
from plumbum import local
from pyspark import StorageLevel, AccumulatorParam
import scipy.sparse as sps
from scipy.stats import ranksums
import shlex
import smart_open
import subprocess
import sys
import threading
import time
import urllib
import zlib

sys.path.append(os.path.dirname(__file__))
import s3_library

#Variable to store Spark Context, needs to be set by the calling script
sc = None
STORAGE = 'S3'

#Indices for canonical subsets in the list passed to calc_mse* (for instance)
SUBSET_TRAIN = 0
SUBSET_VALID = 1
SUBSET_TEST = 2
SUBSET_MAP = {'train':SUBSET_TRAIN,
              'valid':SUBSET_VALID,
              'test':SUBSET_TEST}

MSE_TRAIN = 1
MSE_VALID = 2
MSE_TEST = 3
MSE_MAP = {'train':MSE_TRAIN,
           'valid':MSE_VALID,
           'test':MSE_TEST}

#colors for bigwig tracks
ASSAY_COLORS = {'H3K27me3':'255,0,0',    #red
                'H3K36me3':'0,0,255',    #blue
                'H3K4me3':'0,128,0',     #green
                'H3K4me1':'154,205,50',  #yellowgreen
                'DNase':'255,140,0',     #darkorange
                'H3K27ac':'218,165,32',  #goldenrod
                'H3K4me2':'89,190,121',
                'H3K9me3':'194,102,199',
                'H3K9ac':'77,49,129',
                'H2AZ':'163,165,77',
                'H3K79me2':'84,136,227',
                'H3K79me1':'201,116,39',
                'H4K20me1':'59,167,229',
                'H3K4ac':'181,61,49',
                'H4K8ac':'67,200,172',
                'H3K23ac':'138,46,124',
                'H3K18ac':'197,153,72',
                'H2BK5ac':'106,118,192',
                'H2BK120ac':'139,83,33',
                'H2AK5ac':'187,134,214',
                'H3K14ac':'225,131,94',
                'H2BK12ac':'219,119,185',
                'H4K91ac':'186,71,88',
                'H2BK15ac':'167,69,112'}



class UpdateAccumulatorParam(AccumulatorParam):
#class UpdateAccumulatorParam(object):
    '''Subclass to sum numpy.ma.MaskedArrays and keep track of how many are being
    summed so that we can get the average.
    '''
    def zero(self, initialValue):
        if isinstance(initialValue[0], numpy.ndarray):
            return [numpy.zeros(initialValue[0].shape), 0]
        else:
            return [0,0]

    def addInPlace(self, uaparam, marray):
        uaparam[0] += marray[0]
        uaparam[1] += marray[1]
        return uaparam

class NumpyArrayAccumulatorParam(AccumulatorParam):
    '''Subclass to accumulate pairs of Numpy arrays by summing the corresponding
    arrays.
    '''
    def zero(self, initialValue):
        if isinstance(initialValue, numpy.ndarray):
            return [numpy.zeros(initialValue[0].shape),
                    numpy.zeros(initialValue[1].shape)]
        else:
            return [0,0]

    def addInPlace(self, v1, v2):
        v1[0] += v2[0]
        v1[1] += v2[1]
        return v1

def _custom_load(part_url, num_tries=10):
    while True:
        try:
            with smart_open.smart_open(part_url, 'rb') as pickle_in:
                return pickle.loads(pickle_in.read())
        except s3_library.boto.exception.NoAuthHandlerFound:
            if not num_tries:
                raise
            num_tries -= 1
            time.sleep(10)
        except TypeError:
            if not num_tries:
                raise
            num_tries -= 1
            time.sleep(10)
        else:
            break

def load_custom_save(rdd_url):
    with smart_open.smart_open(os.path.join(rdd_url, 'num_parts.pickle'), 'rb') as pickle_in:
        num_parts = pickle.loads(pickle_in.read())
    part_urls = [os.path.join(rdd_url, 'part-{:05d}.pickle'.format(part_idx)) for part_idx in range(num_parts)]
    url_rdd = sc.parallelize(part_urls, numSlices=num_parts)
    return url_rdd.flatMap(_custom_load).repartition(num_parts)

def load_saved_rdd(rdd_url):
    rdd_bucket, rdd_key = s3_library.parse_s3_url(rdd_url)
    if s3_library.S3.get_bucket(rdd_bucket).get_key(os.path.join(rdd_key, 'num_parts.pickle')):
        return load_custom_save(rdd_url)
    else:
        return sc.pickleFile(rdd_url.replace('s3://', 's3n://'))

def _custom_save(pidx, pdata, out_url):
    '''Takes a partition index, partition data in list form, and a pickle url.
    In the end, each partition saves its own data to S3.
    '''
    part_url = os.path.join(out_url, 'part-{0:05d}.pickle'.format(pidx))
    with smart_open.smart_open(part_url, 'wb') as out:
        out.write(pickle.dumps(list(pdata)))
    yield part_url

def save_rdd_as_pickle(rdd, out_url, num_partitions=None, custom_save=True):
    '''This function attempts to avoid program crashes due to random server failed to respond errors.
    '''
    if custom_save is True:
        if num_partitions is not None:
            urls = rdd.repartition(num_partitions).mapPartitionsWithIndex(lambda x,y: _custom_save(x, y, out_url))
        else:
            urls = rdd.mapPartitionsWithIndex(lambda x,y: _custom_save(x, y, out_url))
        urls.count()
        with smart_open.smart_open(os.path.join(out_url, 'num_parts.pickle'), 'wb') as out:
            out.write(pickle.dumps(urls.getNumPartitions()))
        return

    max_attempts = 4
    attempts = 1
    while True:
        try:
            if num_partitions:
                rdd.saveAsPickleFile(out_url.replace('s3://', 's3n://'), num_partitions)
            else:
                rdd.saveAsPickleFile(out_url.replace('s3://', 's3n://'))
        except Exception as err:
            print('Error saving rdd {!s}. Will try {!s} more times.'.format(out_url, max_attempts - attempts))
            time.sleep(60 * attempts)
            if STORAGE == 'S3':
                bucket_txt, key_txt = parse_s3_url(out_url)
                while s3_library.glob_keys(bucket_txt, key_txt+'*'):
                    print('Found files that should be deleted before next attempt. Deleting.')
                    cmd = ['aws', 's3', 'rm', '--quiet', '--recursive', out_url.replace('s3n://', 's3://')]
                    subprocess.check_call(cmd)
                    time.sleep(60)
                    for key in s3_library.glob_keys(bucket_txt, key_txt+'*'):
                        key.delete()
                    time.sleep(30)
            elif STORAGE == 'BLOB':
                container_txt, blob_txt = parse_azure_url(out_url)[:2]
                blob_list = azure_library.glob_blobs(container_txt, blob_txt+'*')
                while blob_list:
                    print('Found files that should be deleted before next attempt. Deleting.')
                    azure_library.delete_blob_dir(container_txt, blob_txt)
                    time.sleep(60)
                    blob_list = azure_library.glob_blobs(container_txt, blob_txt+'*')
                    time.sleep(30)
            attempts += 1
        else:
            break

def save_model_to_s3(gmean, ct, ct_bias, assay, assay_bias, genome_total, iter_errs, hyperparameters, 
                     out_bucket, out_root, iter_errs_header_line='Iters\tObjective\tTrain\tValid\tTest\n'):
    s3_library.set_pickle_s3(out_bucket, os.path.join(out_root, 'gmean.pickle'), gmean)
    s3_library.set_pickle_s3(out_bucket, os.path.join(out_root, 'ct_factors.pickle'), (ct, ct_bias))
    s3_library.set_pickle_s3(out_bucket, os.path.join(out_root, 'assay_factors.pickle'), (assay, assay_bias))
    print_iter_errs(iter_errs, out_bucket, os.path.join(out_root, 'iter_errs.txt'), 
                    header_line=iter_errs_header_line)
    genome_url = 's3://{!s}/{!s}'.format(out_bucket, os.path.join(out_root, 'genome_factors.rdd.pickle'))
    save_rdd_as_pickle(genome_total.map(lambda x: (x[0], (x[2], x[3]))), genome_url)
    s3_library.set_pickle_s3(out_bucket, os.path.join(out_root, 'hyperparameters.pickle'), hyperparameters)

def load_subsets(pickle_url, fold_idx=-1, valid_fold_idx=-1, trainall=False, folds_fname=None):
    #get subsets
    data_bucket_txt, data_key_txt = s3_library.parse_s3_url(pickle_url)
    data_dir_key = os.path.dirname(data_key_txt)
    if fold_idx < 0:
        subsets_dict = s3_library.get_pickle_s3(data_bucket_txt, os.path.join(data_dir_key, 'subsets.pickle'))
        subsets = [subsets_dict['train'][:,:,0], subsets_dict['valid'][:,:,0], subsets_dict['test'][:,:,0]]
#        for subset in subsets.keys():
#            subsets[subset] = sc.broadcast(subsets[subset][:,:,0])
    elif valid_fold_idx < 0:
        if folds_fname is None:
            subsets_path = os.path.join(data_dir_key, 'folds.5.pickle')
        else:
            subsets_path = os.path.join(data_dir_key, folds_fname)
        print('Fold path: {!s}'.format(os.path.join(data_bucket_txt, subsets_path)))
        subsets_dict = s3_library.get_pickle_s3(data_bucket_txt, subsets_path)[fold_idx]
        subsets = [subsets_dict['train'], subsets_dict['valid'], subsets_dict['test']]
        subsets += [subsets[k] for k in subsets_dict if k not in ['train', 'valid', 'test']]
#        for subset in subsets.keys():
#            if (trainall is True) and subset == 'train':
#                subsets['train'] = sc.broadcast(numpy.zeros(subsets['train'].shape, dtype=bool))
#            else:
#                subsets[subset] = sc.broadcast(subsets[subset])
    else:
        if folds_fname is None:
            fold_path = os.path.join(data_dir_key, 'folds.5.8.pickle')
        else:
            fold_path = os.path.join(data_dir_key, folds_fname)
        print('Fold path: {!s}'.format(os.path.join(data_bucket_txt, fold_path)))
        fold = s3_library.get_pickle_s3(data_bucket_txt, fold_path)[fold_idx]
#        subsets = {}
#        subsets['test'] = sc.broadcast(fold['test'])
#        subsets['train'] = sc.broadcast(fold['train'][valid_fold_idx]['train'])
#        subsets['valid'] = sc.broadcast(fold['train'][valid_fold_idx]['valid'])
        subsets = [fold['train'][valid_fold_idx]['train'],
                   fold['train'][valid_fold_idx]['valid'],
                   fold['test']]
    if trainall is True:
        return [numpy.zeros(subsets[0].shape, dtype=bool)]
    else:
        return subsets

def _csr_remove_nan(csr):
    csr.data = numpy.nan_to_num(csr.data)
    csr.eliminate_zeros()
    return csr

def load_data(pickle_url, win_per_slice=None, fold_idx=-1, valid_fold_idx=-1, trainall=False, no_arcsinh_transform=False, random_fraction=None, random_fraction_seed=20, sort_by_genomic_position=False, folds_fname=None):
    '''Read in data from pickled RDD files prepared using download_and_transform_bigwigs.py,
    compile_transformed_data.py, and get_database_subset.py
    '''
    #get data RDD
    subsets = None
    if pickle_url.startswith('s3'):
        data = load_saved_rdd(pickle_url.replace('s3://', 's3n://'))
        if folds_fname is not None:
            if folds_fname.startswith('s3'):
                subsets = load_subsets(folds_fname, fold_idx=fold_idx, valid_fold_idx=valid_fold_idx, trainall=trainall,
                                       folds_fname=os.path.basename(folds_fname))
            else:
                subsets = load_subsets(pickle_url, fold_idx=fold_idx, valid_fold_idx=valid_fold_idx, trainall=trainall,
                                       folds_fname=folds_fname)
    elif pickle_url.startswith('wasb'):
        data = load_saved_rdd(pickle_url)
        if folds_fname is not None:
            subsets = load_subsets_azure(pickle_url, fold_idx=fold_idx, valid_fold_idx=valid_fold_idx, 
                                         trainall=trainall, folds_fname=folds_fname)
    else:
        raise Exception('Unknown Cloud URL: {!s}'.format(pickle_url))
    #apply optional transformations
    if random_fraction is not None:
        data = data.sample(False, random_fraction, random_fraction_seed)
    if no_arcsinh_transform is True:
        data = data.map(lambda (idx,d): (idx, d.sinh()))
    #persist and load the data
    data = data.mapValues(_csr_remove_nan).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    num_records = data.count()
    #auto-calculate win_per_slice if it is not provided
    if win_per_slice is None:
        win_per_slice = 1e6/len(data.first()[1].nonzero()[0])
    num_partitions = int(numpy.ceil(num_records/float(win_per_slice)))
    if sort_by_genomic_position is True:
        data_sorted = data.sortByKey(numPartitions=num_partitions).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    else:
        data_sorted = data.repartition(num_partitions).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    data_sorted.count()
    data.unpersist()
    del(data)
    
    if subsets is not None:
        return data_sorted, subsets
    else:
        return data_sorted

def load_model(model_url, load_data_too=False):
    bucket_txt, key_txt = s3_library.parse_s3_url(model_url)
    print(bucket_txt, key_txt)
    if not s3_library.S3.get_bucket(bucket_txt).get_key(os.path.join(key_txt, 'ct_factors.pickle')):
        return load_model_old(model_url, load_data_too=load_data_too)
    else:
        gmean = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_txt, 'gmean.pickle'))
        ct, ct_bias = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_txt, 'ct_factors.pickle'))
        assay, assay_bias = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_txt, 'assay_factors.pickle'))
        genome_params = load_saved_rdd(os.path.join(model_url, 'genome_factors.rdd.pickle')).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
        hyperparams = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_txt, 'hyperparameters.pickle'))
        data = [load_saved_rdd(hyperparams['args'].data_url).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)] if load_data_too is True else None
        #recursively load hyperparameters (and data, if requested) of the model this one is based on
        if 'model_url' in hyperparams['args']:
            model_hyps = load_model(hyperparams['args'].model_url, load_data_too=load_data_too)
            model_data_url = model_hyps[6]['args'].data_url
            if 'model_url' in model_hyps[6]['args']:
                model_model_url = model_hyps[6]['args'].model_url
            model_hyps[6].update(hyperparams)
            if model_data_url and isinstance(model_data_url, str):
                model_hyps[6]['args'].data_url = [model_data_url, hyperparams['data_url']]
            else:
                assert isinstance(model_data_url, list), "model_data_url must be list"
                model_hyps[6]['args'].data_url = model_data_url.append(hyperparams['args'].data_url)
            if model_model_url and isinstance(model_model_url, str):
                model_hyps[6]['args'].data_url = [model_model_url, hyperparams['args'].model_url]
            else:
                assert isinstance(model_model_url, list), "model_model_url must be list"
                model_hyps[6]['args'].model_url = model_model_url.append(hyperparams['args'].model_url)
            if load_data_too is True:
                data = model_hyps[7] + data
            hyperparams = model_hyps[6]
        return (gmean, ct, ct_bias, assay, assay_bias, genome_params, hyperparams, data)

def load_model_old(model_url, load_data_too=False):
    '''Takes the url to an S3 output directory created by training a PREDICTD model.
    Returns the parameters of the trained model: cell type, assay, genome,
    cell type bias, assay bias, and genome bias.
    '''
    bucket_txt, key_txt = s3_library.parse_s3_url(model_url)
    #load model parameters
    gmean = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_txt, 'gmean.pickle'))
    ct, ct_bias = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_txt, '3D_model_ct_params.pickle'))
    assay, assay_bias = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_txt, '3D_model_assay_params.pickle'))
    genome_params = load_saved_rdd(os.path.join(model_url, '3D_model_genome_params.rdd.pickle')).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    #load relevant command line parameters
    cmd_key = s3_library.S3.get_bucket(bucket_txt).get_key(os.path.join(key_txt, 'command_line.txt'))
    if not cmd_key:
        cmd_key = s3_library.S3.get_bucket(bucket_txt).get_key(os.path.join(os.path.dirname(key_txt), 'command_line.txt'))
    cmd_line = dict([elt.strip().split('=') if '=' in elt else [elt.strip, True] for elt in shlex.split(cmd_key.get_contents_as_string(headers={'x-amz-request-payer':'requester'}))])
    rc = float(cmd_line.get('--rc', 2.88e-4))
    ra = float(cmd_line.get('--ra', 5.145e-11))
    ri = float(cmd_line.get('--ri', 9.358e-14))
    rbc = float(cmd_line.get('--rbc', 0))
    rba = float(cmd_line.get('--rba', 0))
    rbi = float(cmd_line.get('--rbi', 0))
    learning_rate = float(cmd_line.get('--learning_rate', 0.005))

    if load_data_too is True:
        assert cmd_line.get('--data_url'), "No data_url to get from this model."
        data, subsets = load_data(cmd_line.get('--data_url'), win_per_slice=cmd_line.get('--win_per_slice'), fold_idx=int(cmd_line.get('--fold_idx')) if cmd_line.get('--fold_idx') else -1, valid_fold_idx=int(cmd_line.get('--valid_fold_idx')) if cmd_line.get('--valid_fold_idx') else -1, trainall=False, no_arcsinh_transform=False, random_fraction=None, random_fraction_seed=20, sort_by_genomic_position=False, folds_fname=cmd_line.get('--folds_fname'))
    else:
        data = None

    return gmean, ct, ct_bias, assay, assay_bias, genome_params, rc, ra, ri, rbc, rba, rbi, learning_rate, data, subsets

def vstack_csr_matrices(matrix1, matrix2):
    '''Thanks - http://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
    '''
    new_data = numpy.concatenate((matrix1.data, matrix2.data))
    new_indices = numpy.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = numpy.concatenate((matrix1.indptr, new_ind_ptr))

    return sps.csr_matrix((new_data, new_indices, new_ind_ptr))

def _update_bias_accum(data, subset, axis, accum):
    data_coords = numpy.zeros(data.shape, dtype=bool)
    data_coords[data.nonzero()] = True
    subset_coords = numpy.logical_and(~subset, data_coords)
    accum.add(numpy.array([numpy.sum(numpy.ma.masked_array(data.toarray(), mask=~subset_coords), axis=axis),
                           numpy.sum(subset_coords, axis=axis)]))

def _calc_genome_bias(data, subset):
    data_coords = numpy.zeros(data.shape, dtype=bool)
    data_coords[data.nonzero()] = True
    subset_coords = numpy.logical_and(~subset, data_coords)
    return numpy.mean(data[subset_coords])

def calc_dim_bias(pdata, subset):
    '''Iterate over the pdata RDD and calculate the mean along each dimension of
    the 3D tensor for the coordinates of subset (usually subset contains the
    coordinates of the training data).
    '''
    ct_num, assay_num = pdata.first()[1].shape
    ct_accum = sc.accumulator(numpy.zeros((2, ct_num)), NumpyArrayAccumulatorParam())
    pdata.foreach(lambda (x,y): _update_bias_accum(y, subset, 1, ct_accum))
    ct_bias = numpy.divide(*ct_accum.value)
    ct_resid = pdata.mapValues(lambda x: sps.csr_matrix((numpy.array((x.T - ct_bias).T)[x.nonzero()], x.nonzero()), shape=(ct_num, assay_num)))

    assay_accum = sc.accumulator(numpy.zeros((2, assay_num)),
                                 NumpyArrayAccumulatorParam())
    ct_resid.foreach(lambda (x,y): _update_bias_accum(y, subset, 0, assay_accum))
    assay_bias = numpy.divide(*assay_accum.value)
    assay_resid = ct_resid.mapValues(lambda x: sps.csr_matrix((numpy.array(x - assay_bias)[x.nonzero()], x.nonzero()), shape=(ct_num, assay_num)))
    genome_bias = assay_resid.mapValues(lambda x: _calc_genome_bias(x, subset))
    genome_bias.persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER).count()
    return genome_bias, ct_bias, assay_bias

def init_random_with_seed(rseed, num_factors, uniform_bounds):
    '''Generate a random numpy array using a new RandomState class. This way the
    each seed will act on a number generator with independent state.
    '''
    rs = numpy.random.RandomState(rseed)
    return rs.uniform(*uniform_bounds, size=(num_factors,))

def init_factors(pdata, train_subset, num_factors, init_seed=5, scale=0.5, uniform_bounds=(-0.33, 0.33)):
    '''Initialize the factors of the model: The cell type, assay, and genome
    latent factor matrices, and the cell type, assay, and genome bias vectors.
    '''
    genome_bias, ct_bias, assay_bias = calc_dim_bias(pdata, train_subset)
    numpy.random.seed(init_seed)
    ct = numpy.random.uniform(*uniform_bounds, size=(len(ct_bias), num_factors))
    assay = numpy.random.uniform(*uniform_bounds, size=(len(assay_bias), num_factors))
    genome = pdata.map(lambda (y,x): (y, init_random_with_seed(y[1] + 1 * (init_seed * 2), num_factors, uniform_bounds)), preservesPartitioning=True)
    genome.persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER).count()
    return ct, ct_bias, assay, assay_bias, genome, genome_bias

def _calc_gmean_helper(data, subset=None):
    '''This function is used within a RDD.map() call to calculate the global
    mean of the data corresponding to subset.
    '''
    data_coords = numpy.zeros(data.shape, dtype=bool)
    data_coords[data.nonzero()] = True
    if subset is not None:
        subset_coords = numpy.logical_and(~subset, data_coords)
    else:
        subset_coords = data_coords
    datasum, datacount = numpy.sum(data[subset_coords]), numpy.sum(subset_coords)
    return datasum, datacount

def calc_gmean(pdata, subset=None):
    '''Run a map/reduce over the pdata RDD to calculate the global mean of the
    data points corresponding to subset.
    '''
    res = (pdata.map(lambda (x,y): _calc_gmean_helper(y, subset)).
           reduce(lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2)))
    return numpy.divide(res[0], res[1])

def print_iter_errs(iter_errs, out_bucket, out_key, 
                    header_line='Iters\tObjective\tTrain\tValid\tTest\n'):
    iter_errs_str = header_line
    for idx, elt in enumerate(iter_errs):
        iter_errs_str += ('\t'.join([str(idx)] + [str(elt[i]) for i in range(len(elt))]) + '\n')
    bucket = s3_library.S3.get_bucket(out_bucket)
    key = bucket.new_key(out_key)
    key.set_contents_from_string(iter_errs_str, headers={'x-amz-request-payer':'requester'})

def _calc_mse_helper(gtotal_elt, ct_assay, ct_assay_bias, ri, rbi, subsets=None):
    gidx, data, genome, genome_bias = gtotal_elt[:4]
    ret_vals = []
    #calculate MSE over all data points
    total_pred = numpy.dot(ct_assay.T, genome) + ct_assay_bias + genome_bias
    data = numpy.nan_to_num(data.toarray())
    data_locs = numpy.zeros(data.shape, dtype=bool)
    data_locs[data.nonzero()] = True
    if subsets is None and gtotal_elt[-1] is None:
        subsets = [~data_locs]
    elif subsets is None:
        subsets = gtotal_elt[-1]
    elif gtotal_elt[-1] is not None:
        subsets = [~numpy.logical_and(~e1, ~e2) for e1, e2 in zip(subsets, gtotal_elt[-1])]
    for idx, subset in enumerate(subsets):
        to_test = numpy.logical_and(data_locs, ~subset)
        try:
            total_err = (data.flatten()[to_test.flatten()] - total_pred.flatten()[to_test.flatten()]) ** 2
        except IndexError:
            raise Exception(data.shape, total_pred.shape, subset.shape)
        ret_vals.append((numpy.sum(total_err), numpy.sum(~subset)))
        if idx == 0:
            #calculate L2-norm of the genome factors for use in calculating the objective value
            ret_vals.append((ri * numpy.sum(genome ** 2), 1))
            ret_vals.append((rbi * (genome_bias ** 2), 1))
    return numpy.array(ret_vals)

def calc_mse_gtotal_split(gtotal_train, gtotal_valid, ct, rc, ct_bias, rbc, assay, ra, 
                          assay_bias, rba, ri, rbi, subsets=None):
    ct_assay = numpy.vstack([numpy.outer(ct[:,idx], assay[:,idx]).flatten() for idx in xrange(ct.shape[1])])
    ct_assay_bias = numpy.add.outer(ct_bias, assay_bias).flatten()
    #training MSE
    accum = gtotal_train.map(lambda x: _calc_mse_helper(x, ct_assay, ct_assay_bias, ri, rbi, subsets=subsets))
    accum_res = accum.reduce(lambda x,y: x + y)
    objective = (accum_res[0,0] +
                 (rc * numpy.sum(ct ** 2)) +
                 (rbc * numpy.sum(ct_bias ** 2)) +
                 (ra * numpy.sum(assay ** 2)) +
                 (rba * numpy.sum(assay_bias ** 2)) +
                 accum_res[1,0] + accum_res[2,0])
    train_mse = numpy.divide(accum_res[0,0], accum_res[0,1])
    mse = [objective, train_mse]

    #validation MSE
    accum = gtotal_valid.map(lambda x: _calc_mse_helper(x, ct_assay, ct_assay_bias, ri, rbi, subsets=subsets))
    accum_res = accum.reduce(lambda x,y: x + y)
    mse += [numpy.divide(accum_res[0,0], accum_res[0,1]) ] + list(numpy.divide(accum_res[3:,0], accum_res[3:,1]).flatten())
    #this is a list with first element objective value, second training mse, and the rest
    #are mse values for the other subsets (if provided)
    return mse

def calc_mse(gtotal, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, subsets=None):
    ct_assay = numpy.vstack([numpy.outer(ct[:,idx], assay[:,idx]).flatten() for idx in xrange(ct.shape[1])])
    ct_assay_bias = numpy.add.outer(ct_bias, assay_bias).flatten()
    #training MSE
    accum = gtotal.map(lambda x: _calc_mse_helper(x, ct_assay, ct_assay_bias, ri, rbi, subsets=subsets))
    accum_res = accum.reduce(lambda x,y: x + y)
    objective = (accum_res[0,0] +
                 (rc * numpy.sum(ct ** 2)) +
                 (rbc * numpy.sum(ct_bias ** 2)) +
                 (ra * numpy.sum(assay ** 2)) +
                 (rba * numpy.sum(assay_bias ** 2)) +
                 accum_res[1,0] + accum_res[2,0])
    mse = [objective, numpy.divide(accum_res[0,0], accum_res[0,1])]
    mse += list(numpy.divide(accum_res[3:,0], accum_res[3:,1]))
    #this is a list with first element objective value, second training mse, and the rest
    #are mse values for the other subsets (if provided)
    return mse

def compute_imputed(gtotal, ct, ct_bias, assay, assay_bias, gmean, coords=None):
    gidx, data, genome, genome_bias = gtotal[:4]
    if coords:
        return (gidx, (numpy.sum([numpy.outer(ct[:, idx], assay[:, idx]) * genome[idx] for idx in range(len(genome))], axis=0) + genome_bias + ct_bias[:,None] + assay_bias)[coords])
    else:
        return (gidx, numpy.sum([numpy.outer(ct[:, idx], assay[:, idx]) * genome[idx] for idx in range(len(genome))], axis=0) + genome_bias + ct_bias[:,None] + assay_bias)

def _compute_imputed2_helper(x, ct_assay, ct_assay_bias, gmean, coords=None):
    gidx, data, genome, genome_bias = x[:4]
    imp_vals = (numpy.dot(ct_assay.T, genome[:,None]).flatten() + ct_assay_bias + genome_bias + gmean).reshape(data.shape)
    if coords is None:
        return gidx, imp_vals
    else:
        imp_sparse = sps.csr_matrix((imp_vals[coords], coords), shape=imp_vals.shape)
        return gidx, imp_sparse

def compute_imputed2(gtotal_rdd, ct, ct_bias, assay, assay_bias, gmean, coords=None):
    if coords is None:
        coords = itertools.product(numpy.arange(ct.shape[0]), numpy.arange(assay.shape[0]))
    elif coords == 'data':
        sample_frac = 50.0/gtotal_rdd.count()
        coords = numpy.where(numpy.any(numpy.array([elt[1].toarray() for elt in gtotal_rdd.sample(False, sample_frac).collect()], dtype=bool), axis=0))

#    try:
#        ct_coords = sorted(set(coords[0]))
#    except TypeError:
#        ct_coords = [coords[0]]
#    try:
#        assay_coords = sorted(set(coords[1]))
#    except TypeError:
#        assay_coords = [coords[1]]
#    coords_remap = (numpy.array([ct_coords.index(elt) for elt in (coords[0] if not isinstance(coords[0], int) else [coords[0]])]), 
#                    numpy.array([assay_coords.index(elt) for elt in (coords[1] if not isinstance(coords[1], int) else [coords[1]])]))
    ct_assay = numpy.vstack([numpy.outer(ct[:,idx], assay[:,idx]).flatten() for idx in xrange(ct.shape[1])])
    ct_assay_bias = numpy.add.outer(ct_bias, assay_bias).flatten()
#    raise Exception(ct_assay, ct_assay_bias)
    imputed = gtotal_rdd.map(lambda x: _compute_imputed2_helper(x, ct_assay, ct_assay_bias, gmean, coords))
    return imputed

def _calc_beta1(beta1, num_t, beta_decay=(1 - 1e-6)):
    return beta1 * (beta_decay ** (num_t - 1))

def _calc_lrate(lrate, num_t):
    return lrate * ((1 - 1e-6) ** (num_t - 1))

def one_iteration_ct_only(pidx, pdata, subset, ct, ct_rcoef, ct_bias, ct_bias_rcoef, assay, assay_bias, lrate, batch_size, rseed, ct_accum, ct_bias_accum, ct_m1, ct_m2, ct_t, ct_m1_accum, ct_m2_accum, ct_t_accum, ct_bias_m1, ct_bias_m2, ct_bias_m1_accum, ct_bias_m2_accum, beta1=0.9, beta2=0.999, epsilon=1e-8):
    '''Complete an iteration (or a batch of iterations). Compute updates and
    apply them to the cell type factors.
    '''
    #randomly sample pdata slices (genomic coordinates) to use for stochastic
    #updates
    rs = numpy.random.RandomState(rseed + pidx)
    partition_data = list(pdata)
    partition_len = numpy.arange(len(partition_data))
    updates = {}
    objective_vals = []
    g_idx, data_elt, genome, genome_bias, all_nan, data_coords = [None]*6
    data_points_count = 0
    accum_weight = 0
    ct_t_updates = numpy.zeros(ct_t.shape)
    ct_grad_correction_array = numpy.array([numpy.prod(_calc_beta1(beta1, numpy.arange(1, (t_val + 1 if t_val > 0 else 2)))) for t_val in ct_t])
    while data_points_count < batch_size:
        sample_idx = rs.choice(partition_len)
        if len(partition_data[sample_idx][:4]) == 4:
            g_idx, data_elt, genome, genome_bias = partition_data[sample_idx][:4]
        else:
            raise Exception('Length of partition_data elt not 4: {!s}'.format(partition_data[sample_idx][:4]))
        #randomly pick the data point that we will update on this iteration
        data_coords = list(zip(*numpy.where(~subset))) #take data_coords from training set
        c_coord, a_coord = data_coords[rs.randint(len(data_coords))]
        if data_elt[c_coord, a_coord] == 0 or numpy.isnan(data_elt[c_coord, a_coord]): #in the sparse matrix context, zero == missing
            continue
        ct_t_updates[c_coord] += 1
        ct_t_elt = ct_t[c_coord] + ct_t_updates[c_coord]

        beta1_ct =  _calc_beta1(beta1, ct_t_elt)
        beta1_ct_next = _calc_beta1(beta1, ct_t_elt + 1)

        if data_points_count:
            ct_grad_correction_array[c_coord] *= beta1_ct
        ct_grad_correction = ct_grad_correction_array[c_coord]
        ct_m1_correction = ct_grad_correction * beta1_ct_next

        working_ct_lrate = _calc_lrate(lrate, ct_t_elt)

        #FIRST ORDER UPDATES
        #calculate the predicted value at this data point based on the factors
        #and bias vectors
        pred = (numpy.dot(assay[a_coord], (ct[c_coord] * genome)) +
                assay_bias[a_coord] + ct_bias[c_coord] + genome_bias)
        #calculate the error between the predicted and observed data point and
        #compute the updates to each factor/bias vector
        error = pred - data_elt[c_coord, a_coord]

        #update ct
        axi = assay[a_coord] * genome
        grad = error * axi + (ct_rcoef * ct[c_coord])
        grad_hat = grad/(1 - ct_grad_correction)
        ct_m1[c_coord,:] = beta1_ct * ct_m1[c_coord,:] + (1 - beta1_ct) * grad
        corrected_ct_m1 = ct_m1[c_coord,:]/(1 - ct_m1_correction)
        ct_m2[c_coord,:] = beta2 * ct_m2[c_coord,:] + (1 - beta2) * (grad ** 2)
        corrected_ct_m2 = ct_m2[c_coord,:]/(1 - (beta2 ** ct_t_elt))
        ct_m1_bar = ((1 - beta1_ct) * grad_hat) + (beta1_ct_next * corrected_ct_m1)
        updates['ct'] = working_ct_lrate * (ct_m1_bar/(numpy.sqrt(corrected_ct_m2) + epsilon))
        if numpy.sum(numpy.isnan(updates['ct'])):
            if STORAGE == 'S3':
                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct, genome, error, assay, ct_rcoef, ct_m1, corrected_ct_m1, ct_m2, corrected_ct_m2, updates['ct']))
            if STORAGE == 'BLOB':
                azure_library.load_blob_pickle('encodeimputation', 'ADAM/ct_params.pickle', (ct, genome, error, assay, ct_rcoef, ct_m1, corrected_ct_m1, ct_m2, corrected_ct_m2, updates['ct']))
            raise Exception('NaN at ct update on iteration {!s}'.format(data_points_count))

        #update ct_bias
        grad = error + (ct_bias_rcoef * ct_bias[c_coord])
        grad_hat = grad/(1 - ct_grad_correction)
        ct_bias_m1[c_coord] = beta1_ct * ct_bias_m1[c_coord] + (1 - beta1_ct) * grad
        corrected_ct_bias_m1 = ct_bias_m1[c_coord]/(1 - ct_m1_correction)
        ct_bias_m2[c_coord] = beta2 * ct_bias_m2[c_coord] + (1 - beta2) * (grad ** 2)
        corrected_ct_bias_m2 = ct_bias_m2[c_coord]/(1 - (beta2 ** ct_t_elt))
        ct_bias_m1_bar = ((1 - beta1_ct) * grad_hat) + (beta1_ct_next * corrected_ct_bias_m1)
        updates['ct_bias'] = working_ct_lrate * (ct_bias_m1_bar/(numpy.sqrt(corrected_ct_bias_m2) + epsilon))
        if numpy.sum(numpy.isnan(updates['ct_bias'])):
            if STORAGE == 'S3':
                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct_bias, genome, error, assay, ct_bias_rcoef, ct_bias_m1, corrected_ct_bias_m1, ct_bias_m2, corrected_ct_bias_m2, updates['ct_bias']))
            if STORAGE == 'BLOB':
                azure_library.load_blob_pickle('encodeimputation', 'ADAM/params.pickle', (ct_bias, genome, error, assay, ct_bias_rcoef, ct_bias_m1, corrected_ct_bias_m1, ct_bias_m2, corrected_ct_bias_m2, updates['ct_bias']))
            raise Exception('NaN at ct_bias update on iteration {!s}'.format(data_points_count))

        ct[c_coord,:] -= updates['ct']
        ct_bias[c_coord] -= updates['ct_bias']

        data_points_count += 1
        accum_weight += 1

    #weight these updates based on the weights of the data points they incorporate
    accum_weight = int(round(accum_weight/data_points_count))
    ct_accum.add([ct * accum_weight, accum_weight])
    ct_m1_accum.add([ct_m1 * accum_weight, accum_weight])
    ct_m2_accum.add([ct_m2 * accum_weight, accum_weight])
    ct_t_accum.add([ct_t_updates * accum_weight, accum_weight])
    ct_bias_accum.add([ct_bias * accum_weight, accum_weight])
    ct_bias_m1_accum.add([ct_bias_m1 * accum_weight, accum_weight])
    ct_bias_m2_accum.add([ct_bias_m2 * accum_weight, accum_weight])

    for elt in partition_data:
        yield elt

def train_ct_dim_sgd(gtotal_train, gtotal_valid, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, 
                     learning_rate, out_bucket, out_root, iters_per_mse, batch_size, win_size, win_spacing, 
                     win2_shift, pval, lrate_search_num, init_seed=24, max_iters=None, min_iters=None, subsets=None,
                     beta1=0.9, beta2=0.999, epsilon=1e-8):
    ct_m1 = numpy.zeros(ct.shape)
    ct_m2 = numpy.zeros(ct.shape)
    ct_t = numpy.zeros(ct_bias.shape)
    ct_bias_m1 = numpy.zeros(ct_bias.shape)
    ct_bias_m2 = numpy.zeros(ct_bias.shape)
    iter_errs = [calc_mse_gtotal_split(gtotal_train, gtotal_valid, ct, rc, ct_bias, rbc, 
                                       assay, ra, assay_bias, rba, ri, rbi)]
    min_factors = {'ct': ct.copy(), 'ct_bias': ct_bias.copy(), 'min_idx':0,
                   'ct_m1':ct_m1.copy(), 'ct_m2':ct_m2.copy(), 'ct_bias_m1':ct_bias_m1.copy(),
                   'ct_bias_m2':ct_bias_m2.copy()}
    min_err = iter_errs[0]
    all_iters_count = 0

    #first subset is the training set, so make this set if no subsets list is provided
    if subsets is None:
        data_coords = numpy.sum([elt[1].toarray() for elt in gtotal_train.take(50)], axis=0).nonzero()
        subset = numpy.ones(gtotal_train.first()[1].shape, dtype=bool)
        subset[data_coords] = False
        subsets = [subset]

    #BURN IN
    param = UpdateAccumulatorParam()
    ct_accum = sc.accumulator(param.zero([ct, 1]), UpdateAccumulatorParam())
    ct_bias_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
    ct_m1_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
    ct_m2_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
    ct_t_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
    ct_bias_m1_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
    ct_bias_m2_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())

    sample_frac = min(8000.0/gtotal_train.count(), 1.0)
    burn_in = gtotal_train.sample(False, sample_frac, init_seed).repartition(1).persist()
    burn_in_count = burn_in.count()
    burn_in_batch_size = 1.05 * burn_in_count * len(burn_in.first()[1].data)
    print('Burning in {!s} data points on {!s} loci.'.format(burn_in_batch_size, burn_in_count))
    mapped_burn_in = burn_in.mapPartitionsWithIndex(lambda x,y: one_iteration_ct_only(x, y, subsets[0], ct, rc, ct_bias, rbc, assay, assay_bias, learning_rate, burn_in_batch_size, all_iters_count, ct_accum, ct_bias_accum, ct_m1, ct_m2, ct_t, ct_m1_accum, ct_m2_accum, ct_t_accum, ct_bias_m1, ct_bias_m2, ct_bias_m1_accum, ct_bias_m2_accum, beta1=beta1, beta2=beta2, epsilon=epsilon))
    mapped_burn_in.count()
    print('Burn in complete.')

    ct = numpy.divide(ct_accum.value[0], ct_accum.value[1])
    ct_m1 = numpy.divide(ct_m1_accum.value[0], ct_m1_accum.value[1])
    ct_m2 = numpy.divide(ct_m2_accum.value[0], ct_m2_accum.value[1])
    ct_t = ct_t + numpy.divide(ct_t_accum.value[0], ct_t_accum.value[1])
    ct_bias = numpy.divide(ct_bias_accum.value[0], ct_bias_accum.value[1])
    ct_bias_m1 = numpy.divide(ct_bias_m1_accum.value[0], ct_bias_m1_accum.value[1])
    ct_bias_m2 = numpy.divide(ct_bias_m2_accum.value[0], ct_bias_m2_accum.value[1])
    iters_to_test = []

    while True:
        if not (all_iters_count/float(iters_per_mse)) % 10:
            out_key = os.path.join(out_root, 'iter_errs.txt')
            print_iter_errs(iter_errs, out_bucket, out_key, header_line='Iter\tObjective\tTrain\tValid\n')
        param = UpdateAccumulatorParam()
        ct_accum = sc.accumulator(param.zero([ct, 1]), UpdateAccumulatorParam())
        ct_bias_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_m1_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_m2_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_t_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m1_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m2_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        run_gtotal = gtotal_train.mapPartitionsWithIndex(lambda x,y: one_iteration_ct_only(x, y, subsets[0], ct, rc, ct_bias, rbc, assay, assay_bias, learning_rate, batch_size, all_iters_count, ct_accum, ct_bias_accum, ct_m1, ct_m2, ct_t, ct_m1_accum, ct_m2_accum, ct_t_accum, ct_bias_m1, ct_bias_m2, ct_bias_m1_accum, ct_bias_m2_accum, beta1=beta1, beta2=beta2, epsilon=epsilon))
        run_gtotal.count()

        #update cell type data structures with results from SGD step
        ct = numpy.divide(ct_accum.value[0], ct_accum.value[1])
        print('{!s} Num partitions by ct_accum: {!s}'.format(all_iters_count, ct_accum.value[1]))
        ct_m1 = numpy.divide(ct_m1_accum.value[0], ct_m1_accum.value[1])
        ct_m2 = numpy.divide(ct_m2_accum.value[0], ct_m2_accum.value[1])
        ct_t = ct_t + numpy.array(numpy.around(numpy.divide(ct_t_accum.value[0], ct_t_accum.value[1])), dtype=int)
        ct_bias = numpy.divide(ct_bias_accum.value[0], ct_bias_accum.value[1])
        ct_bias_m1 = numpy.divide(ct_bias_m1_accum.value[0], ct_bias_m1_accum.value[1])
        ct_bias_m2 = numpy.divide(ct_bias_m2_accum.value[0], ct_bias_m2_accum.value[1])

        if not all_iters_count % iters_per_mse:
            mse = calc_mse_gtotal_split(gtotal_train, gtotal_valid, ct, rc, ct_bias, rbc, 
                                        assay, ra, assay_bias, rba, ri, rbi)
            print(mse)
            found_nan = False
            if numpy.any(numpy.isnan(mse)):
                print('Got NaN in MSE result: {!s}. Breaking and returning current min_err after {!s} iterations'.format(mse, all_iters_count))
                found_nan = True
                break
            iter_errs.append(copy.copy(mse))

            #the third index in the mse list is the validation mse
            if mse[MSE_VALID] < min_err[MSE_VALID]:
                print('NEW MIN MSE: {!s} < {!s}'.format(mse[MSE_VALID], min_err[MSE_VALID]))
                min_err = mse
                min_factors = {'ct':ct.copy(), 'ct_bias':ct_bias.copy(), 'min_idx':len(iter_errs) - 1,
                               'ct_m1':ct_m1.copy(), 'ct_m2':ct_m2.copy(), 'ct_bias_m1':ct_bias_m1.copy(),
                               'ct_bias_m2':ct_bias_m2.copy()}
            if not min_iters or len(iter_errs) > min_iters:
                if max_iters and len(iter_errs) > max_iters:
                    break
                if len(iter_errs) > (2 * win_size + win_spacing):
                    if iters_to_test:
                        to_test = list(itertools.chain(*[iter_errs[elt] for elt in iters_to_test]))
                    else:
                        to_test = iter_errs
                    start1 = len(iter_errs) - (2 * win_size + win_spacing)
                    stop1 = start1 + win_size
                    start2 = stop1 + win_spacing
                    win1 = numpy.array([elt[MSE_VALID] for elt in iter_errs[start1:stop1]])
                    win2 = numpy.array([elt[MSE_VALID] for elt in iter_errs[start2:]])
                    test = ranksums(win1, win2 + win2_shift)
                    if test[0] < 0 and test[1]/2 < pval:
                        if lrate_search_num:
                            learning_rate /= 2
                            beta1 -= 1.0 - beta1
                            ct = min_factors['ct']
                            ct_bias = min_factors['ct_bias']
                            ct_m1 = min_factors['ct_m1']
                            ct_m2 = min_factors['ct_m2']
                            ct_bias_m1 = min_factors['ct_bias_m1']
                            ct_bias_m2 = min_factors['ct_bias_m2']
                            lrate_search_num -= 1
                            print('Validation error has stopped improving. Halving learning rate: {!s}. Will do so {!s} more times.'.format(learning_rate, lrate_search_num))
                            #if we found a new minimum since the last lrate decrease
                            if iters_to_test and min_factors['min_idx'] > iters_to_test[-1].start:
                                iters_to_test[-1] = slice(iters_to_test[-1].start, min_factors['min_idx'] + 1)
                                iters_to_test.append(slice(len(iter_errs) - 1, None))
                            #if we did not find a new minimum since the last lrate decrease
                            elif iters_to_test:
                                iters_to_test[-1] = slice(len(iter_errs) - 1, None)
                            #if this is the first time we are decreasing the lrate
                            else:
                                iters_to_test = [slice(0, min_factors['min_idx'] + 1), slice(len(iter_errs) - 1, None)]
                        else:
                            break
        all_iters_count += 1
    return min_factors['ct'], min_factors['ct_bias'], iter_errs

def second_order_genome_updates(gtotal_elt, subset_coords, z_mat, z_h, ac_bias):
    gidx, data, genome, genome_bias = gtotal_elt[:4]
    data_to_use = data.toarray().flatten()[subset_coords]
    genome_plus = numpy.dot(numpy.dot(z_mat, data_to_use - ac_bias).T, z_h).flatten()
    return (gidx, data, genome_plus[:-1], genome_plus[-1]) + gtotal_elt[4:]

def train_genome_dimension(gtotal, ct, ct_bias, assay, assay_bias, ri, subsets=None):
    ct_z = numpy.hstack([ct, numpy.ones((ct.shape[0], 1))])
    assay_z = numpy.hstack([assay, numpy.ones((assay.shape[0], 1))])
    if subsets is not None:
        subset_flat_coords = numpy.where(~subsets[SUBSET_TRAIN].flatten())
    else:
        subset_flat_coords = slice(None)
    ac_bias = numpy.add.outer(ct_bias, assay_bias).flatten()[subset_flat_coords]
    z_mat = numpy.vstack([numpy.outer(ct_z[:,idx], assay_z[:,idx]).flatten()[subset_flat_coords] for idx in xrange(ct_z.shape[1])])
    reg_coef_add = numpy.ones(z_mat.shape[0])
    reg_coef_add[-1] = 0
    reg_coef_add = numpy.diag(reg_coef_add * ri)
    z_h = numpy.linalg.inv(numpy.dot(z_mat, z_mat.T) + reg_coef_add)
    gtotal_tmp = gtotal.map(lambda x: second_order_genome_updates(x, subset_flat_coords, z_mat, z_h, ac_bias)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
    return gtotal_tmp

def one_iteration(pidx, pdata, ct, ct_rcoef, ct_bias, ct_bias_rcoef, assay, assay_rcoef, assay_bias, assay_bias_rcoef, genome_rcoef, genome_bias_rcoef, lrate, batch_size, rseed, ct_accum, ct_bias_accum, assay_accum, assay_bias_accum, ct_m1, ct_m2, ct_t, ct_m1_accum, ct_m2_accum, ct_t_accum, ct_bias_m1, ct_bias_m2, ct_bias_m1_accum, ct_bias_m2_accum, assay_m1, assay_m2, assay_t, assay_m1_accum, assay_m2_accum, assay_t_accum, assay_bias_m1, assay_bias_m2, assay_bias_m1_accum, assay_bias_m2_accum, beta1=0.9, beta2=0.999, epsilon=1e-8, subsets=None):
    '''Complete an iteration (or a batch of iterations). Compute updates and
    apply them to the cell type and assay factors. Aggregate updates for the
    genome factors until they are synchronized into a new master RDD.
    '''
    #randomly sample pdata slices (genomic coordinates) to use for stochastic
    #updates
    rs = numpy.random.RandomState(rseed + pidx)
    partition_data = list(pdata)
    partition_len = numpy.arange(len(partition_data))
    updates = {}
    objective_vals = []
    g_idx, data_elt, genome, genome_bias, genome_m1, genome_m2, genome_t, genome_bias_m1, genome_bias_m2, all_nan, data_coords, data_flat, h_mat, h_bias = [None]*14
    data_points_count = 0
    accum_weight = 0
    ct_t_updates, assay_t_updates = numpy.zeros(ct_t.shape), numpy.zeros(assay_t.shape)
    ct_grad_correction_array = numpy.array([numpy.prod(_calc_beta1(beta1, numpy.arange(1, (t_val + 1 if t_val > 0 else 2)))) for t_val in ct_t])
    assay_grad_correction_array = numpy.array([numpy.prod(_calc_beta1(beta1, numpy.arange(1, (t_val + 1 if t_val > 0 else 2)))) for t_val in assay_t])
    genome_grad_correction_array = numpy.array([numpy.prod(_calc_beta1(beta1, numpy.arange(1, (elt[6] + 1 if elt[6] > 0 else 2)))) for elt in partition_data])
    while data_points_count < batch_size:
        sample_idx = rs.choice(partition_len)
        if len(partition_data[sample_idx][:9]) == 9:
            g_idx, data_elt, genome, genome_bias, genome_m1, genome_m2, genome_t, genome_bias_m1, genome_bias_m2 = partition_data[sample_idx][:9]
            subset_elt = partition_data[sample_idx][-1]
        else:
            raise Exception('Length of partition_data elt not 10: {!s}'.format(partition_data[sample_idx]))
        #randomly pick the data point that we will update on this iteration
        if subsets is not None:
            data_coords = list(zip(*numpy.where(~subsets[SUBSET_TRAIN]))) #take data_coords from training set
        else:
            data_coords = list(zip(*numpy.where(~subset_elt[SUBSET_TRAIN])))
        c_coord, a_coord = data_coords[rs.randint(len(data_coords))]
        if data_elt[c_coord, a_coord] == 0: #in the sparse matrix context, zero == missing
            continue
        ct_t_updates[c_coord] += 1
        ct_t_elt = ct_t[c_coord] + ct_t_updates[c_coord]
        assay_t_updates[a_coord] += 1
        assay_t_elt = assay_t[a_coord] + assay_t_updates[a_coord]
        genome_t += 1

        beta1_ct = _calc_beta1(beta1, ct_t_elt)
        beta1_ct_next = _calc_beta1(beta1, ct_t_elt + 1)
        beta1_assay = _calc_beta1(beta1, assay_t_elt)
        beta1_assay_next = _calc_beta1(beta1, assay_t_elt + 1)
        beta1_genome = _calc_beta1(beta1, genome_t)
        beta1_genome_next = _calc_beta1(beta1, genome_t + 1)

        if data_points_count:
            ct_grad_correction_array[c_coord] *= beta1_ct
            assay_grad_correction_array[a_coord] *= beta1_assay
            genome_grad_correction_array[sample_idx] *= beta1_genome
        ct_grad_correction = ct_grad_correction_array[c_coord]
        ct_m1_correction = ct_grad_correction * beta1_ct_next
        assay_grad_correction = assay_grad_correction_array[a_coord]
        assay_m1_correction = assay_grad_correction * beta1_assay_next
        genome_grad_correction = genome_grad_correction_array[sample_idx]
        genome_m1_correction = genome_grad_correction * beta1_genome_next

        working_ct_lrate = _calc_lrate(lrate, ct_t_elt)
        working_assay_lrate = _calc_lrate(lrate, assay_t_elt)
        working_genome_lrate = _calc_lrate(lrate, genome_t)
        #FIRST ORDER UPDATES
        #calculate the predicted value at this data point based on the factors
        #and bias vectors
        pred = (numpy.dot(assay[a_coord], (ct[c_coord] * genome)) +
                assay_bias[a_coord] + ct_bias[c_coord] + genome_bias)
        #calculate the error between the predicted and observed data point and
        #compute the updates to each factor/bias vector
        error = pred - data_elt[c_coord, a_coord]
        #update assay
        cxi = ct[c_coord] * genome
        grad = (error * cxi) + (assay_rcoef * assay[a_coord])
        grad_hat = grad/(1 - assay_grad_correction)
        assay_m1[a_coord,:] = beta1_assay * assay_m1[a_coord,:] + (1 - beta1_assay) * grad
        corrected_assay_m1 = assay_m1[a_coord,:]/(1 - assay_m1_correction)
        assay_m2[a_coord,:] = beta2 * assay_m2[a_coord,:] + (1 - beta2) * (grad ** 2)
        corrected_assay_m2 = assay_m2[a_coord,:]/(1 - (beta2 ** assay_t_elt))
        assay_m1_bar = ((1 - beta1_assay) * grad_hat) + (beta1_assay_next * corrected_assay_m1)
        updates['assay'] = working_assay_lrate * (assay_m1_bar/(numpy.sqrt(corrected_assay_m2) + epsilon))
        if numpy.sum(numpy.isnan(updates['assay'])):
            if STORAGE == 'S3':
                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct, genome, error, assay, assay_rcoef, assay_m1, corrected_assay_m1, assay_m2, corrected_assay_m2, updates['assay']))
            elif STORAGE == 'BLOB':
                azure_library.load_blob_pickle('encodeimputation', 'ADAM/assay_params.pickle', (ct, genome, error, assay, assay_rcoef, assay_m1, corrected_assay_m1, assay_m2, corrected_assay_m2, updates['assay']))
            raise Exception('NaN at assay update on iteration {!s}\n{!s}'.format(data_points_count, (assay_t_elt, cxi, grad, grad_hat, assay_m1[a_coord,:], corrected_assay_m1, assay_m2[a_coord,:], corrected_assay_m2, assay_m1_bar, updates['assay'])))

        #update assay_bias
        grad = error  + (assay_bias_rcoef * assay_bias[a_coord])
        grad_hat = grad/(1 - assay_grad_correction)
        assay_bias_m1[a_coord] = beta1_assay * assay_bias_m1[a_coord] + (1 - beta1_assay) * grad
        corrected_assay_bias_m1 = assay_bias_m1[a_coord]/(1 - assay_m1_correction)
        assay_bias_m2[a_coord] = beta2 * assay_bias_m2[a_coord] + (1 - beta2) * (grad ** 2)
        corrected_assay_bias_m2 = assay_bias_m2[a_coord]/(1 - (beta2 ** assay_t_elt))
        assay_bias_m1_bar = ((1 - beta1_assay) * grad_hat) + (beta1_assay_next * corrected_assay_bias_m1)
        updates['assay_bias'] = working_assay_lrate * (assay_bias_m1_bar/(numpy.sqrt(corrected_assay_bias_m2) + epsilon))
        if numpy.sum(numpy.isnan(updates['assay_bias'])):
            if STORAGE == 'S3':
                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct, genome, error, assay_bias, assay_bias_rcoef, assay_bias_m1, corrected_assay_bias_m1, assay_bias_m2, corrected_assay_bias_m2, updates['assay_bias']))
            elif STORAGE == 'BLOB':
                azure_library.load_blob_pickle('encodeimputation', 'ADAM/assay_bias_params.pickle', (ct, genome, error, assay_bias, assay_bias_rcoef, assay_bias_m1, corrected_assay_bias_m1, assay_bias_m2, corrected_assay_bias_m2, updates['assay_bias']))
            raise Exception('NaN at assay_bias update on iteration {!s}'.format(data_points_count))

        #update ct
        axi = assay[a_coord] * genome
        grad = error * axi + (ct_rcoef * ct[c_coord])
        grad_hat = grad/(1 - ct_grad_correction)
        ct_m1[c_coord,:] = beta1_ct * ct_m1[c_coord,:] + (1 - beta1_ct) * grad
        corrected_ct_m1 = ct_m1[c_coord,:]/(1 - ct_m1_correction)
        ct_m2[c_coord,:] = beta2 * ct_m2[c_coord,:] + (1 - beta2) * (grad ** 2)
        corrected_ct_m2 = ct_m2[c_coord,:]/(1 - (beta2 ** ct_t_elt))
        ct_m1_bar = ((1 - beta1_ct) * grad_hat) + (beta1_ct_next * corrected_ct_m1)
        updates['ct'] = working_ct_lrate * (ct_m1_bar/(numpy.sqrt(corrected_ct_m2) + epsilon))
        if numpy.sum(numpy.isnan(updates['ct'])):
            if STORAGE == 'S3':
                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct, genome, error, assay, ct_rcoef, ct_m1, corrected_ct_m1, ct_m2, corrected_ct_m2, updates['ct']))
            if STORAGE == 'BLOB':
                azure_library.load_blob_pickle('encodeimputation', 'ADAM/ct_params.pickle', (ct, genome, error, assay, ct_rcoef, ct_m1, corrected_ct_m1, ct_m2, corrected_ct_m2, updates['ct']))
            raise Exception('NaN at ct update on iteration {!s}'.format(data_points_count))

        #update ct_bias
        grad = error + (ct_bias_rcoef * ct_bias[c_coord])
        grad_hat = grad/(1 - ct_grad_correction)
        ct_bias_m1[c_coord] = beta1_ct * ct_bias_m1[c_coord] + (1 - beta1_ct) * grad
        corrected_ct_bias_m1 = ct_bias_m1[c_coord]/(1 - ct_m1_correction)
        ct_bias_m2[c_coord] = beta2 * ct_bias_m2[c_coord] + (1 - beta2) * (grad ** 2)
        corrected_ct_bias_m2 = ct_bias_m2[c_coord]/(1 - (beta2 ** ct_t_elt))
        ct_bias_m1_bar = ((1 - beta1_ct) * grad_hat) + (beta1_ct_next * corrected_ct_bias_m1)
        updates['ct_bias'] = working_ct_lrate * (ct_bias_m1_bar/(numpy.sqrt(corrected_ct_bias_m2) + epsilon))
        if numpy.sum(numpy.isnan(updates['ct_bias'])):
            if STORAGE == 'S3':
                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct_bias, genome, error, assay, ct_bias_rcoef, ct_bias_m1, corrected_ct_bias_m1, ct_bias_m2, corrected_ct_bias_m2, updates['ct_bias']))
            if STORAGE == 'BLOB':
                azure_library.load_blob_pickle('encodeimputation', 'ADAM/params.pickle', (ct_bias, genome, error, assay, ct_bias_rcoef, ct_bias_m1, corrected_ct_bias_m1, ct_bias_m2, corrected_ct_bias_m2, updates['ct_bias']))
            raise Exception('NaN at ct_bias update on iteration {!s}'.format(data_points_count))

        #update genome
        axc = assay[a_coord] * ct[c_coord]
        grad = error * axc + (genome_rcoef * genome)
        grad_hat = grad/(1 - genome_grad_correction)
        genome_m1 = beta1_genome * genome_m1 + (1 - beta1_genome) * grad
        corrected_genome_m1 = genome_m1/(1 - genome_m1_correction)
        genome_m2 = beta2 * genome_m2 + (1 - beta2) * (grad ** 2)
        corrected_genome_m2 = genome_m2/(1 - (beta2 ** genome_t))
        genome_m1_bar = ((1 - beta1_genome) * grad_hat) + (beta1_genome_next * corrected_genome_m1)
        genome -= working_genome_lrate * (genome_m1_bar/(numpy.sqrt(corrected_genome_m2) + epsilon))

        #update genome_bias
        grad = error + (genome_bias_rcoef * genome_bias)
        grad_hat = grad/(1 - genome_grad_correction)
        genome_bias_m1 = beta1_genome * genome_bias_m1 + (1 - beta1_genome) * grad
        corrected_genome_bias_m1 = genome_bias_m1/(1 - genome_m1_correction)
        genome_bias_m2 = beta2 * genome_bias_m2 + (1 - beta2) * (grad ** 2)
        corrected_genome_bias_m2 = genome_bias_m2/(1 - (beta2 ** genome_t))
        genome_bias_m1_bar = ((1 - beta1_genome) * grad_hat) + (beta1_genome_next * corrected_genome_bias_m1)
        genome_bias -= working_genome_lrate * (genome_bias_m1_bar/(numpy.sqrt(corrected_genome_bias_m2) + epsilon))

        assay[a_coord,:] -= updates['assay']
        assay_bias[a_coord] -= updates['assay_bias']
        ct[c_coord,:] -= updates['ct']
        ct_bias[c_coord] -= updates['ct_bias']
        partition_data[sample_idx] = (g_idx, data_elt, genome, genome_bias, genome_m1, genome_m2, genome_t, genome_bias_m1, genome_bias_m2) + partition_data[sample_idx][9:]

        data_points_count += 1
        accum_weight += 1

    #weight these updates based on the weights of the data points they incorporate
    accum_weight = int(round(accum_weight/data_points_count))
    ct_accum.add([ct * accum_weight, accum_weight])
    ct_m1_accum.add([ct_m1 * accum_weight, accum_weight])
    ct_m2_accum.add([ct_m2 * accum_weight, accum_weight])
    ct_t_accum.add([ct_t_updates * accum_weight, accum_weight])
    ct_bias_accum.add([ct_bias * accum_weight, accum_weight])
    ct_bias_m1_accum.add([ct_bias_m1 * accum_weight, accum_weight])
    ct_bias_m2_accum.add([ct_bias_m2 * accum_weight, accum_weight])
    assay_accum.add([assay * accum_weight, accum_weight])
    assay_m1_accum.add([assay_m1 * accum_weight, accum_weight])
    assay_m2_accum.add([assay_m2 * accum_weight, accum_weight])
    assay_t_accum.add([assay_t_updates * accum_weight, accum_weight])
    assay_bias_accum.add([assay_bias * accum_weight, accum_weight])
    assay_bias_m1_accum.add([assay_bias_m1 * accum_weight, accum_weight])
    assay_bias_m2_accum.add([assay_bias_m2 * accum_weight, accum_weight])

    if len(objective_vals):
        if STORAGE == 'S3':
            bucket = s3_library.S3.get_bucket('encodeimputation2')
            key = bucket.new_key('objective_vals/obj_vals.{!s}_{!s}.txt'.format(rseed, pidx))
            key.set_contents_from_string('\n'.join([str(elt) for elt in objective_vals]) + '\n', headers={'x-amz-request-payer':'requester'})
        elif STORAGE == 'BLOB':
            azure_library.load_blob_from_text('encodeimputation', 'objective_vals/obj_vals.{!s}_{!s}.txt'.format(rseed, pidx), '\n'.join([str(elt) for elt in objective_vals]) + '\n')

    for elt in partition_data:
        yield elt

def one_iteration_ct_genome(pidx, pdata, ct, ct_rcoef, ct_bias, ct_bias_rcoef, assay, assay_rcoef, assay_bias, assay_bias_rcoef, genome_rcoef, genome_bias_rcoef, lrate, batch_size, rseed, ct_accum, ct_bias_accum, ct_m1, ct_m2, ct_t, ct_m1_accum, ct_m2_accum, ct_t_accum, ct_bias_m1, ct_bias_m2, ct_bias_m1_accum, ct_bias_m2_accum, beta1=0.9, beta2=0.999, epsilon=1e-8, subsets=None):
    '''Complete an iteration (or a batch of iterations). Compute updates and
    apply them to the cell type and assay factors. Aggregate updates for the
    genome factors until they are synchronized into a new master RDD.
    '''
    #randomly sample pdata slices (genomic coordinates) to use for stochastic
    #updates
    rs = numpy.random.RandomState(rseed + pidx)
    partition_data = list(pdata)
    partition_len = numpy.arange(len(partition_data))
    updates = {}
    objective_vals = []
    g_idx, data_elt, genome, genome_bias, genome_m1, genome_m2, genome_t, genome_bias_m1, genome_bias_m2, all_nan, data_coords, data_flat, h_mat, h_bias = [None]*14
    data_points_count = 0
    accum_weight = 0
    ct_t_updates = numpy.zeros(ct_t.shape)
    ct_grad_correction_array = numpy.array([numpy.prod(_calc_beta1(beta1, numpy.arange(1, (t_val + 1 if t_val > 0 else 2)))) for t_val in ct_t])
#    assay_grad_correction_array = numpy.array([numpy.prod(_calc_beta1(beta1, numpy.arange(1, (t_val + 1 if t_val > 0 else 2)))) for t_val in assay_t])
    genome_grad_correction_array = numpy.array([numpy.prod(_calc_beta1(beta1, numpy.arange(1, (elt[6] + 1 if elt[6] > 0 else 2)))) for elt in partition_data])
    while data_points_count < batch_size:
        sample_idx = rs.choice(partition_len)
        if len(partition_data[sample_idx][:9]) == 9:
            g_idx, data_elt, genome, genome_bias, genome_m1, genome_m2, genome_t, genome_bias_m1, genome_bias_m2 = partition_data[sample_idx][:9]
            subset_elt = partition_data[sample_idx][-1]
        else:
            raise Exception('Length of partition_data elt not 10: {!s}'.format(partition_data[sample_idx]))
        #randomly pick the data point that we will update on this iteration
        if subsets is not None:
            data_coords = list(zip(*numpy.where(~subsets[SUBSET_TRAIN]))) #take data_coords from training set
        else:
            data_coords = list(zip(*numpy.where(~subset_elt[SUBSET_TRAIN])))
        c_coord, a_coord = data_coords[rs.randint(len(data_coords))]
        if data_elt[c_coord, a_coord] == 0: #in the sparse matrix context, zero == missing
            continue
        ct_t_updates[c_coord] += 1
        ct_t_elt = ct_t[c_coord] + ct_t_updates[c_coord]
#        assay_t_updates[a_coord] += 1
#        assay_t_elt = assay_t[a_coord] + assay_t_updates[a_coord]
        genome_t += 1

        beta1_ct =  _calc_beta1(beta1, ct_t_elt)
        beta1_ct_next = _calc_beta1(beta1, ct_t_elt + 1)
#        beta1_assay = _calc_beta1(beta1, assay_t_elt)
#        beta1_assay_next = _calc_beta1(beta1, assay_t_elt + 1)
        beta1_genome = _calc_beta1(beta1, genome_t)
        beta1_genome_next = _calc_beta1(beta1, genome_t + 1)

        if data_points_count:
            ct_grad_correction_array[c_coord] *= beta1_ct
#            assay_grad_correction_array[a_coord] *= beta1_assay
            genome_grad_correction_array[sample_idx] *= beta1_genome
        ct_grad_correction = ct_grad_correction_array[c_coord]
        ct_m1_correction = ct_grad_correction * beta1_ct_next
#        assay_grad_correction = assay_grad_correction_array[a_coord]
#        assay_m1_correction = assay_grad_correction * beta1_assay_next
        genome_grad_correction = genome_grad_correction_array[sample_idx]
        genome_m1_correction = genome_grad_correction * beta1_genome_next

        working_ct_lrate = _calc_lrate(lrate, ct_t_elt)
#        working_assay_lrate = _calc_lrate(lrate, assay_t_elt)
        working_genome_lrate = _calc_lrate(lrate, genome_t)
        #FIRST ORDER UPDATES
        #calculate the predicted value at this data point based on the factors
        #and bias vectors
        pred = (numpy.dot(assay[a_coord], (ct[c_coord] * genome)) +
                assay_bias[a_coord] + ct_bias[c_coord] + genome_bias)
        #calculate the error between the predicted and observed data point and
        #compute the updates to each factor/bias vector
        error = pred - data_elt[c_coord, a_coord]
#        #update assay
#        cxi = ct[c_coord] * genome
#        grad = (error * cxi) + (assay_rcoef * assay[a_coord])
#        grad_hat = grad/(1 - assay_grad_correction)
#        assay_m1[a_coord,:] = beta1_assay * assay_m1[a_coord,:] + (1 - beta1_assay) * grad
#        corrected_assay_m1 = assay_m1[a_coord,:]/(1 - assay_m1_correction)
#        assay_m2[a_coord,:] = beta2 * assay_m2[a_coord,:] + (1 - beta2) * (grad ** 2)
#        corrected_assay_m2 = assay_m2[a_coord,:]/(1 - (beta2 ** assay_t_elt))
#        assay_m1_bar = ((1 - beta1_assay) * grad_hat) + (beta1_assay_next * corrected_assay_m1)
#        updates['assay'] = working_assay_lrate * (assay_m1_bar/(numpy.sqrt(corrected_assay_m2) + epsilon))
#        if numpy.sum(numpy.isnan(updates['assay'])):
#            if STORAGE == 'S3':
#                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct, genome, error, assay, assay_rcoef, assay_m1, corrected_assay_m1, assay_m2, corrected_assay_m2, updates['assay']))
#            elif STORAGE == 'BLOB':
#                azure_library.load_blob_pickle('encodeimputation', 'ADAM/assay_params.pickle', (ct, genome, error, assay, assay_rcoef, assay_m1, corrected_assay_m1, assay_m2, corrected_assay_m2, updates['assay']))
#            raise Exception('NaN at assay update on iteration {!s}\n{!s}'.format(data_points_count, (assay_t_elt, cxi, grad, grad_hat, assay_m1[a_coord,:], corrected_assay_m1, assay_m2[a_coord,:], corrected_assay_m2, assay_m1_bar, updates['assay'])))

#        #update assay_bias
#        grad = error  + (assay_bias_rcoef * assay_bias[a_coord])
#        grad_hat = grad/(1 - assay_grad_correction)
#        assay_bias_m1[a_coord] = beta1_assay * assay_bias_m1[a_coord] + (1 - beta1_assay) * grad
#        corrected_assay_bias_m1 = assay_bias_m1[a_coord]/(1 - assay_m1_correction)
#        assay_bias_m2[a_coord] = beta2 * assay_bias_m2[a_coord] + (1 - beta2) * (grad ** 2)
#        corrected_assay_bias_m2 = assay_bias_m2[a_coord]/(1 - (beta2 ** assay_t_elt))
#        assay_bias_m1_bar = ((1 - beta1_assay) * grad_hat) + (beta1_assay_next * corrected_assay_bias_m1)
#        updates['assay_bias'] = working_assay_lrate * (assay_bias_m1_bar/(numpy.sqrt(corrected_assay_bias_m2) + epsilon))
#        if numpy.sum(numpy.isnan(updates['assay_bias'])):
#            if STORAGE == 'S3':
#                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct, genome, error, assay_bias, assay_bias_rcoef, assay_bias_m1, corrected_assay_bias_m1, assay_bias_m2, corrected_assay_bias_m2, updates['assay_bias']))
#            elif STORAGE == 'BLOB':
#                azure_library.load_blob_pickle('encodeimputation', 'ADAM/assay_bias_params.pickle', (ct, genome, error, assay_bias, assay_bias_rcoef, assay_bias_m1, corrected_assay_bias_m1, assay_bias_m2, corrected_assay_bias_m2, updates['assay_bias']))
#            raise Exception('NaN at assay_bias update on iteration {!s}'.format(data_points_count))

        #update ct
        axi = assay[a_coord] * genome
        grad = error * axi + (ct_rcoef * ct[c_coord])
        grad_hat = grad/(1 - ct_grad_correction)
        ct_m1[c_coord,:] = beta1_ct * ct_m1[c_coord,:] + (1 - beta1_ct) * grad
        corrected_ct_m1 = ct_m1[c_coord,:]/(1 - ct_m1_correction)
        ct_m2[c_coord,:] = beta2 * ct_m2[c_coord,:] + (1 - beta2) * (grad ** 2)
        corrected_ct_m2 = ct_m2[c_coord,:]/(1 - (beta2 ** ct_t_elt))
        ct_m1_bar = ((1 - beta1_ct) * grad_hat) + (beta1_ct_next * corrected_ct_m1)
        updates['ct'] = working_ct_lrate * (ct_m1_bar/(numpy.sqrt(corrected_ct_m2) + epsilon))
        if numpy.sum(numpy.isnan(updates['ct'])):
            if STORAGE == 'S3':
                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct, genome, error, assay, ct_rcoef, ct_m1, corrected_ct_m1, ct_m2, corrected_ct_m2, updates['ct']))
            if STORAGE == 'BLOB':
                azure_library.load_blob_pickle('encodeimputation', 'ADAM/ct_params.pickle', (ct, genome, error, assay, ct_rcoef, ct_m1, corrected_ct_m1, ct_m2, corrected_ct_m2, updates['ct']))
            raise Exception('NaN at ct update on iteration {!s}'.format(data_points_count))

        #update ct_bias
        grad = error + (ct_bias_rcoef * ct_bias[c_coord])
        grad_hat = grad/(1 - ct_grad_correction)
        ct_bias_m1[c_coord] = beta1_ct * ct_bias_m1[c_coord] + (1 - beta1_ct) * grad
        corrected_ct_bias_m1 = ct_bias_m1[c_coord]/(1 - ct_m1_correction)
        ct_bias_m2[c_coord] = beta2 * ct_bias_m2[c_coord] + (1 - beta2) * (grad ** 2)
        corrected_ct_bias_m2 = ct_bias_m2[c_coord]/(1 - (beta2 ** ct_t_elt))
        ct_bias_m1_bar = ((1 - beta1_ct) * grad_hat) + (beta1_ct_next * corrected_ct_bias_m1)
        updates['ct_bias'] = working_ct_lrate * (ct_bias_m1_bar/(numpy.sqrt(corrected_ct_bias_m2) + epsilon))
        if numpy.sum(numpy.isnan(updates['ct_bias'])):
            if STORAGE == 'S3':
                s3_library.set_pickle_s3('encodeimputation2', 'ADAM/params.pickle', (ct_bias, genome, error, assay, ct_bias_rcoef, ct_bias_m1, corrected_ct_bias_m1, ct_bias_m2, corrected_ct_bias_m2, updates['ct_bias']))
            if STORAGE == 'BLOB':
                azure_library.load_blob_pickle('encodeimputation', 'ADAM/params.pickle', (ct_bias, genome, error, assay, ct_bias_rcoef, ct_bias_m1, corrected_ct_bias_m1, ct_bias_m2, corrected_ct_bias_m2, updates['ct_bias']))
            raise Exception('NaN at ct_bias update on iteration {!s}'.format(data_points_count))

        #update genome
        axc = assay[a_coord] * ct[c_coord]
        grad = error * axc + (genome_rcoef * genome)
        grad_hat = grad/(1 - genome_grad_correction)
        genome_m1 = beta1_genome * genome_m1 + (1 - beta1_genome) * grad
        corrected_genome_m1 = genome_m1/(1 - genome_m1_correction)
        genome_m2 = beta2 * genome_m2 + (1 - beta2) * (grad ** 2)
        corrected_genome_m2 = genome_m2/(1 - (beta2 ** genome_t))
        genome_m1_bar = ((1 - beta1_genome) * grad_hat) + (beta1_genome_next * corrected_genome_m1)
        genome -= working_genome_lrate * (genome_m1_bar/(numpy.sqrt(corrected_genome_m2) + epsilon))

        #update genome_bias
        grad = error + (genome_bias_rcoef * genome_bias)
        grad_hat = grad/(1 - genome_grad_correction)
        genome_bias_m1 = beta1_genome * genome_bias_m1 + (1 - beta1_genome) * grad
        corrected_genome_bias_m1 = genome_bias_m1/(1 - genome_m1_correction)
        genome_bias_m2 = beta2 * genome_bias_m2 + (1 - beta2) * (grad ** 2)
        corrected_genome_bias_m2 = genome_bias_m2/(1 - (beta2 ** genome_t))
        genome_bias_m1_bar = ((1 - beta1_genome) * grad_hat) + (beta1_genome_next * corrected_genome_bias_m1)
        genome_bias -= working_genome_lrate * (genome_bias_m1_bar/(numpy.sqrt(corrected_genome_bias_m2) + epsilon))

#        assay[a_coord,:] -= updates['assay']
#        assay_bias[a_coord] -= updates['assay_bias']
        ct[c_coord,:] -= updates['ct']
        ct_bias[c_coord] -= updates['ct_bias']
        partition_data[sample_idx] = (g_idx, data_elt, genome, genome_bias, genome_m1, genome_m2, genome_t, genome_bias_m1, genome_bias_m2) + partition_data[sample_idx][9:]

        data_points_count += 1
        accum_weight += 1

    #weight these updates based on the weights of the data points they incorporate
    accum_weight = int(round(accum_weight/data_points_count))
    ct_accum.add([ct * accum_weight, accum_weight])
    ct_m1_accum.add([ct_m1 * accum_weight, accum_weight])
    ct_m2_accum.add([ct_m2 * accum_weight, accum_weight])
    ct_t_accum.add([ct_t_updates * accum_weight, accum_weight])
    ct_bias_accum.add([ct_bias * accum_weight, accum_weight])
    ct_bias_m1_accum.add([ct_bias_m1 * accum_weight, accum_weight])
    ct_bias_m2_accum.add([ct_bias_m2 * accum_weight, accum_weight])
#    assay_accum.add([assay * accum_weight, accum_weight])
#    assay_m1_accum.add([assay_m1 * accum_weight, accum_weight])
#    assay_m2_accum.add([assay_m2 * accum_weight, accum_weight])
#    assay_t_accum.add([assay_t_updates * accum_weight, accum_weight])
#    assay_bias_accum.add([assay_bias * accum_weight, accum_weight])
#    assay_bias_m1_accum.add([assay_bias_m1 * accum_weight, accum_weight])
#    assay_bias_m2_accum.add([assay_bias_m2 * accum_weight, accum_weight])

    if len(objective_vals):
        if STORAGE == 'S3':
            bucket = s3_library.S3.get_bucket('encodeimputation2')
            key = bucket.new_key('objective_vals/obj_vals.{!s}_{!s}.txt'.format(rseed, pidx))
            key.set_contents_from_string('\n'.join([str(elt) for elt in objective_vals]) + '\n', headers={'x-amz-request-payer':'requester'})
        elif STORAGE == 'BLOB':
            azure_library.load_blob_from_text('encodeimputation', 'objective_vals/obj_vals.{!s}_{!s}.txt'.format(rseed, pidx), '\n'.join([str(elt) for elt in objective_vals]) + '\n')

    for elt in partition_data:
        yield elt

def _sgd_genome_compute_mse(data_list, genome_array, genome_bias_array, ct, assay, ct_bias, assay_bias, subsets):
#    assert not numpy.any(numpy.logical_and(~expanded_train_mask, ~expanded_test_mask)), 'Assert #1'
    mse_vals = numpy.zeros((3,))
    ctassay_factors = numpy.array([numpy.outer(ct[:,idx], assay[:,idx]).flatten() for idx in range(ct.shape[1])])
    bias_sum = numpy.add.outer(genome_bias_array, numpy.add.outer(ct_bias, assay_bias).flatten())
    pred_array = numpy.array([numpy.sum(genome_array[i_coord][:,None] * ctassay_factors, axis=0) for i_coord in xrange(genome_array.shape[0])]) + bias_sum
    data_array = numpy.array([elt.toarray().flatten() for elt in data_list])
    data_array = numpy.ma.masked_array(data_array, mask=data_array == 0)
    se = ((data_array - pred_array) ** 2)

    for idx, subset in enumerate(subsets):
        if len(subset.shape) == 3:
            coords_to_test = numpy.where(~subset.reshape(subset.shape[0], -1))
            mse_vals[idx] = numpy.mean(se[coords_to_test])
        else:
            coord_to_test = numpy.where(~subset)[0]
            mse_vals[idx] = numpy.mean(se[:,coords_to_test])

    return mse_vals

def sgd_genome_only(pidx, pdata, ct, ct_bias, assay, assay_bias, genome_rcoef, genome_bias_rcoef, lrate, batch_size, rseed, beta1=0.9, beta2=0.999, epsilon=1e-8, auto_convergence_detection=True, lrate_search_num=0, iters_per_mse=50000, subsets=None):
    '''Complete an iteration (or a batch of iterations). Compute updates and
    apply them to the cell type and assay factors. Aggregate updates for the
    genome factors until they are synchronized into a new master RDD.
    '''
    #randomly sample pdata slices (genomic coordinates) to use for stochastic updates
    rs = numpy.random.RandomState(rseed + pidx)
    num_elts = 0
    gidx_list = []
    data_list = []
    genome_array = []
    genome_bias_array = []
    genome_m1_array = []
    genome_m2_array = []
    genome_t_array = []
    genome_bias_m1_array = []
    genome_bias_m2_array = []
#    err_weight_array = []
    subset_array_list = []
    for elt in pdata:
        gidx_list.append(elt[0])
        data_list.append(elt[1])
        genome_array.append(elt[2])
        genome_bias_array.append(elt[3])
        genome_m1_array.append(elt[4])
        genome_m2_array.append(elt[5])
        genome_t_array.append(elt[6])
        genome_bias_m1_array.append(elt[7])
        genome_bias_m2_array.append(elt[8])
#        err_weight_array.append(elt[9])
        if subsets is None:
            if elt[-1] is not None:
                subset_array_list.append(elt[-1])
            else:
                raise Exception('Must supply subsets.')
                subset = numpy.ones(elt[1].shape, dtype=bool)
                subset[elt[1].nonzero()] = False
                subset_array_list.append([subset])
        num_elts += 1
#    raise Exception((len(subset_array_list), len(subset_array_list[0]), subset_array_list))
    if subsets is None:
        subsets = [numpy.array(elt) for elt in zip(*subset_array_list)]
    if len(subsets) == 1:
        raise Exception('Validation subset required.')
    genome_array = numpy.array(genome_array, dtype=float)
    genome_bias_array = numpy.array(genome_bias_array, dtype=float)
    genome_m1_array = numpy.array(genome_m1_array, dtype=float)
    genome_m2_array = numpy.array(genome_m2_array, dtype=float)
    genome_t_array = numpy.array(genome_t_array)
    genome_bias_m1_array = numpy.array(genome_bias_m1_array, dtype=float)
    genome_bias_m2_array = numpy.array(genome_bias_m2_array, dtype=float)
#    err_weight_array = numpy.array(err_weight_array)
    batch_size_val = batch_size
    data_points_count = 0
    if auto_convergence_detection is True:
        g_iter_errs = [_sgd_genome_compute_mse(data_list, genome_array, genome_bias_array, ct, assay, ct_bias, assay_bias, subsets)]
        min_genome_array = genome_array.copy()
        min_genome_bias_array = genome_bias_array.copy()
        min_genome_m1_array = genome_m1_array.copy()
        min_genome_bias_m1_array = genome_bias_m1_array.copy()
        min_genome_m2_array = genome_m2_array.copy()
        min_genome_bias_m2_array = genome_bias_m2_array.copy()
        min_genome_t_array = genome_t_array.copy()
    else:
        g_iter_errs = []
        min_genome_array = None
        min_genome_bias_array = None
        min_genome_m1_array = None
        min_genome_bias_m1_array = None
        min_genome_m2_array = None
        min_genome_bias_m2_array = None
        min_genome_t_array = None
    g_iters_to_test = []
    min_mse_idx = 0
    min_genome_grad_correction_array = None
    genome_grad_correction_array = numpy.array([numpy.prod(_calc_beta1(beta1, numpy.arange(1, (genome_t_array[t_idx] + 1 if genome_t_array[t_idx] > 0 else 2)))) for t_idx in xrange(num_elts)])
    data_coords = list(zip(*numpy.where(~subsets[SUBSET_TRAIN])))
    while True:
        #randomly pick the data point that we will update on this iteration
        if len(data_coords[0]) == 2:
            i_coord = rs.randint(num_elts)
            c_coord, a_coord = data_coords[rs.randint(len(data_coords))]
        else:
            i_coord, c_coord, a_coord = data_coords[rs.randint(len(data_coords))]
        if data_list[i_coord][c_coord, a_coord] == 0: #in sparse matrix context, 0 == missing
            continue

        genome_t_array[i_coord] += 1

        beta1_genome = _calc_beta1(beta1, genome_t_array[i_coord])
        beta1_genome_next = _calc_beta1(beta1, genome_t_array[i_coord] + 1)

        if data_points_count:
            genome_grad_correction_array[i_coord] *= beta1_genome

        genome_grad_correction = genome_grad_correction_array[i_coord]
        genome_m1_correction = genome_grad_correction * beta1_genome_next

        working_genome_lrate = _calc_lrate(lrate, genome_t_array[i_coord])

        #FIRST ORDER UPDATES
        #calculate the predicted value at this data point based on the factors
        #and bias vectors
        pred = (numpy.dot(assay[a_coord], (ct[c_coord] * genome_array[i_coord])) +
                assay_bias[a_coord] + ct_bias[c_coord] + genome_bias_array[i_coord])
        #calculate the error between the predicted and observed data point and
        #compute the updates to each factor/bias vector
        error = pred - data_list[i_coord][c_coord, a_coord]

        #update genome
        axc = assay[a_coord] * ct[c_coord]
        grad = error * axc + (genome_rcoef * genome_array[i_coord])
        grad_hat = grad/(1 - genome_grad_correction)
        genome_m1_array[i_coord,:] = beta1_genome * genome_m1_array[i_coord] + (1 - beta1_genome) * grad
        corrected_genome_m1 = genome_m1_array[i_coord]/(1 - genome_m1_correction)
        genome_m2_array[i_coord,:] = beta2 * genome_m2_array[i_coord] + (1 - beta2) * (grad ** 2)
        corrected_genome_m2 = genome_m2_array[i_coord]/(1 - (beta2 ** genome_t_array[i_coord]))
        genome_m1_bar = ((1 - beta1_genome) * grad_hat) + (beta1_genome_next * corrected_genome_m1)
        genome_array[i_coord,:] -= working_genome_lrate * (genome_m1_bar/(numpy.sqrt(corrected_genome_m2) + epsilon))

        #update genome_bias
        grad = error + (genome_bias_rcoef * genome_bias_array[i_coord])
        grad_hat = grad/(1 - genome_grad_correction)
        genome_bias_m1_array[i_coord] = beta1_genome * genome_bias_m1_array[i_coord] + (1 - beta1_genome) * grad
        corrected_genome_bias_m1 = genome_bias_m1_array[i_coord]/(1 - genome_m1_correction)
        genome_bias_m2_array[i_coord] = beta2 * genome_bias_m2_array[i_coord] + (1 - beta2) * (grad ** 2)
        corrected_genome_bias_m2 = genome_bias_m2_array[i_coord]/(1 - (beta2 ** genome_t_array[i_coord]))
        genome_bias_m1_bar = ((1 - beta1_genome) * grad_hat) + (beta1_genome_next * corrected_genome_bias_m1)
        genome_bias_array[i_coord] -= working_genome_lrate * (genome_bias_m1_bar/(numpy.sqrt(corrected_genome_bias_m2) + epsilon))

        data_points_count += 1

        if auto_convergence_detection is True:
            if data_points_count > batch_size_val and not data_points_count % iters_per_mse:
                mse_vals = _sgd_genome_compute_mse(data_list, genome_array, genome_bias_array, ct, assay, ct_bias, assay_bias, subsets)
                if mse_vals[MSE_VALID] < g_iter_errs[min_mse_idx][MSE_VALID]:
                    min_mse_idx = len(g_iter_errs)
                    min_genome_grad_correction_array = genome_grad_correction_array.copy()
                    min_genome_array = genome_array.copy()
                    min_genome_bias_array = genome_bias_array.copy()
                    min_genome_m1_array = genome_m1_array.copy()
                    min_genome_bias_m1_array = genome_bias_m1_array.copy()
                    min_genome_m2_array = genome_m2_array.copy()
                    min_genome_bias_m2_array = genome_bias_m2_array.copy()
                    min_genome_t_array = genome_t_array.copy()
                g_iter_errs.append(mse_vals)

                win_size = 20
                win_spacing = 5
                win2_shift = 1e-5
                pval = 0.01
                if len(g_iter_errs) >= (2 * win_size + win_spacing):
                    if g_iter_errs[0][MSE_VALID] - g_iter_errs[min_mse_idx][MSE_VALID] < 0.01:
                        break
                    if g_iters_to_test:
                        g_to_test = list(itertools.chain(*[g_iter_errs[elt] for elt in g_iters_to_test]))
                        idx_list = list(range(len(g_iter_errs)))
                        min_g_to_test_idx = list(itertools.chain(*[idx_list[elt] for elt in g_iters_to_test])).index(min_mse_idx)
                    else:
                        g_to_test = g_iter_errs
                        min_g_to_test_idx = min_mse_idx
                    start1 = len(g_to_test) - (2 * win_size + win_spacing)
                    stop1 = start1 + win_size
                    start2 = stop1 + win_spacing
                    if ((start1 < 0) or (stop1 > len(g_to_test)) or (start2 > len(g_to_test))):
                        continue
                    win1 = numpy.array(g_to_test[start1:stop1])[:,1]
                    win2 = numpy.array(g_to_test[start2:])[:,1]
                    test = ranksums(win1, win2 + win2_shift)
                    if ((test[0] < 0 and test[1]/2 < pval) or
                        ((len(g_to_test) - min_g_to_test_idx) > (1.5 * (2 * win_size + win_spacing)))):
                        if lrate_search_num:
                            lrate /= 2
                            beta1 -= 1.0 - beta1
                            lrate_search_num -= 1
                            genome_grad_correction_array = min_genome_grad_correction_array
                            genome_array = min_genome_array
                            genome_bias_array = min_genome_bias_array
                            genome_m1_array = min_genome_m1_array
                            genome_bias_m1_array = min_genome_bias_m1_array
                            genome_m2_array = min_genome_m2_array
                            genome_bias_m2_array = min_genome_bias_m2_array
                            genome_t_array = min_genome_t_array

                            #if we found a new minimum since the last lrate decrease
                            if g_iters_to_test and min_mse_idx > g_iters_to_test[-1].start:
                                g_iters_to_test[-1] = slice(g_iters_to_test[-1].start, min_mse_idx + 1)
                                g_iters_to_test.append(slice(len(g_iter_errs) - 1, None))
                            #if we did not find a new minimum since the last lrate decrease
                            elif g_iters_to_test:
                                g_iters_to_test[-1] = slice(len(g_iter_errs) - 1, None)
                            #if this is the first time we are decreasing the lrate
                            else:
                                g_iters_to_test = [slice(0,min_mse_idx + 1), slice(len(g_iter_errs) - 1, None)]
                        else:
                            break
#                    if len(g_iter_errs) - min_mse_idx > 200:
#                        break
        elif data_points_count > batch_size_val:
            break

    if auto_convergence_detection is True and min_genome_array is not None:
        for idx in xrange(num_elts):
            if subset_array_list:
                yield (gidx_list[idx], data_list[idx], min_genome_array[idx], min_genome_bias_array[idx], min_genome_m1_array[idx], min_genome_m2_array[idx], min_genome_t_array[idx], min_genome_bias_m1_array[idx], min_genome_bias_m2_array[idx], subset_array_list[idx])
            else:
                yield (gidx_list[idx], data_list[idx], min_genome_array[idx], min_genome_bias_array[idx], min_genome_m1_array[idx], min_genome_m2_array[idx], min_genome_t_array[idx], min_genome_bias_m1_array[idx], min_genome_bias_m2_array[idx], None)
    else:
        for idx in xrange(num_elts):
            if subset_array_list:
                yield (gidx_list[idx], data_list[idx], genome_array[idx], genome_bias_array[idx], genome_m1_array[idx], genome_m2_array[idx], genome_t_array[idx], genome_bias_m1_array[idx], genome_bias_m2_array[idx], subset_array_list[idx])
            else:
                yield (gidx_list[idx], data_list[idx], genome_array[idx], genome_bias_array[idx], genome_m1_array[idx], genome_m2_array[idx], genome_t_array[idx], genome_bias_m1_array[idx], genome_bias_m2_array[idx], None)

def load_checkpoint(checkpoint_url):
    #load in the last saved model parameters
    if STORAGE == 'S3':
        checkpoint_url = checkpoint_url.replace('s3://', 's3n://')
        bucket_txt, key_txt = s3_library.parse_s3_url(checkpoint_url)
        key_dir = key_txt + '.factors'
        gtotal = sc._checkpointFile(checkpoint_url, AutoBatchedSerializer(PickleSerializer())).persist()
        gtotal.count()
        gmean = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'gmean.pickle'))
        ct = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct.pickle'))
        ct_bias = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_bias.pickle'))
        assay = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay.pickle'))
        assay_bias = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_bias.pickle'))
        ct_m1 = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_m1.pickle'))
        ct_m2 = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_m2.pickle'))
        ct_bias_m1 = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_bias_m1.pickle'))
        ct_bias_m2 = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_bias_m2.pickle'))
        assay_m1 = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_m1.pickle'))
        assay_m2 = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_m2.pickle'))
        assay_bias_m1 = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_bias_m1.pickle'))
        assay_bias_m2 = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_bias_m2.pickle'))
        ct_t = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_t.pickle'))
        assay_t = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_t.pickle'))
        iter_errs = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'iter_errs.pickle'))
        iters_to_test = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'iters_to_test.pickle'))
        test_res = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'test_res.pickle'))
        subsets = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'subsets.pickle'))
        rand_state = s3_library.get_pickle_s3(bucket_txt, os.path.join(key_dir, 'random_state.pickle'))
        min_idx = iter_errs.index(min(iter_errs, key=lambda x: x[MSE_VALID]))
        min_err = iter_errs[min_idx]
        key_min_gtotal = os.path.join(key_dir, 'min_gtotal.{!s}.rdd.pickle'.format(min_idx))
        url_min_gtotal = 's3://{!s}/{!s}'.format(bucket_txt, key_min_gtotal)
        min_factors_no_gtotal_key = os.path.join(key_dir, 'min_factors.{!s}.pickle'.format(min_idx))
        min_factors = s3_library.get_pickle_s3(bucket_txt, min_factors_no_gtotal_key)
        min_factors['gtotal'] = load_saved_rdd(url_min_gtotal.replace('s3://', 's3n://')).persist()
        min_factors['gtotal'].count()
    elif STORAGE == 'BLOB':
        container, blob = parse_azure_url(checkpoint_url)[:2]
        blob_dir = blob + '.factors'
        gtotal = sc._checkpointFile(checkpoint_url, AutoBatchedSerializer(PickleSerializer())).persist()
        gtotal.count()
        gmean = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'gmean.pickle'))
        ct = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'ct.pickle'))
        ct_bias = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'ct_bias.pickle'))
        assay = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'assay.pickle'))
        assay_bias = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'assay_bias.pickle'))
        ct_m1 = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'ct_m1.pickle'))
        ct_m2 = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'ct_m2.pickle'))
        ct_bias_m1 = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'ct_bias_m1.pickle'))
        ct_bias_m2 = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'ct_bias_m2.pickle'))
        assay_m1 = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'assay_m1.pickle'))
        assay_m2 = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'assay_m2.pickle'))
        assay_bias_m1 = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'assay_bias_m1.pickle'))
        assay_bias_m2 = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'assay_bias_m2.pickle'))
        ct_t = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'ct_t.pickle'))
        assay_t = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'assay_t.pickle'))
        iter_errs = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'iter_errs.pickle'))
        iters_to_test = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'iters_to_test.pickle'))
        test_res = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'test_res.pickle'))
        subsets = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'subsets.pickle'))
        rand_state = azure_library.get_blob_pickle(container, os.path.join(blob_dir, 'random_state.pickle'))
        min_idx = iter_errs.index(min(iter_errs, key=lambda x: x[MSE_VALID]))
        min_err = iter_errs[min_idx]
        blob_min_gtotal = os.path.join(blob_dir, 'min_gtotal.{!s}.rdd.pickle'.format(min_idx))
        url_min_gtotal = 'wasbs://{!s}@imputationstoretim.blob.core.windows.net/{!s}'.format(container, blob_min_gtotal)
        min_factors_no_gtotal_key = os.path.join(blob_dir, 'min_factors.{!s}.pickle'.format(min_idx))
        min_factors = azure_library.get_blob_pickle(container, min_factors_no_gtotal_key)
        min_factors['gtotal'] = load_saved_rdd(url_min_gtotal).persist()
        min_factors['gtotal'].count()
    return (gtotal, gmean, ct, ct_bias, assay, assay_bias, ct_m1, ct_m2, ct_bias_m1, ct_bias_m2, assay_m1, assay_m2, assay_bias_m1, assay_bias_m2, ct_t, assay_t, iter_errs, iters_to_test, test_res, min_err, min_factors, subsets, rand_state)

def make_random_subsets(gtotal_elt, num_subsets, seed=None):
    rs = numpy.random.RandomState(seed)
    data_coords_list = list(zip(*gtotal_elt[1].nonzero()))
    subset_size = int(round(len(data_coords_list)/float(num_subsets)))
    subset_coords = [data_coords_list[i:i+subset_size] for i in xrange(0, len(data_coords_list), subset_size)]
    subsets = []
    for sub_coords in subset_coords:
        subset = numpy.ones(gtotal_elt[1].shape, dtype=bool)
        subset[list(zip(*sub_coords))] = False
        subsets.append(subset)
    return gtotal_elt[:-1] + (subsets,)

def _calc_gmean_helper(data, subset):
    '''This function is used within a RDD.map() call to calculate the global
    mean of the data corresponding to subset.
    '''
    #first element is genome coordinate, so just take the second
    data = data[1]
    data_coords = numpy.zeros(data.shape, dtype=bool)
    data_coords[data.nonzero()] = True
    if subset is not None:
        subset_coords = numpy.logical_and(~subset, data_coords)
    else:
        subset_coords = data_coords
    datasum, datacount = numpy.sum(data[subset_coords]), numpy.sum(subset_coords)
    return datasum, datacount

def calc_gmean(pdata, subset=None):
    '''Run a map/reduce over the pdata RDD to calculate the global mean of the
    data points corresponding to subset.
    '''
    res = (pdata.map(lambda x: _calc_gmean_helper(x, subset)).
           reduce(lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2)))
    return numpy.divide(res[0], res[1])

def subtract_from_csr(csr, val):
    csr.data -= val
    return csr

def train_predictd(gtotal, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, learning_rate, run_bucket, out_root, iters_per_mse, batch_size, win_size, win_spacing, win2_shift, pval, lrate_search_num, beta1=0.9, beta2=0.999, epsilon=1e-8, init_seed=1, restart=False, suppress_output=False, min_iters=None, max_iters=None, checkpoint_interval=80, record_param_dist=None, checkpoint=None, subsets=None):
    '''Run training iterations for the 3D additive model.
    '''
    #set checkpoint dir
    if STORAGE == 'S3':
        sc.setCheckpointDir(os.path.join('s3n://{!s}'.format(run_bucket), out_root, 'checkpoints'))
        run_bucket = s3_library.S3.get_bucket(run_bucket)
    elif STORAGE == 'BLOB':
        sc.setCheckpointDir(os.path.join('wasbs://{!s}@imputationstoretim.blob.core.windows.net'.format(run_bucket), out_root, 'checkpoints'))

    if checkpoint:
        (gtotal, gmean, ct, ct_bias, assay, assay_bias, ct_m1, ct_m2, ct_bias_m1, ct_bias_m2, assay_m1, assay_m2, assay_bias_m1, assay_bias_m2, ct_t, assay_t, iter_errs, iters_to_test, test_res, min_err, min_factors, subsets, rand_state) = load_checkpoint(checkpoint)
        rs = numpy.RandomState()
        rs.set_state(rand_state)
        if iters_to_test:
            for _ in range(len(iters_to_test) - 1):
                learning_rate /= 2
                beta1 -= 1.0 - beta1
                lrate_search_num -= 1
        all_iters_count = int(round(((len(iter_errs) - 1) * iters_per_mse), -2)) + 1
    else:
        rs = numpy.random.RandomState(init_seed)
        if subsets is None:
            subset_seed = rs.randint(0, int(1e8))
            gtotal_subsets = gtotal.zipWithIndex().map(lambda (x,y): make_random_subsets(x, 2, seed=subset_seed + y)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
            gtotal_subsets.count()
            gtotal.unpersist()
            del(gtotal)
            gtotal = gtotal_subsets

        iter_errs = [calc_mse(gtotal, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, subsets=subsets)]
        min_err = iter_errs[0]
        ct_m1 = numpy.zeros(ct.shape)
        ct_m2 = numpy.zeros(ct.shape)
        ct_t = numpy.zeros(ct.shape[0])
        ct_bias_m1 = numpy.zeros(ct_bias.shape)
        ct_bias_m2 = numpy.zeros(ct_bias.shape)
        assay_m1 = numpy.zeros(assay.shape)
        assay_m2 = numpy.zeros(assay.shape)
        assay_t = numpy.zeros(assay.shape[0])
        assay_bias_m1 = numpy.zeros(assay_bias.shape)
        assay_bias_m2 = numpy.zeros(assay_bias.shape)
        iters_to_test = []
        test_res = []
        min_factors = {'ct':ct.copy(), 'ct_bias':ct_bias.copy(),
                       'assay':assay.copy(), 'assay_bias':assay_bias.copy(),
                       'gtotal':gtotal, 'min_idx':0, 'min_checkpoint':'',
                       'ct_m1':ct_m1.copy(), 'ct_m2':ct_m2.copy(), 'ct_bias_m1':ct_bias_m1.copy(),
                       'ct_bias_m2':ct_bias_m2.copy(), 'assay_m1':assay_m1.copy(),
                       'assay_m2':assay_m2.copy(), 'assay_bias_m1':assay_bias_m1.copy(),
                       'assay_bias_m2':assay_bias_m2.copy(), 'ct_t':ct_t.copy(),
                       'assay_t':assay_t.copy()}

        #burn in ct and assay parameters
        print('Burning in the model on a small subset of highly-weighted loci.')
        all_iters_count = 0
        param = UpdateAccumulatorParam()
        ct_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_bias_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_m1_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_m2_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_t_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m1_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m2_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        assay_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
        assay_bias_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        assay_m1_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
        assay_m2_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
        assay_t_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        assay_bias_m1_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        assay_bias_m2_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        sample_frac = min(8000.0/gtotal.count(), 1.0)
        burn_in = gtotal.sample(False, sample_frac, init_seed).repartition(1).persist()
        burn_in_count = burn_in.count()
        print('Burn in locus count: {!s}'.format(burn_in_count))
        burn_in_epochs = 0.5
        if subsets is None:
            burn_in_batch_size = burn_in_epochs * burn_in.map(lambda x: numpy.sum(~x[-1][SUBSET_TRAIN]) if x[-1] is not None else len(x[1].nonzero()[0])).sum()
        else:
            burn_in_batch_size = burn_in_epochs * numpy.sum(~subsets[SUBSET_TRAIN]) * burn_in_count
        data_iteration_seed = rs.randint(0, int(1e8))
        burn_in.mapPartitionsWithIndex(lambda x,y: one_iteration(x, y, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, learning_rate, burn_in_batch_size, data_iteration_seed, ct_accum, ct_bias_accum, assay_accum, assay_bias_accum, ct_m1, ct_m2, ct_t, ct_m1_accum, ct_m2_accum, ct_t_accum, ct_bias_m1, ct_bias_m2, ct_bias_m1_accum, ct_bias_m2_accum, assay_m1, assay_m2, assay_t, assay_m1_accum, assay_m2_accum, assay_t_accum, assay_bias_m1, assay_bias_m2, assay_bias_m1_accum, assay_bias_m2_accum, beta1=beta1, beta2=beta2, epsilon=epsilon, subsets=subsets)).count()
        ct = numpy.divide(ct_accum.value[0], ct_accum.value[1])
        print('{!s} Num partitions by ct_accum: {!s}'.format(all_iters_count, ct_accum.value[1]))
        ct_m1 = numpy.divide(ct_m1_accum.value[0], ct_m1_accum.value[1])
        ct_m2 = numpy.divide(ct_m2_accum.value[0], ct_m2_accum.value[1])
        ct_t = ct_t + numpy.array(numpy.around(numpy.divide(ct_t_accum.value[0], ct_t_accum.value[1])), dtype=int)
        ct_bias = numpy.divide(ct_bias_accum.value[0], ct_bias_accum.value[1])
        ct_bias_m1 = numpy.divide(ct_bias_m1_accum.value[0], ct_bias_m1_accum.value[1])
        ct_bias_m2 = numpy.divide(ct_bias_m2_accum.value[0], ct_bias_m2_accum.value[1])
        assay = numpy.divide(assay_accum.value[0], assay_accum.value[1])
        assay_m1 = numpy.divide(assay_m1_accum.value[0], assay_m1_accum.value[1])
        assay_m2 = numpy.divide(assay_m2_accum.value[0], assay_m2_accum.value[1])
        assay_t = assay_t + numpy.array(numpy.around(numpy.divide(assay_t_accum.value[0], assay_t_accum.value[1])), dtype=int)
        assay_bias = numpy.divide(assay_bias_accum.value[0], assay_bias_accum.value[1])
        assay_bias_m1 = numpy.divide(assay_bias_m1_accum.value[0], assay_bias_m1_accum.value[1])
        assay_bias_m2 = numpy.divide(assay_bias_m2_accum.value[0], assay_bias_m2_accum.value[1])
        burn_in.unpersist()
        del(burn_in)
        if subsets is None:
            genome_batch_size = burn_in_epochs * gtotal.map(lambda x: numpy.sum(~x[-1][SUBSET_TRAIN]) if x[-1] is not None else len(x[1].nonzero()[0])).sum()
        else:
            genome_batch_size = burn_in_epochs * numpy.sum(~subsets[SUBSET_TRAIN]) * (int(gtotal.count())/gtotal.getNumPartitions())
        gtotal_tmp = gtotal.mapPartitionsWithIndex(lambda x,y: sgd_genome_only(x, y,  ct, ct_bias, assay, assay_bias, ri, rbi, learning_rate, genome_batch_size, all_iters_count + 1, beta1=beta1, beta2=beta2, epsilon=epsilon, auto_convergence_detection=False, subsets=subsets)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
        gtotal_tmp.count()
        gtotal.unpersist()
#        del(genome)
        gtotal = gtotal_tmp
        print('Burn in complete; continuing training on whole model.')

#    #the stopping criterion is based on two sliding windows with a separation
#    #window size and separation is based on a multiple of the total number of
#    #data points in the training set
#    win_size = args.stop_winsize
#    win_spacing = args.stop_winspacing
#    win2_shift = args.stop_win2shift
#    pval = args.stop_pval

    cur_checkpoint = None
    checkpoint_to_delete = None
    #Train the model
    while True:
        if not (all_iters_count/float(iters_per_mse)) % 10:
#            if STORAGE == 'S3':
#                out_dir = 's3://' + os.path.join(args.run_bucket, args.out_root, 'param_plots')
#            elif STORAGE == 'BLOB':
#                out_dir = 'wasbs://{!s}@imputationstoretim.blob.core.windows.net/{!s}'.format(args.run_bucket, os.path.join(args.out_root, 'param_plots'))
#            plot_parameter_dist(ct, ct_bias, assay, assay_bias, gtotal, out_dir, all_iters_count, genome_frac=0.1)
            out_key = os.path.join(out_root, 'iter_errs.txt')
            print_iter_errs(iter_errs, run_bucket, out_key)
        param = UpdateAccumulatorParam()
        ct_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_bias_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_m1_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_m2_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_t_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m1_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m2_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        assay_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
        assay_bias_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        assay_m1_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
        assay_m2_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
        assay_t_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        assay_bias_m1_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        assay_bias_m2_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        data_iteration_seed = rs.randint(0,int(1e8))
        gtotal_tmp = gtotal.mapPartitionsWithIndex(lambda x,y: one_iteration(x, y, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, learning_rate, batch_size, data_iteration_seed, ct_accum, ct_bias_accum, assay_accum, assay_bias_accum, ct_m1, ct_m2, ct_t, ct_m1_accum, ct_m2_accum, ct_t_accum, ct_bias_m1, ct_bias_m2, ct_bias_m1_accum, ct_bias_m2_accum, assay_m1, assay_m2, assay_t, assay_m1_accum, assay_m2_accum, assay_t_accum, assay_bias_m1, assay_bias_m2, assay_bias_m1_accum, assay_bias_m2_accum, beta1=beta1, beta2=beta2, epsilon=epsilon, subsets=subsets)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
        gtotal_tmp.setName('gtotal' + str(all_iters_count))
        if all_iters_count > 1 and not all_iters_count % checkpoint_interval:
            #checkpoint current gtotal
            gtotal_tmp.checkpoint()
            gtotal_tmp.count()
            if STORAGE == 'S3':
                cur_checkpoint = gtotal_tmp.getCheckpointFile().replace('s3n://', 's3://')
            elif STORAGE == 'BLOB':
                cur_checkpoint = gtotal_tmp.getCheckpointFile()
            print('Just got new checkpoint after iteration {!s}: {!s}'.format(all_iters_count, cur_checkpoint))
        else:
            gtotal_tmp.count()
        if gtotal.name() != min_factors['gtotal'].name():
            gtotal.unpersist()
            del(gtotal)
        gtotal = gtotal_tmp

        ct = numpy.divide(ct_accum.value[0], ct_accum.value[1])
        ct_m1 = numpy.divide(ct_m1_accum.value[0], ct_m1_accum.value[1])
        ct_m2 = numpy.divide(ct_m2_accum.value[0], ct_m2_accum.value[1])
        ct_t = ct_t + numpy.array(numpy.around(numpy.divide(ct_t_accum.value[0], ct_t_accum.value[1])), dtype=int)
        ct_bias = numpy.divide(ct_bias_accum.value[0], ct_bias_accum.value[1])
        ct_bias_m1 = numpy.divide(ct_bias_m1_accum.value[0], ct_bias_m1_accum.value[1])
        ct_bias_m2 = numpy.divide(ct_bias_m2_accum.value[0], ct_bias_m2_accum.value[1])
        assay = numpy.divide(assay_accum.value[0], assay_accum.value[1])
        assay_m1 = numpy.divide(assay_m1_accum.value[0], assay_m1_accum.value[1])
        assay_m2 = numpy.divide(assay_m2_accum.value[0], assay_m2_accum.value[1])
        assay_t = assay_t + numpy.array(numpy.around(numpy.divide(assay_t_accum.value[0], assay_t_accum.value[1])), dtype=int)
        assay_bias = numpy.divide(assay_bias_accum.value[0], assay_bias_accum.value[1])
        assay_bias_m1 = numpy.divide(assay_bias_m1_accum.value[0], assay_bias_m1_accum.value[1])
        assay_bias_m2 = numpy.divide(assay_bias_m2_accum.value[0], assay_bias_m2_accum.value[1])

        print('{!s} Num partitions by ct_accum: {!s}, Mean ct_t: {!s}, Mean assay_t: {!s}'.format(all_iters_count, ct_accum.value[1], numpy.mean(ct_t), numpy.mean(assay_t)))

        #if the model really isn't moving, bail and don't even bother with the statistical test
        if all_iters_count == 1000 and iter_errs[0][MSE_TRAIN] - iter_errs[-1][MSE_TRAIN] < 0.005:
            break

        if not all_iters_count % iters_per_mse:
            #check how we did
            print('Calculating MSE based on all samples.')
            mse = calc_mse(gtotal, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, subsets=subsets)
            if numpy.any(numpy.isnan(mse)):
                print('Got NaN in {!s} MSE result. Breaking and returning current min_mse after {!s} iterations'.format(sname, all_iters_count))
                break
            print(mse)
            iter_errs.append(copy.copy(mse))

            if mse[MSE_VALID] < min_err[MSE_VALID]:
                print('NEW MIN MSE: {!s} < {!s}'.format(mse[MSE_VALID], min_err[MSE_VALID]))
                min_err = mse
                min_factors['gtotal'].unpersist()
                del(min_factors['gtotal'])
                if min_factors['min_checkpoint'] != cur_checkpoint:
                    checkpoint_to_delete = min_factors['min_checkpoint']
                min_factors = {'ct':ct.copy(), 'ct_bias': ct_bias.copy(),
                               'assay':assay.copy(), 'assay_bias':assay_bias.copy(),
                               'gtotal':gtotal, 'min_idx':len(iter_errs) - 1,
                               'ct_m1':ct_m1.copy(),
                               'ct_m2':ct_m2.copy(), 'ct_bias_m1':ct_bias_m1.copy(),
                               'ct_bias_m2':ct_bias_m2.copy(), 'assay_m1':assay_m1.copy(),
                               'assay_m2':assay_m2.copy(), 'assay_bias_m1':assay_bias_m1.copy(),
                               'assay_bias_m2':assay_bias_m2.copy(), 'ct_t':ct_t.copy(),
                               'assay_t':assay_t.copy(), 'min_checkpoint':cur_checkpoint}
            #collecting data for a smarter stopping criterion based on detecting
            #convergence
            if ((max_iters and len(iter_errs) > max_iters) or
                (len(iter_errs) - min_factors['min_idx']) > (2 * win_size + win_spacing)):
                break
            if ((not min_iters or len(iter_errs) > min_iters) and
                len(iter_errs) - min_factors['min_idx'] > win_size):
                if len(iter_errs) > (2 * win_size + win_spacing):
                    if iters_to_test:
                        to_test = list(itertools.chain(*[iter_errs[elt] for elt in iters_to_test]))
                    else:
                        to_test = iter_errs
                    start1 = len(to_test) - (2 * win_size + win_spacing)
                    stop1 = start1 + win_size
                    start2 = stop1 + win_spacing
                    win1 = numpy.array([elt[MSE_VALID] for elt in to_test[start1:stop1]])
                    win2 = numpy.array([elt[MSE_VALID] for elt in to_test[start2:]])
                    test = ranksums(win1, win2 + win2_shift)
                    test_res.append(test)
                    if test[0] < 0 and test[1]/2 < pval:
                        if lrate_search_num:
                            learning_rate /= 2
                            beta1 -= 1.0 - beta1
                            ct = min_factors['ct']
                            ct_bias = min_factors['ct_bias']
                            assay = min_factors['assay']
                            assay_bias = min_factors['assay_bias']
                            ct_m1 = min_factors['ct_m1']
                            ct_m2 = min_factors['ct_m2']
                            ct_bias_m1 = min_factors['ct_bias_m1']
                            ct_bias_m2 = min_factors['ct_bias_m2']
                            assay_m1 = min_factors['assay_m1']
                            assay_m2 = min_factors['assay_m2']
                            assay_bias_m1 = min_factors['assay_bias_m1']
                            asssay_bias_m2 = min_factors['assay_bias_m2']
                            ct_t = min_factors['ct_t']
                            assay_t = min_factors['assay_t']
                            cur_checkpoint = min_factors['min_checkpoint']
                            if min_factors['gtotal'].name() != gtotal.name():
                                gtotal.unpersist()
                                del(gtotal)
                                gtotal = min_factors['gtotal']
                            lrate_search_num -= 1
                            print('Validation error has stopped improving. Halving learning rate: {!s}. Will do '
                                  'so {!s} more times'.format(learning_rate, lrate_search_num))
                            #if we found a new minimum since the last lrate decrease
                            if iters_to_test and min_factors['min_idx'] > iters_to_test[-1].start:
                                iters_to_test[-1] = slice(iters_to_test[-1].start, min_factors['min_idx'] + 1)
                                iters_to_test.append(slice(len(iter_errs) - 1, None))
                            #if we did not find a new minimum since the last lrate decrease
                            elif iters_to_test:
                                iters_to_test[-1] = slice(len(iter_errs) - 1, None)
                            #if this is the first time we are decreasing the lrate
                            else:
                                iters_to_test = [slice(0,min_factors['min_idx'] + 1), slice(len(iter_errs) - 1, None)]
        if (not suppress_output) and (all_iters_count > 0) and (not all_iters_count % checkpoint_interval):
            #save the state of the rest of the model
            if STORAGE == 'S3':
                bucket_txt, key_txt = s3_library.parse_s3_url(cur_checkpoint)
                key_dir = key_txt + '.factors'
#                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'gmean.pickle'), gmean)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct.pickle'), ct)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_bias.pickle'), ct_bias)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay.pickle'), assay)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_bias.pickle'), assay_bias)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_m1.pickle'), ct_m1)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_m2.pickle'), ct_m2)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_bias_m1.pickle'), ct_bias_m1)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_bias_m2.pickle'), ct_bias_m2)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_m1.pickle'), assay_m1)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_m2.pickle'), assay_m2)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_bias_m1.pickle'), assay_bias_m1)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_bias_m2.pickle'), assay_bias_m2)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_t.pickle'), ct_t)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_t.pickle'), assay_t)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'iter_errs.pickle'), iter_errs)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'iters_to_test.pickle'), iters_to_test)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'test_res.pickle'), test_res)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'subsets.pickle'), subsets)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'random_state.pickle'), rs.get_state())
                key_min_gtotal = os.path.join(key_dir, 'min_gtotal.{!s}.rdd.pickle'.format(min_factors['min_idx']))
                url_min_gtotal = 's3://{!s}/{!s}'.format(bucket_txt, key_min_gtotal)
                if not len(s3_library.glob_keys(bucket_txt, os.path.join(key_min_gtotal, '*'))):
                    #now, save the current minimum
                    save_rdd_as_pickle(min_factors['gtotal'], url_min_gtotal)
                    min_factors_no_gtotal = {k:v for k,v in min_factors.items() if k != 'gtotal'}
                    min_factors_no_gtotal_key = os.path.join(key_dir, 'min_factors.{!s}.pickle'.
                                                             format(min_factors['min_idx']))
                    s3_library.set_pickle_s3(bucket_txt, min_factors_no_gtotal_key, min_factors_no_gtotal)
                    del(min_factors_no_gtotal)

            elif STORAGE == 'BLOB':
                container, blob = parse_azure_url(cur_checkpoint)[:2]
                blob_dir = blob + '.factors'
#                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'gmean.pickle'), gmean)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct.pickle'), ct)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_bias.pickle'), ct_bias)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay.pickle'), assay)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_bias.pickle'), assay_bias)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_m1.pickle'), ct_m1)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_m2.pickle'), ct_m2)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_bias_m1.pickle'), ct_bias_m1)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_bias_m2.pickle'), ct_bias_m2)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_m1.pickle'), assay_m1)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_m2.pickle'), assay_m2)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_bias_m1.pickle'), assay_bias_m1)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_bias_m2.pickle'), assay_bias_m2)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_t.pickle'), ct_t)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_t.pickle'), assay_t)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'iter_errs.pickle'), iter_errs)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'iters_to_test.pickle'), iters_to_test)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'test_res.pickle'), test_res)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'subsets.pickle'), subsets)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'random_state.pickle'), rs.get_state())
                blob_min_gtotal = os.path.join(blob_dir, 'min_gtotal.{!s}.rdd.pickle'.format(min_factors['min_idx']))
                url_min_gtotal = 'wasbs://{!s}@imputationstoretim.blob.core.windows.net/{!s}'.format(container, blob_min_gtotal)
                if not len(azure_library.glob_blobs(container, os.path.join(blob_min_gtotal, '*'))):
                    #now, save the current minimum
                    save_rdd_as_pickle(min_factors['gtotal'], url_min_gtotal)
                    min_factors_no_gtotal = {k:v for k,v in min_factors.items() if k != 'gtotal'}
                    min_factors_no_gtotal_blob = os.path.join(blob_dir, 'min_factors.{!s}.pickle'.
                                                              format(min_factors['min_idx']))
                    azure_library.load_blob_pickle(container, min_factors_no_gtotal_blob, min_factors_no_gtotal)
                    del(min_factors_no_gtotal)
            #If a new minimum was found in the just-saved checkpoint, then delete the previous minimum
            if checkpoint_to_delete:
                if STORAGE == 'S3':
                    cmd = ['aws', 's3', 'rm', '--quiet', '--recursive', checkpoint_to_delete]
                    subprocess.check_call(cmd)
                    #remove associated parameters
                    cmd = ['aws', 's3', 'rm', '--quiet', '--recursive', checkpoint_to_delete.rstrip('/') + '.factors/']
                    subprocess.check_call(cmd)
                elif STORAGE == 'BLOB':
                    container, blob = parse_azure_url(checkpoint_to_delete)[:2]
                    azure_library.delete_blob_dir(container, blob)
                    #remove associated parameters
                    azure_library.delete_blob_dir(container, blob.rstrip('/') + '.factors/')
                checkpoint_to_delete = None
                
        all_iters_count += 1
    return min_factors['gtotal'], min_factors['ct'], min_factors['ct_bias'], min_factors['assay'], min_factors['assay_bias'], iter_errs

def train_predictd_ct_genome(gtotal, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, learning_rate, run_bucket, out_root, iters_per_mse, batch_size, win_size, win_spacing, win2_shift, pval, lrate_search_num, beta1=0.9, beta2=0.999, epsilon=1e-8, init_seed=1, restart=False, suppress_output=False, min_iters=None, max_iters=None, checkpoint_interval=80, record_param_dist=None, checkpoint=None, subsets=None):
    '''Run training iterations for the 3D additive model.
    '''
    #set checkpoint dir
    if STORAGE == 'S3':
        sc.setCheckpointDir(os.path.join('s3n://{!s}'.format(run_bucket), out_root, 'checkpoints'))
        run_bucket = s3_library.S3.get_bucket(run_bucket)
    elif STORAGE == 'BLOB':
        sc.setCheckpointDir(os.path.join('wasbs://{!s}@imputationstoretim.blob.core.windows.net'.format(run_bucket), out_root, 'checkpoints'))

    if checkpoint:
        (gtotal, gmean, ct, ct_bias, assay, assay_bias, ct_m1, ct_m2, ct_bias_m1, ct_bias_m2, assay_m1, assay_m2, assay_bias_m1, assay_bias_m2, ct_t, assay_t, iter_errs, iters_to_test, test_res, min_err, min_factors, subsets, rand_state) = load_checkpoint(checkpoint)
        rs = numpy.RandomState()
        rs.set_state(rand_state)
        if iters_to_test:
            for _ in range(len(iters_to_test) - 1):
                learning_rate /= 2
                beta1 -= 1.0 - beta1
                lrate_search_num -= 1
        all_iters_count = int(round(((len(iter_errs) - 1) * iters_per_mse), -2)) + 1
    else:
        rs = numpy.random.RandomState(init_seed)
        if subsets is None:
            subset_seed = rs.randint(0, int(1e8))
            gtotal_subsets = gtotal.zipWithIndex().map(lambda (x,y): make_random_subsets(x, 2, seed=subset_seed + y)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
            gtotal_subsets.count()
            gtotal.unpersist()
            del(gtotal)
            gtotal = gtotal_subsets

        iter_errs = [calc_mse(gtotal, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, subsets=subsets)]
        min_err = iter_errs[0]
        ct_m1 = numpy.zeros(ct.shape)
        ct_m2 = numpy.zeros(ct.shape)
        ct_t = numpy.zeros(ct.shape[0])
        ct_bias_m1 = numpy.zeros(ct_bias.shape)
        ct_bias_m2 = numpy.zeros(ct_bias.shape)
        assay_m1 = numpy.zeros(assay.shape)
        assay_m2 = numpy.zeros(assay.shape)
        assay_t = numpy.zeros(assay.shape[0])
        assay_bias_m1 = numpy.zeros(assay_bias.shape)
        assay_bias_m2 = numpy.zeros(assay_bias.shape)
        iters_to_test = []
        test_res = []
        min_factors = {'ct':ct.copy(), 'ct_bias':ct_bias.copy(),
                       'assay':assay.copy(), 'assay_bias':assay_bias.copy(),
                       'gtotal':gtotal, 'min_idx':0, 'min_checkpoint':'',
                       'ct_m1':ct_m1.copy(), 'ct_m2':ct_m2.copy(), 'ct_bias_m1':ct_bias_m1.copy(),
                       'ct_bias_m2':ct_bias_m2.copy(), 'assay_m1':assay_m1.copy(),
                       'assay_m2':assay_m2.copy(), 'assay_bias_m1':assay_bias_m1.copy(),
                       'assay_bias_m2':assay_bias_m2.copy(), 'ct_t':ct_t.copy(),
                       'assay_t':assay_t.copy()}

        #burn in ct and assay parameters
        print('Burning in the model on a small subset of highly-weighted loci.')
        all_iters_count = 0
        param = UpdateAccumulatorParam()
        ct_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_bias_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_m1_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_m2_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_t_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m1_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m2_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
#        assay_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
#        assay_bias_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
#        assay_m1_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
#        assay_m2_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
#        assay_t_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
#        assay_bias_m1_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
#        assay_bias_m2_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        sample_frac = min(8000.0/gtotal.count(), 1.0)
        burn_in = gtotal.sample(False, sample_frac, init_seed).repartition(1).persist()
        burn_in_count = burn_in.count()
        print('Burn in locus count: {!s}'.format(burn_in_count))
        #burn in for one epoch
        if subsets is None:
            burn_in_batch_size = 1.05 * burn_in.map(lambda x: numpy.sum(~x[-1][SUBSET_TRAIN]) if x[-1] is not None else len(x[1].nonzero()[0])).sum()
        else:
            burn_in_batch_size = 1.05 * numpy.sum(~subsets[SUBSET_TRAIN]) * burn_in_count
        data_iteration_seed = rs.randint(0, int(1e8))
        burn_in.mapPartitionsWithIndex(lambda x,y: one_iteration_ct_genome(x, y, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, learning_rate, burn_in_batch_size, data_iteration_seed, ct_accum, ct_bias_accum, ct_m1, ct_m2, ct_t, ct_m1_accum, ct_m2_accum, ct_t_accum, ct_bias_m1, ct_bias_m2, ct_bias_m1_accum, ct_bias_m2_accum, beta1=beta1, beta2=beta2, epsilon=epsilon, subsets=subsets)).count()
        ct = numpy.divide(ct_accum.value[0], ct_accum.value[1])
        print('{!s} Num partitions by ct_accum: {!s}'.format(all_iters_count, ct_accum.value[1]))
        ct_m1 = numpy.divide(ct_m1_accum.value[0], ct_m1_accum.value[1])
        ct_m2 = numpy.divide(ct_m2_accum.value[0], ct_m2_accum.value[1])
        ct_t = ct_t + numpy.array(numpy.around(numpy.divide(ct_t_accum.value[0], ct_t_accum.value[1])), dtype=int)
        ct_bias = numpy.divide(ct_bias_accum.value[0], ct_bias_accum.value[1])
        ct_bias_m1 = numpy.divide(ct_bias_m1_accum.value[0], ct_bias_m1_accum.value[1])
        ct_bias_m2 = numpy.divide(ct_bias_m2_accum.value[0], ct_bias_m2_accum.value[1])
#        assay = numpy.divide(assay_accum.value[0], assay_accum.value[1])
#        assay_m1 = numpy.divide(assay_m1_accum.value[0], assay_m1_accum.value[1])
#        assay_m2 = numpy.divide(assay_m2_accum.value[0], assay_m2_accum.value[1])
#        assay_t = assay_t + numpy.array(numpy.around(numpy.divide(assay_t_accum.value[0], assay_t_accum.value[1])), dtype=int)
#        assay_bias = numpy.divide(assay_bias_accum.value[0], assay_bias_accum.value[1])
#        assay_bias_m1 = numpy.divide(assay_bias_m1_accum.value[0], assay_bias_m1_accum.value[1])
#        assay_bias_m2 = numpy.divide(assay_bias_m2_accum.value[0], assay_bias_m2_accum.value[1])
        burn_in.unpersist()
        del(burn_in)
        if subsets is None:
            genome_batch_size = 1.05 * gtotal.map(lambda x: numpy.sum(~x[-1][SUBSET_TRAIN]) if x[-1] is not None else len(x[1].nonzero()[0])).sum()
        else:
            genome_batch_size = 1.05 * numpy.sum(~subsets[SUBSET_TRAIN]) * (int(gtotal.count())/gtotal.getNumPartitions())
        gtotal_tmp = gtotal.mapPartitionsWithIndex(lambda x,y: sgd_genome_only(x, y,  ct, ct_bias, assay, assay_bias, ri, rbi, learning_rate, genome_batch_size, all_iters_count + 1, beta1=beta1, beta2=beta2, epsilon=epsilon, auto_convergence_detection=False, subsets=subsets)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
        gtotal_tmp.count()
        gtotal.unpersist()
#        del(genome)
        gtotal = gtotal_tmp
        print('Burn in complete; continuing training on whole model.')

#    #the stopping criterion is based on two sliding windows with a separation
#    #window size and separation is based on a multiple of the total number of
#    #data points in the training set
#    win_size = args.stop_winsize
#    win_spacing = args.stop_winspacing
#    win2_shift = args.stop_win2shift
#    pval = args.stop_pval

    cur_checkpoint = None
    checkpoint_to_delete = None
    #Train the model
    while True:
        if not (all_iters_count/float(iters_per_mse)) % 10:
#            if STORAGE == 'S3':
#                out_dir = 's3://' + os.path.join(args.run_bucket, args.out_root, 'param_plots')
#            elif STORAGE == 'BLOB':
#                out_dir = 'wasbs://{!s}@imputationstoretim.blob.core.windows.net/{!s}'.format(args.run_bucket, os.path.join(args.out_root, 'param_plots'))
#            plot_parameter_dist(ct, ct_bias, assay, assay_bias, gtotal, out_dir, all_iters_count, genome_frac=0.1)
            out_key = os.path.join(out_root, 'iter_errs.txt')
            print_iter_errs(iter_errs, run_bucket, out_key)
        param = UpdateAccumulatorParam()
        ct_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_bias_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_m1_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_m2_accum = sc.accumulator(param.zero([ct,1]), UpdateAccumulatorParam())
        ct_t_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m1_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
        ct_bias_m2_accum = sc.accumulator(param.zero([ct_bias,1]), UpdateAccumulatorParam())
#        assay_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
#        assay_bias_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
#        assay_m1_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
#        assay_m2_accum = sc.accumulator(param.zero([assay,1]), UpdateAccumulatorParam())
#        assay_t_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
#        assay_bias_m1_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
#        assay_bias_m2_accum = sc.accumulator(param.zero([assay_bias,1]), UpdateAccumulatorParam())
        data_iteration_seed = rs.randint(0,int(1e8))
        gtotal_tmp = gtotal.mapPartitionsWithIndex(lambda x,y: one_iteration_ct_genome(x, y, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, learning_rate, batch_size, data_iteration_seed, ct_accum, ct_bias_accum, ct_m1, ct_m2, ct_t, ct_m1_accum, ct_m2_accum, ct_t_accum, ct_bias_m1, ct_bias_m2, ct_bias_m1_accum, ct_bias_m2_accum, beta1=beta1, beta2=beta2, epsilon=epsilon, subsets=subsets)).persist(storageLevel=StorageLevel.MEMORY_AND_DISK_SER)
        gtotal_tmp.setName('gtotal' + str(all_iters_count))
        if all_iters_count > 1 and not all_iters_count % checkpoint_interval:
            #checkpoint current gtotal
            gtotal_tmp.checkpoint()
            gtotal_tmp.count()
            if STORAGE == 'S3':
                cur_checkpoint = gtotal_tmp.getCheckpointFile().replace('s3n://', 's3://')
            elif STORAGE == 'BLOB':
                cur_checkpoint = gtotal_tmp.getCheckpointFile()
            print('Just got new checkpoint after iteration {!s}: {!s}'.format(all_iters_count, cur_checkpoint))
        else:
            gtotal_tmp.count()
        if gtotal.name() != min_factors['gtotal'].name():
            gtotal.unpersist()
            del(gtotal)
        gtotal = gtotal_tmp

        ct = numpy.divide(ct_accum.value[0], ct_accum.value[1])
        ct_m1 = numpy.divide(ct_m1_accum.value[0], ct_m1_accum.value[1])
        ct_m2 = numpy.divide(ct_m2_accum.value[0], ct_m2_accum.value[1])
        ct_t = ct_t + numpy.array(numpy.around(numpy.divide(ct_t_accum.value[0], ct_t_accum.value[1])), dtype=int)
        ct_bias = numpy.divide(ct_bias_accum.value[0], ct_bias_accum.value[1])
        ct_bias_m1 = numpy.divide(ct_bias_m1_accum.value[0], ct_bias_m1_accum.value[1])
        ct_bias_m2 = numpy.divide(ct_bias_m2_accum.value[0], ct_bias_m2_accum.value[1])
#        assay = numpy.divide(assay_accum.value[0], assay_accum.value[1])
#        assay_m1 = numpy.divide(assay_m1_accum.value[0], assay_m1_accum.value[1])
#        assay_m2 = numpy.divide(assay_m2_accum.value[0], assay_m2_accum.value[1])
#        assay_t = assay_t + numpy.array(numpy.around(numpy.divide(assay_t_accum.value[0], assay_t_accum.value[1])), dtype=int)
#        assay_bias = numpy.divide(assay_bias_accum.value[0], assay_bias_accum.value[1])
#        assay_bias_m1 = numpy.divide(assay_bias_m1_accum.value[0], assay_bias_m1_accum.value[1])
#        assay_bias_m2 = numpy.divide(assay_bias_m2_accum.value[0], assay_bias_m2_accum.value[1])

        print('{!s} Num partitions by ct_accum: {!s}, Mean ct_t: {!s}, Mean assay_t: {!s}'.format(all_iters_count, ct_accum.value[1], numpy.mean(ct_t), numpy.mean(assay_t)))

        #if the model really isn't moving, bail and don't even bother with the statistical test
        if all_iters_count == 1000 and iter_errs[0][MSE_TRAIN] - iter_errs[-1][MSE_TRAIN] < 0.02:
            break

        if not all_iters_count % iters_per_mse:
            #check how we did
            print('Calculating MSE based on all samples.')
            mse = calc_mse(gtotal, ct, rc, ct_bias, rbc, assay, ra, assay_bias, rba, ri, rbi, subsets=subsets)
            if numpy.any(numpy.isnan(mse)):
                print('Got NaN in {!s} MSE result. Breaking and returning current min_mse after {!s} iterations'.format(sname, all_iters_count))
                break
            print(mse)
            iter_errs.append(copy.copy(mse))

            if mse[MSE_VALID] < min_err[MSE_VALID]:
                print('NEW MIN MSE: {!s} < {!s}'.format(mse[MSE_VALID], min_err[MSE_VALID]))
                min_err = mse
                min_factors['gtotal'].unpersist()
                del(min_factors['gtotal'])
                if min_factors['min_checkpoint'] != cur_checkpoint:
                    checkpoint_to_delete = min_factors['min_checkpoint']
                min_factors = {'ct':ct.copy(), 'ct_bias': ct_bias.copy(),
                               'assay':assay.copy(), 'assay_bias':assay_bias.copy(),
                               'gtotal':gtotal, 'min_idx':len(iter_errs) - 1,
                               'ct_m1':ct_m1.copy(),
                               'ct_m2':ct_m2.copy(), 'ct_bias_m1':ct_bias_m1.copy(),
                               'ct_bias_m2':ct_bias_m2.copy(), 'assay_m1':assay_m1.copy(),
                               'assay_m2':assay_m2.copy(), 'assay_bias_m1':assay_bias_m1.copy(),
                               'assay_bias_m2':assay_bias_m2.copy(), 'ct_t':ct_t.copy(),
                               'assay_t':assay_t.copy(), 'min_checkpoint':cur_checkpoint}
            #collecting data for a smarter stopping criterion based on detecting
            #convergence
            if max_iters and len(iter_errs) > max_iters:
                break
            if ((not min_iters or len(iter_errs) > min_iters) and
                len(iter_errs) - min_factors['min_idx'] > win_size):
                if len(iter_errs) > (2 * win_size + win_spacing):
                    if iters_to_test:
                        to_test = list(itertools.chain(*[iter_errs[elt] for elt in iters_to_test]))
                    else:
                        to_test = iter_errs
                    start1 = len(to_test) - (2 * win_size + win_spacing)
                    stop1 = start1 + win_size
                    start2 = stop1 + win_spacing
                    win1 = numpy.array([elt[MSE_VALID] for elt in to_test[start1:stop1]])
                    win2 = numpy.array([elt[MSE_VALID] for elt in to_test[start2:]])
                    test = ranksums(win1, win2 + win2_shift)
                    test_res.append(test)
                    if test[0] < 0 and test[1]/2 < pval:
                        if lrate_search_num:
                            learning_rate /= 2
                            beta1 -= 1.0 - beta1
                            ct = min_factors['ct']
                            ct_bias = min_factors['ct_bias']
                            assay = min_factors['assay']
                            assay_bias = min_factors['assay_bias']
                            ct_m1 = min_factors['ct_m1']
                            ct_m2 = min_factors['ct_m2']
                            ct_bias_m1 = min_factors['ct_bias_m1']
                            ct_bias_m2 = min_factors['ct_bias_m2']
                            assay_m1 = min_factors['assay_m1']
                            assay_m2 = min_factors['assay_m2']
                            assay_bias_m1 = min_factors['assay_bias_m1']
                            asssay_bias_m2 = min_factors['assay_bias_m2']
                            ct_t = min_factors['ct_t']
                            assay_t = min_factors['assay_t']
                            cur_checkpoint = min_factors['min_checkpoint']
                            if min_factors['gtotal'].name() != gtotal.name():
                                gtotal.unpersist()
                                del(gtotal)
                                gtotal = min_factors['gtotal']
                            lrate_search_num -= 1
                            print('Validation error has stopped improving. Halving learning rate: {!s}. Will do '
                                  'so {!s} more times'.format(learning_rate, lrate_search_num))
                            #if we found a new minimum since the last lrate decrease
                            if iters_to_test and min_factors['min_idx'] > iters_to_test[-1].start:
                                iters_to_test[-1] = slice(iters_to_test[-1].start, min_factors['min_idx'] + 1)
                                iters_to_test.append(slice(len(iter_errs) - 1, None))
                            #if we did not find a new minimum since the last lrate decrease
                            elif iters_to_test:
                                iters_to_test[-1] = slice(len(iter_errs) - 1, None)
                            #if this is the first time we are decreasing the lrate
                            else:
                                iters_to_test = [slice(0,min_factors['min_idx'] + 1), slice(len(iter_errs) - 1, None)]
        if (not suppress_output) and (all_iters_count > 0) and (not all_iters_count % checkpoint_interval):
            #save the state of the rest of the model
            if STORAGE == 'S3':
                bucket_txt, key_txt = s3_library.parse_s3_url(cur_checkpoint)
                key_dir = key_txt + '.factors'
#                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'gmean.pickle'), gmean)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct.pickle'), ct)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_bias.pickle'), ct_bias)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay.pickle'), assay)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_bias.pickle'), assay_bias)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_m1.pickle'), ct_m1)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_m2.pickle'), ct_m2)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_bias_m1.pickle'), ct_bias_m1)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_bias_m2.pickle'), ct_bias_m2)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_m1.pickle'), assay_m1)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_m2.pickle'), assay_m2)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_bias_m1.pickle'), assay_bias_m1)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_bias_m2.pickle'), assay_bias_m2)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'ct_t.pickle'), ct_t)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'assay_t.pickle'), assay_t)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'iter_errs.pickle'), iter_errs)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'iters_to_test.pickle'), iters_to_test)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'test_res.pickle'), test_res)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'subsets.pickle'), subsets)
                s3_library.set_pickle_s3(bucket_txt, os.path.join(key_dir, 'random_state.pickle'), rs.get_state())
                key_min_gtotal = os.path.join(key_dir, 'min_gtotal.{!s}.rdd.pickle'.format(min_factors['min_idx']))
                url_min_gtotal = 's3://{!s}/{!s}'.format(bucket_txt, key_min_gtotal)
                if not len(s3_library.glob_keys(bucket_txt, os.path.join(key_min_gtotal, '*'))):
                    #now, save the current minimum
                    save_rdd_as_pickle(min_factors['gtotal'], url_min_gtotal)
                    min_factors_no_gtotal = {k:v for k,v in min_factors.items() if k != 'gtotal'}
                    min_factors_no_gtotal_key = os.path.join(key_dir, 'min_factors.{!s}.pickle'.
                                                             format(min_factors['min_idx']))
                    s3_library.set_pickle_s3(bucket_txt, min_factors_no_gtotal_key, min_factors_no_gtotal)
                    del(min_factors_no_gtotal)

            elif STORAGE == 'BLOB':
                container, blob = parse_azure_url(cur_checkpoint)[:2]
                blob_dir = blob + '.factors'
#                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'gmean.pickle'), gmean)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct.pickle'), ct)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_bias.pickle'), ct_bias)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay.pickle'), assay)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_bias.pickle'), assay_bias)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_m1.pickle'), ct_m1)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_m2.pickle'), ct_m2)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_bias_m1.pickle'), ct_bias_m1)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_bias_m2.pickle'), ct_bias_m2)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_m1.pickle'), assay_m1)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_m2.pickle'), assay_m2)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_bias_m1.pickle'), assay_bias_m1)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_bias_m2.pickle'), assay_bias_m2)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'ct_t.pickle'), ct_t)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'assay_t.pickle'), assay_t)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'iter_errs.pickle'), iter_errs)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'iters_to_test.pickle'), iters_to_test)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'test_res.pickle'), test_res)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'subsets.pickle'), subsets)
                azure_library.load_blob_pickle(container, os.path.join(blob_dir, 'random_state.pickle'), rs.get_state())
                blob_min_gtotal = os.path.join(blob_dir, 'min_gtotal.{!s}.rdd.pickle'.format(min_factors['min_idx']))
                url_min_gtotal = 'wasbs://{!s}@imputationstoretim.blob.core.windows.net/{!s}'.format(container, blob_min_gtotal)
                if not len(azure_library.glob_blobs(container, os.path.join(blob_min_gtotal, '*'))):
                    #now, save the current minimum
                    save_rdd_as_pickle(min_factors['gtotal'], url_min_gtotal)
                    min_factors_no_gtotal = {k:v for k,v in min_factors.items() if k != 'gtotal'}
                    min_factors_no_gtotal_blob = os.path.join(blob_dir, 'min_factors.{!s}.pickle'.
                                                              format(min_factors['min_idx']))
                    azure_library.load_blob_pickle(container, min_factors_no_gtotal_blob, min_factors_no_gtotal)
                    del(min_factors_no_gtotal)
            #If a new minimum was found in the just-saved checkpoint, then delete the previous minimum
            if checkpoint_to_delete:
                if STORAGE == 'S3':
                    cmd = ['aws', 's3', 'rm', '--quiet', '--recursive', checkpoint_to_delete]
                    subprocess.check_call(cmd)
                    #remove associated parameters
                    cmd = ['aws', 's3', 'rm', '--quiet', '--recursive', checkpoint_to_delete.rstrip('/') + '.factors/']
                    subprocess.check_call(cmd)
                elif STORAGE == 'BLOB':
                    container, blob = parse_azure_url(checkpoint_to_delete)[:2]
                    azure_library.delete_blob_dir(container, blob)
                    #remove associated parameters
                    azure_library.delete_blob_dir(container, blob.rstrip('/') + '.factors/')
                checkpoint_to_delete = None
                
        all_iters_count += 1
    return min_factors['gtotal'], min_factors['ct'], min_factors['ct_bias'], min_factors['assay'], min_factors['assay_bias'], iter_errs

def _just_write_bdg_coords(part_idx, rdd_part, bdg_path, winsize=25, tmpdir='/data/tmp'):
    rdd_part = list(rdd_part)
    gidx = [elt[0] for elt in rdd_part]
    #save bedgraph coords to join with cell type/assay data
    bdg_coords = numpy.array(gidx, dtype=object)
    bdg_coords = numpy.hstack([bdg_coords, bdg_coords[:,1][:,None] + winsize])
    coords_path = os.path.join(tmpdir, os.path.basename(bdg_path.format('bdg', 'coords', '{:05d}'.format(part_idx))))
    numpy.savetxt(coords_path, bdg_coords, delimiter='\t', fmt=['%s', '%i', '%i'])
    yield part_idx

def _construct_bdg_parts(part_idx, rdd_part, bdg_path, ct_list, assay_list, ct, ct_bias, assay, assay_bias, gmean, winsize=25, sinh=True, coords=None, tmpdir='/data/tmp'):
    rdd_part = list(rdd_part)
    #if this is already an imputed rdd, then no need to call compute_imputed
    if ct is None:
#        raise Exception(rdd_part[0], len(rdd_part), len(rdd_part[0]))
        gidx, imputed_tensor = list(zip(*rdd_part))
        data_tensor = None
    #otherwise, use the factor matrices to generate the imputation
    else:
        gidx, imputed_tensor, data_tensor = list(zip(*[compute_imputed(elt, ct, ct_bias, assay, assay_bias, gmean) + (elt[1],) for elt in rdd_part]))

    #save bedgraph coords to join with cell type/assay data
    bdg_coords = numpy.array(gidx, dtype=object)
    bdg_coords = numpy.hstack([bdg_coords, bdg_coords[:,1][:,None] + winsize])
    coords_path = os.path.join(tmpdir, os.path.basename(bdg_path.format('bdg', 'coords', '{:05d}'.format(part_idx))))
    numpy.savetxt(coords_path, bdg_coords, delimiter='\t', fmt=['%s', '%i', '%i'])

    #save cell type/assay data
    if isinstance(imputed_tensor[0], sps.csr_matrix):
        imputed_tensor = numpy.array([elt.toarray() for elt in imputed_tensor])
    else:
        imputed_tensor = numpy.array(imputed_tensor)
    if data_tensor is not None:
        for elt in data_tensor:
            elt.data += gmean
        data_tensor = numpy.array([elt.toarray() for elt in data_tensor])
    imputed_tensor[numpy.where(imputed_tensor < 0)] = 0
    if sinh is True:
        imputed_tensor = numpy.sinh(imputed_tensor)
        if data_tensor is not None:
            data_tensor = numpy.sinh(data_tensor)
    if coords is None:
        coords = list(itertools.product(range(len(ct_list)), range(len(assay_list))))
    else:
        coords = list(zip(*coords))
    for ct_idx, assay_idx in coords:
        ct_name = ct_list[ct_idx]
        assay_name = assay_list[assay_idx]
        impsave = bdg_path.format(ct_name, assay_name, '{:05d}.imp'.format(part_idx))
        print(impsave)
        try:
            os.makedirs(os.path.dirname(impsave))
        except:
            pass
        numpy.savetxt(impsave, imputed_tensor[:, ct_idx, assay_idx], delimiter='\t', fmt='%.8e')
        if data_tensor is not None:
            obssave = bdg_path.format(ct_name, assay_name, '{:05d}.obs'.format(part_idx))
            numpy.savetxt(obssave, data_tensor[:, ct_idx, assay_idx], delimiter='\t', fmt='%.8e')
    yield part_idx

def _compile_bdg_and_upload(ctassay_part, out_bucket, out_root, bdg_path, make_public=True, tmpdir='/data/tmp'):
    chrom_sizes_path = os.path.join(tmpdir, 'hg19.chrom.sizes')
    chrom_bed_path = os.path.join(tmpdir, 'hg19.chrom.bed')
    while not os.path.exists(chrom_sizes_path):
        try:
            os.makedirs(chrom_sizes_path+'.lock')
        except:
            time.sleep(numpy.random.randint(10,20))
        else:
            #get chrom_sizes
            s3_library.S3.get_bucket('encodeimputation-alldata').get_key('hg19.chrom.sizes').get_contents_to_filename(chrom_sizes_path, headers={'x-amz-request-payer':'requester'})
            s3_library.S3.get_bucket('encodeimputation-alldata').get_key('hg19.chrom.bed').get_contents_to_filename(chrom_bed_path, headers={'x-amz-request-payer':'requester'})
            os.rmdir(chrom_sizes_path+'.lock')
    for ct_name, assay_name in ctassay_part:
        for bdg_type in ['imp', 'obs']:
            out_bdg = os.path.join(tmpdir, os.path.basename(bdg_path.format(ct_name, assay_name, bdg_type)).replace('.*', ''))
            bdg_glob = bdg_path.format(ct_name, assay_name, '*.{!s}'.format(bdg_type))
            bdg_paths = sorted(glob.glob(bdg_glob))
            if not bdg_paths:
                continue
            unixcat, unixpaste, bedtools = local['cat'], local['paste'], local['bedtools']
            chain = unixcat[bdg_paths] | unixpaste[os.path.join(tmpdir, 'bdg_coords.txt'), '-'] | bedtools['intersect', '-wa', '-a', 'stdin', '-b', chrom_bed_path, '-f', '1.0'] > out_bdg
            chain()
            
            out_bw = os.path.splitext(out_bdg)[0] + '.bw'
            bedGraphToBigWig = local['bedGraphToBigWig']
            bedGraphToBigWig(out_bdg, chrom_sizes_path, out_bw)
            os.remove(out_bdg)

            out_key = s3_library.S3.get_bucket(out_bucket).new_key(os.path.join(out_root, os.path.basename(out_bw)))
            out_key.set_contents_from_filename(out_bw, headers={'x-amz-request-payer':'requester'})
            if make_public is True:
                out_key.make_public()
            os.remove(out_bw)
            local['rm']('-f', bdg_paths)
            assay_color = ASSAY_COLORS.get(assay_name, '80,80,80')
            url = 'http://' + out_key.bucket.name + '.s3.amazonaws.com/' + urllib.quote(out_key.name)
            yield ('track type=bigWig maxHeightPixels=50 name={!s} visibility=full color={!s} bigDataUrl={!s}'.
                   format(os.path.basename(out_key.name), assay_color, url))

def _combine_bdg_coords(bdg_coord_glob):
    bdg_coord_parts = sorted(glob.glob(bdg_coord_glob))
    bdg_coord_path = os.path.join(os.path.dirname(bdg_coord_glob), 'bdg_coords.txt')
    unixcat = local['cat']
    (unixcat[bdg_coord_parts] > bdg_coord_path)()
    local['rm']('-f', *bdg_coord_parts)

def write_bigwigs2(gtotal, ct, ct_bias, assay, assay_bias, gmean, 
                   ct_list, assay_list, out_bucket, out_root, winsize=25, 
                   sinh=True, make_public=True, tmpdir='/data/tmp', coords=None, extra_id=None):
    out_root = os.path.join(out_root, 'bigwigs')
    if extra_id is None:
        bdg_path = os.path.join(tmpdir, '{0!s}_{1!s}/{0!s}_{1!s}.{2!s}.txt')
    else:
        bdg_path = os.path.join(tmpdir, '{{0!s}}_{{1!s}}/{{0s}}_{{1!s}}.{:05d}.{{2!s}}.txt'.format(extra_id))
    if coords is None:
        coords = list(zip(*itertools.product(numpy.arange(len(ct_list)), numpy.arange(len(assay_list)))))
    sorted_w_idx = gtotal.map(lambda x: (x[0],x)).sortByKey().map(lambda (x,y): y).mapPartitionsWithIndex(lambda x,y: _construct_bdg_parts(x, y, bdg_path, ct_list, assay_list, ct, ct_bias, assay, assay_bias, gmean, winsize=winsize, sinh=sinh, coords=coords, tmpdir=tmpdir)).count()

    bdg_coord_glob = os.path.join(tmpdir, 'bdg_coords.*.txt')
    sc.parallelize([bdg_coord_glob], numSlices=1).foreach(_combine_bdg_coords)

    if extra_id is not None:
        bdg_path = os.path.join(tmpdir, '{0!s}_{1!s}/{0!s}_{1!s}.*.{2!s}.txt')
    ct_assay_list = [(ct_list[c], assay_list[a]) for c, a in zip(*coords)]
    track_lines = sc.parallelize(ct_assay_list, numSlices=len(ct_assay_list)/2).mapPartitions(lambda x: _compile_bdg_and_upload(x, out_bucket, out_root, bdg_path, tmpdir=tmpdir)).collect()
    out_url = 's3://{!s}/{!s}'.format(out_bucket, os.path.join(out_root, 'track_lines.txt'))
    with smart_open.smart_open(out_url, 'w') as out:
        out.write('\n'.join(track_lines))    
    return

def impute_and_avg(model_urls, coords='test'):
    models = []
    data_rdd = None
    for url in model_urls:
        (gmean, ct, ct_bias, assay, assay_bias, 
         genome_params, hyperparams, data) = load_model(url, load_data_too=True if data_rdd is None else False)
        if data_rdd is None:
            data_rdd = data[0]
        gtotal_rdd = genome_params.join(data_rdd).map(lambda (gidx, ((g, gb), d)): (gidx, d, g, gb))
        if coords in ['train', 'valid', 'test']:
            coords = numpy.where(~hyperparams['subsets'][SUBSET_MAP[coords]])
        imputed_rdd = compute_imputed2(gtotal_rdd, ct, ct_bias, assay, assay_bias, gmean, coords=coords)
        models.append(imputed_rdd)
    num_parts = models[0].getNumPartitions()
    model_count = float(len(models))
    avg_imp = sc.union(models).reduceByKey(lambda x,y: x + y, numPartitions=num_parts).mapValues(lambda x: x/model_count)
    return avg_imp

def debug(gtotal=None):
    gmean = s3_library.get_pickle_s3('encodeimputation-alldata', 'predictd_demo/demo_output13/gmean.pickle')
    assay, assay_bias = s3_library.get_pickle_s3('encodeimputation-alldata', 'predictd_demo/demo_output13/assay_factors.pickle')
    ct, ct_bias = s3_library.get_pickle_s3('encodeimputation-alldata', 'predictd_demo/demo_output13/ct_factors.pickle')
    data_idx = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/data_idx.pickle')
    assay_list = [a[0] for a in sorted(set([(elt[1],elt[-1][1]) for elt in data_idx.values()]), key=lambda x:x[1])]
    print(len(assay_list))
    ct_list = ['Fetal_Spinal_Cord']
    if gtotal is None:
#        gtotal = load_saved_rdd('s3://encodeimputation-alldata/predictd_demo/demo_output13/genome_factors.rdd.pickle')
        gtotal = load_saved_rdd('s3://encodeimputation-alldata/predictd_demo/demo_output/gtotal.rdd.pickle')
#        gtotal.count()

#    write_bigwigs(gtotal.map(lambda (idx, (g, gb)): (idx, None, g, gb)).repartition(1200), ct, ct_bias, assay, assay_bias, gmean, ct_list, assay_list,
#                  'encodeimputation-alldata', 'predictd_demo/demo_output13/bw_test')
    write_bigwigs2(gtotal, ct, ct_bias, assay, assay_bias, gmean, 
                   ct_list, assay_list, 'encodeimputation-alldata', 'predictd_demo/demo_output14', 
                   winsize=25, sinh=True, make_public=True, tmpdir='/data/tmp')
