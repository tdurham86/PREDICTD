#! /usr/bin/env python

'''
After a model is trained, it can be useful to calculate different quality measures, particularly on held out test set data, to assess how well the imputation performed. This script will take in observed and imputed data and calculate the quality measures used in the PREDICTD publication on the ENCODE Pilot Regions and, optionally, also the non-coding human accelerated regions. Note that the peak recovery measure, CatchPeakObs, can currently only be calculated for Roadmap Epigenomics experiments because it must have a called peaks file to refer to.
'''

import argparse
import glob
import gzip
import itertools
import multiprocessing as mp
import numpy
import os
import pickle
import Queue
import scipy.stats as stats
import scipy.sparse as sps
from sklearn import metrics
import smart_open
import sparkpickle
import subprocess
import sys
import tempfile
import time

sys.path.append(os.path.dirname(__file__))
import s3_library
#import transform_imputed_data_test25 as tid

def flatten_csr(csr_mat):
    rownum, colnum = csr_mat.shape
    for i in range(rownum):
        csr_mat.indices[csr_mat.indptr[i]:csr_mat.indptr[i+1]] += colnum * i
    csr_mat.indptr = [0,len(csr_mat.indices)]
    flattened = sps.csr_matrix((csr_mat.data, csr_mat.indices, csr_mat.indptr), shape=(1, rownum * colnum))
    flattened.data[numpy.nan_to_num(flattened.data) < 0] = 0
    return flattened

def load_sparkpickles(in_queue, out_queue, flatten=True):
    while True:
        try:
            path = in_queue.get(True, 10)
        except Queue.Empty:
            return
        tries = 5
        while True:
            try:
                if path.startswith('s3://'):
                    with smart_open.smart_open(path, 'rb') as path_in:
                        if path.endswith('.pickle'):
                            part_data = pickle.loads(path_in.read())
                        else:
                            part_data = sparkpickle.loads(path_in.read())
                else:
                    with open(path, 'rb') as path_in:
                        part_data = sparkpickle.load(path_in)
            except:
                if tries:
                    tries -= 1
                else:
                    raise
            else:
                break
        out_queue.put_nowait([(idx, flatten_csr(mat) if flatten is True else mat) for idx, mat in part_data])
#            out_queue.put_nowait(sparkpickle.load(path_in))
        time.sleep(0.1)
        in_queue.task_done()

def read_in_parts(parts_glob, num_tries=4, num_procs=16, flatten=True):
    part_q = mp.JoinableQueue()
    data_q = mp.Queue()
    if parts_glob.startswith('s3://'):
        bucket_txt, key_txt = s3_library.parse_s3_url(parts_glob)
        glob_res = ['s3://{!s}/{!s}'.format(bucket_txt, elt.name) for elt in s3_library.glob_keys(bucket_txt, key_txt)]
    else:
        glob_res = glob.glob(parts_glob)
    for part in glob_res:
        part_q.put_nowait(part)
        time.sleep(0.005)
    procs = [mp.Process(target=load_sparkpickles, args=(part_q, data_q), kwargs={'flatten':flatten}) for _ in range(num_procs)]
    for proc in procs:
        proc.start()

    part_q.join()

    data = []
    tries = num_tries
    while tries:
        try:
            data.append(data_q.get(True, 10))
        except Queue.Empty:
            tries -= 1
        else:
            tries = num_tries
    data = sorted(itertools.chain(*data), key=lambda x: x[0])
#    return data
    data_gidx, data = list(zip(*data))
    if flatten is True:
        return data_gidx, sps.vstack(data).transpose()
    else:
        return data_gidx, data

def match_keys(gidx1, gidx2):
    '''Takes two sorted gidx lists as returned from read_in_parts() and finds the coordinates in gidx2
    that map to those in gidx1. Note that gidx1 must be a subset of gidx2.
    '''
    offset = 0
    gidx2_mapped_idx = []
    for i in xrange(len(gidx1)):
        while True:
            if gidx1[i] == gidx2[i + offset]:
                gidx2_mapped_idx.append(i + offset)
                break
            else:
                offset += 1
    return gidx2_mapped_idx

def join_data(gidx1, data1, gidx2, data2):
    if data1.shape[0] <= data2.shape[0]:
        data_to_select = match_keys(gidx1, gidx2)
        gidx2 = [gidx2[i] for i in data_to_select]
        data2 = data2[data_to_select, :]
    elif data1.shape[0] > data2.shape[0]:
        data_to_select = match_keys(gidx2, gidx1)
        gidx1 = [gidx1[i] for i in data_to_select]
        data1 = data1[data_to_select, :]
    return gidx1, data1, gidx2, data2

def get_data(parts_glob, working_dir, num_procs, save_to_disk=False):
    parts_path = os.path.join(working_dir, os.path.basename(os.path.dirname(parts_glob)))
    if save_to_disk is False or not os.path.isfile(parts_path):
        parts_gidx, parts = read_in_parts(parts_glob, flatten=True, num_procs=num_procs)
        parts = parts.T
        if save_to_disk is True:
            with open(parts_path, 'wb') as out:
                pickle.dump((parts_gidx, parts), out)
                out.flush()
                os.fsync(out.fileno())
    else:
        with open(parts_path, 'rb') as pickle_in:
            parts_gidx, parts = pickle.load(pickle_in)
    return parts_gidx, parts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_glob', help='The S3 URI to the observed data RDD parts, with appropriate Unix wildcard characters to specify all RDD parts in one string.')
    parser.add_argument('imputed_glob', help='The S3 URI to the imputed data RDD parts, with appropriate Unix wildcard characters to specify all RDD parts in one string.')
    parser.add_argument('--alternative_imputed', help='In the case of three-way comparisons, specify S3 URI to parts of another imputed or observed data RDD, with appropriate Unix wildcard characters to specify all RDD parts in one string.')
    parser.add_argument('--data_idx_url', default='s3://encodeimputation-alldata/25bp/data_idx.pickle', help='S3 URI to the data_idx.pickle file for these imputed/observed data. [default: %(default)s]')
    parser.add_argument('--working_dir', default='/data/peak_pr', help='The local directory to use as a workspace for computing the quality measures. [default: %(default)s]')
    parser.add_argument('--fold_idx', type=int, default=0, help='The index of the test set specifying the experiments on which the quality measures should be calculated. [default: %(default)s]')
#    parser.add_argument('--valid_fold_idx', type=int, default=0, help='The index of the validation set corresponding to the training set to use for ')
    parser.add_argument('--fold_name', help='The basename of the file specifying the test subsets to use. [default: %(default)s]')
    parser.add_argument('--out_name', help='The base filename for the quality measures output. [default: \'metrics_fold{!s}.out\'.format(args.fold_idx)]')
    parser.add_argument('--num_procs', type=int, default=1, help='The number of processes to use when reading in the observed and imputed data. If multiple cores are available, setting this option > 1 can speed up data loading. [default: %(default)s]')
    parser.add_argument('--only_encodePilots', action='store_true', default=False, help='If set, calculate quality measures only over the ENCODE Pilot Regions, even if the RDD elements include the non-coding human accelerated regions as well.')
    args = parser.parse_args()

    data_idx_bucket, data_idx_key = s3_library.parse_s3_url(args.data_idx_url)
    data_idx = s3_library.get_pickle_s3(data_idx_bucket, data_idx_key)
    if args.fold_name is not None:
        if args.fold_name.startswith('s3'):
            subsets_bucket, subsets_key = s3_library.parse_s3_url(args.fold_name)
        else:
            subsets_bucket = data_idx_bucket
            subsets_key = os.path.join(os.path.dirname(data_idx_key), args.fold_name)
            if s3_library.S3.get_bucket(subsets_bucket).get_key(subsets_key):
                subsets = s3_library.get_pickle_s3(data_idx_bucket, subsets_key)[args.fold_idx]
    else:
        #ignore subsets and just calc quality metrics for all experiments in data_idx
        subsets = None
#        subsets_dict = {}
#        if isinstance(subsets, dict):
#            if isinstance(subsets['train'], list):
#                subsets['valid'] = subsets['train'][args.valid_fold_idx]['valid']
#                subsets['train'] = subsets['train'][args.valid_fold_idx]['train']
#        else:
#            subsets_dict['test'] = subsets['train'][args.valid_fold_idx][2]
#            subsets_dict['valid'] = subsets['train'][args.valid_fold_idx][1]
#            subsets_dict['train'] = subsets['train'][args.valid_fold_idx][0]
#            subsets = subsets_dict
#    else:
#        subsets = None
#    data_idx = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/data_idx.pickle')
#    subsets = s3_library.get_pickle_s3('encodeimputation-alldata', os.path.join('25bp',args.fold_name))[args.fold_idx]

    working_dir = args.working_dir
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    if 'chromimpute' in args.imputed_glob or args.only_encodePilots is True:
        window_regions_name = 'hg19.encodePilotRegions.25bp.windows.bed.gz'
    else:
        window_regions_name = 'hars_with_encodePilots.25bp.windows.bed.gz'
    window_regions_path = os.path.join(working_dir, window_regions_name)
    if not os.path.exists(window_regions_path):
        s3_library.S3.get_bucket('encodeimputation-alldata').get_key(os.path.join('25bp', window_regions_name)).get_contents_to_filename(window_regions_path, headers={'x-amz-request-payer':'requester'})
    with gzip.open(window_regions_path) as lines_in:
        window_set = set()
        for line in lines_in:
            line = line.strip().split()
            window_set.add((line[0], int(line[1])))

    print('Getting observed data.')
    data_gidx, data = get_data(args.data_glob, args.working_dir, args.num_procs, save_to_disk=False)
    data_gidx, selector = zip(*[(elt, idx) for idx, elt in enumerate(data_gidx) if elt in window_set])
    data = data[selector,:]
    print('Data shape: {!s}'.format(data.shape))

    print('Observed data retrieved. Getting imputed data.')
    imputed_gidx, imputed = get_data(args.imputed_glob, args.working_dir, args.num_procs, save_to_disk=False)
    imputed_gidx, selector = zip(*[(elt, idx) for idx, elt in enumerate(imputed_gidx) if elt in window_set])
    imputed = imputed[selector,:]
    print('Imputed shape: {!s}'.format(imputed.shape))

    if args.alternative_imputed:
        print('Imputed data retrieved. Getting alternative imputed data.')
        alt_imputed_gidx, alt_imputed = get_data(args.alternative_imputed, args.working_dir, args.num_procs, save_to_disk=False)
        alt_imputed_gidx, selector = zip(*[(elt, idx) for idx, elt in enumerate(alt_imputed_gidx) if elt in window_set])
        alt_imputed = alt_imputed[selector,:]
        print('Alternative imputed shape: {!s}'.format(alt_imputed.shape))

    joined = False
    if imputed.shape[0] != data.shape[0]:
        print('Imputed and Data have different genome dimension lengths. Joining.')
        imputed_gidx, imputed, data_gidx, data = join_data(imputed_gidx, imputed, data_gidx, data)
        joined = True
    if args.alternative_imputed and alt_imputed.shape[0] != data.shape[0]:
        print('Alternative Imputed and Data have different genome dimension lengths. Joining.')
        if alt_imputed.shape[0] < data.shape[0]:
            alt_imputed_gidx, alt_imputed, data_gidx, data = join_data(alt_imputed_gidx, alt_imputed, data_gidx, data)
            alt_imputed_gidx, alt_imputed, imputed_gidx, imputed = join_data(alt_imputed_gidx, alt_imputed, imputed_gidx, imputed)
        else:
            alt_imputed_gidx, alt_imputed, data_gidx, data = join_data(alt_imputed_gidx, alt_imputed, data_gidx, data)
        joined = True
    if joined is True:
        print('Joined data shape: {!s}\nJoined imputed shape: {!s}\nJoined alternative imputed shape: {!s}'.format(data.shape, imputed.shape, alt_imputed.shape))
    print('All data retrieved. Computing performance metrics.')

    out_name = args.out_name if args.out_name else 'metrics_fold{!s}.out'.format(args.fold_idx)
    if out_name.startswith('s3://'):
        out_name = os.path.basename(out_name)

    cols_w_data = set(imputed.nonzero()[1])
    for elt in sorted(data_idx.values(), key=lambda x: x[:2]):
        if subsets is None:
            col_idx = (elt[-1][0] * 24) + elt[-1][1]
        else:
            col_idx = (elt[-1][0] * subsets['test'].shape[1]) + elt[-1][1]
        if subsets is not None and args.fold_idx >= 0 and subsets['test'][elt[-1]]:
            continue
        elif col_idx not in cols_w_data:
            continue
        print(elt[:2])
        data_vals = numpy.nan_to_num(data[:,col_idx].toarray().flatten())
        imputed_vals = numpy.nan_to_num(imputed[:,col_idx].toarray().flatten())
        data_vals_argsort = numpy.argsort(data_vals)
        imputed_vals_argsort = numpy.argsort(imputed_vals)

        #MSEglobal - genome-wide mean squared error
        mseglobal = numpy.mean((data_vals - imputed_vals) ** 2)

        #MSE1obs - MSE on top 1% of obs win
        size_pct1 = int(round(len(data_vals) * 0.01))
        mse1obs = numpy.mean((data_vals[data_vals_argsort[0 - size_pct1:]] - imputed_vals[data_vals_argsort[0 - size_pct1:]]) ** 2)

        #MSE1imp - MSE on top 1% of imp win
        mse1imp = numpy.mean((data_vals[imputed_vals_argsort[0 - size_pct1:]] - imputed_vals[imputed_vals_argsort[0 - size_pct1:]]) ** 2)

        #MSE1altimp - MSE on top 1% of windows from alternative imputation
        if args.alternative_imputed:
            alt_imputed_vals = numpy.nan_to_num(alt_imputed[:, col_idx].toarray().flatten())
            alt_imputed_vals_argsort = numpy.argsort(alt_imputed_vals)
            mse1altimp = numpy.mean((data_vals[alt_imputed_vals_argsort[0 - size_pct1:]] - imputed_vals[alt_imputed_vals_argsort[0 - size_pct1:]]) ** 2)

            mse1altimprecip = numpy.mean((data_vals[imputed_vals_argsort[0 - size_pct1:]] - alt_imputed_vals[imputed_vals_argsort[0 - size_pct1:]]) ** 2)
        else:
            mse1altimp = None
            mse1altimprecip = None

        #GWcorrasinh - genome-wide Pearson correlation on the asinh-transformed data
        gwcorrasinh = stats.pearsonr(data_vals, imputed_vals)

        #GWspcorr - genome-wide Spearman correlation
        gwspcorr = stats.spearmanr(data_vals, imputed_vals)

        #transform back to actual data values
        data_vals = numpy.sinh(data_vals)
        imputed_vals = numpy.sinh(imputed_vals)

        #GWcorr - genome-wide correlation
        gwcorr = stats.pearsonr(data_vals, imputed_vals)

        imputed_pct1_wins = imputed_vals_argsort[0 - size_pct1:]
        data_pct1_wins = data_vals_argsort[0 - size_pct1:]

        #Match1 - overlap between imp & obs in top 1% signal windows
        match1 = len(set(data_pct1_wins) & set(imputed_pct1_wins))/float(size_pct1)

        #Catch1obs - pct of top 1% obs wins in top 5% imp wins
        size_pct5 = round(len(data_vals) * 0.05)
        imputed_pct5_wins = imputed_vals_argsort[0 - size_pct5:]
        catch1obs = len(set(data_pct1_wins) & set(imputed_pct5_wins))/float(len(data_pct1_wins))

        #Catch1imp - pct of top 1% imp wins in top 5% obs wins
        data_pct5_wins = data_vals_argsort[0 - size_pct5:]
        catch1imp = len(set(imputed_pct1_wins) & set(data_pct5_wins))/float(len(imputed_pct1_wins))

        #AucObs1 - recovery of top 1% obs based on all imp wins using ROC AUC
        true_class = numpy.zeros(data_vals.shape)
        true_class[data_pct1_wins] = 1
        aucobs1 = metrics.roc_auc_score(true_class, imputed_vals)

        #AucImp1 - recovery of top 1% imp based on all obs wins using ROC AUC
        true_class = numpy.zeros(imputed_vals.shape)
        true_class[imputed_pct1_wins] = 1
        aucimp1 = metrics.roc_auc_score(true_class, data_vals)

        #CatchPeakObs - auc of recovery of obs peak calls based on imp signal
        peaks_key = 'peak_analysis/narrowPeakProc/{!s}-{!s}.sorted.counted.narrowPeak.gz'.format(elt[0], elt[1].replace('H2AZ', 'H2A'))
        peaks_tmp = os.path.join(working_dir, os.path.basename(peaks_key))
        try:
            s3_library.S3.get_bucket('encodeimputation').get_key(peaks_key).get_contents_to_filename(peaks_tmp, headers={'x-amz-request-payer':'requester'})
        except AttributeError:
            catchpeakobs = None
        else:
            cmd = ['bedtools', 'intersect', '-b', peaks_tmp, '-a', window_regions_path, '-loj']
            bedtools_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            peak_wins = []
            with bedtools_proc.stdout as peak_wins_in:
                for line in peak_wins_in:
                    line = line.strip().split()
                    peak_wins.append(((line[0], int(line[1])), 0 if line[-1] == '.' else 1))
            peak_wins = sorted(peak_wins, key=lambda x: x[0])
            true_class = numpy.array([win[1] for win in peak_wins])
            catchpeakobs = metrics.roc_auc_score(true_class, imputed_vals)
            os.remove(peaks_tmp)

        out_vals = ['{!s}-{!s}'.format(elt[0], elt[1]), elt[-1][0], elt[-1][1], 
                    mseglobal, mse1obs, mse1imp, 
                    gwcorrasinh[0], gwcorrasinh[1], gwspcorr[0], gwspcorr[1], gwcorr[0], gwcorr[1],
                    match1, catch1obs, catch1imp, aucobs1, aucimp1, catchpeakobs, mse1altimp, mse1altimprecip]
#        out_vals = ['{!s}-{!s}'.format(elt[0], elt[1]), elt[-1][0], elt[-1][1], mse1altimp, mse1altimprecip]
        with open(os.path.join(working_dir, out_name), 'a') as out:
            out.write('\t'.join([str(elt) for elt in out_vals]) + '\n')

        if not args.out_name.startswith('s3://'):
            out_bucket, out_key = s3_library.parse_s3_url(os.path.dirname(args.imputed_glob))
            out_key_full = os.path.join(os.path.dirname(out_key), out_name)
        else:
            out_bucket, out_key_full = s3_library.parse_s3_url(args.out_name)
        s3_library.S3.get_bucket(out_bucket).new_key(out_key_full).set_contents_from_filename(os.path.join(working_dir, out_name), headers={'x-amz-request-payer':'requester'})
