#! /usr/bin/env python

import argparse
import gzip
import numpy
import os
import pickle
import scipy.stats as stats
import scipy.sparse as sps
import sklearn.metrics as metrics
import smart_open
import subprocess
import sys
import tempfile

sys.path.append(os.path.dirname(__file__))
import s3_library
import transform_imputed_data_test25 as tid

def match_keys(gidx1, gidx2):
    '''Takes two sorted gidx lists as returned from tid.read_in_parts() and finds the coordinates in gidx2
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
    if data1.shape[0] < data2.shape[0]:
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
        parts_gidx, parts = tid.read_in_parts(parts_glob, flatten=True, num_procs=num_procs)
        parts = parts.T
        if save_to_disk is True:
            with open(parts_path, 'wb') as out:
                pickle.dump((parts_gidx, parts), out)
    else:
        with open(parts_path, 'rb') as pickle_in:
            parts_gidx, parts = pickle.load(pickle_in)
    return parts_gidx, parts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_glob')
    parser.add_argument('imputed_glob')
    parser.add_argument('--alternative_imputed')
    parser.add_argument('--data_idx_url', default='s3://encodeimputation-alldata/25bp/data_idx.pickle')
    parser.add_argument('--working_dir', default='/dev/peak_pr')
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--valid_fold_idx', type=int, default=0)
    parser.add_argument('--fold_name', default='folds.5.pickle')
    parser.add_argument('--out_name')
    parser.add_argument('--num_procs', type=int, default=1)
    parser.add_argument('--only_encodePilots', action='store_true', default=False)
    args = parser.parse_args()

    data_idx_bucket, data_idx_key = s3_library.parse_s3_url(args.data_idx_url)
    data_idx = s3_library.get_pickle_s3(data_idx_bucket, data_idx_key)
    subsets_key = os.path.join(os.path.dirname(data_idx_key), args.fold_name)
    if s3_library.S3.get_bucket(data_idx_bucket).get_key(subsets_key):
        subsets = s3_library.get_pickle_s3(data_idx_bucket, subsets_key)
        if isinstance(subsets, 'dict'):
            if isinstance(subsets['train'], list):
                subsets['valid'] = subsets['train'][args.valid_fold_idx]['valid']
                subsets['train'] = subsets['train'][args.valid_fold_idx]['train']
        else:
            raise Exception('New list-based subsets data structure not yet supported.')
    else:
        subsets = None
#    data_idx = s3_library.get_pickle_s3('encodeimputation-alldata', '25bp/data_idx.pickle')
#    subsets = s3_library.get_pickle_s3('encodeimputation-alldata', os.path.join('25bp',args.fold_name))[args.fold_idx]

    print('Getting observed data.')
    data_gidx, data = get_data(args.data_glob, args.working_dir, args.num_procs, save_to_disk=False)

    print('Observed data retrieved. Getting imputed data.')
    imputed_gidx, imputed = get_data(args.imputed_glob, args.working_dir, args.num_procs, save_to_disk=False)

    print('Imputed data retrieved. Computing performance metrics.')
    if args.alternative_imputed:
        alt_imputed_gidx, alt_imputed = get_data(args.alternative_imputed, args.working_dir, args.num_procs, save_to_disk=False)

    if imputed.shape[0] != data.shape[0]:
        print('Imputed and Data have different genome dimension lengths. Joining.')
        imputed_gidx, imputed, data_gidx, data = join_data(imputed_gidx, imputed, data_gidx, data)
    if args.alternative_imputed and alt_imputed.shape[0] != data.shape[0]:
        print('Alternative Imputed and Data have different genome dimension lengths. Joining.')
        if alt_imputed.shape[0] < data.shape[0]:
            alt_imputed_gidx, alt_imputed, data_gidx, data = join_data(alt_imputed_gidx, alt_imputed, data_gidx, data)
            alt_imputed_gidx, alt_imputed, imputed_gidx, imputed = join_data(alt_imputed_gidx, alt_imputed, imputed_gidx, imputed)
        else:
            alt_imputed_gidx, alt_imputed, data_gidx, data = join_data(alt_imputed_gidx, alt_imputed, data_gidx, data)

    working_dir = args.working_dir
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    out_name = args.out_name if args.out_name else 'metrics_fold{!s}.out'.format(args.fold_idx)

    if 'chromimpute' in args.imputed_glob or args.only_encodePilots is True:
        window_regions_name = 'hg19.encodePilotRegions.25bp.windows.bed.gz'
    else:
        window_regions_name = 'hars_with_encodePilots.25bp.windows.bed.gz'
    window_regions_path = os.path.join(working_dir, window_regions_name)
    if not os.path.exists(window_regions_path):
        s3_library.S3.get_bucket('encodeimputation-alldata').get_key(os.path.join('25bp', window_regions_name)).get_contents_to_filename(window_regions_path, headers={'x-amz-request-payer':'requester'})

    cols_w_data = set(imputed.nonzero()[1])
    for elt in sorted(data_idx.values(), key=lambda x: x[:2]):
        if subsets is None:
            col_idx = (elt[-1][0] * 24) + elt[-1][1]
        else:
            col_idx = (elt[-1][0] * subsets['train'].shape[1]) + elt[-1][1]
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
                    match1, catch1obs, catch1imp, aucobs1, aucimp1, catchpeakobs, mse1altimp]
#        out_vals = ['{!s}-{!s}'.format(elt[0], elt[1]), elt[-1][0], elt[-1][1], mse1altimp, mse1altimprecip]
        with open(os.path.join(working_dir, out_name), 'a') as out:
            out.write('\t'.join([str(elt) for elt in out_vals]) + '\n')

        out_bucket, out_key = s3_library.parse_s3_url(os.path.dirname(args.imputed_glob))
        s3_library.S3.get_bucket(out_bucket).new_key(os.path.join(os.path.dirname(out_key), out_name)).set_contents_from_filename(os.path.join(working_dir, out_name), headers={'x-amz-request-payer':'requester'})
