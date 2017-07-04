'''use Bill's code from bin/split-matrix-uniformly.py to chunk total experiment set into 5 folds, then do the same within each fold to get a set of validation folds.
'''

import argparse
import numpy
import os
import random
import sys

sys.path.append(os.path.dirname(__file__))
import s3_library

def computeSumOfMaxes(numSplits, names, counts):
    returnValue = 0
    for name in names:
        maxValue = 0
        for splitIndex in range(0, numSplits):
            if (name, splitIndex) in counts:
                thisValue = counts[(name, splitIndex)]
            else:
                thisValue = 0
            if (thisValue > maxValue):
                maxValue = thisValue
        returnValue += maxValue
    return(returnValue)

def evaluateSplit(numSplits, splitMatrices):
    # Make a count of the number of times we see each row label
    # associated with a split number.
    rowMarginals = {}
    rowNames = {}
    for myTuple in splitMatrices:
        (myRow, myColumn, mySubset) = myTuple
        rowNames[myRow] = True
        if (myRow, mySubset) in rowMarginals:
            rowMarginals[(myRow, mySubset)] += 1
        else:
            rowMarginals[(myRow, mySubset)] = 1

    # Do the same for columns.
    columnMarginals = {}
    columnNames = {}
    for myTuple in splitMatrices:
        (myRow, myColumn, mySubset) = myTuple
        columnNames[myColumn ] = True
        if (myColumn, mySubset) in columnMarginals:
            columnMarginals[(myColumn, mySubset)] += 1
        else:
            columnMarginals[(myColumn, mySubset)] = 1

    return(computeSumOfMaxes(numSplits, rowNames, rowMarginals)
           + computeSumOfMaxes(numSplits, columnNames, columnMarginals))

def splitMatrix(numSplits, myMatrix):
    random.shuffle(myMatrix)
    splitIndex = 0
    returnMatrix = []
    for matrixEntry in myMatrix:
        returnMatrix.append((matrixEntry[0], matrixEntry[1], splitIndex))
        splitIndex += 1
        if (splitIndex == numSplits):
            splitIndex = 0
    return(returnMatrix)

def trySplits(mat_coord_list, num_splits, num_tries):
    minQuality = 0
    for tryIndex in range(0, num_tries):
        splitMatrices = splitMatrix(num_splits, mat_coord_list)
        thisQuality = evaluateSplit(num_splits, splitMatrices)

        if (minQuality == 0) or (thisQuality < minQuality):
            sys.stderr.write("Split %d has quality %g.\n" % (tryIndex, thisQuality))
            minQuality = thisQuality
            bestSplit = splitMatrices
    folds = [([], []) for _ in range(num_splits)]
    for elt in bestSplit:
        for idx in range(num_splits):
            if idx == elt[-1]:
                folds[idx][1].append(elt[:2])
            else:
                folds[idx][0].append(elt[:2])
#    for idx in range(num_splits):
#      print(idx, len(folds[idx][0]), len(folds[idx][1]))
    return folds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_idx_url')
    parser.add_argument('--test_fold_data_idx_url')
    parser.add_argument('--num_test_folds', type=int, default=5)
    parser.add_argument('--num_valid_folds', type=int, default=8)
    parser.add_argument('--num_tries', type=int, default=1000)
    args = parser.parse_args()

    #a data_idx has been passed in that contains the information to generate a single
    #test fold, now use the rest of the experiments in the data_idx_url parameter to
    #generate validation folds
    if args.test_fold_data_idx_url is not None:
        #get total data_idx and matrix size
        bucket_txt, key_txt = s3_library.parse_s3_url(args.data_idx_url)
        data_idx = s3_library.get_pickle_s3(bucket_txt, key_txt)
        mat_coords = [elt[-1] for elt in data_idx.values()]
        mat_size = tuple(numpy.array(mat_coords).max(axis=0) + 1)
        #get the test data sets and generate test fold
        bucket_txt, key_txt = s3_library.parse_s3_url(args.test_fold_data_idx_url)
        test_data_idx = s3_library.get_pickle_s3(bucket_txt, key_txt)
        test_mat_coords = [elt[-1] for elt in test_data_idx.values()]
        for coord in test_mat_coords:
            del(mat_coords[mat_coords.index(coord)])
        test_folds = [(mat_coords, test_mat_coords)]
    #the tensor description is passed in; generate both test and validation folds
    elif 'data_idx' in os.path.basename(args.data_idx_url):
        bucket_txt, key_txt = s3_library.parse_s3_url(args.data_idx_url)
        data_idx = s3_library.get_pickle_s3(bucket_txt, key_txt)
        mat_coords = [elt[-1] for elt in data_idx.values()]
        mat_size = tuple(numpy.array(mat_coords).max(axis=0) + 1)
        if args.num_test_folds > 0:
            test_folds = trySplits(mat_coords, args.num_test_folds, args.num_tries)
        else:
            test_folds = [(mat_coords, [])]
    #the test folds have already been generated; just add in validation folds
    elif os.path.basename(args.data_idx_url) == 'folds.5.pickle':
        bucket_txt, key_txt = s3_library.parse_s3_url(args.data_idx_url)
        data_idx = None
        test_folds = s3_library.get_pickle_s3(bucket_txt, key_txt)
        mat_size = test_folds[0]['test'].shape
        test_folds = [(list(zip(*numpy.where(numpy.logical_or(~elt['train'], ~elt['valid'])))), list(zip(*numpy.where(~elt['test'])))) for elt in test_folds]

    folds = []
    for rest, test_set in test_folds:
        valid_folds = trySplits(rest, args.num_valid_folds, args.num_tries)
        valid_mats = []
        for train_coords, valid_coords in valid_folds:
            train_mat = numpy.ones(mat_size, dtype=bool)
            train_mat[list(zip(*train_coords))] = False
            valid_mat = numpy.ones(mat_size, dtype=bool)
            valid_mat[list(zip(*valid_coords))] = False
            #if any cell types or assays do not have any entries,
            #just move that row/column from the validation matrix
            #to the training matrix
            rowsums = numpy.sum(~train_mat, axis=1)
            if not numpy.all(rowsums):
                zero_idx = numpy.where(rowsums == 0)[0]
                train_mat[zero_idx,:] = valid_mat[zero_idx,:]
                valid_mat[zero_idx,:] = True
            colsums = numpy.sum(~train_mat, axis=0)
            if not numpy.all(colsums):
                zero_idx = numpy.where(colsums == 0)[0]
                train_mat[:,zero_idx] = valid_mat[:,zero_idx]
                valid_mat[:,zero_idx] = True
            valid_mats.append({'train':train_mat, 'valid':valid_mat})
        test_mat = numpy.ones(mat_size, dtype=bool)
        test_mat[list(zip(*test_set))] = False
        folds.append({'train':valid_mats, 'test':test_mat})

    s3_library.set_pickle_s3(bucket_txt, os.path.join(os.path.dirname(key_txt), 'folds.{!s}.{!s}.pickle'.format(args.num_test_folds, args.num_valid_folds)), folds)
