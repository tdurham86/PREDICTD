'''
After the assembled data has been uploaded to S3, the final step before it can be used for training a PREDICTD model is to convert the bedgraph file into a Spark Resilient Distributed Dataset (RDD) so that it can be read in by Apache Spark and distributed across the cluster. This script reads in each line of the assembled data, creates a compressed sparse row matrix representation of it with `scipy.sparse.csr_matrix`, transforms the data values using the inverse hyperbolic sine function for variance stabilization, and adds this matrix to the RDD keyed by the chromosome and start coordinate of the corresponding genomic window.
'''

import argparse
import numpy
import os
import pickle
from plumbum import local
from pyspark import SparkContext
import scipy.sparse as sps
import smart_open
import sys
import time

sys.path.append(os.path.dirname(__file__))
import s3_library
import predictd_lib as pl

class ValCountException(Exception):
    pass
class NoDataException(Exception):
    pass

def line_to_data(line_vals, data_shape, col_coords, arcsinh=True):
    data = numpy.zeros(data_shape)
    data_elts = numpy.array(line_vals.split('|'), dtype=float)
    if numpy.sum(data_elts) == 0:
        raise NoDataException()
#    elif data_elts.shape[0] != 7:
#        raise ValCountException(data_elts)
    if arcsinh is True:
        data_elts = numpy.arcsinh(data_elts)
    data[col_coords] = data_elts
    return sps.csr_matrix(data)

def read_in_part(part_url, data_shape, col_coords, coords_bed=None, tmpdir='/data/tmp'):
    part_data = []
    if coords_bed:
        #get the local coords bed to the tmpdir
        local_coords_bed = os.path.join(tmpdir, os.path.basename(coords_bed))
        while not os.path.exists(local_coords_bed):
            try:
                os.makedirs(local_coords_bed + '.lock')
            except:
                time.sleep(numpy.random.randint(10,20))
            else:
                try:
                    with smart_open.smart_open(coords_bed) as lines_in, open(local_coords_bed, 'w') as out:
                        for line in lines_in:
                            out.write(line)
                except:
#                    os.remove(local_coords_bed)
                    raise
            finally:
                if os.path.isdir(local_coords_bed + '.lock'):
                    os.rmdir(local_coords_bed + '.lock')
        #extract the relevant data from the part_url passed in
        read_lines_path = os.path.join(os.path.dirname(__file__), 'read_file_lines.py')
        bedtools, read_lines = local['bedtools'], local[read_lines_path]
        extract_regions = (read_lines[part_url] | bedtools['intersect', '-a', 'stdin', '-b', local_coords_bed, '-wa'])
        p = extract_regions.popen()
        to_read = p.stdout
    else:
        to_read = smart_open.smart_open(part_url, 'r')
    with to_read as lines_in:
        for line in lines_in:
            line = line.strip().split()
            try:
                part_data.append(((line[0], int(line[1])),
                                  line_to_data(line[-1], data_shape, col_coords, arcsinh=True)))
            except NoDataException:
                pass
            except ValueError as err:
                raise Exception(line, data_shape, col_coords, err)
#            if len(part_data) == int(1e5):
#                break
##            except ValCountException as err:
##                raise Exception((part_url,) + err.args)
    return part_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_parts_url_base', help='The S3 URI of the base name for the parts of the uploaded assembled bedgraph.')
    parser.add_argument('--coords_bed', help='A bed file describing the subset of regions that should be included in the RDD for training the model. Typically this will be about 1%% of the genome (e.g. the ENCODE Pilot Regions), and each model will train on on a random 10%% of this subset. [default: %(default)s]', default='s3://encodeimputation-alldata/25bp/hars_with_encodePilots.25bp.windows.bed.gz')
#    parser.add_argument('--random_frac', type=float)
    parser.add_argument('--out_url', help='The S3 URI to which the RDD should be written. This is the URI that will be used to refer to this RDD when running the PREDICTD model.')
#    parser.add_argument('--data_shape', default='1,24')
    args = parser.parse_args()

    #set SparkContext
    sc = SparkContext(appName='assemble_rdd',
                      pyFiles=[s3_library.__file__.replace('.pyc', '.py'),
                               pl.__file__.replace('.pyc', '.py')])
    pl.sc = sc

    #get num parts
    num_parts_url = os.path.join(os.path.dirname(args.data_parts_url_base), 'num_parts.pickle')
    with smart_open.smart_open(num_parts_url, 'rb') as pickle_in:
        num_parts = pickle.loads(pickle_in.read())

    #get column coords
    col_coords_url = args.data_parts_url_base + '.coord_order.pickle'
    with smart_open.smart_open(col_coords_url, 'rb') as pickle_in:
        col_coords = list(zip(*pickle.loads(pickle_in.read())))
#    print('col_coords: {!s}'.format(col_coords))

    #set the data_idx to be named after the RDD
    bucket_txt, key_txt = s3_library.parse_s3_url(args.data_parts_url_base)
    data_idx_key_path = os.path.join(os.path.dirname(key_txt), 'data_idx.pickle')
    out_bucket_txt, out_key_txt = s3_library.parse_s3_url(args.out_url)
    out_data_idx = os.path.splitext(out_key_txt)[0] + '.data_idx.pickle'
    data_idx_dst_bucket = s3_library.S3.get_bucket(out_bucket_txt)
    data_idx_dst_bucket.copy_key(out_data_idx, bucket_txt, data_idx_key_path)

    #for each part, extract data and parallelize into RDD
    data_shape = tuple(numpy.max(numpy.array(col_coords), axis=1) + 1)
#    print('data_shape: {!s}'.format(data_shape))
#    sys.exit()
#    data_shape = tuple([int(elt) for elt in args.data_shape.split(',')])
    part_list = []
    for idx in xrange(num_parts):
        part_url = args.data_parts_url_base + '.part{!s}.txt.gz'.format(idx)
        part_list.append(part_url)
#        to_parallelize = read_in_part(part_url, data_shape, col_coords)
#        rdd_elt = sc.parallelize(to_parallelize, numSlices=200).persist()
#        rdd_elt.count()
#        rdd_list.append(rdd_elt)
    total_rdd = sc.parallelize(part_list, numSlices=int(len(part_list)/2.0)).flatMap(lambda x: read_in_part(x, data_shape, col_coords, coords_bed=args.coords_bed)).repartition(2000).persist()
    total_rdd.count()
#    for rdd in rdd_list:
#        rdd.unpersist()
    pl.save_rdd_as_pickle(total_rdd, args.out_url)
    sc.stop()
