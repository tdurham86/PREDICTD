
import argparse
import numpy
import os
import pickle
from pyspark import SparkContext
import scipy.sparse as sps
import smart_open
import sys

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

def read_in_part(part_url, data_shape, col_coords):
    part_data = []
    with smart_open.smart_open(part_url, 'r') as lines_in:
        for line in lines_in:
            line = line.strip().split()
            try:
                part_data.append(((line[0], int(line[1])),
                                  line_to_data(line[-1], data_shape, col_coords)))
            except NoDataException:
                pass
#            except ValCountException as err:
#                raise Exception((part_url,) + err.args)
    return part_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_parts_url_base')
#    parser.add_argument('--coords_bed')
#    parser.add_argument('--random_frac', type=float)
    parser.add_argument('--out_url')
    parser.add_argument('--data_shape', default='1,24')
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

    #for each part, extract data and parallelize into RDD
    data_shape = tuple([int(elt) for elt in args.data_shape.split(',')])
    part_list = []
    for idx in xrange(num_parts):
        part_url = args.data_parts_url_base + '.part{!s}.txt.gz'.format(idx)
        part_list.append(part_url)
#        to_parallelize = read_in_part(part_url, data_shape, col_coords)
#        rdd_elt = sc.parallelize(to_parallelize, numSlices=200).persist()
#        rdd_elt.count()
#        rdd_list.append(rdd_elt)
    total_rdd = sc.parallelize(part_list, numSlices=int(len(part_list)/2.0)).flatMap(lambda x: read_in_part(x, data_shape, col_coords)).repartition(2000).persist()
    total_rdd.count()
#    for rdd in rdd_list:
#        rdd.unpersist()
    pl.save_rdd_as_pickle(total_rdd, args.out_url)
    sc.stop()
