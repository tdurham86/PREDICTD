'''This script provides the main() function for Spearmint hyperparameter optimization.
'''
import argparse
import numpy
import os
import pickle
from pyspark import SparkContext
import subprocess
import sys

sys.path.append(os.path.dirname(__file__))
import s3_library
import train_model

TMPDIR='/data/tmp'
if not os.path.isdir(TMPDIR):
    os.makedirs(TMPDIR)

def main(job_id, params):
    '''Start the SparkContext, train the model on the given parameters.
    '''
    #set the hyperparameters for this model training
    with open(os.path.join(TMPDIR, 'root_params.pickle'), 'rb') as pickle_in:
        args = pickle.load(pickle_in)
    for key in params:
        key_parsed = key.split('zzz')
        if key_parsed[-1] == 'exp':
            setattr(args, key_parsed[0], numpy.exp(params[key][0]))
        else:
            setattr(args, key_parsed[0], params[key][0])
#    args.learning_rate = numpy.exp(params['learning_rate'][0])
#    args.rc = numpy.exp(params['rc'][0])
#    args.ra = numpy.exp(params['ra'][0])
#    args.ri = numpy.exp(params['ri'][0])
#    args.ri2 = numpy.exp(params['ri2'][0])
    args.out_root = os.path.join(args.out_root, 'hypersearch_{:05}'.format(job_id))

    #randomly pick the data subsets to use
    rs = numpy.random.RandomState(args.factor_init_seed + int(job_id))
    args.fold_idx = rs.randint(5)
    args.valid_fold_idx = rs.randint(8)
    args.factor_init_seed = rs.randint(int(1e6))

    if args.data_url.startswith('s3'):
        #write command line to S3
        bucket = s3_library.S3.get_bucket(args.run_bucket)
        key = bucket.new_key(os.path.join(args.out_root, 'command_line.txt'))
        to_write = [os.path.join(os.path.dirname(__file__), 'train_model.py')]
        to_write += ['--{!s}={!s}'.format(*elt) for elt in args.__dict__.items() 
                     if (not isinstance(elt[1], bool)) and 
                        (elt[1] is not None) and 
                        (elt[0] != 'spearmint_config_path')]
        to_write += ['--{!s}'.format(elt[0]) for elt in args.__dict__.items() if elt[1] is True]
        key.set_contents_from_string(' '.join(to_write) + '\n', headers={'x-amz-request-payer':'requester'})

    #start the SparkContext and start model training
    sc = SparkContext(appName='hyp_search',
                      pyFiles=[os.path.join(os.path.dirname(__file__), 'predictd_lib.py'),
                               os.path.join(os.path.dirname(__file__), 's3_library.py'),
                               os.path.join(os.path.dirname(__file__), 'train_model.py')])
    train_model.pl.sc = sc
    try:
        return train_model.train_consolidated(args)
    finally:
        sc.stop()

if __name__ == "__main__":
    parser = train_model.parser
    parser.add_argument('--spearmint_config_path', help='The path to the file containing the configuration parameters for this Spearmint run.')
    args = parser.parse_args()

    #record run in S3
    if args.data_url.startswith('s3'):
        if not s3_library.glob_keys(args.run_bucket, os.path.join(args.out_root, 'command_line.txt')):
            #save the command line to S3
            bucket = s3_library.S3.get_bucket(args.run_bucket)
            key = bucket.new_key(os.path.join(args.out_root, 'command_line.txt'))
            key.set_contents_from_string(' '.join(sys.argv) + '\n', headers={'x-amz-request-payer':'requester'})
            #save the spearmint config file to S3
            key = bucket.new_key(os.path.join(args.out_root, 'config.json'))
            key.set_contents_from_filename(args.spearmint_config_path)
        elif not args.restart:
            raise Exception('Error: Output directory already exists.')

    #save the root parameters as a pickle file
    with open(os.path.join(TMPDIR, 'root_params.pickle'), 'wb') as out:
        pickle.dump(args, out)

    #run the hyperparameter search as a subprocess
    config_base = os.path.basename(args.spearmint_config_path)
    config_dir = os.path.dirname(args.spearmint_config_path)
    if not config_dir:
        config_dir = os.getcwd()
        if not os.path.isfile(os.path.join(config_dir, config_base)):
            raise Exception('Could not find spearmint config file. Please specify full path.')
    cmd = ['spark-submit', '/root/src/Spearmint-master/spearmint/main.py', '--config={!s}'.format(config_base), config_dir]
    subprocess.check_call(cmd)
