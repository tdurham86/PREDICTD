'''Library to encapsulate interactions with AWS S3.
'''

import boto
import fnmatch
import os
import pickle
import sys
from tempfile import NamedTemporaryFile

##with open(os.path.join(os.path.dirname(__file__), 's3_credentials.txt')) as creds:
#with open('/home/ec2-user/code_parallel_stochastic/predictd/s3_credentials.txt') as creds:
#    cred_info = dict([elt.strip().split('=') for elt in creds])
##print(cred_info)
#S3 = boto.connect_s3(**cred_info)
S3 = boto.s3.connect_to_region('us-west-2')

TMPDIR='/data/tmp'
try:
    if not os.path.isdir(TMPDIR):
        os.mkdir(TMPDIR)
except OSError:
    TMPDIR = os.getcwd()

#set up S3 environment variables
creds_path = os.path.join(os.path.expanduser('~'), '.aws/credentials')
if os.path.isfile(creds_path):
    with open(creds_path) as creds_in:
        creds_dict = dict([elt.strip().split(' = ') for elt in creds_in if ' = ' in elt])
    os.environ['AWS_ACCESS_KEY_ID'] = creds_dict['aws_access_key_id']
    os.environ['AWS_SECRET_ACCESS_KEY'] = creds_dict['aws_secret_access_key']
else:
    sys.stderr.write('Warning: No AWS credentials found in /root/.aws directory.')

def parse_s3_url(s3_url):
    '''Parse out and return the bucket name and key path of an s3 url.
    '''
    s3_url = s3_url.split('/')
    bucket_txt = s3_url[2]
    key_txt = '/'.join(s3_url[3:])
    return bucket_txt, key_txt

def get_pickle_s3(bucketname, keyname):
    '''Get a pickled file from AWS S3.
    '''
    key = S3.get_bucket(bucketname).get_key(keyname)
    with NamedTemporaryFile(dir=TMPDIR) as tmp:
        key.get_contents_to_file(tmp, headers={'x-amz-request-payer':'requester'})
        tmp.seek(0)
        pickled = pickle.load(tmp)
    return pickled

def set_pickle_s3(bucketname, keyname, obj):
    '''Pickle and write an object to AWS S3.
    '''
    try:
        bucket = S3.get_bucket(bucketname)
    except boto.exception.S3ResponseError:
        bucket = S3.create_bucket(bucketname)
    key = bucket.get_key(keyname)
    if key is None:
        key = bucket.new_key(keyname)
#    key.set_contents_from_string(pickle.dumps(obj), headers={'x-amz-request-payer':'requester'})
    with NamedTemporaryFile(dir=TMPDIR) as tmp:
        pickle.dump(obj, tmp)
        tmp.seek(0)
        key.set_contents_from_file(tmp, headers={'x-amz-request-payer':'requester'})

def glob_keys(bucketname, glob_str, just_names=False):
    '''Query keys for the specified bucket and return keys with 
    names matching the glob string.
    '''
    bucket = S3.get_bucket(bucketname)
    path_levels = glob_str.split('/')
    return glob_keys_helper(bucket.list(prefix=path_levels[0], delimiter='/'), path_levels, 0, just_names=just_names)

def glob_keys_helper(prefixes, glob_levels, level_idx, just_names=False):
    globbed = []
    for elt in prefixes:
        if fnmatch.fnmatch(elt.name.rstrip('/'), '/'.join(glob_levels[:level_idx + 1]).rstrip('/')):
            if level_idx + 1 == len(glob_levels):
                if just_names is True:
                    key = elt.name
                else:
                    key = elt.bucket.get_key(elt.name)
                if key:
                    globbed.append(key)
            else:
                globbed.extend(glob_keys_helper(elt.bucket.list(prefix=elt.name, delimiter='/'), 
                                                glob_levels, 
                                                level_idx + 1, just_names=just_names))
    return globbed

def glob_buckets(glob_str):
    '''Find buckets with names matching the provided glob string.
    '''
    globbed = []
    for bucket in S3.get_all_buckets():
        if fnmatch.fnmatch(bucket.name, glob_str):
            globbed.append(bucket)
    return globbed
