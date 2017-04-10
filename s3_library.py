'''Library to encapsulate interactions with AWS S3.
'''

import boto
import fnmatch
import os
import pickle
from tempfile import NamedTemporaryFile

##with open(os.path.join(os.path.dirname(__file__), 's3_credentials.txt')) as creds:
#with open('/home/ec2-user/code_parallel_stochastic/predictd/s3_credentials.txt') as creds:
#    cred_info = dict([elt.strip().split('=') for elt in creds])
##print(cred_info)
#S3 = boto.connect_s3(**cred_info)
S3 = boto.connect_s3()
TMPDIR='/data/tmp'

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
        key.get_contents_to_file(tmp)
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
    with NamedTemporaryFile(dir=TMPDIR) as tmp:
        pickle.dump(obj, tmp)
        tmp.seek(0)
        key.set_contents_from_file(tmp)

def glob_keys(bucketname, glob_str):
    '''Query keys for the specified bucket and return keys with 
    names matching the glob string.
    '''
    globbed = []
    bucket = S3.get_bucket(bucketname)
    for key in bucket.list():
        if fnmatch.fnmatch(key.name, glob_str):
            globbed.append(key)
    return globbed

def glob_buckets(glob_str):
    '''Find buckets with names matching the provided glob string.
    '''
    globbed = []
    for bucket in S3.get_all_buckets():
        if fnmatch.fnmatch(bucket.name, glob_str):
            globbed.append(bucket)
    return globbed
