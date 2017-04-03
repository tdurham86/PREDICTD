#! /usr/bin/env python

import argparse
import fnmatch
import io
import json
import numpy
import os
import pickle
import re
import sys

#from azure.storage import BlobService, TableService, Entity
from azure.storage.blob import BlobService
from azure.common import AzureMissingResourceHttpError

with open(os.path.join(os.path.dirname(__file__), 'blob_credentials.txt')) as creds_in:
    cred_info = dict([elt.strip().split('=') for elt in creds_in])
BLOB_SERVICE = BlobService(**cred_info)

def parse_azure_url(azure_url):
    '''Parse out and return the container and blob names from an azure wasb url.
    '''
    re_pattern = r'^wasbs{0,1}://(?P<container>[^@]+)@(?P<account>[^@.]+).blob.core.windows.net/(?P<blob>.+$)'
    match = re.match(re_pattern, azure_url)
    if match:
        return (match.group('container'), match.group('blob'), match.group('account'))
    else:
        raise Exception('Could not parse Azure URL: {!s}'.format(azure_url))

def create_container(container, blob_service=None):
    '''Call the BlobService API to create the specified container.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    blob_service.create_container(container, fail_on_exist=False)

def load_blob_from_path(container, blob_name, path_to_file, blob_service=None):
    '''Load a file into the imputationstore blob storage on Azure.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    create_container(container, blob_service=blob_service)
    blob_service.put_block_blob_from_path(container, blob_name, path_to_file)

def load_blob_from_text(container, blob_name, blob_str, blob_service=None):
    '''Save a string of text as a blob on Azure.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    create_container(container, blob_service=blob_service)
    blob_service.put_block_blob_from_text(container, blob_name, blob_str)

def load_blob_pickle(container, blob_name, obj, blob_service=None):
    '''Takes an object, pickles it to a byte stream, and loads that pickled 
    representation to Azure Blob storage.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    create_container(container, blob_service=blob_service)
    io_stream = io.BytesIO()
    pickle.dump(obj, io_stream)
    io_stream.seek(0)
    blob_service.put_block_blob_from_file(container, blob_name, io_stream)

def get_blob_to_path(container, blob_name, path_to_file, blob_service=None):
    '''Store data to a file from Azure blob storage.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    tries = 3
    while tries:
        try:
            blob_service.get_blob_to_path(container, blob_name, path_to_file)
        except Exception as err:
            tries -= 1
        else:
            break
    else:
        raise err

def get_blob_to_text(container, blob_name, blob_service=None):
    '''Gets a text blob from Azure and returns it as a string.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    return str(blob_service.get_blob_to_text(container, blob_name))

def get_blob_pickle(container, blob_name, blob_service=None):
    '''Gets a blob from Azure Blob Storage to a BytesIO object, unpickles the
    byte stream, and returns the Python object.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    io_stream = io.BytesIO()
    blob_service.get_blob_to_file(container, blob_name, io_stream)
    io_stream.seek(0)
    return pickle.load(io_stream)

def delete_blob_dir(container, blob_prefix, blob_service=None):
    '''Call the blob_service API to find all blobs in a particular virtual
    directory and delete them.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    to_delete = glob_blobs(container, os.path.join(blob_prefix, '*'))
    for blob in to_delete:
        delete_blob(container, blob.name)
    try:
        delete_blob(container, blob_prefix.rstrip('/'))
    except AzureMissingResourceHttpError:
        pass

def delete_blob(container, blob_name, blob_service=None):
    '''Call the blob_service API to delete the specified blob.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    blob_service.delete_blob(container, blob_name)

def delete_container(container, blob_service=None):
    '''Call the BlobService API to delete the specified container.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    blob_service.delete_container(container, fail_not_exist=False)

def ls_blobs(container, max_list_size=5000, blob_service=None):
    '''List the blob elements from the specified container.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    blobs = []
    next_marker = None
    while True:
        blob_page = blob_service.list_blobs(container, maxresults=max_list_size, 
                                            marker=next_marker)
        next_marker = blob_page.next_marker
        blobs.extend([elt for elt in blob_page])
        if not next_marker:
            break
    return blobs

def glob_blobs(container, blob_glob_str, max_list_size=5000, blob_service=None):
    '''Returns blob names that match the provided glob string.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    globbed = []
    for blob in ls_blobs(container, max_list_size=max_list_size, 
                         blob_service=blob_service):
        if fnmatch.fnmatch(blob.name, blob_glob_str):
            globbed.append(blob)
    return globbed

def ls_containers(max_list_size=5000, blob_service=None):
    '''List all containers for the specified blob service 
    (imputationstore by default).
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    containers = []
    next_marker = None
    while True:
        container_page = blob_service.list_containers(marker=next_marker,
                                                      maxresults=max_list_size)
        next_marker = container_page.next_marker
        containers.extend([elt.name for elt in container_page])
        if not next_marker:
            break
    return containers

def glob_containers(container_glob_str, max_list_size=5000, blob_service=BLOB_SERVICE):
    '''Returns container names that match the provided glob string.
    '''
    globbed = []
    for container in ls_containers(max_list_size=max_list_size,
                                   blob_service=blob_service):
        if fnmatch.fnmatch(container, container_glob_str):
            globbed.append(container)
    return globbed

def make_blob_url(container, blob, blob_service=None, prot='wasbs'):
    '''Wrapper for the make_blob_url azure API call.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    if prot in ['http', 'https']:
        return blob_service.make_blob_url(container, blob, protocol=prot)
    elif prot in ['wasb', 'wasbs']:
        return '{!s}://{!s}@{!s}.blob.core.windows.net/{!s}'.format(prot, container, blob_service.account_name, blob)
    else:
        raise Exception('Unrecognized protocol type: {!s}'.format(prot))

def cp_blob(container, src_blob, dst_blob, dst_container=None, blob_service=None):
    '''Copy a blob within a container by default. User can also specify a 
    different destination container to which to copy the source blob if 
    copying a blob between containers.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    src_url = make_blob_url(container, src_blob, prot='https')
    dst_container = container if dst_container is None else dst_container
    blob_service.copy_blob(dst_container, dst_blob, src_url)

def mv_blob(container, src_blob, dst_blob, dst_container=None, blob_service=None):
    '''Run cp_blob, and then delete the source blob once it has completed.
    '''
    blob_service = BLOB_SERVICE if blob_service is None else blob_service
    cp_blob(container, src_blob, dst_blob, dst_container=dst_container, blob_service=blob_service)
    delte_blob(container, src_blob)
