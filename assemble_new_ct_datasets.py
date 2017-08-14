import argparse
import bz2
import fnmatch
import glob
import gzip
import multiprocessing as mp
import numpy
import os
import pickle
import Queue
import random
import shutil
import smart_open
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zlib

sys.path.append(os.path.dirname(__file__))
import s3_library

class ExcProc(mp.Process):

    def __init__(self, paths_queue, exc_queue, out_dir, windows_path):
        mp.Process.__init__(self)
        self.exc_queue = exc_queue
        self.paths_queue = paths_queue
        self.out_dir = out_dir
        self.windows_path = windows_path
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

    def run(self):
        num_to_merge = 2
        try:
            while True:
                paths = []
                tries = 3
                while len(paths) < num_to_merge:
                    try:
                        paths.append(self.paths_queue.get(True, random.randint(10,15)))
                    except Queue.Empty:
                        if tries:
                            tries -= 1
                        else:
                            for elt in paths:
                                self.paths_queue.put_nowait(elt)
                            return
                    else:
                        tries = 3
#                #element 0 is for paths starting with data and element 1 is for dev
#                dircounts = [0,0]
#                inits = []
#                for elt in paths:
#                    if isinstance(elt, tuple):
#                        inits.append(1)
#                        pathroot = elt[1].strip('/').split('/')[0]
#                    else:
#                        inits.append(0)
#                        pathroot = elt.strip('/').split('/')[0]
#                    if pathroot == 'data':
#                        dircounts[0] += 1
#                    elif pathroot == 'dev':
#                        dircounts[1] += 1
#                    else:
#                        raise Exception('Did not recognize root dir: {!s}'.format(pathroot))
#                if dircounts[0] > dircounts[1]:
#                    out_dir = random.choice(['/dev/hierarchical', '/dev/shm/hierarchical'])
#                else:
#                    out_dir = '/data/hierarchical'
                inits = [1 if isinstance(elt, tuple) else 0 for elt in paths]
                for idx, typ in enumerate(inits):
                    if not typ:
                        continue
                    pth = paths[idx][1]
                    if not pth.endswith('.windowed.bdg.gz'):
                        if not pth.startswith('s3://'):
                            raise Exception('Must start with s3 path.')
                        pth = _make_windowed_bdg(pth, os.path.dirname(pth), self.windows_path, overwrite=False)
                    if pth.startswith('s3://'):
                        bucket_txt, key_txt = s3_library.parse_s3_url(pth)
                        local_pth = os.path.join(self.out_dir, os.path.basename(pth))
                        s3_library.S3.get_bucket(bucket_txt).get_key(key_txt).get_contents_to_filename(local_pth, headers={'x-amz-request-payer':'requester'})
                        pth = local_pth
                    paths[idx] = (paths[idx][0], pth)
#                if sum(inits) == 0:
#                    outname = write_to_proc(paths, out_dir=self.out_dir)
#                    self.paths_queue.put_nowait(outname)
#                elif sum(inits) == num_to_merge:
#                    coords, paths = [list(elt) for elt in zip(*paths)]
#                    outname = write_to_proc_init(coords, paths, out_dir=self.out_dir)
#                    self.paths_queue.put_nowait(outname)
#                    paths = []
#                else:
                for idx in [i for i in range(len(inits)) if inits[i]]:
                    coord, path = paths[idx]
                    outname = _write_coords([coord], out_dir=self.out_dir)
                    os.symlink(path, outname.replace('.coord_order.pickle', '.bdg.gz'))
                    paths[idx] = outname.replace('.coord_order.pickle', '.bdg.gz')
                outname = write_to_proc(paths, out_dir=self.out_dir)
                self.paths_queue.put_nowait(outname)

                for elt in paths:
                    if os.path.basename(elt).startswith('tmp'):
                        if os.path.islink(elt):
                            os.unlink(elt)
                        else:
                            os.remove(elt)
                        os.remove(elt.replace('.bdg.gz', '.coord_order.pickle'))
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e

def get_windows_file(path, working_dir):
    tmp_path = os.path.join(working_dir, os.path.basename(path))
    try:
        os.makedirs(tmp_path + '.lock')
    except:
        return tmp_path
    if not os.path.isfile(tmp_path):
        if path.startswith('s3://'):
            path = path.split('/')
            bucket_txt = path[2]
            key_txt = '/'.join(path[3:])
            key = s3_library.S3.get_bucket(bucket_txt).get_key(key_txt)
            key.get_contents_to_filename(tmp_path, headers={'x-amz-request-payer':'requester'})
        else:
            shutil.copy(path, tmp_path)
    return tmp_path

def split_gz_ext(path):
    if path.endswith('.gz'):
        splitpath = os.path.splitext(os.path.splitext(path)[0])
        return splitpath[0], splitpath[1] + '.gz'
    else:
        return os.path.splitext(path)

def _read_in_file_to_proc(path, proc_stdin):
    try:
        with smart_open.smart_open(path, 'rb') as lines_in:
            for line in lines_in:
#                with open(os.path.basename(path) + '.test.in', 'a') as out:
#                    out.write(line)
                proc_stdin.write(line)
    finally:
        proc_stdin.close()

def _make_windowed_bdg(path, s3_root, windows_path, overwrite=False):
    transformed_url = os.path.join(s3_root, os.path.basename(split_gz_ext(path)[0]) + '.windowed.bdg')
    bucket_txt, key_txt = s3_library.parse_s3_url(transformed_url)
    if s3_library.S3.get_bucket(bucket_txt).get_key(key_txt + '.gz'):
        return transformed_url + '.gz'
    cmd = ['bedtools', 'map', '-a', windows_path, '-b', 'stdin', '-c', '4', '-o', 'mean']
    map_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    read_file_thread = threading.Thread(target=_read_in_file_to_proc, args=[path, map_proc.stdin])
    read_file_thread.daemon = True
    read_file_thread.start()

    compressor = zlib.compressobj(6, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    with map_proc.stdout as bedgraph_in:
        with smart_open.smart_open(transformed_url, 'wb') as data_out:
            line_buffer, val_buffer = [], []
            for line in bedgraph_in:
#                with open(os.path.basename(transformed_url) + '.test', 'a') as out:
#                    out.write(line)
                line = line.strip()
                if line[-1] == '.':
                    line = line.rstrip('.') + '0'
                line_buffer.append(line)
                if len(line_buffer) == 1e5:
                    lines = '\n'.join(line_buffer) + '\n'
                    data_out.write(compressor.compress(lines))
                    line_buffer, val_buffer = [], []
            else:
                lines = '\n'.join(line_buffer) + '\n'
                data_out.write(compressor.compress(lines))
                data_out.write(compressor.flush(zlib.Z_FINISH))
    key = s3_library.S3.get_bucket(bucket_txt).get_key(key_txt)
    key.copy(bucket_txt, key_txt + '.gz')
    key.delete()
    return transformed_url + '.gz'

def _unionbg(paths, outname):
    bedtools_cmd = ['bedtools', 'unionbedg', '-i'] + paths
    bedtools_proc = subprocess.Popen(bedtools_cmd, stdout=subprocess.PIPE)
    with gzip.open(outname, 'wb') as out:
        with bedtools_proc.stdout as lines_in:
            for line in lines_in:
                line = line.strip().split()
                if len(line[4:]) != (len(paths) - 1):
                    raise Exception('Line does not have enough fields: {!s}'.format((bedtools_cmd, line)))
                out.write('\t'.join(line[:4]) + '|' + '|'.join(line[4:]) + '\n')

def _write_coords(coord_list, out_dir):
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.coord_order.pickle', dir=out_dir, delete=False) as out:
        outname = out.name
        pickle.dump(coord_list, out)
    return outname

#def write_to_proc_init(coords, paths, out_dir):
#    outname = _write_coords(list(coords), out_dir=out_dir).replace('.coord_order.pickle', '.bdg.gz')
#    _unionbg(list(paths), outname)
#    return outname

def write_to_proc(paths, out_dir):
    coord_list = []
    for path in paths:
        with open(split_gz_ext(path)[0] + '.coord_order.pickle', 'rb') as pickle_in:
            coord_list.extend(pickle.load(pickle_in))

    outname = _write_coords(coord_list, out_dir=out_dir).replace('.coord_order.pickle', '.bdg.gz')
    _unionbg(paths, outname)
    return outname

#cached_keys = None
#def get_key_url(cell_type, assay, s3_root, use_cached_keys=True):
#    global cached_keys
#    bucket_txt, key_root_txt = s3_library.parse_s3_url(s3_root)
#    if cached_keys is None or use_cached_keys is False:
#        bedgraph_glob = os.path.join(key_root_txt, '*', '*', '*.bdg.gz')
#        cached_keys = s3_library.glob_keys(bucket_txt, bedgraph_glob)
#    bedgraph_glob = os.path.join(key_root_txt, cell_type, assay, '*.bdg.gz')
#    try:
#        bedgraph_key = [elt for elt in cached_keys if fnmatch.fnmatch(elt.name, bedgraph_glob)][0]
#    except IndexError:
#        raise Exception('Could not find matching s3 key for {!s}'.format(bedgraph_glob))
#    return os.path.join('/data', os.path.basename(bedgraph_key.name).replace('.bz2', '.gz'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_map')
#    parser.add_argument('--data_root')
    parser.add_argument('--working_dir', default=os.getcwd())
    parser.add_argument('--windows_file', default='s3://encodeimputation-alldata/25bp/hg19.25bp.windows.bed.gz')
#    parser.add_argument('--data_idx', default='s3://encodeimputation-alldata/25bp/data_idx.pickle')
    parser.add_argument('--procnum', type=int, default=4)
    args = parser.parse_args()

    print('Getting windows file.')
    windows_file = get_windows_file(args.windows_file, args.working_dir)

    #read in filemap and set order (row coordinates) for new cell types
    print('Reading in file map and setting tensor cell type and assay indices')
    with smart_open.smart_open(args.file_map, 'r') as lines_in:
        filemap = numpy.array([elt.strip().split() for elt in lines_in])
    print(filemap)
    sorted_cts = sorted(set(filemap[:,1]))

    #get assay order (column coordinates)
    sorted_assays = sorted(set(filemap[:,2]))
#    bucket_txt, key_txt = s3_library.parse_s3_url(args.data_idx)
#    data_idx = s3_library.get_pickle_s3(bucket_txt, key_txt)
#    sorted_assays = [elt[0] for elt in 
#                     sorted(set((elt[1], elt[-1][1]) for elt in data_idx.values()), key=lambda x:x[1])]

    #set coordinates for each experiment in the filemap
    coords = numpy.array([(sorted_cts.index(filemap[i,1]), sorted_assays.index(filemap[i,2]))
                          for i in range(filemap.shape[0])])

    #save pickled filemap with coordinates
    print('Saving a copy of the file map with tensor coordinates in Python pickle format.')
    fmap_w_coords = numpy.hstack([filemap, coords])
    fmap_w_coords_path = os.path.splitext(args.file_map)[0] + '.w_coords.pickle'
    with smart_open.smart_open(fmap_w_coords_path, 'wb') as out:
        out.write(pickle.dumps(fmap_w_coords))
    #also save a local copy so that the upload script can transform it to data_idx
    with open(os.path.join(args.working_dir, os.path.basename(fmap_w_coords_path)), 'wb') as out:
        pickle.dump(fmap_w_coords, out)

    #set up multiprocessing queue and feed in the paths to be joined with unionbg
    print('Combining data files.')
    paths_queue = mp.Queue()
    for i in xrange(filemap.shape[0]):
        paths_queue.put_nowait((tuple(coords[i,:]), filemap[i][0]))

    print('Starting Path Queue Length: {!s}'.format(paths_queue.qsize()))

    procnum = args.procnum
    proclist = []
    exc_queues = []
    for i in range(procnum):
        exc_queues.append(mp.Queue())
        proclist.append(ExcProc(paths_queue, exc_queues[-1], args.working_dir, windows_file))
        proclist[-1].start()

    try:
        while True:
            alive_count = 0
            for idx in range(len(proclist)):
                proc = proclist[idx]
                exc_queue = exc_queues[idx]
                if not proc.is_alive():
                    try:
                        exc = exc_queue.get(block=False)
                    except Queue.Empty:
                        pass
                    else:
                        exc_type, exc_obj, exc_trace = exc
                        raise Exception(exc_type, exc_obj, traceback.format_exc(exc_trace))
                else:
                    alive_count += 1
            else:
                if alive_count:
                    print('{!s} processes still running. Paths Queue Length: {!s}'.format(alive_count, paths_queue.qsize()))
                    time.sleep(60)
                else:
                    break
    except Exception as err:
        for elt in proclist:
            elt.terminate()
        raise err

    print('Multiprocessing complete.')
#    print('Multiprocessing complete. Finishing last bedtools unionbedg.')
#    data_files = glob.glob('/data/hierarchical/*.bedGraph.gz')
#    outname = write_to_proc(data_files)
#    os.rename(outname, '/data/hierarchical/alldata.25bp.bedGraph.gz')
#    os.rename(outname.replace('.bedGraph.gz', '.coord_order.pickle'), '/data/hierarchical/alldata.25bp.coord_order.pickle')
    print('Done.')
