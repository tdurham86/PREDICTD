
import argparse
import glob
import gzip
import os
import pickle
import smart_open
import sys
import zlib

sys.path.append(os.path.dirname(__file__))
import s3_library

def write_lines(lines, out_path):
    compressor = zlib.compressobj(6, zlib.DEFLATED, zlib.MAX_WBITS | 16)
    with smart_open.smart_open(out_path, 'wb') as out:
        stepsize = int(1e5)
        idx = 0
        while idx < len(lines):
            next_idx = idx + stepsize
            out.write(compressor.compress(''.join(lines[idx:next_idx])))
            idx = next_idx
        else:
            out.write(compressor.flush(zlib.Z_FINISH))
    out_bucket_txt, out_key_txt = s3_library.parse_s3_url(out_path)
    out_key = s3_library.S3.get_bucket(out_bucket_txt).get_key(out_key_txt)
    out_key.copy(out_bucket_txt, out_key_txt + '.gz')
    out_key.delete()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_bdg', help='Local gzipped bedgraph file that is the '
                        'output of assemble_new_ct_datasets.py')
    parser.add_argument('--out_url_base', default='s3://encodeimputation-alldata/25bp/alldata-parts/alldata')
    parser.add_argument('--lines_per_file', type=int, default=1000000)
    args = parser.parse_args()

    part_no = 0
    out_path_base = args.out_url_base + '.part{!s}.txt'
    bucket_txt, key_txt = s3_library.parse_s3_url(args.out_url_base)
    out_bucket = s3_library.S3.get_bucket(bucket_txt)

    #generate the data_idx.pickle for these data based on the file_map + coords
    fmap_w_coords_path = glob.glob(os.path.join(os.path.dirname(args.input_bdg), '*.w_coords.pickle'))[0]
    with open(fmap_w_coords_path, 'rb') as pickle_in:
        fmap_w_coords = pickle.load(pickle_in)
    data_idx = {fmap_w_coords[i][0]:(fmap_w_coords[i][1], fmap_w_coords[i][2], tuple([int(elt) for elt in fmap_w_coords[i][3:5]])) 
                for i in range(fmap_w_coords.shape[0])}
    data_idx_key = out_bucket.new_key(os.path.join(os.path.dirname(key_txt), 'data_idx.pickle'))
    data_idx_key.set_contents_from_string(pickle.dumps(data_idx), headers={'x-amz-request-payer':'requester'})

    #save the column coordinates to s3
    coord_key = out_bucket.new_key(key_txt + '.coord_order.pickle')
    coord_key.set_contents_from_filename(args.input_bdg.replace('.bdg.gz', '.coord_order.pickle'), headers={'x-amz-request-payer':'requester'})

    #save the data in parts
    with gzip.open(args.input_bdg, 'rb') as gzip_in:
        line_buf = []
        num_lines = 0
        for line in gzip_in:
            line_buf.append(line)
            num_lines += 1
            if not num_lines % args.lines_per_file:
                out_path = out_path_base.format(part_no)
                write_lines(line_buf, out_path)
                print(out_path + '.gz')
                line_buf = []
                part_no += 1
        else:
            out_path = out_path_base.format(part_no)
            write_lines(line_buf, out_path)
            print(out_path + '.gz')

    #record the number of parts saved for ease of access later
    num_parts = os.path.join(os.path.dirname(args.out_url_base), 'num_parts.pickle')
    with smart_open.smart_open(num_parts, 'wb') as out:
        out.write(pickle.dumps(part_no + 1))

