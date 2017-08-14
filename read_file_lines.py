#! /usr/bin/env python

import argparse
import smart_open
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', help='Path to a file. Can be a local file or s3 url.')
    args = parser.parse_args()

    with smart_open.smart_open(args.fpath) as lines_in:
        for line in lines_in:
            sys.stdout.write(line)
    sys.stdout.close()
