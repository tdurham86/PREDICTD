#! /usr/bin/env bash

#1) Point the code to the file map with file_url\tct\tassay and assemble the individual
#   data sets into a single bedgraph file.
python ~/proj/updated_imp_code/aws/predictd/assemble_new_ct_datasets.py --file_map=../fetal_spinal_cord_file_map.local.tab --working_dir=./

#2) Upload the assembled data to AWS S3 in manageable chunks
python ~/proj/updated_imp_code/aws/predictd/upload_genome_file_in_parts.py --input_bdg=./tmpYQZSLg.bdg.gz --out_url_base=s3://encodeimputation-alldata/predictd_demo/data/assembled_data --lines_per_file=1000000

#3) Use the data to create an RDD for imputation.
python /root/predictd/assembled_data_to_rdd.py --data_parts_url_base=s3://encodeimputation-alldata/predictd_demo/data/assembled_data --out_url=s3://encodeimputation-alldata/predictd_demo/data/fetal_spinal_cord.rdd.pickle --data_shape=1,24

#4) Train model and impute data

