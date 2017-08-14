#! /usr/bin/env bash

#1) Point the code to the file map with file_url\tct\tassay and assemble the individual
#   data sets into a single bedgraph file.
#python /root/predictd/assemble_new_ct_datasets.py --file_map=s3://data-tdurham/fetalspinalcord_analysis/signal_files/fetal_spinal_cord_filemap.txt --working_dir=/data/tmp/ --windows_file=s3://encodeimputation-alldata/25bp/hg19.25bp.windows.bed.gz --procnum=4

#2) Upload the assembled data to AWS S3 in manageable chunks
#bdg_path=`ls tmp*.bdg.gz`
#python /root/predictd/upload_genome_file_in_parts.py --input_bdg=${bdg_path[0]} --out_url_base=s3://data-tdurham/fetalspinalcord_analysis/signal_data/assembled_data --lines_per_file=1000000

#3) Use the data to create an RDD for imputation.
spark-submit /root/predictd/assembled_data_to_rdd.py --data_parts_url_base=s3://data-tdurham/fetalspinalcord_analysis/signal_data/assembled_data --coords_bed=s3://encodeimputation-alldata/25bp/hars_with_encodePilots.25bp.windows.bed.gz --out_url=s3://data-tdurham/fetalspinalcord_analysis/fetalspinalcord_data.hars_with_encodePilots.25bp.rdd.pickle

#4) Train model and impute data

