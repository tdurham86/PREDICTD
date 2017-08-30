#! /usr/bin/env bash

#e.g. s3://data-tdurham/fetalspinalcord_analysis/signal_files/fetal_spinal_cord_filemap.txt
filemap_url=$1
#e.g. s3://data-tdurham/fetalspinalcord_analysis/
out_root_url=$2

if [ -z "$filemap_url" ] || [ -z $out_root_url ]
then
    echo "Usage:  new_ct_demo.sh s3://file_map_bucket/file/map/key s3://output_bucket/out/root/key"
    exit 1
fi

#1) Point the code to the file map with file_url\tct\tassay and assemble the individual
#   data sets into a single bedgraph file.
echo "Assembling cell type data sets"
python /root/predictd/assemble_new_ct_datasets.py --file_map=$filemap_url --working_dir=/data/tmp/ --windows_file=s3://encodeimputation-alldata/25bp/hg19.25bp.windows.bed.gz --procnum=4

#2) Upload the assembled data to AWS S3 in manageable chunks
echo "Uploading assembled data to S3"
bdg_path=`ls tmp*.bdg.gz`
assembled_data_url_base=$out_root_url/signal_data/assembled_data
python /root/predictd/upload_genome_file_in_parts.py --input_bdg=${bdg_path[0]} --out_url_base=$assembled_data_url_base --lines_per_file=1000000

#3) Use the data to create an RDD for imputation.
echo "Transforming assembled data to rdd tensor data structure"
new_rdd_url=$out_root_url/fetalspinalcord_data.hars_with_encodePilots.25bp.rdd.pickle
spark-submit /root/predictd/assembled_data_to_rdd.py --data_parts_url_base=$assembled_data_url_base --coords_bed=s3://encodeimputation-alldata/25bp/hars_with_encodePilots.25bp.windows.bed.gz --out_url=$new_rdd_url

#4) Train model
echo "Training PREDICTD model"
model_out=$out_root_url/test_out
strip_prot=${model_out#s3://}
model_out_bucket=${strip_prot%%/*}
model_out_key=${strip_prot#*/}
roadmap_data_root=s3://encodeimputation-alldata/25bp
roadmap_data_rdd=$roadmap_data_root/alldata.25bp.hars_plus_encodePilots.2500.coords_adjusted.rdd.pickle

spark-submit /root/predictd/impute_data.py --data_url=$roadmap_data_rdd --addl_data_url=$new_rdd_url --run_bucket=$model_out_bucket --out_root=$model_out_key --fold_idx=0 --valid_fold_idx=0 --latent_factors=100 --learning_rate=0.005 --beta1=0.9 --beta2=0.999 --epsilon=1e-8 --batch_size=5000 --win_per_slice=1000 --rc=3.66 --ra=1.23e-7 --ri=1.23e-5 --rbc=0 --rba=0 --rbi=0 --stop_winsize=15 --stop_winspacing=5 --min_iters=50 --lrate_search_num=3 --iters_per_mse=3 --folds_fname=folds.5.8.pickle --training_fraction=0.01 --factor_init_seed=132 --no_browser_tracks
#spark-submit /root/predictd/train_bags.py --data_url=$roadmap_data_rdd --addl_data_url=$new_rdd_url --run_bucket=$model_out_bucket --out_root=$model_out_key --fold_idx=0 --latent_factors=100 --learning_rate=0.005 --beta1=0.9 --beta2=0.999 --epsilon=1e-8 --batch_size=5000 --win_per_slice=1000 --rc=3.66 --ra=1.23e-7 --ri=1.23e-5 --rbc=0 --rba=0 --rbi=0 --stop_winsize=15 --stop_winspacing=5 --min_iters=50 --lrate_search_num=3 --iters_per_mse=3 --folds_fname=folds.5.8.pickle --training_fraction=0.01 --factor_init_seed=132 --no_browser_tracks

#5) Impute data
echo "Generating whole genome imputed tracks"
spark-submit ./impute_with_models.py --model_url=$model_out --model_data_idx=$model_out/data_idx.pickle --addl_agg_coord_order=$assembled_data_url_base.coord_order.pickle --addl_agg_data_idx=${assembled_data_url_base%/*}/data_idx.pickle --agg_data_idx=$roadmap_data_root/data_idx.pickle --agg_coord_order=$roadmap_data_root/alldata.column_coords.pickle --agg_parts=s3://data-tdurham/fetalspinalcord_analysis/smalltest_alldata --cts_to_impute=Fetal_Spinal_Cord --out_root_url=$model_out/imputed_tracks
