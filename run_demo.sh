#! /usr/bin/env bash

out_url=$1
strip_prot=${out_url#s3://}
out_bucket=${strip_prot%%/*}
out_key=${strip_prot#*/}

DATA_URL=s3://encodeimputation-alldata/25bp/alldata.25bp.hars_plus_encodePilots.2500.coords_adjusted.rdd.pickle

spark-submit ./impute_roadmap_consolidated_data.py --data_url=$DATA_URL --run_bucket=$out_bucket --out_root=$out_key --fold_idx=0 --valid_fold_idx=0 --factor_init_seed=123 -f 100 -l 0.005 --beta1=0.9 --beta2=0.999 --epsilon=1e-8 --batch_size=5000 --win_per_slice=10000 --rc=3.66 --ra=1.23e-7 --ri=1.23e-5 --ri2=2.9 --rbc=0 --rba=0 --rbi=0 --stop_winsize=15 --stop_winspacing=5 --stop_win2shift=1e-05 --stop_pval=0.05 --min_iters=50 --iters_per_mse=3 --lrate_search_num=3 --folds_fname=folds.5.8.pickle --no_bw_sinh --training_fraction=0.01 --training_fraction_seed=89
