#!/bin/bash
# Evaluate predictions produced by SIGMORPHON baseline. The predictions are required to be in one directory (`pred_dir`)
# Their expected names are as follows:
# SIG_UD_Basque-BDT_large_tagtransformer_ts150000_bs400_ds_dev.pred
#SIG_UD_Basque-BDT_large_tagtransformer_ts150000_bs400_ds_test.pred
#SIG_UD_Basque-BDT_large_tagtransformer_ts150000_bs800_ds_dev.pred
#SIG_UD_Basque-BDT_large_tagtransformer_ts150000_bs800_ds_test.pred
#SIG_UD_Basque-BDT_large_transformer_ts150000_bs400_ds_dev.pred
#SIG_UD_Basque-BDT_large_transformer_ts150000_bs400_ds_test.pred
#SIG_UD_Basque-BDT_large_transformer_ts150000_bs800_ds_dev.pred
#SIG_UD_Basque-BDT_large_transformer_ts150000_bs800_ds_test.pred
#SIG_UD_Breton-KEB_large_tagtransformer_ts150000_bs400_ds_dev.pred
#SIG_UD_Breton-KEB_large_tagtransformer_ts150000_bs400_ds_test.pred
#SIG_UD_Breton-KEB_large_tagtransformer_ts150000_bs800_ds_dev.pred
#SIG_UD_Breton-KEB_large_tagtransformer_ts150000_bs800_ds_test.pred
#SIG_UD_Breton-KEB_large_transformer_ts150000_bs400_ds_dev.pred
#SIG_UD_Breton-KEB_large_transformer_ts150000_bs400_ds_test.pred
#SIG_UD_Breton-KEB_large_transformer_ts150000_bs800_ds_dev.pred
#SIG_UD_Breton-KEB_large_transformer_ts150000_bs800_ds_test.pred
#SIG_UD_Czech-PDT_large_tagtransformer_ts150000_bs400_ds_dev.pred
#SIG_UD_Czech-PDT_large_tagtransformer_ts150000_bs400_ds_test.pred
#SIG_UD_Czech-PDT_large_tagtransformer_ts150000_bs800_ds_dev.pred
#SIG_UD_Czech-PDT_large_tagtransformer_ts150000_bs800_ds_test.pred
#SIG_UD_Czech-PDT_large_transformer_ts150000_bs400_ds_dev.pred
#SIG_UD_Czech-PDT_large_transformer_ts150000_bs400_ds_test.pred
#SIG_UD_Czech-PDT_large_transformer_ts150000_bs800_ds_dev.pred
#SIG_UD_Czech-PDT_large_transformer_ts150000_bs800_ds_test.pred
#SIG_UD_English-EWT_large_tagtransformer_ts150000_bs400_ds_dev.pred
#SIG_UD_English-EWT_large_tagtransformer_ts150000_bs400_ds_test.pred
#SIG_UD_English-EWT_large_tagtransformer_ts150000_bs800_ds_dev.pred
#SIG_UD_English-EWT_large_tagtransformer_ts150000_bs800_ds_test.pred
#SIG_UD_English-EWT_large_transformer_ts150000_bs400_ds_dev.pred
#SIG_UD_English-EWT_large_transformer_ts150000_bs400_ds_test.pred
#SIG_UD_English-EWT_large_transformer_ts150000_bs800_ds_dev.pred
#SIG_UD_English-EWT_large_transformer_ts150000_bs800_ds_test.pred
#SIG_UD_Spanish-AnCora_large_tagtransformer_ts150000_bs400_ds_dev.pred
#SIG_UD_Spanish-AnCora_large_tagtransformer_ts150000_bs400_ds_test.pred
#SIG_UD_Spanish-AnCora_large_tagtransformer_ts150000_bs800_ds_dev.pred
#SIG_UD_Spanish-AnCora_large_tagtransformer_ts150000_bs800_ds_test.pred
#SIG_UD_Spanish-AnCora_large_transformer_ts150000_bs400_ds_dev.pred
#SIG_UD_Spanish-AnCora_large_transformer_ts150000_bs400_ds_test.pred
#SIG_UD_Spanish-AnCora_large_transformer_ts150000_bs800_ds_dev.pred
#SIG_UD_Spanish-AnCora_large_transformer_ts150000_bs800_ds_test.pred

pred_dir="logs/sig-neural-bsln/2025-02-15_predictions/all_langs"

HOMEDIR=/storage/brno2/home/LOGIN/inflector
cd $HOMEDIR

module add python/python-3.10.4-intel-19.0.4-sc7snnf
PYTHON="/storage/brno2/home/LOGIN/dp-local/.venv/bin/python3"


corpora=("UD_Czech-PDT" "UD_English-EWT" "UD_Spanish-AnCora" "UD_Basque-BDT" "UD_Breton-KEB")
#corpora=("UD_Breton-KEB")
#corpora=("UD_Breton-KEB" "UD_Basque-BDT")


for model in tagtransformer_ts150000_bs400 tagtransformer_ts150000_bs800 transformer_ts150000_bs800 transformer_ts150000_bs400 ; do
  # Construct the dev_prediction_files variable
  dev_prediction_files=""

  for corpus in "${corpora[@]}"; do
      dev_prediction_files+="${pred_dir}/SIG_${corpus}_large_${model}_ds_dev.pred;"
  done

  # Remove the trailing semicolon
  dev_prediction_files=${dev_prediction_files%;}

  # Print the result
  echo "$dev_prediction_files"

  test_prediction_files=${dev_prediction_files//dev/test}

  echo "$test_prediction_files"

  resfile="logs/sig-neural-bsln/2025-02-15_predictions/results/${model}.res"
  fresfile="logs/sig-neural-bsln/2025-02-15_predictions/results/${model}.tsv"

  # Create a semicolon-separated corpora string
  corpora_string=$(IFS=";"; echo "${corpora[*]}")

  $PYTHON inflector_dataset.py --expid "first-sig-bsln" --corpora "${corpora_string}" --model_name $model --results_file $resfile --full_results_file $fresfile --dev_pred_files "${dev_prediction_files}" --test_pred_files "${test_prediction_files}"
done
