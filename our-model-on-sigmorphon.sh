#!/bin/bash
# Downloads SIGMORPHON 2022 and 2023 data, trains the system for all the languages, runs official SIGMORPHON evaluation, and prints a nice TeX comparison table comparing all our systems with nonneural and neural baseline on DEV.

# TODO: download (or just stage them in the repository) also the official results of bslns: from https://github.com/sigmorphon/2022InflectionST/tree/main/evaluation/baseline_results/part1 , rename files, remove consecutive TAB characters
# DONE: process results on test set, print comparison table comparing our system with all competing systems and OOV paper
# TODO: process results of SIGMORPHON 2023: unfortunately there are no dev results of bslns released, we would have to produce them ourselves
# TODO: incorporate checkpoint selection
# TODO: incorporate using better hyperparams
# TODO: rewrite such that all the sigmorphon scripts could be placed in different than the root directory

PYTHON=$1
ROOTDIR=$2

# Set boolean variables, which years to run
run_sig22=true
run_sig23=true

# TODO: in OOVs paper, we had 9.5k epochs (!), but with checkpoint selection

# 960
EPOCHS=960
DESC="mono-${EPOCHS}"

RUN_SETUP="inflector.py --eval_also_on_test  --checkpoint_selection --acc_for_ckpt_selection accuracy --weighted_accuracy_during_training --weight_decay 0.01 --clip_grad_norm 1.0 --lr 0.001 --lr_decay cosine --batch_size 512 --epochs ${EPOCHS}  --joint_vocab --trm_layer_dim 256 --trm_layer_count 3 --trm_attn_heads 4 --trm_ff_nn_dim 64 --trm_drop 0.15 --trm_attn_drop 0.1 --trm_feed_drop 0.35 --trm_layer_drop 0.2 --jobid ${PBS_JOBID} --model transformer"

# with ckpt selection
RUN_SETUP_MULTI="inflector.py --checkpoint_selection --acc_for_ckpt_selection multiling-acc --eval_also_on_test --multilingual_training --multilingual_corpus_down_up_sampling_temperature 0.5 --weighted_multilingual_accuracy_during_training --weight_decay 0.01 --clip_grad_norm 1.0 --lr 0.001 --lr_decay cosine --batch_size 512 --epochs ${EPOCHS}  --joint_vocab --trm_layer_dim 256 --trm_layer_count 3 --trm_attn_heads 4 --trm_ff_nn_dim 64 --trm_drop 0.15 --trm_attn_drop 0.1 --trm_feed_drop 0.35 --trm_layer_drop 0.2 --jobid $PBS_JOBID --model transformer"

# Download data, convert to our format and put to out directories
bash prepare_sigmorphon_data.sh

DATE="$(date +'%Y-%m-%d_%H-%M-%S')"

prediction_dir="logs/sigmorphon/$DATE"

if [ "$run_sig22" = true ]; then
  pred22=$prediction_dir/22
  mkdir -p $pred22


  results_dir="results/sigmorphon/22/large-trainset-full"
  mkdir -p $results_dir

  table_dir="results/sigmorphon/22/large-trainset-feats-overlap/comparison"
  mkdir -p $table_dir

  results_file_pref="$results_dir/${DATE}_epochs=${DESC}"

  # RUN TRAINING AND EVALUATION ON SIGMORPHON 2022

  corpus22="ang;ara;asm;evn;got;heb;hun;kat;khk;kor;krl;lud;non;pol;poma;slk;tur;vep"

  #$PYTHON inflector.py --data "sig22" --args "bp_rnn" --corpus="ang" --predictions_dir $pred22 --epochs 4

  expid="${DESC}-sig22"

  $PYTHON $RUN_SETUP --data "sig22" --corpus $corpus22 --predictions_dir $pred22 --expid $expid #--datadir $ROOTDIR/sig-data/22
  #$PYTHON inflector.py --epochs $EPOCHS --data "sig22" --args $PARAMS --corpus $corpus22 --predictions_dir $pred22

  cd $pred22

  # rename the prediction files to have `_large` in the name, since it is expected from the eval script
  # and copy the form column 3times, since the format is expected to be "sth \t form \t sth" (the rest is ignored but it has to be there)
  for file in $(ls .); do
    # Extract the base name and extension
    lang_set="${file%.*}"  # This removes the extension
    set="${file##*.}"      # This gets the extension part after the last dot

    # Construct the new file name
    new_name="${lang_set}_large.${set}"

    # create dummy lemma-file with correct length (string "lemma" repeated the correct number of times)
    awk '{print "lemma"}' $file > dummy_lemma_file.txt

    paste dummy_lemma_file.txt "$file" "$file" > "$new_name"
    rm $file
    rm dummy_lemma_file.txt
  done

  cd ../../../../

  for SET in dev test; do
    $PYTHON official_sigmorphon_evaluation.py $pred22 data/raw/2022InflectionST/part1/development_languages/ --evaltype "${SET}" --sig_year 2022 > "${results_file_pref}.${SET}"
    $PYTHON merge_sigmorphon_results.py --directory $results_dir --suffix $SET > $table_dir/$DATE.$SET.tex
    echo "Result comparison printed to file $table_dir/$DATE.$SET.tex"
  done

  # column 4, as we are interested in accuracy on feature-overlap on SIG22
  $PYTHON merge_sigmorphon_results.py --directory $results_dir --suffix test --existing_table "results/sigmorphon/22/large-trainset-feats-overlap/oovs-paper-results.tex" --skip_langs="LARGE;SMALL;TOTAL;_small;kaz;hye;heb;evn" --column 4 > $table_dir/$DATE.test.FULL-TABLE.tex



fi


##########################################################

if [ "$run_sig23" = true ]; then

  pred23=$prediction_dir/23
  mkdir -p $pred23

  results_dir="results/sigmorphon/23/large-trainset-full"
  mkdir -p $results_dir

  results_file_pref="$results_dir/${DATE}_epochs=${DESC}"

  table_dir="results/sigmorphon/23/comparison"
  mkdir -p $table_dir

  corpus23="afb;amh;arz;bel;dan;deu;eng;fin;fra;grc;heb;heb_unvoc;hun;hye;ita;jap;kat;klr;mkd;nav;rus;san;sme;spa;sqi;swa;tur"

  # RUN TRAINING AND EVALUATION ON SIGMORPHON 2023
  expid="${DESC}-sig23"

  #$PYTHON inflector.py --data "sig23" --args "bp_rnn" --corpus="eng" --predictions_dir $pred23 --epochs 4
  $PYTHON $RUN_SETUP --data "sig23" --corpus $corpus23 --predictions_dir $pred23  --expid $expid  # --datadir $ROOTDIR/sig-data/23

  cd $pred23

  # copy the form column 3times, since the format is expected to be "sth \t form \t sth" (the rest is ignored but it has to be there)
  for file in $(ls .); do
    
    # create dummy lemma-file with correct length (string "lemma" repeated the correct number of times)
    awk '{print "lemma"}' $file > dummy_lemma_file.txt

    paste dummy_lemma_file.txt "$file" "$file" > tmp
    mv tmp $file
    rm dummy_lemma_file.txt
  done

  cd ../../../../

  for SET in dev test; do
    $PYTHON official_sigmorphon_evaluation.py $pred23 data/raw/2023InflectionST/part1/data/ --evaltype $SET --sig_year 2023 > "${results_file_pref}.${SET}"
  done
  
  # column 1, as we are interested in overall accuracy on SIG23
  $PYTHON merge_sigmorphon_results.py --directory $results_dir --suffix test --existing_table "results/sigmorphon/23/original_paper_table.tex" --column 1 > $table_dir/$DATE.test.FULL-TABLE.tex
fi

