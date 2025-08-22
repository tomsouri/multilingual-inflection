#!/bin/bash

# if running as a job, also pass job ID for identification
JOBID="$PBS_JOBID"

DATADIR="data/processed/ud-inflection/ud-treebanks-v2.14"
PYTHON=".venv/bin/python"

mkdir -p logs/
mkdir -p results/


development_langs="UD_Breton-KEB;UD_Spanish-AnCora;UD_Czech-PDT;UD_English-EWT;UD_Basque-BDT"
all="UD_Breton-KEB;UD_Spanish-AnCora;UD_Czech-PDT;UD_English-EWT;UD_Basque-BDT;UD_Slovak-SNK;UD_Afrikaans-AfriBooms;UD_Ancient_Greek-PROIEL;UD_Ancient_Hebrew-PTNK;UD_Arabic-PADT;UD_Armenian-ArmTDP;UD_Belarusian-HSE;UD_Bulgarian-BTB;UD_Catalan-AnCora;UD_Chinese-GSDSimp;UD_Classical_Armenian-CAVaL;UD_Classical_Chinese-Kyoto;UD_Coptic-Scriptorium;UD_Croatian-SET;UD_Danish-DDT;UD_Dutch-Alpino;UD_Erzya-JR;UD_Estonian-EDT;UD_Finnish-TDT;UD_French-GSD;UD_Galician-TreeGal;UD_Georgian-GLC;UD_German-GSD;UD_Gothic-PROIEL;UD_Greek-GDT;UD_Hebrew-HTB;UD_Hindi-HDTB;UD_Hungarian-Szeged;UD_Icelandic-Modern;UD_Indonesian-GSD;UD_Irish-IDT;UD_Italian-ISDT;UD_Japanese-GSDLUW;UD_Korean-Kaist;UD_Kyrgyz-KTMU;UD_Latin-ITTB;UD_Latvian-LVTB;UD_Lithuanian-ALKSNIS;UD_Low_Saxon-LSDC;UD_Maghrebi_Arabic_French-Arabizi;UD_Manx-Cadhan;UD_Marathi-UFAL;UD_Naija-NSC;UD_North_Sami-Giella;UD_Norwegian-Bokmaal;UD_Old_Church_Slavonic-PROIEL;UD_Old_East_Slavic-TOROT;UD_Old_French-PROFITEROLE;UD_Ottoman_Turkish-BOUN;UD_Persian-PerDT;UD_Polish-PDB;UD_Pomak-Philotis;UD_Portuguese-Bosque;UD_Romanian-RRT;UD_Russian-SynTagRus;UD_Sanskrit-Vedic;UD_Scottish_Gaelic-ARCOSG;UD_Slovenian-SSJ;UD_Swedish-Talbanken;UD_Tamil-TTB;UD_Turkish-BOUN;UD_Ukrainian-IU;UD_Urdu-UDTB;UD_Uyghur-UDT;UD_Vietnamese-VTB;UD_Welsh-CCG;UD_Western_Armenian-ArmTDP;UD_Wolof-WTB"

model="transformer"


########### A TRIAL TO CHECK THAT EVERYTHING RUNS ##############################
expid="exp0-first-experiment"
corpus="UD_Breton-KEB;UD_English-EWT"
e=1

# tr-mono - mini (1 epoch, 2 langs, tiny capacity), just to check that it runs smoothly
$PYTHON inflector.py --eval_also_on_test  --checkpoint_selection --acc_for_ckpt_selection "w-accuracy" --weighted_accuracy_during_training --weight_decay 0.01 --clip_grad_norm 1.0 --lr 0.001 --lr_decay "cosine" --batch_size 4 --epochs $e  --joint_vocab --trm_layer_dim 2 --trm_layer_count 1 --trm_attn_heads 1 --trm_ff_nn_dim 2 --trm_drop 0.15 --trm_attn_drop 0.1 --trm_feed_drop 0.35 --trm_layer_drop 0.2 --expid $expid --datadir $DATADIR --jobid $JOBID --model $model --corpus $corpus

# tr-multi - mini (1 epoch, 2 langs, tiny capacity), just to check that it runs smoothly
$PYTHON inflector.py --eval_also_on_test --multilingual_training --multilingual_corpus_down_up_sampling_temperature 0.5 --weighted_multilingual_accuracy_during_training --weight_decay 0.01 --clip_grad_norm 1.0 --lr 0.001 --lr_decay "cosine" --batch_size 4 --epochs $e  --joint_vocab --trm_layer_dim 2 --trm_layer_count 1 --trm_attn_heads 1 --trm_ff_nn_dim 2 --trm_drop 0.15 --trm_attn_drop 0.1 --trm_feed_drop 0.35 --trm_layer_drop 0.2 --expid $expid --datadir $DATADIR --jobid $JOBID --model $model --corpus $corpus

echo "Finished trial run."

################################################################################

# ACTUAL RUN (UNCOMMENT)

corpus="${all}"
e=960

# tr-mono (with ckpt selection based on token accuracy)
# $PYTHON inflector.py --eval_also_on_test  --checkpoint_selection --acc_for_ckpt_selection "w-accuracy" --weighted_accuracy_during_training --weight_decay 0.01 --clip_grad_norm 1.0 --lr 0.001 --lr_decay "cosine" --batch_size 512 --epochs $e  --joint_vocab --trm_layer_dim 256 --trm_layer_count 3 --trm_attn_heads 4 --trm_ff_nn_dim 64 --trm_drop 0.15 --trm_attn_drop 0.1 --trm_feed_drop 0.35 --trm_layer_drop 0.2 --expid $expid --datadir $DATADIR --jobid $JOBID --model $model --corpus $corpus

# tr-multi
# $PYTHON inflector.py --eval_also_on_test --multilingual_training --multilingual_corpus_down_up_sampling_temperature 0.5 --weighted_multilingual_accuracy_during_training --weight_decay 0.01 --clip_grad_norm 1.0 --lr 0.001 --lr_decay "cosine" --batch_size 512 --epochs $e  --joint_vocab --trm_layer_dim 256 --trm_layer_count 3 --trm_attn_heads 4 --trm_ff_nn_dim 64 --trm_drop 0.15 --trm_attn_drop 0.1 --trm_feed_drop 0.35 --trm_layer_drop 0.2 --expid $expid --datadir $DATADIR --jobid $JOBID --model $model --corpus $corpus


# Logs appear in logs/, results in results/, both marked by datetime and expid.
