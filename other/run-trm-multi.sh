#!/bin/bash
#PBS -N multi-no-ckpt
#PBS -q gpu
#PBS -m ea
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:gpu_mem=32gb:scratch_local=4gb:cl_galdor=False
#PBS -l walltime=24:00:00
# The 4 lines above are options for the scheduling system: the job will run 1 hour at maximum, 1 machine with 4 processors + 4gb RAM memory + 10gb scratch memory are requested

# can be run by `qsub run-trm.sh`

# adapted from https://docs.metacentrum.cz/computing/run-basic-job/

### Checks if local directory `dp-local` exists, if so, use it. If not, backup to global directory in `brno2`.

export TMPDIR=$SCRATCHDIR

DEVDIR="/storage/brno2/home/LOGIN/inflector"

# define a DATADIR variable: directory where the input files are taken from and where the output will be copied to
LOCAL_ROOTDIR="${HOME}/dp-local"
BACKUP_ROOTDIR="/storage/brno2/home/LOGIN/dp-local"

# Check if LOCAL_ROOTDIR exists and contains .venv and data directories
if [[ -d "$LOCAL_ROOTDIR" && -d "$LOCAL_ROOTDIR/.venv" && -d "$LOCAL_ROOTDIR/data" ]]; then
    echo "LOCAL_ROOTDIR is valid."
    ROOTDIR="$LOCAL_ROOTDIR"
else
    echo "LOCAL_ROOTDIR is invalid. Setting to BACKUP_ROOTDIR."
    ROOTDIR="$BACKUP_ROOTDIR"
fi

PYTHON=$ROOTDIR"/.venv/bin/python"
DATADIR=$ROOTDIR"/data"


LOGDIR=$DEVDIR"/logs"
mkdir -p $LOGDIR
mkdir -p $DEVDIR/results/

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $LOGDIR/jobs_info.txt

# add python module
module add python/python-3.10.4-intel-19.0.4-sc7snnf

# test if the scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

WORKDIR=$SCRATCHDIR/inflector

mkdir -p $WORKDIR

# if the copy operation fails, issue an error message and exit
cp -r $DEVDIR/*.py $WORKDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cp -r $DEVDIR/*.sh $WORKDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cp -r $DEVDIR/configs $WORKDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

mkdir -p $WORKDIR/logs

# go into scratch directory
cd $WORKDIR

######## STD EXPERIMENTS #############################################################################
expid="exp31-base-multi-nockpt"

base="UD_Breton-KEB;UD_Spanish-AnCora;UD_Czech-PDT;UD_English-EWT;UD_Basque-BDT"
basesk="${base};UD_Slovak-SNK"
basepor="${base};UD_Portuguese-Bosque"
baseger="${base};UD_German-GSD"
all="UD_Breton-KEB;UD_Spanish-AnCora;UD_Czech-PDT;UD_English-EWT;UD_Basque-BDT;UD_Slovak-SNK;UD_Afrikaans-AfriBooms;UD_Ancient_Greek-PROIEL;UD_Ancient_Hebrew-PTNK;UD_Arabic-PADT;UD_Armenian-ArmTDP;UD_Belarusian-HSE;UD_Bulgarian-BTB;UD_Catalan-AnCora;UD_Chinese-GSDSimp;UD_Classical_Armenian-CAVaL;UD_Classical_Chinese-Kyoto;UD_Coptic-Scriptorium;UD_Croatian-SET;UD_Danish-DDT;UD_Dutch-Alpino;UD_Erzya-JR;UD_Estonian-EDT;UD_Finnish-TDT;UD_French-GSD;UD_Galician-TreeGal;UD_Georgian-GLC;UD_German-GSD;UD_Gothic-PROIEL;UD_Greek-GDT;UD_Hebrew-HTB;UD_Hindi-HDTB;UD_Hungarian-Szeged;UD_Icelandic-Modern;UD_Indonesian-GSD;UD_Irish-IDT;UD_Italian-ISDT;UD_Japanese-GSDLUW;UD_Korean-Kaist;UD_Kyrgyz-KTMU;UD_Latin-ITTB;UD_Latvian-LVTB;UD_Lithuanian-ALKSNIS;UD_Low_Saxon-LSDC;UD_Maghrebi_Arabic_French-Arabizi;UD_Manx-Cadhan;UD_Marathi-UFAL;UD_Naija-NSC;UD_North_Sami-Giella;UD_Norwegian-Bokmaal;UD_Old_Church_Slavonic-PROIEL;UD_Old_East_Slavic-TOROT;UD_Old_French-PROFITEROLE;UD_Ottoman_Turkish-BOUN;UD_Persian-PerDT;UD_Polish-PDB;UD_Pomak-Philotis;UD_Portuguese-Bosque;UD_Romanian-RRT;UD_Russian-SynTagRus;UD_Sanskrit-Vedic;UD_Scottish_Gaelic-ARCOSG;UD_Slovenian-SSJ;UD_Swedish-Talbanken;UD_Tamil-TTB;UD_Turkish-BOUN;UD_Ukrainian-IU;UD_Urdu-UDTB;UD_Uyghur-UDT;UD_Vietnamese-VTB;UD_Welsh-CCG;UD_Western_Armenian-ArmTDP;UD_Wolof-WTB"

#corpus="UD_Breton-KEB"
corpus="${base}"
#corpus="${basesk}"
#corpus="${basepor}"
#corpus="${baseger}"
#corpus="${all}"


model="transformer"

e=960

# TODO: may easily fail due to walltime

# with ckpt selection
#$PYTHON inflector.py --checkpoint_selection --acc_for_ckpt_selection "multiling-acc" --eval_also_on_test --multilingual_training --multilingual_corpus_down_up_sampling_temperature 0.5 --weighted_multilingual_accuracy_during_training --weight_decay 0.01 --clip_grad_norm 1.0 --lr 0.001 --lr_decay "cosine" --batch_size 512 --epochs $e  --joint_vocab --trm_layer_dim 256 --trm_layer_count 3 --trm_attn_heads 4 --trm_ff_nn_dim 64 --trm_drop 0.15 --trm_attn_drop 0.1 --trm_feed_drop 0.35 --trm_layer_drop 0.2 --expid $expid --datadir $DATADIR --jobid $PBS_JOBID --model $model --corpus $corpus || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }

# w/o ckpt selection
$PYTHON inflector.py --eval_also_on_test --multilingual_training --multilingual_corpus_down_up_sampling_temperature 0.5 --weighted_multilingual_accuracy_during_training --weight_decay 0.01 --clip_grad_norm 1.0 --lr 0.001 --lr_decay "cosine" --batch_size 512 --epochs $e  --joint_vocab --trm_layer_dim 256 --trm_layer_count 3 --trm_attn_heads 4 --trm_ff_nn_dim 64 --trm_drop 0.15 --trm_attn_drop 0.1 --trm_feed_drop 0.35 --trm_layer_drop 0.2 --expid $expid --datadir $DATADIR --jobid $PBS_JOBID --model $model --corpus $corpus || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }

######## END OF STD EXPERIMENTS #############################################################################

# copy back logs and integration test results and results
cp -r logs/* $DEVDIR/logs/ && cp -r results/* $DEVDIR/results/ || ( export CLEAN_SCRATCH=false && { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; })

clean_scratch
