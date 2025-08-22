#!/bin/bash
#PBS -N sigdata-run
#PBS -q gpu
#PBS -m ea
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:gpu_mem=20gb:scratch_local=10gb
#PBS -l walltime=8:00:00
# The 4 lines above are options for the scheduling system: the job will run 1 hour at maximum, 1 machine with 4 processors + 4gb RAM memory + 10gb scratch memory are requested

# can be run by `qsub run.sh`

# adapted from https://docs.metacentrum.cz/computing/run-basic-job/
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


mkdir -p $WORKDIR/results/sigmorphon
cp -r $DEVDIR/results/sigmorphon/* $WORKDIR/results/sigmorphon/ || { echo >&2 "Error while copying input file(s)!"; exit 2; }



#############################################################################


bash our-model-on-sigmorphon.sh $PYTHON $ROOTDIR

# move the output to user's DATADIR or exit in case of failure # clean the SCRATCH directory
#cp $PBS_JOBID.out $DEVDIR/job_outputs/
cp -r logs/* $DEVDIR/logs/
cp -r results/* $DEVDIR/results/
#|| ( export CLEAN_SCRATCH=false && { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; })

clean_scratch
