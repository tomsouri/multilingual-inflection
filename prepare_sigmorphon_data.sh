#!/bin/bash

# Downloads SIGMORPHON 2022 and 2023 raw data. Converts them to our format and puts them to our directories.
# Renames files, changes directory structure, swaps columns (if needed), replaces tag separator.

PYTHON="python3"

# download if not present
RAW_PREFIX="data/raw"
PROC_PREFIX="data/processed"
dirname_22="2022InflectionST"
dirname_23="2023InflectionST"

mkdir -p $RAW_PREFIX
mkdir -p $PROC_PREFIX

cd $RAW_PREFIX

if [ ! -d "$dirname_22" ]; then
    echo "Directory $dirname_22 does not exist. Downloading..."

    git clone https://github.com/sigmorphon/2022InflectionST.git

    echo "Download complete."
else
    echo "Directory $dirname_22 already exists."
fi

if [ ! -d "$dirname_23" ]; then
    echo "Directory $dirname_23 does not exist. Downloading..."

    git clone https://github.com/sigmorphon/2023InflectionST.git

    echo "Download complete."
else
    echo "Directory $dirname_23 already exists."
fi

cd ../../

echo "Processing SIGMORPHON 2022 shared task data..."

# process them to our format - using inflector_dataset - sigmorphon compatibility
# source path to original data files
path_22=$RAW_PREFIX"/2022InflectionST/part1/development_languages"
train_suf_22=".train"
dev_suf_22=".dev"
test_suf_22=".gold"
tag_separator_22=";"
path_22_tmp=$PROC_PREFIX/tmp/22

# target path: where to put the processed data
tgt_dir_22=$PROC_PREFIX/2022_SIGMORPHON_Inflection_ST
mkdir -p $path_22_tmp
mkdir -p $tgt_dir_22

cp $path_22/* $path_22_tmp/

# we do not work with small training data, remove it
rm $path_22_tmp/*small*
# and rename the large training data, such that the files of the same corpus would have the same name (except for suffix)
for f in $(ls $path_22_tmp/*_large*); do mv "$f" "${f//_large/}"; done


# Process languages one by one, convert the data
languages=$(ls $path_22_tmp | cut -f1 -d. | uniq)

for language in $languages; do
    echo "Processing language: $language"
    train_file=$path_22_tmp/$language$train_suf_22
    dev_file=$path_22_tmp/$language$dev_suf_22
    test_file=$path_22_tmp/$language$test_suf_22
    tgt_dir=$tgt_dir_22/$language
    mkdir -p $tgt_dir

    $PYTHON prepare_sigmorphon_data.py --train_file=$train_file --dev_file=$dev_file --test_file=$test_file --tgt_dir=$tgt_dir
done

# remove the tmp directory
rm -r $path_22_tmp


# The same for 2023 data, with the difference that it is needed to swap the columns, and the files have different names


echo "Processing SIGMORPHON 2023 shared task data"

path_23=$RAW_PREFIX"/2023InflectionST/part1/data"
train_suf_23=".trn"
dev_suf_23=".dev"
test_suf_23=".tst"
tag_separator_23=";"
path_23_tmp=$PROC_PREFIX/tmp/23
tgt_dir_23=$PROC_PREFIX/2023_SIGMORPHON_Inflection_ST
mkdir -p $path_23_tmp
mkdir -p $tgt_dir_23


cp $path_23/* $path_23_tmp

# remove files with missing `form` column (those with .covered. in filename)
rm $path_23_tmp/*.covered.*

# switch columns, as in 2023 they are in different order than in 2022
for f in $(ls $path_23_tmp); do cat $path_23_tmp/$f | awk -F'\t' '{print $1 "\t" $3 "\t" $2}' > $path_23_tmp/tmp && cat $path_23_tmp/tmp > $path_23_tmp/$f ; done


# Process languages one by one, convert the data
languages=$(ls $path_23_tmp | cut -f1 -d. | uniq)

for language in $languages; do
    echo "Processing language: $language"
    train_file=$path_23_tmp/$language$train_suf_23
    dev_file=$path_23_tmp/$language$dev_suf_23
    test_file=$path_23_tmp/$language$test_suf_23
    tgt_dir=$tgt_dir_23/$language
    mkdir -p $tgt_dir

    $PYTHON prepare_sigmorphon_data.py --train_file=$train_file --dev_file=$dev_file --test_file=$test_file --tgt_dir=$tgt_dir
done
