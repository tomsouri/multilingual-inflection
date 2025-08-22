#!/bin/bash

# if cmdline argument(s) is/are given, they are each represented as a corpus name to process

# Should the split be weighted by the frequencies or rather uniform? (if weighted, first sample to train randomly weighted by occurrence counts, then uniformly randomly to dev/test, otherwise uniformly randomly to train/dev/test)
SPLIT="lemma-disjoint-weighted" # "lemma-disjoint-uniform"

# How should by the train:dev:test 8:1:1 ratio ensured? By-counts means that the 8:1:1 is the ratio of total occurrence counts, while ignoring-counts means that it is the ratio of number of lemmas.
RATIO="lemmata-by-counts" # "lemmata-ignoring-counts"

# All 73 corpora used in the thesis.
DEFAULT_CORPORA=(
# development langs (5)
	"UD_Breton-KEB"
	"UD_Spanish-AnCora"
	"UD_Czech-PDT"
	"UD_English-EWT"
	"UD_Basque-BDT"
# surprise langs: Indo-European langs written in Latin script (28)
	"UD_Afrikaans-AfriBooms"
	"UD_Catalan-AnCora"
	"UD_Croatian-SET"
	"UD_Danish-DDT"
	"UD_Dutch-Alpino"
	"UD_French-GSD"
	"UD_Galician-TreeGal"
	"UD_German-GSD"
	"UD_Gothic-PROIEL"
	"UD_Icelandic-Modern"
	"UD_Irish-IDT"
	"UD_Italian-ISDT"
	"UD_Latin-ITTB"
	"UD_Latvian-LVTB"
	"UD_Lithuanian-ALKSNIS"
	"UD_Low_Saxon-LSDC"
	"UD_Manx-Cadhan"
	"UD_Old_French-PROFITEROLE"
	"UD_Polish-PDB"
	"UD_Pomak-Philotis"
	"UD_Portuguese-Bosque"
	"UD_Romanian-RRT"
	"UD_Sanskrit-Vedic"
	"UD_Scottish_Gaelic-ARCOSG"
	"UD_Slovak-SNK"
	"UD_Slovenian-SSJ"
	"UD_Swedish-Talbanken"
	"UD_Welsh-CCG"
# surprise langs: the rest of langs (40)
	"UD_Ancient_Greek-PROIEL"
	"UD_Ancient_Hebrew-PTNK"
	"UD_Arabic-PADT"
	"UD_Armenian-ArmTDP"
	"UD_Belarusian-HSE"
	"UD_Bulgarian-BTB"
	"UD_Chinese-GSDSimp"
	"UD_Classical_Armenian-CAVaL"
	"UD_Classical_Chinese-Kyoto"
	"UD_Coptic-Scriptorium"
	"UD_Erzya-JR"
	"UD_Estonian-EDT"
	"UD_Finnish-TDT"
	"UD_Georgian-GLC"
	"UD_Greek-GDT"
	"UD_Hebrew-HTB"
	"UD_Hindi-HDTB"
	"UD_Hungarian-Szeged"
	"UD_Indonesian-GSD"
	"UD_Japanese-GSDLUW"
	"UD_Korean-Kaist"
	"UD_Kyrgyz-KTMU"
	"UD_Maghrebi_Arabic_French-Arabizi"
	"UD_Marathi-UFAL"
	"UD_Naija-NSC"
	"UD_North_Sami-Giella"
	"UD_Norwegian-Bokmaal"
	"UD_Old_Church_Slavonic-PROIEL"
	"UD_Old_East_Slavic-TOROT"
	"UD_Ottoman_Turkish-BOUN"
	"UD_Persian-PerDT"
	"UD_Russian-SynTagRus"
	"UD_Tamil-TTB"
	"UD_Turkish-BOUN"
	"UD_Ukrainian-IU"
	"UD_Urdu-UDTB"
	"UD_Uyghur-UDT"
	"UD_Vietnamese-VTB"
	"UD_Western_Armenian-ArmTDP"
	"UD_Wolof-WTB"
)

# if no cmdline args are given, process default corpora
if [ $# -eq 0 ]; then
  corpora=("${DEFAULT_CORPORA[@]}")
else
  corpora=("$@")
fi



# Generate a date-time string in the format "YYYY-MM-DD_HH-MM-SS"
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")

# Set TMPDIR to include the current date-time
TMPDIR="tmp_$current_datetime"

mkdir -p $TMPDIR

mkdir -p "data/raw/"
TGZ_PATH="data/raw/ud-treebanks-v2.14.tgz"
DATADIR="data/processed/ud-inflection/ud-treebanks-v2.14"
UD_URL="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502/ud-treebanks-v2.14.tgz"
PYTHON=".venv/bin/python3"

for CORPUS in "${corpora[@]}" ; do
  $PYTHON preprocess.py --ud_tgz_path $TGZ_PATH --ud_url $UD_URL --datadir $DATADIR --seed 42 --corpus $CORPUS --split_type $SPLIT --split_ratio_type $RATIO ;
done


$PYTHON compute_statistics.py > "data-stats.${current_datetime}.txt"

rm -r $TMPDIR
