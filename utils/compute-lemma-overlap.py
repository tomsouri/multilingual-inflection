"""Computes overlap of unique lemmas in train-dev and train-test pairs in given conllu files from UD corpus."""

"""Example call:
> /storage/brno2/home/souradat/dp-sourada-dev/data/raw/ud-treebanks-v2.14$ for d in UD_Czech-PDT UD_Spanish-AnCora UD_English-EWT UD_Basque-BDT ; do echo $d && python3 compute-lemma-overlap.py $d/*train*.conllu $d/*dev*.conllu $d/*test*.conllu ; done
UD_Czech-PDT
Reading file UD_Czech-PDT/cs_pdt-ud-train.conllu...
rozklížený
vědomí
singlový
Rusňák
Reading file UD_Czech-PDT/cs_pdt-ud-dev.conllu...
248
halový
Tonak
Hopmanův
Reading file UD_Czech-PDT/cs_pdt-ud-test.conllu...
248
halový
vědomí
bonbónek
Dev overlap: 14701/18131 (81.08%)
Test overlap: 15215/18787 (80.99%)
UD_Spanish-AnCora
Reading file UD_Spanish-AnCora/es_ancora-ud-train.conllu...
0,35
Unidos
Ojos
Jhon
Reading file UD_Spanish-AnCora/es_ancora-ud-dev.conllu...
cuarentena
Pippin
pleno
oportuno
Reading file UD_Spanish-AnCora/es_ancora-ud-test.conllu...
pleno
Unidos
reina
Chamil
Dev overlap: 5901/7346 (80.33%)
Test overlap: 5930/7464 (79.45%)
UD_English-EWT
Reading file UD_English-EWT/en_ewt-ud-train.conllu...
Stanford
healthy
Spain
omelet
Reading file UD_English-EWT/en_ewt-ud-dev.conllu...
speed
pedi
responsibility
:(
Reading file UD_English-EWT/en_ewt-ud-test.conllu...
pedi
speed
responsibility
Stanford
Dev overlap: 2994/4247 (70.50%)
Test overlap: 2999/4414 (67.94%)
UD_Basque-BDT
Reading file UD_Basque-BDT/eu_bdt-ud-train.conllu...
mezulari
oholtza
morroi
erretaula
Reading file UD_Basque-BDT/eu_bdt-ud-dev.conllu...
aurren
hots
Coast
blokeo
Reading file UD_Basque-BDT/eu_bdt-ud-test.conllu...
aurren
hots
ingurune-baldintza
blokeo
Dev overlap: 3380/4704 (71.85%)
Test overlap: 3359/4693 (71.57%)

"""

import sys

def read_conllu(file_path):
    """Read a CoNLL-U file and extract 1-tuples (column 3)."""
    print(f"Reading file {file_path}...")
    values = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            columns = line.strip().split('\t')
            if len(columns) >= 3:
                values.add(columns[2])
    for i, lemma in enumerate(values):
        print(lemma)
        if i>2:
            break
    return values

def compute_overlap(train_values, target_values):
    """Compute the percentage and absolute count of target_values in train_values."""
    common = target_values & train_values
    overlap_count = len(common)
    total_count = len(target_values)
    percentage = (overlap_count / total_count * 100) if total_count > 0 else 0
    return overlap_count, total_count, percentage

def main(train_path, dev_path, test_path):
    train_values = read_conllu(train_path)
    dev_values = read_conllu(dev_path)
    test_values = read_conllu(test_path)
    
    dev_overlap, dev_total, dev_percentage = compute_overlap(train_values, dev_values)
    test_overlap, test_total, test_percentage = compute_overlap(train_values, test_values)
    
    print(f"Dev overlap: {dev_overlap}/{dev_total} ({dev_percentage:.2f}%)")
    print(f"Test overlap: {test_overlap}/{test_total} ({test_percentage:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py train.conllu dev.conllu test.conllu")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
