"""Compute overlap of unique lemma-tag-form triples, in the train-dev and train-test pairs of CoNLL-U files from a UD corpus. For the tag, consider the tuple of POS and Features column."""

"""Example call:
> /storage/brno2/home/souradat/dp-sourada-dev/data/raw/ud-treebanks-v2.14$ for d in UD_Czech-PDT UD_Spanish-AnCora UD_English-EWT UD_Basque-BDT ; do echo $d && python3 compute-item-overlap.py $d/*train*.conllu $d/*dev*.conllu $d/*test*.conllu ; done
UD_Czech-PDT
Dev overlap: 30497/43674 (69.83%)
Test overlap: 31601/45641 (69.24%)
UD_Spanish-AnCora
Dev overlap: 8581/11625 (73.82%)
Test overlap: 8710/11836 (73.59%)
UD_English-EWT
Dev overlap: 4191/6305 (66.47%)
Test overlap: 4157/6400 (64.95%)
UD_Basque-BDT
Dev overlap: 5069/9133 (55.50%)
Test overlap: 4996/9243 (54.05%)

"""

import sys
import random

def read_conllu(file_path):
    """Read a CoNLL-U file and extract quadruples (columns 2, 3, 4, 6) (!! not 5, that is XPOS).
    1 ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes (decimal numbers can be lower than 1 but must be greater than 0).
    2 FORM: Word form or punctuation symbol.
    3 LEMMA: Lemma or stem of word form.
    4 UPOS: Universal part-of-speech tag.
    5 XPOS: Optional language-specific (or treebank-specific) part-of-speech / morphological tag; underscore if not available.
    6 FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    """
    print(f"Reading file {file_path}...")
    quadruples = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            columns = line.strip().split('\t')
            if len(columns) >= 6:
                quadruples.add((columns[1], columns[2], columns[3], columns[5]))
    for (i,q) in enumerate(quadruples):
        print(q)
        if i>2:
            break
    #first_three = [next(iter(quadruples)) for _ in range(3)]
    #for item in first_three:
    #    print(item)
    return quadruples

def compute_overlap(train_quads, target_quads):
    """Compute the percentage and absolute count of target_quads in train_quads."""
    common = target_quads & train_quads
    overlap_count = len(common)
    total_count = len(target_quads)
    percentage = (overlap_count / total_count * 100) if total_count > 0 else 0
    return overlap_count, total_count, percentage

def main(train_path, dev_path, test_path):
    train_quads = read_conllu(train_path)
    dev_quads = read_conllu(dev_path)
    test_quads = read_conllu(test_path)
    
    dev_overlap, dev_total, dev_percentage = compute_overlap(train_quads, dev_quads)
    test_overlap, test_total, test_percentage = compute_overlap(train_quads, test_quads)
    
    print(f"Dev overlap: {dev_overlap}/{dev_total} ({dev_percentage:.2f}%)")
    print(f"Test overlap: {test_overlap}/{test_total} ({test_percentage:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py train.conllu dev.conllu test.conllu")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
