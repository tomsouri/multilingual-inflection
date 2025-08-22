import argparse
from pathlib import Path
import random
import numpy as np

from process_ud_dir import UD

parser = argparse.ArgumentParser()
parser.add_argument("--ud_url",
                    default="https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502/ud-treebanks-v2.14.tgz",
                    type=str, help="URL for downloading the raw UD data.")
parser.add_argument("--ud_tgz_path",
                    default="data/raw/ud-treebanks-v2.14.tgz",
                    type=str, help="Path to the tgz file where to save the downloaded UD data.")
parser.add_argument("--datadir", default="data/processed/ud-inflection/ud-treebanks-v2.14", type=str,
                    help="Directory of the processed UD data directory.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--corpus", default="UD_Spanish-AnCora", type=str,
                    help="Corpus to run the preprocessing on.")
parser.add_argument("--split_type", default="lemma-disjoint-weighted",
                    choices=["lemma-disjoint-uniform", "lemma-disjoint-weighted"],
                    help="Type of split to perform. Options: `lemma-disjoint-uniform` - split lemmata uniformly randomly between train-dev-test, `lemma-disjoint-weighted` - first, sample train lemmata randomly with weights (given by counts), then split the rest randomly between dev and test.")
parser.add_argument("--split_ratio", default="8:1:1", type=str, help="Split ratio - train:dev:test.")
parser.add_argument("--split_ratio_type", default="lemmata-by-counts",
                    choices=["lemmata-by-counts", "lemmata-ignoring-counts"],
                    help="Split ratio type. Should the given ratio (e.g., 8:1:1) be the ratio of the number of unique lemmata in train/dev/test (`lemmata-ignoring-counts`), or the ratio between occurrence counts of lemmata in train/dev/test (`lemmata-by-counts`)?")




def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    UD.TGZ_PATH = Path(args.ud_tgz_path)
    UD.URL = args.ud_url
    UD.INFLECTION_DIR = Path(args.datadir)

    if args.split_type == "lemma-disjoint-uniform":
        weighting = "uniform"
    elif args.split_type == "lemma-disjoint-weighted":
        weighting = "weighted"
    else:
        raise RuntimeError(f"Invalid option for split type: {args.split_type}")

    if args.split_ratio_type == "lemmata-ignoring-counts":
        ensure_ratio = "lemmata"
    elif args.split_ratio_type == "lemmata-by-counts":
        ensure_ratio = "occurrences"
    else:
        raise RuntimeError(f"Invalid option for split ratio type: {args.split_ratio_type}")

    if ensure_ratio == "occurrences" and weighting == "uniform":
        raise NotImplementedError("This split type is implemented with a bug, because the final ratio of counts is not 8:1:1 at all. This is probably because it allows sampling a really frequent lemma as the last sampled one from the train set, which leads to massive exceeding the desired number number of counts in the train set.")

    UD.resplit_corpus(args.corpus, args.split_ratio, ensure_ratio, weighting)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
