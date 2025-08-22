#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the
# Attribution-NonCommercial-ShareAlike 4.0 International license.
# If a copy of the CC license was not distributed with this
# file, You can obtain one at  https://creativecommons.org/licenses/by-nc-sa/4.0/ .


import os
import sys
from typing import Any, BinaryIO, Callable, Iterable, Sequence, TextIO, TypedDict, Optional
from morpho_dataset import MorphoDataset
from process_ud_dir import UD
import urllib.request
import zipfile

from pathlib import Path


#
# import torch


# A class for managing mapping between strings and indices.
# It provides:
# - `__len__`: number of strings in the vocabulary
# - `__iter__`: iterator over strings in the vocabulary
# - `string(index: int) -> str`: string for a given index to the vocabulary
# - `strings(indices: Sequence[int]) -> list[str]`: list of strings for given indices
# - `index(string: str) -> int`: index of a given string in the vocabulary
# - `indices(strings: Sequence[str]) -> list[int]`: list of indices for given strings
class Vocabulary:
    PAD: int = 0
    UNK: int = 1

    def __init__(self, strings: Sequence[str]) -> None:
        self._strings = ["[PAD]", "[UNK]"]
        self._strings.extend(strings)
        self._string_map = {string: index for index, string in enumerate(self._strings)}

    def __len__(self) -> int:
        return len(self._strings)

    def __iter__(self) -> Iterable[str]:
        return iter(self._strings)

    def string(self, index: int) -> str:
        return self._strings[index]

    def strings(self, indices: Sequence[int]) -> list[str]:
        return [self._strings[index] for index in indices]

    def index(self, string: str) -> int:
        return self._string_map.get(string, Vocabulary.UNK)

    def indices(self, strings: Sequence[str]) -> list[int]:
        return [self._string_map.get(string, Vocabulary.UNK) for string in strings]


# Loads a morphological dataset in a vertical format.
# - The data consists of three datasets
#   - `train`
#   - `dev`
#   - `test`
# - Each dataset is a `torch.utils.data.Dataset` providing
#   - `__len__`: number of sentences in the dataset
#   - `__getitem__`: return the requested sentence as an `Element`
#     instance, which is a dictionary with keys "forms"/"lemmas"/"tags",
#     each being a list of strings
#   - `forms`, `lemmas`, `tags`: instances of type `Factor` containing
#     the following fields:
#     - `strings`: a Python list containing input sentences, each being
#       a list of strings (forms/lemmas/tags)
#     - `word_vocab`: a `Vocabulary` object capable of mapping words to
#       indices. It is constructed on the train set and shared by the dev
#       and test sets
#     - `char_vocab`: a `Vocabulary` object capable of mapping characters
#       to  indices. It is constructed on the train set and shared by the dev
#       and test sets
#   - `cle_batch`: a method for creating inputs for character-level embeddings.
#     It takes a list of sentences, each being a list of string forms, and produces
#     a tuple of two tensors:
#     - `unique_forms` with shape `[num_unique_forms, max_form_length]` containing
#       each unique form as a sequence of character ids
#     - `forms_indices` with shape `[num_sentences, max_sentence_length]`
#       containing for every form its index in `unique_forms`
class InflectionDataset:
    # PAD: int = 0
    # UNK: int = 1
    # BOW: int = 2
    # EOW: int = 3
    #
    # _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"
    #
    # Element = TypedDict("Element", {"forms": list[str], "lemmas": list[str], "tags": list[str]})
    #
    # class Factor:
    #     word_vocab: Vocabulary
    #     char_vocab: Vocabulary
    #     strings: list[list[str]]
    #
    #     def __init__(self) -> None:
    #         self.strings = []
    #
    #     def finalize(self, train = None) -> None:
    #         # Create vocabularies
    #         if train:
    #             self.word_vocab = train.word_vocab
    #             self.char_vocab = train.char_vocab
    #         else:
    #             strings = sorted(set(string for sentence in self.strings for string in sentence))
    #             self.word_vocab = Vocabulary(strings)
    #
    #             bow_eow = ["[BOW]", "[EOW]"]
    #             self.char_vocab = Vocabulary(bow_eow + sorted(set(char for string in strings for char in string)))
    #
    # class Dataset(torch.utils.data.Dataset):
    #     def __init__(self, data_file: BinaryIO, train = None, max_sentences: int | None = None) -> None:
    #         # Create factors
    #         self._factors = (MorphoDataset.Factor(), MorphoDataset.Factor(), MorphoDataset.Factor())
    #         self._factors_tensors = None
    #
    #         # Load the data
    #         self._size = 0
    #         in_sentence = False
    #         for line in data_file:
    #             line = line.decode("utf-8").rstrip("\r\n")
    #             if line:
    #                 if not in_sentence:
    #                     for factor in self._factors:
    #                         factor.strings.append([])
    #                     self._size += 1
    #
    #                 columns = line.split("\t")
    #                 assert len(columns) == len(self._factors)
    #                 for column, factor in zip(columns, self._factors):
    #                     factor.strings[-1].append(column)
    #
    #                 in_sentence = True
    #             else:
    #                 in_sentence = False
    #                 if max_sentences is not None and self._size >= max_sentences:
    #                     break
    #
    #         # Finalize the mappings
    #         for i, factor in enumerate(self._factors):
    #             factor.finalize(train._factors[i] if train else None)
    #
    #     @property
    #     def forms(self) -> "MorphoDataset.Factor":
    #         return self._factors[0]
    #
    #     @property
    #     def lemmas(self) -> "MorphoDataset.Factor":
    #         return self._factors[1]
    #
    #     @property
    #     def tags(self) -> "MorphoDataset.Factor":
    #         return self._factors[2]
    #
    #     def __len__(self) -> int:
    #         return self._size
    #
    #     def __getitem__(self, index: int) -> "MorphoDataset.Element":
    #         return {"forms": self.forms.strings[index],
    #                 "lemmas": self.lemmas.strings[index],
    #                 "tags": self.tags.strings[index]}
    #
    #     def transform(self, transform: Callable[["MorphoDataset.Element"], Any]) -> "MorphoDataset.TransformedDataset":
    #         return MorphoDataset.TransformedDataset(self, transform)
    #
    #     def cle_batch(self, forms: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
    #         unique_strings = list(set(form for sentence in forms for form in sentence))
    #         unique_string_map = {form: index + 1 for index, form in enumerate(unique_strings)}
    #         unique_forms = torch.nn.utils.rnn.pad_sequence(
    #             [torch.tensor([MorphoDataset.UNK])]
    #             + [torch.tensor(self.forms.char_vocab.indices(form)) for form in unique_strings], batch_first=True)
    #         forms_indices = torch.nn.utils.rnn.pad_sequence(
    #             [torch.tensor([unique_string_map[form] for form in sentence]) for sentence in forms], batch_first=True)
    #         return unique_forms, forms_indices
    #
    # class TransformedDataset(torch.utils.data.Dataset):
    #     def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
    #         self._dataset = dataset
    #         self._transform = transform
    #
    #     def __len__(self) -> int:
    #         return len(self._dataset)
    #
    #     def __getitem__(self, index: int) -> Any:
    #         item = self._dataset[index]
    #         return self._transform(*item) if isinstance(item, tuple) else self._transform(item)
    #
    #     def transform(self, transform: Callable[..., Any]) -> "MorphoDataset.TransformedDataset":
    #         return MorphoDataset.TransformedDataset(self, transform)
    #
    # def __init__(self, dataset, max_sentences=None):
    #     path = "{}.zip".format(dataset)
    #     if not os.path.exists(path):
    #         print("Downloading dataset {}...".format(dataset), file=sys.stderr)
    #         urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
    #         os.rename("{}.tmp".format(path), path)
    #
    #     with zipfile.ZipFile(path, "r") as zip_file:
    #         for dataset in ["train", "dev", "test"]:
    #             with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
    #                 setattr(self, dataset, self.Dataset(
    #                     dataset_file, train=getattr(self, "train", None), max_sentences=max_sentences))
    #
    # train: Dataset
    # dev: Dataset
    # test: Dataset
    #
    # # Evaluation infrastructure.
    # @staticmethod
    # def evaluate(gold_dataset: "MorphoDataset.Factor", predictions: Sequence[str]) -> float:
    #     gold_sentences = gold_dataset.strings
    #
    #     predicted_sentences, in_sentence = [], False
    #     for line in predictions:
    #         line = line.rstrip("\n")
    #         if not line:
    #             in_sentence = False
    #         else:
    #             if not in_sentence:
    #                 predicted_sentences.append([])
    #                 in_sentence = True
    #             predicted_sentences[-1].append(line)
    #
    #     if len(predicted_sentences) != len(gold_sentences):
    #         raise RuntimeError("The predictions contain different number of sentences than gold data: {} vs {}".format(
    #             len(predicted_sentences), len(gold_sentences)))
    #
    #     correct, total = 0, 0
    #     for i, (predicted_sentence, gold_sentence) in enumerate(zip(predicted_sentences, gold_sentences)):
    #         if len(predicted_sentence) != len(gold_sentence):
    #             raise RuntimeError("Predicted sentence {} has different number of words than gold: {} vs {}".format(
    #                 i + 1, len(predicted_sentence), len(gold_sentence)))
    #         correct += sum(predicted == gold for predicted, gold in zip(predicted_sentence, gold_sentence))
    #         total += len(predicted_sentence)
    #
    #     return 100 * correct / total
    #
    # @staticmethod
    # def evaluate_file(gold_dataset: "MorphoDataset.Factor", predictions_file: TextIO) -> float:
    #     predictions = predictions_file.readlines()
    #     return MorphoDataset.evaluate(gold_dataset, predictions)

    @staticmethod
    def evaluate_predictions(gold_forms: list[str], pred_forms: list[str],
                             weights: Optional[list[float]] = None) -> float:
        """Evaluate predictions using weights. If weights are not given, perform macro evaluation (each instance with equal weight)"""

        assert len(gold_forms) == len(pred_forms), f"Gold and predicted forms must have the same length: gold={len(gold_forms)}!=pred={len(pred_forms)}"

        # If no weights are provided, set equal weight for each instance
        if weights is None:
            weights = [1] * len(gold_forms)
        else:
            # Ensure that the provided weights list matches the length of the input lists
            assert len(weights) == len(
                gold_forms), f"Weights list must have the same length as the gold and predicted forms.gold={len(gold_forms)}!=weights={len(weights)}"

        # Initialize metrics for evaluation
        total_weight = sum(weights)
        weighted_correct = 0

        # Loop over each prediction and its corresponding gold form, using weights
        for gold, pred, weight in zip(gold_forms, pred_forms, weights):
            if gold == pred:
                weighted_correct += weight

        # Return the weighted accuracy (or another evaluation metric)
        weighted_accuracy = weighted_correct / total_weight
        return weighted_accuracy

    @staticmethod
    def evaluate_files(gold_file: TextIO, predictions_file: TextIO, macro_evaluation: bool) -> float:
        """Evaluate a file of predicted forms w.r.t. gold file containing the gold forms.
        The gold file is expected in format lemma\tform\ttag\tcount per line.
        The predictions file is expected to contain one form per line.
        The count of lines in each of the files must be the same.
        Accuracy (between 0 and 1) is returned."""

        predicted_forms = [line.strip() for line in predictions_file.readlines()]

        gold_forms, counts = zip(*[(line.split('\t')[1], int(line.split('\t')[3])) for line in gold_file])

        if macro_evaluation:
            counts = None

        return InflectionDataset.evaluate_predictions(gold_forms, predicted_forms, counts)

    @staticmethod
    def evaluate_on_specific_indices(eval_predictions, weights, gold_predictions, indices) -> dict:
        """
        Evaluate predicted forms against gold forms, counting only specified indices. Return both weighted (micro-avg) and
        uniform (macro-avg) accuracy, and also the count (and percentage) of the indices included to evaluation.
        `indices`: [0,1]-list, with 0 meaning not to evaluate and 1 evaluate
        """

        def product(xs, ys):
            """Scalar product of two float lists."""
            return [x * y for x, y in zip(xs, ys)]

        sum_indices = sum(indices)



        return {
            "count": sum_indices,

            # the percentage of items included in this evaluation
            "%": sum_indices / len(indices),

            # the percentage of overall corpus-frequency-counts of the items included in this evaluation
            "%-weights": sum(product(weights,indices)) / sum(weights),

            "weighted": 0 if sum_indices == 0 else InflectionDataset.evaluate_predictions(gold_forms=gold_predictions,
                                                                                          pred_forms=eval_predictions,
                                                                                          weights=product(weights,
                                                                                                          indices)),
            "uniform": 0 if sum_indices == 0 else InflectionDataset.evaluate_predictions(gold_forms=gold_predictions,
                                                                                         pred_forms=eval_predictions,
                                                                                         weights=indices)
        }

    @staticmethod
    def full_evaluation_on_dataset(eval_predictions, gold_dataset, train_dataset):
        """Perform full evaluation of given predictions, on all relevant subsets of the evaluation dataset."""
        all_train_tagsets = set(train_dataset.tags.strings)
        all_train_lemmas = set(train_dataset.lemmas.strings)
        gold_forms = gold_dataset.forms.strings
        weights = gold_dataset.counts

        eval_types_and_indices = [
            # include all data to evaluation
            ("all", [1 for _ in range(len(gold_forms))]),

            # include only trivial items (gold_lemma == gold_form)
            ("trivial", [1 if gold_forms[i] == gold_dataset.lemmas.strings[i] else 0 for i in range(len(gold_forms))]),

            # include only non-trivial items (gold_lemma != gold_form)
            ("non-trivial",
             [1 if gold_forms[i] != gold_dataset.lemmas.strings[i] else 0 for i in range(len(gold_forms))]),

            # include only items with feature overlap (specific tag combination seen in training data)
            ("feat-overlap",
             [1 if gold_dataset.tags.strings[i] in all_train_tagsets else 0 for i in range(len(gold_forms))]),

            # include only items without feature overlap (specific tag combination not seen in training data)
            ("feat-disjoint",
             [1 if gold_dataset.tags.strings[i] not in all_train_tagsets else 0 for i in range(len(gold_forms))]),


            # newly added criteria which are necessary for multilingual setting where we cannot guarantee lemma-disjointness
            ("lemma-overlap",
             [1 if gold_dataset.lemmas.strings[i] in all_train_lemmas else 0 for i in range(len(gold_forms))]),

            ("lemma-disjoint",
             [1 if gold_dataset.lemmas.strings[i] not in all_train_lemmas else 0 for i in range(len(gold_forms))]),

            ("lemma-only-overlap",
             [1 if (gold_dataset.lemmas.strings[i] in all_train_lemmas and gold_dataset.tags.strings[
                 i] not in all_train_tagsets) else 0 for i in range(len(gold_forms))]),

            ("feats-only-overlap",
             [1 if (gold_dataset.lemmas.strings[i] not in all_train_lemmas and gold_dataset.tags.strings[
                 i] in all_train_tagsets) else 0 for i in range(len(gold_forms))]),

            ("both-overlap",
             [1 if (gold_dataset.lemmas.strings[i] in all_train_lemmas and gold_dataset.tags.strings[
                 i] in all_train_tagsets) else 0 for i in range(len(gold_forms))]),

            ("no-overlap",
             [1 if (gold_dataset.lemmas.strings[i] not in all_train_lemmas and gold_dataset.tags.strings[
                 i] not in all_train_tagsets) else 0 for i in range(len(gold_forms))]),
        ]
        return {
            eval_type: InflectionDataset.evaluate_on_specific_indices(
                eval_predictions=eval_predictions,
                weights=weights,
                gold_predictions=gold_dataset.forms.strings,
                indices=indices
            ) for eval_type, indices in eval_types_and_indices
        }

    @staticmethod
    def full_evaluation_by_corpus(eval_predictions, corpus_name: str, test: bool = False) -> dict:
        """Perform full evaluation of predictions given the corpus_name. Evaluate either against the test set or the dev set."""
        morpho = MorphoDataset(corpus_name, max_sentences=None)
        if test:
            eval_dataset = morpho.test
        else:
            eval_dataset = morpho.dev

        train_dataset = morpho.train

        return InflectionDataset.full_evaluation_on_dataset(eval_predictions, eval_dataset, train_dataset)

    @staticmethod
    def full_evaluation_on_corpora(corpora_list: list[str], prediction_dev_files: list[str],
                                   prediction_test_files: list[str]) -> dict:
        """
        Evaluation of dev and test predictions for a given list of corpora.
        Suitable for evaluating predictions of other than our systems, e.g. SIGMORPHON baselines, which first
        produce all predictions (both for dev and test) and we then need to evaluate them separately.
        The order of the given corpora and prediction files (both dev and test) must match.
        """
        result = dict()
        for corpus_name, pred_dev_file, pred_test_file in zip(corpora_list, prediction_dev_files,
                                                              prediction_test_files):
            dev_predictions = extract_predictions_from_file(pred_dev_file)
            test_predictions = extract_predictions_from_file(pred_test_file)

            result[corpus_name] = {
                "dev": InflectionDataset.full_evaluation_by_corpus(dev_predictions, corpus_name, test=False),
                "test": InflectionDataset.full_evaluation_by_corpus(test_predictions, corpus_name, test=True),
            }

        def average_dict(dict_list):
            """Compute an average of a dictionary list"""
            # If the list is empty, return an empty dictionary
            if not dict_list:
                return {}

            # Initialize a result dictionary with the same structure as the first dictionary
            def recursive_avg(dictionaries):
                avg_dict = {}
                for key in dictionaries[0].keys():
                    # Collect values for the current key across all dictionaries
                    values = [d[key] for d in dictionaries if key in d]

                    # If the value is a dictionary, recurse
                    if isinstance(values[0], dict):
                        avg_dict[key] = recursive_avg(values)
                    else:
                        # If the value is a number (float), compute the average
                        avg_dict[key] = sum(values) / len(values)
                return avg_dict

            return recursive_avg(dict_list)

        result["macro-avg"] = average_dict(list(result.values()))

        return result

    @staticmethod
    def pretty_print_of_full_results(full_accuracies: dict) -> str:
        """Convert full results to a string in tsv format in a nice human-readable way."""
        str_result = ""
        header = ["lang", "dataset", "eval_type", "count", "%", "%-weighted", "weighted", "uniform"]
        str_result += "\t".join(header) + "\n"
        for lang, datasets in full_accuracies.items():
            for dataset, eval_types in datasets.items():
                for eval_type, metrics in eval_types.items():
                    # Extract the values and format them
                    row = [
                        lang,
                        dataset,
                        eval_type,
                        f"{metrics['count']}",
                        f"{metrics['%']:.4f}",
                        f"{metrics['%-weights']:.4f}",
                        f"{metrics['weighted']:.4f}",
                        f"{metrics['uniform']:.4f}"
                    ]
                    str_result += "\t".join(row) + "\n"
        return str_result


def extract_predictions_from_file(filepath: str) -> list[str]:
    """
    Read a file containing predictions and load them.
    The file should contain one prediction per a line.
    """
    with open(filepath, "r") as f:
        preds = f.read().split("\n")

        # remove last empty line
        if preds[-1].strip() == "":
            preds = preds[:-1]

    return preds

# class SigmorphonCompatibility:
#     shared_task_data_directory = Path("data/raw/2023InflectionST/part1/data")
#     shared_task_converted_data_directory = Path("data/processed/2023_SIGMPORHON_Inflection_ST")
#
#     # TODO dwn original sig data, if not present
#
#     @staticmethod
#     def convert_sigmorphon_file_to_inflector_format(sigmorphon_file: TextIO, inflector_file: TextIO) -> None:
#         """Converts file in SIGMORPHON 2023 Shared task format to OUR format.
#         It changes the order of the items in a row (lemma, tag, form -> lemma, form, tag) and adds count column
#         (always 1, the SIGMORPHON data are missing the information about counts), and changes the format of tag
#         (replaces ';' with '|', such that the morphological features would always be split by '|' character).
#
#         Example of SIGMORPHON 2023 format:
#         Arabize V;PRS;NOM(3,SG) Arabizes
#         Arabize V;V.PTCP;PRS    Arabizing
#
#         Example of our format (the same data after conversion):
#         Arabize	V|PRS|NOM(3,SG)	Arabizes	1
#         Arabize	V|V.PTCP|PRS	Arabizing	1
#         """
#         print(f"Processing {sigmorphon_file.name} file...", file=sys.stderr)
#         for line in sigmorphon_file:
#             lemma, tag, form = line.strip().split("\t")
#             tags = tag.split(";")
#             inflector_file.write(f"{lemma}\t{form}\t{'|'.join(tags)}\t1\n")
#
#     @staticmethod
#     def convert_whole_sigmorphon_directory(sigmorphon_directory: Path, output_directory: Path) -> None:
#         """Call `convert_sigmorphon_file_to_inflector_format` method on all files in the given directory, and place
#         the converted files into the output_directory."""
#
#         print(f"Converting all files from directory {sigmorphon_directory} into our format...", file=sys.stderr)
#
#         # Ensure the output directory exists
#         output_directory.mkdir(parents=True, exist_ok=True)
#
#         # Iterate over all files in the sigmorphon directory
#         for sigmorphon_file in sigmorphon_directory.glob('*'):
#             # Check if the item is a file
#             if sigmorphon_file.is_file() and "covered" not in sigmorphon_file.name:
#                 # omit files missing the `form` column (those with "covered" in filename, such as `eng.covered.tst`)
#
#                 # Define the output file path
#                 output_file = output_directory / sigmorphon_file.name
#
#                 # Open the input and output files
#                 with sigmorphon_file.open('r', encoding='utf-8') as input_file, \
#                         output_file.open('w', encoding='utf-8') as output_file:
#                     # Call the conversion function
#                     SigmorphonCompatibility.convert_sigmorphon_file_to_inflector_format(input_file, output_file)
#
#         print(f"Converted files in our format were saved to {output_directory}", file=sys.stderr)

# @staticmethod
# def convert_all_sigmorphon_files() -> None:
#     SigmorphonCompatibility.convert_whole_sigmorphon_directory(
#         SigmorphonCompatibility.shared_task_data_directory,
#         SigmorphonCompatibility.shared_task_converted_data_directory
#     )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", default=None, type=str, help="Gold file to evaluate")
    parser.add_argument("--pred_file", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--macro_eval", default=False, action="store_true",
                        help="Perform macro-average evaluation (ignore frequencies of test entries).")

    parser.add_argument("--corpora", default=None, type=str, help=";-separated list of corpora for full evaluation")
    parser.add_argument("--dev_pred_files", default=None, type=str, help=";-separated list of dev-prediction files for full evaluation")
    parser.add_argument("--test_pred_files", default=None, type=str,
                        help=";-separated list of test-prediction files for full evaluation")
    parser.add_argument("--results_file", default=None, type=str, help="File to print the basic results")
    parser.add_argument("--full_results_file", default=None, type=str, help="File to print the full results")
    parser.add_argument("--model_name", default=None, type=str, help="Description of the model")
    parser.add_argument("--expid", default="none", type=str, help="If not set to 'none', results on all corpora will be printed.")
    parser.add_argument("--datadir", default="data/processed/ud-inflection/ud-treebanks-v2.14", type=str,
                        help="Directory of the processed UD data directory. If not specified, filled in based on the `data` argument.")


    # parser.add_argument("--convert_sigmorphon_data", default=False, action="store_true",
    #                     help="Convert data from sigmorphon format to our format.")
    args = parser.parse_args()

    # if args.convert_sigmorphon_data:
    #     # convert data from sigmorphon format to our format
    #     SigmorphonCompatibility.convert_whole_sigmorphon_directory(
    #         SigmorphonCompatibility.shared_task_data_directory,
    #         SigmorphonCompatibility.shared_task_converted_data_directory
    #     )
    if args.corpora is not None and args.dev_pred_files is not None and args.test_pred_files is not None:
        # Perform full evaluation of predictions from other system, e.g. from SIGMORPHON baseline.

        UD.INFLECTION_DIR = Path(args.datadir)

        corpora = args.corpora.split(";")
        dev_pred_files = args.dev_pred_files.split(";")
        test_pred_files = args.test_pred_files.split(";")

        print(f"Performing evaluation on the following corpora: {corpora}...")

        # perform the evaluation
        results = InflectionDataset.full_evaluation_on_corpora(
            corpora_list = corpora,
            prediction_dev_files = dev_pred_files,
            prediction_test_files=test_pred_files
        )

        # write the full results to a tsv file
        with open(args.full_results_file, "w") as f:
            f.write(InflectionDataset.pretty_print_of_full_results(results))

        # write basic results to a res file
        with open(args.results_file, "w") as f:
            simple_accuracies = {
                corpus: (
                results[corpus]['dev']['all']['weighted'], results[corpus]['dev']['all']['uniform'])
                for corpus
                in results
            }
            for corpus, accs in simple_accuracies.items():
                print(f"{args.model_name}\t{corpus}\t{accs[0]:.4f}\t{accs[1]:.4f}")
                f.write(f"{args.model_name}\t{corpus}\t{accs[0]:.4f}\t{accs[1]:.4f}\n")

            if args.expid != "none":
                # Code for experiments, where we want to print out also the averages, and the corpora results in specific order
                relevant_langs = ["UD_Spanish-AnCora", "UD_Czech-PDT", "UD_English-EWT"]
                relevant_accs = [simple_accuracies[lang] for lang in relevant_langs]
                macro_avg_weighted_acc = sum(relevant_accs[i][0] for i in range(len(relevant_accs))) / len(
                    relevant_accs)
                macro_avg_uniform_acc = sum(relevant_accs[i][1] for i in range(len(relevant_accs))) / len(relevant_accs)

                simple_accuracies["MACRO-AVG(CZ,ESP,EN)"] = (macro_avg_weighted_acc, macro_avg_uniform_acc)
                for corpus in ["MACRO-AVG(CZ,ESP,EN)"]:
                    accs = simple_accuracies[corpus]
                    print(f"{args.model_name}\t{corpus}\t{accs[0]:.4f}\t{accs[1]:.4f}")
                    f.write(f"{args.model_name}\t{corpus}\t{accs[0]:.4f}\t{accs[1]:.4f}\n")

                f.write(f"{args.model_name}\tALL\t")
                for corpus in ["MACRO-AVG(CZ,ESP,EN)", "UD_Czech-PDT", "UD_English-EWT", "UD_Spanish-AnCora",
                               "UD_Basque-BDT", "UD_Breton-KEB"]:
                    accs = simple_accuracies[corpus]
                    f.write(f"{accs[0]:.4f}\t{accs[1]:.4f}\t")
                f.write("\n")




    if args.gold_file is not None and args.pred_file is not None:
        with open(args.gold_file, "r") as gold, open(args.pred_file, "r") as pred:
            print(
                f"The accuracy on the provided eval set is: {InflectionDataset.evaluate_files(gold, pred, args.macro_eval):.15f}")

    # parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    # parser.add_argument("--corpus", default="czech_pdt", type=str, help="The corpus to evaluate")
    # parser.add_argument("--dataset", default="dev", type=str, help="The dataset to evaluate (dev/test)")
    # parser.add_argument("--task", default="tagger", type=str, help="Task to evaluate (tagger/lemmatizer)")
    # args = parser.parse_args()
    #
    # if args.evaluate:
    #     gold = getattr(MorphoDataset(args.corpus), args.dataset)
    #     if args.task == "tagger":
    #         gold = gold.tags
    #     elif args.task == "lemmatizer":
    #         gold = gold.lemmas
    #     else:
    #         raise ValueError("Unknown task '{}', valid values are only 'tagger' or 'lemmatizer'".format(args.task))
    #
    #     with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
    #         accuracy = MorphoDataset.evaluate_file(gold, predictions_file)
    #     print("{} accuracy: {:.2f}%".format(args.task.title(), accuracy))
