import os
import sys
from typing import Any, BinaryIO, Callable, Iterable, Sequence, TextIO, TypedDict, Union
import urllib.request
import zipfile

import torch
from pathlib import Path

from process_ud_dir import UD


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
    SEP: int = 2

    def __init__(self, strings: Sequence[str]) -> None:
        self._strings = ["[PAD]", "[UNK]", "[SEP]"]
        self._strings.extend(strings)
        self._string_map = {string: index for index, string in enumerate(self._strings)}

    def __len__(self) -> int:
        return len(self._strings)

    def __iter__(self) -> Iterable[str]:
        return iter(self._strings)

    def string(self, index: int) -> str:
        return self._strings[index]

    def strings(self, indices: Sequence[int]) -> list[str]:
        # If there is an invalid index, do not raise IndexError, but rather just print "[INVALID]" instead of that token

        # remove this, as it is only a check whether it happens any time that an invalid index is generated
        # for index in indices:
        #     if index >= len(self._strings):
        #         raise RuntimeError(f"Rewriting in vocab: [INVALID={index}]" )
        return [self._strings[index] if index < len(self._strings) else f"[INVALID={index}]" for index in indices]

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


# Verze, kde je zmenene jen to ukladani instanci po vetach, a ktera funguje.
#
class MorphoDataset:
    PAD: int = 0
    UNK: int = 1
    SEP: int = 2
    BOW: int = 3
    EOW: int = 4
    DECODER_START_TOKEN_ID: int = 5
    # DECODER_START_TOKEN_ID : int = 4

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    DATA_DIRECTORY = UD.INFLECTION_DIR

    Element = TypedDict("Element", {"form": str, "lemma": str, "tag": str, "count": int,
                                    "tag_separated": tuple[str], "lang_id": Union[str | None],
                                    "corpus_len_coef": Union[int | None]})

    class Factor:
        word_vocab: Vocabulary
        char_vocab: Vocabulary
        strings: list[str]

        def __init__(self) -> None:
            self.strings = []

        def finalize(self, train=None) -> None:
            # Create vocabularies
            if train:
                self.word_vocab = train.word_vocab
                self.char_vocab = train.char_vocab
            else:
                # TODO: magic fix: remove sorted to allow self.strings containing both lemmata (true strings) and tag-sets (tuples of strings, tag-parts)
                # strings = sorted(set(self.strings))
                strings = list(set(self.strings))

                self.word_vocab = Vocabulary(strings)

                bow_eow = ["[BOW]", "[EOW]", "[DEC_START]"]
                self.char_vocab = Vocabulary(bow_eow + sorted(set(char for string in strings for char in string)))

    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self._factors = (
                MorphoDataset.Factor(), MorphoDataset.Factor(), MorphoDataset.Factor(), MorphoDataset.Factor(),
                MorphoDataset.Factor())
            self._factors_tensors = None
            self._counts = []

            # used only for encoding/decoding with its vocabulary. It's items are not used at all.
            self._joint_lemma_tag_factor = MorphoDataset.Factor()

            # used only for encoding/decoding with its vocabulary. It's items are not used at all.
            self._joint_lemma_tag_form_factor = MorphoDataset.Factor()

            # used only for multilingual datasets
            self._lang_ids_factor = MorphoDataset.Factor()

        @property
        def forms(self) -> "MorphoDataset.Factor":
            return self._factors[1]

        @property
        def lemmas(self) -> "MorphoDataset.Factor":
            return self._factors[0]

        @property
        def tags(self) -> "MorphoDataset.Factor":
            return self._factors[2]

        @property
        def counts(self) -> list[int]:
            return self._counts

        @property
        def tags_separated(self) -> "MorphoDataset.Factor":
            return self._factors[4]

        @property
        def lemma_tag_joint(self) -> "MorphoDataset.Factor":
            return self._joint_lemma_tag_factor

        @property
        def lemma_tag_form_joint(self) -> "MorphoDataset.Factor":
            return self._joint_lemma_tag_form_factor

        def __len__(self) -> int:
            return self._size

        def transform(self, transform: Callable[["MorphoDataset.Element"], Any]) -> "MorphoDataset.TransformedDataset":
            return MorphoDataset.TransformedDataset(self, transform)

    class MonolingualDataset(Dataset):
        def __init__(self, data_file: TextIO, train=None, max_sentences: int | None = None) -> None:
            super().__init__()

            # Load the data
            self._size = 0
            for line in data_file:
                line = line.strip()
                if line:
                    columns = line.split("\t")

                    # This is artificial adjustment which concats lemma and tag
                    # it should be removed soon
                    # lemma = lemma + tag
                    # columns[0] = columns[0] + "#" + columns[2]
                    # print(columns[0])
                    # input()

                    # make the tag-set sorted tuple to be hashable
                    tag_set = tuple(sorted(columns[2].split("|")))
                    columns.append(tag_set)
                    # Now, columns = [lemma, form, tag-atomic, count, tag-separated], where tag-separated=["tag-part-01", "tag-part-02", ...]
                    # TODO: zbytecne mame factor pro lemma, count, tag-atomic, tag-separated, ktere nikde nevyuzivame.

                    # TODO: need to add the assertion after fixing adding counts
                    # assert len(columns) == len(self._factors)

                    (lemma, form, tag_atomic, count, tag_separated) = tuple(columns)

                    # skip rows where lemma is longer than 500 characters
                    max_lemma_len = 500
                    if len(lemma) > max_lemma_len:
                        print(f"Skipping data entry with lemma longer than {max_lemma_len}: {lemma}", file=sys.stderr)
                        pass
                    else:
                        for column, factor in zip(columns, self._factors):
                            factor.strings.append(column)
                        self._joint_lemma_tag_factor.strings.extend([lemma, tag_separated])
                        self._joint_lemma_tag_form_factor.strings.extend([lemma, tag_separated, form])
                        self._counts.append(int(count))

                        self._size += 1

            # Finalize the mappings
            for i, factor in enumerate(self._factors):
                factor.finalize(train._factors[i] if train else None)

            self._joint_lemma_tag_factor.finalize(train._joint_lemma_tag_factor if train else None)
            self._joint_lemma_tag_form_factor.finalize(train._joint_lemma_tag_form_factor if train else None)

            # maximal length of a form
            self.max_len_of_form = max([len(form) for form in self.forms.strings])

        # def cle_batch(self, forms: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        #     unique_strings = list(set(forms))
        #     unique_string_map = {form: index + 1 for index, form in enumerate(unique_strings)}
        #     unique_forms = torch.nn.utils.rnn.pad_sequence(
        #         [torch.tensor([MorphoDataset.UNK])]
        #         + [torch.tensor(self.forms.char_vocab.indices(form)) for form in unique_strings], batch_first=True)
        #
        #     # TODO: tadyto moc nevim jak upravit...
        #     forms_indices = torch.nn.utils.rnn.pad_sequence(
        #         [torch.tensor([unique_string_map[form]])  for form in forms], batch_first=True)
        #     return unique_forms, forms_indices

        def __getitem__(self, index: int) -> "MorphoDataset.Element":
            return {"form": self.forms.strings[index],
                    "lemma": self.lemmas.strings[index],
                    "tag": self.tags.strings[index],
                    "count": self.counts[index],
                    "tag_separated": self.tags_separated.strings[index],
                    "lang_id": None,
                    "corpus_len_coef": None,
                    }

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def transform(self, transform: Callable[..., Any]) -> "MorphoDataset.TransformedDataset":
            return MorphoDataset.TransformedDataset(self, transform)

    class MultilingualDataset(Dataset):
        def __init__(self, filenames: list[Path], lang_ids: list[str], train=None,
                     max_sentences: int | None = None) -> None:
            """`lang_ids`: the identifiers of the languages of each corpus, to be used as identification of the language
            for model generation etc. - corpus_names can be passed if there are no 2 corpora with the same language
            or we do not want to use the knowledge of having the same language in two corpora
            `filenames`: one filename per language"""

            # Create factors
            super().__init__()

            assert len(filenames) == len(
                lang_ids), f"The filenames given and lang ids given to the MultilingualDataset must have the same count! {len(filenames)}!={len(lang_ids)}"

            # Load the data
            self._size = 0
            self._corpus_lens_unique = []
            self._corpus_lens_total = []

            for filename, lang_id in zip(filenames, lang_ids):

                # add lang_id as a tuple, such that it would be treated as an atomic token - hack: it has to be twice in it
                lang_id_token = tuple([f"LANG={lang_id}"])
                self._joint_lemma_tag_form_factor.strings.append(lang_id_token)
                self._joint_lemma_tag_factor.strings.append(lang_id_token)

                # the number of unique items in the corpus (number of lemma-form-tag-count quadruples)
                corpus_len_unique = count_lines(filename)

                print(f"Multilingual dataset: processing corpus {lang_id}")
                print(f"Corpus len: {corpus_len_unique}")
                #input()

                # the sum of occurrences of all items in the corpus (original word-count of the corpus, sum of corpus counts)
                corpus_len_total = sum_fourth_column(filename)

                with open(filename) as data_file:
                    for line in data_file:
                        line = line.strip()
                        if line:
                            columns = line.split("\t")

                            # make the tag-set sorted tuple to be hashable
                            tag_set = tuple(sorted(columns[2].split("|")))
                            columns.append(tag_set)

                            (lemma, form, tag_atomic, count, tag_separated) = tuple(columns)


                            # skip rows where lemma is longer than 500 characters
                            max_lemma_len = 500
                            if len(lemma) > max_lemma_len:
                                print(f"Skipping data entry with lemma longer than {max_lemma_len}: {lemma}",
                                      file=sys.stderr)
                                pass
                            else:
                                for column, factor in zip(columns, self._factors):
                                    factor.strings.append(column)

                                self._joint_lemma_tag_factor.strings.extend([lemma, tag_separated])
                                self._joint_lemma_tag_form_factor.strings.extend([lemma, tag_separated, form])
                                self._counts.append(int(count))
                                self._lang_ids_factor.strings.append(lang_id_token)

                                self._corpus_lens_unique.append(corpus_len_unique)
                                self._corpus_lens_total.append(corpus_len_total)

                                self._size += 1



            self._corpus_len_coefficients_for_weighted_avg_evaluation_during_training = transform_list(self._corpus_lens_unique)

            print("Corpus len coefficients for weighted avg evaluation during training:")
            print(get_shortened_list_string(self._corpus_len_coefficients_for_weighted_avg_evaluation_during_training))
            #print(self._corpus_len_coefficients_for_weighted_avg_evaluation_during_training == [11]*1204 + [1]*13026 + [2]*6584)
            #input()

            # Finalize the mappings
            for i, factor in enumerate(self._factors):
                factor.finalize(train._factors[i] if train else None)

            self._joint_lemma_tag_factor.finalize(train._joint_lemma_tag_factor if train else None)
            self._joint_lemma_tag_form_factor.finalize(train._joint_lemma_tag_form_factor if train else None)
            self._lang_ids_factor.finalize(train._lang_ids_factor if train else None)

            # maximal length of a form
            self.max_len_of_form = max([len(form) for form in self.forms.strings])

        @property
        def lang_ids(self) -> "MorphoDataset.Factor":
            return self._lang_ids_factor

        @property
        def corpus_lens_unique(self) -> list[int]:
            return self._corpus_lens_unique

        @property
        def corpus_lens_total(self) -> list[int]:
            return self._corpus_lens_total


        def __getitem__(self, index: int) -> "MorphoDataset.Element":
            return {"form": self.forms.strings[index],
                    "lemma": self.lemmas.strings[index],
                    "tag": self.tags.strings[index],
                    "count": self.counts[index],
                    "tag_separated": self.tags_separated.strings[index],
                    "lang_id": self.lang_ids.strings[index],
                    "corpus_len_coef": self._corpus_len_coefficients_for_weighted_avg_evaluation_during_training[index],
                    }

    def __init__(self, corpus_name: str = None, max_sentences=None, multilingual: bool = False,
                 corpus_names: list[str] = None):
        """If `multilingual=True`, `corpus_name` is not used, but `corpus_names` must be passed.
        A multilingual dataset is created, with a single source and single target Vocabulary, and also a single train set,
        a single (merged) dev set and a dict of individual dev-sets and dict of individual test-sets."""

        # TODO: add max_tokens parameter instead of max_sentences
        # path = "{}.zip".format(dataset)
        # if not os.path.exists(path):
        #     print("Downloading dataset {}...".format(dataset), file=sys.stderr)
        #     urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
        #     os.rename("{}.tmp".format(path), path)

        # TODO: call method for getting the train, dev, test paths (UD staticmethod), which inside ensures presence of the data:
        # do not go through the files using "train", 'dev" "test"

        # TODO: download & resplit if the train-dev-test split is not present
        # TODO: allow non-existence of train/dev data, or somewhere else, allow setup for running the true prediction
        if multilingual:
            corpora_filenames = [UD.get_corpus_files(corp_name) for corp_name in corpus_names]
            train_filenames = [corpus_filenames.train for corpus_filenames in corpora_filenames]
            dev_filenames = [corpus_filenames.dev for corpus_filenames in corpora_filenames]
            test_filenames = [corpus_filenames.test for corpus_filenames in corpora_filenames]

            for (dataset_name, filenames) in zip(["train", "dev", "test"],
                                                 [train_filenames, dev_filenames, test_filenames]):
                setattr(self, dataset_name,
                        self.MultilingualDataset(filenames=filenames, train=getattr(self, "train", None),
                                                 max_sentences=max_sentences, lang_ids=corpus_names))

            # create monolingual dev/test datasets for evaluation
            # these need to be MultilingualDatasets, since we need the lang_id flag there
            self.monolingual_dev_sets = dict()
            self.monolingual_test_sets = dict()
            for dev_test_dict, dev_test_filenames in zip([self.monolingual_dev_sets, self.monolingual_test_sets],
                                                         [dev_filenames, test_filenames]):
                for corpus_name, filename in zip(corpus_names, dev_test_filenames):
                    dev_test_dict[corpus_name] = MorphoDataset.MultilingualDataset([filename],
                                                                                   train=getattr(self,
                                                                                                 "train",
                                                                                                 None),
                                                                                   max_sentences=max_sentences,
                                                                                   lang_ids=[corpus_name])

        else:
            corpus_filenames = UD.get_corpus_files(corpus_name)
            for (dataset_name, filename) in zip(["train", "dev", "test"],
                                                [corpus_filenames.train, corpus_filenames.dev, corpus_filenames.test]):
                with open(filename, "r") as dataset_file:
                    setattr(self, dataset_name,
                            self.MonolingualDataset(dataset_file, train=getattr(self, "train", None),
                                                    max_sentences=max_sentences))

        # corpus_dir = MorphoDataset.DATA_DIRECTORY / corpus_name
        #
        # for dataset_name in ["train", "dev", "test"]:
        #     filename = corpus_dir / f"{dataset_name}.tsv"
        #     with open(filename, "r") as dataset_file:
        #         setattr(self, dataset_name, self.Dataset(dataset_file, train=getattr(self, "train", None), max_sentences=max_sentences))

    train: Dataset
    dev: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: "MorphoDataset.Factor", predictions: Sequence[str]) -> float:
        gold_sentences = gold_dataset.strings

        predicted_sentences, in_sentence = [], False
        for line in predictions:
            line = line.rstrip("\n")
            if not line:
                in_sentence = False
            else:
                if not in_sentence:
                    predicted_sentences.append([])
                    in_sentence = True
                predicted_sentences[-1].append(line)

        if len(predicted_sentences) != len(gold_sentences):
            raise RuntimeError("The predictions contain different number of sentences than gold data: {} vs {}".format(
                len(predicted_sentences), len(gold_sentences)))

        correct, total = 0, 0
        for i, (predicted_sentence, gold_sentence) in enumerate(zip(predicted_sentences, gold_sentences)):
            if len(predicted_sentence) != len(gold_sentence):
                raise RuntimeError("Predicted sentence {} has different number of words than gold: {} vs {}".format(
                    i + 1, len(predicted_sentence), len(gold_sentence)))
            correct += sum(predicted == gold for predicted, gold in zip(predicted_sentence, gold_sentence))
            total += len(predicted_sentence)

        return 100 * correct / total

    @staticmethod
    def evaluate_file(gold_dataset: "MorphoDataset.Factor", predictions_file: TextIO) -> float:
        predictions = predictions_file.readlines()
        return MorphoDataset.evaluate(gold_dataset, predictions)

def sum_fourth_column(filename):
    """Returns the sum of the values in the 4th column of a TSV file."""
    total = 0
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            columns = line.strip().split("\t")  # Split by tab
            if len(columns) >= 4:  # Ensure there are at least 4 columns
                try:
                    total += int(columns[3])  # Convert to integer and add
                except ValueError:
                    pass  # Ignore lines where the 4th column is not a number
    return total

def count_lines(filename):
    """Returns the number of lines in the given file."""
    with open(filename, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def transform_list(lst):
    """
    Transforms a list of integers by computing their reciprocals, normalizing them so that the smallest value is 1,
    and rounding all results to the nearest integer.

    Steps:
    1. Compute the reciprocal (1/x) for each element.
    2. Normalize by dividing all values by the minimum reciprocal.
    3. Round the results (multiplied by 100 for higher precision) to the nearest integer.

    Parameters:
        lst (list of int): A list of integers (nonzero values).

    Returns:
        list of int: A list of transformed integers.

    Raises:
        ZeroDivisionError: If the input list contains zero.
    """
    if not lst:
        return []  # Handle empty list case

    reciprocal_list = [1 / x for x in lst]  # Step 1: Compute reciprocals
    min_value = min(reciprocal_list)  # Find the minimum
    normalized_list = [x / min_value for x in reciprocal_list]  # Step 2: Normalize
    rounded_list = [round(100 * x) for x in normalized_list]  # Step 3: Round to integers

    return rounded_list


def get_shortened_list_string(lst):
    def shortstring(item):
        if isinstance(item, float) or item != int(item):
            if abs(item) < 1e-4:
                    return f"{item:.2e}"
            else:
                return f"{item:.4f}"
        else:
            return f"{item}"

    if not lst:
        return ""

    result = []  # List to hold the shortened version of the list as strings
    count = 1  # Initialize the count for the first item
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            count += 1  # Increment count if the current item is the same as the previous one
        else:
            if count > 1:
                result.append(f"{shortstring(lst[i - 1])} ({count}x)")  # Append the count of the previous number to result
            else:
                result.append(f"{shortstring(lst[i - 1])}")
            count = 1  # Reset the count for the new number

    # Append the last sequence after the loop finishes
    result.append(f"{shortstring(lst[-1])} ({count}x)")

    # Join the list of results into a single string separated by newlines
    return "[" + ", ".join(result) + "]"

#
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
#     parser.add_argument("--corpus", default="czech_pdt", type=str, help="The corpus to evaluate")
#     parser.add_argument("--dataset", default="dev", type=str, help="The dataset to evaluate (dev/test)")
#     parser.add_argument("--task", default="tagger", type=str, help="Task to evaluate (tagger/lemmatizer)")
#     args = parser.parse_args()
#
#     if args.evaluate:
#         gold = getattr(MorphoDataset(args.corpus), args.dataset)
#         if args.task == "tagger":
#             gold = gold.tags
#         elif args.task == "lemmatizer":
#             gold = gold.lemmas
#         else:
#             raise ValueError("Unknown task '{}', valid values are only 'tagger' or 'lemmatizer'".format(args.task))
#
#         with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
#             accuracy = MorphoDataset.evaluate_file(gold, predictions_file)
#         print("{} accuracy: {:.2f}%".format(args.task.title(), accuracy))
