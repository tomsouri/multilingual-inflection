import os
import sys
from collections import Counter
from typing import Iterator, Iterable, TextIO
from pathlib import Path
import argparse
import random
import numpy as np
from collections import defaultdict
from sortedcontainers import SortedSet
from dataclasses import dataclass

import tarfile
import urllib.request

import ufal.udpipe

# TODO: make arguments
FILE_SUFFIX = ".tsv"
# ALL_TRIPLETS_NAME = "all" + FILE_SUFFIX
TRAIN_NAME = "train" + FILE_SUFFIX
DEV_NAME = "dev" + FILE_SUFFIX
TEST_NAME = "test" + FILE_SUFFIX


@dataclass(frozen=True, order=True)
class MorphologicalTriplet:
    """Class representing a unique lemma-form-tag triplet from a corpus."""
    lemma: str
    form: str
    tag: str

    def __str__(self) -> str:
        return "\t".join([self.lemma, self.form, self.tag])


@dataclass(frozen=True, order=True)
class MorphologicalTripletCounted:
    """Morphological triplet, containing also absolute count of occurrences of the triplet from the corpus."""
    triplet: MorphologicalTriplet
    count: int

    @property
    def lemma(self) -> str:
        return self.triplet.lemma

    @property
    def form(self) -> str:
        return self.triplet.form

    @property
    def tag(self) -> str:
        return self.triplet.tag

    def __str__(self) -> str:
        return "\t".join([str(self.triplet), str(self.count)])


@dataclass(frozen=True)
class SplitProbabilities:
    train: float
    dev: float
    test: float

    def __post_init__(self):
        if not 0 <= self.train <= 1 or not 0 <= self.dev <= 1 or not 0 <= self.test <= 1:
            raise ValueError("Probabilities must be between 0 and 1.")
        if not abs(self.train + self.dev + self.test - 1) < 1e-6:
            raise ValueError("Probabilities must sum to 1.")


@dataclass(frozen=True)
class CorpusFilePaths:
    train: Path
    dev: Path
    test: Path
    corpus_dir: Path

    def __iter__(self):
        # Iterate over all split files
        return iter((self.train, self.dev, self.test))

    def __str__(self) -> str:
        return f"train: {self.train}\ndev: {self.dev}\ntest: {self.test}\n"


# to arguments
# TRAIN_PROB = 0.8
# DEV_PROB = 0.1
# TEST_PROB = 0.1
# SPLIT_PROBABILITIES = SplitProbabilities(TRAIN_PROB, DEV_PROB, TEST_PROB)


class UD:
    """Interface:
    - method `get_corpus_files -> returns paths to train-dev-test files of given corpus
    - method `resplit_corpus -> ensures presence of raw data and performs resplit of given corpus
    """

    # These are substituted by the main script
    URL: str = ""
    TGZ_PATH: Path = Path("")
    INFLECTION_DIR: Path = Path("")

    ALL_UD_CORPORA = ["UD_Abaza-ATB", "UD_Abkhaz-AbNC", "UD_Afrikaans-AfriBooms", "UD_Akkadian-PISANDUB",
                      "UD_Akkadian-RIAO", "UD_Akuntsu-TuDeT", "UD_Albanian-TSA", "UD_Amharic-ATT",
                      "UD_Ancient_Greek-Perseus", "UD_Ancient_Greek-PROIEL", "UD_Ancient_Greek-PTNK",
                      "UD_Ancient_Hebrew-PTNK", "UD_Apurina-UFPA", "UD_Arabic-NYUAD", "UD_Arabic-PADT", "UD_Arabic-PUD",
                      "UD_Armenian-ArmTDP", "UD_Armenian-BSUT", "UD_Assyrian-AS", "UD_Azerbaijani-TueCL",
                      "UD_Bambara-CRB", "UD_Basque-BDT", "UD_Bavarian-MaiBaam", "UD_Beja-NSC", "UD_Belarusian-HSE",
                      "UD_Bengali-BRU", "UD_Bhojpuri-BHTB", "UD_Bororo-BDT", "UD_Breton-KEB", "UD_Bulgarian-BTB",
                      "UD_Buryat-BDT", "UD_Cantonese-HK", "UD_Cappadocian-TueCL", "UD_Catalan-AnCora", "UD_Cebuano-GJA",
                      "UD_Chinese-Beginner", "UD_Chinese-CFL", "UD_Chinese-GSD", "UD_Chinese-GSDSimp", "UD_Chinese-HK",
                      "UD_Chinese-PatentChar", "UD_Chinese-PUD", "UD_Chukchi-HSE", "UD_Classical_Armenian-CAVaL",
                      "UD_Classical_Chinese-Kyoto", "UD_Classical_Chinese-TueCL", "UD_Coptic-Scriptorium",
                      "UD_Croatian-SET", "UD_Czech-CAC", "UD_Czech-CLTT", "UD_Czech-FicTree", "UD_Czech-PDT",
                      "UD_Czech-Poetry", "UD_Czech-PUD", "UD_Danish-DDT", "UD_Dutch-Alpino", "UD_Dutch-LassySmall",
                      "UD_Egyptian-UJaen", "UD_English-Atis", "UD_English-CTeTex", "UD_English-ESLSpok",
                      "UD_English-EWT", "UD_English-GENTLE", "UD_English-GUM", "UD_English-GUMReddit",
                      "UD_English-LinES", "UD_English-ParTUT", "UD_English-Pronouns", "UD_English-PUD", "UD_Erzya-JR",
                      "UD_Estonian-EDT", "UD_Estonian-EWT", "UD_Faroese-FarPaHC", "UD_Faroese-OFT", "UD_Finnish-FTB",
                      "UD_Finnish-OOD", "UD_Finnish-PUD", "UD_Finnish-TDT", "UD_French-FQB", "UD_French-GSD",
                      "UD_French-ParisStories", "UD_French-ParTUT", "UD_French-PUD", "UD_French-Rhapsodie",
                      "UD_French-Sequoia", "UD_Frisian_Dutch-Fame", "UD_Galician-CTG", "UD_Galician-PUD",
                      "UD_Galician-TreeGal", "UD_Georgian-GLC", "UD_German-GSD", "UD_German-HDT", "UD_German-LIT",
                      "UD_German-PUD", "UD_Gheg-GPS", "UD_Gothic-PROIEL", "UD_Greek-GDT", "UD_Greek-GUD",
                      "UD_Guajajara-TuDeT", "UD_Guarani-OldTuDeT", "UD_Gujarati-GujTB", "UD_Haitian_Creole-Autogramm",
                      "UD_Hausa-NorthernAutogramm", "UD_Hausa-SouthernAutogramm", "UD_Hebrew-HTB",
                      "UD_Hebrew-IAHLTwiki", "UD_Highland_Puebla_Nahuatl-ITML", "UD_Hindi-HDTB", "UD_Hindi-PUD",
                      "UD_Hittite-HitTB", "UD_Hungarian-Szeged", "UD_Icelandic-GC", "UD_Icelandic-IcePaHC",
                      "UD_Icelandic-Modern", "UD_Icelandic-PUD", "UD_Indonesian-CSUI", "UD_Indonesian-GSD",
                      "UD_Indonesian-PUD", "UD_Irish-Cadhan", "UD_Irish-IDT", "UD_Irish-TwittIrish", "UD_Italian-ISDT",
                      "UD_Italian-MarkIT", "UD_Italian-Old", "UD_Italian-ParlaMint", "UD_Italian-ParTUT",
                      "UD_Italian-PoSTWITA", "UD_Italian-PUD", "UD_Italian-TWITTIRO", "UD_Italian-Valico",
                      "UD_Italian-VIT", "UD_Japanese-BCCWJ", "UD_Japanese-BCCWJLUW", "UD_Japanese-GSD",
                      "UD_Japanese-GSDLUW", "UD_Japanese-PUD", "UD_Japanese-PUDLUW", "UD_Javanese-CSUI",
                      "UD_Kaapor-TuDeT", "UD_Kangri-KDTB", "UD_Karelian-KKPP", "UD_Karo-TuDeT", "UD_Kazakh-KTB",
                      "UD_Khunsari-AHA", "UD_Kiche-IU", "UD_Komi_Permyak-UH", "UD_Komi_Zyrian-IKDP",
                      "UD_Komi_Zyrian-Lattice", "UD_Korean-GSD", "UD_Korean-Kaist", "UD_Korean-PUD", "UD_Kurmanji-MG",
                      "UD_Kyrgyz-KTMU", "UD_Kyrgyz-TueCL", "UD_Latgalian-Cairo", "UD_Latin-CIRCSE", "UD_Latin-ITTB",
                      "UD_Latin-LLCT", "UD_Latin-Perseus", "UD_Latin-PROIEL", "UD_Latin-UDante", "UD_Latvian-Cairo",
                      "UD_Latvian-LVTB", "UD_Ligurian-GLT", "UD_Lithuanian-ALKSNIS", "UD_Lithuanian-HSE",
                      "UD_Livvi-KKPP", "UD_Low_Saxon-LSDC", "UD_Luxembourgish-LuxBank", "UD_Macedonian-MTB",
                      "UD_Madi-Jarawara", "UD_Maghrebi_Arabic_French-Arabizi", "UD_Makurap-TuDeT", "UD_Malayalam-UFAL",
                      "UD_Maltese-MUDT", "UD_Manx-Cadhan", "UD_Marathi-UFAL", "UD_Mbya_Guarani-Dooley",
                      "UD_Mbya_Guarani-Thomas", "UD_Middle_French-PROFITEROLE", "UD_Moksha-JR", "UD_Munduruku-TuDeT",
                      "UD_Naija-NSC", "UD_Nayini-AHA", "UD_Neapolitan-RB", "UD_Nheengatu-CompLin",
                      "UD_North_Sami-Giella", "UD_Norwegian-Bokmaal", "UD_Norwegian-Nynorsk",
                      "UD_Old_Church_Slavonic-PROIEL", "UD_Old_East_Slavic-Birchbark", "UD_Old_East_Slavic-RNC",
                      "UD_Old_East_Slavic-Ruthenian", "UD_Old_East_Slavic-TOROT", "UD_Old_French-PROFITEROLE",
                      "UD_Old_Irish-DipSGG", "UD_Old_Irish-DipWBG", "UD_Old_Turkish-Clausal", "UD_Ottoman_Turkish-BOUN",
                      "UD_Ottoman_Turkish-DUDU", "UD_Paumari-TueCL", "UD_Persian-PerDT", "UD_Persian-Seraji",
                      "UD_Polish-LFG", "UD_Polish-PDB", "UD_Polish-PUD", "UD_Pomak-Philotis", "UD_Portuguese-Bosque",
                      "UD_Portuguese-CINTIL", "UD_Portuguese-GSD", "UD_Portuguese-PetroGold",
                      "UD_Portuguese-Porttinari", "UD_Portuguese-PUD", "UD_Romanian-ArT", "UD_Romanian-Nonstandard",
                      "UD_Romanian-RRT", "UD_Romanian-SiMoNERo", "UD_Romanian-TueCL", "UD_Russian-GSD",
                      "UD_Russian-Poetry", "UD_Russian-PUD", "UD_Russian-SynTagRus", "UD_Russian-Taiga",
                      "UD_Sanskrit-UFAL", "UD_Sanskrit-Vedic", "UD_Scottish_Gaelic-ARCOSG", "UD_Serbian-SET",
                      "UD_Sinhala-STB", "UD_Skolt_Sami-Giellagas", "UD_Slovak-SNK", "UD_Slovenian-SSJ",
                      "UD_Slovenian-SST", "UD_Soi-AHA", "UD_South_Levantine_Arabic-MADAR", "UD_Spanish-AnCora",
                      "UD_Spanish-COSER", "UD_Spanish-GSD", "UD_Spanish-PUD", "UD_Swedish-LinES", "UD_Swedish-PUD",
                      "UD_Swedish_Sign_Language-SSLC", "UD_Swedish-Talbanken", "UD_Swiss_German-UZH", "UD_Tagalog-TRG",
                      "UD_Tagalog-Ugnayan", "UD_Tamil-MWTT", "UD_Tamil-TTB", "UD_Tatar-NMCTT", "UD_Teko-TuDeT",
                      "UD_Telugu_English-TECT", "UD_Telugu-MTG", "UD_Thai-PUD", "UD_Tswana-Popapolelo",
                      "UD_Tupinamba-TuDeT", "UD_Turkish-Atis", "UD_Turkish-BOUN", "UD_Turkish-FrameNet",
                      "UD_Turkish-GB", "UD_Turkish_German-SAGT", "UD_Turkish-IMST", "UD_Turkish-Kenet",
                      "UD_Turkish-Penn", "UD_Turkish-PUD", "UD_Turkish-Tourism", "UD_Ukrainian-IU",
                      "UD_Umbrian-IKUVINA", "UD_Upper_Sorbian-UFAL", "UD_Urdu-UDTB", "UD_Uyghur-UDT", "UD_Veps-VWT",
                      "UD_Vietnamese-TueCL", "UD_Vietnamese-VTB", "UD_Warlpiri-UFAL", "UD_Welsh-CCG",
                      "UD_Western_Armenian-ArmTDP", "UD_Western_Sierra_Puebla_Nahuatl-ITML", "UD_Wolof-WTB",
                      "UD_Xavante-XDT", "UD_Xibe-XDT", "UD_Yakut-YKTDT", "UD_Yoruba-YTB", "UD_Yupik-SLI",
                      "UD_Zaar-Autogramm"]

    # CANONIC_CORPORA = {
    #     "cz": "UD_Czech-PDT",
    #     "es": "UD_Spanish-AnCora",
    #     "en": "UD_English-EWT",
    #     "eus": "UD_Basque-BDT",
    #     "bre": "UD_Breton-KEB"
    #     # lang_code : # corpus_name
    # }

    @staticmethod
    def _get_corpus_filepaths(corpus_name: str) -> CorpusFilePaths:
        """get paths to corpus files (only paths, existence of files nor directories is not ensured)"""
        corpus_dir = UD.INFLECTION_DIR / corpus_name
        return CorpusFilePaths(
            train=corpus_dir / TRAIN_NAME,
            dev=corpus_dir / DEV_NAME,
            test=corpus_dir / TEST_NAME,
            corpus_dir=corpus_dir
        )

    @staticmethod
    def get_corpus_files(corpus_name: str) -> CorpusFilePaths:
        """Get paths to the corpus files"""
        # TODO: check existence of files, if non-existent, raise error

        return UD._get_corpus_filepaths(corpus_name)

    @staticmethod
    def resplit_corpus(corpus_name: str, split_ratio: str, ensure_ratio: str, weighting: str) -> None:
        """Joins the original conllu files from train-dev-test split of the corpus, performs lemma-disjoint resplit, and
        prints the new train-dev-test files to the output directory. If the raw data is not present, downloads it."""

        def write_list_to_file(filename: Path, items: Iterable) -> None:
            with open(filename, "w") as f:
                for item in items:
                    f.write(str(item) + "\n")

        def ensure_raw_data_presence() -> None:
            """Check if the tgz file with raw UD data is present, and if not, download it."""
            if not os.path.exists(UD.TGZ_PATH):
                # zajisti ze bude existovat aspon directory
                UD.TGZ_PATH.parent.mkdir(parents=True, exist_ok=True)
                print("Downloading UD data from {}...".format(UD.URL), file=sys.stderr)
                urllib.request.urlretrieve(UD.URL, filename="{}.tmp".format(UD.TGZ_PATH))
                os.rename("{}.tmp".format(UD.TGZ_PATH), UD.TGZ_PATH)
                print("Successfully downloaded UD data and saved to {}...".format(UD.TGZ_PATH),
                      file=sys.stderr)

        def extract_all_conllu_content_from_tgz_for_corpus(corpus_name: str, tgz_path: Path) -> str:
            """Read the tgz file, and from all files containing `corpus_name` in the filename with extension ".conllu"
            extract the content, merge it and return."""
            text = ""
            # Open the tgz file
            with tarfile.open(tgz_path, "r:gz") as tar:
                # Iterate over each file in the tgz archive
                for member in sorted(tar.getmembers(), key=lambda member: member.name):
                    # Check if the filename contains the corpus name and has a .conllu extension
                    if corpus_name in member.name and member.name.endswith('.conllu'):
                        # Extract the file content
                        f = tar.extractfile(member)
                        if f:
                            text += f.read().decode('utf-8') + "\n\n"
            return text

        def conllu_text_to_counted_triplets(text: str) -> Iterable[MorphologicalTripletCounted]:
            """Reads conllu text and returns Iterator over morphological triplets, each triplet once, with the absolute count of occurrences."""

            def conllu_text_to_udpipe_words(text: str) -> Iterator[ufal.udpipe.Word]:
                """Converts text in conllu to Iterator over UDPipe words."""
                tokenizer = ufal.udpipe.InputFormat.newConlluInputFormat()
                sentence = ufal.udpipe.Sentence()
                processing_error = ufal.udpipe.ProcessingError()
                tokenizer.setText(text)

                while tokenizer.nextSentence(sentence, processing_error):
                    for word in sentence.words[
                                1:]:  # iterate from index 1, ignore <root>, which is a technical item
                        yield word

                if processing_error.occurred():
                    raise RuntimeError(processing_error.message)

            def count_triplets(triplets: Iterable[MorphologicalTriplet]) -> Iterable[
                MorphologicalTripletCounted]:
                """Gets a sequence of (possibly) repeating triplets, returns sequence of unique triplets with absolute occurrence counts."""
                counter = Counter(triplets)
                # rather use .items() instead of .most_common() to avoid triplets sorted according to frequency
                for triplet, count in counter.items():  # counter.most_common():
                    yield MorphologicalTripletCounted(triplet, count)

            words = conllu_text_to_udpipe_words(text)
            triplets = map(UD.WordToTripletConversions.word_to_triplet, words)
            counted = count_triplets(triplets)
            return counted


        ensure_raw_data_presence()

        print("Performing re-split of UD corpus {}...".format(corpus_name), file=sys.stderr)

        probs = tuple([float(val) for val in split_ratio.split(":")])
        # normalize to sum up to 1
        train_prob, dev_prob, test_prob = tuple([prob / sum(probs) for prob in probs])
        split_probabilities = SplitProbabilities(train_prob, dev_prob, test_prob)

        # TODO: check that corpus_name is valid
        if corpus_name == "UD_ToyDataset":
            # The toy dataset is from Czech, and contains only first 100 lemma-tag-form-count entries
            conllu_text = extract_all_conllu_content_from_tgz_for_corpus("UD_Czech-PDT", UD.TGZ_PATH)
            triplets_counted = list(conllu_text_to_counted_triplets(text=conllu_text))[:100]

            # toy dataset split has lemma overlap 100%
            split = triplets_counted, triplets_counted[:10], triplets_counted[-10:]

        else:
            conllu_text = extract_all_conllu_content_from_tgz_for_corpus(corpus_name, UD.TGZ_PATH)

            triplets_counted = list(conllu_text_to_counted_triplets(text=conllu_text))

            if ensure_ratio == "lemmata" and weighting == "uniform":
                split = UD.uniform_lemma_disjoint_split_ensure_lemma_count_ratio(triplets_counted, split_probabilities)
            elif ensure_ratio == "occurrences" and weighting == "uniform":
                split = UD.uniform_lemma_disjoint_split_ensure_occurrence_counts_ratio(triplets_counted,
                                                                                       split_probabilities)
            elif ensure_ratio == "lemmata" and weighting == "weighted":
                split = UD.frequency_weighted_lemma_disjoint_split_ensure_lemma_count_ratio(
                    triplets_counted,
                    split_probabilities
                )
            elif ensure_ratio == "occurrences" and weighting == "weighted":
                # This is probably the desired option
                split = UD.frequency_weighted_lemma_disjoint_split_ensure_occurrence_counts_ratio(
                    triplets_counted,
                    split_probabilities
                )
            else:
                raise RuntimeError(f"Invalid options for ensure ratio and weighting: {ensure_ratio}, {weighting}")

        train_triplets, dev_triplets, test_triplets = split

        print(UD.count_statistics_for_split(train_triplets, dev_triplets, test_triplets), file=sys.stderr)


        corpus_filepaths = UD._get_corpus_filepaths(corpus_name)

        corpus_filepaths.corpus_dir.mkdir(parents=True, exist_ok=True)
        # Write all split sets to files.
        # write_list_to_file(output_directory / ALL_TRIPLETS_NAME, triplets_counted)
        write_list_to_file(corpus_filepaths.train, train_triplets)
        write_list_to_file(corpus_filepaths.dev, dev_triplets)
        write_list_to_file(corpus_filepaths.test, test_triplets)

        print("Re-split of UD corpus {} performed.".format(corpus_name), file=sys.stderr)
        print("Resplit UD corpus files were saved to files\n" + str(corpus_filepaths) + "\n")

    class WordToTripletConversions:
        """Different conversions of UDPipe.Word to Triplet, with differing ways of matching casing of the form to the casing of the lemma."""

        @staticmethod
        def word_to_triplet(word: ufal.udpipe.Word) -> MorphologicalTriplet:
            """Convert UDPipe Word instance to morphological triplet (omitting unnecessary information)"""
            return UD.WordToTripletConversions._word_to_triplet_match_capitalization(word)

        @staticmethod
        def _word_to_triplet_match_capitalization(word: ufal.udpipe.Word) -> MorphologicalTriplet:
            """Convert UDPipe Word instance to morphological triplet (omitting unnecessary information).
            For every form, match its capitalization to the capitalization of the corresponding lemma."""

            def match_capitalization(lemma: str, form: str) -> str:
                """Match the capitalization of the given form to the capitalization of the given lemma."""
                if lemma.islower():
                    # All lowercase, e.g. "apple"
                    return form.lower()
                elif lemma.isupper():
                    # All uppercase, e.g. "USA"
                    return form.upper()
                elif lemma.istitle():
                    # Capitalized (first letter uppercase, rest lowercase), e.g. "Apple"
                    return form.capitalize()
                else:
                    # If the lemma has mixed capitalization (e.g., "McDonald"), return it as is
                    return form

            form = match_capitalization(word.lemma, word.form)

            return MorphologicalTriplet(
                lemma=word.lemma,
                form=form,
                tag=f"UPOS={word.upostag}{'|' if word.feats != '' else ''}{word.feats}"
            )

    @staticmethod
    def count_statistics_for_split(train_set: list[MorphologicalTripletCounted],
                                   dev_set: list[MorphologicalTripletCounted],
                                   test_set: list[MorphologicalTripletCounted]) -> str:

        stats_str = "split\t#unique_lemmas\t#unique_triplets\ttotal_occurrences\n"

        # print("split & #unique lemmas & #unique triplets & total occurrences \\\\", file=sys.stderr)
        # Count statistics for the split
        all_lemmas = 0
        all_triplets = 0
        all_occs = 0

        for split_set, name in [(train_set, "train"), (dev_set, "dev"), (test_set, "test")]:
            unique_lemmas_count = len(list(set([triplet.triplet.lemma for triplet in split_set])))
            all_lemmas += unique_lemmas_count

            unique_lemma_tag_form_triplets = len(split_set)
            all_triplets += unique_lemma_tag_form_triplets

            total_occurrences = sum(triplet.count for triplet in split_set)
            all_occs += total_occurrences

            stats_str += f"{name}\t{unique_lemmas_count}\t{unique_lemma_tag_form_triplets}\t{total_occurrences}\n"

            # print(f"{name} & {unique_lemmas_count} & {unique_lemma_tag_form_triplets} & {total_occurrences} \\\\", file=sys.stderr)

        stats_str += f"all\t{all_lemmas}\t{all_triplets}\t{all_occs}\n"
        # print(f"all & {all_lemmas} & {all_triplets} & {all_occs} \\\\", file=sys.stderr)

        return stats_str

    @staticmethod
    def uniform_lemma_disjoint_split_ensure_lemma_count_ratio(triplets: Iterable[MorphologicalTripletCounted],
                                                              probs: SplitProbabilities) -> \
            tuple[
                list[MorphologicalTripletCounted], list[MorphologicalTripletCounted], list[
                    MorphologicalTripletCounted]]:
        """Perform uniform lemma-disjoint split of morphological triplets into train, dev and test.
        First, create a dict {lemma: Set[triplets with such lemma]}, and then, for each lemma, randomly decide whether
        it goes to train, dev or test (according to the predefined probabilities, train+dev+test probabilities = 1)."""

        # Create a dictionary {lemma: Set[triplets with such lemma]}
        lemma_to_triplets = defaultdict(SortedSet)

        for triplet in triplets:
            lemma_to_triplets[triplet.lemma].add(triplet)

        # print(f"# unique lemmas: {len(lemma_to_triplets.items())}", file=sys.stderr)

        # Define containers for train, dev, and test sets
        train_set = []
        dev_set = []
        test_set = []

        # Iterate over the lemma to decide where to place its triplets
        for lemma, lemma_triplets in lemma_to_triplets.items():
            rand_val = random.random()

            if rand_val < probs.train:
                train_set.extend(lemma_triplets)
            elif rand_val < probs.train + probs.dev:
                dev_set.extend(lemma_triplets)
            else:
                test_set.extend(lemma_triplets)

        # TODO: co kdyz je test set nebo dev set prazdny?

        return train_set, dev_set, test_set

    @staticmethod
    def uniform_lemma_disjoint_split_ensure_occurrence_counts_ratio(triplets: Iterable[MorphologicalTripletCounted],
                                                                    probs: SplitProbabilities) -> \
            tuple[
                list[MorphologicalTripletCounted], list[MorphologicalTripletCounted], list[
                    MorphologicalTripletCounted]]:
        """Perform uniform lemma-disjoint split of morphological triplets into train, dev and test.
        First, create a dict {lemma: Set[triplets with such lemma]}, and then, for each lemma, randomly decide whether
        it goes to train, dev or test (according to the predefined probabilities, train+dev+test probabilities = 1)."""

        # Step 1: Organize triplets by lemma
        lemma_to_triplets = defaultdict(list)
        lemma_weights = defaultdict(int)

        for triplet in triplets:
            lemma_to_triplets[triplet.lemma].append(triplet)
            lemma_weights[triplet.lemma] += triplet.count

        total_count = sum(lemma_weights.values())
        train_target = int(total_count * probs.train)
        dev_target = int(total_count * probs.dev)
        test_target = total_count - train_target - dev_target

        # Step 2: Weighted sampling for train set
        train_lemmata = []
        current_train_count = 0

        # print(f"Train target: {train_target}")
        # input()

        remaining_lemmata = list(lemma_weights.keys())

        while current_train_count < train_target and remaining_lemmata:
            sampled_lemma = random.sample(remaining_lemmata, 1)[0]
            train_lemmata.append(sampled_lemma)
            current_train_count += lemma_weights[sampled_lemma]
            # print(f"Current train count: {current_train_count}")

            idx = remaining_lemmata.index(sampled_lemma)
            del remaining_lemmata[idx]

        # input()
        # print(f"DEV TARGET {dev_target}")
        # input()

        # Step 3: Weighted sampling for dev set
        dev_lemmata = []
        current_dev_count = 0
        while current_dev_count < dev_target and remaining_lemmata:
            sampled_lemma = random.sample(remaining_lemmata, 1)[0]
            dev_lemmata.append(sampled_lemma)
            current_dev_count += lemma_weights[sampled_lemma]
            # print(f"Current train count: {current_dev_count}")
            idx = remaining_lemmata.index(sampled_lemma)
            del remaining_lemmata[idx]
        # input()

        # Remaining lemmata go to test set
        test_lemmata = remaining_lemmata

        # Step 4: Partition triplets
        train = [triplet for lemma in train_lemmata for triplet in lemma_to_triplets[lemma]]
        dev = [triplet for lemma in dev_lemmata for triplet in lemma_to_triplets[lemma]]
        test = [triplet for lemma in test_lemmata for triplet in lemma_to_triplets[lemma]]

        return train, dev, test

    @staticmethod
    def frequency_weighted_lemma_disjoint_split_ensure_lemma_count_ratio(
            triplets: Iterable[MorphologicalTripletCounted],
            probs: SplitProbabilities) -> \
            tuple[
                list[MorphologicalTripletCounted], list[MorphologicalTripletCounted], list[
                    MorphologicalTripletCounted]]:
        """Perform frequency-weighted lemma-disjoint split of morphological triplets into train, dev and test.
        First, create a dict {lemma: Set[triplets with such lemma]}, determine the number of training lemmata to sample
        (according to the predefined probabilities, train+dev+test probabilities = 1), randomly sample the given number
        of training lemmata (but not uniformly randomly, but with weights equal to total number of occurrences of each
        lemma (sum of counts).
        Then, take the rest of lemmata as dev+test, and split lemmata uniformly to dev/test."""

        def weighted_random_choice_of_k_elems(arr: list, sample_weights: list[int], size: int) -> list:
            """Select randomly `size` elements (w/o replacement) from given array. Sample with given weights."""
            total_weight = sum(sample_weights)
            probabilities = np.array([w / total_weight for w in sample_weights])
            selected_elems = np.random.choice(np.array(arr), size=size, replace=False, p=probabilities)
            return selected_elems

        # Step 1: Organize triplets by lemma
        lemma_to_triplets = defaultdict(SortedSet)
        lemma_weights = defaultdict(int)

        for triplet in triplets:
            lemma_to_triplets[triplet.lemma].add(triplet)
            lemma_weights[triplet.lemma] += triplet.count

        lemmata = list(lemma_to_triplets.keys())
        weights = [lemma_weights[lemma] for lemma in lemmata]

        # Step 2: Determine sizes
        total_lemmas = len(lemmata)
        train_size = int(probs.train * total_lemmas)
        dev_size = int(probs.dev * total_lemmas)
        test_size = total_lemmas - train_size - dev_size

        # Step 3: Weighted sampling for train
        train_lemmata = weighted_random_choice_of_k_elems(arr=lemmata, sample_weights=weights, size=train_size)
        # train_lemmata = set(random.choices(lemmata, weights=weights, k=train_size))

        # Remaining lemmata for dev+test
        remaining_lemmata = [lemma for lemma in lemmata if lemma not in train_lemmata]

        # Uniform split for dev/test
        random.shuffle(remaining_lemmata)
        dev_lemmata = remaining_lemmata[:dev_size]
        test_lemmata = remaining_lemmata[dev_size:]

        # Step 4: Partition triplets
        train = [triplet for lemma in train_lemmata for triplet in lemma_to_triplets[lemma]]
        dev = [triplet for lemma in dev_lemmata for triplet in lemma_to_triplets[lemma]]
        test = [triplet for lemma in test_lemmata for triplet in lemma_to_triplets[lemma]]

        return train, dev, test

    @staticmethod
    def frequency_weighted_lemma_disjoint_split_ensure_occurrence_counts_ratio(
            triplets: Iterable[MorphologicalTripletCounted],
            probs: SplitProbabilities) -> \
            tuple[
                list[MorphologicalTripletCounted], list[MorphologicalTripletCounted], list[
                    MorphologicalTripletCounted]]:
        """Perform frequency-weighted lemma-disjoint split of morphological triplets into train, dev and test.
        First, create a dict {lemma: Set[triplets with such lemma]}, determine the total occurrence count of training
        lemmata to sample (according to the predefined probabilities, train+dev+test probabilities = 1), randomly sample
        lemmata one by one (but not uniformly randomly, but with weights equal to total number of occurrences of each
        lemma (sum of counts), until the desired count_sum is achieved.
        Then, take the rest of lemmata as dev+test, and sample lemmata to dev one by one, again, ensuring that the
        count_sum is as it should be, but now, uniformly randomly!!"""

        # Step 1: Organize triplets by lemma
        lemma_to_triplets = defaultdict(SortedSet)
        lemma_weights = defaultdict(int)

        for triplet in triplets:
            lemma_to_triplets[triplet.lemma].add(triplet)
            lemma_weights[triplet.lemma] += triplet.count

        total_count = sum(lemma_weights.values())
        train_target = int(total_count * probs.train)
        dev_target = int(total_count * probs.dev)
        test_target = total_count - train_target - dev_target

        # Step 2: Weighted sampling for train set
        train_lemmata = []
        current_train_count = 0
        remaining_lemmata = list(lemma_weights.keys())
        remaining_weights = [lemma_weights[lemma] for lemma in remaining_lemmata]

        while current_train_count < train_target and remaining_lemmata:
            sampled_lemma = random.choices(remaining_lemmata, weights=remaining_weights, k=1)[0]
            train_lemmata.append(sampled_lemma)
            current_train_count += lemma_weights[sampled_lemma]

            # Update remaining lemmata and weights
            idx = remaining_lemmata.index(sampled_lemma)
            del remaining_lemmata[idx]
            del remaining_weights[idx]

        # Step 3: Weighted sampling for dev set
        dev_lemmata = []
        current_dev_count = 0
        while current_dev_count < dev_target and remaining_lemmata:
            # sampled_lemma = random.choices(remaining_lemmata, weights=remaining_weights, k=1)[0]

            # sample to dev uniformly randomly!!
            sampled_lemma = random.choices(remaining_lemmata, k=1)[0]
            dev_lemmata.append(sampled_lemma)
            current_dev_count += lemma_weights[sampled_lemma]

            # Update remaining lemmata and weights
            idx = remaining_lemmata.index(sampled_lemma)
            del remaining_lemmata[idx]
            del remaining_weights[idx]

        # Remaining lemmata go to test set
        test_lemmata = remaining_lemmata

        # Step 4: Partition triplets
        train = [triplet for lemma in train_lemmata for triplet in lemma_to_triplets[lemma]]
        dev = [triplet for lemma in dev_lemmata for triplet in lemma_to_triplets[lemma]]
        test = [triplet for lemma in test_lemmata for triplet in lemma_to_triplets[lemma]]

        return train, dev, test
