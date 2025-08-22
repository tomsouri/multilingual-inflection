import argparse
import os
from pathlib import Path


def load_file(filepath):
    """Load the content of a TSV file into a list of rows."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return [line.strip().split('\t') for line in file]


def save_file(filepath, content):
    """Save a list of rows into a TSV file."""
    with open(filepath, 'w', encoding='utf-8') as file:
        file.writelines(['\t'.join(row) + '\n' for row in content])


def enrich_testset(test_filepath, freq_filepath, freqtest_filepath, key_level):
    """
    Enrich the test set by adding frequency data from the frequency list.

    Args:
        test_filepath (str): Path to the test set file.
        freq_filepath (str): Path to the frequency list file.
        freqtest_filepath (str): Path to save the enriched test set.
        key_level (int): Level of key to use (1=lemma, 2=lemma+form, 3=lemma+form+tag).
    """
    # Load files
    test_data = load_file(test_filepath)
    freq_data = load_file(freq_filepath)

    # Create a frequency dictionary
    freq_dict = {}
    for row in freq_data:
        if len(row) >= 4:
            key = tuple(row[:key_level])  # Key based on the selected level

            # If key_level==0 means that only the form should be used as a key
            if key_level == 0:
                key = row[1]
            freq_dict[key] = row[3]  # count

    # Enrich the test set
    enriched_test = []
    missing_count = 0

    for row in test_data:
        if len(row) >= 3:
            key = tuple(row[:key_level])

            # If key_level==0 means that only the form should be used as a key
            if key_level == 0:
                key = row[1]

            count = freq_dict.get(key, '0')  # Default frequency is '0' if not found
            if count == '0':
                missing_count += 1
            # else:
                # pass
                # print(row)
                # print(count)
                # input()
            enriched_test.append(row + [count])
        # else:
        #     print(row)
        #     input()

    # Calculate and print percentage of missing items
    total_items = len(test_data)
    missing_percentage = (missing_count / total_items) * 100 if total_items > 0 else 0
    #print(f"Missing items: {missing_count}/{total_items} ({missing_percentage:.2f}%)")

    # Save enriched test set
    save_file(freqtest_filepath, enriched_test)

    return f"{missing_percentage:.2f}%\t{100-missing_percentage:.2f}%\t{total_items - missing_count}/{total_items}"


def enrich_all_data() -> None:
    dirs = {
        21: Path("2021Task0/part1/ground-truth/"),
        22: Path("2022InflectionST/part1/development_languages/")
    }
    langs = {
        21 : [("ara", "ar", "ara"), ("deu", "de", "deu"), ("spa", "es", "esp"), ("tur", "tr", "tur")],
        #21 : [("ara", "ar", "ara")],
        22 : [("ara", "ar", "ara"), ("tur", "tr", "tur")]
    }

    suffixes = {
        21 : ".test",
        22 : ".gold"
    }

    tgt_dir = Path("gold-data/freq/")
    freq_dir = Path("ACL2023_RealityCheck/data/freq_lists/")
    print("\t".join(["year", "lang", "key-for-match", "missing (%)", "overlap (%)", "present/all"]))
    for include_to_key in ["form", "lemma", "lemma-form", "lemma-form-tag"]:
    #for include_to_key in ["lemma", "lemma-form", "lemma-form-tag"]:
        key_count = 1 if include_to_key == "lemma" else 2 if include_to_key == "lemma-form" else 3 if include_to_key=="lemma-form-tag" else 0
        for year in [21, 22]:
            for lang in langs[year]:
                lang_orig = lang[0]
                lang_freq = lang[1]
                lang_tgt = lang[2]
                orig_testfile = dirs[year] / f"{lang_orig}{suffixes[year]}"
                freq_file = freq_dir / f"{lang_freq}_freq.txt"
                tgt_file = tgt_dir / str(year) / f"{lang_tgt}.test"
                os.makedirs(tgt_dir / str(year), exist_ok=True)

                percentage_missing = enrich_testset(test_filepath=orig_testfile, freq_filepath=freq_file, freqtest_filepath=tgt_file, key_level=key_count)
                print("\t".join([str(year), lang_tgt, include_to_key, percentage_missing]))
        print("-------------------------")

def main():

    # TODO: pokud beru jako klic pouze lemma, nebo lemma-form, musim ty freqs z freq listu scitat, ne zamenovat jeden za druhy! (a pak idealne vydelit poctem lemma-tag-form triples, mezi ktere to rozdeluju)

    enrich_all_data()
    exit(0)

    parser = argparse.ArgumentParser(description="Enrich a test set with frequency data.")
    parser.add_argument("--test", required=True, help="Path to the test set file.")
    parser.add_argument("--freq", required=True, help="Path to the frequency list file.")
    parser.add_argument("--freqtest", required=True, help="Path to save the enriched test set.")
    parser.add_argument(
        "--key", type=int, choices=[1, 2, 3], default=3,
        help="Key level to match (1=lemma, 2=lemma+form, 3=lemma+form+tag). Default is 3."
    )

    args = parser.parse_args()

    # Validate file paths
    if not os.path.exists(args.test):
        raise FileNotFoundError(f"Test file not found: {args.test}")
    if not os.path.exists(args.freq):
        raise FileNotFoundError(f"Frequency file not found: {args.freq}")

    # Enrich the test set
    enrich_testset(args.test, args.freq, args.freqtest, args.key)


if __name__ == "__main__":
    main()
