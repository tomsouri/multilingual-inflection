import argparse
import os, sys
from pathlib import Path
from typing import TextIO


class SigmorphonCompatibility:
    TRAIN_FILENAME="train.tsv"
    DEV_FILENAME="dev.tsv"
    TEST_FILENAME="test.tsv"
    #
    # shared_task_data_directory = Path("data/raw/2023InflectionST/part1/data")
    # shared_task_converted_data_directory = Path("data/processed/2023_SIGMPORHON_Inflection_ST")


    @staticmethod
    def convert_sigmorphon_file_to_inflector_format(sigmorphon_file: TextIO, inflector_file: TextIO) -> None:
        """Converts file in SIGMORPHON 2023 Shared task format to OUR format.
        It changes the order of the items in a row (lemma, tag, form -> lemma, form, tag) and adds count column
        (always 1, the SIGMORPHON data are missing the information about counts), and changes the format of tag
        (replaces ';' with '|', such that the morphological features would always be split by '|' character).

        Example of SIGMORPHON 2023 format:
        Arabize V;PRS;NOM(3,SG) Arabizes
        Arabize V;V.PTCP;PRS    Arabizing

        Example of our format (the same data after conversion):
        Arabize	V|PRS|NOM(3,SG)	Arabizes	1
        Arabize	V|V.PTCP|PRS	Arabizing	1
        """
        print(f"Processing {sigmorphon_file.name} file...", file=sys.stderr)
        for line in sigmorphon_file:
            lemma, form, tag = line.strip().split("\t")
            tags = tag.split(";")
            inflector_file.write(f"{lemma}\t{form}\t{'|'.join(tags)}\t1\n")
    #
    # @staticmethod
    # def convert_whole_sigmorphon_directory(sigmorphon_directory: Path, output_directory: Path) -> None:
    #     """Call `convert_sigmorphon_file_to_inflector_format` method on all files in the given directory, and place
    #     the converted files into the output_directory."""
    #
    #     print(f"Converting all files from directory {sigmorphon_directory} into our format...", file=sys.stderr)
    #
    #     # Ensure the output directory exists
    #     output_directory.mkdir(parents=True, exist_ok=True)
    #
    #     # Iterate over all files in the sigmorphon directory
    #     for sigmorphon_file in sigmorphon_directory.glob('*'):
    #         # Check if the item is a file
    #         if sigmorphon_file.is_file() and "covered" not in sigmorphon_file.name:
    #             # omit files missing the `form` column (those with "covered" in filename, such as `eng.covered.tst`)
    #
    #             # Define the output file path
    #             output_file = output_directory / sigmorphon_file.name
    #
    #             # Open the input and output files
    #             with sigmorphon_file.open('r', encoding='utf-8') as input_file, \
    #                     output_file.open('w', encoding='utf-8') as output_file:
    #                 # Call the conversion function
    #                 SigmorphonCompatibility.convert_sigmorphon_file_to_inflector_format(input_file, output_file)
    #
    #     print(f"Converted files in our format were saved to {output_directory}", file=sys.stderr)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Prepare Sigmorphon data.")

    # Adding command-line arguments
    parser.add_argument('--train_file', type=str, required=True, help="Path to the training file.")
    parser.add_argument('--dev_file', type=str, required=True, help="Path to the development file.")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the test file.")
    parser.add_argument('--tgt_dir', type=str, required=True, help="Directory to store the prepared data.")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments for debugging
    print(f"Training file: {args.train_file}")
    print(f"Development file: {args.dev_file}")
    print(f"Test file: {args.test_file}")
    print(f"Target directory: {args.tgt_dir}")

    # Ensure the target directory exists
    if not os.path.exists(args.tgt_dir):
        print(f"Creating target directory: {args.tgt_dir}")
        os.makedirs(args.tgt_dir)

    # Logic for preparing the data (example)
    # This would normally involve reading the files and processing them accordingly
    # For this example, we'll just copy the input files into the target directory
    print("Preparing data...")

    for file_type, src_file_path, tgt_filename in zip(["train", "dev", "test"],
                                    [args.train_file, args.dev_file, args.test_file],
                                    [SigmorphonCompatibility.TRAIN_FILENAME, SigmorphonCompatibility.DEV_FILENAME, SigmorphonCompatibility.TEST_FILENAME]):
        if os.path.exists(src_file_path):
            target_file_path = os.path.join(args.tgt_dir, tgt_filename)
            # In real case, you'd process the file; here, we just copy it for demonstration
            with open(src_file_path, 'r') as src, open(target_file_path, 'w') as tgt:
                SigmorphonCompatibility.convert_sigmorphon_file_to_inflector_format(
                    sigmorphon_file=src,
                    inflector_file=tgt
                )

            print(f"Prepared {file_type} data at: {target_file_path}")
        else:
            print(f"{file_type.capitalize()} file does not exist: {src_file_path}")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
