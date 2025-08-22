#!/usr/bin/env python3

"""Merge original sigmorphon baseline results on dev/test data with our results, select given column, round the accuracies, make the winning system bold and print as a TeX table.
Example usages:

# print comparison table for all .dev files in the given directory, computing avg accuracy for each system
python3 merge_sigmorphon_results.py --directory path/to/results/directory --suffix dev

# merge an existing TeX table (from previous paper) with the new columns (new competing systems), skip rows for given languages, compute avg
python3 merge_sigmorphon_results.py --directory path/to/results/directory --suffix test --existing_table "results/sigmorphon/22/large-trainset-feats-overlap/oovs-paper-results.tex" --skip_langs="LARGE;SMALL;TOTAL;_small;kaz;hye;heb;evn"

# e.g.
.venv/bin/python3 merge_sigmorphon_results.py --directory sig-results/22/large-trainset-full --suffix test --existing_table "sig-results/22/large-trainset-feats-overlap/oovs-paper-results.tex" --skip_langs="LARGE;SMALL;TOTAL;_small;kaz;hye;heb;evn"

.venv/bin/python3 merge_sigmorphon_results.py --directory sig-results/23/large-trainset-full --suffix test --existing_table "sig-results/23/original_paper_table.tex" --column 5


"""

import argparse
import os, sys
import csv
import pandas as pd
from collections import defaultdict
import re


def read_tsv(file_path, column_index):
    """
    Read a TSV file and extract the 'lang' column and the specified data column.
    Returns a dictionary where keys are 'lang' values and values are the corresponding data column values.
    """
    lang_data = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        header = next(reader, None)  # Try to read the header, return None if empty file
        if header is None:  # or not any(reader):  # Skip empty files or files without data rows
            print(f"Skipping empty file: {file_path}", file=sys.stderr)
            return {}
        for row in reader:
            lang = row[0]  # The first column is 'lang'
            if lang.startswith("PREDICTION"):
                # TODO: check whether there is a missing empty line at the end
                # skip lines of type "PREDICTION (1999) AND EVAL (2000) FILES HAVE DIFFERENT LENGTHS. SKIPPING hun_large..."
                continue
            data = row[column_index]  # The given column
            lang_data[lang] = data
    return lang_data


def add_columns_to_latex_table(latex_string, additional_columns):
    """
    Adds additional columns to the 'tabular' environment in a LaTeX string.

    Parameters:
    - latex_string: str, the LaTeX string containing the table.
    - additional_columns: int, the number of columns to add.

    Returns:
    - str, the modified LaTeX string with added columns.
    """
    # Step 1: Use regular expressions to match the column format in the tabular environment
    match = re.search(r'\\begin{tabular}\{(.*?)\}', latex_string)

    if match:
        # Extract the existing column format
        existing_columns = match.group(1)

        # Step 2: Add the additional columns to the format
        new_columns = existing_columns + "|" + "c" * additional_columns  # Adding 'c' for each new column

        # Step 3: Replace the existing column format with the new one
        latex_string = latex_string.replace(existing_columns, new_columns, 1)

    return latex_string


def extract_textbf_value(input_string):
    """
    Extracts the value inside \textbf{} from the input string.
    If the string doesn't match the pattern, returns the original string.

    Parameters:
    input_string (str): The input string to process.

    Returns:
    str: The extracted value or the original string.
    """
    match = re.match(r"\\textbf\{(.+?)\}", input_string)
    return match.group(1) if match else input_string


def load_existing_table(file_path):
    """
    Load an existing LaTeX table, extracting the header, footer, and data.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header, footer, tabular_block = [], [], []
    in_tabular = False

    for line in lines:
        if in_tabular:
            tabular_block.append(line)
            if '\\end{tabular}' in line:
                in_tabular = False
                footer.extend(lines[lines.index(line):])
                break
        else:
            header.append(line)
            if "multicolumn" in line or "lang" in line:
                in_tabular = True

    if not tabular_block:
        raise ValueError("Invalid LaTeX table format in the provided file.")

    # Extract column names and data rows
    tabular_text = ''.join(tabular_block)
    rows = re.findall(r'.*\\\\', tabular_text)

    data = []
    for row in rows:
        if "multicolumn" in row:
            header.append(row)
        elif "average" in row:
            continue
        else:
            cells = re.split(r'&', row.strip('\\').strip())
            cells = [cell.strip() for cell in cells]
            cells = [extract_textbf_value(cell) for cell in cells]
            data.append(cells)

    # Create a DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])

    header = "".join(header)
    footer = "".join(footer)

    return header, footer, df


def remove_datetime_from_string(input_string):
    # Regular expression to match the datetime format YYYY-MM-DD-HH-MM-SS
    pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_'

    # Use re.sub to remove all occurrences of the datetime pattern
    result = re.sub(pattern, '', input_string)

    # Return the modified string
    return result


def main():
    # Set up argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Process TSV files and compute inner product.")
    parser.add_argument("--directory", type=str, default="results/sigmorphon/22/large-trainset-full",
                        help="Path to the source directory")
    parser.add_argument("--suffix", type=str, choices=["dev", "test"], default="dev",
                        help="Suffix of files to process (e.g., dev)")
    parser.add_argument("--column", type=int, default=4, help="The column number to process (0-indexed)")
    parser.add_argument("--precision", type=int, default=1, help="Number of digits for rounding")
    parser.add_argument("--existing_table", type=str, default=None,
                        help="Path to an existing LaTeX table with results from previous paper")
    parser.add_argument("--skip_langs", type=str, default=None, help="Which rows from evaluation should be skipped.")

    args = parser.parse_args()

    # Ensure the directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory {args.directory} does not exist.", file=sys.stderr)
        return

    # Dictionary to store lang values and corresponding data for each file
    lang_data_dict = defaultdict(dict)

    new_systems_count = 0

    # Iterate over all files in the directory and process those matching the suffix
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file.endswith(args.suffix):
                file_path = os.path.join(root, file)
                file_name_without_suffix = os.path.splitext(file)[0]  # Get the file name without extension
                file_name_without_suffix = remove_datetime_from_string(file_name_without_suffix)
                print(f"Processing file: {file_path}", file=sys.stderr)

                # Read the file and extract lang and the specified column
                lang_data = read_tsv(file_path, args.column)

                # Store the extracted data with the file name as the key
                for lang, value in lang_data.items():
                    lang_data_dict[lang][file_name_without_suffix] = value

                if lang_data:
                    new_systems_count += 1

    # Convert the lang_data_dict to a DataFrame for easy manipulation
    df = pd.DataFrame.from_dict(lang_data_dict, orient='index')

    # Fill any missing values with NaN (if a lang was not found in a particular file)
    df = df.fillna('NaN')

    # Round all numeric values to 1 decimal place
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, invalid parsing will be NaN

    # Modify the first column (index):
    # 1. Remove rows matching --skip_langs
    if args.skip_langs:
        skip_langs = args.skip_langs.split(";")
        mask = df.index.to_series().apply(lambda x: not any(skip in x for skip in skip_langs))
        df = df[mask]

    # 2. Remove "_large" from cells in the first column
    df.index = df.index.to_series().str.replace("_large", "", regex=False)

    # If an existing table is provided, load it and merge
    if args.existing_table:
        header, footer, existing_df = load_existing_table(args.existing_table)
        existing_df.set_index(existing_df.columns[0], inplace=True)
        combined_df = existing_df.join(df, how='outer')
    else:
        header = "\\begin{table}[htbp]\n\centering\n"
        footer = "\\end{table}"
        combined_df = df

    # Round all numeric values to the specified precision
    combined_df = combined_df.apply(pd.to_numeric, errors='coerce').round(args.precision)

    # Calculate the mean of each column (ignoring the index column)
    avg_row = combined_df.mean()

    # Add a new row with index 'avg'
    combined_df.loc['\\textbf{macro avg}'] = avg_row

    combined_df = combined_df.apply(pd.to_numeric, errors='coerce').round(args.precision)

    for index, row in combined_df.iterrows():
        max_value = row.max()  # Find the max value in the row
        if max_value != 0.0:
            max_columns = row[row == max_value].index  # Get all columns with the max value (in case of ties)

            for col in max_columns:
                combined_df[col] = combined_df[col].astype(str)
                combined_df.at[index, col] = f"\\textbf{{{max_value}}}"

    combined_df = combined_df.map(lambda x: f"{x:.{args.precision}f}" if isinstance(x, (int, float)) else x)

    # Replace all underscores with hyphens in the header (column names)
    combined_df.columns = combined_df.columns.str.replace('_', '-', regex=True)

    combined_df = combined_df.replace('nan', '-')

    # Generate and print the LaTeX table
    combined_df.reset_index(inplace=True)
    latex_table = combined_df.to_latex(index=False, escape=False)  # escape=False to preserve LaTeX commands

    if args.existing_table:
        # Extract only the raw data (lines between \toprule and \bottomrule)
        raw_data = "\n".join(
            line for line in latex_table.splitlines()
            if not (line.startswith("\\begin{") or line.startswith("\\end{") or line.startswith(
                "\\toprule") or line.startswith("\\midrule") or line.startswith("\\bottomrule"))
        )

        header = add_columns_to_latex_table(header, new_systems_count)
    else:
        raw_data = latex_table

    print(header + raw_data + "\n" + footer)


if __name__ == "__main__":
    main()
