import argparse

import pandas as pd
import numpy as np
import json
from collections import Counter, defaultdict

from pathlib import Path


# Load a split from the given directory
def load_split(directory: Path):
    """
    Loads train, dev, and test splits from a given directory.

    Args:
        directory (str): Path to the directory containing train.tsv, dev.tsv, and test.tsv.

    Returns:
        dict: A dictionary with keys 'train', 'dev', and 'test', where each value is a pandas DataFrame.

    The files must be tab-separated with columns: lemma, form, tag, and count.
    """
    splits = {}
    for split in ['train', 'dev', 'test']:
        file_path = directory / f"{split}.tsv"
        splits[split] = pd.read_csv(file_path,
                                    delimiter="\t",  # Use tab as the delimiter
                                    quoting=3,  # 3 means ignore quotes completely
                                    header=None,  # No header row
                                    names=["lemma", "form", "tag", "count"]  # Specify column names
                                    )
    return splits


def general_counts(splits):
    """
    Computes general statistics and ratios for each split.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict:
            - General counts including the total number of quadruples, unique lemmas, unique forms, unique tags, and total count for each split.
            - Train:Dev:Test ratios for number of unique lemmas, total quadruples, and total count, normalized such that Test = 1.

    Importance:
        Provides an overview of dataset size and diversity. Ratios help verify that the splits are balanced, ensuring the test set is appropriately sized for evaluation.
    """
    # Compute statistics for each split
    stats = {
        split: {
            "total_quadruples": len(data),
            "unique_lemmas": data['lemma'].nunique(),
            "unique_forms": data['form'].nunique(),
            "unique_tags": data['tag'].nunique(),
            "total_count": data['count'].sum()
        }
        for split, data in splits.items()
    }

    # Extract values for ratio calculation
    train_stats = stats['train']
    dev_stats = stats['dev']
    test_stats = stats['test']

    # Normalize test to 1 for ratios
    def normalize_to_test(train_value, dev_value, test_value):
        # Determine if the value is an integer or float and format accordingly
        def format_value(value):
            return f"{int(round(value, 0))}" if round(value, 1).is_integer() else f"{value:.1f}"

        return f"{format_value(train_value / test_value)}:{format_value(dev_value / test_value)}:1"

    # Compute ratios
    ratios = {
        "unique_lemmas_ratio": normalize_to_test(
            train_stats["unique_lemmas"], dev_stats["unique_lemmas"], test_stats["unique_lemmas"]
        ),
        "total_quadruples_ratio": normalize_to_test(
            train_stats["total_quadruples"], dev_stats["total_quadruples"], test_stats["total_quadruples"]
        ),
        "total_count_ratio": normalize_to_test(
            train_stats["total_count"], dev_stats["total_count"], test_stats["total_count"]
        ),
    }

    return {
        "general_counts": stats,
        "ratios": ratios
    }


def tag_distribution(splits):
    """
    Computes tag frequency distribution for each split.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict: Tag frequencies and total tag counts for each split.

    Importance:
        Helps assess whether the tags are evenly distributed across splits. Discrepancies may indicate data leakage or biases.
    """
    return {
        split: {
            "tag_frequencies": dict(Counter(data['tag'])),
            "total_tags": len(data)
        }
        for split, data in splits.items()
    }


def lemma_distribution(splits):
    """
    Computes lemma frequency distribution for each split.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict: Frequencies of lemmas for each split.

    Importance:
        Ensures that lemmas are distributed proportionally across splits. Imbalanced lemma distributions can affect model performance.
    """
    return {
        split: dict(Counter(data['lemma']))
        for split, data in splits.items()
    }


def form_distribution(splits):
    """
    Computes form frequency distribution for each split.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict: Frequencies of forms for each split.

    Importance:
        Tracks diversity in word forms. High or low diversity in forms can indicate issues in the dataset's representativeness.
    """
    return {
        split: dict(Counter(data['form']))
        for split, data in splits.items()
    }


def count_statistics(splits):
    """
    Computes statistics related to the `count` column for each split.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict:
            - Statistics including average, median, min, max count values for each split.
            - The triplet (lemma, tag, form) associated with the min and max count.
            - The percentage of the max count relative to the total count for each split.
            - The total count and ratio for the top 10 highest counts, along with their associated triplets.
            - The total count and ratio for the top 10% highest counts, along with their associated triplets.

    Importance:
        Highlights the distribution of counts, indicating if there are particularly dominant items (high max count)
        or very rare items (low min count). Including aggregated statistics for top counts helps identify if a small
        portion of the data dominates the dataset.
    """
    result = {}
    for split, data in splits.items():
        total_count = data['count'].sum()

        # Find max and min counts and their associated triplets
        max_count_idx = data['count'].idxmax()
        min_count_idx = data['count'].idxmin()

        max_count = data.loc[max_count_idx, 'count']
        min_count = data.loc[min_count_idx, 'count']

        max_triplet = (
            data.loc[max_count_idx, 'lemma'],
            data.loc[max_count_idx, 'tag'],
            data.loc[max_count_idx, 'form']
        )
        min_triplet = (
            data.loc[min_count_idx, 'lemma'],
            data.loc[min_count_idx, 'tag'],
            data.loc[min_count_idx, 'form']
        )

        # Sort by count to get top counts and triplets
        sorted_data = data.sort_values(by='count', ascending=False).reset_index()

        # Top 10 counts and triplets
        top_10 = sorted_data.head(10)
        top_10_sum = top_10['count'].sum()
        top_10_ratio = top_10_sum / total_count if total_count > 0 else 0
        top_10_triplets = list(zip(top_10['lemma'], top_10['tag'], top_10['form']))

        # Top 10% of counts and triplets
        top_10_percent_count = max(1, int(len(sorted_data) * 0.1))  # Ensure at least one entry
        top_10_percent = sorted_data.head(top_10_percent_count)
        top_10_percent_sum = top_10_percent['count'].sum()
        top_10_percent_ratio = top_10_percent_sum / total_count if total_count > 0 else 0
        top_10_percent_triplets = list(zip(top_10_percent['lemma'], top_10_percent['tag'], top_10_percent['form']))

        result[split] = {
            "average_count": data['count'].mean(),
            "median_count": data['count'].median(),
            "min_count": min_count,
            "min_triplet": min_triplet,
            "max_count": max_count,
            "max_triplet": max_triplet,
            "max_count_ratio": max_count / total_count if total_count > 0 else 0,
            "top_10_count_sum": top_10_sum,
            "top_10_count_ratio": top_10_ratio,
            "top_10_triplets": top_10_triplets,
            "top_10_percent_count_sum": top_10_percent_sum,
            f"top_10_percent_count_ratio": (top_10_percent_ratio, f"(#triplets={len(top_10_percent_triplets)})"),
            #"top_10_percent_triplets": top_10_percent_triplets
        }
    return result

def lemma_count_statistics(splits):
    """
    Computes statistics related to the total count aggregated by lemmas for each split.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict:
            - Statistics including average, median, min, and max values for the total count aggregated by lemma for each split.
            - The max lemma count and its ratio to the total count for each split.
            - The total count, ratio, and list of lemmas for the top 10 most frequent lemmas.
            - The total count, ratio, and list of lemmas for the top 10% of the most frequent lemmas.

    Importance:
        Aggregating counts by lemmas helps identify how balanced the dataset is with respect to the distribution of data across different lemmas.
        Including the actual lists of frequent lemmas provides further insight into which lemmas dominate the dataset.
    """
    result = {}
    for split, data in splits.items():
        # Aggregate counts by lemma
        lemma_counts = data.groupby("lemma")["count"].sum().sort_values(ascending=False)
        total_count = lemma_counts.sum()
        min_lemma = lemma_counts.idxmin()
        max_lemma_count = lemma_counts.max()
        max_lemma = lemma_counts.idxmax()  # Retrieve the lemma with the maximum count

        # Top 10 lemmas
        top_10_lemmas = lemma_counts.head(10).index.tolist()
        top_10_sum = lemma_counts.head(10).sum()
        top_10_ratio = top_10_sum / total_count if total_count > 0 else 0

        # Top 10% of lemmas
        top_10_percent_count = max(1, int(len(lemma_counts) * 0.1))  # Ensure at least one lemma is included
        top_10_percent_lemmas = lemma_counts.head(top_10_percent_count).index.tolist()
        top_10_percent_sum = lemma_counts.head(top_10_percent_count).sum()
        top_10_percent_ratio = top_10_percent_sum / total_count if total_count > 0 else 0

        result[split] = {
            "average_lemma_count": lemma_counts.mean(),
            "median_lemma_count": lemma_counts.median(),
            "min_lemma": min_lemma,
            "min_lemma_count": lemma_counts.min(),
            "max_lemma": max_lemma,
            "max_lemma_count": max_lemma_count,
            "ratio_of_counts_max_lemma": max_lemma_count / total_count if total_count > 0 else 0,
            "top_10_lemma_count_sum": top_10_sum,
            "ratio_of_top_10_lemmas": top_10_ratio,
            "top_10_lemmas": top_10_lemmas,
            "top_10_percent_lemma_count_sum": top_10_percent_sum,
            f"ratio_of_top_10_percent_lemmas": (top_10_percent_ratio, f"#lemmas={len(top_10_percent_lemmas)}"),
        }
    return result


def compute_coverage(train_set, dev_or_test_set, column):
    """
    Computes the percentage coverage of a column in dev/test relative to train.

    Args:
        train_set (DataFrame): The training set.
        dev_or_test_set (DataFrame): The dev or test set.
        column (str): Column to compute coverage for ('lemma' or 'tag').

    Returns:
        float: Percentage of dev/test values covered by train.

    Importance:
        Ensures that critical linguistic features in dev/test (e.g., lemmas, tags) are present in train. Low coverage indicates gaps in training data.
    """
    train_values = set(train_set[column])
    dev_test_values = set(dev_or_test_set[column])
    return len(dev_test_values & train_values) / len(dev_test_values) if len(dev_test_values) > 0 else 0


def coverage_metrics(splits):
    """
    Computes coverage of lemmas and tags in train relative to dev and test.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict: Coverage metrics for lemmas and tags.

    Importance:
        Reveals potential data gaps and assesses whether train sufficiently represents dev/test sets.
    """
    train = splits['train']
    dev = splits['dev']
    test = splits['test']
    return {
        "lemma_coverage": {
            "train-dev": compute_coverage(train, dev, 'lemma'),
            "train-test": compute_coverage(train, test, 'lemma')
        },
        "tag_coverage": {
            "train-dev": compute_coverage(train, dev, 'tag'),
            "train-test": compute_coverage(train, test, 'tag')
        }
    }


def morphological_variety(splits):
    """
    Computes statistics related to morphological variety (number of forms per lemma) for each split,
    and additional insights about entries where lemma == form and lemma != form.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict:
            - Statistics including average, median, min, max number of forms per lemma for each split.
            - The lemma with the maximal number of forms and the list of its associated forms.
            - The number of entries where lemma == form and lemma != form.
            - Weighted counts for the same.
            - Ratios of lemma == form and lemma != form entries to all entries (unweighted and weighted).

    Importance:
        Understanding morphological variety is crucial for evaluating dataset richness and identifying any potential biases
        toward highly or sparsely inflected lemmas. Tracking lemma == form vs. lemma != form provides a clearer picture
        of trivial vs. non-trivial entries in the dataset.
    """
    result = {}
    for split, data in splits.items():
        # Calculate the number of unique forms for each lemma
        forms_per_lemma = data.groupby("lemma")["form"].nunique()
        max_forms = forms_per_lemma.max()
        max_forms_lemma = forms_per_lemma.idxmax()  # Lemma with the maximal number of forms

        # Retrieve all forms associated with the lemma with maximal forms
        max_forms_list = data[data["lemma"] == max_forms_lemma]["form"].unique().tolist()

        # Compute lemma == form statistics
        lemma_equals_form = data[data["lemma"] == data["form"]]
        lemma_equals_form_count = len(lemma_equals_form)  # Number of entries where lemma == form
        lemma_equals_form_weighted_count = lemma_equals_form["count"].sum()  # Weighted count

        # Compute lemma != form statistics
        lemma_not_equals_form = data[data["lemma"] != data["form"]]
        lemma_not_equals_form_count = len(lemma_not_equals_form)  # Number of entries where lemma != form
        lemma_not_equals_form_weighted_count = lemma_not_equals_form["count"].sum()  # Weighted count

        # Compute total counts
        total_entries = len(data)  # Total number of entries in the split
        total_weighted_count = data["count"].sum()  # Sum of all counts in the split

        # Compute ratios
        lemma_equals_form_ratio = lemma_equals_form_count / total_entries if total_entries > 0 else 0
        lemma_equals_form_weighted_ratio = (
            lemma_equals_form_weighted_count / total_weighted_count if total_weighted_count > 0 else 0
        )
        lemma_not_equals_form_ratio = lemma_not_equals_form_count / total_entries if total_entries > 0 else 0
        lemma_not_equals_form_weighted_ratio = (
            lemma_not_equals_form_weighted_count / total_weighted_count if total_weighted_count > 0 else 0
        )

        result[split] = {
            "average_forms_per_lemma": forms_per_lemma.mean(),
            "median_forms_per_lemma": forms_per_lemma.median(),
            "min_forms_per_lemma": forms_per_lemma.min(),
            "max_forms_per_lemma": max_forms,
            "max_forms_lemma": max_forms_lemma,
            "max_forms_list": max_forms_list,
            "lemma_equals_form_count": lemma_equals_form_count,
            "lemma_equals_form_weighted_count": lemma_equals_form_weighted_count,
            "lemma_equals_form_ratio": lemma_equals_form_ratio,
            "lemma_equals_form_weighted_ratio": lemma_equals_form_weighted_ratio,
            "lemma_not_equals_form_count": lemma_not_equals_form_count,
            "lemma_not_equals_form_weighted_count": lemma_not_equals_form_weighted_count,
            "lemma_not_equals_form_ratio": lemma_not_equals_form_ratio,
            "lemma_not_equals_form_weighted_ratio": lemma_not_equals_form_weighted_ratio,
        }
    return result


def consistency_check(splits):
    """
    Checks for consistency across splits, ensuring they are disjoint.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict: Number of duplicated entries and whether the splits are disjoint.

    Importance:
        Prevents data leakage by verifying that splits do not share any identical quadruples.
    """
    all_data = pd.concat([splits[split].assign(split=split) for split in splits], ignore_index=True)
    duplicated = all_data.duplicated(subset=["lemma", "form", "tag", "count"], keep=False)
    return {
        "duplicated_entries": duplicated.sum(),
        "is_disjoint": duplicated.sum() == 0
    }


def aggregate_statistics(splits):
    """
    Aggregates all computed statistics for the dataset splits.

    Args:
        splits (dict): A dictionary of pandas DataFrames for train, dev, and test splits.

    Returns:
        dict: Comprehensive statistics for all metrics.

    Importance:
        Provides a unified summary of dataset properties, enabling informed decisions about its quality and suitability.
    """
    results =  {
        "general_counts": general_counts(splits),
        # "tag_distribution": tag_distribution(splits),
        # "lemma_distribution": lemma_distribution(splits),
        # "form_distribution": form_distribution(splits),
        "statistics_of_lemma_counts": lemma_count_statistics(splits),
        "statistics_of_form-tag_counts": count_statistics(splits),
        "coverage_metrics": coverage_metrics(splits),
        "morphological_variety": morphological_variety(splits),
        "consistency_check": consistency_check(splits)
    }
    return {
        metric.upper(): format_floats_and_lists(result) for metric, result in results.items()
    }


def format_floats_and_lists(data):
    """
    Recursively formats all float values in a dictionary, list, or other nested structures to have two decimal places.

    Args:
        data: The input data, which can be a dictionary, list, or other data structure.

    Returns:
        The same data structure with all float values formatted to two decimal places.
    """
    if isinstance(data, np.int64):
        return format_floats_and_lists(int(data))  # Convert int64 to int
    elif isinstance(data, np.float64):
        return format_floats_and_lists(float(data))
    elif isinstance(data, np.bool_):
        return bool(data)  # Convert numpy.bool_ to bool
    elif isinstance(data, dict):
        return {key: format_floats_and_lists(value) for key, value in data.items()}
    # elif isinstance(data, tuple):
    # elif isinstance(data, list):
    #     return [format_floats(item) for item in data]
    elif isinstance(data, float):
        return round(data, 2)  # Formats the float to two decimal places
    elif isinstance(data, list) or isinstance(data, tuple):
        return f"[{', '.join([str(format_floats_and_lists(item)) for item in data])}]"
    else:
        return data


# Custom function to handle non-serializable types
def custom_serializer(obj):
    if isinstance(obj, np.int64):
        return int(obj)  # Convert int64 to int
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)  # Convert numpy.bool_ to bool

    raise TypeError(f"Type {type(obj)} not serializable")


def filter_trivial_entries(splits):
    """
    Filters out trivial entries (where lemma == form) from the given dataset splits.

    Args:
        splits (dict): A dictionary containing pandas DataFrames for the train, dev, and test splits.

    Returns:
        dict: A new dictionary with the same structure, but without trivial entries (lemma == form).

    Importance:
        This function removes entries that do not add morphological complexity (trivial entries),
        allowing for analysis or modeling on purely non-trivial data.
    """
    filtered_splits = {}
    for split, data in splits.items():
        # Filter out rows where lemma == form
        filtered_data = data[data["lemma"] != data["form"]]
        filtered_splits[split] = filtered_data
    return filtered_splits

def compare_dicts(original, filtered):
    """
    Compare two dictionaries with the same structure and return a comparison dictionary.

    :param original: The original dictionary.
    :param filtered: The filtered dictionary.
    :return: A new dictionary with comparisons.
    """
    def compare_values(orig_val, filt_val):
        if isinstance(orig_val, int):
            ratio = filt_val / orig_val if orig_val != 0 else 0
            return f"{filt_val} ({orig_val}: {ratio:.2f})"
        else:
            return f"{filt_val} ({orig_val})"

    def compare_recursive(orig, filt):
        if isinstance(orig, dict) and isinstance(filt, dict):
            return {key: compare_recursive(orig[key], filt[key]) for key in orig.keys()}
        else:
            return compare_values(orig, filt)

    return compare_recursive(original, filtered)


def compute_statistics(directory: Path):
    """
    Loads the dataset splits, computes all statistics, and prints them.

    Args:
        directory (str): Path to the directory containing the dataset splits.

    Returns:
        None

    Importance:
        A single method to analyze a dataset directory, compute all statistics, and present the findings.
    """
    full_result = {}
    splits = load_split(directory)
    non_trivial_filtered_splits = filter_trivial_entries(splits)

    stats = aggregate_statistics(splits)
    full_result["FULL"] = stats

    filtered_stats = aggregate_statistics(non_trivial_filtered_splits)

    compared_stats = compare_dicts(original=stats, filtered=filtered_stats)

    full_result["FILTERED (the same data after removing all entries with lemma==form)"] = compared_stats

    # for (split_type, split_dict) in [("full", splits), ("filtered_from_trivial", non_trivial_filtered_splits)]:
    #     stats = aggregate_statistics(split_dict)
    #     full_result[split_type.upper()] = {
    #         metric.upper(): format_floats_and_lists(result) for metric, result in stats.items()
    #     }
        # for metric, result in stats.items():
        #     full_result[metric.upper()] = format_floats_and_lists(result)

    return full_result


def list_directories(path: Path):
    try:
        # List only directories
        directories = [d.name for d in path.iterdir() if d.is_dir()]
        return directories
    except FileNotFoundError:
        return "The specified path does not exist."
    except PermissionError:
        return "You do not have permission to access this directory."
    
def compute_latex_summary_table(results):
    table = defaultdict(dict)
    
    for corpus, corpus_data in results.items():
        full_data = corpus_data["FULL"]

        # Extract necessary counts
        lemma_stats = full_data["GENERAL_COUNTS"]["general_counts"]
        triples_stats = {split: len(set(zip(
            lemma_stats[split]["unique_lemmas"],
            splits[split]["tag"],
            splits[split]["form"]
        ))) for split in ["train", "dev", "test"]}
        
        total_lemmas = sum(lemma_stats[split]["unique_lemmas"] for split in ["train", "dev", "test"])
        total_triples = sum(triples_stats[split] for split in ["train", "dev", "test"])
        total_count = sum(lemma_stats[split]["total_count"] for split in ["train", "dev", "test"])

        # Compute ratios (normalize test to 1)
        def ratio_string(t, d, te):
            if te == 0:
                return "0:0:1"
            return f"{round(t/te, 1)}:{round(d/te, 1)}:1"

        table["Unique lemmas"][corpus] = [
            lemma_stats["train"]["unique_lemmas"],
            lemma_stats["dev"]["unique_lemmas"],
            lemma_stats["test"]["unique_lemmas"],
            total_lemmas,
            ratio_string(
                lemma_stats["train"]["unique_lemmas"],
                lemma_stats["dev"]["unique_lemmas"],
                lemma_stats["test"]["unique_lemmas"]
            )
        ]

        table["Lemma-tag-form triples"][corpus] = [
            triples_stats["train"],
            triples_stats["dev"],
            triples_stats["test"],
            total_triples,
            ratio_string(triples_stats["train"], triples_stats["dev"], triples_stats["test"])
        ]

        table["Occurrences"][corpus] = [
            lemma_stats["train"]["total_count"],
            lemma_stats["dev"]["total_count"],
            lemma_stats["test"]["total_count"],
            total_count,
            ratio_string(
                lemma_stats["train"]["total_count"],
                lemma_stats["dev"]["total_count"],
                lemma_stats["test"]["total_count"]
            )
        ]

    # Generate LaTeX table
    corpus_names = list(results.keys())
    header = " & Metric & " + " & ".join(corpus_names) + " \\\\"
    lines = ["\\begin{tabular}{ll" + "r"*len(corpus_names) + "}", "\\toprule", header, "\\midrule"]
    for section in ["Unique lemmas", "Lemma-tag-form triples", "Occurrences"]:
        lines.append(f"\\multirow{{6}}{{*}}{{{section}}}")
        for i, label in enumerate(["Train", "Dev", "Test", "Total", "Train:Dev:Test ratio"]):
            if i > 0:
                lines.append(" & " + label)
            else:
                lines[-1] += f" & {label}"
            for corpus in corpus_names:
                lines[-1] += f" & {table[section][corpus][i]}"
            lines[-1] += " \\\\"  # end row
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"  # replace last midrule with bottomrule
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    datadir = Path(args.datadir)
    if args.corpora:
        corpora = args.corpora.split(";")
    else:
        corpora = list_directories(datadir)

    results = {}
    for corpus in corpora:
        corpus_path = datadir / corpus
        results[corpus] = compute_statistics(corpus_path)

    print(json.dumps(results, indent=4, default=custom_serializer, ensure_ascii=False, sort_keys=True))
    
    if args.latex:
        with open(args.latex, "w") as f:
            f.write(compute_latex_summary_table(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data/processed/ud-inflection/ud-treebanks-v2.14", type=str,
                        help="Directory of the processed UD data directory.")
    parser.add_argument("--corpora", default=None, type=str,
                        help="';'-separated list of corpora to compute the statistics for. If None, computes stats for all corpora present in the datadir.")
    parser.add_argument("--latex", default=None, type=str,
                        help="Print LaTeX table with simplified stats to given file.")
    

    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
