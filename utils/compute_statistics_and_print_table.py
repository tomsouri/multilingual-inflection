#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Načte data pro jednotlivé splity ze zadaného adresáře.
def load_split(directory: Path):
    splits = {}
    for split in ['train', 'dev', 'test']:
        file_path = directory / f"{split}.tsv"
        splits[split] = pd.read_csv(
            file_path,
            delimiter="\t",
            quoting=3,  # ignoruje uvozovky
            header=None,
            names=["lemma", "form", "tag", "count"]
        )
    return splits

# Formátovací funkce – pokud hodnota zaokrouhlená na 1 des. místo je celé číslo, vytiskne se bez desetinné části.
def format_float_no_trailing(value):
    rounded = round(value, 1)
    if rounded == int(rounded):
        return str(int(rounded))
    else:
        return str(rounded)

# Formátovací funkce pro celá čísla s oddělovačem tisíců.
def format_int(n):
    return format(n, ",d")

# Vypočítá poměr ve formátu train:dev:test s normalizací (test = 1)
# Použije funkci format_float_no_trailing pro odstranění zbytečných desetinných míst.
def ratio_string(train_val, dev_val, test_val):
    if test_val == 0:
        return "0:0:1"
    t = train_val/test_val
    d = dev_val/test_val
    # Použijeme formátování podle zadání
    return f"{format_float_no_trailing(t)}:{format_float_no_trailing(d)}:1"

# Funkce pro odvození ISO kódu z názvu korpusu.
# Pokud je název ve formátu např. 'UD_English-EWT', vezme se část mezi 'UD_' a prvním pomlčkou,
# a pomocí předdefinované mapy se převede např. 'English' -> 'en'.
def get_iso_code(corpus_name):
    mapping = {
        "English": "eng",
        "German": "de",
        "French": "fr",
        "Spanish": "spa",
        "Italian": "it",
        "Russian": "ru",
        "Arabic": "ar",
        "Basque": "eus",
        "Breton": "bre",
        "Czech": "ces",
        "Chinese": "zh",
        "Portuguese": "pt",
        # Další jazyky lze přidat dle potřeby
    }
    if corpus_name.startswith("UD_"):
        without_prefix = corpus_name[3:]
        # Vezmeme první část oddělenou pomlčkou nebo podtržítkem
        for sep in ["-", "_"]:
            if sep in without_prefix:
                lang = without_prefix.split(sep)[0]
                break
        else:
            lang = without_prefix
        return mapping.get(lang, lang[:3].lower())
    return corpus_name

# Pro daný korpus (adresář) spočítá požadované statistiky:
# 1. Počet unikátních lemmat (pro train, dev, test, celkem).
# 2. Počet unikátních trojic (lemma, tag, form) (pro train, dev, test, celkem).
# 3. Součet četností (sloupec count) (pro train, dev, test, celkem).
def compute_corpus_stats(corpus_path: Path):
    splits = load_split(corpus_path)
    stats = {}

    # Unikátní lemmata pro jednotlivé splity
    lemmas = {}
    for split, df in splits.items():
        lemmas[split] = df['lemma'].nunique()
    lemmas["total"] = pd.concat(splits.values())['lemma'].nunique()

    # Unikátní trojice (lemma, tag, form)
    triples = {}
    for split, df in splits.items():
        triples[split] = len(set(zip(df['lemma'], df['tag'], df['form'])))
    all_df = pd.concat(splits.values())
    triples["total"] = len(set(zip(all_df['lemma'], all_df['tag'], all_df['form'])))

    # Součet četností
    occ = {}
    for split, df in splits.items():
        occ[split] = df['count'].sum()
    occ["total"] = sum(occ[split] for split in ['train', 'dev', 'test'])

    stats["lemmas"] = {
        "train": lemmas["train"],
        "dev": lemmas["dev"],
        "test": lemmas["test"],
        "total": lemmas["total"],
        "ratio": ratio_string(lemmas["train"], lemmas["dev"], lemmas["test"])
    }
    stats["triples"] = {
        "train": triples["train"],
        "dev": triples["dev"],
        "test": triples["test"],
        "total": triples["total"],
        "ratio": ratio_string(triples["train"], triples["dev"], triples["test"])
    }
    stats["occurrences"] = {
        "train": occ["train"],
        "dev": occ["dev"],
        "test": occ["test"],
        "total": occ["total"],
        "ratio": ratio_string(occ["train"], occ["dev"], occ["test"])
    }
    return stats

# Funkce, která ze slovníku se statistikami vytvoří LaTeX tabulku.
# Tabulka obsahuje:
# - První sloupec: název sekce (Unique lemmas, Lemma-tag-form triples, Occurrences).
# - Druhý sloupec: řádkové popisky (Train, Dev, Test, Total, Ratio).
# - Další sloupce: pro každý korpus, s názvem ve formátu \iso{lang}.
# V caption je uveden popis mappingu ISO kódů na plné názvy korpusů.
def compute_latex_summary_table(corpus_stats):
    row_labels = ["train", "dev", "test", "total", "ratio"]
    sections = ["lemmas", "triples", "occurrences"]

    # Pro každý korpus získáme data pro každou sekci – hodnoty budou list 5 položek.
    table = {section: {} for section in sections}
    for corpus, stats in corpus_stats.items():
        table["lemmas"][corpus] = [
            stats["lemmas"]["train"],
            stats["lemmas"]["dev"],
            stats["lemmas"]["test"],
            stats["lemmas"]["total"],
            stats["lemmas"]["ratio"]
        ]
        table["triples"][corpus] = [
            stats["triples"]["train"],
            stats["triples"]["dev"],
            stats["triples"]["test"],
            stats["triples"]["total"],
            stats["triples"]["ratio"]
        ]
        table["occurrences"][corpus] = [
            stats["occurrences"]["train"],
            stats["occurrences"]["dev"],
            stats["occurrences"]["test"],
            stats["occurrences"]["total"],
            stats["occurrences"]["ratio"]
        ]

    # Připravíme hlavičku s ISO kódy místo celých názvů
    corpus_iso = {corpus: f"\\iso{{{get_iso_code(corpus)}}}" for corpus in corpus_stats.keys()}
    corpus_names = list(corpus_stats.keys())
    header = " & set & " + " & ".join(corpus_iso[corpus] for corpus in corpus_names) + " \\\\"
    lines = []
    lines.append("\\begin{tabular}{" + "ll" + "r" * len(corpus_names) + "}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    # Pro každou sekci vytvoříme multirow se 5 řádky
    for section in sections:
        lines.append(f"\\multirow{{5}}{{*}}{{{section}}} & {row_labels[0]}")
        for corpus in corpus_names:
            # Formátování hodnot - pro int použijeme oddělovač tisíců, pro řetězec (ratio) necháme jak je.
            val = table[section][corpus][0]
            cell = format_int(val) if isinstance(val, (int,np.integer)) else val
            lines[-1] += f" & {cell}"
        lines[-1] += " \\\\"
        for i in range(1, len(row_labels)):
            lines.append(" & " + row_labels[i])
            for corpus in corpus_names:
                val = table[section][corpus][i]
                # If the value is either a Python int or a NumPy integer, convert it to a native int.
                if isinstance(val, (int, np.integer)):
                    cell = format_int(int(val))
                elif isinstance(val, float):
                    cell = format_float_no_trailing(val)
                else:
                    cell = val
                lines[-1] += f" & {cell}"
            lines[-1] += " \\\\"
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"  # nahradí poslední \\midrule dolní čárou
    lines.append("\\end{tabular}")
    # Přidáme caption, která obsahuje popis mappingu ISO kódů na korpusy
    #caption = "% Caption: Column headers show ISO codes for languages. Mapping: " + ", ".join(
        #f\"\\iso{{{get_iso_code(corpus)}}} => {corpus}\" for corpus in corpus_names
    #)
    return "\n".join(lines) + "\n"

# Hlavní funkce, která pro každý korpus ze zadaného datadir načte data, spočítá statistiky a vytiskne výsledky.
def main(args: argparse.Namespace) -> None:
    datadir = Path(args.datadir)
    if args.corpora:
        corpora = args.corpora.split(";")
    else:
        corpora = [d.name for d in datadir.iterdir() if d.is_dir()]
    corpus_stats = {}
    for corpus in corpora:
        corpus_path = datadir / corpus
        try:
            stats = compute_corpus_stats(corpus_path)
            corpus_stats[corpus] = stats
        except Exception as e:
            print(f"Error processing corpus '{corpus}': {e}")
    #print("JSON output:")
    #print(json.dumps(corpus_stats, indent=4, ensure_ascii=False))
    print("\nLaTeX Table:\n")
    latex_table = compute_latex_summary_table(corpus_stats)
    print(latex_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="data/processed/ud-inflection/ud-treebanks-v2.14", type=str,
                        help="Directory of the processed UD data.")
    parser.add_argument("--corpora", default=None, type=str,
                        help="';'-separated list of corpora to compute the statistics for. If not provided, all corpora in datadir are processed.")
    args = parser.parse_args()
    main(args)
