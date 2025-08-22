# multilingual-inflection - Flexing in 73 Languages: A Single Small Model for Multilingual Inflection

This repository contains all code from the paper Flexing in 73 Languages: A Single Small Model for Multilingual
Inflection. It is available under the CC BY-NC-SA 4.0 license. The work is described in the paper:

The paper by Tomáš Sourada and Jana Straková, presented at TSD 2025: [Flexing in 73 Languages: A Single Small Model for Multilingual Inflection](https://link.springer.com/chapter/10.1007/978-3-032-02551-7_5)

[A master thesis Neural Models for Multilingual Inflection](http://hdl.handle.net/20.500.11956/199280) describes the work more profoundly, including all theoretical and technical details.

## Multilingual inflection

We present a compact, single-model approach to multilingual
inflection, the task of generating inflected word forms from base
lemmas to express grammatical categories. Our model, trained jointly
on data from 73 languages, is lightweight, robust to unseen words, and
outperforms monolingual baselines in most languages. This demonstrates
the effectiveness of multilingual modeling for inflection and highlights its
practical benefits: simplifying deployment by eliminating the need to
manage and retrain dozens of separate monolingual models.
In addition to the standard SIGMORPHON shared task benchmarks, we
evaluate our monolingual and multilingual models on 73 Universal Dependencies 
(UD) treebanks, extracting lemma-tag-form triples and their
frequency counts. To ensure realistic data splits, we introduce a novel
frequency-weighted, lemma-disjoint train-dev-test resampling procedure.
Our work addresses the lack of an open-source, general-purpose, multilingual 
morphological inflection system capable of handling unseen words
across a wide range of languages, including Czech.


### On Linux
To prepare the virtual environment and data and run a toy training, run:
`bash init.sh`

This:
1) creates the venv and installs dependencies, 
2) downloads and re-splits the data, 
3) prints the re-split statistics to a file `data-stats.<datetime>.txt` (this should be the same as the provided `stats.txt`)
4) runs a toy example of monolingual and multilingual training on 2 languages

For actually running some experiments, uncomment the appropriate lines in `example_run.sh` (bottom of the file) and run it (you need to have the venv and data prepared first).

To inspect what is being done, look into `example_run.sh`.
