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

# TODO docstring, citation, installation, usage examples

import os
import time
import argparse
import datetime
import re
import sys
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torchmetrics

from typing import Iterable

from inflector_dataset import InflectionDataset
from process_ud_dir import UD
from morpho_dataset import MorphoDataset, Vocabulary, get_shortened_list_string

from inflector_model import RNNEncoderDecoderWithAttention, BartWrapper, BaseInflectorModel, CopyBaseline

# Force CPU fallback for PyTorch with the MPS device due to some operators not
# implemented in MPS.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

os.environ.setdefault("KERAS_BACKEND", "torch")

# This is only set for debugging to force all CUDA calls to be synchronous.
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#
# import transformers

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--integration", default=None, type=str,
                    help="Identifier of an integration test, which will be run. Ignore the rest of arguments, set them to pre-defined values, and run integration test (identified by the given name) and prepend its results to the results file.")
parser.add_argument("--integration_results_file", default="logs/integration_tests.txt", type=str,
                    help="File to prepend integration results.")

parser.add_argument("--run_lemma_tag_implies_form_exp", default=False, action="store_true",
                    help="Run the `lemma-tag-implies-form` experiment, computing the upper bound on performance of models on given corpora.")

parser.add_argument("--checkpoint_selection", default=False, action="store_true",
                    help="After training, choose the best checkpoint (based on eval on the dev set).")
parser.add_argument("--acc_for_ckpt_selection", default="w-accuracy", type=str, choices=["w-accuracy", "accuracy", "multiling-acc"],
                    help="Accuracy value to be used for checkpoint selection. `w-accuracy`=weighted (micro avg), `accuracy`=uniform (macro avg). `--weighted_accuracy_during_training` must be passed if weighted acc is used for ckpt selection.")

parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--show_results_every_batch", default=10000, type=int, help="Show results every given batch.")
parser.add_argument("--print_first_preds_on_dev", default=100, type=int, help="Print first `given` predictions on dev set.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--corpus", default="UD_SimpleToyDataset", type=str,
                    help="Corpus to run the experiments on, or a separated list of corpora (;-separated by default, a custom separator can be defined in the `--corpus_name_separator` argument).")
parser.add_argument("--corpus_name_separator", default=";", type=str,
                    help="Separator used to separate multiple corpora names")
parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
parser.add_argument("--result_dir", default="results", type=str, help="Results-dir name.")
parser.add_argument("--eval_also_on_test", default=False, action="store_true",
                    help="Evaluate trained model also on the test set.")


parser.add_argument("--jobid", default="0", type=str, help="Job ID to identify different runs of the same experiment.")
parser.add_argument("--datadir", default=None, type=str,
                    help="Directory of the processed UD data directory. If not specified, filled in based on the `data` argument.")
parser.add_argument("--data", default="ud", type=str, choices=["none", "ud", "sig22", "sig23"],
                    help="Specification of the data to be used.")
parser.add_argument("--predictions_dir", default=None, type=str, help="Path to print predictions.")
parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size for transformer model")
parser.add_argument("--weight_decay", default=None, type=float, help="Weight decay to use in L2 regularization.")
parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--lr_decay", default="none", choices=["none", "linear", "cosine"], help="Learning rate decay.")
parser.add_argument("--clip_grad_norm", default=None, type=float, help="Use gradient clipping with given norm.")
parser.add_argument("--weighted_accuracy_during_training", default=False, action="store_true",
                    help="Evaluate also the weighted accuracy during training. Cannot be combined with `weighted_multilingual_accuracy_during_training`.")
parser.add_argument("--weighted_multilingual_accuracy_during_training", default=False, action="store_true",
                    help="Macro-avg accuracy over all languages will be stored in multiling-acc during training (it is the same as macro-avg of uniform accuracy over all langs). Cannot be combined with `weighted_accuracy_during_training`.")

parser.add_argument("--multilingual_training", default=False, action="store_true",
                    help="Train a single model on a joint train set of all corpora, during training evaluate on a joint dev set, and then evaluate separately on each dev/test set.")

parser.add_argument("--frequency_weighted_training_temperature", default=None, type=float,
                    help="Use frequencies of items during training, adjusted by given temperature (0.0 - uniform weights, 1.0 - exact corpus frequencies)")
parser.add_argument("--multilingual_corpus_down_up_sampling_temperature", default=None, type=float,
                    help="When training in multilingual setting, use 1/corpus_len as weights for items during training, adjusted by given temperature (0.0 - uniform weights, 1.0 - exact 1/corpus_len weights)")
parser.add_argument("--multilingual_corpus_down_up_sampling_weights", default="unique", choices=["unique", "total"],
                    help="If `multilingual_corpus_down_up_sampling_temperature` is not None, which corpus_len coefficients should be used to compute the 1/corpus_len weights? `unique`: corpus_len=#unique lemma-tag-form triples in the corpus, `total`: corpus_len=sum(corpus_freq for each item in dataset)")

parser.add_argument("--model", default="rnn", choices=["rnn", "transformer", "copy-bsln"],
                    help="Model to be used.")

parser.add_argument("--add_separator_token", default=False, action="store_true",
                    help="Add separator token to separate lemma from tags in the input.")



# parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--rnn_embedding_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--rnn_layer_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--rnn_layer_count", default=1, type=int,
                    help="Number of layers in RNN cell in the encoder. (In decoder, it is always 1.)")
parser.add_argument("--rnn_drop", default=0.0, type=float, help="Dropout in RNN cells.")
parser.add_argument("--rnn_hidden_drop", default=0.0, type=float, help="Dropout on the encoder output.")
parser.add_argument("--rnn_attn_drop", default=0.0, type=float,
                    help="Attention dropout in the decoder. Probably there is an unproved bug in the implementation of the attention dropout, rather do not use it.")
parser.add_argument("--rnn_unidirectional", default=False, action="store_true",
                    help="Use unidirectional RNN in the encoder. (If not set, bidirectional RNN is used.)")
parser.add_argument("--rnn_tie_embeddings", default=False, action="store_true", help="Tie target embeddings.")

parser.add_argument("--trm_layer_dim", default=256, type=int, help="Dimensionality of the layers and the pooler layer.")
parser.add_argument("--trm_layer_count", default=3, type=int,
                    help="Number of layers both in the encoder and the decoder.")
parser.add_argument("--trm_attn_heads", default=4, type=int,
                    help="Number of attention heads for each attention layer in the Transformer decoder and encoder.")

parser.add_argument("--trm_ff_nn_dim", default=512, type=int,
                    help="Dimensionality of the “intermediate” (often named feed-forward) layer in decoder and encoder.")

parser.add_argument("--trm_drop", default=0.0, type=float,
                    help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.")
parser.add_argument("--trm_attn_drop", default=0.0, type=float,
                    help="The dropout ratio for the attention probabilities.")
parser.add_argument("--trm_feed_drop", default=0.0, type=float,
                    help="The dropout ratio for activations inside the fully connected layer (called `activation_dropout` in BART config).")
parser.add_argument("--trm_layer_drop", default=0.0, type=float,
                    help="The LayerDrop probability for the decoder and encoder in transformer. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more details.")

parser.add_argument("--trm_add_final_layer_norm", default=False, action="store_true",
                    help="Transformer: final layer normalization")
parser.add_argument("--trm_scale_embedding", default=False, action="store_true",
                    help="Transformer: scale embeddings by diving by sqrt(d_model)")
parser.add_argument("--trm_normalize_before", default=False, action="store_true",
                    help="Transformer: normalization before")

parser.add_argument("--trm_gen_num_beams", default=None, type=int,
                    help="Transformer generation: beam search. If >1, beam search is turned of with the specified number of beams.")
parser.add_argument("--trm_gen_penalty_alpha", default=None, type=float,
                    help="Transformer generation: contrastive search, penalty_alpha parameter. Recommended values are from 0.6-1.0. The values balance the model confidence and the degeneration penalty in contrastive search decoding.")
parser.add_argument("--trm_gen_top_k", default=None, type=int,
                    help="Transformer generation: contrastive search, top_k parameter. Recommended values are from 4 to 10. The number of highest probability vocabulary tokens to keep for top-k-filtering.")
parser.add_argument("--trm_gen_turn_sampling_off_during_training", default=False, action="store_true",
                    help="Transformer generation: if True, sampling is turned off during training (for faster dev evaluation between epochs) and turned on after training. Leads to difference in printout results during training and the final results of our evaluation.")

parser.add_argument("--joint_vocab", default=False, action="store_true",
                    help="Use joint vocabulary for source and target sequences?")

parser.add_argument("--args", choices=["none", "bp_rnn"], default="none",
                    help="Set cmdline arguments to predefined values, such as the ones used in BP.")
parser.add_argument("--expid", default="none", type=str,
                    help="Experiment ID to be included in the results file.")

def check_and_fix_args(args: argparse.Namespace) -> None:
    """
    Modifies args in-place.
    Performs the following checks:
    (0) if `weighted_multilingual_accuracy_during_training` == True and `weighted_accuracy_during_training` == True:
        raise Runtime Error, that both cannot be set to True
    (1) if `model` == "transformer", but `joint_vocab` == False, raise warning and set `joint_vocab` to True
    (2) if `multilingual_training` == False, and:
        (a) `multilingual_corpus_down_up_sampling_temperature` != None: raise warning that it has no effect
        (b) `multilingual_corpus_down_up_sampling_weights` == "total": raise warning that it has no effect
    (3) if `multilingual_training` == True, and:
        (a) `multilingual_corpus_down_up_sampling_temperature` == None: raise warning that multilingual training is turned on, but no temperature for up/down-sampling is given
        (b) `weighted_multilingual_accuracy_during_training` == False: raise warning that it was turned off, and turn it on (set to True)
    (4) if `checkpoint_selection` == True, and:
        (a) `acc_for_ckpt_selection` == "w-accuracy" and `weighted_accuracy_during_training` == False:
            raise warning that for choosing checkpoint based on "w-accuracy", `weighted_accuracy_during_training` has to be set to True, and that we set it
            and set `weighted_accuracy_during_training` == True
        (b) `acc_for_ckpt_selection` == "multiling-acc", and:
            (i) `multilingual_training` == False: raise Runtime error, that we cannot choose checkpoint based on multiling accuracy, if multilingual training is turned off
            (ii) `weighted_multilingual_accuracy_during_training` == False:
                raise warning that for choosing checkpoint based on "w-accuracy", `weighted_multilingual_accuracy_during_training` has to be set to True, and that we set it
                and set `weighted_multilingual_accuracy_during_training` == True
        (c) `multilingual_training` == True and `acc_for_ckpt_selection` != "multiling-acc":
            raise warning that `acc_for_ckpt_selection` should be set to "multiling-acc" during multilingual training, and that we set it that way,
            and set `acc_for_ckpt_selection` = "multiling-acc"
    (5) if `weighted_multilingual_accuracy_during_training` == True and `weighted_accuracy_during_training` == True:
        raise Runtime Error, that both cannot be set to True (yes, do it again)
    """

    def warning(text: str) -> None:
        print(f"Warning: {text}")


    print("Arguments before validation:", vars(args))
    original_args = vars(args).copy()
    # TODO: print args before the check
    # TODO: check all args, and if necessary, adjust them and raise warnings/runtime errors
    # TODO: print args after the check, emphasizing, which has been changed
    # (0) Ensure mutually exclusive options are not both set
    if args.weighted_multilingual_accuracy_during_training and args.weighted_accuracy_during_training:
        raise RuntimeError("Both weighted_multilingual_accuracy_during_training and weighted_accuracy_during_training cannot be True.")

    # (1) Transformer model requires joint vocabulary
    if args.model == "transformer" and not args.joint_vocab:
        raise RuntimeError("Transformer model requires joint_vocab=True. Setting joint_vocab=True.")
        args.joint_vocab = True

    # (2) Multilingual corpus down/up sampling settings only apply to multilingual training
    if not args.multilingual_training:
        if args.multilingual_corpus_down_up_sampling_temperature is not None:
            warning("multilingual_corpus_down_up_sampling_temperature has no effect when multilingual_training=False.")
        if args.multilingual_corpus_down_up_sampling_weights == "total":
            warning("multilingual_corpus_down_up_sampling_weights has no effect when multilingual_training=False.")

    # (3) Adjust settings for multilingual training
    if args.multilingual_training:
        if args.multilingual_corpus_down_up_sampling_temperature is None:
            warning("multilingual_training is enabled, but no temperature for up/down-sampling is given.")
        if not args.weighted_multilingual_accuracy_during_training:
            warning("weighted_multilingual_accuracy_during_training is turned off, enabling it.")
            args.weighted_multilingual_accuracy_during_training = True

    # (4) Adjust checkpoint selection settings
    if args.checkpoint_selection:
        if args.acc_for_ckpt_selection == "w-accuracy" and not args.weighted_accuracy_during_training:
            raise RuntimeError("For choosing checkpoint based on 'w-accuracy', weighted_accuracy_during_training must be True. Enabling it.")
            args.weighted_accuracy_during_training = True
        if args.acc_for_ckpt_selection == "multiling-acc":
            if not args.multilingual_training:
                raise RuntimeError("Cannot choose checkpoint based on multiling-acc when multilingual_training is disabled.")
            if not args.weighted_multilingual_accuracy_during_training:
                raise RuntimeError("For choosing checkpoint based on 'multiling-acc', weighted_multilingual_accuracy_during_training must be True. Enabling it.")
                args.weighted_multilingual_accuracy_during_training = True
        if args.multilingual_training and args.acc_for_ckpt_selection != "multiling-acc":
            warning(
                "`acc_for_ckpt_selection` is NOT set to 'multiling-acc', although you are running in multilingual setting. Is it intended?")
            #args.acc_for_ckpt_selection = "multiling-acc"

    # (5) Double-check mutually exclusive options again
    if args.weighted_multilingual_accuracy_during_training and args.weighted_accuracy_during_training:
        raise RuntimeError("Both weighted_multilingual_accuracy_during_training and weighted_accuracy_during_training cannot be True.")

    modified_args = {k: v for k, v in vars(args).items() if original_args[k] != v}
    print("Arguments after validation:", vars(args))
    if modified_args:
        print("Changed arguments:", modified_args)



TOTAL_TRAINING_TIME = timedelta(0)
TOTAL_EVALUATION_TIME = timedelta(0)
LAST_TIME_CHECKPOINT = datetime.datetime.now()

def format_timedelta(elapsed_time) -> str:
    # Convert elapsed time to HH:MM:SS format
    formatted_time = str(elapsed_time).split(".")[0]
    return formatted_time

def get_current_timedelta_string(training: bool = True) -> str:
    """Resets the LAST_TIME_CHECKPOINT to current time, updates either TOTAL_EVAL_TIME or
    TOTAL_TRAINING_TIME (based on what is given in `training`) and returns a string representing the last timedelta."""

    global LAST_TIME_CHECKPOINT, TOTAL_EVALUATION_TIME, TOTAL_TRAINING_TIME

    end_time = datetime.datetime.now()
    elapsed_time = end_time - LAST_TIME_CHECKPOINT

    LAST_TIME_CHECKPOINT = end_time

    if training:
        TOTAL_TRAINING_TIME += elapsed_time
    else:
        TOTAL_EVALUATION_TIME += elapsed_time

    return format_timedelta(elapsed_time)


# dict {arg_name (str): bool (was explicitly set from cmdline?)}
EXPLICITLY_SET_ARGS = {
    action.dest: (
            f"--{action.dest}" in sys.argv or
            f"--{action.option_strings[0]}" in sys.argv or
            any(arg.startswith(f"--{action.dest}=") for arg in
                sys.argv) or  # such that it would also work with --arg1="value"
            any(arg.startswith(f"--{action.option_strings[0]}=") for arg in sys.argv)
    )
    for action in parser._actions
}


def _run_lemma_tag_implies_form_experiment():
    """Run the experiment to count the percentages of instances of type `lemma-tag-implies-form`.
    """
    from inflector_model import MostCommonFormBaseline
    from inflector_dataset import InflectionDataset

    def lemmas_tags_from_dataset_file(filename):
        """Extract lemmas and tags from a given dataset file."""
        lemmas = []
        tags = []

        with open(filename, "r") as file:
            for line in file:
                lemma, form, tag, count = line.strip().split("\t")
                lemmas.append(lemma)
                tags.append(tag)
        return lemmas, tags

    def write_list_to_file(filename: Path, items: Iterable) -> None:
        with open(filename, "w") as f:
            for item in items:
                f.write(str(item) + "\n")

    for evalset in ["train", "dev", "test"]:
        for corpus in args.corpus.split(args.corpus_name_separator):
            #process_files(42, corpora=corpus, langs="")
            full_file = Path(args.datadir) / corpus / f"{evalset}.tsv"

            most_common_bsln = MostCommonFormBaseline()

            most_common_bsln.train(train_filename=full_file)

            lemmas, tags = lemmas_tags_from_dataset_file(full_file)

            # print(lemmas[:10])

            pred_forms = most_common_bsln.predict_batch(lemmas, tags)

            pred_dir = Path("logs/lemma-tag-implies-form/preds/")

            os.makedirs(pred_dir, exist_ok=True)

            pred_filename = pred_dir / f"{corpus}_{evalset}.pred"
            write_list_to_file(filename=pred_filename, items=pred_forms)

            accs = dict()
            for uniform_acc in [False, True]:
                with open(pred_filename, "r") as pred, open(full_file, "r") as gold:
                    acc = InflectionDataset.evaluate_files(gold, pred, macro_evaluation=uniform_acc)
                    accs[uniform_acc] = acc

            # print( corpus dev/test wacc uacc )
            print(f"{corpus}\t{evalset}\t{(accs[False]):.4f}\t{accs[True]:.4f}")


def make_model_predict(model, dataset, char_vocab) -> list[str]:
    def decode_form(form_encoded, char_vocab) -> str:
        """Convert a (predicted) form from the np.Array of indices to a string"""
        return "".join(char_vocab.strings(form_encoded))

    predictions_encoded = model.predict(dataloader=dataset)
    predictions_decoded = [decode_form(form_encoded, char_vocab) for form_encoded in predictions_encoded]
    return predictions_decoded


# def run_training_and_eval_on_corpus(args: argparse.Namespace, corpus: str) -> dict:
#     """Runs the training and evaluation on a given corpus (given by name), and returns the weighted dev accuracy and uniform dev accuracy."""
#


def get_dataloader_from_dataset(dataset: MorphoDataset.Dataset, args: argparse.Namespace, src_vocab: Vocabulary,
                                tgt_vocab: Vocabulary, training: bool) -> torch.utils.data.DataLoader:
    def prepare_tagging_data(example):
        """Convert the dataset dict to tuple of what we want"""
        # lemma and tag is input, form is output, count is weight
        if args.multilingual_training:
            return example["lemma"], example["tag_separated"], example["form"], example["count"], example["lang_id"], \
                example["corpus_len_coef"]
        else:
            return example["lemma"], example["tag_separated"], example["form"], example["count"]

    def prepare_batch(data, training: bool):
        """Map the characters to their indices using the `morpho.train.__.char_vocab` vocabulary.
        Then create a tensor by padding the forms to the length of the longest one in the batch."""

        if args.multilingual_training:
            lemmas, tags, forms, counts, lang_ids, corpus_len_coefs = zip(*data)

            # INPUTS: l e m m a lang_id t1 t2 t3 (all encoded as indices)
            inputs = [
                # src_vocab.indices([MorphoDataset.BOW]) +
                src_vocab.indices(lemma) +
                # add separator index, if required
                ([MorphoDataset.SEP] if args.add_separator_token else []) +
                # src_vocab.indices([MorphoDataset.SEP]) +
                src_vocab.indices(lang_id) +
                src_vocab.indices(tag)  # +
                # src_vocab.indices([MorphoDataset.EOW])
                for (lemma, tag, lang_id)
                in zip(lemmas, tags, lang_ids)
            ]
        else:
            # if not multilingual training, no language ID token is needed
            lemmas, tags, forms, counts = zip(*data)

            # INPUTS: l e m m a t1 t2 t3 (all encoded as indices)
            inputs = [
                # src_vocab.indices([MorphoDataset.BOW]) +
                src_vocab.indices(lemma) +
                # add separator index, if required
                ([MorphoDataset.SEP] if args.add_separator_token else []) +
                # src_vocab.indices([MorphoDataset.SEP]) +
                src_vocab.indices(tag)  # +
                # src_vocab.indices([MorphoDataset.EOW])
                for (lemma, tag)
                in zip(lemmas, tags)
            ]

        # INPUTS: l e m m a t1 t2 t3 PAD PAD PAD (padded to the same length in the batch)
        inputs = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(input_sequence) for input_sequence in inputs],
                                                 batch_first=True,
                                                 padding_value=MorphoDataset.PAD)

        # TARGETS: f o r m EOW (encoded as indices)
        forms = [
            # [MorphoDataset.BOW] +
            tgt_vocab.indices(form) +
            [MorphoDataset.EOW]
            for form in forms
        ]

        # TARGETS: f o r m EOW PAD PAD PAD (padded to the same length in the batch)
        forms = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(form) for form in forms],
                                                batch_first=True,
                                                padding_value=MorphoDataset.PAD)

        # In the training regime, we pass the gold `forms` also as inputs.

        # return inputs, targets                (if not training)
        # return (inputs, targets), targets     (if training)

        # if some weights should be used during training to compute weighted accuracy (either macro-avg over langs in
        # multilingual setting (a) or micro-avg-accuracy in monolinugual setting (b))
        if args.multilingual_training and args.weighted_multilingual_accuracy_during_training:  # (A)
            # if desired (set in args), use the corpus_len_coefs here as weights for evaluation
            weights = torch.as_tensor(corpus_len_coefs)
            return ((inputs, forms) if training else inputs), forms, weights

        elif args.weighted_accuracy_during_training:  # (B)
            # optionally adjust weights or use sth else as weights for weighted evaluation during training:
            # either the corpus counts (for dev evaluation in micro-avg/weighted accuracy),
            # or somehow adjusted corpus len coefficients
            weights = torch.as_tensor(counts)
            return ((inputs, forms) if training else inputs), forms, weights
        else:
            return ((inputs, forms) if training else inputs), forms

    def prepare_training_batch(data):
        return prepare_batch(data, training=True)

    def prepare_dev_batch(data):
        return prepare_batch(data, training=False)

    transformed_dataset = dataset.transform(prepare_tagging_data)

    if training:
        # === WEIGHTED SAMPLING ACCORDING TO CORPUS COUNTS, WEIGHTED BY THE TEMPERATURE ===
        # weighted sampling according to counts (given from the dataset) with application of temperature
        # example: https://github.com/ufal/nametag3/blob/main/nametag3_dataset_collection.py
        # but we do it in an opposite way (our temperature corresponds to `1-temperature` in nametag)

        weights = None

        if args.multilingual_training and args.multilingual_corpus_down_up_sampling_temperature is not None:

            # multilingual_training multilingual_corpus_down_up_sampling_temperature multilingual_corpus_down_up_sampling_weights

            if args.multilingual_corpus_down_up_sampling_weights == "unique":
                corpus_lens = dataset.corpus_lens_unique
            elif args.multilingual_corpus_down_up_sampling_weights == "total":
                corpus_lens = dataset.corpus_lens_total
            else:
                raise RuntimeError(
                    f"Invalid option for args.multilingual_corpus_down_up_sampling_weights: must be `total` or `unique`, is: {args.multilingual_corpus_down_up_sampling_weights}")

            # first, compute the reciprocal of the corpus lengths
            multilingual_weights_orig = [1 / w for w in corpus_lens]

            #multilingual_weights_orig //= sum(multilingual_weights_orig)

            # adjust the weights by the given temperature
            multilingual_weights = [w ** args.multilingual_corpus_down_up_sampling_temperature for w in
                                    multilingual_weights_orig]

            # normalize the weights, such that the largest would be equal to 1.0
            max_item = max(multilingual_weights)
            multilingual_weights = [w / max_item for w in multilingual_weights]

            weights = multilingual_weights

            print("INFO: Multilingual down/up-sampling of corpora will be performed.", file=sys.stderr)
            print(
                f"INFO: {args.multilingual_corpus_down_up_sampling_weights} corpus lengths will be used to compute the 1/corpus_len coefficient.",
                file=sys.stderr)
            print(
                f"INFO: The coefficient will be adjusted by given temperature: {args.multilingual_corpus_down_up_sampling_temperature}",
                file=sys.stderr)
            print(f"INFO: Original 1/corpus_len weights: {get_shortened_list_string(multilingual_weights_orig)}",
                  file=sys.stderr)
            print(
                f"INFO: Adjusted weights (normalized, such that the largest one == 1): {get_shortened_list_string(multilingual_weights)}",
                file=sys.stderr)
            # input()

        if args.frequency_weighted_training_temperature is not None:
            # if it is 0.0, it should (almost) be the same as if it is None - 0 temperature means uniform weights
            # however, with `args.frequency_weighted_training_temperature=None` we are sure that every items will be seen exactly once during one epoch.
            # with 0 temperature we only know that every item has the same probability to appear in the epoch

            training_counts = dataset.counts
            training_weights = [count ** args.frequency_weighted_training_temperature for count in training_counts]

            print("INFO: Weighting of items with higher corpus frequency will be performed during training.",
                  file=sys.stderr)
            print(f"Original counts: {training_counts[:100]}", file=sys.stderr)
            tmp = [f"{w:.2f}" for w in training_weights[:100]]
            print(
                f"Counts adjusted by temperature ({args.frequency_weighted_training_temperature}): {tmp}",
                file=sys.stderr)
            # input()

            if weights is None:
                weights = training_weights
                print("INFO: ONLY the corpus freqs will be used to weights training items.", file=sys.stderr)
                # input()
            else:
                weights = [multi_w * train_w for (multi_w, train_w) in zip(weights, training_weights)]
                print("INFO: BOTH the corpus freqs and corpus_len coeffs will be used to weights training items.",
                      file=sys.stderr)
                print("INFO: Final weights of training examples:", file=sys.stderr)
                tmp = [f'{w:.4f}' for w in weights[:100]]
                print(f"INFO: {tmp}")
                # input()

        if weights is not None:
            print(f"Created weighted random sampler, that will, in each epoch, generate {len(dataset)} samples",
                  file=sys.stderr)
            random_sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(dataset), replacement=True,
            )
            # when sampler is provided, do not specify `shuffle`
            return torch.utils.data.DataLoader(transformed_dataset, args.batch_size, collate_fn=prepare_training_batch,
                                               sampler=random_sampler)
        else:
            return torch.utils.data.DataLoader(transformed_dataset, args.batch_size, collate_fn=prepare_training_batch,
                                               shuffle=True)
    # === ===
    else:
        return torch.utils.data.DataLoader(transformed_dataset, args.batch_size, collate_fn=prepare_dev_batch)

    # dev = torch.utils.data.DataLoader(dev, args.batch_size, collate_fn=prepare_dev_batch)
    # train_for_eval = torch.utils.data.DataLoader(train_for_eval, args.batch_size, collate_fn=prepare_dev_batch)
    # test = torch.utils.data.DataLoader(test, args.batch_size, collate_fn=prepare_dev_batch)


def train_on_dataset(args: argparse.Namespace, morpho: MorphoDataset, corpus: str = None):
    """
    Train a model (with given params) on a given morpho dataset (training on its train set, dev evaluation during training on its dev set)
    """

    # if corpus name is not given (we are training in multilingual setting, not on a particular corpus, the logdir is just the original logdir, without the corpus name
    logdir = Path(args.logdir) / corpus if corpus else Path(args.logdir)

    os.makedirs(logdir, exist_ok=True)

    # Save the source-target vocabulary for encoding/decoding the inputs and outputs
    if args.joint_vocab:
        src_vocab = tgt_vocab = morpho.train.lemma_tag_form_joint.char_vocab
        print(f"Joint char vocab size for src and tgt = {len(src_vocab)}")
    else:
        src_vocab = morpho.train.lemma_tag_joint.char_vocab
        tgt_vocab = morpho.train.forms.char_vocab
        print(f"Char vocab size for lemma-tags={len(src_vocab)}")
        print(f"Char vocab size for forms={len(tgt_vocab)}")

    print(f"Src vocab:")
    print(f"{src_vocab._strings}")
    print()
    print("Tgt vocab: ")
    print(f"{tgt_vocab._strings}")
    print()
    # input()

    if args.model.startswith("rnn"):
        model = RNNEncoderDecoderWithAttention(args=args, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    elif args.model.startswith("transformer"):
        if not args.joint_vocab:
            raise RuntimeError(
                "Transformer model only supports joint src-tgt vocabulary. Please, call with `--joint_vocab`.")
        model = BartWrapper(args, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    elif args.model.startswith("copy-bsln"):
        if not args.joint_vocab:
            raise RuntimeError(
                "Copy baseline model only supports joint src-tgt vocabulary. Please, call with `--joint_vocab`.")

        if not args.add_separator_token:
            raise RuntimeError(
                "Copy baseline model only supports using separator token. Please, call with `--add_separator_token`.")
        model = CopyBaseline()
    else:
        raise RuntimeError(f"Invalid model specification: {args.model}")

    def count_params(model, is_human: bool = False):
        # from https://michaelwornow.net/2024/01/18/counting-params-in-transformer
        params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f"{params / 1e6:.2f}M" if is_human else params

    print(model)
    print("Total # of params:", count_params(model, is_human=True))

    train = get_dataloader_from_dataset(morpho.train, args, src_vocab, tgt_vocab, training=True)
    dev = get_dataloader_from_dataset(morpho.dev, args, src_vocab, tgt_vocab, training=False)

    # from torchsummary import summary
    #
    # data_iter = iter(train_for_eval)
    # inputs, labels = next(data_iter)
    #
    # # Automatically obtain the input shape (excluding the batch size)
    # input_shape = inputs.shape[1:]
    #
    # summary(model, input_shape)
    #
    # exit(0)

    # TODO: Warmup scheduler:
    # A warmup strategy is commonly used where the learning rate is initially small and gradually increases
    # to the desired value over a set number of steps or epochs.

    # Let's use a `LinearLR` scheduler from PyTorch with a warmup period.
    # We will also use the learning rate scheduler along with the optimizer.

    # from torch.optim.lr_scheduler import LambdaLR
    #
    # # Number of warmup steps, this could be a fraction of the total number of steps (e.g., 0.1 of training steps)
    # warmup_steps = 1000  # This is just an example, you may adjust it based on your training data
    #
    # # Lambda function for linear warmup
    # def warmup_lr_lambda(current_step: int):
    #     if current_step < warmup_steps:
    #         return float(current_step) / float(max(1, warmup_steps))  # Linear increase in LR
    #     return 1.0  # After warmup, keep the LR constant at the base value
    #
    # # Add the warmup scheduler to the model configuration
    # scheduler = LambdaLR(
    #     optimizer=model.optimizer,  # Use the optimizer from the model configuration
    #     lr_lambda=warmup_lr_lambda,  # Apply the warmup function
    # )

    # === L2 regularization ===
    if args.weight_decay:
        opt = torch.optim.AdamW(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,  # This is the weight decay (L2 regularization). Adjust as needed.
        )
    else:
        opt = torch.optim.Adam(
            params=model.parameters(),
            lr=args.lr
        )
    # === ===

    # === LR DECAY SETUP ===
    number_of_train_steps_to_perform = args.epochs * len(train)
    print(f"Expected number of train steps: {args.epochs} * {len(train)} = {number_of_train_steps_to_perform}",
          file=sys.stderr)

    # startovni epocha/startovni pocet steps
    if args.lr_decay == "none":
        scheduler = None
    elif args.lr_decay == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=opt,
            start_factor=1,  # go linearly from the original learning rate to 0
            end_factor=0,
            total_iters=number_of_train_steps_to_perform
        )

    elif args.lr_decay == "cosine":
        # start on the initial learning and decrease to 0 in a cosine manner (LR reaches 0 after the last train step)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt,
            T_max=number_of_train_steps_to_perform
        )
    else:
        scheduler = None
    # === ===

    metrics = {"accuracy": torchmetrics.MeanMetric()}
    if args.weighted_accuracy_during_training:
        metrics["w-accuracy"] = torchmetrics.MeanMetric()
    if args.weighted_multilingual_accuracy_during_training:
        metrics["multiling-acc"] = torchmetrics.MeanMetric()

    model.configure(
        # TODO: with warmup steps
        optimizer=opt,
        schedule=scheduler,
        # Pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation.
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        metrics=metrics,
        logdir=logdir,
        clip_grad_norm=args.clip_grad_norm,
        checkpoint_selection=args.checkpoint_selection,
        accuracy_for_ckpt_selection=args.acc_for_ckpt_selection,

    )

    # Print dataset sizes
    print(f"Training set size: {len(train.dataset)}", file=sys.stderr)
    print(f"Development set size: {len(dev.dataset)}", file=sys.stderr)

    print(
        f"Info: When printing the predictions during training, '·' char is used in the source sequence in the printouts, only to show the borders of the tokens.",
        file=sys.stderr)

    logs = model.fit(train, dev=dev, epochs=args.epochs, verbose=2 if args.show_results_every_batch else 1)

    # save to args the variables that were created here, but are used in `eval_on_dataset()`
    args._src_vocab = src_vocab
    args._tgt_vocab = tgt_vocab
    args._logdir = logdir

    return model


def eval_on_dataset(args: argparse.Namespace, corpus: str, morpho: MorphoDataset, model: BaseInflectorModel):
    """
    Evaluate the given model on a given morpho dataset (and given corpus): in monolingual setting, just evaluate on the train and dev (and test?) set of the morpho dataset,
    in multilingual setting, evaluate on a dev set given by the corpus name.
    """

    src_vocab = args._src_vocab
    tgt_vocab = args._tgt_vocab
    logdir = args._logdir

    if args.multilingual_training:
        # extract the monolingual dev set from the multilingual dataset
        dev_dataset = morpho.monolingual_dev_sets[corpus]
        test_dataset = morpho.monolingual_test_sets[corpus]
    else:
        dev_dataset = morpho.dev
        test_dataset = morpho.test

    print(f"Evaluating on a monolingual dataset for {corpus}:")
    print(f"Dev len={len(dev_dataset)}")
    print(f"Test len={len(test_dataset)}")
    print(f"Train len={len(morpho.train)}")

    dev = get_dataloader_from_dataset(dev_dataset, args, src_vocab, tgt_vocab, training=False)
    train_for_eval = get_dataloader_from_dataset(morpho.train, args, src_vocab, tgt_vocab, training=False)
    test = get_dataloader_from_dataset(test_dataset, args, src_vocab, tgt_vocab, training=False)

    dev_predicted_forms = make_model_predict(model, dev, tgt_vocab)
    train_predicted_forms = make_model_predict(model, train_for_eval, tgt_vocab)
    test_predicted_forms = make_model_predict(model, test, tgt_vocab)

    j = args.print_first_preds_on_dev
    print(f"First {j} predicted forms on {corpus}-dev, compared to gold forms:", file=sys.stderr)
    for (predicted, gold) in zip(dev_predicted_forms[:j], dev_dataset.forms.strings[:j]):
        print(f"P: {predicted:<20}; G: {gold}", file=sys.stderr)

    filename = Path(logdir) / f"{corpus}.dev.pred"

    def write_predictions_to_file(filename, predictions):
        with open(filename, "w") as f:
            for form in predictions:
                f.write(f"{form}\n")
            print(f"Predictions successfully written to file {filename}.", file=sys.stderr)

    if args.predictions_dir:
        # Used for printing sigmorphon predictions
        os.makedirs(args.predictions_dir, exist_ok=True)
        write_predictions_to_file(Path(args.predictions_dir) / f"{corpus}.dev", dev_predicted_forms)
        write_predictions_to_file(Path(args.predictions_dir) / f"{corpus}.test", test_predicted_forms)

    write_predictions_to_file(filename, dev_predicted_forms)

    # weighted_dev_acc = InflectionDataset.evaluate_predictions(
    #     gold_forms=morpho.dev.forms.strings,
    #     pred_forms=dev_predicted_forms,
    #     weights=morpho.dev.counts
    # )
    #
    # uniform_dev_acc = InflectionDataset.evaluate_predictions(
    #     gold_forms=morpho.dev.forms.strings,
    #     pred_forms=dev_predicted_forms,
    #     weights=None
    # )
    #
    # uniform_train_acc = InflectionDataset.evaluate_predictions(
    #     gold_forms=morpho.train.forms.strings,
    #     pred_forms=train_predicted_forms,
    #     weights=None
    # )

    # TODO: adjust: split to branches for multilinguality

    accuracies = {
        "dev": InflectionDataset.full_evaluation_on_dataset(eval_predictions=dev_predicted_forms,
                                                            gold_dataset=dev_dataset, train_dataset=morpho.train),
        "train": InflectionDataset.full_evaluation_on_dataset(eval_predictions=train_predicted_forms,
                                                              gold_dataset=morpho.train, train_dataset=morpho.train)
    }

    if args.eval_also_on_test:
        accuracies["test"] = InflectionDataset.full_evaluation_on_dataset(
            eval_predictions=test_predicted_forms,
            gold_dataset=test_dataset, train_dataset=morpho.train
        )

    # wda = accuracies["dev"]["all"]["weighted"]
    # uda = accuracies["dev"]["all"]["uniform"]
    # uta = accuracies["train"]["all"]["uniform"]
    # print(f"Weighted dev accuracy on corpus {corpus} is {weighted_dev_acc:.4f} OR {wda:.4f}", file=sys.stderr)
    # print(f"Uniform dev accuracy on corpus {corpus} is {uniform_dev_acc:.4f} OR {uda:.4f}", file=sys.stderr)
    # print(f"Uniform train accuracy on corpus {corpus} is {uniform_train_acc:.4f} OR {uta:.4f}", file=sys.stderr)

    print(f"Weighted dev accuracy on corpus {corpus} is {accuracies['dev']['all']['weighted']:.4f}", file=sys.stderr)
    print(f"Uniform dev accuracy on corpus {corpus} is {accuracies['dev']['all']['uniform']:.4f}", file=sys.stderr)
    print(f"Uniform train accuracy on corpus {corpus} is {accuracies['train']['all']['uniform']:.4f}", file=sys.stderr)

    return accuracies


def average_dict(dict_list):
    "Compute an average of a dictionary list"
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

EPOCH_SELECTED_BY_CKPT_SELECTION = dict()

def run_training_and_eval_on_corpora(args: argparse.Namespace, corpora: list[str]) -> dict[str, dict]:
    print(f"Going to perform training and evaluation on the corpora {corpora}...", file=sys.stderr)

    global EPOCH_SELECTED_BY_CKPT_SELECTION

    accuracies = dict()

    if args.multilingual_training:
        # Load the multilingual dataset
        morpho = MorphoDataset(corpus_name=None, max_sentences=args.max_sentences, multilingual=True,
                               corpus_names=corpora)

        # train the model
        model = train_on_dataset(args, morpho, corpus=None)

        timedelta = get_current_timedelta_string(training=True)
        print(f"# TIME: Training the multilingual model took {timedelta}")

        for corpus in corpora:
            # evaluate the model
            accuracies[corpus] = eval_on_dataset(args=args, corpus=corpus, morpho=morpho, model=model)
            EPOCH_SELECTED_BY_CKPT_SELECTION[corpus] = model.checkpoint_epoch

        timedelta = get_current_timedelta_string(training=False)
        print(f"# TIME: Evaluating the multilingual model took {timedelta}")

    else:

        for corpus in corpora:
            print(f"Going to perform monolingual training and evaluation on the corpus {corpus}...", file=sys.stderr)

            # Load the data
            morpho = MorphoDataset(corpus, max_sentences=args.max_sentences)
            # train the model
            model = train_on_dataset(args=args, morpho=morpho, corpus=corpus)

            EPOCH_SELECTED_BY_CKPT_SELECTION[corpus] = model.checkpoint_epoch

            timedelta = get_current_timedelta_string(training=True)
            print(f"# TIME: Training the monolingual model on {corpus} took {timedelta}")

            # evaluate the model
            accuracies[corpus] = eval_on_dataset(args=args, corpus=corpus, morpho=morpho, model=model)

            timedelta = get_current_timedelta_string(training=False)
            print(f"# TIME: Evaluating the monolingual model on {corpus} took {timedelta}")

            # accuracies[corpus] = run_training_and_eval_on_corpus(args, corpus=corpus)

    accuracies["macro-avg"] = average_dict(list(accuracies.values()))

    print(f"# TIME: TOTAL TRAINING TIME: {format_timedelta(TOTAL_TRAINING_TIME)}")
    print(f"# TIME: TOTAL EVALUATION TIME: {format_timedelta(TOTAL_EVALUATION_TIME)}")

    return accuracies


def run_integration_test(args: argparse.Namespace) -> None:
    """Ignore arguments, set them to the predefined values, and run integration test, prepend results to the integration
    tests file.
    """
    print(f"Running integration test: {args.integration}", file=sys.stderr)

    # def extract_first_entry(file_path):
    #     # Initialize dictionary to store the result
    #     corpus_dict = {}
    #
    #     # If the file does not exist (no integration test has yet been run), return dict with 0 accuracies.
    #     if not os.path.exists(file_path):
    #         for corpus in corpora_for_experiments:
    #             corpus_dict[corpus] = {
    #                 "weighted": 0.0,
    #                 "uniform": 0.0
    #             }
    #         return corpus_dict
    #
    #     with open(file_path, 'r') as file:
    #         lines = file.readlines()
    #
    #     # Find the first entry (corpus name and weighted/uniform values)
    #     for line in lines:
    #         # Skip lines that do not contain relevant data
    #         if not line.strip() or line.startswith("Date") or line.startswith("Start Time") or line.startswith(
    #                 "End Time") or line.startswith("Integration test"):
    #             continue
    #
    #         # Extract only first entry
    #         if "=============" in line:
    #             break
    #
    #         # Split by tabs and process each corpus and its values
    #         parts = [part for part in line.split() if part]
    #
    #         corpus_name = parts[0]
    #
    #         # strip the last character ":" from the second part
    #         acc_type = parts[1][:-1]
    #         acc_value = float(parts[2].strip())
    #
    #         if corpus_name in corpus_dict:
    #             corpus_dict[corpus_name][acc_type] = acc_value
    #         else:
    #             corpus_dict[corpus_name] = {
    #                 acc_type: acc_value
    #             }
    #
    #     return corpus_dict

    # def compare_floats(old_acc: float, new_acc: float) -> bool:
    #     """If the new accuracy is considerably worse than the old accuracy, return False, else return True"""
    #     epsilon = 0.01  # 1% accuracy
    #     if new_acc + epsilon < old_acc:
    #         return False
    #     return True

    # def prepend_to_file(file_path, new_content):
    #     """Prepend content to a given file. Used to prepend the integration test results, such that the newest results would
    #     always be at the beginning of the file."""
    #
    #     # Check if the file exists
    #     if os.path.exists(file_path):
    #         # Read the current content of the file
    #         with open(file_path, 'r') as file:
    #             current_content = file.read()
    #     else:
    #         # If the file does not exist, start with an empty string
    #         current_content = ''
    #
    #     # Prepend the new content
    #     new_content = new_content + current_content
    #
    #     # Write the new content back to the file
    #     with open(file_path, 'w') as file:
    #         file.write(new_content)

    def get_datetime_string() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    def get_date_string() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d")

    # corpora_for_experiments = ["UD_Breton-KEB", "UD_Spanish-AnCora", "UD_Czech-PDT", "UD_English-EWT", "UD_Basque-BDT"]

    # previous_results = extract_first_entry(args.integration_results_file)

    string_result = f"""Integration test: {args.integration}
Date: {get_date_string()}
Start Time: {get_datetime_string()}
"""

    # set_args_to_predefined_values(args, "integration_test", corpora=corpora_for_experiments)

    accuracies = main(args)

    string_result += f"End Time: {get_datetime_string()}\n"

    for corpus, accs in accuracies.items():
        weighted_acc, uniform_acc = accs
        string_result += f"{corpus:<20}# weighted # {weighted_acc:.6f}\n"
        # {'GOOD' if compare_floats(old_acc=previous_results[corpus]['weighted'], new_acc=weighted_acc) else 'WORSE'}\n"
        string_result += f"{corpus:<20}# uniform  # {uniform_acc:.6f}\n"
        # {'GOOD' if compare_floats(old_acc=previous_results[corpus]['uniform'], new_acc=uniform_acc) else 'WORSE'}\n"

    import json
    string_result += "\nPARAMETERS ====================\n"
    string_result += json.dumps(vars(args), indent=4)
    string_result += "\n===================================================================================\n\n"

    # prepend_to_file(args.integration_results_file, new_content=string_result)
    with open(args.integration_results_file, "w") as f:
        f.write(string_result)

    print(f"Integration test {args.integration} finished.", file=sys.stderr)


def set_args_to_predefined_values(args: argparse.Namespace, configuration: str, corpora: list = None) -> None:
    """Modifies the args namespace in place. Sets arguments to predefined values (only those that were not explicitly set from commandline)."""

    # if configuration == "integration_test":
    # # FOR INTEGRATION TEST
    # argument_values = {
    #     'batch_size': 256,
    #     'rnn_embedding_dim': 200,
    #     'epochs': 30,  # 2
    #     'max_sentences': None,
    #     'rnn_layer_dim': 200,
    #     'seed': 42,
    #     'show_results_every_batch': 100,
    #     'rnn_tie_embeddings': True,
    #     'rnn_layer_count': 2,
    #     'rnn_drop': 0.3,
    #     'rnn_unidirectional': False,
    #     'threads': 1,
    #     'corpus': ';'.join(corpora),
    #     'corpus_name_separator': ';',
    # }
    if configuration == "bp_rnn":
        """- LSTM
            - 13 epochs
        
            - Adam default values of βs learning rate 0.001
            - warm-up 4k steps
            - batch 256
            - 2 layers
            - size 200
            - shared embedding
            - emb dimension 128
            - bi-directional encoder
            - Luong attention
            - dropout=0.3
            - attention dropout = 0.1
            """
        argument_values = {
            # - LSTM
            # TODO: implement LSTM
            # - 13 epochs
            "epochs": 13,

            # - Adam default values of βs learning rate 0.001
            # Is the only supported option right now

            # - warm-up 4k steps
            # TODO implement warpup steps
            # - batch 256
            "batch_size": 256,
            # - 2 layers
            "rnn_layer_count": 2,
            # - size 200
            "rnn_layer_dim": 200,

            # - shared embedding
            "rnn_tie_embeddings": True,

            # - emb dimension 128
            # "cle_dim": 128,
            # "cle_dim" : 200, # TODO: fix the bug that we cannot have different cle dim than rnn dim if we have tie embeddings=True
            # Tenhle bug je konceptuální. Vzhledem k tomu, jak je to napsané, při tied embeddings se prostě očekává, že rnn zachovává dimenzi,
            # a tedy že jeho hidden states budou stejně velké jako input embeddings. Pokud bychom chtěli dovolit cle_dim jinačí,
            # budeme muset implementovat a nějak to pořešit. Viz https://piazza.com/class/lsun0f9xzja243/post/684

            # - bi-directional encoder
            "rnn_unidirectional": False,

            # - Luong attention
            # TODO implement support Luong

            # - dropout=0.3
            "rnn_drop": 0.3,

            # TODO implement attention dropout
        }
        argument_values["rnn_embedding_dim"] = argument_values["rnn_layer_dim"]
    # elif configuration == "bp_rnn_improved":
    #     argument_values = {
    #         "epochs": 13, # 13
    #         "batch_size": 256, # 256
    #         "num_layers": 2, # 2
    #         "rnn_layer_dim": 200, # 200
    #         "rnn_tie_embeddings": True, # True
    #         "rnn_unidirectional": False, # False
    #         "dropout": 0.3, # 0.3
    #     }
    #     argument_values["cle_dim"] = argument_values["rnn_layer_dim"]
    else:
        raise RuntimeError(f"Invalid configuration option for setting cmdline args: {configuration}")

    # Set the predefined attributes iff they are not set by the user from cmdline (i.e., if they have the default value)
    for argument_name, value in argument_values.items():
        if not EXPLICITLY_SET_ARGS[argument_name]:
            setattr(args, argument_name, value)


def set_datadir_based_on_data(args: argparse.Namespace) -> None:
    """Modify args in place.
    Sets datadir according to the argument `data`.
    """
    datadir_dict = {
        "ud": "data/processed/ud-inflection/ud-treebanks-v2.14",
        "sig22": "data/processed/2022_SIGMORPHON_Inflection_ST",
        "sig23": "data/processed/2023_SIGMORPHON_Inflection_ST"
    }

    if args.datadir:
        # if the datadir is explicitly set, do not reset it
        pass
    else:
        if args.data in datadir_dict:
            args.datadir = datadir_dict[args.data]
        else:
            raise RuntimeError(
                "If the argument `data` is set to 'none', `datadir` argument has to be set explicitly.")


def main(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    """Runs the experiments on given corpora, and returns the list of (weighted,uniform) accuracies."""

    # UD.RAW_DIR = Path(args.ud_raw_dir)
    UD.INFLECTION_DIR = Path(args.datadir)
    # UD.TGZ_PATH = Path(args.ud_tgz_path)
    # UD.URL = args.ud_url

    # TODO : set seed at all places needed
    # keras.utils.set_random_seed(args.seed)
    # keras.config.disable_traceback_filtering()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    logname = create_log_name(args)

    args.logdir = "{}/{}".format(args.logdir, logname)
    args.result_filename = "{}/{}.res".format(args.result_dir, logname)
    args.full_result_filename1 = "{}/{}.full-res1.tsv".format(args.result_dir, logname)
    args.full_result_filename2 = "{}/{}.full-res2.tsv".format(args.result_dir, logname)

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    def touch_file(path):
        with open(path, 'a'):
            os.utime(path, times=None)  # Updates access and modification times to the current time

    # Create the log and res files before running the experiments: quick fix of the problem "filename too long": at least it will fail here at the beginning
    for filepath in [args.result_filename, args.full_result_filename1, args.full_result_filename2]:
        touch_file(filepath)

    # copy all src files to args.logdir/src
    copy_files_and_dirs(dest_dir=f"{args.logdir}/src", suffixes=['.py', 'configs', '.sh'])

    corpora = args.corpus.split(args.corpus_name_separator)

    full_accuracies = run_training_and_eval_on_corpora(args, corpora)


    with open(args.full_result_filename1, "w") as f:
        eval_criteria_count = 11 # 5
        accuracy_values_reported_per_criterion = 5

        splits_count = 3 if args.eval_also_on_test else 2

        # full results 1, with one line of actual values, the rest being header
        for lang in full_accuracies:
            f.write(lang + ''.join(['\t'] * splits_count * eval_criteria_count * accuracy_values_reported_per_criterion))
        f.write("\n")
        for lang in full_accuracies:
            for dataset in full_accuracies[lang]:
                f.write(dataset + ''.join(['\t'] * eval_criteria_count * accuracy_values_reported_per_criterion))
        f.write("\n")
        for lang in full_accuracies:
            for dataset in full_accuracies[lang]:
                for eval_type in full_accuracies[lang][dataset]:
                    f.write(eval_type + ''.join(['\t'] * accuracy_values_reported_per_criterion))
        f.write("\n")
        for lang in full_accuracies:
            for dataset in full_accuracies[lang]:
                for eval_type in full_accuracies[lang][dataset]:
                    for eval_value in full_accuracies[lang][dataset][eval_type]:
                        f.write(eval_value + '\t')
        f.write("\n")
        for lang in full_accuracies:
            for dataset in full_accuracies[lang]:
                for eval_type in full_accuracies[lang][dataset]:
                    for eval_value in full_accuracies[lang][dataset][eval_type]:
                        actual_value = full_accuracies[lang][dataset][eval_type][eval_value]
                        if eval_value == "count":
                            actual_value = f"{actual_value}"
                        else:
                            actual_value = f"{actual_value:.4f}"
                        f.write(actual_value + '\t')
        f.write("\n")

    with open(args.full_result_filename2, "w") as f:
        # second version of full results, with a full table
        f.write(InflectionDataset.pretty_print_of_full_results(full_accuracies))

    def print_simple_results_to_file(simple_accuracies, filename):
        """Prints simple accuracies to given file."""
        with open(filename, "w") as f:
            # basic results, for looking into it regularly. first, one line per corpus, then, if we there is EXPID
            # specified, also macro-avg over CZ, EN and ESP, followed by all results in one line in a pre-specified order
            f.write(f"datetime\targs\tcorpus\tckpt-selected-epoch\tw-acc\tu-acc\n")

            for corpus, accs in simple_accuracies.items():
                print(
                    f"{args.datetime_string}\t{args.args_string}\t{corpus}\t{EPOCH_SELECTED_BY_CKPT_SELECTION.get(corpus, None)}\t{accs[0]:.4f}\t{accs[1]:.4f}")
                f.write(
                    f"{args.datetime_string}\t{args.args_string}\t{corpus}\t{EPOCH_SELECTED_BY_CKPT_SELECTION.get(corpus, None)}\t{accs[0]:.4f}\t{accs[1]:.4f}\n")

            relevant_langs = ["UD_Spanish-AnCora", "UD_Czech-PDT", "UD_English-EWT"]
            if args.expid != "none" and all(corp in args.corpus for corp in relevant_langs):
                # Code for experiments, where we want to print out also the averages, and the corpora results in specific order

                relevant_accs = [simple_accuracies[lang] for lang in relevant_langs]
                macro_avg_weighted_acc = sum(relevant_accs[i][0] for i in range(len(relevant_accs))) / len(
                    relevant_accs)
                macro_avg_uniform_acc = sum(relevant_accs[i][1] for i in range(len(relevant_accs))) / len(relevant_accs)

                simple_accuracies["MACRO-AVG(CZ,ESP,EN)"] = (macro_avg_weighted_acc, macro_avg_uniform_acc)
                for corpus in ["MACRO-AVG(CZ,ESP,EN)"]:
                    accs = simple_accuracies[corpus]
                    print(f"{args.datetime_string}\t{args.args_string}\t{corpus}\t{accs[0]:.4f}\t{accs[1]:.4f}")
                    f.write(f"{args.datetime_string}\t{args.args_string}\t{corpus}\t{accs[0]:.4f}\t{accs[1]:.4f}\n")

                corpora = ["MACRO-AVG(CZ,ESP,EN)", "UD_Czech-PDT", "UD_English-EWT", "UD_Spanish-AnCora",
                           "UD_Basque-BDT", "UD_Breton-KEB"]

                # add the corpora from args that are not yet present in the list of corpora to be printed
                additional_corpora = list(set(args.corpus.split(args.corpus_name_separator)) - set(corpora))

                # sort, for in all runs it would be in the same order
                additional_corpora.sort()
                corpora += additional_corpora

                f.write(f"datetime\targs\tALL\t")
                for corpus in corpora:
                    f.write(f"{corpus}-w\t{corpus}-u\t")
                f.write("\n")

                f.write(f"{args.datetime_string}\t{args.args_string}\tALL\t")
                for corpus in corpora:
                    accs = simple_accuracies[corpus]
                    f.write(f"{accs[0]:.4f}\t{accs[1]:.4f}\t")
                f.write("\n")
                
                
            # write w-acc u-acc
            f.write(f"{args.datetime_string}\t{args.args_string}\tFIRST-W-THEN-U\t")
            for corpus, accs in sorted(simple_accuracies.items()):
                f.write("w-acc\t")
            f.write("\t")
            for corpus, accs in sorted(simple_accuracies.items()):
                f.write("u-acc\t")
            f.write("\n")
            
            # write corpus names
            f.write(f"{args.datetime_string}\t{args.args_string}\tFIRST-W-THEN-U\t")
            for _ in range(2):
                for corpus, accs in sorted(simple_accuracies.items()):
                    f.write(f"{corpus}\t")
                f.write("\t")
            f.write("\n")
            
            # write actual accuracies
            f.write(f"{args.datetime_string}\t{args.args_string}\tFIRST-W-THEN-U\t")
            for jj in range(2):
                for corpus, accs in sorted(simple_accuracies.items()):
                    f.write(f"{accs[jj]:.4f}\t")
                f.write("\t")
            f.write("\n")

    # PRINTOUT ALL RELEVANT RESULTS
    # extract only the most basic accuracy values, for backward compatibility
    simple_dev_accuracies = {
        corpus: (full_accuracies[corpus]['dev']['all']['weighted'], full_accuracies[corpus]['dev']['all']['uniform'])
        for corpus
        in full_accuracies
    }

    print_simple_results_to_file(simple_dev_accuracies, args.result_filename)




    if args.eval_also_on_test:
        simple_test_accuracies = {
            corpus: (
            full_accuracies[corpus]['test']['all']['weighted'], full_accuracies[corpus]['test']['all']['uniform'])
            for corpus
            in full_accuracies
        }
        print_simple_results_to_file(simple_test_accuracies, args.result_filename + ".test")
    #
    # if args.sigmorphon_evaluation_orig_datadir:
    #     # TODO: run evaluation using official sigmorphon script, first pasting with the original dataset
    #     if args.data == "sig22":
    #         ...
    #     elif args.data == "sig23":
    #         ...
    #     ...
    #



def create_log_name(args: argparse.Namespace) -> str:
    # Create logdir
    logargs = dict(vars(args).items())

    # delete arguments that you don't need to store in the logdir name
    irrelevant_args = [
        # "seed",
        "corpus_name_separator", "threads", "show_results_every_batch",
        "datadir", "integration_results_file", "integration", "logdir",  # "ud_tgz_path", "ud_url",
        "corpus", "max_sentences", "result_dir", "jobid", "predictions_dir", "expid",
        "args", "seed", "print_first_preds_on_dev",
        # args of trm now always set to False
        "trm_add_final_layer_norm", "trm_scale_embedding", "trm_normalize_before",
        # only affects logs
        "weighted_accuracy_during_training", "weighted_multilingual_accuracy_during_training",
        "trm_gen_turn_sampling_off_during_training",
        "eval_also_on_test"
    ]

    if not args.multilingual_training:
        irrelevant_args += [
            "multilingual_corpus_down_up_sampling_temperature",
            "multilingual_corpus_down_up_sampling_weights"
        ]

    if not args.checkpoint_selection:
        irrelevant_args += [
            "acc_for_ckpt_selection"
        ]

    if args.model.startswith("rnn") or args.model.startswith("copy-bsln"):
        # remove all args corresponding to transformer since they are irrelevant
        irrelevant_args += [
            "trm_layer_dim", "trm_layer_count", "trm_attn_heads", "trm_ff_nn_dim", "trm_drop", "trm_attn_drop",
            "trm_feed_drop", "trm_layer_drop"
        ]
    if args.model.startswith("transformer") or args.model.startswith("copy-bsln"):
        # remove all args corresponding to RNN since they are irrelevant
        irrelevant_args += [
            "rnn_embedding_dim", "rnn_layer_dim", "rnn_layer_count", "rnn_drop", "rnn_unidirectional",
            "rnn_tie_embeddings", "rnn_hidden_drop", "rnn_attn_drop",
            "joint_vocab", # since it is obligatory

        ]


    for key in irrelevant_args:
        del logargs[key]

    args.args_string = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key),
                                                re.sub("^.*/", "", value) if type(value) == str else value)
                                 for key, value in sorted(logargs.items())))
    long_short_arg_values = {
        "True": "t",
        "False": "f",
        "cosine": "cos",
        "transformer": "trm",
        "None": "n",
        "none": "n",
        "w-accuracy": "w",
        "accuracy": "a",
        "multiling-acc": "m",
        "unique": "u",
        "total": "tot",
    }
    for (long_value, short_value) in long_short_arg_values.items():
        args.args_string = args.args_string.replace(long_value, short_value)

    args.datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    import json
    print("PARAMETERS ====================")
    print(json.dumps(vars(args), indent=4))
    print("===============================")

    # remove non-numeric characters from job ID
    args.jobid = ''.join([char for char in args.jobid if char.isdigit()])

    logname = "{}-{}-{}-{}".format(
        args.datetime_string,
        args.args_string,
        args.jobid,
        args.expid
    )

    return logname


def copy_files_and_dirs(dest_dir, suffixes):
    """
    Copies files and directories matching the given suffixes to the destination directory.
    Directories are copied recursively.
    :param dest_dir: The destination directory where files/directories will be copied.
    :param suffixes: List of suffixes (file extensions or directory names) to match.
    """

    import shutil
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Function to copy files or directories recursively
    def copy_item(src, dst):
        if os.path.isdir(src):
            # Copy directory recursively
            shutil.copytree(src, dst)
        else:
            # Copy file
            shutil.copy(src, dst)

    # Loop through all items in the script directory
    for item in os.listdir(script_dir):
        item_path = os.path.join(script_dir, item)

        # print(f"checking to copy the following item: {item_path}")

        # Check if the item matches any suffix in the list
        if any(item.endswith(suffix) for suffix in suffixes):
            destination = os.path.join(dest_dir, item)
            copy_item(item_path, destination)

    print(f"All matching files and directories have been copied to {dest_dir}.")


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.args != "none":
        set_args_to_predefined_values(args, configuration=args.args)

    check_and_fix_args(args)

    set_datadir_based_on_data(args)

    if args.run_lemma_tag_implies_form_exp:
        _run_lemma_tag_implies_form_experiment()
    else:
        if args.integration:
            run_integration_test(args)
        else:
            main(args)

    end_time = datetime.datetime.now()
    print(f"# TIME: TOTAL TIME OF RUNNING THE SCRIPT: {format_timedelta(end_time-start_time)}")
