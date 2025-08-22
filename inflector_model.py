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
import sys
import json5

PRINTOUTS = False

WACC_PRINTOUTS = False


def my_print(string):
    def pretty_print_list_of_lists(data, indent=4):
        print("[")
        for sublist in data:
            print(" " * indent + "[", end="")
            print(", ".join(map(str, sublist)), end="")
            print("]")
        print("]")

    if isinstance(string, list):
        import json
        print(pretty_print_list_of_lists(string, indent=4))
    else:
        print(string)


from typing import Any, BinaryIO, Callable, Iterable, Sequence, TextIO, Optional
from collections import defaultdict

from transformers import BartConfig, BartForConditionalGeneration, GenerationConfig


class MostCommonFormBaseline:
    """Baseline, that from training data for every lemma-tag pair extracts the most common form, and during prediction
    returns it. For unseen lemma-tag pairs returns empty string.
    Useful for determining the percentage of instances of type `lemma-tag-implies-form`, but completely useless as a
    baseline in our setting, where the dev/test-train lemma overlap is empty."""

    def __init__(self):
        self.most_common_forms = dict()

    def train(self, train_filename: str):
        """Train self on the training data. TODO: adjust to work with datasets, not files"""
        # Initialize a defaultdict to store the form with the highest count for each (lemma, tag) pair
        lemma_tag_dict = defaultdict(lambda: ("", 0))  # (form, count)

        # Open and process the file
        with open(train_filename, "r") as file:
            for line in file:
                lemma, form, tag, count = line.strip().split("\t")
                count = int(count)  # Convert count to integer

                # Check if this form has a higher count for the given (lemma, tag) pair
                if count > lemma_tag_dict[(lemma, tag)][1]:
                    lemma_tag_dict[(lemma, tag)] = (form, count)

        # Create the final dictionary with (lemma, tag) as keys and the best form as values
        self.most_common_forms = {key: value[0] for key, value in lemma_tag_dict.items()}

    def predict(self, lemma: str, tag: str) -> Optional[str]:
        return self.most_common_forms.get((lemma, tag), None)

    def predict_batch(self, lemmas: list[str], tags: list[str]) -> list[Optional[str]]:
        return [self.predict(lemma, tag) for (lemma, tag) in zip(lemmas, tags)]


# Adapted from Milan Straka's course of Deep Learning, academic year 2023/24 on MFF CUNI, assignment 10_lemmatizer_attn.

import argparse
import os

import numpy as np
import torch
import torchmetrics

from morpho_dataset import MorphoDataset, Vocabulary


def unpack_batch(batch, device):
    """Unpack a batch, that is given as a tuple, either as 2-tuple of xs, y, or as a 3-tuple of xs, y and weights w."""
    if len(batch) == 3:
        xs, y, w = batch
        w = w.to(device)

    elif len(batch) == 2:
        xs, y = batch
        w = None
    else:
        raise ValueError("Tuple must have 2 or 3 elements")

    # move everything to self.device (and for some reason, ensure that xs is a tuple, even if it is a tensor
    xs, y = tuple(x.to(device) for x in (xs if isinstance(xs, tuple) else (xs,))), y.to(device)
    return xs, y, w


class TrainableModule(torch.nn.Module):
    """A simple Keras-like module for training with raw PyTorch.

    The module provides fit/evaluate/predict methods, computes loss and metrics,
    and generates both TensorBoard and console logs. By default, it uses GPU
    if available, and CPU otherwise. Additionally, it offers a Keras-like
    initialization of the weights.

    The current implementation supports models with either single input or
    a tuple of inputs; however, only one output is currently supported.
    """
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    from time import time as _time
    from tqdm import tqdm as _tqdm

    def configure(self, *, optimizer=None, schedule=None, loss=None, metrics={}, logdir=None, device="auto",
                  clip_grad_norm=None, checkpoint_selection: bool = False, accuracy_for_ckpt_selection: str = None):
        """Configure the module process.

        - `optimizer` is the optimizer to use for training;
        - `schedule` is an optional learning rate scheduler used after every batch;
        - `loss` is the loss function to minimize;
        - `metrics` is a dictionary of additional metrics to compute;
        - `logdir` is an optional directory where TensorBoard logs should be written;
        - `device` is the device to use; when "auto", `cuda` is used when available, `cpu` otherwise.
        """
        self.optimizer = optimizer
        self.schedule = schedule
        self.loss, self.loss_metric = loss, torchmetrics.MeanMetric()
        self.metrics = torchmetrics.MetricCollection(metrics)
        self.logdir, self._writers = logdir, {}
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.to(self.device)

        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_selection = checkpoint_selection
        self.accuracy_for_ckpt_selection = accuracy_for_ckpt_selection

        if self.checkpoint_selection:
            self.path_to_best_checkpoint = "best_checkpoint.model"

        self.checkpoint_epoch = -1

    def load_weights(self, path, device="auto"):
        """Load the model weights from the given path."""
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.load_state_dict(torch.load(path, map_location=self.device))

    def save_weights(self, path):
        """Save the model weights to the given path."""
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def fit(self, dataloader, epochs, dev=None, callbacks=[], verbose=1):
        """Train the model on the given dataset.

        - `dataloader` is the training dataset, each element a pair of inputs and an output;
          the inputs can be either a single tensor or a tuple of tensors;
        - `dev` is an optional development dataset;
        - `epochs` is the number of epochs to train;
        - `callbacks` is a list of callbacks to call after each epoch with
          arguments `self`, `epoch`, and `logs`;
        - `verbose` controls the verbosity: 0 for silent, 1 for persistent progress bar,
          2 for a progress bar only when writing to a console.
        """
        best_ckpt_acc = float("-inf")
        best_ckpt_epochs = -1

        logs = {}
        for epoch in range(epochs):
            # if epoch % 12 == 0:
            #     my_print("a")
            self.train()
            self.loss_metric.reset()
            self.metrics.reset()
            start = self._time()
            epoch_message = f"Epoch={epoch + 1}/{epochs}"
            data_and_progress = self._tqdm(
                dataloader, epoch_message, unit="batch", leave=False, disable=None if verbose == 2 else not verbose)

            for batch in data_and_progress:
                # If there are no weights in the batch (it is only a 2-tuple, None is assigned to the weights)
                xs, y, weights_for_evaluation = unpack_batch(batch, device=self.device)
                # for xs, y in data_and_progress:
                # During training, (that is, now) xs=(inputs, targets), y=targets

                if WACC_PRINTOUTS:
                    print("-- Training --")
                    print(f"xs: {type(xs)}, len={len(xs)}")
                    print(f"xs[0]: {xs[0].shape}")
                    print(f"xs[1]: {xs[1].shape}")
                    print(f"y: {y.shape}")
                    print(f"w: {len(weights_for_evaluation)}")
                    print()
                    print(f"xs[1]: {xs[1]}")
                    print(f"y:     {y}")
                    print(f"w: {weights_for_evaluation}")
                    input()

                logs = self.train_step(xs, y, weights_for_evaluation=weights_for_evaluation)

                message = [epoch_message] + [f"{k}={v:.{0 < abs(v) < 2e-4 and '3g' or '4f'}}" for k, v in logs.items()]
                data_and_progress.set_description(" ".join(message), refresh=False)
            if dev is not None:
                dev_res = self.evaluate(dev, verbose=0)

                # if we want to compute what is the best checkpoint and store its weights
                if self.checkpoint_selection:

                    # extract the desired accuracy measure (weighted/uniform)
                    current_dev_acc = dev_res[self.accuracy_for_ckpt_selection]

                    # možná v případě rovnosti podle téhle accuracy, porovnávat i podle druhé accuracy?
                    # asi netřeba, v reálném případě nikdy nenastane rovnost
                    if current_dev_acc > best_ckpt_acc:
                        best_ckpt_epochs = epoch + 1
                        best_ckpt_acc = current_dev_acc
                        self.save_weights(self.path_to_best_checkpoint)

                logs |= {"dev_" + k: v for k, v in dev_res.items()}
            for callback in callbacks:
                callback(self, epoch, logs)
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("dev_")}, epoch + 1)
            self.add_logs("dev", {k[4:]: v for k, v in logs.items() if k.startswith("dev_")}, epoch + 1)
            verbose and print(epoch_message, "{:.1f}s".format(self._time() - start),
                              *[f"{k}={v:.{0 < abs(v) < 2e-4 and '3g' or '4f'}}" for k, v in logs.items()])

        # after training, load the best checkpoint weights
        if self.checkpoint_selection:
            self.load_weights(self.path_to_best_checkpoint)
            self.checkpoint_epoch = best_ckpt_epochs

            print("Training finished, best checkpoint loaded:")
            print(f"Epoch {best_ckpt_epochs}/{epochs}, {self.accuracy_for_ckpt_selection}: {best_ckpt_acc:.4f}")

        return logs

    def train_step(self, xs, y, weights_for_evaluation=None):
        """An overridable method performing a single training step.

        A dictionary with the loss and metrics should be returned.

        xs: tuple(inputs, targets)
        y: targets (the same as in xs)

        INPUTS: l e m m a t1 t2 t3 PAD PAD PAD (padded to the same length in the batch)
        TARGETS: f o r m EOW PAD PAD PAD (padded to the same length in the batch)

        inputs.shape = [batch_size, seq_len1]
        targets.shape = [batch_size, seq_len2]

        # TARGETS (for bs=4):
        # f o r m 1 EOW PAD PAD PAD
        # f o o o o r   m   2   EOW
        # f o r m m 3   EOW PAD PAD
        # f o o o r m   4   EOW PAD
        """

        if PRINTOUTS:
            print(f"Input to `train_steps`: {xs[0].shape}, {xs[1].shape}")
            print(f"{xs[0]}")
            print(f"{xs[1]}")
            print(f"Expected output: {y.shape}")
            print(f"Expected output: {y}")

        # vynuluj gradient
        self.zero_grad()

        # get the predictions (more precisely, logits)
        # y_pred.shape = [batch_size, tgt_vocab_size, seq_len]
        y_pred = self.forward(*xs)  # .forward(inputs, targets)

        if PRINTOUTS:
            print(f"After forward: {y_pred.shape}")

        # *xs are ignored during loss computation.
        loss = self.compute_loss(y_pred, y, *xs)

        if PRINTOUTS:
            print(loss)
        loss.backward()

        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)

        # y_forwarded = y_pred.argmax(dim=-2)
        # y_generated = self._generate(xs)
        # self.train()
        # result_forw = self.evaluate_batch(y_pred, y, *xs, training=True)
        # result_gen = self.evaluate_batch(y_generated, y, *xs, training=False)
        # if not torch.equal(result_forw, result_gen):
        #     print("forward result:")
        #     print(result_forw)
        #     print("generate result:")
        #     print(result_gen)
        #     print()
        #
        #     print("y_gold:")
        #     print(y)
        #     print(f"y_forwarded:")
        #     print(y_forwarded)
        #     print(f"y_generated:")
        #     print(y_generated)
        #     print()
        #     print("y_gold:")
        #     self.print_tensor(y)
        #     print("y_forwarded:")
        #     self.print_tensor(y_forwarded)
        #     print("y_generated:")
        #     self.print_tensor(y_generated)

        with torch.no_grad():
            self.optimizer.step()
            self.schedule is not None and self.schedule.step()
            self.loss_metric.update(loss)
            return {"loss": self.loss_metric.compute()} \
                | ({"lr": self.schedule.get_last_lr()[0]} if self.schedule else {
                    "lr": self.optimizer.param_groups[0]["lr"]}) \
                | self.compute_metrics(y_pred, y, *xs, training=True, weights_for_evaluation=weights_for_evaluation)

    def compute_loss(self, y_pred, y, *xs):
        """Compute the loss of the model given the inputs, predictions, and target outputs.
        y_pred.shape = [batch_size, tgt_vocab_size, seq_len]
        y.shape =      [batch_size, seq_len]
        """

        loss = self.loss(y_pred, y)
        return loss

    def compute_metrics(self, y_pred, y, *xs, training, weights_for_evaluation=None):
        """Compute and return metrics given the inputs, predictions, and target outputs.
        `weights_for_evaluation` are ignored in this implementation."""
        self.metrics.update(y_pred, y)
        return self.metrics.compute()

    def evaluate(self, dataloader, verbose=1):
        """An evaluation of the model on the given dataset.
        Called from `fit()` after each epoch for evaluation on dev.

        - `dataloader` is the dataset to evaluate on, each element a pair of inputs
          and an output, the inputs either a single tensor or a tuple of tensors;
        - `verbose` controls the verbosity: 0 for silent, 1 for a single message."""
        self.eval()
        self.loss_metric.reset()
        self.metrics.reset()

        for batch in dataloader:
            xs, y, weights_for_evaluation = unpack_batch(batch, device=self.device)
            # for xs, y in dataloader:
            if WACC_PRINTOUTS:
                print("-- Dev eval --")
                print(f"xs: {type(xs)}")
                print(f"xs: {xs.shape}")
                print(f"y: {y.shape}")
                print()
                print(f"xs: {xs}")
                print(f"y:  {y}")
                input()

            logs = self.test_step(xs, y, weights_for_evaluation=weights_for_evaluation)

        verbose and print("Evaluation", *[f"{k}={v:.{0 < abs(v) < 2e-4 and '3g' or '4f'}}" for k, v in logs.items()])
        return logs

    def test_step(self, xs, y, weights_for_evaluation=None):
        """An overridable method performing a single evaluation step.

        A dictionary with the loss and metrics should be returned."""

        with torch.no_grad():
            y_pred = self.forward(*xs)
            self.loss_metric.update(self.compute_loss(y_pred, y, *xs))
            return {"loss": self.loss_metric.compute()} | self.compute_metrics(y_pred, y, *xs, training=False,
                                                                               weights_for_evaluation=weights_for_evaluation)

    def predict(self, dataloader, as_numpy=True):
        """Compute predictions for the given dataset.

        - `dataloader` is the dataset to predict on, each element either
          directly the input or a tuple whose first element is the input;
          the input can be either a single tensor or a tuple of tensors;
        - `as_numpy` is a flag controlling whether the output should be
          converted to a numpy array or kept as a PyTorch tensor.

        The method returns a Python list whose elements are predictions
        of the individual examples. Note that if the input was padded, so
        will be the predictions, which will then need to be trimmed."""
        self.eval()
        predictions = []
        for batch in dataloader:
            xs = batch[0] if isinstance(batch, tuple) else batch
            xs = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,)))
            prediction = self.predict_step(xs, as_numpy=as_numpy)
            predictions.extend(prediction)
        return predictions

    def predict_step(self, xs, as_numpy=True):
        """An overridable method performing a single prediction step."""
        with torch.no_grad():
            batch = self.forward(*xs)
            return batch.numpy(force=True) if as_numpy else batch

    def writer(self, writer):
        """Possibly create and return a TensorBoard writer for the given name."""
        if writer not in self._writers:
            self._writers[writer] = self._SummaryWriter(os.path.join(self.logdir, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        """Log the given dictionary to TensorBoard with a given name and step number."""
        if logs and self.logdir:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    @staticmethod
    def keras_init(module):
        """Initialize weights using the Keras defaults."""
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                               torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
            torch.nn.init.uniform_(module.weight, -0.05, 0.05)
        if isinstance(module, (torch.nn.RNNBase, torch.nn.RNNCellBase)):
            for name, parameter in module.named_parameters():
                "weight_ih" in name and torch.nn.init.xavier_uniform_(parameter)
                "weight_hh" in name and torch.nn.init.orthogonal_(parameter)
                "bias" in name and torch.nn.init.zeros_(parameter)
                if "bias" in name and isinstance(module, (torch.nn.LSTM, torch.nn.LSTMCell)):
                    parameter.data[module.hidden_size:module.hidden_size * 2] = 1


class BaseInflectorModel(TrainableModule):
    """Base class for seq2seq models for inflection, implementing common methods for all subclasses, such as evaluation and prediction."""

    def __init__(self, args: argparse.Namespace, src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        super().__init__()

        self._source_vocab = src_vocab
        self._target_vocab = tgt_vocab

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

        # apart from len(lemma-tag), how many tokens can be generated during generation?
        self.max_new_tokens_for_generation = 10

    def print_batch_of_forms(self, xs):
        """Takes a batch of predicted forms and prints it in a human-readable way (decoded)."""
        for i in range(xs.shape[0]):
            print("\t" + "".join(self._target_vocab.strings(np.trim_zeros(xs[i].numpy(force=True)))))

    def train_step(self, xs, y, weights_for_evaluation=None):
        """The same as in RNN model. First, call the `TrainableModule`'s implementation of the `train_step`, and then print the intermediate predictions if desired."""
        result = super().train_step(xs, y, weights_for_evaluation=weights_for_evaluation)

        self._batches += 1
        if self._batches % self._show_results_every_batch == 0:
            self._tqdm.write("{}: {} -> {}".format(
                self._batches,
                "·".join(self._source_vocab.strings(np.trim_zeros(xs[0][0].numpy(force=True)))),
                "".join(self._target_vocab.strings(self.predict_step((xs[0][:1],))[0]))))

        return result

    # Override compute metrics, because it is called from TrainableModule's `train_step()`
    def compute_metrics(self, y_pred, y, *xs, training, weights_for_evaluation=None):
        # TODO: tady se počítá accuracy
        # TODO: až tomu přidám váhy (k trénování), tady bych měl umět spočítat i tu weighted accuracy, pokud si ty váhy někde tady taky předám

        if training:
            # take max over all logits
            y_pred = y_pred.argmax(dim=-2)

        # Crop the rest of y_pred sequences that is longer than y_gold
        y_pred = y_pred[:, :y.shape[-1]]

        # if y_pred is shorter than y_gold, pad with padding values
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)

        correct_predictions = torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1)
        self.metrics["accuracy"](correct_predictions)

        # Update weighted accuracy
        if "w-accuracy" in self.metrics and weights_for_evaluation is not None:
            # `correct_predictions` is a 0-1 tensor of shape [batch_size], where 0 means incorrect prediction, 1 means correct prediction
            # we want to repeat each 0/1 the number of times equal to the corresponding weight for evaluation, which is
            # exactly what `torch.repeat_interleave()` does
            repeated_correct_predictions = torch.repeat_interleave(correct_predictions, weights_for_evaluation)
            if WACC_PRINTOUTS:
                print(f"Compute metrics evaluation")
                print(f"evaluated predictions shape: {correct_predictions.shape}")
                print(f"sum of weights of the examples in batch: {int(weights_for_evaluation.sum().item())}")
                print(f"repeated evaluated predictions shape: {repeated_correct_predictions.shape}")

            self.metrics["w-accuracy"].update(repeated_correct_predictions)

        # Multilingual macro-avg over langs
        elif "multiling-acc" in self.metrics and weights_for_evaluation is not None:
            repeated_correct_predictions = torch.repeat_interleave(correct_predictions, weights_for_evaluation)
            self.metrics["multiling-acc"].update(repeated_correct_predictions)

        if WACC_PRINTOUTS:
            print(f"Training: {training}")
            print(f"y: {y.shape}")
            print(f"y_pred: {y_pred.shape}")
            input()

        return self.metrics.compute()

    def test_step(self, xs, y, weights_for_evaluation=None):

        with torch.no_grad():
            y_pred = self.forward(*xs)
            return self.compute_metrics(y_pred, y, *xs, training=False, weights_for_evaluation=weights_for_evaluation)

    def predict_step(self, xs, as_numpy=True):
        """Predict step for autoregressive prediction.
        Used both for printouts of predictions during training, and from `predict()` method of TrainableModule, which is used to generate predictions for a whole dataset (called from `make_model_predict` from main)"""
        with torch.no_grad():
            batch = self.forward(*xs)
            # If `as_numpy==True`, trim the predictions at the first EOW.
            # Useful when printing the results or returning them to a user.
            # Irrelevant when just computing loss or accuracy.
            if as_numpy:
                batch = [example[np.cumsum(example == MorphoDataset.EOW) == 0] for example in batch.numpy(force=True)]
            return batch

    def evaluate_batch(self, y_pred, y, *xs, training):
        """Evaluation of a single batch, used for printing intermediate results."""
        if training:
            # take max over all logits
            y_pred = y_pred.argmax(dim=-2)
        # Crop the rest of y_pred sequences that is longer than y_gold
        y_pred = y_pred[:, :y.shape[-1]]

        # if y_pred is shorter than y_gold, pad with padding values
        y_pred = torch.nn.functional.pad(y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD)
        return torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1)


class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention vo the given RNN cell."""

    def __init__(self, cell, attention_dim, attention_drop):
        super().__init__()
        self._cell = cell

        # - `self._project_encoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs.
        # - `self._project_decoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs
        # - `self._output_layer` as a linear layer with `attention_dim` inputs and 1 output
        self._project_encoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)
        self._project_decoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)
        self._output_layer = torch.nn.Linear(attention_dim, 1)

        if attention_drop > 0:
            self.attention_dropout = torch.nn.Dropout(attention_drop)
        else:
            self.attention_dropout = None

    def setup_memory(self, encoded):
        self._encoded = encoded
        # Pass the `encoded` through the `self._project_encoder_layer` and store
        # the result as `self._encoded_projected`.
        self._encoded_projected = self._project_encoder_layer(encoded)

    def forward(self, inputs, states):
        # Compute the attention.
        # - According to the definition, we need to project the encoder states, but we have
        #   already done that in `setup_memory`, so we just take `self._encoded_projected`.
        # - Compute projected decoder state by passing the given state through the `self._project_decoder_layer`.
        # - Sum the two projections. However, the first has shape `[batch_size, input_sequence_len, attention_dim]`
        #   and the second just `[batch_size, attention_dim]`, so the second needs to be expanded so that
        #   the sum of the projections broadcasts correctly.
        projected = self._encoded_projected + self._project_decoder_layer(states)[:, np.newaxis, :]
        # - Pass the sum through the `torch.tanh` and then through the `self._output_layer`.
        # - Then, run softmax on a suitable axis, generating `weights`.
        weights = torch.softmax(self._output_layer(torch.tanh(projected)), dim=1)

        if self.attention_dropout:
            weights = self.attention_dropout(weights)

        # - Multiply the original (non-projected) encoder states `self._encoded` with `weights` and sum
        #   the result in the axis corresponding to characters, generating `attention`. Therefore,
        #   `attention` is a fixed-size representation for every batch element, independently on
        #   how many characters the corresponding input form had.
        attention = torch.sum(self._encoded * weights, dim=1)
        # - Finally, concatenate `inputs` and `attention` (in this order), and call the `self._cell`
        #   on this concatenated input and the `states`, returning the result.

        if PRINTOUTS:
            print(f"inputs shape: {inputs.shape}")
            print(f"attention shape: {attention.shape}")
            print(f"cat: {torch.cat([inputs, attention], dim=1).shape}")

        return self._cell(torch.cat([inputs, attention], dim=1), states)


class RNNEncoderDecoderWithAttention(BaseInflectorModel):
    def __init__(self, args: argparse.Namespace, src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> None:
        super().__init__(args, src_vocab, tgt_vocab)

        embedding_dim = args.rnn_embedding_dim
        layer_dim = args.rnn_layer_dim
        layer_count = args.rnn_layer_count
        dropout = args.rnn_drop
        self.bidirectional = not args.rnn_unidirectional
        self.num_layers = layer_count

        # - `self._source_embedding` as an embedding layer of source characters into `embedding_dim` dimensions
        # - `self._source_rnn` as a bidirectional GRU with `layer_dim` units processing embedded source chars
        self._source_embedding = torch.nn.Embedding(len(self._source_vocab), embedding_dim)

        # self._source_rnn = torch.nn.GRU(embedding_dim, layer_dim, batch_first=True, bidirectional=True)

        ### NUM_LAYERS
        # Update the `_source_rnn` to include `layer_count`
        self._source_rnn = torch.nn.GRU(
            embedding_dim, layer_dim, num_layers=layer_count,
            batch_first=True, bidirectional=self.bidirectional, dropout=dropout if layer_count > 1 else 0
        )

        # - `self._target_rnn_cell` as a `WithAttention` with `attention_dim=layer_dim`, employing as the
        #   underlying cell the `torch.nn.GRUCell` with `layer_dim`. The cell will process concatenated
        #   target character embeddings and the result of the attention mechanism.

        self._target_rnn_cell = WithAttention(
            cell=torch.nn.GRUCell(input_size=embedding_dim + layer_dim,
                                  hidden_size=layer_dim
                                  ),
            attention_dim=layer_dim,
            attention_drop=args.rnn_attn_drop
        )

        # - `self._target_output_layer` as a linear layer into as many outputs as there are unique target chars
        self._target_output_layer = torch.nn.Linear(layer_dim, len(self._target_vocab))

        if not args.rnn_tie_embeddings:
            # `self._target_embedding` as an embedding layer of the target
            # characters into `embedding_dim` dimensions.
            self._target_embedding = torch.nn.Embedding(len(self._target_vocab), embedding_dim)
        else:
            #  Create a function `self._target_embedding` computing the embedding of given
            # target characters. When called, use `torch.nn.functional.embedding` to suitably
            # index the shared embedding matrix `self._target_output_layer.weight`
            # multiplied by the square root of `layer_dim`.
            self._target_embedding = lambda inputs: (layer_dim ** 0.5) * torch.nn.functional.embedding(
                inputs, self._target_output_layer.weight)

        if args.rnn_hidden_drop > 0:
            self.hidden_dropout = torch.nn.Dropout(args.rnn_hidden_drop)
        else:
            self.hidden_dropout = None

        # Initialize the layers using the Keras-inspired initialization. You can try
        # removing this line to see how much worse the default PyTorch initialization is.
        self.apply(self.keras_init)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """Forward call to the inflection model.
        If `targets` are given, a training forward pass is performed (using teacher forcing in decoder prediction).
        If `targets`==None, a prediction with auto-regressive decoding is performed

        INPUTS: l e m m a t1 t2 t3 PAD PAD PAD (padded to the same length in the batch)
        TARGETS: f o r m EOW PAD PAD PAD (padded to the same length in the batch)

        inputs.shape = [batch_size, seq_len1]
        targets.shape = [batch_size, seq_len2]

        # TARGETS (for bs=4):
        # f o r m 1 EOW PAD PAD PAD
        # f o o o o r   m   2   EOW
        # f o r m m 3   EOW PAD PAD
        # f o o o r m   4   EOW PAD

        returns:
        in training regime (`targets` are given): logits: logits.shape = [batch_size, tgt_vocab_size, seq_len2]
        in prediction regime (`targets`==None): predicted_forms: predicted_forms.shape = [batch_size, seq_len2]
        """
        encoded = self.encoder(inputs)

        if self.hidden_dropout:
            # Use dropout on the hidden state (optional, depending on your model)
            encoded = self.hidden_dropout(encoded)

        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=inputs.shape[1] + self.max_new_tokens_for_generation)

    def encoder(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the encoder on the given batch of inputs (lemma-tag sequences)."""

        # Embed the inputs using `self._source_embedding`.
        hidden = self._source_embedding(inputs)

        # Run the `self._source_rnn` on the embedded sequences, correctly handling
        # padding. Anew, the result should be encoding of every sequence element,
        # summing results in the opposite directions.
        forms_len = torch.sum(inputs != MorphoDataset.PAD, dim=-1).cpu()

        if PRINTOUTS:
            print("forms len: " + str(forms_len))

        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, forms_len, batch_first=True, enforce_sorted=False)

        # Handle hidden states from multiple layers when using bidirectional RNN
        hidden, _ = self._source_rnn(packed)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        if PRINTOUTS:
            print(hidden.shape)

        if self.bidirectional:
            hidden = sum(torch.chunk(hidden, 2, dim=-1))

        if PRINTOUTS:
            print('after summing directions')
            print(hidden.shape)

        return hidden

    def decoder_training(self, encoded: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Run the decoder with teacher forcing in training."""

        # Generate inputs for the decoder, which are obtained from `targets` by
        # - prepending `MorphoDataset.BOW` as the first element of every batch example,
        # - dropping the last element of `targets`.
        decoder_inputs = torch.nn.functional.pad(targets[:, :-1], (1, 0), value=MorphoDataset.BOW)

        # Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.
        self._target_rnn_cell.setup_memory(encoded)

        # Process the generated inputs by
        # - the `self._target_embedding` layer to obtain embeddings,
        # - repeatedly call the `self._target_rnn_cell` on the sequence of embedded
        #   inputs and the previous states, starting with state `encoded[:, 0]`,
        #   obtaining outputs for all target hidden states,
        # - the `self._target_output_layer` to obtain logits,
        # - finally, permute dimensions so that the logits are in the dimension 1,
        # and return the result.
        hidden = self._target_embedding(decoder_inputs)

        if PRINTOUTS:
            print(f'embedding: {hidden.shape}')

        hiddens, states = [], encoded[:, 0]
        for i in range(hidden.shape[1]):
            hiddens.append(states := self._target_rnn_cell(hidden[:, i], states))
        hidden = self._target_output_layer(torch.stack(hiddens, dim=1))
        hidden = hidden.permute(0, 2, 1)
        return hidden

    def decoder_prediction(self, encoded: torch.Tensor, max_length: int) -> torch.Tensor:
        """Run the decoder with auto-regressive decoding. Used in prediction."""

        batch_size = encoded.shape[0]

        # Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.
        self._target_rnn_cell.setup_memory(encoded)

        # Define the following variables, that we will use in the cycle:
        # - `index`: the time index, initialized to 0;
        # - `inputs`: a tensor of shape `[batch_size]` containing the `MorphoDataset.BOW` symbols,
        # - `states`: initial RNN state from the encoder, i.e., `encoded[:, 0]`.
        # - `results`: an empty list, where generated outputs will be stored;
        # - `result_lengths`: a tensor of shape `[batch_size]` filled with `max_length`,
        index = 0
        inputs = torch.full([batch_size], MorphoDataset.BOW, dtype=torch.int32, device=encoded.device)
        states = encoded[:, 0]
        results = []
        result_lengths = torch.full([batch_size], max_length, dtype=torch.int32, device=encoded.device)

        while index < max_length and torch.any(result_lengths == max_length):
            # - First embed the `inputs` using the `self._target_embedding` layer.
            # - Then call `self._target_rnn_cell` using two arguments, the embedded `inputs`
            #   and the current `states`. The call returns a single tensor, which you should
            #   store as both a new `hidden` and a new `states`.
            # - Pass the outputs through the `self._target_output_layer`.
            # - Generate the most probable prediction for every batch example.
            hidden = self._target_embedding(inputs)
            hidden = states = self._target_rnn_cell(hidden, states)
            hidden = self._target_output_layer(hidden)
            predictions = hidden.argmax(dim=-1)

            # Store the predictions in the `results` and update the `result_lengths`
            # by setting it to current `index` if an EOW was generated for the first time.
            results.append(predictions)
            result_lengths[(predictions == MorphoDataset.EOW) & (result_lengths > index)] = index + 1

            # Finally,
            # - set `inputs` to the `predictions`,
            # - increment the `index` by one.
            inputs = predictions
            index += 1

        results = torch.stack(results, dim=1)
        return results


# BartForConditionalGeneration (https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration):
#
# The BART Model with a language modeling head. Can be used for summarization.
# This model inherits from PreTrainedModel. Check the superclass documentation
# for the generic methods the library implements for all its model (such as
# downloading or saving, resizing the input embeddings, pruning heads etc.)
#
# This model is also a PyTorch torch.nn.Module subclass. Use it as a regular
# PyTorch Module and refer to the PyTorch documentation for all matter related
# to general usage and behavior.
# as the documentation says:
#
# https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration.config
#
# config (BartConfig) — Model configuration class with all the parameters
# of the model. Initializing with a config file does not load the weights
# associated with the model, only the configuration. Check out the
# from_pretrained() method to load the model weights.
#
# The BartConfig and the key-value pairs is documented here:
#
# https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartConfig
#
# An example of BART Base-sized model can be found here:
#
# https://huggingface.co/facebook/bart-base
#
# The config file is under the Files and versions tab:
#
# https://huggingface.co/facebook/bart-base/tree/main
#
# The config itself here:
#
# https://huggingface.co/facebook/bart-base/blob/main/config.json

class BartWrapper(BaseInflectorModel):
    """
    Wrapper of BartForConditionalGeneration, used for training a transformer model for inflection from scratch.
    """

    def __init__(self, args: argparse.Namespace, src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> None:

        # Initialize TrainableModule parent class
        super().__init__(args, src_vocab, tgt_vocab)

        # TODO: get the config path from args
        config_filepath = "configs/transformer_bart.json"
        config_dict = read_json_with_comments(config_filepath)
        config = BartConfig.from_dict(config_dict)

        config_dict_for_update = {
                # source and target vocab should be the same, but just to be sure, use max
                "vocab_size": max([len(self._source_vocab), len(self._target_vocab)]),
                "decoder_start_token_id": MorphoDataset.DECODER_START_TOKEN_ID,
                "bos_token_id": MorphoDataset.BOW,
                "eos_token_id": MorphoDataset.EOW,
                "forced_eos_token_id": MorphoDataset.EOW,
                "pad_token_id": MorphoDataset.PAD,

                # model capacity
                "num_hidden_layers": args.trm_layer_count,
                "encoder_layers": args.trm_layer_count,
                "decoder_layers": args.trm_layer_count,
                "d_model": args.trm_layer_dim,
                "decoder_attention_heads": args.trm_attn_heads,
                "encoder_attention_heads": args.trm_attn_heads,
                "encoder_ffn_dim": args.trm_ff_nn_dim,
                "decoder_ffn_dim": args.trm_ff_nn_dim,

                # regularization
                "activation_dropout": args.trm_feed_drop,
                "attention_dropout": args.trm_attn_drop,
                "dropout": args.trm_drop,
                "decoder_layerdrop": args.trm_layer_drop,
                "encoder_layerdrop": args.trm_layer_drop,

                # normalization
                "add_final_layer_norm": args.trm_add_final_layer_norm,
                "scale_embedding": args.trm_scale_embedding,
                "normalize_before": args.trm_normalize_before,
            }

        config.update(config_dict_for_update)

        # more generation config
        self.num_beams = args.trm_gen_num_beams
        self.penalty_alpha = args.trm_gen_penalty_alpha
        self.top_k = args.trm_gen_top_k
        self.do_sample = (self.penalty_alpha is not None)

        # generation config:
        # generation_config = {
        #     "num_beams": args.trm_gen_num_beams
        # }
        # if args.trm_gen_penalty_alpha is not None:
        #     generation_config["penalty_alpha "] = args.trm_gen_penalty_alpha
        # if args.trm_gen_top_k is not None:
        #     generation_config["top_k"] = args.trm_gen_top_k
        #     # `top_k` is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
        #     generation_config["do_sample"] = True
        # print(f"generation config dict: {generation_config}")
        # self._generation_config = GenerationConfig.from_dict(generation_config)
        #
        # print("Generation configuration:")
        # print(self._generation_config)


        self.turn_sampling_off_during_training = args.trm_gen_turn_sampling_off_during_training

        if not self.turn_sampling_off_during_training:
            print("Config updated by the generation configuration during initialization. If sampling is turned on, it will be used already during dev evaluation between epochs.")
            #self.generation_config = self._generation_config
            self.generation_sampling_turned_on = True
        else:
            #self.generation_config = None
            self.generation_sampling_turned_on = False



        print("Running BART from scratch with the following configuration: ")
        print(config)

        # Initialize BART architecture without weights from custom config
        self._model = BartForConditionalGeneration(config)

        print(
            f"GENERATION INFO: maximal allowed length of generated form is len(lemma-tag) + {self.max_new_tokens_for_generation} chars",
            file=sys.stderr)

        self.apply(self.keras_init)

    # override fit to allow turning on sampling in generation after training the model
    def fit(self, dataloader, epochs, dev=None, callbacks=[], verbose=1):
        logs = super().fit(dataloader, epochs, dev, callbacks, verbose)

        # turn on sampling at the end of training
        if self.turn_sampling_off_during_training:
            print("Turning on sampling after training.")
            #self.generation_config=self._generation_config
            self.generation_sampling_turned_on = True

        return logs

    # Override forward method for training
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor | None = None):
        """Forward call to the BART model.
        If `targets` are given, a training forward pass is performed (using teacher forcing in decoder prediction).
        If `targets`==None, a prediction with auto-regressive decoding is performed

        INPUTS: l e m m a t1 t2 t3 PAD PAD PAD (padded to the same length in the batch)
        TARGETS: f o r m EOW PAD PAD PAD (padded to the same length in the batch)

        inputs.shape = [batch_size, seq_len1]
        targets.shape = [batch_size, seq_len2]

        # TARGETS (for bs=4):
        # f o r m 1 EOW PAD PAD PAD
        # f o o o o r   m   2   EOW
        # f o r m m 3   EOW PAD PAD
        # f o o o r m   4   EOW PAD

        returns:
        in training regime (`targets` are given): logits: logits.shape = [batch_size, tgt_vocab_size, seq_len2]
        in prediction regime (`targets`==None): predicted_forms: predicted_forms.shape = [batch_size, seq_len2]
        """

        if PRINTOUTS:
            print(f"`forward`: forms {inputs.shape}")
            self.print_tensor(inputs)

        # As we received the token integers in a rectangular batch, padded
        # by a padding value, we need to exclude these padded values from
        # the attention by specifying an attention mask.
        attention_mask = inputs != MorphoDataset.PAD  # whichever padding value we are using to pad

        if targets is not None:
            # We are running in the training mode. Run teacher forcing decoding.
            #
            # Generate inputs for the decoder, which are obtained from `targets` by
            # - prepending `MorphoDataset.DECODER_START_TOKEN_ID` as the first element of every batch example,
            # - dropping the last element of `targets`.
            #
            # Why do we do this?
            # Je to proto, aby vycházely velikosti. Konkrétně chceme, aby výstupů (=timestepů) z dekodéru bylo přesně tolik, jak velké jsou zlaté labels. Z těch zlatých labels děláme vstupy do dekodéru tím, že na začátek předřadíme BOW – takže kdybychom nesmazali poslední znak, byly by o jedna delší než zlaté výstupy, a dekodér by vygeneroval o jeden timestep větší výstup.
            # Jinak řečeno, když konstruujeme pro dekodér hodnoty, které “jakože vygeneroval” v předchozím kroce, tak nepotřebujeme úplně poslední znak (protože pro něj už žádný pokračovací negenerujeme).
            # A ohledně toho, že toj sou samé PADy – nejsou, alespoň jeden z nich je EOW (konstruujeme batche tak, aby byly nejmenší možné, takže alespoň jedna zlatá sekvence je dlouhá přesně jako batch, a končí EOW).
            # BTW, kdybychom místo uřezání posledního vstupu pustili dekodér na tom neuřezaném a pak uřezali výstup dekodéru (aby bylo možné spustit loss), tak by to dopadlo identicky.

            gold_decoder_input_ids_for_teacher_forcing = torch.nn.functional.pad(targets[:, :-1], (1, 0),
                                                                                 value=MorphoDataset.DECODER_START_TOKEN_ID)

            # Preprended with DECODER_START_TOKEN_ID,
            # removed last item, thus the longest form will be without EOW.
            # TARGETS (for bs=4):
            # DECODER_START_TOKEN_ID f o r m 1 EOW PAD PAD
            # DECODER_START_TOKEN_ID f o o o o r   m   2
            # DECODER_START_TOKEN_ID f o r m m 3   EOW PAD
            # DECODER_START_TOKEN_ID f o o o r m   4   EOW

            # passing unshifted targets as labels is the same as passing shifted targets (with prepended decoder-start-token-id) as decoder_input_ids
            # outputs = self._model(input_ids=forms, attention_mask=attention_mask, labels=targets)

            # pass shifted targets for teacher forcing. Do not pass labels, since they are only used to produce these shifted targets (not needed) and to compute loss (we compute it ourselves in TrainableModule.
            outputs = self._model(input_ids=inputs, attention_mask=attention_mask,
                                  decoder_input_ids=gold_decoder_input_ids_for_teacher_forcing)

            reshaped_logits = outputs.logits.permute(0, 2, 1)

            if PRINTOUTS:
                print(f'logits: {outputs.logits.shape}')
                print(f"reshaped logits: {reshaped_logits.shape}")

            return reshaped_logits
        else:
            # We are running in the prediction mode, generate predictions
            return self._generate(inputs)

    def _generate(self, xs):
        """Call `self._model.generate(xs) and properly treat input-output."""

        should_be_in_training_mode = self.training

        # Check if the model is in training mode
        if should_be_in_training_mode:
            # Turn on the evaluation mode, because for generating we always want the eval mode without dropouts etc.
            # but remember to turn on the training after the generation again!
            self.eval()

        # if also the gold targets are given in xs (in a tuple with inputs), extract only the inputs
        xs = xs[0] if isinstance(xs, tuple) else xs

        # during dev eval, print xs
        # if not should_be_in_training_mode:
        #     print(xs)
        #     for i in range(xs.shape[0]):
        #         print("·".join(self._source_vocab.strings(np.trim_zeros(xs[i].numpy(force=True)))))

        with torch.no_grad():
            if self.generation_sampling_turned_on:

                if self.num_beams is not None:
                    batch = self._model.generate(
                        inputs=xs,
                        num_beams=self.num_beams,
                        max_new_tokens=xs.shape[1] + self.max_new_tokens_for_generation + 1
                        # (the 1 is for `decoder_start_token_id`, which is always generated)
                    )
                elif self.top_k is not None and self.penalty_alpha is not None and self.do_sample:
                    batch = self._model.generate(
                        inputs=xs,
                        top_k=self.top_k,
                        penalty_alpha=self.penalty_alpha,
                        do_sample=self.do_sample,
                        max_new_tokens=xs.shape[1] + self.max_new_tokens_for_generation + 1
                        # (the 1 is for `decoder_start_token_id`, which is always generated)
                    )
                else:
                    batch = self._model.generate(
                        inputs=xs,
                        max_new_tokens=xs.shape[1] + self.max_new_tokens_for_generation + 1
                        # (the 1 is for `decoder_start_token_id`, which is always generated)
                    )
            else:
                batch = self._model.generate(
                    inputs=xs,
                    max_new_tokens=xs.shape[1] + self.max_new_tokens_for_generation + 1
                    # (the 1 is for `decoder_start_token_id`, which is always generated)
                )
            # , num_beams=2, min_length=0)#,decoder_start_token_id=self._model.config.pad_token_id)

            # remove the DECODER_START_TOKEN_ID from each sequence
            batch = batch[:, 1:]

        # If we are in a training process, turn the training mode on again
        if should_be_in_training_mode:
            self.train()

        return batch


def read_json_with_comments(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Parse the JSON with comments using json5
            return json5.load(file)
    except Exception as e:
        print(f"Error reading or parsing the file: {e}")
        return None


class CopyBaseline(TrainableModule):
    def __init__(self) -> None:
        super().__init__()

        # only to achieve that the model would have some parameters (needed for passing to optimizer for compatibility)
        self.dummy = torch.nn.Embedding(1,1)

    def fit(self, dataloader, epochs, dev=None, callbacks=[], verbose=1):
        pass

    def predict_step(self, xs, as_numpy=True):
        """An overridable method performing a single prediction step."""
        batch = xs[0] if isinstance(xs, tuple) else xs
        batch = [example[np.cumsum(example == MorphoDataset.SEP) == 0] for example in batch.numpy(force=True)]
        return batch

