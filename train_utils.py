# Copyright 2020 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Lint as: python3
"""Util functions for training models."""

import copy
import datetime
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from absl import flags
from absl import logging
import attr
import numpy as np
from sklearn import metrics as sklearn_metrics
import tensorflow.compat.v1 as tf
import tree as nest
import smart_module
import dataset_utils
import models


@attr.s
class GraphVars:
  eval_loss: Any = attr.ib()
  regularization_loss: Any = attr.ib()
  accuracy: Any = attr.ib()
  unmasked_count: Any = attr.ib()
  predictions: Any = attr.ib()
  labels: Any = attr.ib()


class L1Regularizer:
  """L1 regularizer.

  >>> reg = L1Regularizer(0.01)
  >>> reg([tf.constant([1.0, 2.0, 3.0])])
  <tf.Tensor: ...>
  """

  def __init__(self, scale: float):
    """Create an L1 regularizer.

    Args:
      scale: A non-negative regularization factor.

    Raises:
      ValueError: if scale is <0.
    """
    if scale < 0:
      raise ValueError(f"Expected non-negative scale. Got '{scale}'.")
    self.scale = scale

  def __repr__(self):
    return "L1(scale={})".format(self.scale)

  __str__ = __repr__

  def __call__(self, tensors: Sequence[tf.Tensor]) -> tf.Tensor:
    """Apply a regularizer.

    Args:
      tensors: A sequence of tensors to regularize.

    Returns:
      Combined regularization loss for the given tensors.
    """
    if not tensors:
      return tf.zeros_like(self.scale)

    return self.scale * tf.add_n([tf.reduce_sum(tf.abs(t)) for t in tensors])


def sample_batch(batch_size: int,
                 data_split: Dict[str, Any]) -> Dict[str, Any]:
  """Sample a randomly indexed batch from the data split."""
  idx = np.random.choice(range(data_split["label"].shape[1]), size=batch_size)
  sampled_data_split = {
      "sequence": data_split["sequence"][:, idx, ...],
      "label": data_split["label"][:, idx, ...]}
  if "mask" in data_split and data_split["mask"] is not None:
    sampled_data_split["mask"] = data_split["mask"][:, idx, ...]
  else:
    sampled_data_split["mask"] = (
        np.ones_like(sampled_data_split["label"]).astype(bool))
  return sampled_data_split


def get_extended_training_dir(checkpoint_directory: str,
                              checkpoint_dir_extension: str,
                              backup_directory_name: str) -> str:
  """Extends original checkpoint directory with a folder for train objects.

  Args:
    checkpoint_directory: the original checkpoint directory for storing all
      model checkpoints.
    checkpoint_dir_extension: the folder extension the includes unique training
      information.
    backup_directory_name: what to use for the original checkpoint directory
      when checkpoint directory is None.

  Returns:
    A new checkpoint dir with folder extensions.
  """
  if checkpoint_directory is None:
    checkpoint_dir = os.path.join(backup_directory_name,
                                  checkpoint_dir_extension)
  else:
    checkpoint_dir = os.path.join(
        checkpoint_directory, checkpoint_dir_extension)
  return checkpoint_dir


def get_placeholder_variables(feature_ndims, num_targets):
  """Creates placeholder variables for feeding in data."""
  sequences_ph = tf.placeholder(
      name="sequences", shape=(None, None, feature_ndims), dtype=tf.float32)
  labels_ph = tf.placeholder(
      name="labels", shape=(None, None, num_targets), dtype=tf.float32)
  mask_ph = tf.placeholder(
      name="mask", shape=(None, None, num_targets), dtype=tf.float32)
  return sequences_ph, labels_ph, mask_ph


def get_graph_variables(
    model: smart_module.SmartModuleExport, sequences: tf.Tensor,
    labels: tf.Tensor, l1_regularization_scale: float,
    mask: Optional[tf.Tensor], mode: str) -> GraphVars:
  """Gets loss and accuracy computation graph variables."""
  if mode not in {"train", "eval"}:
    raise ValueError(f"Unrecognized model mode: {mode}.")
  logits = (model.unroll_train(sequences) if mode == "train"
            else model.unroll_eval(sequences))
  logits = tf.reshape(logits, [-1])
  labels = tf.reshape(labels, [-1])
  mask = tf.reshape(mask, [-1])
  logits = tf.boolean_mask(logits, mask)
  labels = tf.boolean_mask(labels, mask)
  predictions = tf.nn.sigmoid(logits)
  loss_per_sample = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=logits)
  loss_op = tf.reduce_mean(loss_per_sample)
  reg = L1Regularizer(l1_regularization_scale)
  regularization_loss = reg(model.trainable_variables)
  unmasked_count = tf.shape(logits)
  # Accuracy
  correct_predictions = tf.equal(
      tf.cast(predictions > 0.5, tf.int32),
      tf.cast(labels, tf.int32))
  correct_predictions = tf.cast(correct_predictions, tf.float32)
  accuracy_op = tf.reduce_mean(correct_predictions)
  return GraphVars(eval_loss=loss_op,
                   regularization_loss=regularization_loss,
                   accuracy=accuracy_op,
                   unmasked_count=unmasked_count,
                   predictions=predictions,
                   labels=labels)


def convert_batch_to_time_major(batch):
  """Converts relevant batch arrays to time-major format."""
  batch["sequence"] = tf.transpose(batch["sequence"], [1, 0, 2])
  batch["label"] = tf.transpose(batch["label"], [1, 0, 2])
  if "mask" in batch and batch["mask"] is not None:
    batch["mask"] = tf.transpose(batch["mask"], [1, 0, 2])
  return batch


def get_mask_for_batch(batch):
  """Gets mask from batch object."""
  if "mask" in batch:
    mask = batch["mask"]
  else:
    mask = tf.ones_like(batch["label"])
  # Cast tensors into compatible dtype.
  mask = tf.cast(mask, tf.bool)
  return mask


def train_model(
    dataset_path: str,
    cell_type: str,
    hidden_sizes: List[int],
    input_dropout_prob: float = 0.0,
    output_dropout_prob: float = 0.0,
    state_dropout_prob: float = 0.0,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    num_train_steps: int = 10000,
    seed: int = None,
    checkpoint_base_path: str = None,
    logging_interval: int = 200,
    l1_regularization_scale: float = 0.0
    ) -> Dict[str, Union[int, float]]:
  """Train and export a DeepRNN model.

  Args:
    dataset_path: A path pointing to the either the dataset dir or pickled file.
    cell_type: Cell type of model.
    hidden_sizes: Size of each RNN layer.
    input_dropout_prob: Dropout probability for inputs to hidden layers.
    output_dropout_prob: Dropout probability for outputs to hiddne layers.
    state_dropout_prob: Dropout probability for hidden state activations.
    batch_size: Size of a batch.
    learning_rate: Learning rate for the optimizer.
    num_train_steps: Number of training steps.
    seed: Seed used by random number generators.
    checkpoint_base_path: Base path for checkpointing and model export.
    logging_interval: Logging every x steps (default 50).
    l1_regularization_scale: Optional scale for L1 regularization. If none is
      provided, no regularization will be collected and incorporated in the
      objective.

  Returns:
    Results dict with final loss and accuracies for train and test cohorts.
  """

  if seed is not None:
    logging.info("Setting random seed to %d.", seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)

  if True:
    dataset = dataset_utils.load_pickled_data(dataset_path)
    feature_ndims = dataset["train_split"]["sequence"].shape[-1]
    num_targets = dataset["train_split"]["label"].shape[-1]
    train_sequences, train_labels, train_mask = get_placeholder_variables(
        feature_ndims, num_targets)
    test_sequences, test_labels, test_mask = get_placeholder_variables(
        feature_ndims, num_targets)

  # Model and outputs.
  model_kwargs = dict(
      cell_type=cell_type,
      hidden_sizes=hidden_sizes,
      num_targets=num_targets,
      input_dropout_prob=input_dropout_prob,
      output_dropout_prob=output_dropout_prob,
      state_dropout_prob=state_dropout_prob,
      input_dims=feature_ndims
  )
  model = smart_module.SmartModuleExport(
      lambda: models.BinarySequenceClassifier(**model_kwargs))
  # Placeholders for dummy calls.
  step_sequences_ph = tf.placeholder(
      name="step_sequences", shape=(None, feature_ndims), dtype=tf.float32)
  initial_state_ph = nest.map_structure(
      lambda v: tf.placeholder(  # pylint: disable=g-long-lambda
          name=v.op.name, shape=[None] + v.shape[1:], dtype=v.dtype),
      model.initial_state(100))
  # dummy calls to SmartModule
  model(step_sequences_ph, initial_state_ph)
  model.step(step_sequences_ph, initial_state_ph)

  train_graph_vars = get_graph_variables(
      model,
      train_sequences,
      train_labels,
      l1_regularization_scale,
      train_mask,
      "train")
  test_graph_vars = get_graph_variables(
      model,
      test_sequences,
      test_labels,
      l1_regularization_scale,
      test_mask,
      "eval")

  if l1_regularization_scale:
    total_loss = (
        train_graph_vars.eval_loss + train_graph_vars.regularization_loss)
  else:
    total_loss = train_graph_vars.eval_loss

  # Optimizer and train op.
  with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_loss)

  state = type("DummyState", (object,), {"step": 0})

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train the model
    train_eval_losses = []
    train_reg_losses = []
    train_accuracies = []
    train_unmasked_counts = []
    train_predictions_list = []
    train_labels_list = []
    while state.step < num_train_steps:
      state.step += 1
      if True:
        dataset_batch = sample_batch(batch_size, dataset["train_split"])
        feed_dict = {
            train_sequences: dataset_batch["sequence"],
            train_labels: dataset_batch["label"],
            train_mask: dataset_batch["mask"]
        }
      _, train_graph_vals = sess.run(
          [train_op, train_graph_vars], feed_dict=feed_dict)
      train_eval_losses.append(train_graph_vals.eval_loss)
      train_reg_losses.append(train_graph_vals.regularization_loss)
      train_accuracies.append(train_graph_vals.accuracy)
      train_unmasked_counts.append(train_graph_vals.unmasked_count)
      train_predictions_list.append(train_graph_vals.predictions)
      train_labels_list.append(train_graph_vals.labels)

      if (state.step > 1 and ((state.step - 1) % logging_interval == 0) or
          state.step == num_train_steps):
        if True:
          feed_dict = {
              test_sequences: dataset["test_split"]["sequence"],
              test_labels: dataset["test_split"]["label"]
          }
          if ("mask" in dataset["test_split"]
              and dataset["test_split"]["mask"] is not None):
            feed_dict[test_mask] = dataset["test_split"]["mask"]
          else:
            feed_dict[test_mask] = (
                np.ones_like(dataset["test_split"]["label"]).astype(bool))
          test_graph_vals = sess.run(test_graph_vars, feed_dict=feed_dict)
        train_predictions_list = np.concatenate(train_predictions_list, axis=0)
        train_labels_list = np.concatenate(train_labels_list, axis=0)
        results = dict(
            step=state.step,
            train_eval_loss=np.mean(train_eval_losses),
            train_reg_loss=np.mean(train_reg_losses),
            train_accuracy=np.mean(train_accuracies),
            train_unmasked_count=np.mean(train_unmasked_counts),
            train_prauc=sklearn_metrics.average_precision_score(
                train_labels_list, train_predictions_list),
            train_incidence=(
                np.sum(train_labels_list)/np.size(train_labels_list)),
            test_eval_loss=test_graph_vals.eval_loss,
            test_reg_loss=test_graph_vals.regularization_loss,
            test_accuracy=test_graph_vals.accuracy,
            test_unmasked_count=test_graph_vals.unmasked_count,
            test_prauc=sklearn_metrics.average_precision_score(
                test_graph_vals.labels, test_graph_vals.predictions),
            test_incidence=(
                np.sum(test_graph_vals.labels)/np.size(test_graph_vals.labels)))
        logging.info(results)

        train_eval_losses = []
        train_reg_losses = []
        train_accuracies = []
        train_unmasked_counts = []
        train_predictions_list = []
        train_labels_list = []
    # Export SmartModule.
    export_path = os.path.join(checkpoint_base_path, "tfhub")
    model.export(export_path, sess)

  return results
