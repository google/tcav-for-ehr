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
"""TCAV Utils."""

import collections
import concurrent.futures
import functools
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import logging

import numpy as np
from sklearn import base
from sklearn import linear_model
from sklearn import model_selection
from sklearn import utils

import timing_utils


PermutationTestResult = collections.namedtuple(
    "PermutationTestResult",
    ["pvalue", "baseline_score", "baseline_score_std", "permuted_scores"])


def create_classifier(
    model_type: str,
    reg_coef: float = 1e-4,
    max_iter: int = 1000,
) -> base.BaseEstimator:
  """Create a classifier."""
  if model_type == "linear":
    lm = linear_model.SGDClassifier(
        loss="hinge", alpha=reg_coef, max_iter=max_iter,
        class_weight="balanced")
  elif model_type == "logistic":
    lm = linear_model.SGDClassifier(
        loss="log", alpha=reg_coef, max_iter=max_iter,
        class_weight="balanced")
  else:
    raise ValueError("Invalid model_type: {}".format(model_type))

  return lm


def create_classifier_cross_validated_reg_coef(
    model_type: str,
    reg_coefs: Optional[List[float]] = None,
    max_iter: int = 1000,
    penalty: str = "l2",
    early_stopping: Optional[bool] = None,
    n_iter_no_change: Optional[int] = None,
) -> base.BaseEstimator:
  """Create a linear classifier with a cross-validated regularization param."""
  loss = {
      "linear": "hinge",
      "logistic": "log",
  }

  if reg_coefs is None:
    reg_coefs = np.logspace(-4, 4, 9).tolist()

  param_grid = {
      "loss": [loss[model_type]],
      "max_iter": [max_iter],
      "alpha": reg_coefs,
      "class_weight": ["balanced"],
      "n_jobs": [-1],
      "penalty": [penalty],
  }
  if early_stopping is not None:
    param_grid["early_stopping"] = [early_stopping]
  if n_iter_no_change is not None:
    param_grid["n_iter_no_change"] = [n_iter_no_change]

  lm = model_selection.GridSearchCV(linear_model.SGDClassifier(),
                                    param_grid=param_grid)

  return lm


def _resample(
    dataset_size: int,
    num_train: int,
    num_samples: int,
) -> List[Tuple[List[int], List[int]]]:
  """Sample with replacement from range(dataset_size).

  Used for training classifiers on bootstrapped versions of the input dataset.

  Args:
    dataset_size: The number of samples in the dataset of interest.
    num_train: The number of samples in each training set.
    num_samples: The number of bootstrapped datasets (i.e train/test pairs) to
      be sampled.

  Returns:
    A list containing `num_samples` tuples. Each tuple contains two elements.
    The first element is the indices of the train set.
    The second element is the indices of the test set.
  """
  samples = []
  for _ in range(num_samples):
    train = np.random.choice(
        range(dataset_size), size=num_train, replace=True)
    test = utils.shuffle([k for k in range(dataset_size) if k not in set(train)
                         ])
    samples.append((train, test))
  return samples


def _sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))


def _bootstrap(
    xs_all: np.ndarray,
    ys_all: np.ndarray,
    bootstrap_indices: List[Tuple[List[int], List[int]]],
    model: base.BaseEstimator,
    metrics: List[Any],
    executor: concurrent.futures.Executor,
    permute_labels: bool = False,
) -> Tuple[List[List[Sequence[float]]], List[base.BaseEstimator]]:
  """Evaluate a model trained on bootstrapped versions of [X, y].

  Args:
    xs_all: The entire feature space. Shape [n_samples, n_features].
    ys_all: The corresponding binary labels. Shape [n_samples].
    bootstrap_indices: The indices of the train/test samples for each
      bootstrapped dataset. Typically obtained via `_resample`.
    model: The linear model to be trained.
    metrics: List of metrics to evaluate the performance of the trained models.
    executor: Number of worker threads.
    permute_labels: Whether to permute the labels to train the model.

  Returns:
    A tuple containing 2 elements:
      The score of the trained models.
      The trained models.
  """
  def fit_and_predict(train_idx, test_idx):
    xs_train, xs_test = xs_all[train_idx], xs_all[test_idx]
    ys_train, ys_test = ys_all[train_idx], ys_all[test_idx]

    new_model = base.clone(model)
    new_model.fit(
        xs_train, utils.shuffle(ys_train) if permute_labels else ys_train)
    ys_pred = _sigmoid(new_model.decision_function(xs_test))
    new_score = [metric(y_true=ys_test, y_pred=ys_pred) for metric in metrics]
    return new_score, new_model

  computations = []
  for train_idx, test_idx in bootstrap_indices:
    computations.append(executor.submit(fit_and_predict, train_idx, test_idx))
  results = [f.result() for f in computations]
  scores, models = zip(*results)

  return scores, models


def train_models_with_significance_test(
    xs: np.ndarray,
    ys: np.ndarray,
    model: base.BaseEstimator,
    metrics: List[Any],
    num_bootstrapped_datasets: int = 100,
    num_train_per_bootstrap: Optional[int] = None,
    num_permutations: int = 10,
    max_workers: Optional[int] = 16,
) -> Tuple[Dict[str, PermutationTestResult], List[base.BaseEstimator],
           List[base.BaseEstimator]]:
  """Train models and compute significance using a permutation test.

  The true models are trained using bootstrapped datasets.

  Using the same datasets, significance test is performed by permuting the
  labels, then training models using the same bootstrapped datasets,
  repeated num_permutations times.

  For each metric, the p-value of the score for the true models is computed by
  comparing how many of the models trained using permuted labels were able to
  achieve scores at least as extreme.

  Args:
    xs: An array of shape (n_samples, n_features) input features.
    ys: An array of shape (n_samples,) containing target labels.
    model: A sklearn model with fit and predict methods.
    metrics: A list of metrics from sklearn.metrics
    num_bootstrapped_datasets: Number of bootstrapped datasets to be sampled
      from (xs, ys).
    num_train_per_bootstrap: If specified, the number of samples to be used in
      each training bootstrapped dataset. If nothing is specified, `len(xs)`
      will be used. The rest is used for testing.
    num_permutations: Number of permutations used by significance test.
    max_workers: Number of worker threads.

  Returns:
    results: A list of PermutationTestResult.
    models: A list of all models produced on bootstrapped datasets using the
      true labels. It has `num_bootstrapped_datasets` elements.
    permuted models: A list of models trained on permuted labels. It has
      `num_permutations * num_bootstrapped_datasets` elements.
  """
  if not metrics:
    raise ValueError(
        "You should provide a list of metrics for evaluating the linear "
        "classifiers but the provided list was empty.")
  executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

  if num_train_per_bootstrap is None:
    num_train_per_bootstrap = len(xs)
  bootstrap_indices = _resample(
      len(xs), num_train_per_bootstrap, num_bootstrapped_datasets)

  bootstrap_fn = functools.partial(
      _bootstrap,
      xs_all=xs,
      bootstrap_indices=bootstrap_indices,
      model=model,
      metrics=metrics,
      executor=executor)

  # Train and evaluate on "true" labels.
  with timing_utils.time_this("training_true_models", _print_and_flush):
    model_scores, models = bootstrap_fn(ys_all=ys, permute_labels=False)

  # Compute "true" model scores.
  model_scores = list(zip(*model_scores))
  model_scores_mean = np.mean(model_scores, axis=-1)
  model_scores_std = np.std(model_scores, axis=-1)
  for ind, m in enumerate(metrics):
    logging.info(
        "%s: %0.3f (std=%0.3f, std/sqrt(%d)=%0.3f)",
        m.__name__,
        model_scores_mean[ind],
        model_scores_std[ind],
        len(models),
        model_scores_std[ind] / np.sqrt(len(models)),
    )

  # Train and evaluate on "permuted" labels
  permuted_scores = []
  permuted_models = []
  with timing_utils.time_this("training_permutation_models", _print_and_flush):
    for _ in range(1, num_permutations + 1):
      scores, perm_models = bootstrap_fn(ys_all=ys, permute_labels=True)
      permuted_scores.append(scores)
      permuted_models += perm_models

  # Concatenate over bootstraps and reshape over metrics.
  permuted_scores = np.reshape(
      np.array(permuted_scores),
      (num_permutations * num_bootstrapped_datasets, len(metrics)))
  permuted_scores = list(zip(*permuted_scores))

  # Significance test
  results = {}
  for res in zip(metrics, model_scores_mean, model_scores_std, permuted_scores):
    metric, model_score, model_score_std, permuted_score = res
    num_extremes = np.sum(np.array(permuted_score) >= model_score)
    pvalue = (num_extremes + 1) / float(len(permuted_score) + 1)
    results[metric.__name__] = PermutationTestResult(
        pvalue=pvalue,
        baseline_score=model_score,
        baseline_score_std=model_score_std,
        permuted_scores=permuted_score,
    )

  return results, models, permuted_models


def compute_cav(models: List[base.BaseEstimator]) -> Tuple[np.ndarray,
                                                           List[np.ndarray]]:
  """Compute Concept Activation Vector."""
  concept_vecs = []
  for model in models:
    cvec = model.coef_[0]  # pytype: disable=attribute-error
    cvec = cvec / np.linalg.norm(cvec)
    concept_vecs.append(cvec)

  mean_concept_vec = sum(concept_vecs)
  mean_concept_vec = mean_concept_vec / np.linalg.norm(mean_concept_vec)
  return mean_concept_vec, concept_vecs


def testing_with_cav(concept_vectors: Sequence[np.ndarray],
                     layer_gradients: np.ndarray,
                     normalise_gradients: bool = False,
                     near_zero: float = 1e-12) -> Tuple[float, np.ndarray]:
  """Testing with Concept Activation Vectors.

  Args:
    concept_vectors: A list or 2D array (N x D) of concept activation vectors.
    layer_gradients: A list or 2D array (M x D) of gradient/activations vectors.
    normalise_gradients: Whether the gradients should be normalised before
      computing CAV alignments and TCAV scores.
    near_zero: A near zero value to prevent division by zero when computing
      gradient norms.

  Returns:
    A tuple with 2 elements: a scalar TCAV score and the CAV alignments (N x M).
  """
  if normalise_gradients:
    layer_norm = np.linalg.norm(layer_gradients, axis=-1)
    layer_norm[layer_norm < near_zero] = 1.0  # Avoid division by zero.
    layer_gradients = layer_gradients / np.expand_dims(layer_norm, axis=-1)
  cav_alignments = np.matmul(layer_gradients, np.array(concept_vectors).T)
  tcav_score = np.mean(cav_alignments > 0)
  return tcav_score, cav_alignments


def _print_and_flush(*args, **kwargs) -> None:
  """This is useful for working around buffering in notebooks."""
  print(*args, **kwargs)
  sys.stdout.flush()
