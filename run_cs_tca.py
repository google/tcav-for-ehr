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
"""Script for running CS/tCA eval."""
import collections
import itertools
import os
import time
from typing import List, Mapping, Optional
from absl import flags
from absl import logging
import numpy as np

import dataset_utils
import model_utils
import tcav_eval_utils
from absl import app

flags.DEFINE_string("dataset_path", None, "Path to the data.")
flags.DEFINE_string("model_path", None, "Path for trained model restore. May"
                    " end with 'tfhub'.")
flags.DEFINE_string("cavs_path", None, "Path to the pickled cavs file.")
flags.DEFINE_string("output_dir", None, "Dir to result output.")
flags.DEFINE_multi_integer(
    "targets", None, "Indices of the target for which to compute cs/tca.")
flags.DEFINE_multi_integer(
    "layers", None, "Which model layers to compute cs/tca for. They are "
    "indexed starting at 0 (the first hidden layer. See "
    "`models.BinarySequenceClassifier.step_from_state`).")
flags.DEFINE_integer(
    "synthetic_n_eval", None, "Number of data elements to use in eval dataset "
    "for synthetic data. Does not need to be provided if creating a different "
    "dataset type.")

FLAGS = flags.FLAGS


def _get_output_subdir(
    output_dir: str,
    target: int,
    concept: str,
    layer: int,
) -> str:
  return os.path.join(output_dir, f"target={target}", f"concept={concept}",
                      f"layer={layer}")


def compute_cs(unit_cav: np.ndarray, grads: np.ndarray):
  """Compute the conceptual sensitivity as defined by the original TCAV paper."""
  grads = grads.squeeze()
  assert len(unit_cav.shape) == 1
  assert len(grads.shape) == 3
  assert grads.shape[-1] == unit_cav.shape[0]
  return np.sum(grads * unit_cav, -1)


def _exponential_moving_average(array: np.ndarray, ratio: float) -> np.ndarray:
  averages = array.copy()
  for t in range(1, array.shape[0]):
    averages[t, ...] = (
        ratio * averages[t - 1, ...] + (1 - ratio) * array[t, ...])
  return averages


def compute_tca(unit_cav: np.ndarray,
                activations: np.ndarray,
                lag: int = 25,
                normalized: bool = True,
                smoothed: bool = False) -> np.ndarray:
  """Compute temporal Concept Angle.

  This quantity is defined in Kim et al., 2018 for a lag of 0, and in
  Baur et al., 2020.

  Args:
    unit_cav: A unit concept activation vector.
    activations: A unroll_length x batch_size x num_cavs array.
    lag: Difference in activations at time t is computed between t-lag and t.
    normalized: Whether to normalize the differece in activations.
    smoothed: Whether to smooth the activations at t-lags.

  Returns:
    The tCA scores.
  """
  nsteps = activations.shape[0]
  # Replicate first time step until lag is reached
  source_times = range(-lag, nsteps - lag)
  source_times = np.maximum(source_times, 0)
  if smoothed:
    smoothed_acts = _exponential_moving_average(activations, ratio=0.8)
  else:
    smoothed_acts = activations
  activations = activations - smoothed_acts[source_times, ...]
  if normalized:
    layer_norm = np.linalg.norm(activations, axis=-1)
    layer_norm[layer_norm < 1e-12] = 1.0  # Avoid division by zero.
    activations /= np.expand_dims(layer_norm, axis=-1)
  # activations /= act_norms
  assert len(unit_cav.shape) == 1
  assert len(activations.shape) == 3
  assert activations.shape[-1] == unit_cav.shape[0]
  return np.sum(activations * unit_cav, -1)


def run_cs_tca(
    dataset_path: str,
    model_path: str,
    cavs_path: str,
    targets: List[int],
    layers: List[int],
    output_dir: str,
    dataset_type: dataset_utils.DatasetType,
    synthetic_n_eval: Optional[int] = None,
) -> None:
  """Runs and saves CS and tCA on the given dataset with the given model.

  Args:
    dataset_path: Where the dataset is saved.
    model_path: Where the trained model is saved.
    cavs_path: Where the cavs are saved.
    targets: Indices of the target for which to compute cs/tca.
    layers: Which model layers to compute cs/tca for. 0 is the first hidden
      layer.
    output_dir: Where to save the output.
    dataset_type: The type of dataset.
    synthetic_n_eval: How many samples to use for running eval. Takes first
      N values from test dataset split. Does not need to be provided if creating
      a different dataset type.
  """
  logging.info("Loading CAVs from %s...", cavs_path)
  all_cavs = dataset_utils.load_pickled_data(cavs_path)
  cavs_metrics_path = os.path.join(
      os.path.dirname(cavs_path), "cav_metrics.pkl")
  if os.path.exists(cavs_metrics_path):
    logging.info("Loading CAV Metrics from %s...", cavs_path)
    all_cav_metrics = dataset_utils.load_pickled_data(cavs_metrics_path)
    # logging.info("CAV metrics: %s", all_cav_metrics)
    for concept_name, concept_metrics in all_cav_metrics.items():
      for target_name, target_metrics in concept_metrics.items():
        for metric_name, metrics in target_metrics.items():
          logging.info("%s %s %s: baseline=%s, pvalue=%s", concept_name,
                       target_name, metric_name, metrics.baseline_score,
                       metrics.pvalue)
  # Get CAVs in backward compatible way.
  if tcav_eval_utils.CavType.BOOTSTRAP in all_cavs:
    all_bootstrap_cavs = all_cavs[tcav_eval_utils.CavType.BOOTSTRAP]
    all_permuted_cavs = all_cavs[tcav_eval_utils.CavType.PERMUTED]
  else:
    all_bootstrap_cavs = all_cavs["CAV"]
    all_permuted_cavs = all_cavs["permuted_CAV"]
  if dataset_type == dataset_utils.DatasetType.STANDARD_SYNTHETIC:
    logging.info("Loading synthetic data for CS/TCA eval...")
    data_loading_start_time = time.time()
    dataset = dataset_utils.load_pickled_data(dataset_path)
    eval_sequence = dataset["test_split"]["sequence"][:, :synthetic_n_eval, :]
    eval_label = dataset["test_split"]["label"][:, :synthetic_n_eval, :]
    logging.info("Synthetic data loading time: %fs",
                 time.time() - data_loading_start_time)
  else:
    raise ValueError(f"dataset_type arg {dataset_type} not recognized.")
  logging.info("Creating activations for eval data...")
  eval_activations_start = time.time()
  eval_model_output = model_utils.unroll_model_per_step(
      model_path,
      eval_sequence,
      eval_label,
      use_sigmoid_gradients=True)

  eval_activations = eval_model_output["activations"]
  eval_gradients = eval_model_output["gradients"]
  logging.info(
      "Eval data activations time: %fs.", time.time() - eval_activations_start)

  logging.info("Computing CS and tCA on eval dataset...")
  cs_tca_start = time.time()
  concept_names = list(all_bootstrap_cavs.keys())
  for concept_name, target, layer in itertools.product(
      concept_names, targets, layers):
    logging.info("Concept name: %s, Target: %d, Layer: %d",
                 concept_name, target, layer)
    output_subdir = _get_output_subdir(output_dir, target, concept_name, layer)
    results = collections.defaultdict(dict)
    bootstrap_cavs = all_bootstrap_cavs[concept_name][layer]
    permuted_cavs = all_permuted_cavs[concept_name][layer]
    if len(bootstrap_cavs.shape) != 2 or len(permuted_cavs.shape) != 2:
      raise ValueError(
          "CAV arrays must have rank 2. Bootstrap rank: "
          f"{len(bootstrap_cavs.shape)}, permuted rank: "
          f"{len(permuted_cavs.shape)}.")
    unit_bootstrap_cavs = (
        bootstrap_cavs / np.linalg.norm(bootstrap_cavs, axis=1, keepdims=True))
    unit_permuted_cavs = (
        permuted_cavs / np.linalg.norm(permuted_cavs, axis=1, keepdims=True))
    # Compute CS.
    # The indexing for gradients is a bit weird.
    cs_scores = [
        compute_cs(unit_cav, eval_gradients[target][:, layer, ...])
        for unit_cav in unit_bootstrap_cavs
    ]
    # T x n_test x n_CAVs
    results["bootstrap_CS"] = np.stack(cs_scores, axis=-1)
    perm_cs_scores = [
        compute_cs(unit_cav, eval_gradients[target][:, layer, ...])
        for unit_cav in unit_permuted_cavs
    ]
    # T x n_test x n_CAVs
    results["permuted_CS"] = np.stack(perm_cs_scores, axis=-1)
    if not os.path.exists(os.path.dirname(output_subdir)):
      logging.info("Creating output_dir %s", output_subdir)
      os.makedirs(os.path.dirname(output_subdir))
    cs_results_path = os.path.join(output_subdir, "cs_results.pkl")
    logging.info("Saving CS results to %s...", cs_results_path)
    dataset_utils.save_pickled_data(dict(results), cs_results_path)

    # Compute tCA.
    results = collections.defaultdict(dict)
    tca_scores = [
        compute_tca(unit_cav, eval_activations[layer])
        for unit_cav in unit_bootstrap_cavs
    ]
    # T x n_test x n_CAVs
    results["bootstrap_tCA"] = np.stack(tca_scores, axis=-1)
    perm_tca_scores = [
        compute_tca(unit_cav, eval_activations[layer])
        for unit_cav in unit_permuted_cavs
    ]
    # T x n_test x n_CAVs
    results["permuted_tCA"] = np.stack(perm_tca_scores, axis=-1)
    tca_results_path = os.path.join(output_subdir, "tca_results.pkl")
    logging.info("Saving tCA results to %s...", tca_results_path)
    dataset_utils.save_pickled_data(dict(results), tca_results_path)
  logging.info("CS and tCA computation time: %fs", time.time() - cs_tca_start)


def main(argv):
  del argv  # unused
  flags.mark_flags_as_required([
      "dataset_path",
      "model_path",
      "cavs_path",
      "output_dir",
      "targets",
      "layers",
      "synthetic_n_eval"
  ])
  dataset_type = dataset_utils.DatasetType.STANDARD_SYNTHETIC
  run_cs_tca(
      dataset_path=FLAGS.dataset_path,
      model_path=FLAGS.model_path,
      cavs_path=FLAGS.cavs_path,
      targets=FLAGS.targets,
      layers=FLAGS.layers,
      output_dir=FLAGS.output_dir,
      synthetic_n_eval=FLAGS.synthetic_n_eval,
      dataset_type=dataset_type,
  )


if __name__ == "__main__":
  app.run(main)
