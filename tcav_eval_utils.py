# Lint as: python3
"""Utils functions for TCAV colabs."""
import collections
import enum
import functools
from typing import Any, Collection, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from absl import logging
import numpy as np
import sklearn
from sklearn.base import BaseEstimator

import model_utils
import tcav_utils


ALIGNMENT_MODES = [
    "gradients",
    "activation_diff_with_fixed_lag",
    "activation_diff_with_fixed_lag_smoothed",
    "activation_diff_baseline",
]


class CavType:
  BOOTSTRAP = "bootstrap"
  PERMUTED = "permuted"


@enum.unique
class CavClassifierMode(enum.Enum):
  """Contains the modes currently supported for training CAV classifiers.

  t1_only: CAVs are trained only on model activations at the last timestamp of
    the concept trajectory.
  t0_to_t1: CAVS are trained on model activations from all timestamps within the
    concept trajectory.
  t0_to_t1_diff: CAVS are trained on the difference in activations between the
    first and last timestamp of the concept trajectory.
  """
  T1_ONLY = "t1_only"
  T0_TO_T1 = "t0_to_t1"
  T0_TO_T1_DIFF = "t0_to_t1_diff"


WriterType = Any


def _compute_window_of_interest_for_classifier(
    classifier_mode: CavClassifierMode,
    t1: List[int],
    t0: Optional[List[int]] = None,
    classifier_step: int = 1) -> List[List[int]]:
  """Computes window of interest given classifier mode."""
  window_of_interest = []
  for idx in range(len(t1)):
    if classifier_mode == CavClassifierMode.T1_ONLY:
      window_of_interest.append([t1[idx]])
    elif classifier_mode == CavClassifierMode.T0_TO_T1:
      window = list(range(t1[idx], t0[idx], -classifier_step))
      window_of_interest.append(window[::-1])
    elif classifier_mode == CavClassifierMode.T0_TO_T1_DIFF:
      window_of_interest.append([t0[idx], t1[idx]])
    else:
      raise ValueError(f"classifier_mode arg {classifier_mode} not recognized.")
  return window_of_interest


def build_linear_classifier_per_layer(
    xs: np.ndarray,
    ys: np.ndarray,
    metrics: List[Any],
    model_type: str = "linear",
    reg_coef: float = 1e-4,
    num_bootstrapped_datasets: int = 100,
    num_permutations: int = 10,
    cross_val_reg: bool = True,
    early_stopping: bool = False
    ) -> Tuple[
        Dict[str, tcav_utils.PermutationTestResult],
        List[BaseEstimator],
        List[BaseEstimator]]:
  """Builds a linear classifier based on an array of activations."""
  shuffled_xs, shuffled_ys = sklearn.utils.shuffle(xs, ys)
  # Train models.
  if cross_val_reg:
    base_model = tcav_utils.create_classifier_cross_validated_reg_coef(
        model_type,
        early_stopping=early_stopping)
  else:
    base_model = tcav_utils.create_classifier(model_type, reg_coef)
  results, models, perm_models = tcav_utils.train_models_with_significance_test(
      shuffled_xs,
      shuffled_ys,
      base_model,
      metrics=metrics,
      num_bootstrapped_datasets=num_bootstrapped_datasets,
      num_permutations=num_permutations)
  if cross_val_reg:
    return (
        results,
        [m.best_estimator_ for m in models],  # pytype: disable=attribute-error
        [m.best_estimator_ for m in perm_models],  # pytype: disable=attribute-error
    )
  else:
    return results, models, perm_models


def get_results_writer(writer_description: str) -> WriterType:
  """Returns a file-like object to write train/eval output to."""
  import sys
  writer = sys.stdout
  writer._write = writer.write

  def write_wrapper(self, results):
    return self._write(str(results))

  writer.write = write_wrapper
  writer.close = lambda: None
  return writer


def create_standard_cav_training_dataset_synthetic(
    model_path: str, data_split: Dict[str, Any], label_names: Sequence[str],
    concept_names: Sequence[str], ntrains: int,
    changepoint_lookahead: int, changepoint_lookback: int,
    classifier_mode: CavClassifierMode, layer: Optional[int] = None,
    classifier_step: int = 1,
    accuracy_threshold: Optional[float] = None,
    accuracy_threshold_label_name: Optional[str] = None
    ) -> Mapping[str, Tuple[List[np.ndarray], np.ndarray]]:
  """Generates data required for training cavs.

  This is the standard strategy applied to the dense synthetic dataset. It
  involves selecting the first ntrains data elements, filtered by an accuracy
  threshold, and collecting t0 and t1 as a window range around concept
  changepoints.

  Args:
    model_path: Path to model checkpoint.
    data_split: Mapping from data split fields to corresponding arrays.
    label_names: Sequence of label string names.
    concept_names: Sequence of concept string names.
    ntrains: Number of train elements to collect.
    changepoint_lookahead: Number of timesteps after changepoint to call t1.
    changepoint_lookback: Number of timesteps before changepoint to call t0.
    classifier_mode: Training data mode for CAV classifiers. Options are
      't1_only', 't0_to_t1', & 't0_to_t1_diff'.
    layer: Index of the layer for which to compute ICS.
    classifier_step: Determines the step size between training examples when
      using classifier mode 't0_to_t1'.
    accuracy_threshold: Threshold of accuracy, above which to include training
      elements.
    accuracy_threshold_label_name: Which label to use for computing accuracy.

  Returns:
    A mapping from concept name to a tuple of training examples. The first
      element of the tuple is a list, with each element being an array of
      training activations of shape [num_training_examples, act_dim] from a
      different model layer. The second element is an array of length
      [num_training_examples,]
  """
  model_output = model_utils.unroll_model_per_step(
      model_path,
      data_split["sequence"],
      data_split["label"],
      use_sigmoid_gradients=True)
  concept_names_to_seqs = {}
  concept_names_to_times = {}
  for i, concept_name in enumerate(concept_names):
    concept_names_to_seqs[concept_name] = (
        data_split["concept_sequence"][:, :, i])
    changes = data_split["changes"][:, i]
    concept_names_to_times[concept_name] = (
        (changes - changepoint_lookback), (changes + changepoint_lookahead))
  activations = model_output["activations"]
  if layer is not None:
    activations = [activations[layer]]
  max_time_idx = data_split["label"].shape[0] - 1
  if accuracy_threshold is not None:
    if accuracy_threshold_label_name is None:
      raise ValueError("Provided threshold accuracy but no label name "
                       "to compute accuracy for.")
    label_idx = label_names.index(accuracy_threshold_label_name)
    logging.info("Filtering by accuracy...")
    preds = model_output["predictions"]
    logging.info("Predictions shape:")
    logging.info(preds.shape)  # pytype: disable=attribute-error
    logging.info("Label shape:")
    logging.info(data_split["label"].shape)
    correct_preds = np.equal(
        preds[..., label_idx], data_split["label"][..., label_idx])
    accuracy = (
        np.sum(correct_preds.astype(int), axis=0) / data_split["label"].shape[0]
        )
    logging.info("Accuracy shape:")
    logging.info(accuracy.shape)
    accuracy_filter = np.squeeze(accuracy >= accuracy_threshold)
    logging.info("Keeping %d out of %d examples.",
                 np.sum(accuracy_filter), accuracy_filter.shape[0])
    activations = [
        act[:, accuracy_filter, :] for act in activations]
    concept_names_to_seqs = {
        name: seq[:, accuracy_filter]
        for name, seq in concept_names_to_seqs.items()}
    concept_names_to_times = {
        name: (t0[accuracy_filter], t1[accuracy_filter])
        for name, (t0, t1) in concept_names_to_times.items()}
  activations = [act[:, :ntrains, :] for act in activations]
  concept_names_to_seqs = {
      name: seq[:, :ntrains] for name, seq in concept_names_to_seqs.items()}
  concept_names_to_times = {
      name: (list(np.clip(t0[:ntrains], a_min=0, a_max=max_time_idx)),
             list(np.clip(t1[:ntrains], a_min=0, a_max=max_time_idx)))
      for name, (t0, t1) in concept_names_to_times.items()}
  concept_name_to_training_data = {}
  for concept_name in concept_names:
    t0, t1 = concept_names_to_times[concept_name]
    labels = concept_names_to_seqs[concept_name]
    window_of_interest = _compute_window_of_interest_for_classifier(
        classifier_mode=classifier_mode,
        t1=t1,
        t0=t0,
        classifier_step=classifier_step)
    ys = [labels[window_of_interest[idx], idx]
          for idx in range(len(window_of_interest))]
    if classifier_mode == CavClassifierMode.T0_TO_T1_DIFF:
      # We take the difference between 2 points as xs, and the last label as ys.
      ys = [y[-1:] for y in ys]
    ys = np.concatenate(ys)
    xs_layers = []
    for acts in activations:
      xs = [acts[window_of_interest[idx], idx, :]
            for idx in range(len(window_of_interest))]
      if classifier_mode == CavClassifierMode.T0_TO_T1_DIFF:
        # We take the difference between the points as xs.
        xs = [np.diff(x, axis=0) for x in xs]
      xs = np.concatenate(xs, axis=0)
      xs_layers.append(xs)
    concept_name_to_training_data[concept_name] = xs_layers, ys
  return concept_name_to_training_data


def train_classifiers(
    concept_to_training_data: Mapping[str, Tuple[List[np.ndarray], np.ndarray]],
    metrics: List[Any],
    model_type: str = "linear",
    reg_coef: float = 1e-4,
    num_bootstrapped_datasets: int = 100,
    num_permutations: int = 10,
    cross_val_reg: bool = True,
    early_stopping: bool = False
) -> Tuple[Dict[str, Dict[int, List[BaseEstimator]]], Dict[str, Dict[
    int, List[BaseEstimator]]], Dict[str, Dict[int, Dict[
        str, tcav_utils.PermutationTestResult]]]]:
  """Train CAV classifiers for each concept and layer."""
  trained_models = {}
  permuted_models = {}
  metric_results = {}
  for concept_name, (xs_layers, ys) in concept_to_training_data.items():
    trained_models[concept_name] = {}
    permuted_models[concept_name] = {}
    metric_results[concept_name] = {}
    for layer, xs in enumerate(xs_layers):
      results, models, perm_models = build_linear_classifier_per_layer(
          xs,
          ys,
          metrics=metrics,
          model_type=model_type,
          reg_coef=reg_coef,
          num_bootstrapped_datasets=num_bootstrapped_datasets,
          num_permutations=num_permutations,
          cross_val_reg=cross_val_reg,
          early_stopping=early_stopping)
      trained_models[concept_name][layer] = models
      permuted_models[concept_name][layer] = perm_models
      metric_results[concept_name][layer] = results
  return trained_models, permuted_models, metric_results


def evaluate_tcav(linear_classifiers: List[BaseEstimator],
                  model_directions: np.ndarray) -> np.ndarray:
  """Computes CAV from linear classifiers and estimates alignments."""

  _, concept_vectors = tcav_utils.compute_cav(linear_classifiers)
  _, cav_alignments = tcav_utils.testing_with_cav(
      concept_vectors, model_directions, normalise_gradients=True)
  return cav_alignments


def _write_training_results(
    results: Dict[str, Dict[int, Dict[str, tcav_utils.PermutationTestResult]]],
    writer: WriterType) -> None:
  """Logs the CAV training results in a writer."""
  for concept_name, concept_results in results.items():
    for layer, layer_results in concept_results.items():
      results_row = dict(
          concept=concept_name,
          layer=layer,
      )
      for metric_name, metric_result in layer_results.items():
        pvalue, score_avg, score_std, permuted_scores = metric_result
        worst_num_extremes = np.sum(
            np.array(permuted_scores) >= (score_avg - score_std))
        worst_pvalue = ((worst_num_extremes + 1) /
                        float(len(permuted_scores) + 1))
        results_row.update({
            metric_name + "_avg": score_avg,
            metric_name + "_std": score_std,
            metric_name + "_pvalue": pvalue,
            metric_name + "_pvalue_worst": worst_pvalue,
        })
      logging.info(results_row)
      writer.write(results_row)
      writer.flush()


def _evaluate_cav_and_write(activations: np.ndarray,
                            labels: np.ndarray,
                            models: List[BaseEstimator],
                            writer: WriterType,
                            concept_name: str,
                            layer: int,
                            filter_eval: np.ndarray = None) -> None:
  """Evaluates CAV performance on activations and saves results."""
  # Flatten everything
  flat_acts = np.reshape(activations, [-1, activations.shape[-1]])
  flat_concepts = np.reshape(labels, [-1])
  if filter_eval is not None:
    flat_filter_eval = np.reshape(filter_eval, [-1])

  predictions = [m.predict(flat_acts) for m in models]  # pytype: disable=attribute-error

  # eval only
  accs_eval = [sklearn.metrics.accuracy_score(flat_concepts[flat_filter_eval],
                                              preds[flat_filter_eval])
               for preds in predictions]

  # concept present
  flat_filter_eval_pos = np.logical_and(flat_filter_eval, flat_concepts > 0.5)
  accs_pos = [
      sklearn.metrics.accuracy_score(flat_concepts[flat_filter_eval_pos],
                                     preds[flat_filter_eval_pos])
      for preds in predictions]

  # concept absent
  flat_filter_eval_neg = np.logical_and(flat_filter_eval, flat_concepts < 0.5)
  accs_neg = [
      sklearn.metrics.accuracy_score(flat_concepts[flat_filter_eval_neg],
                                     preds[flat_filter_eval_neg])
      for preds in predictions]

  results_row = dict(
      concept=concept_name,
      layer=layer,
      accs_avg=np.mean(accs_eval),
      accs_std=np.std(accs_eval),
      accs_pos_avg=np.mean(accs_pos),
      accs_pos_std=np.std(accs_pos),
      accs_neg_avg=np.mean(accs_neg),
      accs_neg_std=np.std(accs_neg),
  )
  logging.info(results_row)
  writer.write(results_row)
  writer.flush()


def extract_model_direction(alignment_mode: str,
                            activations: np.ndarray,
                            gradients: np.ndarray,
                            baseline_activations: np.ndarray,
                            alignment_diff: int = 1) -> np.ndarray:
  """Extracts which vectors to compute cosine similarity with CAV."""

  if alignment_mode == "gradients":
    model_vectors = gradients.copy()
  elif alignment_mode == "activation_diff_with_fixed_lag":
    nsteps = activations.shape[0]
    source_times = range(-alignment_diff, nsteps - alignment_diff)
    source_times = np.maximum(source_times, 0)
    model_vectors = activations - activations[source_times, ...]
  elif alignment_mode == "activation_diff_with_fixed_lag_smoothed":
    nsteps = activations.shape[0]
    source_times = range(-alignment_diff, nsteps - alignment_diff)
    source_times = np.maximum(source_times, 0)
    smoothed_acts = _exponential_moving_average(activations, ratio=0.8)
    model_vectors = activations - smoothed_acts[source_times, ...]
  elif alignment_mode == "activation_diff_baseline":
    model_vectors = activations - baseline_activations
  return model_vectors


def analyse_alignments(cav_alignments: np.ndarray,
                       concept_labels: np.ndarray,
                       target_labels: np.ndarray,
                       time_of_interests: List[List[Tuple[Any, Any]]],
                       writer: WriterType,
                       concept_name: str,
                       layer: int = None,
                       eval_mask: np.ndarray = None) -> None:
  """Analyse CAV alignment results at specific time points t0 and t1."""

  for label_val in [0, 1]:  # For each label value
    for concept_val in [0, 1]:  # For each concept value (i.e. present/absent)
      per_sample_filter = np.logical_and(target_labels == label_val,
                                         concept_labels == concept_val)
      if eval_mask:
        per_sample_filter = np.logical_and(per_sample_filter, eval_mask)

      results_row = dict(
          concept_name=concept_name,
          layer=layer,
          label=(label_val == 1),
          concept=(concept_val == 1),
          count=np.sum(per_sample_filter))
      for toi in time_of_interests:  # For each time or window of interest
        cav_at_toi_filtered = _extract_alignments(
            cav_alignments[:, per_sample_filter],
            np.array(toi)[per_sample_filter].tolist())
        if isinstance(toi[0], collections.Sequence):
          toi_name = "_".join(str(toi[0][x] for x in toi[0]))
        else:
          toi_name = str(toi)
        results_row.update({
            "TCAV_AVG_{}".format(toi_name): np.mean(cav_at_toi_filtered > 0),
            "TCAV_STD_{}".format(toi_name): np.std(cav_at_toi_filtered > 0),
            "CAV_AVG_{}".format(toi_name): np.mean(cav_at_toi_filtered),
            "CAV_STD_{}".format(toi_name): np.std(cav_at_toi_filtered),
        })
        writer.write(results_row)
        writer.flush()


def _extract_alignments(alignments: np.ndarray,
                        time_of_interest: List[Any]) -> np.ndarray:
  """Extract alignments over periods or times of interest.

  Args:
    alignments: np.ndarray of [num_time_steps, num_samples, num_features] of CAV
      alignment similarity scores.
    time_of_interest: List of times or windows to compute alignment at. Should
      have length num_samples and be either an integer index in [0,
      num_time_steps] or a tuple of integers to estimate alignment over a window
      rather than specific time steps.

  Returns:
    Array of CAV alignment similarity scores at or between times of interest.
  """

  if alignments.shape[1] != len(time_of_interest):
    raise ValueError("alignment.shape={}, len(time_of_interest)={}".format(
        alignments.shape, len(time_of_interest)))

  _, num_seqs, _ = alignments.shape

  if time_of_interest:
    if isinstance(time_of_interest[0], collections.Sequence):
      if len(time_of_interest[0]) != 2:
        raise ValueError(
            "Expected an int or a tuple of two ints for time of interest, found {}"
            .format(time_of_interest))
      cav_at_toi = []
      for idx in range(num_seqs):
        cav_at_toi.append(
            alignments[time_of_interest[idx][0]:time_of_interest[idx][1], idx])
      if cav_at_toi:
        cav_at_toi = np.concatenate(cav_at_toi, axis=0)
      else:
        cav_at_toi = np.array(cav_at_toi)
    else:
      cav_at_toi = np.array(
          [alignments[time_of_interest[idx], idx] for idx in range(num_seqs)])
  else:
    cav_at_toi = np.array([])

  return cav_at_toi


def _exponential_moving_average(array: np.ndarray, ratio: float) -> np.ndarray:
  averages = array.copy()
  for t in range(1, array.shape[0]):
    averages[t, ...] = (
        ratio * averages[t - 1, ...] + (1 - ratio) * array[t, ...])
  return averages
