# Lint as: python3
"""Util functions for generating synthetic datasets."""
import collections
import copy
import enum
import os
from typing import Any, Dict, List, MutableMapping, Optional, Tuple
from absl import logging
import attr
import dill
import numpy as np
import tensorflow.compat.v1 as tf
import dataset_specs
import feature_scaling_utils



def get_dataset_id(dataset_path: str) -> str:
  """Extracts an id from the dataset path, used for naming related files."""
  dataset_id = os.path.basename(dataset_path).split(".")[0]
  return dataset_id


def load_pickled_data(path: str) -> Dict[str, Any]:
  """Loads pickled data from disk."""
  with open(path, "rb") as f:
    data = dill.load(f)
  return data


def save_pickled_data(data: MutableMapping[str, Any],
                      output_filepath: str) -> None:
  """Saves pickled data to disk."""
  if not os.path.exists(os.path.dirname(output_filepath)):
    logging.warning(
        "The output directory does not exist and will be created: "
        "%s.", os.path.dirname(output_filepath))
    os.makedirs(os.path.dirname(output_filepath))
  with open(output_filepath, "wb") as fout:
    dill.dump(data, fout)


@enum.unique
class DatasetType(enum.Enum):
  """Contains the dataset types that can be used for informativeness work."""
  STANDARD_SYNTHETIC = "standard_synthetic"


@attr.s(auto_attribs=True)
class DataSplit:
  """Contains arrays for a time series data split."""
  sequence: Optional[np.ndarray] = None
  label: Optional[np.ndarray] = None
  concept: Optional[np.ndarray] = None
  changes: Optional[np.ndarray] = None
  concept_sequence: Optional[np.ndarray] = None
  features: Optional[np.ndarray] = None
  mask: Optional[np.ndarray] = None


@attr.s(auto_attribs=True)
class Dataset:
  """Contains data splits and config comprising a time series dataset."""
  train_split: Optional[DataSplit] = None
  valid_split: Optional[DataSplit] = None
  test_split: Optional[DataSplit] = None
  config: Optional[Dict[str, Any]] = None


def generate_concept_sequence(
    num_unroll: int, concept_indicators: np.ndarray,
    concept_changepoints: np.ndarray) -> np.ndarray:
  """Creates array reflecting the presence of concepts in samples.

  Args:
    num_unroll: Number of timesteps in time series.
    concept_indicators: Bool array of concept indicators of shape
      [batch_size, num_concepts].
    concept_changepoints: Int array of concept changepoints of shape
      [batch_size, num_concepts].

  Returns:
    An array of shape [num_unroll, batch_size, num_concepts]
  """
  batch_size, num_concepts = concept_indicators.shape
  concept_sequence = np.zeros(shape=(num_unroll, batch_size, num_concepts))
  for batch_idx in range(batch_size):
    for concept_idx in range(num_concepts):
      concept_indicator = concept_indicators[batch_idx, concept_idx]
      concept_cp = concept_changepoints[batch_idx, concept_idx]
      concept_sequence[concept_cp:, batch_idx, concept_idx] = int(
          concept_indicator)
  return concept_sequence


def create_numerical_feature_sequence(
    num_steps: int, feature_spec: dataset_specs.NumericalFeatureSpec,
    feature_changepoints_and_patterns: List[
        Tuple[int, List[dataset_specs.FeaturePattern]]]
    ):
  """Returns a time series sequence for a single numerical feature.

  All feature patterns are additive.

  Args:
    num_steps: Number of timesteps to generate.
    feature_spec: Feature specification for the sequence.
    feature_changepoints_and_patterns: A list of pairs of timestep changepoints
      and the feature patterns at those changepoints.
  """
  sequence = np.zeros(num_steps)
  # Fill in background values.
  sequence += (
      feature_spec.bg_mean +
      np.random.normal(size=num_steps) * feature_spec.bg_std)
  for changepoint, feature_patterns in feature_changepoints_and_patterns:
    # Fill in manifested values.
    if dataset_specs.FeaturePattern.SINE in feature_patterns:
      pattern_values = np.sin(
          np.array(np.arange(num_steps) * feature_spec.sine_freq) * 0.2 +
          np.random.normal() * 10)
      pattern_values = (
          pattern_values * feature_spec.sine_amp +
          np.random.normal(size=pattern_values.shape) * feature_spec.sine_std
          )
      sequence[changepoint:] += pattern_values[changepoint:]
    if dataset_specs.FeaturePattern.OFFSET in feature_patterns:
      pattern_values = np.ones(shape=num_steps) * feature_spec.offset_step
      sequence[changepoint:] += np.cumsum(
          pattern_values[changepoint:], axis=0)
  return sequence


def create_binary_feature_sequence(
    num_steps: int, feature_spec: dataset_specs.BinaryFeatureSpec,
    feature_changepoints_and_patterns: List[
        Tuple[int, List[dataset_specs.FeaturePattern]]]
    ):
  """Returns a time series sequence for a single binary feature.

  All feature patterns are additive.

  Args:
    num_steps: Number of timesteps to generate.
    feature_spec: Feature specification for the sequence.
    feature_changepoints_and_patterns: A list of pairs of timestep changepoints
      and the feature patterns at those changepoints.
  """
  sequence = np.zeros(num_steps)
  # Fill in background values.
  sequence[:] = (
      np.random.uniform(size=num_steps) < feature_spec.bg_prob)
  for changepoint, feature_patterns in feature_changepoints_and_patterns:
    if dataset_specs.FeaturePattern.PRESENCE in feature_patterns:
      pattern_values = (
          np.random.uniform(size=num_steps) < feature_spec.presence_prob)
      sequence[changepoint:] += pattern_values[changepoint:]
      sequence[sequence > 1] = 1
  return sequence


def create_full_feature_sequence(
    num_steps: int,
    feature_specs: List[dataset_specs.FeatureSpecType],
    concept_specs: List[dataset_specs.ConceptSpec],
    concept_indicators: List[bool], concept_changepoints: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
  """Create a single time series sequence with multiple features.

  Args:
    num_steps: Number of timesteps to generate.
    feature_specs: List of feature specifications.
    concept_specs: List of concept specifications.
    concept_indicators: List of boolean indicators for which state each concept
      has been sampled in.
    concept_changepoints: List of sampled timesteps for each concept.
  Returns:
    A tuple of two arrays:
      - Float sequence array of shape [num_steps, ndims].
      - Boolean concept-feature indicator array of shape
        [num_concepts, num_features].
  """
  ndims = len(feature_specs)
  concept_feature_indicators = np.zeros(
      shape=(len(concept_specs), ndims)).astype(bool)
  feature_idx_to_changepoints_and_patterns = collections.defaultdict(list)

  for concept_idx, concept_spec in enumerate(concept_specs):
    for feature_idx in concept_spec.feature_idxs:
      feature_patterns = concept_spec.feature_idx_to_patterns[feature_idx]
      agreement = concept_spec.feature_idx_to_agreement[feature_idx]
      agreed = np.random.uniform() < agreement
      if agreed:
        concept_feature_indicator = concept_indicators[concept_idx]
      else:
        concept_feature_indicator = np.random.uniform() < 0.5
      concept_feature_indicators[concept_idx, feature_idx] = (
          concept_feature_indicator)
      if concept_feature_indicator:
        feature_idx_to_changepoints_and_patterns[feature_idx].append(
            (concept_changepoints[concept_idx], feature_patterns))

  sequence = np.zeros(shape=(num_steps, ndims))
  for d in range(ndims):
    if isinstance(feature_specs[d], dataset_specs.NumericalFeatureSpec):
      feature_sequence = create_numerical_feature_sequence(
          num_steps, feature_specs[d],
          feature_idx_to_changepoints_and_patterns[d])
    elif isinstance(feature_specs[d], dataset_specs.BinaryFeatureSpec):
      feature_sequence = create_binary_feature_sequence(
          num_steps, feature_specs[d],
          feature_idx_to_changepoints_and_patterns[d])
    else:
      raise ValueError("Unsupported spec {}".format(feature_specs[d]))
    sequence[:, d] = feature_sequence

  return sequence, concept_feature_indicators


def create_batch(
    feature_specs: List[dataset_specs.FeatureSpecType],
    concept_specs: List[dataset_specs.ConceptSpec],
    label_specs: List[dataset_specs.LabelSpec],
    batch_size: int) -> DataSplit:
  """Creates a batch of features and labels, along with other metadata.

  Args:
    feature_specs: List of feature specifications.
    concept_specs: List of concept specifications.
    label_specs: List of label specifications.
    batch_size: Size of batch to create.

  Returns:
    A DataSplit object containing five arrays:
      sequence: Float sequence of features of shape
        [num_unroll, batch_size, ndims].
      label: Float array of labels of shape [num_unroll, batch_size, num_labels]
      concept: Bool array of concept indicators of shape
        [batch_size, num_concepts].
      changes: Int array of concept changepoints of shape
        [batch_size, num_concepts].
      concept_sequence: Int array of concept indicators on a timestep resolution
        of shape [num_unroll, batch_size, num_concepts].
      features: Bool array of concept-feature indicators of shape
        [batch_size, num_concepts, ndims]
  """

  num_unroll = 100

  batch_sequence = []
  batch_labels = []
  batch_concept_indicators = []
  batch_concept_changepoints = []
  batch_concept_feature_indicators = []

  for _ in range(batch_size):

    # Sample concepts and changepoints.
    concept_indicators = []
    concept_changepoints = []
    for _ in range(len(concept_specs)):
      concept_indicators.append(np.random.choice([True, False]))
      concept_changepoints.append(np.random.choice(range(0, 100)))

    # Sample labels.
    labels = []
    for label_spec in label_specs:
      label_concept_idxs = label_spec.concept_idxs
      label_concept_indicators = [
          concept_indicators[idx] for idx in label_concept_idxs]
      label_concept_changepoints = [
          concept_changepoints[idx] for idx in label_concept_idxs]
      label = np.zeros(num_unroll)
      label_concept_sort_idxs = np.argsort(label_concept_changepoints)
      contingency_idx = [0] * len(label_concept_idxs)
      for sort_idx in label_concept_sort_idxs:
        if label_concept_indicators[sort_idx]:
          contingency_idx[sort_idx] = 1
        label_prob = label_spec.contingency_table[tuple(contingency_idx)]
        label_indicator = int(np.random.uniform() < label_prob)
        if label_indicator:
          label[label_concept_changepoints[sort_idx]:] = 1
          break
      labels.append(label)

    sequence, concept_feature_indicators = create_full_feature_sequence(
        num_unroll,
        feature_specs,
        concept_specs,
        concept_indicators,
        concept_changepoints)

    batch_sequence.append(sequence)
    batch_labels.append(np.transpose(np.stack(labels)))
    batch_concept_indicators.append(np.array(concept_indicators))
    batch_concept_changepoints.append(np.array(concept_changepoints))
    batch_concept_feature_indicators.append(concept_feature_indicators)

  batch_sequence = np.stack(batch_sequence, axis=1).astype(np.float32)
  batch_labels = np.stack(batch_labels, axis=1).astype(np.float32)
  batch_concept_indicators = np.stack(batch_concept_indicators, axis=0)
  batch_concept_changepoints = np.stack(batch_concept_changepoints, axis=0)
  batch_concept_feature_indicators = np.stack(
      batch_concept_feature_indicators, axis=0)
  batch_concept_sequence = generate_concept_sequence(
      num_unroll, batch_concept_indicators, batch_concept_changepoints)

  return DataSplit(
      sequence=batch_sequence,
      label=batch_labels,
      concept=batch_concept_indicators,
      changes=batch_concept_changepoints,
      concept_sequence=batch_concept_sequence,
      features=batch_concept_feature_indicators)


def scale_sequences(
    scaler: feature_scaling_utils.FeatureScaler,
    feature_specs: List[dataset_specs.FeatureSpecType],
    train_sequence: np.ndarray,
    test_sequence: np.ndarray,
    valid_sequence: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
  """Scales sequences with provided scaler object.

    Fits on train sequence, scales both train and test.

  Args:
    scaler: object for scaling.
    feature_specs: list of specs with feature metadata.
    train_sequence: train array of shape [num_unroll, batch_size, num_features].
    test_sequence: test array of shape [num_unroll, batch_size, num_features].
    valid_sequence: validation array of shape
      [num_unroll, batch_size, num_features].

  Returns:
    A tuple of train, test, and validation sequences, scaled accordingly.
  """
  scaler.fit(train_sequence, feature_specs)
  train_sequence = scaler.transform(train_sequence)
  test_sequence = scaler.transform(test_sequence)
  if valid_sequence:
    valid_sequence = scaler.transform(valid_sequence)

  return train_sequence, test_sequence, valid_sequence


def generate_dataset(
    num_trains: int,
    num_tests: int,
    feature_specs: List[dataset_specs.FeatureSpecType],
    concept_specs: List[dataset_specs.ConceptSpec],
    label_specs: List[dataset_specs.LabelSpec],
    scaling_type: dataset_specs.ScalingType,
    seed: int,
) -> Dataset:
  """Generate a synthetic dataset.

  Args:
    num_trains: number of samples to generate for the training set.
    num_tests: number of samples to generate for the test set.
    feature_specs: list of feature spec objects with feature metadata.
    concept_specs: list of concept spec objects.
    label_specs: list of label spec objects.
    scaling_type: one of 'none', 'meanstd', 'unitrange'.
    seed: which seed to set, for reproducibility.

  Returns:
    A Dataset object containing the three dataset splits.
  """
  np.random.seed(seed)
  dataset_kwargs = dict(
      feature_specs=feature_specs,
      concept_specs=concept_specs,
      label_specs=label_specs)
  train_datasplit = create_batch(batch_size=num_trains, **dataset_kwargs)
  test_datasplit = create_batch(batch_size=num_tests, **dataset_kwargs)
  valid_datasplit = create_batch(batch_size=num_tests, **dataset_kwargs)
  # Scaling
  if scaling_type == dataset_specs.ScalingType.NONE:
    scaler = None
  elif scaling_type == dataset_specs.ScalingType.MEAN_STD_STANDARDIZATION:
    scaler = feature_scaling_utils.MeanStdStandardScaler()
  elif scaling_type == dataset_specs.ScalingType.UNIT_RANGE_NORMALIZATION:
    scaler = feature_scaling_utils.UnitRangeNormScaler()
  else:
    raise ValueError("Unsupported scaling type {}".format(scaling_type))
  if scaler:
    train_seq, test_seq, valid_seq = scale_sequences(
        scaler=scaler,
        feature_specs=feature_specs,
        train_sequence=train_datasplit.sequence,
        test_sequence=test_datasplit.sequence,
        valid_sequence=valid_datasplit.sequence,
    )
    train_datasplit.sequence = train_seq
    test_datasplit.sequence = test_seq
    valid_datasplit.sequence = valid_seq

  return Dataset(
      train_split=train_datasplit,
      valid_split=valid_datasplit,
      test_split=test_datasplit)


def _convert_dict_to_concept_spec(
    concept_spec: Dict[str, Any]) -> dataset_specs.ConceptSpec:
  """Converts dict object to ConceptSpec object."""
  return dataset_specs.ConceptSpec(**concept_spec)


def _convert_dict_to_feature_spec(
    feature_spec: Dict[str, Any]) -> dataset_specs.FeatureSpecType:
  """Converts dict object to FeatureSpecType object."""
  if "bg_prob" in feature_spec:
    return dataset_specs.BinaryFeatureSpec(**feature_spec)
  else:
    return dataset_specs.NumericalFeatureSpec(**feature_spec)


def _convert_dict_to_label_spec(
    label_spec: Dict[str, Any]) -> dataset_specs.LabelSpec:
  """Converts dict object to LabelSpec object."""
  return dataset_specs.LabelSpec(**label_spec)
