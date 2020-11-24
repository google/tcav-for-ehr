# Lint as: python3
"""Util functions for normalizing synthetic datasets."""

from typing import List

import numpy as np

import dataset_specs


class Error(Exception):
  pass


class FitError(Error):
  """Raised when Scaler object transform is used before being fitted."""


class FeatureScaler(object):
  """General feature scaler class.

  Attributes:
    _feature_specs: list of configs.FeatureSpec objects with feature metadata.
    _ndims: number of feature dimensions of the array.
    _fitted: whether the object has been fit to an array or not.
  """

  def __init__(self):
    self._feature_specs = None
    self._ndims = None
    self._fitted = False

  def fit(self, array: np.ndarray,
          specs: List[dataset_specs.FeatureSpecType]) -> None:
    raise NotImplementedError

  def transform(self, array: np.ndarray) -> np.ndarray:
    raise NotImplementedError


class MeanStdStandardScaler(FeatureScaler):
  """Mean & standard deviation standardization scaler class.

  Applies standardization to all numerical features of an array.

  Attributes:
    _means: list of means for features. None values for binary.
    _stds: list of standard deviations for features. None values for binary.
  """

  def __init__(self):
    super(MeanStdStandardScaler, self).__init__()
    self._means = None
    self._stds = None

  def fit(self, array: np.ndarray,
          specs: List[dataset_specs.FeatureSpecType]) -> None:
    """Collects value parameters useful for mean std standardization."""
    self._feature_specs = specs
    self._ndims = len(specs)
    self._means, self._stds = [None] * self._ndims, [None] * self._ndims
    for d in range(self._ndims):
      if isinstance(specs[d], dataset_specs.NumericalFeatureSpec):
        values = array[:, :, d].flatten()
        self._means[d] = np.mean(values)
        self._stds[d] = np.std(values)
    self._fitted = True

  def transform(self, array: np.ndarray) -> np.ndarray:
    """Mean & standard deviation standardization of dataset.

      For all numerical features, values x with mean mu and std sigma are
      replaced with (x - mu)/sigma, resulting in zero mean and unit std.

    Args:
      array: array of shape [num_unroll, batch_size, num_features].

    Returns:
      A new standardized sequences array of identical shape to the input array.

    Raises:
      FitError: Must run .fit() before .transform()
    """
    if not self._fitted:
      raise FitError("Must run .fit() method before .transform().")
    array = array.copy()
    for d in range(self._ndims):
      if isinstance(self._feature_specs[d], dataset_specs.NumericalFeatureSpec):
        array[:, :, d] -= self._means[d]
        array[:, :, d] /= self._stds[d]
    return array


class UnitRangeNormScaler(FeatureScaler):
  """Unit range normalization scaler class.

  Applies normalization to all numerical features of an array.

  Attributes:
    _mins: list of minimums for features. None values for binary.
    _ranges: list of (max-min) for features. None values for binary.
  """

  def __init__(self):
    super(UnitRangeNormScaler, self).__init__()
    self._mins = None
    self._ranges = None

  def fit(self, array: np.ndarray,
          specs: List[dataset_specs.FeatureSpecType]) -> None:
    """Collects value parameters useful for unit range normalization."""
    self._feature_specs = specs
    self._ndims = len(specs)
    self._mins, self._ranges = [None] * self._ndims, [None] * self._ndims
    for d in range(self._ndims):
      if isinstance(specs[d], dataset_specs.NumericalFeatureSpec):
        values = array[:, :, d].flatten()
        value_max = np.max(values)
        self._mins[d] = np.min(values)
        self._ranges[d] = value_max - self._mins[d]
    self._fitted = True

  def transform(self, array: np.ndarray) -> np.ndarray:
    """Unit range normalization of the dataset.

      For all numerical features, values x with max and min are replaced with
      (x - min)/range, resulting in unit range distribution.

    Args:
      array: array of shape [num_unroll, batch_size, num_features].

    Returns:
      A new normalized sequences array of identical shape to the input array.

    Raises:
      FitError: Must run .fit() before .transform()
    """
    if not self._fitted:
      raise FitError("Must run .fit() method before .transform().")
    array = array.copy()
    for d in range(self._ndims):
      if isinstance(self._feature_specs[d], dataset_specs.NumericalFeatureSpec):
        array[:, :, d] -= self._mins[d]
        array[:, :, d] /= self._ranges[d]
    return array
