# Lint as: python3
"""Types used to define features included to the synthetic datasets."""

import enum
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar
import attr
import numpy as np


class ScalingType(enum.Enum):
  """Defines a list of possible scaling methods."""
  MEAN_STD_STANDARDIZATION = "meanstd"
  UNIT_RANGE_NORMALIZATION = "unitrange"
  NONE = "none"


class FeaturePattern(enum.Enum):
  """The feature patterns that can be induced by a concept."""
  SINE = "sine"
  OFFSET = "offset"
  PRESENCE = "presence"

ConceptSpecType = TypeVar("ConceptSpecType", bound="ConceptSpec")


@attr.s(auto_attribs=True, frozen=True)
class ConceptSpec:
  """The specifications for a synthetic dataset concept.

  Concepts are binary latent variables that are sampled once for each sequence.
  These concepts, coupled with a "changepoint" (timestep at which the concept
  manifests), determine the feature patterns that manifest in the sequence.

  Attributes:
    name: Name of the concept for reference.
    feature_idxs: List of feature indexes influenced by the concept.
    feature_idx_to_patterns: A mapping between feature index and the pattern
      induced by the concept.
    feature_idx_to_agreement: A mapping between feature index and its agreement
      with the concept. The relationship between feature agreement and
      probability of the feature pattern occurring (pF) is the following:
        (pF | Concept) = (No pF | No Concept) = agreement + (1 - agreement) / 2
      These probabilities are not required to be equal, but we do so by
      convention.
  """
  name: str
  feature_idxs: List[int]
  feature_idx_to_patterns: Mapping[int, List[FeaturePattern]]
  feature_idx_to_agreement: Mapping[int, float]

  def __attrs_post_init__(self):
    for idx in self.feature_idxs:
      if (idx not in self.feature_idx_to_patterns or
          idx not in self.feature_idx_to_agreement):
        raise ValueError(
            f"Relevant feature idx {idx} is missing from other attributes.")

  @classmethod
  def with_same_feature_patterns(
      cls: Type[ConceptSpecType], name: str, feature_idxs: List[int],
      agreements: List[float], feature_patterns: List[FeaturePattern]
      ) -> ConceptSpecType:
    """Create concept spec in which each feature has the same pattern(s)."""
    return cls(name=name,
               feature_idxs=feature_idxs,
               feature_idx_to_agreement=dict(zip(feature_idxs, agreements)),
               feature_idx_to_patterns={
                   idx: feature_patterns for idx in feature_idxs})

LabelSpecType = TypeVar("LabelSpecType", bound="LabelSpec")


@attr.s(auto_attribs=True, frozen=True)
class LabelSpec:
  """The specifications for a synthetic dataset label.

  Labels are targets to predict, and they are influenced by the presence or
  absence of latent concepts.

  Attributes:
    name: Name of the label for reference.
    concept_idxs: List of concept indexes that influence the label.
    contingency_table: An N-dimensional binary matrix of label probabilities,
      conditioned on the presence or absence of each concept. Each dimension of
      the table should correspond to the concept idx at the same dimension of
      the concept_idxs attribute - order matters!
  """
  name: str
  concept_idxs: List[int]
  contingency_table: np.ndarray

  def __attrs_post_init__(self):
    if len(self.concept_idxs) != len(self.contingency_table.shape):
      raise ValueError(
          "Contingency table dimensions do not match number of label concepts.")
    if any(dim != 2 for dim in self.contingency_table.shape):
      raise ValueError("Contingency table should include only binary concepts.")

  @classmethod
  def from_single_concept(
      cls: Type[LabelSpecType], name: str, concept_idx: int,
      pos_concept_prob: float, neg_concept_prob: Optional[float] = None
      ) -> LabelSpecType:
    """Returns a label spec from a single concept and pair of probabilities.

    Args:
      name: Name of the label for reference.
      concept_idx: The index of the concept_specs list to use for this label.
      pos_concept_prob: The conditional probability of the label given a
        positive concept sample: p(label | concept).
      neg_concept_prob: The conditional probability of the label given a
        negative concept sample: p(label | no concept). If a neg_concept_prob is
        not provided, we use the complementary probability of pos_concept_prob
        as a matter of convention.
    """
    if neg_concept_prob is None:
      neg_concept_prob = 1 - pos_concept_prob
    return cls(name=name,
               concept_idxs=[concept_idx],
               contingency_table=np.array([neg_concept_prob, pos_concept_prob]))


@attr.s(auto_attribs=True, frozen=True)
class NumericalFeatureSpec:
  """The specifications for a synthetic dataset numerical feature."""
  offset_step: float = 0.04
  sine_freq: float = 1.0
  sine_amp: float = 1.0
  sine_std: float = 0.2
  bg_mean: float = 0.0
  bg_std: float = 0.5


@attr.s(auto_attribs=True, frozen=True)
class BinaryFeatureSpec:
  """The specifications for a synthetic dataset binary feature."""
  presence_prob: float = 0.95
  bg_prob: float = 0.5

FeatureSpecType = TypeVar("FeatureSpecType",
                          NumericalFeatureSpec, BinaryFeatureSpec)


def default_dataset_config(
    feature_specs: List[FeatureSpecType], concept_specs: List[ConceptSpec],
    label_specs: List[LabelSpec]
    ) -> Dict[str, Any]:
  return dict(
      num_trains=100000,
      num_tests=10000,
      feature_specs=feature_specs,
      concept_specs=concept_specs,
      label_specs=label_specs,
      scaling_type=ScalingType.NONE.value,
      seed=42)
