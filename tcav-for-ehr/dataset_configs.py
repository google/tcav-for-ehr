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
"""Dataset configs staged to be released to the public."""

import sys
from typing import Any, Dict

import dataset_specs

this_module = sys.modules[__name__]


def main_paper_config():
  """Two sine concepts, 5 features each concept with equal probability."""
  feature_specs = [dataset_specs.NumericalFeatureSpec() for _ in range(10)]
  concept_specs = [
      dataset_specs.ConceptSpec.with_same_feature_patterns(
          name="SINE0",
          feature_idxs=list(range(5)), agreements=[0.7] * 5,
          feature_patterns=[dataset_specs.FeaturePattern.SINE]),
      dataset_specs.ConceptSpec.with_same_feature_patterns(
          name="SINE1",
          feature_idxs=list(range(5, 10)), agreements=[0.7] * 5,
          feature_patterns=[dataset_specs.FeaturePattern.SINE]),
  ]
  label_specs = [
      dataset_specs.LabelSpec.from_single_concept(
          name="SINE0", concept_idx=0, pos_concept_prob=1.0),
      dataset_specs.LabelSpec.from_single_concept(
          name="SINE1", concept_idx=1, pos_concept_prob=1.0)]
  return dataset_specs.default_dataset_config(
      feature_specs, concept_specs, label_specs)


def get_config(name: str, num_trains: int, num_tests: int,
               scaling_type: dataset_specs.ScalingType) -> Dict[str, Any]:
  config = getattr(this_module, name + "_config")()
  config["num_trains"] = num_trains
  config["num_tests"] = num_tests
  config["scaling_type"] = scaling_type
  return config
