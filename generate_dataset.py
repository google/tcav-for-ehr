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
"""Generate synthetic dataset."""

import datetime
import os

from absl import flags
from absl import logging

import attr

import dataset_configs
import dataset_specs
import dataset_utils

from absl import app



SCALING_TYPES = [stype.value for stype in dataset_specs.ScalingType]
VERSION = 3

flags.DEFINE_list("datasets", None, "Dataset names.")
flags.DEFINE_string("export_dir", "datasets/", "Export path.")
flags.DEFINE_list("scaling_types", [dataset_specs.ScalingType.NONE.value],
                  f"Feature scaling types to run. Options are {SCALING_TYPES}")
flags.DEFINE_integer("num_train", 100000,
                     "Number of samples in the 'train' split.")
flags.DEFINE_integer("num_test", 10000,
                     "Number of samples in the 'test' split.")
flags.DEFINE_string("suffix", "",
                    "Suffix to add to the end of the automated dataset name")
FLAGS = flags.FLAGS

flags.register_validator(
    "scaling_types",
    lambda stypes: all([stype in SCALING_TYPES for stype in stypes]),
    message=f"--scaling_types must all be one of {SCALING_TYPES}")


def _write_pickled_dataset(dataset, export_path):
  # Attr classes are tricky to serialize, so convert to dicts.
  dataset = attr.asdict(dataset)
  logging.info("Saving dataset to %s", f"{export_path}.pkl")
  dataset_utils.save_pickled_data(dataset, f"{export_path}.pkl")


def main(argv):
  del argv
  flags.mark_flag_as_required("datasets")
  for dataset_name in FLAGS.datasets:
    for scaling_type in FLAGS.scaling_types:
      logging.info("Generating %s with scaling type %s.",
                   dataset_name, scaling_type)
      scaling_type_id = next(i for i in dataset_specs.ScalingType
                             if i.value == scaling_type)
      config = dataset_configs.get_config(dataset_name, FLAGS.num_train,
                                          FLAGS.num_test, scaling_type_id)
      dataset = dataset_utils.generate_dataset(**config)
      dataset.config = config

      timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

      suffix = FLAGS.suffix
      if suffix:
        suffix = "_" + suffix

      # include scaling_type in export_filename if not none
      if scaling_type == "none":
        dataset_basename = (
            f"{dataset_name}_v{VERSION:03d}_{timestamp}{suffix}")
      else:
        dataset_basename = (
            f"{dataset_name}_{scaling_type}_v{VERSION:03d}_{timestamp}{suffix}")

      if not os.path.exists(FLAGS.export_dir):
        os.makedirs(FLAGS.export_dir)

      export_path = os.path.join(FLAGS.export_dir, dataset_basename)

      _write_pickled_dataset(dataset, export_path)
      logging.info("Save complete.")


if __name__ == "__main__":
  flags.mark_flag_as_required("datasets")
  app.run(main)
