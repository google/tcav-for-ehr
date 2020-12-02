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
"""Train models."""

import os
import random
from absl import flags
import dataset_utils
import train_utils
from absl import app



flags.DEFINE_string("dataset_path", None, "Path to the data.")
flags.DEFINE_string("model_cell_type", "LSTM", "The RNN cell type.")
flags.DEFINE_list("hidden_sizes", ["64", "64", "64"], "Size of RNN layers.")
flags.DEFINE_integer("batch_size", 32, "Size of training batches.")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for optimizer.")
flags.DEFINE_integer("num_train_steps", 10000,
                     "Number of training steps/batches.")
flags.DEFINE_integer("model_seed", 1, "RNG seed.")
flags.DEFINE_integer("logging_interval", 50, "Logging interval (in steps)")
flags.DEFINE_string("checkpoint_directory", None,
                    "Path to save the model checkpoint.")
flags.DEFINE_string(
    "training_id", None,
    "String identifier that is appended to the "
    "saved model checkpoint directory.")
flags.DEFINE_float("l1_regularization_scale", 0.0,
                   "Scale for L1 regularization.")
flags.DEFINE_float("input_dropout_prob", 0.0, "Input dropout prob.")
flags.DEFINE_float("output_dropout_prob", 0.0, "Output dropout prob.")
flags.DEFINE_float("state_dropout_prob", 0.0, "State dropout prob.")

FLAGS = flags.FLAGS


def get_train_checkpoint_dir_extension(dataset_path, model_cell_type,
                                       model_seed):
  """Gets unique folder extension for training."""
  dataset_id = dataset_utils.get_dataset_id(dataset_path)
  checkpoint_dir_extension = "{}/{}_{}".format(
      dataset_id, model_cell_type, model_seed)
  return checkpoint_dir_extension


def main(argv):
  del argv

  flags.mark_flag_as_required("dataset_path")
  checkpoint_dir_extension = get_train_checkpoint_dir_extension(
      FLAGS.dataset_path, FLAGS.model_cell_type, FLAGS.model_seed)
  checkpoint_dir = train_utils.get_extended_training_dir(
      FLAGS.checkpoint_directory, checkpoint_dir_extension,
      backup_directory_name="model_checkpoints")
  if FLAGS.training_id:
    checkpoint_dir = f"{checkpoint_dir}_{FLAGS.training_id}"
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  seed = FLAGS.model_seed or random.randint(1, 2**20)
  train_utils.train_model(
      FLAGS.dataset_path,
      cell_type=FLAGS.model_cell_type,
      hidden_sizes=[int(hs) for hs in FLAGS.hidden_sizes],
      input_dropout_prob=FLAGS.input_dropout_prob,
      output_dropout_prob=FLAGS.output_dropout_prob,
      state_dropout_prob=FLAGS.state_dropout_prob,
      batch_size=FLAGS.batch_size,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      checkpoint_base_path=checkpoint_dir,
      seed=seed,
      logging_interval=FLAGS.logging_interval,
      l1_regularization_scale=FLAGS.l1_regularization_scale)


if __name__ == "__main__":
  app.run(main)
