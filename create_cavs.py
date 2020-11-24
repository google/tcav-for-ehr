# Lint as: python3
"""Script for training CAVs."""
import os
import time
from typing import Any, List, Mapping, Optional
from absl import flags
from absl import logging
import numpy as np
from sklearn.base import BaseEstimator
import dataset_utils
import tcav_eval_utils
import metrics
from absl import app


flags.DEFINE_string("dataset_path", None, "Path to the data.")
flags.DEFINE_string("model_path", None, "Path for trained model restore. May"
                    " end with 'tfhub'.")
flags.DEFINE_string("output_dir", None, "Directory to output the results to.")
flags.DEFINE_boolean(
    "save_cav_classifiers", True,
    "Whether to save CAV classifiers, of type sklearn.base.BaseEstimator.")
flags.DEFINE_string(
    "linear_model_type", "linear", "What kind of linear model should be used "
    "for training the CAVs. One of 'logistic' or 'linear'.")
flags.DEFINE_integer(
    "num_bootstrapped_datasets", 100, "Number of CAVs to be trained. Each CAV "
    "is trained on a different bootstrapped dataset.")
flags.DEFINE_integer(
    "num_permutations", 10, "Number of datasets with permuted labels to be "
    "used for training random CAVs on each bootstrapped dataset. The "
    "total number of models with permuted labels will be "
    "`num_bootstrapped_datasets * num_permutations`.")
flags.DEFINE_enum(
    "cav_classifier_mode", tcav_eval_utils.CavClassifierMode.T0_TO_T1.value,
    [tcav_eval_utils.CavClassifierMode.T1_ONLY.value,
     tcav_eval_utils.CavClassifierMode.T0_TO_T1.value,
     tcav_eval_utils.CavClassifierMode.T0_TO_T1_DIFF.value],
    "Training data mode for CAV classifiers. Options are 't1_only', 't0_to_t1',"
    " & t0_to_t1_diff.")
flags.DEFINE_boolean(
    "cav_classifier_early_stopping", True,
    "Whether to use early stopping when training CAV classifiers.")
flags.DEFINE_integer(
    "synthetic_ntrains", None,
    "Number of data elements to include in the CAV "
    "training dataset for synthetic data. Will take the first N elements that "
    "satisfy the accuracy threshold condition.")
flags.DEFINE_float(
    "synthetic_accuracy_threshold", 0.0,
    "Threshold model accuracy to filter CAV training examples for the standard "
    "synthetic dataset.")
flags.DEFINE_integer(
    "synthetic_changepoint_lookback", 0,
    "The number of timesteps before the change point to"
    " select as t0 for standard synthetic CAV datasets.")
flags.DEFINE_integer(
    "synthetic_changepoint_lookahead", 25,
    "The number of timesteps after the change point to"
    " select as t1 for standard synthetic CAV datasets.")
flags.DEFINE_list(
    "metrics", ["balanced_accuracy", "rocauc"],
    "List of metrics names to be used for evaluating CAVs.")

FLAGS = flags.FLAGS


def _extract_cavs(
    models: Mapping[str, Mapping[int, List[BaseEstimator]]]
    ) -> Mapping[str, Mapping[int, np.ndarray]]:
  """Extract the CAVs from trained models for all concepts and layers."""
  cavs = {}
  for concept_name, all_models in models.items():
    cavs[concept_name] = {}
    for layer, models in all_models.items():
      cavs[concept_name][layer] = np.stack(
          [m.coef_.squeeze() for m in models])  # pytype: disable=attribute-error
  return cavs


def create_cavs(
    dataset_path: str,
    model_path: str,
    output_dir: str,
    cav_classifier_mode: tcav_eval_utils.CavClassifierMode,
    num_bootstrapped_datasets: int,
    num_permutations: int,
    dataset_type: dataset_utils.DatasetType,
    metric_list: List[Any],
    save_cav_classifiers: bool,
    linear_model_type: str = "linear",
    changepoint_lookback: int = 0,
    changepoint_lookahead: int = 25,
    cav_classifier_early_stopping: bool = True,
    synthetic_accuracy_threshold: float = 0.0,
    synthetic_ntrains: Optional[int] = None,
) -> None:
  """Creates a CAV training dataset and trains CAVs for each concept and layer.

  Args:
    dataset_path: Where the dataset is saved.
    model_path: Where the trained model is saved.
    output_dir: Where to save the output.
    cav_classifier_mode: Training data mode for CAV classifiers. Options are
      't1_only', 't0_to_t1', & t0_to_t1_diff.
    num_bootstrapped_datasets: Number of bootstrap samples to take of CAV
      training dataset.
    num_permutations: Number of permuted-label samples to generate of CAV
      training dataset.
    dataset_type: The type of CAV training dataset to generate.
    metric_list: List of metrics to evaluate the performance of the trained
      linear models.
    save_cav_classifiers: Whether to save CAV classifiers, of type
      sklearn.base.BaseEstimator.
    linear_model_type: What kind of linear model should be used to define CAVs.
      One of "linear" or "logistic".
    changepoint_lookback: The number of timesteps before the change point to
      select as t0 for standard synthetic CAV datasets.
    changepoint_lookahead: The number of timesteps after the change point to
      select as t1 for standard synthetic CAV datasets.
    cav_classifier_early_stopping: Whether to use early stopping when training
      CAV classifiers.
    synthetic_accuracy_threshold: Threshold model accuracy to
      filter CAV training examples.
    synthetic_ntrains: Number of data elements to include in the CAV training
      dataset for synthetic data. Will take the first N elements that satisfy
      the accuracy threshold condition. Does not need to be provided if creating
      a different dataset type.
  """
  if dataset_type == dataset_utils.DatasetType.STANDARD_SYNTHETIC:
    logging.info("Loading synthetic data...")
    data_loading_start_time = time.time()
    dataset = dataset_utils.load_pickled_data(dataset_path)
    concept_names = (
        [spec["name"] for spec in dataset["config"]["concept_specs"]])
    label_names = [spec["name"] for spec in dataset["config"]["label_specs"]]
    logging.info("Synthetic data loading time: %fs",
                 time.time() - data_loading_start_time)
    logging.info("Creating standard synthetic CAV training dataset...")
    cav_training_data_start_time = time.time()
    concept_to_training_data = (
        tcav_eval_utils.create_standard_cav_training_dataset_synthetic(
            model_path=model_path,
            data_split=dataset["valid_split"],
            label_names=label_names,
            concept_names=concept_names,
            ntrains=synthetic_ntrains,
            changepoint_lookback=changepoint_lookback,
            changepoint_lookahead=changepoint_lookahead,
            classifier_mode=cav_classifier_mode,
            classifier_step=1,
            accuracy_threshold=synthetic_accuracy_threshold,
            accuracy_threshold_label_name=label_names[0]))
    logging.info("Synthetic CAV training dataset creation time: %fs",
                 time.time() - cav_training_data_start_time)
  else:
    raise ValueError(f"dataset_type arg {dataset_type} not recognized.")
  cav_training_start = time.time()
  logging.info("Training CAVs...")
  bootstrap_models, permuted_models, metric_results = (
      tcav_eval_utils.train_classifiers(
          concept_to_training_data,
          metrics=metric_list,
          model_type=linear_model_type,
          num_bootstrapped_datasets=num_bootstrapped_datasets,
          num_permutations=num_permutations,
          cross_val_reg=True,
          early_stopping=cav_classifier_early_stopping))
  logging.info("CAV training time: %fs", time.time() - cav_training_start)
  if save_cav_classifiers:
    cav_classifier_path = os.path.join(output_dir, "cav_classifiers.pkl")
    dataset_utils.save_pickled_data(bootstrap_models, cav_classifier_path)
  bootstrap_cavs = _extract_cavs(bootstrap_models)
  permuted_cavs = _extract_cavs(permuted_models)
  all_cavs = {tcav_eval_utils.CavType.BOOTSTRAP: bootstrap_cavs,
              tcav_eval_utils.CavType.PERMUTED: permuted_cavs}

  cavs_path = os.path.join(output_dir, "cavs.pkl")
  logging.info("Saving CAVS to %s...", cavs_path)
  dataset_utils.save_pickled_data(all_cavs, cavs_path)
  cav_metrics_path = os.path.join(output_dir, "cav_metrics.pkl")
  logging.info("Saving CAV metrics to %s...", cav_metrics_path)
  dataset_utils.save_pickled_data(metric_results, cav_metrics_path)


def main(argv):
  del argv  # unused
  flags.mark_flags_as_required([
      "dataset_path",
      "model_path",
      "output_dir",
      "synthetic_ntrains"
  ])
  # Convert strings to enums.
  cav_classifier_mode = tcav_eval_utils.CavClassifierMode(
      FLAGS.cav_classifier_mode)
  dataset_type = dataset_utils.DatasetType.STANDARD_SYNTHETIC
  create_cavs(
      dataset_path=FLAGS.dataset_path,
      model_path=FLAGS.model_path,
      output_dir=FLAGS.output_dir,
      save_cav_classifiers=FLAGS.save_cav_classifiers,
      synthetic_ntrains=FLAGS.synthetic_ntrains,
      cav_classifier_mode=cav_classifier_mode,
      num_bootstrapped_datasets=FLAGS.num_bootstrapped_datasets,
      num_permutations=FLAGS.num_permutations,
      dataset_type=dataset_type,
      metric_list=[metrics.METRICS[metric] for metric in FLAGS.metrics],
      linear_model_type=FLAGS.linear_model_type,
      changepoint_lookback=FLAGS.synthetic_changepoint_lookback,
      changepoint_lookahead=FLAGS.synthetic_changepoint_lookahead,
      synthetic_accuracy_threshold=(
          FLAGS.synthetic_accuracy_threshold),
      cav_classifier_early_stopping=FLAGS.cav_classifier_early_stopping,
  )


if __name__ == "__main__":
  app.run(main)
