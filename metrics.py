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
"""Metrics for evaluating CAVs."""

import numpy as np
from sklearn import metrics


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Compute accuracy for binary classification models at 50%."""
  return metrics.accuracy_score(y_true=y_true, y_pred=y_pred > .5)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Compute balanced accuracy for binary classification models at 50%."""
  return metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred > .5)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Compute recall for binary classification models at 50% threshold."""
  return metrics.recall_score(y_true=y_true, y_pred=y_pred > .5)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Compute precision for binary classification models at 50% threshold."""
  return metrics.precision_score(y_true=y_true, y_pred=y_pred > .5)


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Compute ROC AUC for binary classification models."""
  return metrics.roc_auc_score(y_true=y_true, y_score=y_pred)


def pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Compute PR AUC for binary classification models."""
  return metrics.average_precision_score(y_true=y_true, y_score=y_pred)


METRICS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "recall": recall,
    "precision": precision,
    "rocauc": roc_auc,
    "prauc": pr_auc
}
