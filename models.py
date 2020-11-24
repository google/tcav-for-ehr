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
"""RNN models used for synthetic data experiments."""

from typing import Any, List, Optional, Tuple, Union

import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tree as nest

import smart_module


class MLP(snt.RNNCore):
  """A RNN like MLP module."""

  def __init__(self, num_units: int, activation_fn: str, name: str = "MLP"):
    super(MLP, self).__init__(name=name)
    with self._enter_variable_scope():
      self._cell = snt.Linear(num_units)
      self._act_fn = getattr(tf.nn, activation_fn)

  def _build(self, inputs: tf.Tensor, state: tf.Tensor) -> Tuple[tf.Tensor,
                                                                 tf.Tensor]:
    return self._act_fn(self._cell(inputs)), state

  @property
  def state_size(self) -> int:
    return 1

  @property
  def output_size(self) -> int:
    return self._cell.output_size


class SimpleDeepRNN(snt.RNNCore):
  """A simple deep RNN with skip connections from inputs to every layer."""

  def __init__(self, layers: List[snt.RNNCore],
               skip_connections: bool = True,
               name: str = "simple_deep_rnn"):
    super(SimpleDeepRNN, self).__init__(name=name)
    self._layers = layers
    self._skip_connections = skip_connections

  def _build(self, inputs: tf.Tensor,
             prev_states: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    outputs = []
    states = []
    concatenate = lambda *args: tf.concat(args, axis=-1)

    next_output = inputs
    for idx, (layer, prev_state) in enumerate(zip(self._layers, prev_states)):
      if self._skip_connections and idx > 0:
        next_output = nest.map_structure(concatenate, inputs, next_output)
      next_output, next_state = layer(next_output, prev_state)
      outputs.append(next_output)
      states.append(next_state)
    return outputs, states

  def step_from_hidden_layer(
      self,
      acts: tf.Tensor,
      from_layer: int,
      all_prev_states: List[tf.Tensor],
  ) -> List[tf.Tensor]:
    """Run a RNN step from a specific layer.

    It expects (but does not verify) that `acts` has a shape compatible with the
    hidden layer of interest.

    Args:
      acts: Activation at the considered hidden layer. Shape [None,
        num_features_for_considered_layer].
      from_layer: Indicates what hidden layer to start the forward pass from.
        0 is the first hidden layer.
      all_prev_states: A list of ALL the previous hidden states (including those
        from previous layers).

    Returns:
      A list of the outputs of all subsequent layers.
    """
    if from_layer > len(self._layers):
      raise ValueError(
          f"`from_layer` was {from_layer} but there are only "
          f"{len(self._layers)} layers.")
    if len(all_prev_states) != len(self._layers):
      raise ValueError(
          f"You should provide all the previous hidden states. There are "
          f"{len(self._layers)} but you provided {len(all_prev_states)} "
          f"hidden states")
    if from_layer >= 0:
      if self.output_size[from_layer] != acts.shape.as_list()[-1]:
        raise ValueError(
            f"The last dimension of `acts` was {acts.shape.as_list()[-1]} but "
            f"should have been the hidden size of the output of the layer "
            f"{from_layer}, i.e. {self.output_size[from_layer]}.")

    outputs = []

    next_output = acts
    for _, (layer, prev_state) in enumerate(
        zip(self._layers[from_layer + 1:], all_prev_states[from_layer + 1:])):
      next_output, _ = layer(next_output, prev_state)
      outputs.append(next_output)
    return outputs

  def zero_state(self, batch_size: int,
                 dtype: tf.dtypes.DType) -> List[tf.Tensor]:
    return [layer.zero_state(batch_size, dtype) for layer in self._layers]

  @property
  def output_size(self) -> List[int]:
    try:
      return [layer.output_size.as_list()[0] for layer in self._layers]
    except AttributeError:
      # Some cells do not have the same return type.
      return [layer.output_size for layer in self._layers]

  @property
  def state_size(self) -> List[int]:
    return [layer.state_size for layer in self._layers]


def _create_cell(
    cell_type: str, num_units: int, layer: int
    ) -> Union[snt.RNNCore, tf.nn.rnn_cell.RNNCell]:
  """Creates different RNN cells. Dropout only supported for LSTMs."""
  with tf.variable_scope("layer_{}".format(layer)):
    if cell_type.lower() == "lstm":
      return tf.nn.rnn_cell.LSTMCell(
          num_units=num_units)
    elif cell_type.lower() == "gru":
      return snt.GRU(hidden_size=num_units)
    elif cell_type.lower() == "mlp_tanh":
      return MLP(num_units=num_units, activation_fn="tanh")
    elif cell_type.lower() == "mlp_relu":
      return MLP(num_units=num_units, activation_fn="relu")
    elif cell_type.lower() == "mlp_sigmoid":
      return MLP(num_units=num_units, activation_fn="sigmoid")
    else:
      raise ValueError("Unsupported cell type {}".format(cell_type))


class BinarySequenceClassifier(snt.AbstractModule):
  """Simple RNN (no skip connections) based per-step classifier."""

  def __init__(self,
               cell_type: str,
               hidden_sizes: List[int],
               num_targets: int = 1,
               name: str = "sequence_classifier",
               input_dims: Optional[int] = None,
               input_dropout_prob: float = 0.0,
               output_dropout_prob: float = 0.0,
               state_dropout_prob: float = 0.0):
    super(BinarySequenceClassifier, self).__init__(name=name)
    self._hidden_sizes = hidden_sizes
    def input_size_fn(layer_i):
      if layer_i == 0 and input_dims is None:
        raise ValueError("input_dims is required for LSTM with dropout.")
      return input_dims if layer_i == 0 else hidden_sizes[layer_i - 1]
    with self._enter_variable_scope():
      self._eval_layers = (
          [_create_cell(cell_type, num_units=sz, layer=idx)
           for idx, sz in enumerate(hidden_sizes)])
      if max(input_dropout_prob, output_dropout_prob, state_dropout_prob):
        dropout_kwargs = dict(
            input_keep_prob=(1 - input_dropout_prob),
            output_keep_prob=(1 - output_dropout_prob),
            state_keep_prob=(1 - state_dropout_prob),
            variational_recurrent=True,
            dtype=tf.float32)
        self._train_layers = [
            tf.nn.rnn_cell.DropoutWrapper(
                cell=cell, input_size=input_size_fn(layer), **dropout_kwargs)
            for layer, cell in enumerate(self._eval_layers)]
      else:
        self._train_layers = self._eval_layers
      self._train_model = SimpleDeepRNN(
          self._train_layers, skip_connections=False)
      self._eval_model = SimpleDeepRNN(
          self._eval_layers, skip_connections=False)
      self._head = snt.Linear(num_targets, allow_many_batch_dims=True)
      self.train()

  def train(self):
    self._main_model = self._train_model

  def eval(self):
    self._main_model = self._eval_model

  def _get_model(self, mode: str) -> SimpleDeepRNN:
    if mode == "train":
      return self._train_model
    elif mode == "eval":
      return self._eval_model
    else:
      raise ValueError(f"Unrecognized model mode: {mode}")

  def unroll_train(self, inputs: tf.Tensor) -> tf.Tensor:
    return self._unroll(inputs, "train")

  def unroll_eval(self, inputs: tf.Tensor) -> tf.Tensor:
    return self._unroll(inputs, "eval")

  @snt.reuse_variables
  def _unroll(self, inputs: tf.Tensor, mode: str = "train") -> tf.Tensor:
    model = self._get_model(mode)

    init_state = model.initial_state(batch_size=tf.shape(inputs)[1])

    outputs, unused_final_state = tf.nn.dynamic_rnn(
        model,
        inputs,
        time_major=True,
        dtype=tf.float32,
        initial_state=init_state)

    logits = self._head(outputs[-1])

    return logits

  def step(
      self, inputs: tf.Tensor,
      prev_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor]]:
    outputs, next_states = self.__call__(inputs, prev_state)

    logits = self._head(outputs[-1])

    return logits, outputs, next_states

  def _build(self, inputs: tf.Tensor,
             prev_states: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    outputs, next_states = self._main_model(inputs, prev_states)
    return outputs, next_states

  def initial_state(self, batch_size: int) -> List[tf.Tensor]:
    return self._main_model.initial_state(batch_size)


def load_module(module_path: str) -> smart_module.SmartModuleImport:
  return smart_module.SmartModuleImport(hub.Module(module_path))
