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
"""Compute TCAV."""

from typing import Dict, List, Optional, Tuple, Union

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tree as nest

import smart_module

ModelStepOutput = Tuple[List[tf.Variable], tf.Tensor,
                        tf.Tensor, tf.Tensor, tf.Tensor]


def step_with_state(model: smart_module.SmartModuleImport,
                    inputs: tf.Tensor,
                    labels: tf.Tensor,
                    initial_state: List[tf.Tensor],
                    use_sigmoid_gradients: bool = False,
                    from_layer: Optional[int] = None,
                    modify_state: bool = True) -> ModelStepOutput:
  """Step through RNN with persistent state.

  Args:
    model: Model to be used to make predictions.
    inputs: Input at current time. Shape [batch_size, num_features].
    labels: Label to predict at this time. Shape [batch_size, num_labels]. Note
      that there should be as many labels as the model predicts.
    initial_state: The initial hidden state of the model. One per layer.
    use_sigmoid_gradients: Whether to take the gradients of the predicted
      probabilities or logits.
    from_layer: Which layer to run the forward pass from. In that case, the
      inputs should be the hidden activations at that layer.
    modify_state: Whether to update the hidden state with its new value.

  Returns:
    The hidden states, prediction, loss, activations, logits, and gradients
    w.r.t. activations of the model for this timestep.
  """

  def create_state(t: tf.Tensor) -> tf.Variable:
    """Create and initialize a local variable with a value for persistence."""
    return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

  with tf.variable_scope("state", reuse=tf.AUTO_REUSE):
    prev_state_vars = nest.map_structure(create_state, initial_state)
  prev_state = nest.map_structure(lambda x: x.read_value(), prev_state_vars)

  if from_layer is None:
    logits, activations, next_state = model.step(inputs, prev_state)
  else:
    if modify_state:
      raise ValueError(
          "When specifying a layer to run the forward pass from, `modify_state`"
          " should be False.")
    activations = [inputs]
    logits = model.step_from_hidden_layer(
        acts=inputs,
        from_layer=from_layer,
        all_prev_states=prev_state,
    )

  # loss and gradients
  loss_per_sample = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=logits)
  loss = tf.reduce_mean(loss_per_sample)
  if use_sigmoid_gradients:
    gradients = [
        tf.gradients(tf.nn.sigmoid(logits[:, k]), activations)
        for k in range(logits.shape.as_list()[-1])
    ]
  else:
    gradients = [
        tf.gradients(logits[:, k], activations)
        for k in range(logits.shape.as_list()[-1])
    ]

  if modify_state:
    assign_states = nest.map_structure(lambda x, v: x.assign(v),
                                       prev_state_vars, next_state)
    with tf.control_dependencies(nest.flatten(assign_states)):
      loss = tf.identity(loss)
      logits = tf.identity(logits)
      activations = nest.map_structure(tf.identity, activations)
      gradients = nest.map_structure(tf.identity, gradients)

  return prev_state_vars, loss, logits, activations, gradients


def unroll_model_per_step(
    model_path: str,
    sequences: np.ndarray,
    labels: np.ndarray,
    use_sigmoid_gradients: bool = False,
) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
  """Evaluate the model one step at a time.

  Args:
    model_path: Where the model to be restored is saved.
    sequences: Input time series. Shape [num_timesteps, batch_size,
      num_features].
    labels: Label to predicts. Shape [num_timesteps, batch_size, num_labels].
      Note that there should be as many labels as the model predicts.
    use_sigmoid_gradients: Whether to take the gradients of the predicted
      probabilities or logits.

  Returns:
    The prediction, loss, activations, logits, gradients w.r.t. activations
    of the model, and maybe the states.
  """

  ndims = sequences.shape[-1]

  with tf.Graph().as_default():
    with tf.variable_scope(None, default_name="step_with_state"):
      step_sequences_ph = tf.placeholder(
          name="step_sequences", shape=(None, ndims), dtype=tf.float32)
      step_labels_ph = tf.placeholder(
          name="step_labels", shape=(None, labels.shape[-1]), dtype=tf.float32)

      model = smart_module.SmartModuleImport(hub.Module(model_path))
      initial_state = model.initial_state(tf.shape(step_sequences_ph)[0])
      logging.info("Loaded %s", model_path)

      step_state, _, step_logits, step_acts, step_grads = step_with_state(
          model,
          step_sequences_ph,
          step_labels_ph,
          initial_state=initial_state,
          use_sigmoid_gradients=use_sigmoid_gradients)

    with tf.Session() as sess:
      sess.run(
          [tf.global_variables_initializer(),
           tf.local_variables_initializer()],
          feed_dict={
              step_sequences_ph: sequences[0, :],
              step_labels_ph: labels[0, :],
          })

      fetches = [step_state, tf.nn.sigmoid(step_logits), step_logits, step_acts,
                 step_grads]
      run_step = sess.make_callable(fetches,
                                    [step_sequences_ph, step_labels_ph])

      states, probs, logits, acts, grads = zip(
          *[run_step(seq, lbl) for seq, lbl in zip(sequences, labels)])

  states = [np.stack(x, axis=0) for x in zip(*states)]
  probs = np.stack(probs, axis=0)
  logits = np.stack(logits, axis=0)
  predictions = np.stack([x > 0 for x in logits], axis=0)
  acts = [np.stack(x, axis=0) for x in zip(*acts)]
  grads = [np.stack(x, axis=0) for x in zip(*grads)]

  return dict(
      states=states,
      probs=probs,
      logits=logits,
      predictions=predictions,
      activations=acts,
      gradients=grads,
  )


def unroll_model(model_path: str, sequences: np.ndarray,
                 labels: np.ndarray) -> Dict[str, np.ndarray]:
  """Evaluate the model."""

  ndims = sequences.shape[-1]

  with tf.Graph().as_default():
    # Placeholders
    sequences_ph = tf.placeholder(
        name="sequences", shape=(None, None, ndims), dtype=tf.float32)
    labels_ph = tf.placeholder(
        name="labels", shape=(None, None, labels.shape[-1]), dtype=tf.float32)

    # Model and outputs.
    model = smart_module.SmartModuleImport(hub.Module(model_path))
    logits = model.unroll_eval(sequences_ph)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      test_logits, test_probs, test_preds = sess.run(
          [logits, tf.nn.sigmoid(logits), logits > 0],
          feed_dict={
              sequences_ph: sequences,
              labels_ph: labels,
          },
      )

  return dict(predictions=test_preds, probs=test_probs, logits=test_logits)
