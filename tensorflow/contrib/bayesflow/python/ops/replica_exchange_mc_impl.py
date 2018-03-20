# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Replica Exchange Monte Carlo.

@@kernel
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.contrib.bayesflow.python.ops import hmc
from tensorflow.contrib.bayesflow.python.ops import metropolis_hastings

__all__ = [
    "kernel"
]

KernelResults = collections.namedtuple(
    "KernelResults",
    [
        "inner_kernel_results",
        "next_replica_idx",
        "exchange_proposed",
        "exchange_proposed_n"
    ])


def default_exchange_proposed_fn(freq):
  """Default function for `exchange_proposed_fn` of `kernel`.

  Depending on the probability of `freq`, decide whether to propose combinations
  of replica for exchange.
  When exchanging, create combinations of adjacent replicas from 0 or 1 index.
  """
  def _default_exchange_proposed_fn(n_replica):
    u = random_ops.random_uniform([])
    exist = u > freq

    i = random_ops.random_uniform_int([1], minval=0, maxval=2)[0]
    a = math_ops.range(i, n_replica - 1, 2)
    exchange_proposed = array_ops.transpose(
        array_ops.concat(([a], [a + 1]), axis=0))
    exchange_proposed_n = math_ops.to_int32(math_ops.floor(
        (math_ops.to_int32(n_replica) - i) / 2))

    exchange_proposed = control_flow_ops.cond(
        exist, lambda: math_ops.to_int32([]), lambda: exchange_proposed)
    exchange_proposed_n = control_flow_ops.cond(
        exist, lambda: 0, lambda: exchange_proposed_n)
    return exchange_proposed, exchange_proposed_n

  return _default_exchange_proposed_fn


def kernel(target_log_prob_fn,
           current_state,
           replica_state,
           mode,
           inverse_temperatures,
           exchange_proposed_fn=default_exchange_proposed_fn(0.5),
           seed=None,
           current_target_log_prob=None,
           name=None,
           **kwargs):
  """Runs one iteration of Replica Exchange Monte Carlo.

  Replica Exchange Monte Carlo is a Markov chain Monte Carlo (MCMC) algorithm
  that is also known as Parallel Tempering. This algorithm perform multiple
  sampling with different temperatures in parallel, and exchange those samplings
  according to the Metropolis-Hastings criterion. By using the sampling result
  of high temperature, sampling with less influence of the local solution
  becomes possible.

  Args:
    target_log_prob_fn: Python callable which takes an argument like
      `current_state` (or `*current_state` if it's a list) and returns its
      (possibly unnormalized) log-density under the target distribution.
    current_state: `Tensor` or Python `list` of `Tensor`s representing the
      current state(s) of the Markov chain(s). The first `r` dimensions index
      independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
    replica_state: `list` of variables made in the same way as
      `current_state`, representing the current replica state(s) of the Markov
      chain(s).. This have same length as inverse_temperatures.
    mode: `str`, 'mh' or 'hmc'.
    inverse_temperatures: sequence of inverse temperatures to perform samplings
      with each replica.
    exchange_proposed_fn: Python callable which take a number of replicas, and
      return combinations of replicas for exchange.
    seed: Python integer to seed the random number generator.
    current_target_log_prob: (Optional) `Tensor` representing the value of
      `target_log_prob_fn` at the `current_state`. The only reason to
      specify this argument is to reduce TF graph size.
      Default value: `None` (i.e., compute as needed).
    name: A name of the operation (optional).
    **kwargs: Arguments for inner kernel.

  Returns:
    next_state: Tensor or Python list of `Tensor`s representing the state(s)
      of the Markov chain(s) at each result step. Has same shape as
      `current_state`.
    next_replica_sample: `list` of variables made in the same way as
      `current_state`, representing the replica state(s) of the Markov chain(s)
      at each result step.
    kernel_results: `collections.namedtuple` of internal calculations used to
      advance the chain.

  #### Examples

  We illustrate Replica Exchange Monte Carlo on a Mixture Normal distribution.

  ```python
  tfd = tf.contrib.distributions
  tfp = tf.contrib.bayesflow

  def target_log_prob_fn(x):
    prob = tf.exp(-tf.reduce_sum(tf.square((x - 5.))))
    prob += tf.exp(-tf.reduce_sum(tf.square((x + 5.))))
    return tf.log(prob)

  x = tf.get_variable("x", initializer=[1.,1.])

  replica_x = []
  for i in range(5):
    replica_x.append(tf.get_variable("x_%d" % i, initializer=[1.,1.]))

  next_x, next_replica_x, kernel_results \
      = tfp.replica_exchange_mc.kernel(
          target_log_prob_fn=target_log_prob_fn,
          current_state=x,
          replica_state=replica_x,
          mode="mh",
          inverse_temperatures=np.logspace(0, -2, 5),
          proposal_fn=tfp.metropolis_hastings.proposal_normal([1., 1.]))

  x_update = x.assign(next_x)
  replica_x_update = [replica_x[i].assign(next_replica_x[i]) for i in range(5)]
  ```
  """

  with ops.name_scope(
      name, "replica_exchange_mc_kernel",
      [current_state, replica_state, mode, inverse_temperatures, seed,
       current_target_log_prob, kwargs]):

    inverse_temperatures = math_ops.to_float(inverse_temperatures)
    n_replica = inverse_temperatures.shape[0]

    replica_state[0] = current_state

    if mode == "mh":
      inner_kernel = metropolis_hastings.kernel
    elif mode == "hmc":
      inner_kernel = hmc.kernel
    else:
      raise ValueError("`mode` must be 'mh' or 'hnc' ")

    sampling_state = []
    inner_kernel_results = []
    for i in range(n_replica):
      next_state, kernel_results = inner_kernel(
          target_log_prob_fn=lambda *x:
              inverse_temperatures[i] * target_log_prob_fn(*x),
          current_state=replica_state[i], seed=seed,
          current_target_log_prob=current_target_log_prob,
          name=name, **kwargs)
      sampling_state.append(next_state)
      inner_kernel_results.append(kernel_results)

    maybe_expand = lambda x: list(x) if _is_list_like(x) else [x]
    sampling_ratio = []
    for i in range(n_replica):
      sampling_state_parts = maybe_expand(sampling_state[i])
      sampling_ratio.append([target_log_prob_fn(*sampling_state_parts)])
    sampling_ratio = array_ops.concat(sampling_ratio, axis=0)

    next_replica_idx = math_ops.range(n_replica)
    exchange_proposed, exchange_proposed_n = exchange_proposed_fn(n_replica)
    i = array_ops.constant(0)

    def cond(i, next_replica_idx):
      return math_ops.less(i, exchange_proposed_n)

    def body(i, next_replica_idx):
      ratio = sampling_ratio[next_replica_idx[exchange_proposed[i, 0]]] - \
          sampling_ratio[next_replica_idx[exchange_proposed[i, 1]]]
      ratio *= inverse_temperatures[exchange_proposed[i, 1]] - \
          inverse_temperatures[exchange_proposed[i, 0]]
      u = random_ops.random_uniform([], dtype=ratio.dtype)
      exchange = math_ops.log(u) < ratio
      exchange_op = sparse_ops.sparse_to_dense(
          [exchange_proposed[i, 0], exchange_proposed[i, 1]], [n_replica],
          [next_replica_idx[exchange_proposed[i, 1]] -
           next_replica_idx[exchange_proposed[i, 0]],
           next_replica_idx[exchange_proposed[i, 0]] -
           next_replica_idx[exchange_proposed[i, 1]]])
      next_replica_idx = control_flow_ops.cond(
          exchange,
          lambda: next_replica_idx + exchange_op,
          lambda: next_replica_idx)
      return [i + 1, next_replica_idx]

    next_replica_idx = control_flow_ops.while_loop(
        cond, body, loop_vars=[i, next_replica_idx])[1]

    next_replica_sample = []
    for i in range(n_replica):
      next_replica_sample.append(
          control_flow_ops.case({math_ops.equal(next_replica_idx[i], j):
                                 _stateful_lambda(sampling_state[j])
                                 for j in range(n_replica)}, exclusive=True))
    next_state = next_replica_sample[0]

    return [
        next_state,
        next_replica_sample,
        KernelResults(
            inner_kernel_results=inner_kernel_results,
            next_replica_idx=next_replica_idx,
            exchange_proposed=exchange_proposed,
            exchange_proposed_n=exchange_proposed_n
        ),
    ]


class _stateful_lambda:
  """Class to use instead of lambda.
  `lambda` is affected by the change of `x`,
  so `_stateful_lambda(x)()`` output `x` at the time of definition.
  """

  def __init__(self, x):
    self.x = x

  def __call__(self):
    return self.x


def _is_list_like(x):
  """Helper which returns `True` if input is `list`-like."""
  return isinstance(x, (tuple, list))
