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
"""Tests for Replica Exchange Monte Carlo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.bayesflow.python.ops import replica_exchange_mc_impl as remc
from tensorflow.contrib.bayesflow.python.ops import metropolis_hastings_impl as mh
from tensorflow.contrib.distributions.python.ops import mvn_tril as mvn_tril_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test


class ReplicaExchangeMCTest(test.TestCase):

  def testKernelStateTensor(self):
    """Test that transition kernel works with tensor input to `state`."""
    loc = variable_scope.get_variable("loc", initializer=0.)
    replica_loc = [variable_scope.get_variable("loc_%d" % i, initializer=0.)
                   for i in range(5)]

    def target_log_prob_fn(loc):
      return normal_lib.Normal(loc=0.0, scale=0.1).log_prob(loc)

    new_state, new_replica_state, _ = remc.kernel(
        target_log_prob_fn=target_log_prob_fn,
        current_state=loc,
        replica_state=replica_loc,
        mode="mh",
        inverse_temperatures=np.logspace(0, -2, 5),
        proposal_fn=mh.proposal_normal(scale=0.05),
        seed=231251)

    loc_update = loc.assign(new_state)
    replica_loc_update = [replica_loc[i].assign(new_replica_state[i])
                          for i in range(5)]

    init = variables.initialize_all_variables()
    with self.test_session() as sess:
      sess.run(init)
      loc_samples = []
      for _ in range(2500):
        loc_sample, replica_loc_sample = sess.run([loc_update,
                                                   replica_loc_update])
        loc_samples.append(loc_sample)
    loc_samples = loc_samples[500:]  # drop samples for burn-in

    self.assertAllClose(np.mean(loc_samples), 0.0, rtol=1e-5, atol=1e-1)
    self.assertAllClose(np.std(loc_samples), 0.1, rtol=1e-5, atol=1e-1)

  def testKernelStateList(self):
    """Test that transition kernel works with list input to `state`."""
    num_chains = 2
    loc_one = variable_scope.get_variable(
        "loc_one", [num_chains],
        initializer=init_ops.zeros_initializer())
    loc_two = variable_scope.get_variable(
        "loc_two", [num_chains], initializer=init_ops.zeros_initializer())
    replica_loc = []
    for i in range(5):
      replica_loc.append(
          [variable_scope.get_variable(
              "loc_one_%d" % i, [num_chains],
              initializer=init_ops.zeros_initializer()),
           variable_scope.get_variable(
              "loc_two_%d" % i, [num_chains],
              initializer=init_ops.zeros_initializer())])

    def target_log_prob_fn(loc_one, loc_two):
      loc = array_ops.stack([loc_one, loc_two])
      log_prob = mvn_tril_lib.MultivariateNormalTriL(
          loc=constant_op.constant([0., 0.]),
          scale_tril=constant_op.constant([[0.1, 0.1], [0.0, 0.1]])).log_prob(
              loc)
      return math_ops.reduce_sum(log_prob, 0)

    def proposal_fn(loc_one, loc_two):
      loc_one_proposal = mh.proposal_normal(scale=0.05)
      loc_two_proposal = mh.proposal_normal(scale=0.05)
      loc_one_sample, _ = loc_one_proposal(loc_one)
      loc_two_sample, _ = loc_two_proposal(loc_two)
      return [loc_one_sample, loc_two_sample], None

    new_state, new_replica_state, _ = remc.kernel(
        target_log_prob_fn=target_log_prob_fn,
        current_state=[loc_one, loc_two],
        replica_state=replica_loc,
        mode="mh",
        inverse_temperatures=np.logspace(0, -2, 5),
        proposal_fn=proposal_fn,
        seed=231251)

    loc_one_update = loc_one.assign(new_state[0])
    loc_two_update = loc_two.assign(new_state[1])
    replica_loc_one_update = \
        [replica_loc[i][0].assign(new_replica_state[i][0]) for i in range(5)]
    replica_loc_two_update = \
        [replica_loc[i][1].assign(new_replica_state[i][1]) for i in range(5)]

    init = variables.initialize_all_variables()
    with self.test_session() as sess:
      sess.run(init)
      loc_one_samples = []
      loc_two_samples = []
      for _ in range(10000):
        loc_one_sample, loc_two_sample, _, _ = sess.run(
            [loc_one_update, loc_two_update,
             replica_loc_one_update, replica_loc_two_update])
        loc_one_samples.append(loc_one_sample)
        loc_two_samples.append(loc_two_sample)

    loc_one_samples = np.array(loc_one_samples)
    loc_two_samples = np.array(loc_two_samples)
    loc_one_samples = loc_one_samples[1000:]  # drop samples for burn-in
    loc_two_samples = loc_two_samples[1000:]  # drop samples for burn-in

    self.assertAllClose(np.mean(loc_one_samples, 0),
                        np.array([0.] * num_chains),
                        rtol=1e-5, atol=1e-1)
    self.assertAllClose(np.mean(loc_two_samples, 0),
                        np.array([0.] * num_chains),
                        rtol=1e-5, atol=1e-1)
    self.assertAllClose(np.std(loc_one_samples, 0),
                        np.array([0.1] * num_chains),
                        rtol=1e-5, atol=1e-1)
    self.assertAllClose(np.std(loc_two_samples, 0),
                        np.array([0.1] * num_chains),
                        rtol=1e-5, atol=1e-1)

  def testDocstringExample(self):
    """Tests the simplified docstring example."""

    def target_log_prob_fn(x):
      prob = math_ops.exp(-math_ops.reduce_sum(math_ops.square((x - 5.))))
      prob += math_ops.exp(-math_ops.reduce_sum(math_ops.square((x + 5.))))
      return math_ops.log(prob)

    x = variable_scope.get_variable("x", initializer=[1., 1.])

    replica_x = []
    for i in range(5):
      replica_x.append(
          variable_scope.get_variable("x_%d" % i, initializer=[1., 1.]))

    next_x, next_replica_x, kernel_results \
        = remc.kernel(
            target_log_prob_fn=target_log_prob_fn,
            current_state=x,
            replica_state=replica_x,
            mode="mh",
            inverse_temperatures=np.logspace(0, -2, 5),
            proposal_fn=mh.proposal_normal([1., 1.]))

    x_update = x.assign(next_x)
    replica_x_update = [replica_x[i].assign(next_replica_x[i])
                        for i in range(5)]

    init = variables.initialize_all_variables()
    with self.test_session() as sess:
      sess.run(init)
      # Run the chains for a total of 1000 steps.
      for _ in range(10):
        sess.run([x_update, replica_x_update])

if __name__ == "__main__":
  test.main()
