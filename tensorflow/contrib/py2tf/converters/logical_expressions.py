# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Converter for logical expressions.

e.g. `a and b -> tf.logical_and(a, b)`. This is not done automatically in TF.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import templates
from tensorflow.contrib.py2tf.pyct import transformer


# TODO(mdan): Properly extrack boolean ops according to lazy eval rules.
# Note that this isn't completely safe either, because tensors may have control
# dependencies.
# Note that for loops that should be done after the loop was converted to
# tf.while_loop so that the expanded conditionals are properly scoped.

# Used to signal that an operand is safe for non-lazy evaluation.
SAFE_BOOLEAN_OPERAND = 'SAFE_BOOLEAN_OPERAND'


class LogicalExpressionTransformer(transformer.Base):
  """Converts logical expressions to corresponding TF calls."""

  def __init__(self, context):
    super(LogicalExpressionTransformer, self).__init__(context)
    # TODO(mdan): Look into replacing with bitwise operators instead.
    self.op_mapping = {
        gast.And: 'logical_and',
        gast.Eq: 'equal',
        gast.Gt: 'greater',
        gast.GtE: 'greater_equal',
        gast.Lt: 'less',
        gast.LtE: 'less_equal',
        gast.Not: 'logical_not',
        gast.NotEq: 'not_equal',
        gast.Or: 'logical_or',
        gast.USub: 'negative',
        gast.Is: 'py2tf_utils.dynamic_is',
        gast.IsNot: 'py2tf_utils.dynamic_is_not'
    }

  def _expect_simple_symbol(self, operand):
    if isinstance(operand, gast.Name):
      return
    if anno.hasanno(operand, SAFE_BOOLEAN_OPERAND):
      return
    raise NotImplementedError(
        'only simple local variables are supported in logical and compound '
        'comparison expressions; for example, we support "a or b" but not '
        '"a.x or b"; for a workaround, assign the expression to a local '
        'variable and use that instead, for example "tmp = a.x", "tmp or b"')

  def _matching_tf_op(self, operator):
    op_type = type(operator)
    mapped_op = self.op_mapping.get(op_type)
    if not mapped_op:
      raise NotImplementedError('operator %s is not yet supported' % op_type)
    return mapped_op

  def _inline_tf_op(self, op_name, args):
    if 'py2tf_utils' in op_name:
      # TODO(alexbw): explicitly spelling out the attribute function name
      # until fix for issue highlighted in cl/188931581 lands.
      template = """
      py2tf_utils.op_name(args)
    """
      op_name = op_name.replace('py2tf_utils.', '')
    else:
      template = """
        tf.op_name(args)
      """
    replacement = templates.replace_as_expression(
        template, op_name=op_name, args=args)
    anno.setanno(replacement, SAFE_BOOLEAN_OPERAND, True)
    return replacement

  def visit_Compare(self, node):
    node = self.generic_visit(node)
    ops_and_comps = list(zip(node.ops, node.comparators))
    left = node.left
    op_tree = None

    # Repeated comparisons are converted to conjunctions:
    #   a < b < c   ->   a < b and b < c
    while ops_and_comps:
      op, right = ops_and_comps.pop(0)
      binary_comparison = self._inline_tf_op(self._matching_tf_op(op),
                                             (left, right))
      if isinstance(left, gast.Name) and isinstance(right, gast.Name):
        anno.setanno(binary_comparison, SAFE_BOOLEAN_OPERAND, True)
      if op_tree:
        self._expect_simple_symbol(right)
        op_tree = self._inline_tf_op('logical_and',
                                     (binary_comparison, op_tree))
      else:
        op_tree = binary_comparison
      left = right
    assert op_tree is not None
    return op_tree

  def visit_UnaryOp(self, node):
    node = self.generic_visit(node)
    return self._inline_tf_op(self._matching_tf_op(node.op), node.operand)

  def visit_BoolOp(self, node):
    node = self.generic_visit(node)
    node_values = node.values
    right = node.values.pop()
    self._expect_simple_symbol(right)
    while node_values:
      left = node_values.pop()
      self._expect_simple_symbol(left)
      right = self._inline_tf_op(self._matching_tf_op(node.op), (left, right))
    return right


def transform(node, context):
  return LogicalExpressionTransformer(context).visit(node)
