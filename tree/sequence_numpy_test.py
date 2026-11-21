# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Regression tests for reconstructing NumPy sequences (issue #110)."""

import unittest

import numpy as np
from tree import sequence


class SequenceNumpyTest(unittest.TestCase):

  def test_long_numpy_vector_reconstructs_without_dimension_overflow(self):
    original = np.zeros(52, dtype=np.float32)
    actual = sequence._sequence_like(original, list(range(52)))
    self.assertIsInstance(actual, np.ndarray)
    self.assertEqual(actual.shape, (52,))
    np.testing.assert_array_equal(actual, np.arange(52))

  def test_short_numpy_vector_reconstructs_as_values_not_shape(self):
    actual = sequence._sequence_like(np.zeros(3), [5., 3., 8.])
    np.testing.assert_array_equal(actual, [5., 3., 8.])
    self.assertEqual(actual.shape, (3,))

  def test_original_array_does_not_change(self):
    original = np.zeros(52)
    sequence._sequence_like(original, list(range(52)))
    np.testing.assert_array_equal(original, np.zeros(52))

  def test_custom_leaf_objects_are_not_coerced_to_numpy_scalars(self):
    class TensorLike:
      def __init__(self, data):
        self.data = data

    leaves = [TensorLike(i) for i in range(52)]
    rebuilt = sequence._sequence_like(np.zeros(52), leaves)
    self.assertEqual(rebuilt.shape, (52,))
    self.assertEqual(rebuilt.dtype, object)
    for index, leaf in enumerate(leaves):
      self.assertIs(rebuilt[index], leaf)

  def test_standard_sequences_still_reconstruct(self):
    self.assertEqual(sequence._sequence_like((1, 2), [3, 4]), (3, 4))
    self.assertEqual(sequence._sequence_like([1, 2], [3, 4]), [3, 4])


if __name__ == "__main__":
  unittest.main()
