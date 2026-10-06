# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Count mappings must retain their keys and mapped values on reconstruction."""

from collections import Counter
from collections import defaultdict
from collections import OrderedDict
import types
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tree


class SubCounter(Counter):
  pass


class CounterReconstructionTest(parameterized.TestCase):

  @parameterized.parameters(Counter, SubCounter)
  def test_map_preserves_count_values_order_and_counter_type(
      self, counter_type
  ):
    value = counter_type({"z": 3, "a": 0, "m": -2})
    original = value.copy()
    result = tree.map_structure(lambda n: 2 * n, value)
    self.assertIs(type(result), counter_type)
    self.assertEqual(dict(result), {"z": 6, "a": 0, "m": -4})
    self.assertEqual(list(result), list(value))
    self.assertEqual(result["missing"], 0)
    self.assertEqual(value, original)

  @parameterized.parameters("map", "path", "up_to", "unflatten", "traverse")
  def test_all_reconstruction_paths_keep_original_count_keys(self, operation):
    source = {"counts": Counter({"pear": 4, "apple": 2}), "other": [7]}
    if operation == "map":
      result = tree.map_structure(lambda n: n + 1, source)
    elif operation == "path":
      result = tree.map_structure_with_path(lambda path, n: n + 1, source)
    elif operation == "up_to":
      result = tree.map_structure_up_to(source, lambda n: n + 1, source)
    elif operation == "unflatten":
      result = tree.unflatten_as(source, [n + 1 for n in tree.flatten(source)])
    else:
      result = tree.traverse(
          lambda n: n + 1 if isinstance(n, int) else None, source
      )
    self.assertEqual(dict(result["counts"]), {"pear": 5, "apple": 3})
    self.assertEqual(result["other"], [8])
    self.assertIsInstance(result["counts"], Counter)

  def test_tensor_count_values_are_not_used_as_mapping_keys(self):
    source = Counter({"left": 2, "right": 3})
    result = tree.map_structure(lambda n: np.array([n, n + 1]), source)
    self.assertEqual(list(result), list(source))
    np.testing.assert_array_equal(result["left"], [2, 3])
    np.testing.assert_array_equal(result["right"], [3, 4])

  def test_two_count_trees_map_without_merging_or_recounting_entries(self):
    a = Counter({"a": 2, "b": 3})
    b = Counter({"b": 5, "a": 7})
    result = tree.map_structure(lambda x, y: x - y, a, b)
    self.assertEqual(dict(result), {"a": -5, "b": -2})

  def test_empty_counter_and_other_mapping_types_keep_existing_behavior(self):
    for source in [
        Counter(),
        {"x": 1},
        OrderedDict([("x", 1)]),
        defaultdict(int, x=1),
        types.MappingProxyType({"x": 1}),
    ]:
      result = tree.map_structure(lambda x: x * 3, source)
      self.assertIs(type(result), type(source))
      self.assertEqual(
          dict(result), {key: value * 3 for key, value in source.items()}
      )


if __name__ == "__main__":
  absltest.main()
