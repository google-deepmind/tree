
# Copyright 2018 The Tree Authors. All Rights Reserved.
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

"""## Functions for working with objects or nested structures of objects.

This module can perform operations on nest structures. A nest is a Python object
or a nested structure of objects. A nested structure is a Python sequence,
tuple (including `namedtuple`), or dict that can contain further objects or
nested structures.

attr.s decorated classes (http://www.attrs.org) are also supported, in the
same way as `namedtuple`.

single_object ::= <Python object>
nested_structure ::= <list(nest)> | <tuple(nest)> | <dict(nest)>
nest ::= <single_object> | <nested_structure>

The utilities here assume (and do not check) that the nested structures form a
'tree', i.e., it should be a Directed acyclic graph: no references in the
structure of the input of these functions should be recursive.

Example structures:
- `((3, 4), 5, (6, 7, (9, 10), 8))`
- `(np.array(0), (np.array([3, 4]), tf.constant([3, 4])))`


NOTE: this set of utilities is designated to be the final replacement for
tensorflow/python/util/nest.py. Use when possible.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six
from six.moves import zip
from tree import _tree

# Note: this is *not* the same as `six.string_types`, which in Python3 is just
#       `(str,)` (i.e. it does not include byte strings).
_TEXT_OR_BYTES = (six.text_type, six.binary_type)

_SHALLOW_TREE_HAS_INVALID_KEYS = (
    "The shallow_tree's keys are not a subset of the input_tree's keys. The "
    "shallow_tree has the following keys that are not in the input_tree: {}.")

_STRUCTURES_HAVE_MISMATCHING_TYPES = (
    "The two structures don't have the same sequence type. Input structure has "
    "type {input_type}, while shallow structure has type {shallow_type}.")

_STRUCTURES_HAVE_MISMATCHING_LENGTHS = (
    "The two structures don't have the same sequence length. Input "
    "structure has length {input_length}, while shallow structure has length "
    "{shallow_length}."
)

_INPUT_TREE_SMALLER_THAN_SHALLOW_TREE = (
    "The input_tree has fewer elements than the shallow_tree. Input structure "
    "has length {input_size}, while shallow structure has length "
    "{shallow_size}.")

_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ = (
    "If shallow structure is a sequence, input must also be a sequence. "
    "Input has type: {}.")


def _get_attrs_items(obj):
  """Returns a list of (name, value) pairs from an attrs instance.

  The list will be sorted by name.

  Args:
    obj: an object.

  Returns:
    A list of (attr_name, attr_value) pairs.
  """
  return [(attr.name, getattr(obj, attr.name))
          for attr in obj.__class__.__attrs_attrs__]


def _sorted(dictionary):
  """Returns a sorted list of the dict keys, with error if keys not sortable."""
  try:
    return sorted(dictionary)
  except TypeError:
    raise TypeError("nest only supports dicts with sortable keys.")


def _is_attrs(instance):
  return _tree.is_attrs(instance)


def _is_namedtuple(instance, strict=False):
  """Returns True iff `instance` is a `namedtuple`.

  Args:
    instance: An instance of a Python object.
    strict: If True, `instance` is considered to be a `namedtuple` only if
        it is a "plain" namedtuple. For instance, a class inheriting
        from a `namedtuple` will be considered to be a `namedtuple`
        iff `strict=False`.

  Returns:
    True if `instance` is a `namedtuple`.
  """
  return _tree.is_namedtuple(instance, strict)


def _sequence_like(instance, args):
  """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`, or
        `collections.OrderedDict`.
    args: elements to be converted to the `instance` type.

  Returns:
    `args` with the type of `instance`.
  """
  if isinstance(instance, (dict, collections.Mapping)):
    # Pack dictionaries in a deterministic order by sorting the keys.
    # Notice this means that we ignore the original order of `OrderedDict`
    # instances. This is intentional, to avoid potential bugs caused by mixing
    # ordered and plain dicts (e.g., flattening a dict but using a
    # corresponding `OrderedDict` to pack it back).
    result = dict(zip(_sorted(instance), args))
    keys_and_values = ((key, result[key]) for key in instance)
    if isinstance(instance, collections.defaultdict):
      # `defaultdict` requires a default factory as the first argument.
      return type(instance)(instance.default_factory, keys_and_values)
    else:
      return type(instance)(keys_and_values)
  elif _is_namedtuple(instance) or _is_attrs(instance):
    return type(instance)(*args)
  else:
    # Not a namedtuple
    return type(instance)(args)


def _yield_value(iterable):
  for _, v in _yield_sorted_items(iterable):
    yield v


def _yield_sorted_items(iterable):
  """Yield (key, value) pairs for `iterable` in a deterministic order.

  For Sequences, the key will be an int, the array index of a value.
  For Mappings, the key will be the dictionary key.
  For objects (e.g. namedtuples), the key will be the attribute name.

  In all cases, the keys will be iterated in sorted order.

  Args:
    iterable: an iterable.

  Yields:
    The iterable's (key, value) pairs, in order of sorted keys.
  """
  if isinstance(iterable, collections.Mapping):
    # Iterate through dictionaries in a deterministic order by sorting the
    # keys. Notice this means that we ignore the original order of `OrderedDict`
    # instances. This is intentional, to avoid potential bugs caused by mixing
    # ordered and plain dicts (e.g., flattening a dict but using a
    # corresponding `OrderedDict` to pack it back).
    for key in _sorted(iterable):
      yield key, iterable[key]
  elif _is_attrs(iterable):
    for item in _get_attrs_items(iterable):
      yield item
  elif _is_namedtuple(iterable):
    for field in iterable._fields:
      yield (field, getattr(iterable, field))
  else:
    for item in enumerate(iterable):
      yield item


def _num_elements(tree):
  if _is_attrs(tree):
    return len(getattr(tree.__class__, "__attrs_attrs__"))
  else:
    return len(tree)


def is_nested(seq):
  """Returns `True` if its input is a collections.Sequence (except strings).

  Args:
    seq: an input sequence.

  Returns:
    True if the sequence is a not a string and is a collections.Sequence or a
    dict.
  """
  return _tree.is_sequence(seq)


is_sequence = is_nested  # TODO(b/124396266): Need to deprecate


def flatten(nest):
  """Returns a flat list from a given nested structure.

  >>> flatten([[1, 2, 3], [4, [5], [[6]]]])
  [1, 2, 3, 4, 5, 6]

  If `nest` is not a sequence, tuple, or dict, then returns a single-element
  list: `[item]`.

  >>> flatten(None)
  [None]

  >>> flatten(1)
  [1]

  In the case of dict instances, the sequence consists of the values, sorted by
  key to ensure deterministic behavior. This is true also for `OrderedDict`
  instances: their sequence order is ignored, the sorting order of keys is
  used instead. The same convention is followed in `pack_sequence_as`. This
  correctly repacks dicts and `OrderedDict`s after they have been flattened,
  and also allows flattening an `OrderedDict` and then repacking it back using
  a corresponding plain dict, or vice-versa.
  Dictionaries with non-sortable keys cannot be flattened.

  >>> flatten({100: 'world!', 6: 'Hello'})
  ['Hello', 'world!']

  Args:
    nest: an arbitrarily nested structure or a scalar object. Note, numpy
        arrays are considered scalars.

  Returns:
    A list, the flattened version of the input.

  Raises:
    TypeError: The nest is or contains a dict with non-sortable keys.
  """
  return _tree.flatten(nest)


def _same_namedtuples(nest1, nest2):
  """Returns True if the two namedtuples have the same name and fields."""
  return _tree.same_namedtuples(nest1, nest2)


class _DotString(object):

  def __str__(self):
    return "."

  def __repr__(self):
    return "."


_DOT = _DotString()


def assert_same_structure(nest1, nest2, check_types=True):
  """Asserts that two structures are nested in the same way.


  >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
  >>> assert_same_structure(Foo(0, 1), Foo(2, 3))

  Note that namedtuples with identical name and fields are always considered
  to have the same shallow structure (even with `check_types=True`).

  >>> Foo2 = collections.namedtuple('Foo', ['a', 'b'])
  >>> assert_same_structure(Foo(0, 1), Foo2(2, 3))

  Named tuples with different names are considered to have different shallow
  structures:

  >>> Bar = collections.namedtuple('Bar', ['a', 'b'])
  >>> assert_same_structure(Foo(0, 1), Bar(2, 3))
  Traceback (most recent call last):
    ...
  TypeError: The two structures don't have the same nested structure.
  ...

  Args:
    nest1: an arbitrarily nested structure.
    nest2: an arbitrarily nested structure.
    check_types: if `True` (default) types of sequences are checked as
        well, including the keys of dictionaries. If set to `False`, for example
        a list and a tuple of objects will look the same if they have the same
        size. Note that namedtuples with identical name and fields are always
        considered to have the same shallow structure.

  Raises:
    ValueError: If the two structures do not have the same number of elements or
      if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any of
      their substructures. Only possible if `check_types` is `True`.
  """
  try:
    _tree.assert_same_structure(nest1, nest2, check_types)
  except (ValueError, TypeError) as e:
    str1 = str(map_structure(lambda _: _DOT, nest1))
    str2 = str(map_structure(lambda _: _DOT, nest2))
    raise type(e)("%s\n"
                  "Entire first structure:\n%s\n"
                  "Entire second structure:\n%s"
                  % (e, str1, str2))


def flatten_dict_items(dictionary):
  """Returns a dictionary with flattened keys and values.

  This function flattens the keys and values of a dictionary, which can be
  arbitrarily nested structures, and returns the flattened version of such
  structures:

  >>> example_dictionary = {(4, 5, (6, 8)): ("a", "b", ("c", "d"))}
  >>> result = {4: "a", 5: "b", 6: "c", 8: "d"}
  >>> assert flatten_dict_items(example_dictionary) == result

  The input dictionary must satisfy two properties:

  1. Its keys and values should have the same exact nested structure.
  2. The set of all flattened keys of the dictionary must not contain repeated
     keys.

  Args:
    dictionary: the dictionary to zip

  Returns:
    The zipped dictionary.

  Raises:
    TypeError: If the input is not a dictionary.
    ValueError: If any key and value do not have the same structure layout, or
      if keys are not unique.
  """
  if not isinstance(dictionary, (dict, collections.Mapping)):
    raise TypeError("input must be a dictionary")

  flat_dictionary = {}
  for i, v in six.iteritems(dictionary):
    if not is_nested(i):
      if i in flat_dictionary:
        raise ValueError(
            "Could not flatten dictionary: key %s is not unique." % i)
      flat_dictionary[i] = v
    else:
      flat_i = flatten(i)
      flat_v = flatten(v)
      if len(flat_i) != len(flat_v):
        raise ValueError(
            "Could not flatten dictionary. Key had %d elements, but value had "
            "%d elements. Key: %s, value: %s."
            % (len(flat_i), len(flat_v), flat_i, flat_v))
      for new_i, new_v in zip(flat_i, flat_v):
        if new_i in flat_dictionary:
          raise ValueError(
              "Could not flatten dictionary: key %s is not unique."
              % (new_i,))
        flat_dictionary[new_i] = new_v
  return flat_dictionary


def _packed_nest_with_indices(structure, flat, index):
  """Helper function for pack_sequence_as.

  Args:
    structure: Substructure (list / tuple / dict) to mimic.
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.

  Returns:
    The tuple (new_index, child), where:
      * new_index - the updated index into `flat` having processed `structure`.
      * packed - the subset of `flat` corresponding to `structure`,
                 having started at `index`, and packed into the same nested
                 format.

  Raises:
    ValueError: if `structure` contains more elements than `flat`
      (assuming indexing starts from `index`).
  """
  packed = []
  for s in _yield_value(structure):
    if is_nested(s):
      new_index, child = _packed_nest_with_indices(s, flat, index)
      packed.append(_sequence_like(s, child))
      index = new_index
    else:
      packed.append(flat[index])
      index += 1
  return index, packed


def pack_sequence_as(structure, flat_sequence):
  """Returns a given flattened sequence packed into a given structure.

  >>> structure = [[1, 2], [[3], [4]]]
  >>> pack_sequence_as(structure, [5, 6, 7, 8])
  [[5, 6], [[7], [8]]]

  If `structure` is a scalar, `flat_sequence` must be a single-element list;
  in this case the return value is `flat_sequence[0]`.

  >>> pack_sequence_as(None, [1])
  1

  If `structure` is or contains a dict instance, the keys will be sorted to
  pack the flat sequence in deterministic order. This is true also for
  `OrderedDict` instances: their sequence order is ignored, the sorting order of
  keys is used instead. The same convention is followed in `flatten`.
  This correctly repacks dicts and `OrderedDict`s after they have been
  flattened, and also allows flattening an `OrderedDict` and then repacking it
  back using a corresponding plain dict, or vice-versa.
  Dictionaries with non-sortable keys cannot be flattened.

  >>> structure = {1: None, 2: None}
  >>> pack_sequence_as(structure, ['Hello', 'world!'])
  {1: 'Hello', 2: 'world!'}

  Args:
    structure: Nested structure, whose structure is given by nested lists,
        tuples, and dicts. Note: numpy arrays and strings are considered
        scalars.
    flat_sequence: flat sequence to pack.

  Returns:
    packed: `flat_sequence` converted to have the same recursive structure as
      `structure`.

  Raises:
    ValueError: If `flat_sequence` and `structure` have different
      element counts.
    TypeError: `structure` is or contains a dict with non-sortable keys.
  """
  if not is_nested(flat_sequence):
    raise TypeError("flat_sequence must be a sequence not a {}:\n{}".format(
        type(flat_sequence), flat_sequence))

  if not is_nested(structure):
    if len(flat_sequence) != 1:
      raise ValueError("Structure is a scalar but len(flat_sequence) == %d > 1"
                       % len(flat_sequence))
    return flat_sequence[0]

  flat_structure = flatten(structure)
  if len(flat_structure) != len(flat_sequence):
    raise ValueError(
        "Could not pack sequence. Structure had %d elements, but flat_sequence "
        "had %d elements.  Structure: %s, flat_sequence: %s."
        % (len(flat_structure), len(flat_sequence), structure, flat_sequence))

  _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
  return _sequence_like(structure, packed)


def map_structure(func, *structure, **check_types_dict):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(x[0], x[1], ...)` where x[i] is an entry in
  `structure[i]`.  All structures in `structure` must have the same arity,
  and the return value will contain the results with the same structure layout.

  >>> structure = [[1], [2], [3]]
  >>> double_fn = lambda v: v * 2
  >>> map_structure(double_fn, structure)
  [[2], [4], [6]]

  >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
  >>> structure = Foo(a=1, b=2)
  >>> map_structure(double_fn, structure)
  Foo(a=2, b=4)

  Args:
    func: A callable that accepts as many arguments as there are structures.
    *structure: scalar, or tuple or list of constructed scalars and/or other
      tuples/lists, or scalars.  Note: numpy arrays are considered as scalars.
    **check_types_dict: only valid keyword argument is `check_types`. If set to
      `True` (default) the types of iterables within the structures have to be
      same (e.g. `map_structure(func, [1], (1,))` raises a `TypeError`
      exception). To allow this set this argument to `False`.
      Note that namedtuples with identical name and fields are always
      considered to have the same shallow structure.

  Returns:
    A new structure with the same arity as `structure`, whose values correspond
    to `func(x[0], x[1], ...)` where `x[i]` is a value in the corresponding
    location in `structure[i]`. If there are different sequence types and
    `check_types` is `False` the sequence types of the first structure will be
    used.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    TypeError: If `check_types` is not `False` and the two structures differ in
      the type of sequence in any of their substructures.
    ValueError: If no structure is provided or if a keyword argument other than
      `check_types` is provided.
  """
  if not callable(func):
    raise TypeError("func must be callable, got: %s" % func)

  if not structure:
    raise ValueError("Must provide at least one structure")

  if check_types_dict:
    if "check_types" not in check_types_dict or len(check_types_dict) > 1:
      raise ValueError("Only valid keyword argument is check_types")
    check_types = check_types_dict["check_types"]
  else:
    check_types = True

  for other in structure[1:]:
    assert_same_structure(structure[0], other, check_types=check_types)

  flat_structure = [flatten(s) for s in structure]
  entries = zip(*flat_structure)

  return pack_sequence_as(
      structure[0], [func(*x) for x in entries])


def map_structure_with_paths(func, *structure, **kwargs):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(path, x[0], x[1], ..., **kwargs)` where x[i] is an entry in
  `structure[i]` and `path` is the common path to x[i] in the structures.  All
  structures in `structure` must have the same arity, and the return value will
  contain the results with the same structure layout.

  Args:
    func: A callable with the signature func(path, *values, **kwargs) that is
      evaluated on the leaves of the structure.
    *structure: A variable number of compatible structures to process.
    **kwargs: Optional kwargs to be passed through to func. Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.
      `map_structure(func, [1], (1,))` raises a `TypeError` exception). To allow
      this set this argument to `False`.

  Returns:
    A structure of the same form as the input structures whose leaves are the
    result of evaluating func on corresponding leaves of the input structures.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    TypeError: If `check_types` is not `False` and the two structures differ in
      the type of sequence in any of their substructures.
    ValueError: If no structures are provided.
  """
  def wrapper_func(tuple_path, *inputs, **kwargs):
    string_path = "/".join(str(s) for s in tuple_path)
    return func(string_path, *inputs, **kwargs)

  return map_structure_with_tuple_paths_up_to(structure[0],
                                              wrapper_func,
                                              *structure,
                                              **kwargs)


def map_structure_with_tuple_paths(func, *structure, **kwargs):
  """Applies `func` to each entry in `structure` and returns a new structure.

  Applies `func(tuple_path, x[0], x[1], ..., **kwargs)` where `x[i]` is an entry
  in `structure[i]` and `tuple_path` is a tuple of indices and/or dictionary
  keys (as returned by `nest.yield_flat_paths`), which uniquely specifies the
  common path to x[i] in the structures. All structures in `structure` must have
  the same arity, and the return value will contain the results in the same
  structure.

  Args:
    func: A callable with the signature `func(tuple_path, *values, **kwargs)`
      that is evaluated on the leaves of the structure.
    *structure: A variable number of compatible structures to process.
    **kwargs: Optional kwargs to be passed through to func. Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.
      `map_structure(func, [1], (1,))` raises a `TypeError` exception). To allow
      this set this argument to `False`.

  Returns:
    A structure of the same form as the input structures whose leaves are the
    result of evaluating func on corresponding leaves of the input structures.

  Raises:
    TypeError: If `func` is not callable or if the structures do not match
      each other by depth tree.
    TypeError: If `check_types` is not `False` and the two structures differ in
      the type of sequence in any of their substructures.
    ValueError: If no structures are provided.
  """
  return map_structure_with_tuple_paths_up_to(structure[0],
                                              func,
                                              *structure,
                                              **kwargs)


def _yield_flat_up_to(shallow_tree, input_tree, path=()):
  """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

  Args:
    shallow_tree: Nested structure. Traverse no further than its leaf nodes.
    input_tree: Nested structure. Return the paths and values from this tree.
      Must have the same upper structure as shallow_tree.
    path: Tuple. Optional argument, only used when recursing. The path from the
      root of the original shallow_tree, down to the root of the shallow_tree
      arg of this recursive call.

  Yields:
    Pairs of (path, value), where path the tuple path of a leaf node in
    shallow_tree, and value is the value of the corresponding node in
    input_tree.
  """
  if (isinstance(shallow_tree, _TEXT_OR_BYTES) or
      not (isinstance(shallow_tree, (collections.Mapping,
                                     collections.Sequence)) or
           _is_namedtuple(shallow_tree) or
           _is_attrs(shallow_tree))):
    yield (path, input_tree)
  else:
    input_tree = dict(_yield_sorted_items(input_tree))
    for shallow_key, shallow_subtree in _yield_sorted_items(shallow_tree):
      subpath = path + (shallow_key,)
      input_subtree = input_tree[shallow_key]
      for leaf_path, leaf_value in _yield_flat_up_to(shallow_subtree,
                                                     input_subtree,
                                                     path=subpath):
        yield (leaf_path, leaf_value)


def assert_shallow_structure(shallow_tree, input_tree, check_types=True,
                             check_subtrees_length=True):
  """Asserts that `shallow_tree` is a shallow structure of `input_tree`.

  That is, this function recursively tests if each key in shallow_tree has its
  corresponding key in input_tree.

  Examples:

  The following code will raise an exception:

  >>> shallow_tree = {"a": "A", "b": "B"}
  >>> input_tree = {"a": 1, "c": 2}
  >>> assert_shallow_structure(shallow_tree, input_tree)
  Traceback (most recent call last):
    ...
  ValueError: The shallow_tree's keys are not a subset of the input_tree's ...

  The following code will raise an exception:

  >>> shallow_tree = ["a", "b"]
  >>> input_tree = ["c", ["d", "e"], "f"]
  >>> assert_shallow_structure(shallow_tree, input_tree)
  Traceback (most recent call last):
    ...
  ValueError: The two structures don't have the same sequence length. ...

  The following code will NOT raise an exception:

  >>> shallow_tree = ["a", "b"]
  >>> input_ = ["c", ["d", "e"], "f"]
  >>> assert_shallow_structure(shallow_tree, input_,check_subtrees_length=False)

  By setting check_types=False, we drop the requirement that corresponding
  nodes in shallow_tree and input_tree have to be the same type. Sequences
  are treated equivalently to Mappables that map integer keys (indices) to
  values. The following code will therefore not raise an exception:

  >>> assert_shallow_structure({0: "foo"}, ["foo"], check_types=False)

  Args:
    shallow_tree: an arbitrarily nested structure.
    input_tree: an arbitrarily nested structure.
    check_types: if `True` (default) the sequence types of `shallow_tree` and
      `input_tree` have to be the same.
    check_subtrees_length: if `True` (default) the subtrees `shallow_tree` and
      `input_tree` have to be the same length. If `False` sequences are treated
      as key-value like mappings allowing them to be considered as valid
      subtrees. Note that this may drop parts of the `input_tree`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`. Only raised if `check_types` is `True`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  if is_nested(shallow_tree):
    if not is_nested(input_tree):
      raise TypeError(_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(
          type(input_tree)))

    if check_types and not isinstance(input_tree, type(shallow_tree)):
      # Duck-typing means that nest should be fine with two different
      # namedtuples with identical name and fields.
      shallow_is_namedtuple = _is_namedtuple(shallow_tree, False)
      input_is_namedtuple = _is_namedtuple(input_tree, False)
      if shallow_is_namedtuple and input_is_namedtuple:
        if not _same_namedtuples(shallow_tree, input_tree):
          raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
              input_type=type(input_tree),
              shallow_type=type(shallow_tree)))
      elif not (isinstance(shallow_tree, collections.Mapping)
                and isinstance(input_tree, collections.Mapping)):
        raise TypeError(_STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            input_type=type(input_tree),
            shallow_type=type(shallow_tree)))

    if check_subtrees_length and _num_elements(input_tree) != _num_elements(
        shallow_tree):
      raise ValueError(
          _STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
              input_length=len(input_tree), shallow_length=len(shallow_tree)))
    elif _num_elements(input_tree) < _num_elements(shallow_tree):
      raise ValueError(
          _INPUT_TREE_SMALLER_THAN_SHALLOW_TREE.format(
              input_size=_num_elements(input_tree),
              shallow_size=_num_elements(shallow_tree)))

    shallow_iter = _yield_sorted_items(shallow_tree)
    input_iter = _yield_sorted_items(input_tree)

    def get_matching_input_branch(shallow_key):
      for input_key, input_branch in input_iter:
        if input_key == shallow_key:
          return input_branch

      raise ValueError(_SHALLOW_TREE_HAS_INVALID_KEYS.format([shallow_key]))

    for shallow_key, shallow_branch in shallow_iter:
      input_branch = get_matching_input_branch(shallow_key)
      assert_shallow_structure(shallow_branch,
                               input_branch,
                               check_types=check_types,
                               check_subtrees_length=check_subtrees_length)


def flatten_up_to(shallow_tree, input_tree, check_types=True,
                  check_subtrees_length=True):
  """Flattens `input_tree` up to `shallow_tree`.

  Any further depth in structure in `input_tree` is retained as elements in the
  partially flatten output.

  If `shallow_tree` and `input_tree` are not sequences, this returns a
  single-element list: `[input_tree]`.

  Use Case:

  Sometimes we may wish to partially flatten a nested sequence, retaining some
  of the nested structure. We achieve this by specifying a shallow structure,
  `shallow_tree`, we wish to flatten up to.

  The input, `input_tree`, can be thought of as having the same structure layout
  as `shallow_tree`, but with leaf nodes that are themselves tree structures.

  Examples:

  ```python
  input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
  shallow_tree = [[True, True], [False, True]]

  flattened_input_tree = flatten_up_to(shallow_tree, input_tree)
  flattened_shallow_tree = flatten_up_to(shallow_tree, shallow_tree)

  # Output is:
  # [[2, 2], [3, 3], [4, 9], [5, 5]]
  # [True, True, False, True]
  ```

  ```python
  input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
  shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]

  input_tree_flattened_as_shallow_tree = flatten_up_to(shallow_tree, input_tree)
  input_tree_flattened = flatten(input_tree)

  # Output is:
  # [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
  # ['a', 1, 'b', 2, 'c', 3, 'd', 4]
  ```

  Non-Sequence Edge Cases:

  ```python
  flatten_up_to(0, 0)  # Output: [0]
  flatten_up_to(0, [0, 1, 2])  # Output: [[0, 1, 2]]
  flatten_up_to([0, 1, 2], 0)  # Output: TypeError
  flatten_up_to([0, 1, 2], [0, 1, 2])  # Output: [0, 1, 2]
  ```

  Args:
    shallow_tree: a possibly pruned structure of input_tree.
    input_tree: an arbitrarily nested structure or a scalar object.
      Note, numpy arrays are considered scalars.
    check_types: bool. If True, check that each node in shallow_tree has the
      same type as the corresponding node in input_tree.
    check_subtrees_length: if `True` (default) the subtrees `shallow_tree` and
      `input_tree` have to be the same length. If `False` sequences are treated
      as key-value like mappings allowing them to be considered as valid
      subtrees. Note that this may drop parts of the `input_tree`.

  Returns:
    A list, the partially flattened version of `input_tree` according to
    the structure of `shallow_tree`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  assert_shallow_structure(shallow_tree, input_tree, check_types=check_types,
                           check_subtrees_length=check_subtrees_length)
  # Discard paths returned by _yield_flat_up_to.
  return [v for _, v in _yield_flat_up_to(shallow_tree, input_tree)]


def flatten_with_tuple_paths_up_to(shallow_tree, input_tree, check_types=True,
                                   check_subtrees_length=True):
  """Flattens `input_tree` up to `shallow_tree`.

  Any further depth in structure in `input_tree` is retained as elements in the
  partially flattened output.

  Returns a list of (path, value) pairs, where value a leaf node in the
  flattened tree, and path is the tuple path of that leaf in input_tree.

  If `shallow_tree` and `input_tree` are not sequences, this returns a
  single-element list: `[((), input_tree)]`.

  Use Case:

  Sometimes we may wish to partially flatten a nested sequence, retaining some
  of the nested structure. We achieve this by specifying a shallow structure,
  `shallow_tree`, we wish to flatten up to.

  The input, `input_tree`, can be thought of as having the same structure layout
  as `shallow_tree`, but with leaf nodes that are themselves tree structures.

  Examples:

  ```python
  input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
  shallow_tree = [[True, True], [False, True]]

  flattened_input_tree = flatten_with_tuple_paths_up_to(shallow_tree,
                                                        input_tree)
  flattened_shallow_tree = flatten_with_tuple_paths_up_to(shallow_tree,
                                                          shallow_tree)

  # Output is:
  # [((0, 0), [2, 2]),
  #  ((0, 1), [3, 3]),
  #  ((1, 0), [4, 9]),
  #  ((1, 1), [5, 5])]
  #
  # [((0, 0), True),
  #  ((0, 1), True),
  #  ((1, 0), False),
  #  ((1, 1), True)]
  ```

  ```python
  input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]
  shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]

  input_tree_flattened_as_shallow_tree = flatten_up_to(shallow_tree, input_tree)
  input_tree_flattened = flatten(input_tree)

  # Output is:
  # [((0, 0), ('a', 1)),
  #  ((0, 1, 0), ('b', 2)),
  #  ((0, 1, 1, 0), ('c', 3)),
  #  ((0, 1, 1, 1), ('d', 4))]
  # ['a', 1, 'b', 2, 'c', 3, 'd', 4]
  ```

  Non-Sequence Edge Cases:

  ```python
  flatten_with_tuple_paths_up_to(0, 0)  # Output: [(), 0]

  flatten_with_tuple_paths_up_to(0, [0, 1, 2])  # Output: [(), [0, 1, 2]]

  flatten_with_tuple_paths_up_to([0, 1, 2], 0)  # Output: TypeError

  flatten_with_tuple_paths_up_to([0, 1, 2], [0, 1, 2])
  # Output: [((0,) 0), ((1,), 1), ((2,), 2)]
  ```

  Args:
    shallow_tree: a possibly pruned structure of input_tree.
    input_tree: an arbitrarily nested structure or a scalar object.
      Note, numpy arrays are considered scalars.
    check_types: bool. If True, check that each node in shallow_tree has the
      same type as the corresponding node in input_tree.
    check_subtrees_length: if `True` (default) the subtrees `shallow_tree` and
      `input_tree` have to be the same length. If `False` sequences are treated
      as key-value like mappings allowing them to be considered as valid
      subtrees. Note that this may drop parts of the `input_tree`.

  Returns:
    A list, the partially flattened version of `input_tree` according to
    the structure of `shallow_tree`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.
  """
  assert_shallow_structure(shallow_tree, input_tree, check_types=check_types,
                           check_subtrees_length=check_subtrees_length)
  return list(_yield_flat_up_to(shallow_tree, input_tree))


def apply_to_structure(branch_fn, leaf_fn, structure):
  """`apply_to_structure` applies branch_fn and leaf_fn to branches and leaves.

  This function accepts two separate callables depending on whether the
  structure is a sequence.

  Args:
    branch_fn: A function to call on a struct if is_nested(struct) is `True`.
    leaf_fn: A function to call on a struct if is_nested(struct) is `False`.
    structure: A nested structure containing arguments to be applied to.

  Returns:
    A nested structure of function outputs.

  Raises:
    TypeError: If `branch_fn` or `leaf_fn` is not callable.
    ValueError: If no structure is provided.
  """
  if not callable(leaf_fn):
    raise TypeError("leaf_fn must be callable, got: %s" % leaf_fn)

  if not callable(branch_fn):
    raise TypeError("branch_fn must be callable, got: %s" % branch_fn)

  if not is_nested(structure):
    return leaf_fn(structure)

  processed = branch_fn(structure)

  new_structure = [
      apply_to_structure(branch_fn, leaf_fn, value)
      for value in _yield_value(processed)
  ]
  return _sequence_like(processed, new_structure)


def map_structure_up_to(shallow_tree, func, *inputs, **kwargs):
  """Applies a function or op to a number of partially flattened inputs.

  The `inputs` are flattened up to `shallow_tree` before being mapped.

  Use Case:

  Sometimes we wish to apply a function to a partially flattened
  sequence (for example when the function itself takes sequence inputs). We
  achieve this by specifying a shallow structure, `shallow_tree` we wish to
  flatten up to.

  The `inputs`, can be thought of as having the same structure layout as
  `shallow_tree`, but with leaf nodes that are themselves tree structures.

  This function therefore will return something with the same base structure as
  `shallow_tree`.

  Examples:

  ```python
  shallow_tree = [None, None]
  inp_val = [1, 2, 3]
  out = map_structure_up_to(shallow_tree, lambda x: 2 * x, inp_val)

  # Output is: [2, 4]
  ```

  ```python
  ab_tuple = collections.namedtuple("ab_tuple", "a, b")
  op_tuple = collections.namedtuple("op_tuple", "add, mul")
  inp_val = ab_tuple(a=2, b=3)
  inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
  out = map_structure_up_to(inp_val, lambda val, ops: (val + ops.add) * ops.mul,
                            inp_val, inp_ops)

  # Output is: ab_tuple(a=6, b=15)
  ```

  ```python
  data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
  name_list = ['evens', ['odds', 'primes']]
  out = map_structure_up_to(
      name_list,
      lambda name, sec: "first_{}_{}".format(len(sec), name),
      name_list, data_list)

  # Output is: ['first_4_evens', ['first_5_odds', 'first_3_primes']]
  ```

  Args:
    shallow_tree: a shallow tree, common to all the inputs.
    func: callable which will be applied to each input individually.
    *inputs: arbitrarily nested combination of objects that are compatible with
        shallow_tree. The function `func` is applied to corresponding
        partially flattened elements of each input, so the function must support
        arity of `len(inputs)`.
    **kwargs: kwargs to feed to func(). Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.
      `map_structure(func, [1], (1,))` raises a `TypeError` exception). To allow
      this set this argument to `False`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.

  Returns:
    result of repeatedly applying `func`, with the same structure layout as
    `shallow_tree`.
  """
  return map_structure_with_tuple_paths_up_to(
      shallow_tree,
      lambda _, *values: func(*values),  # Discards the path arg.
      *inputs,
      **kwargs)


def map_structure_with_tuple_paths_up_to(shallow_tree, func, *inputs, **kwargs):
  """Applies a function or op to a number of partially flattened inputs.

  Like map_structure_up_to(), except that the 'func' argument takes a path
  tuple as its first argument, followed by the corresponding values from
  *inputs.

  Args:
    shallow_tree: a shallow tree, common to all the inputs.
    func: callable that takes args (path, inputs_0_value, ... , inputs_N_value),
      where path is a tuple path to a leaf node in shallow_tree, and
      inputs_i_value is the corresponding value from inputs[i].
    *inputs: nested structures that are all structurally compatible with
        shallow_tree.
    **kwargs: kwargs to feed to func(). Special kwarg
      `check_types` is not passed to func, but instead determines whether the
      types of iterables within the structures have to be same (e.g.
      `map_structure(func, [1], (1,))` raises a `TypeError` exception). To allow
      this set this argument to `False`.

  Raises:
    TypeError: If `shallow_tree` is a sequence but one of `*inputs` is not.
    TypeError: If the sequence types of `shallow_tree` are different from
      `input_tree`.
    ValueError: If the sequence lengths of `shallow_tree` are different from
      `input_tree`.

  Returns:
    Result of repeatedly applying `func`. Has the same structure layout
    as `shallow_tree`.
  """
  if not inputs:
    raise ValueError("Cannot map over no sequences")

  check_types = kwargs.pop("check_types", True)
  check_subtrees_length = kwargs.pop("check_subtrees_length", True)

  for input_tree in inputs:
    assert_shallow_structure(
        shallow_tree,
        input_tree,
        check_types=check_types,
        check_subtrees_length=check_subtrees_length)

  # Flatten each input separately, apply the function to corresponding elements,
  # then repack based on the structure of the first input.
  flat_value_lists = [
      flatten_up_to(  # pylint: disable=g-complex-comprehension
          shallow_tree,
          input_tree,
          check_types,
          check_subtrees_length=check_subtrees_length) for input_tree in inputs
  ]
  flat_path_list = [path for path, _
                    in _yield_flat_up_to(shallow_tree, inputs[0])]
  results = [func(*args, **kwargs) for args in zip(flat_path_list,
                                                   *flat_value_lists)]
  return pack_sequence_as(structure=shallow_tree, flat_sequence=results)


def get_traverse_shallow_structure(traverse_fn, structure):
  """Generates a shallow structure from a `traverse_fn` and `structure`.

  `traverse_fn` must accept any possible subtree of `structure` and return
  a depth=1 structure containing `True` or `False` values, describing which
  of the top-level subtrees may be traversed.  It may also
  return scalar `True` or `False` "traversal is OK / not OK for all subtrees."

  Examples are available in the unit tests (nest_test.py).

  Args:
    traverse_fn: Function taking a substructure and returning either a scalar
      `bool` (whether to traverse that substructure or not) or a depth=1
      shallow structure of the same type, describing which parts of the
      substructure to traverse.
    structure: The structure to traverse.

  Returns:
    A shallow structure containing python bools, which can be passed to
    `map_structure_up_to` and `flatten_up_to`.

  Raises:
    TypeError: if `traverse_fn` returns a sequence for a non-sequence input,
      or a structure with depth higher than 1 for a sequence input,
      or if any leaf values in the returned structure or scalar are not type
      `bool`.
  """
  to_traverse = traverse_fn(structure)
  if not is_nested(structure):
    if not isinstance(to_traverse, bool):
      raise TypeError("traverse_fn returned structure: %s for non-structure: %s"
                      % (to_traverse, structure))
    return to_traverse
  level_traverse = []
  if isinstance(to_traverse, bool):
    if not to_traverse:
      # Do not traverse this substructure at all.  Exit early.
      return False
    else:
      # Traverse the entire substructure.
      for branch in _yield_value(structure):
        level_traverse.append(
            get_traverse_shallow_structure(traverse_fn, branch))
  elif not is_nested(to_traverse):
    raise TypeError("traverse_fn returned a non-bool scalar: %s for input: %s"
                    % (to_traverse, structure))
  else:
    # Traverse some subset of this substructure.
    assert_shallow_structure(to_traverse, structure)
    for t, branch in zip(_yield_value(to_traverse), _yield_value(structure)):
      if not isinstance(t, bool):
        raise TypeError(
            "traverse_fn didn't return a depth=1 structure of bools.  saw: %s "
            " for structure: %s" % (to_traverse, structure))
      if t:
        level_traverse.append(
            get_traverse_shallow_structure(traverse_fn, branch))
      else:
        level_traverse.append(False)
  return _sequence_like(structure, level_traverse)


def yield_flat_paths(nest):
  """Yields paths for some nested structure.

  Paths are lists of objects which can be str-converted, which may include
  integers or other types which are used as indices in a dict.

  The flat list will be in the corresponding order as if you called
  `flatten` on the structure. This is handy for naming Tensors such
  the TF scope structure matches the tuple structure.

  E.g. if we have a tuple `value = Foo(a=3, b=Bar(c=23, d=42))`

  >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
  >>> Bar = collections.namedtuple('Bar', ['c', 'd'])
  >>> value = Foo(a=3, b=Bar(c=23, d=42))

  >>> flatten(value)
  [3, 23, 42]

  >>> list(yield_flat_paths(value))
  [('a',), ('b', 'c'), ('b', 'd')]

  >>> list(yield_flat_paths({'a': [3]}))
  [('a', 0)]

  >>> list(yield_flat_paths({'a': 3}))
  [('a',)]

  Args:
    nest: the value to produce a flattened paths list for.

  Yields:
    Tuples containing index or key values which form the path to a specific
      leaf value in the nested structure.
  """
  for k, _ in _yield_flat_up_to(nest, nest):
    yield k


def flatten_with_joined_string_paths(structure, separator="/"):
  """Returns a list of (string path, data element) tuples.

  The order of tuples produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each data element was located. See `nest.yield_flat_paths`
  for more information.

  Args:
    structure: the nested structure to flatten.
    separator: string to separate levels of hierarchy in the results, defaults
      to '/'.

  Returns:
    A list of (string, data element) tuples.
  """
  flat_paths = yield_flat_paths(structure)
  def stringify_and_join(path_elements):
    return separator.join(str(path_element) for path_element in path_elements)
  flat_string_paths = [stringify_and_join(path) for path in flat_paths]
  return list(zip(flat_string_paths, flatten(structure)))


def flatten_with_tuple_paths(structure):
  """Returns a list of `(tuple_path, leaf_element)` tuples.

  The order of pairs produced matches that of `nest.flatten`. This allows you
  to flatten a nested structure while keeping information about where in the
  structure each data element was located. See `nest.yield_flat_paths`
  for more information about tuple paths.

  Args:
    structure: the nested structure to flatten.

  Returns:
    A list of `(tuple_path, leaf_element)` tuples. Each `tuple_path` is a tuple
    of indices and/or dictionary keys that uniquely specify the path to
    `leaf_element` within `structure`.
  """
  flat_paths = yield_flat_paths(structure)
  return list(zip(flat_paths, flatten(structure)))
