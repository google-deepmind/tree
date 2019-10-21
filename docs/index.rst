Tree Documentation
==================

.. toctree::
   :maxdepth: 2

``tree`` is a library for working with nested data structures. In a way,
``tree`` generalizes the builtin :func:`map` function which only supports 
flat sequences, and allows to apply a function to each "leaf" preserving 
the overall structure.

Here's a quick example::

  >>> tree.map_structure(lambda v: v**2, [[1], [[[2, 3]]], [4]])
  [[1], [[[4, 9]]], [16]]

.. note::

   ``tree`` has originally been part of TensorFlow and is available
   as ``tf.nest``.

Installation
------------

Install ``tree`` by running::

  $ pip install dm_tree


Reference
---------

All ``tree`` functions operate on nested tree-like structures. A *structure*
is recursively defined as::

  Structure = Union[
      Any,
      Sequence['Structure'],
      Mapping[Any, 'Structure'],
      'AnyNamedTuple',
      'AnyAttrsClass'
  ]

.. TODO(slebedev): support @dataclass classes.

A single (non-nested) Python object is a perfectly valid structure::

  >>> tree.map_structure(lambda v: v * 2, 42)
  84
  >>> tree.flatten(42)
  [42]

You could check whether a structure is actually nested via
:func:`~tree.is_nested`::

  >>> tree.is_nested(42)
  False
  >>> tree.is_nested([42])
  True

Note that ``tree`` only supports acyclic structures. The behavior for
structures with cycle references is undefined.

.. currentmodule:: tree

.. autofunction:: is_nested

.. autofunction:: assert_same_structure

.. autofunction:: pack_sequence_as

.. autofunction:: flatten

.. autofunction:: flatten_up_to

.. autofunction:: flatten_with_tuple_paths

.. autofunction:: flatten_with_tuple_paths_up_to

.. autofunction:: map_structure

.. autofunction:: map_structure_up_to

.. autofunction:: map_structure_with_tuple_paths

.. autofunction:: map_structure_with_tuple_paths_up_to

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/deepmind/tree/issues>`_.

License
-------

Sonnet is licensed under the Apache 2.0 License.
