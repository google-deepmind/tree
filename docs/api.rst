#############
API Reference
#############

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

.. autofunction:: unflatten_as

.. autofunction:: flatten

.. autofunction:: flatten_up_to

.. autofunction:: flatten_with_path

.. autofunction:: flatten_with_path_up_to

.. autofunction:: map_structure

.. autofunction:: map_structure_up_to

.. autofunction:: map_structure_with_path

.. autofunction:: map_structure_with_path_up_to
