# Tree

`tree` is a library for working with nested data structures. The `tree` package
is not new, it has previously existed (as `nest`) in libraries such as
[TensorFlow](https://github.com/tensorflow/tensorflow) and
[Sonnet](https://github.com/deepmind/sonnet).

In a way, tree generalizes the builtin map function which only supports flat
sequences, and allows to apply a function to each "leaf" preserving the overall
structure.

Tree is a standalone Python extension with an optimized C++ implementation
making it suitable for use in demanding applications (such as machine learning
models).

Tree is designed to be easy to use:

```python
>>> import tree
>>> structure = [[1], [[[2, 3]]], [4]]

>>> # Extracting leaves from a structure is simple:
>>> tree.flatten(structure)
[1, 2, 3, 4]

>>> # Additionally it is easy map while preserving structure:
>>> double_fn = lambda v: v * 2
>>> tree.map_structure(double_fn, structure)
[[2], [[[4, 6]]], [8]]

>>> # This works for a subset of user-defined types like attrs and namedtuples:
>>> import collections
>>> Foo = collections.namedtuple('Foo', ['a', 'b'])
>>> structure = Foo(a=1, b=2)
>>> tree.map_structure(double_fn, structure)
Foo(a=2, b=4)
```

## Features

- `tree.flatten(..)` extract leaf values from a structure.
- `tree.map_structure(..)` map leaf values from one or more structures while
  preserving the structure of the input.
- `tree.assert_same_structure(..)` test that one or more structures are identical.
- .. and more!

## Installation

Install `tree` by running:

```shell
$ pip install dm-tree
```

## Contribute

- Issue Tracker: github.com/deepmind/tree/issues
- Source Code: github.com/deepmind/tree

## Support

If you are having issues, please let us know.
We have a mailing list located at: dm-tree@google.com

## License

The project is licensed under the Apache license.
