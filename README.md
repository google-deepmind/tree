# Tree

`tree` is a library for working with nested data structures. The `tree` package
is not new, it has previously existed (as `nest`) in libraries such as
[TensorFlow](https://github.com/tensorflow/tensorflow) and
[Sonnet](https://github.com/deepmind/sonnet).

Tree is a standalone Python extension with an optimized C++ implementation
making it suitable for use in demanding applications (such as machine learning
models).

## Quick start

### Installation

```shell
$ pip install dm-tree
```

### Usage

```python
>>> import tree
>>> structure = [[1], [2], [3]]

>>> tree.flatten(structure)
[1, 2, 3]

>>> tree.map_structure(lambda v: v * 2, structure)
[[2], [4], [6]]

>>> import collections
>>> Foo = collections.namedtuple('Foo', ['a', 'b'])
>>> structure = Foo(a=1, b=2)
>>> tree.map_structure(double_fn, structure)
Foo(a=2, b=4)
```
