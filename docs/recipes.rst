############
Recipes
############


Concatenate nested array structures
===================================
>>> tree.map_structure(lambda *args: np.concatenate(args, axis=1),
...                    {'a': np.ones((2, 1))},
...                    {'a': np.zeros((2, 1))})
{'a': array([[1., 0.],
       [1., 0.]])}

>>> tree.map_structure(lambda *args: np.concatenate(args, axis=0),
...                    {'a': np.ones((2, 1))},
...                    {'a': np.zeros((2, 1))})
{'a': array([[1.],
       [1.],
       [0.],
       [0.]])}


Exclude "meta" keys while mapping across structures
===================================================
>>> d1 = {'key_to_exclude': None, 'a': 1}
>>> d2 = {'key_to_exclude': None, 'a': 2}
>>> d3 = {'a': 3}
>>> tree.map_structure_up_to({'a': True}, lambda x, y, z: x+y+z, d1, d2, d3)
{'a': 6}


Broadcast a value across a reference structure
==============================================
>>> reference_tree = {'a': 1, 'b': (2, 3)}
>>> value = np.inf
>>> tree.map_structure(lambda _: value, reference_tree)
{'a': inf, 'b': (inf, inf)}
