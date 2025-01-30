#########
Changelog
#########

Version 0.1.9
=============

Released 2025-01-30

* Dropped support for Python <3.10.

Version 0.1.8
=============

Released 2022-12-19

* Bumped pybind11 to v2.10.1 to support Python 3.11.
* Dropped support for Python 3.6.

Version 0.1.7
=============

Released 2022-04-10

* The build is now done via CMake instead of Bazel.

Version 0.1.6
=============

Released 2021-04-12

* Dropped support for Python 2.X.
* Added a generalization of ``tree.traverse`` which keeps track of the
  current path during traversal.

Version 0.1.5
=============

Released 2020-04-30

* Added a new function ``tree.traverse`` which allows to traverse a nested
  structure and apply a function to each subtree.

Version 0.1.4
=============

Released 2020-03-27

* Added support for ``types.MappingProxyType`` on Python 3.X.

Version 0.1.3
=============

Released 2020-01-30

* Fixed ``ImportError`` when ``wrapt`` was not available.

Version 0.1.2
=============

Released 2020-01-29

* Added support for ``wrapt.ObjectWrapper`` objects.
* Added ``StructureKV[K, V]`` and ``Structure = Structure[Text, V]`` types.

Version 0.1.1
=============

Released 2019-11-07

* Ensured that the produced Linux wheels are manylinux2010-compatible.

Version 0.1.0
=============

Released 2019-11-05

* Initial public release.
