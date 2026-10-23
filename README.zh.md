# Tree

[English](README.md) | [简体中文](README.zh.md)

`tree`（在 PyPI 上以 `dm-tree` 发布）是一个用于处理嵌套数据结构（nested data structures）的高性能核心库。从某种意义上说，`tree` 推广了 Python 内置仅支持扁平序列的 `map` 函数，允许将任意函数映射应用到每个“叶节点”（leaf），同时严格保持原有的整体嵌套层次结构。

```python
>>> import tree
>>> structure = [[1], [[[2, 3]]], [4]]
>>> tree.flatten(structure)
[1, 2, 3, 4]
>>> tree.map_structure(lambda v: v**2, structure)
[[1], [[[4, 9]]], [16]]
```

`tree` 底层由高度优化的 C++ 实现支撑，非常适合用于对性能要求严苛的应用场景（例如机器学习模型与张量计算框架）。

> **说明**：`tree` 最初作为 TensorFlow 的核心组件（即 `tf.nest`），现作为独立的高性能 C++/Python 库由 Google DeepMind 维护，并作为 JAX、TensorFlow、Haiku、Optax 等现代深度学习生态的底层基石。

## 安装指南 (Installation)

通过 PyPI 安装：

```shell
$ pip install dm-tree
```

直接通过 GitHub 使用 pip 安装：

```shell
$ pip install git+git://github.com/deepmind/tree.git
```

从源码构建：

```shell
$ python setup.py install
```

## 技术支持 (Support)

如果您在使用过程中遇到任何问题，欢迎在我们的 [问题追踪器 (Issue Tracker)](https://github.com/deepmind/tree/issues) 提交 Issue。

## 开源协议 (License)

本项目采用 Apache 2.0 开源许可协议。

---

> **说明**：本文档为基于英文原文的中文翻译版本。如翻译内容与英文原文存在不一致之处，请以 [英文原版 (README.md)](README.md) 为准。
