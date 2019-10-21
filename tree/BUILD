# Description:
#   Tree provides utilities for working with tree data structures.

load(
    "//tree:build_defs.bzl",
    "tree_py_library",
    "tree_py_extension",
    "tree_py_test",
)

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

tree_py_library(
    name = "tree",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    tree_extension = ":_tree",
    visibility = ["//visibility:public"],
    deps = [
        # absl/logging
        # six
    ],
)

tree_py_extension(
    name = "_tree",
    srcs = ["tree.cc"],
    hdrs = ["tree.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        # base,
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/strings",
        "@pybind11_archive//:pybind11",
        "@python_headers",
    ],
)

tree_py_test(
    name = "tree_test",
    srcs = ["tree_test.py"],
    deps = [
        ":tree",
        # absl/testing:parameterized
        # attr
        # numpy
    ],
)

tree_py_test(
    name = "tree_benchmark",
    srcs = ["tree_benchmark.py"],
    deps = [
        ":tree",
    ],
)
