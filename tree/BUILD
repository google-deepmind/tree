# Description:
#   Tree provides utilities for working with nested data structures.

load(
    "//tree:build_defs.bzl",
    "COMPAT_USERS",
    "tree_py_extension",
    "tree_py_test",
)

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tree",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":_tree",
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
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
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
        # wrapt
    ],
)

tree_py_test(
    name = "tree_benchmark",
    srcs = ["tree_benchmark.py"],
    deps = [
        ":tree",
    ],
)

package_group(
    name = "compat_users",
    packages = COMPAT_USERS,
)

py_library(
    name = "compat",
    srcs = [],  # compat.py
    srcs_version = "PY2AND3",
    visibility = [
        ":compat_users",
    ],
    deps = [
        ":tree",
    ],
)

tree_py_test(
    name = "compat_test",
    srcs = [],  # compat_test.py
    deps = [
        ":compat",
        # util_nest_test_main_lib,
    ],
)
