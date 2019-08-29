"""Open source rules for building the tree python extension."""

def tree_py_extension(name, srcs, hdrs, copts, features, deps):
    return native.cc_binary(
        name = name + ".so",
        linkshared = 1,
        linkstatic = 1,
        srcs = srcs + hdrs,
        copts = copts,
        features = features,
        deps = deps,
    )

def tree_py_library(name, srcs, srcs_version, tree_extension, visibility, deps = []):
    return native.py_library(
        name = name,
        srcs = srcs,
        srcs_version = srcs_version,
        visibility = visibility,
        data = [tree_extension.lstrip(":") + ".so"],
        deps = deps,
    )

def tree_py_test(name, srcs, deps):
    return native.py_test(
        name = name,
        srcs = srcs,
        deps = deps,
    )
