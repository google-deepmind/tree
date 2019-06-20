workspace(name = "tree")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "abseil-cpp",
    sha256 = "583e5801372a0bb12eb561858532e3bb9a3528f15f65cfc87b2c0f4c1ab1a0ca",
    strip_prefix = "abseil-cpp-111ca7060a6ff50115ca85b59f6b5d8c8c5e9105",
    urls = [
        "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/111ca7060a6ff50115ca85b59f6b5d8c8c5e9105.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/111ca7060a6ff50115ca85b59f6b5d8c8c5e9105.tar.gz",
    ],
)

http_archive(
    name = "pybind11_archive",
    build_file = "pybind11.BUILD",
    sha256 = "b69e83658513215b8d1443544d0549b7d231b9f201f6fc787a2b2218b408181e",
    strip_prefix = "pybind11-2.2.4",
    urls = [
        "https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz",
    ],
)

new_local_repository(
    name = "python_system",
    build_file = "external/python.BUILD",
    path = "/usr",
)
