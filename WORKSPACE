workspace(name = "tree")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "a8677c64e2a58eb113f305784e6af9759cfa3f9a6eacb4d40531fe1bd6356aca",
    strip_prefix = "bazel-skylib-0.9.0",
    url = "https://github.com/bazelbuild/bazel-skylib/archive/0.9.0.zip",
)

http_archive(
    name = "com_google_absl",
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
    sha256 = "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d",
    strip_prefix = "pybind11-2.4.3",
    urls = [
        "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
    ],
)

new_local_repository(
    name = "python_headers",
    build_file = "external/python_headers.BUILD",
    path = "/usr/include/python2.7",  # May be overwritten by setup.py.
)
