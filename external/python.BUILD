# Description:
#   Build rule for Python.
#   This rule works for Debian and Ubuntu. Other platforms might keep the
#   headers in different places.

cc_library(
    name = "python",
    hdrs = glob(["include/python2.7/*.h"]),
    includes = ["include/python2.7"],
    visibility = ["//visibility:public"],
)
