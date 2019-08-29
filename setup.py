# Copyright 2018 The Tree Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import shutil

from distutils import sysconfig
import setuptools
from setuptools.command import build_ext

__version__ = '0.0.1'

PROJECT_NAME = 'tree'

REQUIRED_PACKAGES = [
    'absl-py >= 0.6.1',
    'attrs >= 18.2.0',
    'numpy >= 1.15.4',
    'six >= 1.12.0',
]

WORKSPACE_PYTHON_HEADERS_PATTERN = re.compile(
    r'(?<=path = ").*(?=",  # May be overwritten by setup\.py\.)')


class BazelExtension(setuptools.Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(self, bazel_target):
    self.bazel_target = bazel_target
    self.relpath, self.target_name = (
        os.path.relpath(bazel_target, '//').split(':'))
    ext_name = os.path.join(self.relpath, self.target_name)
    setuptools.Extension.__init__(self, ext_name, sources=[])


class BuildBazelExtension(build_ext.build_ext):
  """A command that runs Bazel to build a C/C++ extension."""

  def run(self):
    for ext in self.extensions:
      self.bazel_build(ext)
    build_ext.build_ext.run(self)

  def bazel_build(self, ext):
    with open('WORKSPACE', 'r') as f:
      workspace_contents = f.read()

    with open('WORKSPACE', 'w') as f:
      f.write(WORKSPACE_PYTHON_HEADERS_PATTERN.sub(
          sysconfig.get_python_inc(), workspace_contents))

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    shared_lib_suffix = '.' + sysconfig.get_config_var('SO').split('.')[-1]

    self.spawn([
        'bazel',
        'build',
        ext.bazel_target + shared_lib_suffix,
        '--symlink_prefix=' + os.path.join(self.build_temp, 'bazel-'),
        '--compilation_mode=' + ('dbg' if self.debug else 'opt'),
    ])

    ext_bazel_bin_path = os.path.join(
        self.build_temp, 'bazel-bin',
        ext.relpath, ext.target_name + shared_lib_suffix)
    ext_dest_path = self.get_ext_fullpath(ext.name)
    ext_dest_dir = os.path.dirname(ext_dest_path)
    if not os.path.exists(ext_dest_dir):
      os.makedirs(ext_dest_dir)
    shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


setuptools.setup(
    name=PROJECT_NAME,
    version=__version__,
    description='Tree is a library for working with tree data structures.',
    author='DeepMind',
    # Contained modules and scripts.
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    cmdclass=dict(build_ext=BuildBazelExtension),
    ext_modules=[BazelExtension('//tree:_tree')],
    zip_safe=False,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tree nest flatten',
)
