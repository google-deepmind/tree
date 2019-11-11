# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
import posixpath
import re
import shutil
import sys

from distutils import sysconfig
import setuptools
from setuptools.command import build_ext


here = os.path.dirname(os.path.abspath(__file__))


IS_WINDOWS = sys.platform.startswith('win')


def _get_tree_version():
  """Parse the version string from tree/__init__.py."""
  with open(os.path.join(here, 'tree', '__init__.py')) as f:
    try:
      version_line = next(
          line for line in f if line.startswith('__version__'))
    except StopIteration:
      raise ValueError('__version__ not defined in tree/__init__.py')
    else:
      ns = {}
      exec(version_line, ns)  # pylint: disable=exec-used
      return ns['__version__']


def _parse_requirements(path):
  with open(os.path.join(here, path)) as f:
    return [
        line.rstrip() for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


class BazelExtension(setuptools.Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(self, bazel_target):
    self.bazel_target = bazel_target
    self.relpath, self.target_name = (
        posixpath.relpath(bazel_target, '//').split(':'))
    ext_name = os.path.join(
        self.relpath.replace(posixpath.sep, os.path.sep), self.target_name)
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
      f.write(re.sub(
          r'(?<=path = ").*(?=",  # May be overwritten by setup\.py\.)',
          sysconfig.get_python_inc().replace(os.path.sep, posixpath.sep),
          workspace_contents))

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    bazel_argv = [
        'bazel',
        'build',
        ext.bazel_target,
        '--symlink_prefix=' + os.path.join(self.build_temp, 'bazel-'),
        '--compilation_mode=' + ('dbg' if self.debug else 'opt'),
    ]

    if IS_WINDOWS:
      # Link with python*.lib.
      for library_dir in self.library_dirs:
        bazel_argv.append('--linkopt=/LIBPATH:' + library_dir)

    self.spawn(bazel_argv)

    shared_lib_suffix = '.dll' if IS_WINDOWS else '.so'
    ext_bazel_bin_path = os.path.join(
        self.build_temp, 'bazel-bin',
        ext.relpath, ext.target_name + shared_lib_suffix)
    ext_dest_path = self.get_ext_fullpath(ext.name)
    ext_dest_dir = os.path.dirname(ext_dest_path)
    if not os.path.exists(ext_dest_dir):
      os.makedirs(ext_dest_dir)
    shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


setuptools.setup(
    name='dm-tree',
    version=_get_tree_version(),
    url='https://github.com/deepmind/tree',
    description='Tree is a library for working with nested data structures.',
    author='DeepMind',
    author_email='tree-copybara@google.com',
    long_description=open(os.path.join(here, 'README.md')).read(),
    long_description_content_type='text/markdown',
    # Contained modules and scripts.
    packages=setuptools.find_packages(),
    install_requires=_parse_requirements('requirements.txt'),
    tests_require=_parse_requirements('requirements-test.txt'),
    test_suite='tree',
    cmdclass=dict(build_ext=BuildBazelExtension),
    ext_modules=[BazelExtension('//tree:_tree')],
    zip_safe=False,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tree nest flatten',
)
