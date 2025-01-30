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

import os
import platform
import shutil
import subprocess
import sys
import sysconfig

import setuptools
from setuptools.command import build_ext

here = os.path.dirname(os.path.abspath(__file__))


def _get_tree_version():
  """Parse the version string from tree/__init__.py."""
  with open(os.path.join(here, 'tree', '__init__.py')) as f:
    try:
      version_line = next(line for line in f if line.startswith('__version__'))
    except StopIteration:
      raise ValueError('__version__ not defined in tree/__init__.py')
    else:
      ns = {}
      exec(version_line, ns)  # pylint: disable=exec-used
      return ns['__version__']


class CMakeExtension(setuptools.Extension):
  """An extension with no sources.

  We do not want distutils to handle any of the compilation (instead we rely
  on CMake), so we always pass an empty list to the constructor.
  """

  def __init__(self, name, source_dir=''):
    super().__init__(name, sources=[])
    self.source_dir = os.path.abspath(source_dir)


class BuildCMakeExtension(build_ext.build_ext):
  """Our custom build_ext command.

  Uses CMake to build extensions instead of a bare compiler (e.g. gcc, clang).
  """

  def run(self):
    self._check_build_environment()
    for ext in self.extensions:
      self.build_extension(ext)

  def _check_build_environment(self):
    """Check for required build tools: CMake, C++ compiler, and python dev."""
    try:
      subprocess.check_call(['cmake', '--version'])
    except OSError as e:
      ext_names = ', '.join(e.name for e in self.extensions)
      raise RuntimeError(
          f'CMake must be installed to build the following extensions: {ext_names}'
      ) from e
    print('Found CMake')

  def build_extension(self, ext):
    extension_dir = os.path.abspath(
        os.path.dirname(self.get_ext_fullpath(ext.name)))
    build_cfg = 'Debug' if self.debug else 'Release'
    cmake_args = [
        f'-DPython3_ROOT_DIR={sys.prefix}',
        f'-DPython3_EXECUTABLE={sys.executable}',
        f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extension_dir}',
        f'-DCMAKE_BUILD_TYPE={build_cfg}'
    ]
    if platform.system() != 'Windows':
      cmake_args.extend([
          f'-DPython3_LIBRARY={sysconfig.get_paths()["stdlib"]}',
          f'-DPython3_INCLUDE_DIR={sysconfig.get_paths()["include"]}',
      ])
    if platform.system() == 'Darwin' and os.environ.get('ARCHFLAGS'):
      osx_archs = []
      if '-arch x86_64' in os.environ['ARCHFLAGS']:
        osx_archs.append('x86_64')
      if '-arch arm64' in os.environ['ARCHFLAGS']:
        osx_archs.append('arm64')
      cmake_args.append(f'-DCMAKE_OSX_ARCHITECTURES={";".join(osx_archs)}')
    os.makedirs(self.build_temp, exist_ok=True)
    subprocess.check_call(
        ['cmake', '-S', ext.source_dir, '-B', self.build_temp] + cmake_args)
    num_jobs = ()
    if self.parallel:
      num_jobs = (f'-j{self.parallel}',)
    subprocess.check_call([
        'cmake', '--build', self.build_temp, *num_jobs, '--config', build_cfg
    ])

    # Force output to <extension_dir>/. Amends CMake multigenerator output paths
    # on Windows and avoids Debug/ and Release/ subdirs, which is CMake default.
    tree_dir = os.path.join(extension_dir, 'tree')  # pylint:disable=unreachable
    for cfg in ('Release', 'Debug'):
      cfg_dir = os.path.join(extension_dir, cfg)
      if os.path.isdir(cfg_dir):
        for f in os.listdir(cfg_dir):
          shutil.move(os.path.join(cfg_dir, f), tree_dir)


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
    python_requires='>=3.10',
    install_requires=[
        'absl-py>=0.6.1',
        'attrs>=18.2.0',
        'numpy>=1.21',
        "numpy>=1.21.2; python_version>='3.10'",
        "numpy>=1.23.3; python_version>='3.11'",
        "numpy>=1.26.0; python_version>='3.12'",
        "numpy>=2.1.0; python_version>='3.13'",
        'wrapt>=1.11.2',
    ],
    test_suite='tree',
    cmdclass=dict(build_ext=BuildCMakeExtension),
    ext_modules=[CMakeExtension('_tree', source_dir='tree')],
    zip_safe=False,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tree nest flatten',
)
