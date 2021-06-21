#!/bin/bash

mkdir tmp && pushd tmp
git clone https://github.com/stellaraccident/manylinux-bazel
pushd manylinux-bazel
bash ./setup_dev_env_2014.sh
bash ./fetch_sources.sh
bash ./patch_gcc.sh
export PATH=/opt/python/cp38-cp38/bin:$PATH
bash ./build.sh
popd && popd
