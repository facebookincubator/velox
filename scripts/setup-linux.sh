#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
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

set -eufx -o pipefail
SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
source $SCRIPTDIR/setup-common.sh

function install_hdfs {
  github_checkout apache/hawq master
  libhdfs3_dir=$DEPENDENCY_DIR/hawq/depends/libhdfs3
  sed -i "/FIND_PACKAGE(GoogleTest REQUIRED)/d" $libhdfs3_dir/CMakeLists.txt
  sed -i "s/dumpversion/dumpfullversion/" $libhdfs3_dir/CMake/Platform.cmake
  # Dependencies for Hadoop testing
  wget_and_untar https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz hadoop
  ${SUDO} cp -a hadoop /usr/local/
  cmake_install ${libhdfs3_dir}
}
