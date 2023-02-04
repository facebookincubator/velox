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

set -efx -o pipefail
# Some of the packages must be build with the same compiler flags
# so that some low level types are the same size. Also, disable warnings.
SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
source $SCRIPTDIR/setup-helper-functions.sh
CPU_TARGET="${CPU_TARGET:-avx}"
NPROC=$(getconf _NPROCESSORS_ONLN)
export CFLAGS=$(get_cxx_flags $CPU_TARGET)  # Used by LZO.
export CXXFLAGS=$CFLAGS  # Used by boost.
export CPPFLAGS=$CFLAGS  # Used by LZO.

function dnf_install {
  dnf install -y -q --setopt=install_weak_deps=False "$@"
}

dnf_install epel-release dnf-plugins-core # For ccache, ninja
dnf config-manager --set-enabled powertools
dnf_install ninja-build ccache gcc-toolset-9 git wget which libevent-devel \
  openssl-devel re2-devel libzstd-devel lz4-devel double-conversion-devel \
  libdwarf-devel curl-devel libicu-devel

dnf remove -y gflags

# Required for Thrift
dnf_install autoconf automake libtool bison flex python3

dnf_install conda

# install sphinx for doc gen
pip3 install sphinx sphinx-tabs breathe sphinx_rtd_theme

# Activate gcc9; enable errors on unset variables afterwards.
source /opt/rh/gcc-toolset-9/enable || exit 1
set -u

function cmake_install {
  cmake -B "$1-build" -GNinja -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="${CFLAGS}" -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release -Wno-dev "$@"
  ninja -C "$1-build" install
}

function wget_and_untar {
  local URL=$1
  local DIR=$2
  mkdir -p "${DIR}"
  wget -q --max-redirect 3 -O - "${URL}" | tar -xz -C "${DIR}" --strip-components=1
}

# untar cmake binary release directly to /usr.
wget_and_untar https://github.com/Kitware/CMake/releases/download/v3.17.5/cmake-3.17.5-Linux-x86_64.tar.gz /usr &

# Fetch sources.
wget_and_untar https://github.com/gflags/gflags/archive/v2.2.2.tar.gz gflags &
wget_and_untar https://github.com/google/glog/archive/v0.4.0.tar.gz glog &
wget_and_untar http://www.oberhumer.com/opensource/lzo/download/lzo-2.10.tar.gz lzo &
wget_and_untar https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.gz boost &
wget_and_untar https://github.com/google/snappy/archive/1.1.8.tar.gz snappy &
wget_and_untar https://github.com/fmtlib/fmt/archive/8.0.1.tar.gz fmt &
wget_and_untar https://github.com/facebook/folly/archive/v2022.11.14.00.tar.gz folly &
#  wget_and_untar https://github.com/ericniebler/range-v3/archive/0.11.0.tar.gz ranges-v3 &
wget_and_untar https://archive.apache.org/dist/hadoop/common/hadoop-2.10.1/hadoop-2.10.1.tar.gz hadoop
wget_and_untar https://github.com/protocolbuffers/protobuf/releases/download/v21.4/protobuf-all-21.4.tar.gz protobuf &

wait  # For cmake and source downloads to complete.

cp -a hadoop /usr/local/

# Build & install.
(
  cd lzo
  ./configure --prefix=/usr --enable-shared --disable-static --docdir=/usr/share/doc/lzo-2.10
  make "-j$(nproc)"
  make install
)

(
  cd boost
  ./bootstrap.sh --prefix=/usr/local
  ./b2 "-j$(nproc)" -d0 install threading=multi
)

(
  cd protobuf
  ./configure --prefix=/usr
  make "-j${NPROC}"
  make install
  ldconfig
)

cmake_install gflags -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON -DLIB_SUFFIX=64 -DCMAKE_INSTALL_PREFIX:PATH=/usr
cmake_install glog -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr
cmake_install snappy -DSNAPPY_BUILD_TESTS=OFF
cmake_install fmt -DFMT_TEST=OFF
cmake_install folly
# cmake_install ranges-v3

dnf clean all
