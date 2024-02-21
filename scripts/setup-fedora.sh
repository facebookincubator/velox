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
FB_OS_VERSION=v2023.12.04.00
FMT_VERSION=10.1.1
DEPENDENCY_DIR=${DEPENDENCY_DIR:-$(pwd)}
export CMAKE_BUILD_TYPE=Release

function dnf_install {
 dnf install -y -q --setopt=install_weak_deps=False "$@"
}

dnf_install ninja-build ccache gcc g++ git wget which libevent-devel \
 openssl-devel re2-devel libzstd-devel lz4-devel double-conversion-devel \
 libdwarf-devel curl-devel cmake libicu-devel python3-devel zlib-devel boost-devel libsodium-devel

dnf remove -y gflags

# Required for Thrift
dnf_install autoconf automake libtool bison flex python3


dnf_install conda

set -u

function cmake_install_deps {
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

# Fetch sources.
wget_and_untar https://github.com/gflags/gflags/archive/v2.2.2.tar.gz gflags &
wget_and_untar https://github.com/google/glog/archive/v0.4.0.tar.gz glog &
wget_and_untar http://www.oberhumer.com/opensource/lzo/download/lzo-2.10.tar.gz lzo &
wget_and_untar https://github.com/google/snappy/archive/1.1.8.tar.gz snappy &


wait  # For cmake and source downloads to complete.

# Build & install.
(
 cd lzo
 ./configure --prefix=/usr --enable-shared --disable-static --docdir=/usr/share/doc/lzo-2.10
 make "-j$(nproc)"
 make install
)

# (
#  cd boost
#  ./bootstrap.sh --prefix=/usr/local
#  ./b2 "-j$(nproc)" -d0 install threading=multi
# )

cmake_install_deps gflags -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON -DLIB_SUFFIX=64 -DCMAKE_INSTALL_PREFIX:PATH=/usr
cmake_install_deps glog -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr
cmake_install_deps snappy -DSNAPPY_BUILD_TESTS=OFF


dnf clean all

function install_fmt {
  github_checkout fmtlib/fmt "${FMT_VERSION}"
  cmake_install -DFMT_TEST=OFF
}

function install_folly {
  github_checkout facebook/folly "${FB_OS_VERSION}"
  cmake_install -DBUILD_TESTS=OFF -DFOLLY_HAVE_INT128_T=ON
}

function install_fizz {
  github_checkout facebookincubator/fizz "${FB_OS_VERSION}"
  cmake_install -DBUILD_TESTS=OFF -S fizz
}

function install_wangle {
  github_checkout facebook/wangle "${FB_OS_VERSION}"
  cmake_install -DBUILD_TESTS=OFF -S wangle
}

function install_mvfst {
  github_checkout facebook/mvfst "${FB_OS_VERSION}"
  cmake_install -DBUILD_TESTS=OFF
}

function install_fbthrift {
  github_checkout facebook/fbthrift "${FB_OS_VERSION}"
  cmake_install -DBUILD_TESTS=OFF
}

function install_velox_deps {
  run_and_time install_fmt
  run_and_time install_folly
  run_and_time install_fizz
  run_and_time install_wangle
  run_and_time install_mvfst
  run_and_time install_fbthrift
  run_and_time install_conda
}

(return 2> /dev/null) && return # If script was sourced, don't run commands.

(
  if [[ $# -ne 0 ]]; then
    for cmd in "$@"; do
      run_and_time "${cmd}"
    done
  else
    install_velox_deps
  fi
)

echo "All dependencies for Velox installed!"
