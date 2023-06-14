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
DEPENDENCY_DIR=${DEPENDENCY_DIR:-/tmp/velox-deps}
CPU_TARGET="${CPU_TARGET:-avx}"
NPROC=$(getconf _NPROCESSORS_ONLN)
export CFLAGS=$(get_cxx_flags $CPU_TARGET)  # Used by LZO.
export CXXFLAGS=$CFLAGS  # Used by boost.
export CPPFLAGS=$CFLAGS  # Used by LZO.

# shellcheck disable=SC2037
SUDO="sudo -E"

function dnf_install {
  $SUDO dnf install -y -q --setopt=install_weak_deps=False "$@"
}

$SUDO dnf makecache

dnf_install epel-release dnf-plugins-core # For ccache, ninja
$SUDO dnf config-manager --set-enabled powertools
dnf_install ninja-build ccache gcc-toolset-9 git wget which libevent-devel \
  openssl-devel re2-devel libzstd-devel lz4-devel double-conversion-devel \
  libdwarf-devel curl-devel cmake libicu-devel libxml2-devel libgsasl-devel \
  libuuid-devel

$SUDO dnf remove -y gflags

# Required for Thrift
dnf_install autoconf automake libtool bison flex python3

dnf_install conda

# Activate gcc9; enable errors on unset variables afterwards.
source /opt/rh/gcc-toolset-9/enable || exit 1
set -u

function cmake_install_deps {
  cmake -B "$1-build" -GNinja -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="${CFLAGS}" -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release -Wno-dev "$@"
  ninja -C "$1-build"
  $SUDO ninja -C "$1-build" install
}

function wget_and_untar {
  local URL=$1
  local DIR=$2
  mkdir -p "${DIR}"
  wget -q --max-redirect 3 -O - "${URL}" | tar -xz -C "${DIR}" --strip-components=1
}

function install_gtest {
  cd "${DEPENDENCY_DIR}"
  wget https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
  tar -xzf release-1.12.1.tar.gz
  cd googletest-release-1.12.1
  mkdir -p build && cd build && cmake -DBUILD_GTEST=ON -DBUILD_GMOCK=ON -DINSTALL_GTEST=ON -DINSTALL_GMOCK=ON -DBUILD_SHARED_LIBS=ON ..
  make "-j$(nproc)"
  $SUDO make install
}

FB_OS_VERSION=v2022.11.14.00
function install_folly {
  cd "${DEPENDENCY_DIR}"
  github_checkout facebook/folly "${FB_OS_VERSION}"
  cmake_install -DBUILD_TESTS=OFF
}

function install_libhdfs3 {
  cd "${DEPENDENCY_DIR}"
  github_checkout apache/hawq master
  cd depends/libhdfs3
  sed -i "/FIND_PACKAGE(GoogleTest REQUIRED)/d" ./CMakeLists.txt
  sed -i "s/dumpversion/dumpfullversion/" ./CMake/Platform.cmake
  sed -i "s/dfs.domain.socket.path\", \"\"/dfs.domain.socket.path\", \"\/var\/lib\/hadoop-hdfs\/dn_socket\"/g" src/common/SessionConfig.cpp
  sed -i "s/pos < endOfCurBlock/pos \< endOfCurBlock \&\& pos \- cursor \<\= 128 \* 1024/g" src/client/InputStreamImpl.cpp
  cmake_install
}

function install_protobuf {
  cd "${DEPENDENCY_DIR}"
  wget https://github.com/protocolbuffers/protobuf/releases/download/v21.4/protobuf-all-21.4.tar.gz
  tar -xzf protobuf-all-21.4.tar.gz
  cd protobuf-21.4
  ./configure  CXXFLAGS="-fPIC"  --prefix=/usr/local
  make "-j$(nproc)"
  $SUDO make install
}

function install_awssdk {
  github_checkout aws/aws-sdk-cpp 1.9.379 --depth 1 --recurse-submodules
  cmake_install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS:BOOL=OFF -DMINIMIZE_SIZE:BOOL=ON -DENABLE_TESTING:BOOL=OFF -DBUILD_ONLY:STRING="s3;identity-management" 
} 

[ -f "${DEPENDENCY_DIR}" ] || mkdir -p "${DEPENDENCY_DIR}"
cd "${DEPENDENCY_DIR}"

# Fetch sources.
wget_and_untar https://github.com/gflags/gflags/archive/v2.2.2.tar.gz gflags &
wget_and_untar https://github.com/google/glog/archive/v0.4.0.tar.gz glog &
wget_and_untar http://www.oberhumer.com/opensource/lzo/download/lzo-2.10.tar.gz lzo &
wget_and_untar https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.gz boost &
wget_and_untar https://github.com/google/snappy/archive/1.1.8.tar.gz snappy &
wget_and_untar https://github.com/fmtlib/fmt/archive/8.0.1.tar.gz fmt &
wget_and_untar https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_0l.tar.gz openssl &

wait  # For cmake and source downloads to complete.

# Build & install.
(
  cd lzo
  ./configure --prefix=/usr --enable-shared --disable-static --docdir=/usr/share/doc/lzo-2.10
  make "-j$(nproc)"
  $SUDO make install
)

(
  cd boost
  ./bootstrap.sh --prefix=/usr/local
  ./b2 "-j$(nproc)" -d0 threading=multi
  $SUDO ./b2 "-j$(nproc)" -d0 install threading=multi
)

(
  # openssl static library
  cd openssl
  ./config no-shared
  make depend
  make
  $SUDO cp libcrypto.a /usr/local/lib64/
  $SUDO cp libssl.a /usr/local/lib64/
)

cmake_install_deps gflags -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON -DLIB_SUFFIX=64 -DCMAKE_INSTALL_PREFIX:PATH=/usr
cmake_install_deps glog -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr
cmake_install_deps snappy -DSNAPPY_BUILD_TESTS=OFF
cmake_install_deps fmt -DFMT_TEST=OFF
