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
export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:/usr/lib64/pkgconfig:/usr/lib/pkgconfig:$PKG_CONFIG_PATH
FB_OS_VERSION=v2022.11.14.00

# shellcheck disable=SC2037
SUDO="sudo -E"

function run_and_time {
  time "$@"
  { echo "+ Finished running $*"; } 2> /dev/null
}

function dnf_install {
  $SUDO dnf install -y -q --setopt=install_weak_deps=False "$@"
}

function yum_install {
  $SUDO yum install -y "$@"
}

function cmake_install_deps {
  cmake -B"$1-build" -GNinja -DCMAKE_CXX_STANDARD=17 \
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

function install_cmake {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar https://cmake.org/files/v3.25/cmake-3.25.1.tar.gz cmake-3
  cd cmake-3
  ./bootstrap --prefix=/usr/local
  make -j$(nproc)
  $SUDO make install
  cmake --version
}

function install_ninja {
  cd "${DEPENDENCY_DIR}"
  github_checkout ninja-build/ninja v1.11.1
  ./configure.py --bootstrap
  cmake -Bbuild-cmake
  cmake --build build-cmake
  $SUDO cp ninja /usr/local/bin/  
}

function install_fmt {
  cd "${DEPENDENCY_DIR}"
  github_checkout fmtlib/fmt 8.0.0
  cmake_install -DFMT_TEST=OFF
}

function install_folly {
  cd "${DEPENDENCY_DIR}"
  github_checkout facebook/folly "${FB_OS_VERSION}"
  cmake_install -DBUILD_TESTS=OFF
}

function install_conda {
  cd "${DEPENDENCY_DIR}"
  mkdir -p conda && cd conda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  MINICONDA_PATH=/opt/miniconda-for-velox
  bash Miniconda3-latest-Linux-x86_64.sh -b -u $MINICONDA_PATH
}

function install_openssl {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1s.tar.gz openssl
  cd openssl
  ./config no-shared
  make depend
  make
  $SUDO make install
}

function install_gflags {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar https://github.com/gflags/gflags/archive/v2.2.2.tar.gz gflags
  cd gflags
  cmake_install -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON -DLIB_SUFFIX=64 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local
}

function install_glog {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar https://github.com/google/glog/archive/v0.5.0.tar.gz glog
  cd glog
  cmake_install -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=/usr/local  
}

function install_snappy {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar https://github.com/google/snappy/archive/1.1.8.tar.gz snappy
  cd snappy
  cmake_install -DSNAPPY_BUILD_TESTS=OFF  
}

function install_dwarf {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar https://github.com/davea42/libdwarf-code/archive/refs/tags/20210528.tar.gz dwarf
  cd dwarf
  #local URL=https://github.com/davea42/libdwarf-code/releases/download/v0.5.0/libdwarf-0.5.0.tar.xz
  #local DIR=dwarf
  #mkdir -p "${DIR}"
  #wget -q --max-redirect 3 "${URL}"
  #tar -xf libdwarf-0.5.0.tar.xz -C "${DIR}"
  #cd dwarf/libdwarf-0.5.0
  ./configure --enable-shared=no
  make
  make check
  $SUDO make install
}

function install_re2 {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar https://github.com/google/re2/archive/refs/tags/2023-03-01.tar.gz re2
  cd re2
  $SUDO make install
}

function install_flex {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz flex
  cd flex
  ./autogen.sh
  ./configure
  $SUDO make install
}

function install_lzo {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar http://www.oberhumer.com/opensource/lzo/download/lzo-2.10.tar.gz lzo
  cd lzo
  ./configure --prefix=/usr/local --enable-shared --disable-static --docdir=/usr/local/share/doc/lzo-2.10
  make "-j$(nproc)"
  $SUDO make install
}

function install_boost {
  cd "${DEPENDENCY_DIR}"
  wget_and_untar https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.gz boost
  cd boost
  ./bootstrap.sh --prefix=/usr/local --with-python=/usr/bin/python3 --with-python-root=/usr/lib/python3.6 --without-libraries=python
  $SUDO ./b2 "-j$(nproc)" -d0 install threading=multi
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
  cd "${DEPENDENCY_DIR}"
  github_checkout aws/aws-sdk-cpp 1.9.379 --depth 1 --recurse-submodules
  cmake_install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS:BOOL=OFF -DMINIMIZE_SIZE:BOOL=ON -DENABLE_TESTING:BOOL=OFF -DBUILD_ONLY:STRING="s3;identity-management" 
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

function install_prerequisites {
  run_and_time install_lzo
  run_and_time install_boost
  run_and_time install_re2
  run_and_time install_flex
  run_and_time install_openssl
  run_and_time install_gflags
  run_and_time install_glog
  run_and_time install_snappy
  run_and_time install_dwarf
}

function install_velox_deps {
  run_and_time install_fmt
  run_and_time install_folly
  run_and_time install_conda
}

$SUDO dnf makecache

# dnf install dependency libraries
dnf_install epel-release dnf-plugins-core # For ccache, ninja
# PowerTools only works on CentOS8
# dnf config-manager --set-enabled powertools
dnf_install ccache git wget which libevent-devel \
  openssl-devel libzstd-devel lz4-devel double-conversion-devel \
  curl-devel cmake libxml2-devel libgsasl-devel libuuid-devel

$SUDO dnf remove -y gflags

# Required for Thrift
dnf_install autoconf automake libtool bison python3 python3-devel

# Required for build flex
dnf_install gettext-devel texinfo help2man

# dnf_install conda

# Activate gcc9; enable errors on unset variables afterwards.
# GCC9 install via yum and devtoolset
# dnf install gcc-toolset-9 only works on CentOS8

$SUDO yum makecache
yum_install centos-release-scl
yum_install devtoolset-9
source /opt/rh/devtoolset-9/enable || exit 1
gcc --version
set -u

# Build from source
[ -d "$DEPENDENCY_DIR" ] || mkdir -p "$DEPENDENCY_DIR"

run_and_time install_cmake
run_and_time install_ninja

install_prerequisites
install_velox_deps
