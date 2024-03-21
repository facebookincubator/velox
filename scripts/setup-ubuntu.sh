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

# Minimal setup for Ubuntu 20.04.
set -eufx -o pipefail
SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
source $SCRIPTDIR/setup-helper-functions.sh

# Folly must be built with the same compiler flags so that some low level types
# are the same size.
CPU_TARGET="${CPU_TARGET:-avx}"
COMPILER_FLAGS=$(get_cxx_flags "$CPU_TARGET")
export COMPILER_FLAGS
NPROC=$(getconf _NPROCESSORS_ONLN)
DEPENDENCY_DIR=${DEPENDENCY_DIR:-$(pwd)}
export CMAKE_BUILD_TYPE=Release
SUDO="${SUDO:-"sudo --preserve-env"}"

FB_OS_VERSION="v2024.02.26.00"
FMT_VERSION="10.1.1"
BOOST_VERSION="boost-1.84.0"

# Install all velox and folly dependencies.
# The is an issue on 22.04 where a version conflict prevents glog install,
# installing libunwind first fixes this.

${SUDO} apt update
${SUDO} apt install -y libunwind-dev
${SUDO} apt install -y \
  g++ \
  cmake \
  ccache \
  curl \
  ninja-build \
  checkinstall \
  git \
  libc-ares-dev \
  libcurl4-openssl-dev \
  libssl-dev \
  libicu-dev \
  libdouble-conversion-dev \
  libgoogle-glog-dev \
  libbz2-dev \
  libgflags-dev \
  libgmock-dev \
  libevent-dev \
  liblz4-dev \
  libzstd-dev \
  libre2-dev \
  libsnappy-dev \
  libsodium-dev \
  libthrift-dev \
  liblzo2-dev \
  libelf-dev \
  libdwarf-dev \
  bison \
  flex \
  libfl-dev \
  tzdata \
  wget

function install_fmt {
  wget_and_untar https://github.com/fmtlib/fmt/archive/${FMT_VERSION}.tar.gz fmt
  (
    cd fmt
    cmake_install -DFMT_TEST=OFF
  )
}

function install_boost {
  wget_and_untar https://github.com/boostorg/boost/releases/download/${BOOST_VERSION}/${BOOST_VERSION}.tar.gz boost
  (
   cd boost
   ./bootstrap.sh --prefix=/usr/local
   ${SUDO} ./b2 "-j$(nproc)" -d0 install threading=multi
  )
}

function install_folly {
  wget_and_untar https://github.com/facebook/folly/archive/refs/tags/${FB_OS_VERSION}.tar.gz folly
  (
    cd folly
    cmake_install -DBUILD_TESTS=OFF -DFOLLY_HAVE_INT128_T=ON
  )
}

function install_fizz {
  wget_and_untar https://github.com/facebookincubator/fizz/archive/refs/tags/${FB_OS_VERSION}.tar.gz fizz
  (
    cd fizz/fizz
    cmake_install -DBUILD_TESTS=OFF
  )
}

function install_wangle {
  wget_and_untar https://github.com/facebook/wangle/archive/refs/tags/${FB_OS_VERSION}.tar.gz wangle
  (
    cd wangle/wangle
    cmake_install -DBUILD_TESTS=OFF
  )
}

function install_mvfst {
  wget_and_untar https://github.com/facebook/mvfst/archive/refs/tags/${FB_OS_VERSION}.tar.gz mvfst
  (
   cd mvfst
   cmake_install -DBUILD_TESTS=OFF
  )
}

function install_fbthrift {
  wget_and_untar https://github.com/facebook/fbthrift/archive/refs/tags/${FB_OS_VERSION}.tar.gz fbthrift
  (
    cd fbthrift
    cmake_install -Denable_tests=OFF -DBUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF
  )
}

function install_conda {
  MINICONDA_PATH=/opt/miniconda-for-velox
  if [ -e ${MINICONDA_PATH} ]; then
    echo "File or directory already exists: ${MINICONDA_PATH}"
    return
  fi
  ARCH=$(uname -m)
  if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
    echo "Unsupported architecture: $ARCH"
    exit 1
  fi
  
  mkdir -p conda && cd conda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$ARCH.sh
  bash Miniconda3-latest-Linux-$ARCH.sh -b -p $MINICONDA_PATH
}


function install_velox_deps {
  run_and_time install_fmt
  run_and_time install_boost
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
