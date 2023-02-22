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
FB_OS_VERSION=v2022.11.14.00
NPROC=$(getconf _NPROCESSORS_ONLN)
DEPENDENCY_DIR=${DEPENDENCY_DIR:-$(pwd)}
export CMAKE_BUILD_TYPE=Release

# Install all velox and folly dependencies.
sudo --preserve-env apt update && apt install -y \
  g++ \
  cmake \
  ccache \
  ninja-build \
  checkinstall \
  git \
  libssl-dev \
  libboost-all-dev \
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
  liblzo2-dev \
  bison \
  flex \
  tzdata \
  wget

function run_and_time {
  time "$@"
  { echo "+ Finished running $*"; } 2> /dev/null
}

function prompt {
  (
    while true; do
      local input="${PROMPT_ALWAYS_RESPOND:-}"
      echo -n "$(tput bold)$* [Y, n]$(tput sgr0) "
      [[ -z "${input}" ]] && read input
      if [[ "${input}" == "Y" || "${input}" == "y" || "${input}" == "" ]]; then
        return 0
      elif [[ "${input}" == "N" || "${input}" == "n" ]]; then
        return 1
      fi
    done
  ) 2> /dev/null
}

function install_fmt {
  github_checkout fmtlib/fmt 8.0.1
  cmake_install -DFMT_TEST=OFF
}

function install_folly {
  github_checkout facebook/folly "${FB_OS_VERSION}"
  cmake_install -DBUILD_TESTS=OFF
}

function install_conda {
  mkdir -p conda && cd conda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  MINICONDA_PATH=/opt/miniconda-for-velox
  bash Miniconda3-latest-Linux-x86_64.sh -b -p $MINICONDA_PATH
}

function install_velox_deps {
  run_and_time install_fmt
  run_and_time install_folly
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

echo "All deps for Velox installed! Now try \"make\""
