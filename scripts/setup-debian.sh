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
export COMPILER_FLAGS=$(get_cxx_flags $CPU_TARGET)
FB_OS_VERSION=v2022.03.14.00
NPROC=$(getconf _NPROCESSORS_ONLN)
DEPENDENCY_DIR=${DEPENDENCY_DIR:-$(pwd)}
VELOX_DIR=$DEPENDENCY_DIR

# Add the Debian Source to apt-get
echo "deb http://http.us.debian.org/debian stable main contrib non-free" >> /etc/apt/sources.list
apt-get update && apt-get upgrade -y

# Install all velox and folly dependencies.
apt-get install -y \
    build-essential \
    bison \
    libcurl4-openssl-dev \
    libdouble-conversion-dev \
    libsodium-dev \
    libmsgpack* \
    sudo \
    libgflags-dev \
    uuid-dev \
    curl \
    libzstd-dev \
    ninja-build \
    git \
    cmake \
    libkrb5-dev \
    libprotobuf-dev \
    g++ \
    libsnappy-dev \
    ccache \
    python3-pip \
    libssl-dev \
    dpkg-dev \
    libgmock-dev \
    libgoogle-glog-dev \
    libbz2-dev \
    flex \
    unzip \
    gperf \
    liblzo2-dev \
    libre2-dev \
    libevent-dev \
    ca-certificates \
    antlr4 \
    wget \
    libxml2-dev \
    libboost-all-dev \
    libgtest-dev \
    fakeroot \
    libghc-gsasl-dev \
    checkinstall \
    rapidjson-dev \
    protobuf-compiler \
    liblz4-dev \
    pkg-config

# Install python 2.7 and pip
cd /tmp
wget https://www.python.org/ftp/python/2.7.15/Python-2.7.15.tar.xz --no-check-certificate
tar -xvJf Python-2.7.15.tar.xz -C /tmp
cd Python-2.7.15
./configure
make
make install
cd ..
rm Python-2.7.15.tar.xz

cd /tmp
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py \
  && python get-pip.py --no-setuptools \
  && rm -f get-pip.py \
  && pip --no-cache-dir install setuptools==42.0.2

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
  github_checkout fmtlib/fmt 8.0.0
  cmake_install -DFMT_TEST=OFF
}

function install_folly {
  github_checkout facebook/folly "${FB_OS_VERSION}"
  cmake_install -DBUILD_TESTS=OFF
}

function install_velox_deps {
  run_and_time install_fmt
  run_and_time install_folly
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
