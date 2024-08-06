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

# This script documents setting up a Centos9 host for Velox
# development.  Running it should make you ready to compile.
#
# Environment variables:
# * INSTALL_PREREQUISITES="N": Skip installation of packages for build.
# * PROMPT_ALWAYS_RESPOND="n": Automatically respond to interactive prompts.
#     Use "n" to never wipe directories.
#
# You can also run individual functions below by specifying them as arguments:
# $ scripts/setup-centos9.sh install_googletest install_fmt
#

set -efx -o pipefail
# Some of the packages must be build with the same compiler flags
# so that some low level types are the same size. Also, disable warnings.
SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
source $SCRIPTDIR/setup-linux.sh

SUDO="${SUDO:-""}"

export CFLAGS=$(get_cxx_flags)  # Used by LZO.
export CXXFLAGS=$CFLAGS  # Used by boost.
export CPPFLAGS=$CFLAGS  # Used by LZO.

export CC=/opt/rh/gcc-toolset-12/root/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/bin/g++

function dnf_install {
  dnf install -y -q --setopt=install_weak_deps=False "$@"
}

# Install packages required for build.
function install_build_prerequisites {
  dnf update -y
  dnf_install epel-release dnf-plugins-core # For ccache, ninja
  dnf config-manager --set-enabled crb
  dnf update -y
  dnf_install ninja-build cmake ccache gcc-toolset-12 git wget which
  dnf_install autoconf automake python3-devel pip libtool
  pip install cmake==3.28.3
}

# Install dependencies from the package managers.
function install_velox_deps_from_dnf {
  dnf_install libevent-devel \
    openssl-devel re2-devel libzstd-devel lz4-devel double-conversion-devel \
    libdwarf-devel elfutils-libelf-devel curl-devel libicu-devel bison flex \
    libsodium-devel zlib-devel gmock gtest

  # install sphinx for doc gen
  pip install sphinx sphinx-tabs breathe sphinx_rtd_theme
}

function install_conda {
  dnf_install conda
}

function install_cuda {
  # See https://developer.nvidia.com/cuda-downloads
  dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
  dnf install -y cuda-nvcc-$(echo $1 | tr '.' '-') cuda-cudart-devel-$(echo $1 | tr '.' '-')
}

function install_s3 {
  dnf -y install libxml2-devel libgsasl-devel libuuid-devel krb5-devel

  install_aws_deps

  local MINIO_ARCH="amd64"
  local MINIO_OS="linux"
  install_minio ${MINIO_ARCH} ${MINIO_OS}
}

function install_gcs {
  # Dependencies of GCS, probably a workaround until the docker image is rebuilt
  dnf -y install npm curl-devel c-ares-devel
  install_gcs-sdk-cpp
}

function install_abfs {
  # Dependencies of Azure Storage Blob cpp
  dnf -y install perl-IPC-Cmd openssl
  install_azure-storage-sdk-cpp
}

function install_adapters {
  run_and_time install_s3
  run_and_time install_gcs
  run_and_time install_abfs
  run_and_time install_hdfs
}

function install_velox_deps {
  run_and_time install_velox_deps_from_dnf
  run_and_time install_conda
  run_and_time install_gflags
  run_and_time install_glog
  run_and_time install_lzo
  run_and_time install_snappy
  run_and_time install_boost
  run_and_time install_protobuf
  run_and_time install_fmt
  run_and_time install_folly
  run_and_time install_fizz
  run_and_time install_wangle
  run_and_time install_mvfst
  run_and_time install_fbthrift
  run_and_time install_duckdb
  run_and_time install_arrow
  run_and_time install_xsimd
  run_and_time install_simdjson
}

(return 2> /dev/null) && return # If script was sourced, don't run commands.

(
  if [[ $# -ne 0 ]]; then
    # Activate gcc12; enable errors on unset variables afterwards.
    source /opt/rh/gcc-toolset-12/enable || exit 1
    set -u
    for cmd in "$@"; do
      run_and_time "${cmd}"
    done
    echo "All specified dependencies installed!"
  else
    if [ "${INSTALL_PREREQUISITES:-Y}" == "Y" ]; then
      echo "Installing build dependencies"
      run_and_time install_build_prerequisites
    else
      echo "Skipping installation of build dependencies since INSTALL_PREREQUISITES is not set"
    fi
    # Activate gcc12; enable errors on unset variables afterwards.
    source /opt/rh/gcc-toolset-12/enable || exit 1
    set -u
    install_velox_deps
    echo "All dependencies for Velox installed!"
    dnf clean all
  fi
)

