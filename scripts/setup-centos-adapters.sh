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
# shellcheck source-path=SCRIPT_DIR

# This script installs addition dependencies for the adapters build
# of Velox. The scrip expects base dependencies to already be installed.
#
# This script is split of from setup-centos9.sh to improve docker caching
#
# Environment variables:
# * INSTALL_PREREQUISITES="N": Skip installation of packages for build.
# * PROMPT_ALWAYS_RESPOND="n": Automatically respond to interactive prompts.
#     Use "n" to never wipe directories.
# * VELOX_CUDA_VERSION="12.9": Which version of CUDA to install, will pick up
#   CUDA_VERSION from the env
# * VELOX_UCX_VERSION="1.19.0": Which version of ucx to install, will pick up
#   UCX_VERSION from the env

set -efx -o pipefail

VELOX_CUDA_VERSION=${CUDA_VERSION:-"12.9"}
VELOX_UCX_VERSION=${UCX_VERSION:-"1.19.0"}
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$SCRIPT_DIR"/setup-centos9.sh

function install_ucx {
  dnf_install rdma-core-devel
  local UCX_REPO_NAME="openucx/ucx"
  local NEEDS_AUTOGEN=false

  if [ "${VELOX_UCX_VERSION}" == "master" ]; then
    github_checkout "${UCX_REPO_NAME}" "${VELOX_UCX_VERSION}"
    NEEDS_AUTOGEN=true
  else
    wget_and_untar https://github.com/openucx/ucx/releases/download/v"${VELOX_UCX_VERSION}"/ucx-"${VELOX_UCX_VERSION}".tar.gz ucx
  fi

  (
    cd "${DEPENDENCY_DIR}"/ucx || exit
    if [ "${NEEDS_AUTOGEN}" = true ]; then
      ./autogen.sh
    fi

    local CUDA_FLAG=""
    if [ -d "/usr/local/cuda" ]; then
      CUDA_FLAG="--with-cuda=/usr/local/cuda"
    fi

    mkdir build-linux && cd build-linux

    ../contrib/configure-release --prefix="${INSTALL_PREFIX}" --with-sysroot --enable-cma \
      --enable-mt --with-gnu-ld --with-rdmacm --with-verbs \
      --without-go --without-java ${CUDA_FLAG}
    make "-j${NPROC}"
    make install
  )
}

function install_cuda {
  # See https://developer.nvidia.com/cuda-downloads
  local arch
  arch="$(uname -m)"
  local repo_url
  version="${1:-$VELOX_CUDA_VERSION}"

  if [[ $arch == "x86_64" ]]; then
    repo_url="https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo"
  elif [[ $arch == "aarch64" ]]; then
    # Using SBSA (Server Base System Architecture) repository for ARM64 servers
    repo_url="https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa/cuda-rhel9.repo"
  else
    echo "Unsupported architecture: $arch" >&2
    return 1
  fi

  dnf config-manager --add-repo "$repo_url"
  local dashed
  dashed="$(echo "$version" | tr '.' '-')"
  dnf_install \
    cuda-compat-"$dashed" \
    cuda-driver-devel-"$dashed" \
    cuda-minimal-build-"$dashed" \
    cuda-nvrtc-devel-"$dashed" \
    libcufile-devel-"$dashed" \
    libnvjitlink-devel-"$dashed" \
    cuda-nvml-devel-"$dashed" \
    numactl-devel
}

function install_adapters_deps_from_dnf {
  local gcs_deps=(curl-devel c-ares-devel re2-devel)
  local azure_deps=(perl-IPC-Cmd openssl-devel libxml2-devel)
  local hdfs_deps=(libxml2-devel libgsasl-devel libuuid-devel krb5-devel java-1.8.0-openjdk-devel)

  dnf_install "${azure_deps[@]}" "${gcs_deps[@]}" "${hdfs_deps[@]}"
}

function install_s3 {
  install_aws_deps
  local MINIO_OS="linux"
  install_minio ${MINIO_OS}
}

function install_adapters {
  run_and_time install_adapters_deps_from_dnf
  run_and_time install_s3
  run_and_time install_gcs_sdk_cpp
  run_and_time install_azure_storage_sdk_cpp
  run_and_time install_hdfs_deps
  run_and_time install_avro_cpp
}

(return 2>/dev/null) && return # If script was sourced, don't run commands.

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
    # Activate gcc12; enable errors on unset variables afterwards.
    source /opt/rh/gcc-toolset-12/enable || exit 1
    set -u
    install_cuda "$VELOX_CUDA_VERSION"
    install_adapters
    echo "All dependencies for the Velox Adapters installed!"
    if [[ ${USE_CLANG} != "false" ]]; then
      echo "To use clang for the Velox build set the CC and CXX environment variables in your session."
      echo "  export CC=/usr/bin/clang-15"
      echo "  export CXX=/usr/bin/clang++-15"
    fi
    dnf clean all
  fi
)
