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

# Propagate errors and improve debugging.
set -eufx -o pipefail

CUDA_VERSION="12.8"
CUDA_VERSION_DASHED="${CUDA_VERSION//./-}"

function install_cuda_deps {
   # /etc/os-release is a standard way to query various distribution
   # information and is available everywhere
   LINUX_DISTRIBUTION=$(. /etc/os-release && echo ${ID})
   if [[ "$LINUX_DISTRIBUTION" == "ubuntu" || "$LINUX_DISTRIBUTION" == "debian" ]]; then
      apt update -y
      apt install -y \
         cuda-compat-${CUDA_VERSION_DASHED} \
         cuda-minimal-build-${CUDA_VERSION_DASHED} \
         cuda-nvrtc-dev-${CUDA_VERSION_DASHED}
   else # Assume CentOS/Fedora/RHEL-like
      dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
      dnf -y install \
         cuda-compat-${CUDA_VERSION_DASHED} \
         cuda-minimal-build-${CUDA_VERSION_DASHED} \
         cuda-nvrtc-devel-${CUDA_VERSION_DASHED}
   fi
}

install_cuda_deps

_ret=$?
if [ $_ret -eq 0 ] ; then
   echo "CUDA dependencies installed!"
fi
