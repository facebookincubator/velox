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

# github_checkout $REPO $VERSION $GIT_CLONE_PARAMS clones or re-uses an existing clone of the
# specified repo, checking out the requested version.
function get_cxx_flags {
  local OS
  OS=$(uname)
  local MACHINE
  MACHINE=$(uname -m)
  ADDITIONAL_FLAGS=""

  if [ "$OS" = "Darwin" ]; then
    if [ "$MACHINE" = "x86_64" ]; then
      local CPU_CAPABILITIES
      CPU_CAPABILITIES=$(sysctl -a | grep machdep.cpu.features | awk '{print tolower($0)}')

      if [[ $CPU_CAPABILITIES =~ "avx" ]]; then
        echo -n "-mavx2 -mfma -mavx -mf16c -mlzcnt -std=c++17 -mbmi2 $ADDITIONAL_FLAGS"
      else
        echo -n "-msse4.2 -std=c++17 $ADDITIONAL_FLAGS"
      fi
    elif [[ $(sysctl -a | grep machdep.cpu.brand_string) =~ "Apple" ]]; then
      # Apple silicon.
      echo -n "-mcpu=apple-m1+crc -std=c++17 -fvisibility=hidden $ADDITIONAL_FLAGS"
    fi
  elif [ "$OS" = "Linux" ]; then
    local CPU_CAPABILITIES
    CPU_CAPABILITIES=$(cat /proc/cpuinfo | grep flags | head -n 1 | awk '{print tolower($0)}')

    if [[ "$CPU_CAPABILITIES" =~ "avx" ]]; then
      echo -n "-mavx2 -mfma -mavx -mf16c -mlzcnt -std=c++17 -mbmi2 $ADDITIONAL_FLAGS"
    elif [[ "$CPU_CAPABILITIES" =~ "sse" ]]; then
      echo -n "-msse4.2 -std=c++17 $ADDITIONAL_FLAGS"
    elif [ "$MACHINE" = "aarch64" ]; then
      echo -n "-mcpu=neoverse-n1 -std=c++17 $ADDITIONAL_FLAGS"
    fi
  else
    echo -n "Architecture not supported!"
  fi
}

function cmake_install {
  local NAME=$(basename "$(pwd)")
  local BINARY_DIR=_build
  if [ -d "${BINARY_DIR}" ] && prompt "Do you want to rebuild ${NAME}?"; then
    rm -rf "${BINARY_DIR}"
  fi
  mkdir -p "${BINARY_DIR}"
  COMPILER_FLAGS=$(get_cxx_flags)

  # CMAKE_POSITION_INDEPENDENT_CODE is required so that Velox can be built into dynamic libraries \
  cmake -Wno-dev -B"${BINARY_DIR}" \
    -GNinja \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=17 \
    "${INSTALL_PREFIX+-DCMAKE_PREFIX_PATH=}${INSTALL_PREFIX-}" \
    "${INSTALL_PREFIX+-DCMAKE_INSTALL_PREFIX=}${INSTALL_PREFIX-}" \
    -DCMAKE_CXX_FLAGS="$COMPILER_FLAGS" \
    -DBUILD_TESTING=OFF \
    "$@"
  ninja -C "${BINARY_DIR}" install
}
