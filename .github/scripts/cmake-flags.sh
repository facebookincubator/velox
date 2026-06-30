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

# shellcheck shell=bash
# shellcheck disable=SC2034 # CMAKE_FLAGS is consumed by callers that source this file.

# Source this file (do not exec it) from a workflow `run:` step to
# populate the CMAKE_FLAGS array for one of the supported build
# profiles. The caller must set BUILD_PROFILE in env before sourcing;
# USE_CLANG (default false) gates compiler-conditional flags.
#
# Supported profiles:
#   * adapters             — Linux release with adapters (linux-build-base.yml).
#   * dep-graph            — Dep-graph generator action; same as adapters minus
#                            WAVE/CUDF (they gate experimental code, which the
#                            selective-build planner short-circuits to full).
#   * ubuntu-debug         — Ubuntu debug with system dependencies.
#   * ubuntu-bundled-deps  — Ubuntu debug with velox's BUNDLED dependency
#                            resolution; mirrors ubuntu-debug minus
#                            REMOTE_FUNCTIONS (velox doesn't bundle the FB_OS
#                            suite — fbthrift / fizz / wangle / mvfst — so
#                            that flag has no business in BUNDLED-deps mode).
#   * fedora-debug         — Fedora debug.
#   * macos                — macOS debug/release.
#
# Truly job-local extras stay at the call site (e.g. the macOS job
# appends `-DCMAKE_BUILD_TYPE=$BUILD_TYPE` per matrix entry). Anything
# that's a property of the profile itself belongs here.
#
# Usage:
#   BUILD_PROFILE=adapters [USE_CLANG=true] source .github/scripts/cmake-flags.sh
#   EXTRA_CMAKE_FLAGS=("${CMAKE_FLAGS[@]}" -DVELOX_ENABLE_X=ON)

case "${BUILD_PROFILE:?BUILD_PROFILE must be set before sourcing cmake-flags.sh}" in
adapters | dep-graph)
  # adapters and dep-graph are sister profiles: dep-graph regenerates
  # the planner's source-of-truth graph for what adapters builds, so
  # they share most flags. MONO=OFF on both — dep-graph needs granular
  # per-component target edges, and adapters benefits from the same
  # leaf-library surface the selective-build planner targets. The only
  # delta is WAVE / CUDF (adapters-only; see below).
  CMAKE_FLAGS=(
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DVELOX_MONO_LIBRARY=OFF
    -DVELOX_ENABLE_BENCHMARKS=ON
    -DVELOX_ENABLE_EXAMPLES=ON
    -DVELOX_ENABLE_ARROW=ON
    -DVELOX_ENABLE_GEO=ON
    -DVELOX_ENABLE_PARQUET=ON
    -DVELOX_ENABLE_HDFS=ON
    -DVELOX_ENABLE_S3=ON
    -DVELOX_ENABLE_GCS=ON
    -DVELOX_ENABLE_ABFS=ON
  )
  if [[ ${USE_CLANG:-false} != "true" ]]; then
    # Faiss has a link issue under Clang; the remote function service
    # has open issues under Clang too (#13897).
    CMAKE_FLAGS+=(
      -DVELOX_ENABLE_FAISS=ON
      -DVELOX_ENABLE_REMOTE_FUNCTIONS=ON
    )
  fi
  if [[ $BUILD_PROFILE == "adapters" ]]; then
    # WAVE / CUDF gate code under velox/experimental/, which the
    # planner short-circuits to full mode via FULL_BUILD_PREFIXES,
    # so dep-graph doesn't need their targets. CUDF is GCC-only.
    CMAKE_FLAGS+=(-DVELOX_ENABLE_WAVE=ON)
    if [[ ${USE_CLANG:-false} != "true" ]]; then
      CMAKE_FLAGS+=(-DVELOX_ENABLE_CUDF=ON)
    fi
  fi
  ;;

ubuntu-debug | ubuntu-bundled-deps)
  CMAKE_FLAGS=(
    -DCMAKE_LINK_LIBRARIES_STRATEGY=REORDER_FREELY
    -DVELOX_BUILD_SHARED=ON
    -DVELOX_ENABLE_BENCHMARKS=ON
    -DVELOX_ENABLE_EXAMPLES=ON
    -DVELOX_ENABLE_ARROW=ON
    -DVELOX_ENABLE_GEO=ON
    -DVELOX_ENABLE_PARQUET=ON
    -DVELOX_MONO_LIBRARY=ON
  )
  if [[ ${USE_CLANG:-false} != "true" ]]; then
    CMAKE_FLAGS+=(-DVELOX_ENABLE_FAISS=ON)
  fi
  if [[ $BUILD_PROFILE == "ubuntu-debug" ]]; then
    # libnuma is baked into the velox-dev:ubuntu-22.04 image. ubuntu-bundled-deps
    # runs on a bare runner that doesn't install it, so it can't enable CXL.
    CMAKE_FLAGS+=(-DVELOX_ENABLE_CXL=ON)
  fi
  if [[ $BUILD_PROFILE == "ubuntu-debug" && ${USE_CLANG:-false} != "true" ]]; then
    # REMOTE_FUNCTIONS pulls in fbthrift / fizz / wangle / mvfst, which
    # velox doesn't bundle. ubuntu-debug runs in the velox-dev container
    # which has them pre-installed; ubuntu-bundled-deps runs on a bare
    # runner where they don't exist, so it must omit this flag.
    CMAKE_FLAGS+=(-DVELOX_ENABLE_REMOTE_FUNCTIONS=ON)
  fi
  ;;

fedora-debug)
  # fedora-debug is always GCC; no USE_CLANG branch needed.
  CMAKE_FLAGS=(
    -DVELOX_ENABLE_PARQUET=ON
    -DVELOX_ENABLE_EXAMPLES=ON
    -DVELOX_ENABLE_FAISS=ON
    # libnuma is baked into the velox-dev:fedora image.
    -DVELOX_ENABLE_CXL=ON
  )
  ;;

macos)
  # macOS is always Clang; no USE_CLANG branch needed.
  CMAKE_FLAGS=(
    -DTREAT_WARNINGS_AS_ERRORS=1
    -DENABLE_ALL_WARNINGS=1
    -DVELOX_ENABLE_PARQUET=ON
    -DVELOX_MONO_LIBRARY=ON
    -DVELOX_BUILD_SHARED=ON
    -DVELOX_ENABLE_FAISS=ON
  )
  ;;

*)
  echo "::error::unknown BUILD_PROFILE: ${BUILD_PROFILE}" >&2
  return 1
  ;;
esac
