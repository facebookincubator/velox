#!/bin/sh
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

# Generates the inlined headers for TorchWave JIT.
#
# Run in the velox checkout root.

TORCHWAVE=velox/experimental/torchwave
if [ -z "$STRINGIFY" ]; then
  STRINGIFY=stringify
fi

head --lines 16 velox/experimental/wave/common/Cuda.h >$TORCHWAVE/Headers.h
{
  echo ""
  echo "#pragma once"
  echo ""
  echo "namespace facebook::velox::wave {"

  echo "bool registerHeader(const char* text);"

  $STRINGIFY "velox/experimental/torchwave/KernelParams.h"
  $STRINGIFY "velox/experimental/torchwave/Core.cuh"
  $STRINGIFY "velox/experimental/torchwave/Elementwise.cuh"
  $STRINGIFY "velox/experimental/torchwave/Scan.cuh"
  $STRINGIFY "velox/experimental/torchwave/Views.cuh"
  $STRINGIFY "velox/experimental/torchwave/Hash.cuh"

  echo "}"
} >>$TORCHWAVE/Headers.h

clang-format -i $TORCHWAVE/Headers.h
