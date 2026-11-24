/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "velox/experimental/wave/common/Cuda.h"
#include "velox/experimental/wave/dwio/decode/DecodeOp.h"

namespace facebook::velox::wave {

/// The template arguments of the device function that executes a GpuDecode.
/// GpuDecodes with equal signatures run the same instantiation. Fields that
/// do not affect the instantiation for 'step' are left at their defaults so
/// that equal instantiations have equal signatures.
struct DecodeSignature {
  DecodeStep step{DecodeStep::kUnsupported};
  DecodeStep encoding{DecodeStep::kUnsupported};
  /// Value type, normalized to the integer type of the same width.
  WaveTypeKind dataType{WaveTypeKind::UNKNOWN};
  /// Type of the decoded dictionary index if different from 'dataType'.
  WaveTypeKind indexType{WaveTypeKind::UNKNOWN};
  WaveFilterKind filterKind{WaveFilterKind::kAlwaysTrue};
  NullMode nullMode{NullMode::kDenseNonNull};
  DictMode dictMode{DictMode::kNone};
  /// True if values are materialized, false for a filter only column.
  bool hasResult{true};
  /// Step specific compile time constant, e.g. the bit count stride of
  /// kCountBits.
  int32_t variant{0};

  auto tie() const {
    return std::tie(
        step,
        encoding,
        dataType,
        indexType,
        filterKind,
        nullMode,
        dictMode,
        hasResult,
        variant);
  }

  bool operator==(const DecodeSignature& other) const {
    return tie() == other.tie();
  }

  bool operator<(const DecodeSignature& other) const {
    return tie() < other.tie();
  }

  /// Compact text for use in a cache key.
  std::string toString() const;

  /// C++ text of the statement that executes a GpuDecode 'op' with this
  /// signature in a TB of 'blockSize' threads.
  std::string call(int32_t blockSize) const;
};

/// Returns a process wide id for a comma separated list of accessor types of
/// a file format, e.g. "NFbw<uint64_t>, NConstant" for Nimble. Column ops
/// carry the id of the accessors of their chunks in ColumnAccess::accessorsId
/// and the signature carries it in 'variant'.
int32_t accessorListId(const std::string& accessors);

/// Returns the accessor list for an id from accessorListId().
std::string accessorList(int32_t id);

/// Returns the signature of 'op'. Throws if 'op' has a combination of step,
/// type, encoding and filter that has no device implementation.
DecodeSignature decodeSignature(const GpuDecode& op);

/// The normalized description of a compiled reader kernel: the sorted,
/// distinct signatures of the ops of a launch and the TB size. The opcode of a
/// GpuDecode is the position of its signature in 'signatures'. Ops of
/// different columns in a different order produce the same key.
struct ReaderKernelKey {
  int32_t blockSize{0};
  std::vector<DecodeSignature> signatures;

  std::string toString() const;

  /// Returns the position of 'signature' in 'signatures'.
  int32_t opcode(const DecodeSignature& signature) const;

  /// Dynamic shared memory for a launch of the kernel, including alignment
  /// padding.
  int32_t sharedMemorySize() const;
};

/// Makes the key for a launch of 'ops' and sets the opcode of each op.
ReaderKernelKey makeReaderKernelKey(
    const std::vector<GpuDecode*>& ops,
    int32_t blockSize);

/// Entry points of a compiled reader kernel.
enum class ReaderEntry : int32_t {
  /// Takes GpuDecodeParams by value. Runs a sequence of ops per TB.
  kPrograms = 0,
  /// Takes a GpuDecode*. Runs the op at blockIdx.x.
  kSingle = 1
};

/// Returns the CUDA source of the kernel for 'key'.
KernelSpec readerKernelSpec(const ReaderKernelKey& key);

struct ReaderKernel {
  std::unique_ptr<CompiledKernel> kernel;
  /// True if the key was not requested before in this process, i.e. this
  /// request started its compilation.
  bool isNew{false};
};

/// Returns the compiled kernel for 'key'. Compiles in the background on first
/// use. Launching waits for the compilation.
ReaderKernel getReaderKernel(const ReaderKernelKey& key);

/// Runs ops[i] on TB i of a grid of 'numOps' TBs of 'blockSize' threads.
/// 'ops' must be readable and writable from host and device, e.g. unified
/// memory.
void launchDecodeOps(
    GpuDecode* ops,
    int32_t numOps,
    int32_t blockSize,
    Stream* stream);

} // namespace facebook::velox::wave
