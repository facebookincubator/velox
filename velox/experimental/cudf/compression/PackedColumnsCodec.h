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

#include <cudf/packed_types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace facebook::velox::cudf_velox::compression {
namespace detail {
class PackedColumnsCodecState;
}

/** Numeric transform applied before optional entropy coding. */
enum class NumericTransform {
  /** Chooses FOR or delta-FOR independently for each eligible region. */
  kAutomatic,

  /** Produces independently addressable FOR values. */
  kFrameOfReference,

  /** Produces delta-FOR values that require a prefix sum to reconstruct. */
  kDeltaFrameOfReference,
};

/** Optional entropy-coding stage applied after the numeric transform. */
enum class EntropyEncoding {
  /** Entropy-codes transformed byte planes and eligible residual regions. */
  kAns,

  /** Keeps transformed byte planes directly accessible and residuals raw. */
  kNone,
};

struct CompressionOptions {
  NumericTransform numericTransform{NumericTransform::kAutomatic};
  EntropyEncoding entropyEncoding{EntropyEncoding::kAns};
};

/**
 * @brief Opaque metadata required to reconstruct compressed packed columns.
 *
 * The serialized representation contains only fixed-width signed integers so
 * it can be carried by transports without depending on codec internals.
 */
class PackedColumnsDescriptor {
 public:
  /** Constructs a descriptor from trusted serialized words. */
  explicit PackedColumnsDescriptor(std::vector<int64_t> words);

  /** Parses a serialized descriptor without throwing on invalid input. */
  [[nodiscard]] static std::optional<PackedColumnsDescriptor> deserialize(
      std::span<const int64_t> words);

  /** Returns a read-only view of the serialized descriptor. */
  [[nodiscard]] std::span<const int64_t> serializedView() const noexcept;

  /** Returns a transport-neutral owning copy of this descriptor. */
  [[nodiscard]] std::vector<int64_t> serialize() const;

 private:
  std::vector<int64_t> words_;
};

struct CompressedPackedColumns {
  rmm::device_buffer data;
  PackedColumnsDescriptor descriptor;
};

/**
 * @brief Compresses the GPU allocation owned by `cudf::packed_columns`.
 *
 * The codec inspects the packed column layout and applies the requested
 * transform and entropy policy. It has no transport state or link-rate policy.
 * Each instance is bound to one CUDA stream and is not safe for concurrent
 * calls.
 *
 * Both operations synchronize the bound stream before returning. Inputs may
 * therefore be released on return, and returned buffers are ready for use.
 */
class PackedColumnsCodec {
 public:
  PackedColumnsCodec(rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref temporaryMemoryResource,
                     rmm::device_async_resource_ref outputMemoryResource);
  ~PackedColumnsCodec();

  PackedColumnsCodec(const PackedColumnsCodec&) = delete;
  PackedColumnsCodec& operator=(const PackedColumnsCodec&) = delete;
  PackedColumnsCodec(PackedColumnsCodec&&) = delete;
  PackedColumnsCodec& operator=(PackedColumnsCodec&&) = delete;

  /**
   * Returns no value when the encoded representation fails the configured
   * byte-reduction safeguard. FOR with no entropy coding leaves numeric byte
   * planes directly addressable. The current decompression API still
   * reconstructs the complete packed allocation.
   */
  [[nodiscard]] std::optional<CompressedPackedColumns> compress(
      const cudf::packed_columns& input,
      CompressionOptions options = {});

  /** Reconstructs the packed GPU allocation byte-exactly. */
  [[nodiscard]] rmm::device_buffer decompress(
      cudf::device_span<const uint8_t> input,
      const PackedColumnsDescriptor& descriptor);

 private:
  std::unique_ptr<detail::PackedColumnsCodecState> state_;
};

} // namespace facebook::velox::cudf_velox::compression
