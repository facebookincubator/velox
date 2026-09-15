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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace nvcomp {
struct ANSManager;
}

namespace facebook::velox::cudf_velox::compression::detail {
// Internal native-ANS chunk size passed to nvCOMP. The codec's descriptor
// indexes the larger frames submitted to the manager, not these inner chunks.
inline constexpr std::size_t kNvcompAnsChunkSize = 64u << 10;
inline constexpr double kMinimumEncodedByteReduction = 0.02;

// One pinned size entry is required for every ANS segment submitted together.
// Callers statically assert that their largest batch fits this shared staging.
inline constexpr std::size_t kAnsSizeStagingCapacity = 8;

/**
 * Owns the nvCOMP ANS manager and pinned size staging for one CUDA stream.
 * The context is not safe for concurrent calls and must not outlive its stream.
 */
class AnsCodecContext {
 public:
  AnsCodecContext(rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref temporaryMemoryResource);
  ~AnsCodecContext();

  AnsCodecContext(const AnsCodecContext&) = delete;
  AnsCodecContext& operator=(const AnsCodecContext&) = delete;

  [[nodiscard]] rmm::cuda_stream_view stream() const noexcept;
  [[nodiscard]] rmm::device_async_resource_ref temporaryMemoryResource()
      const noexcept;
  [[nodiscard]] nvcomp::ANSManager& manager();
  [[nodiscard]] std::span<std::size_t> sizeStaging(std::size_t count);

 private:
  rmm::cuda_stream_view stream_;
  rmm::device_async_resource_ref temporaryMemoryResource_;
  std::unique_ptr<nvcomp::ANSManager> manager_;
  cudf::detail::host_vector<std::size_t> sizeStaging_;
};

struct AnsCompressedData {
  rmm::device_buffer data;
  std::vector<uint32_t> segmentSizes;
};

/** Compresses one bounded batch of independent ANS inputs. */
[[nodiscard]] AnsCompressedData compressAnsBatch(
    std::span<const cudf::device_span<const uint8_t>> inputs,
    AnsCodecContext& context);

/**
 * Compresses one contiguous device span into native nvCOMP ANS frames.
 *
 * Large inputs are divided into bounded segments. The returned device buffer
 * contains 16-byte-aligned frame extents. `segmentSizes` records each frame's
 * true length, and all transmitted padding is initialized to zero. This
 * function synchronizes the context stream before returning.
 */
[[nodiscard]] std::optional<AnsCompressedData> compressAns(
    cudf::device_span<const uint8_t> input,
    std::size_t minimumInputSize,
    AnsCodecContext& context);

/**
 * Decompresses native nvCOMP ANS frames byte-exactly.
 *
 * This function validates frame and output sizes and synchronizes the context
 * stream before returning.
 */
[[nodiscard]] rmm::device_buffer decompressAns(
    cudf::device_span<const uint8_t> input,
    std::span<const uint32_t> segmentSizes,
    std::size_t uncompressedSize,
    AnsCodecContext& context);

} // namespace facebook::velox::cudf_velox::compression::detail
