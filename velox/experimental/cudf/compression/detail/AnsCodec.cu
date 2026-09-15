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
#include "velox/experimental/cudf/compression/detail/AnsCodec.h"
#include "velox/experimental/cudf/compression/detail/SizeUtils.h"

#include <cudf/utilities/error.hpp>

// clang-format off (CudfNoDefaults must follow all cuDF headers)
#include "velox/experimental/cudf/CudfNoDefaults.h"
// clang-format on

#include <nvcomp/ans.hpp>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::compression::detail {
namespace {

// Splitting large inputs bounds the maximum temporary output allocation made
// for any one frame.
constexpr std::size_t kAnsSegmentSize = 32u << 20;

// At most five maximum-sized frames are configured together, bounding a
// residual batch to about 160 MiB of input while retaining batched execution.
constexpr std::size_t kMaximumSegmentsPerBatch = 5;

// The packed-column transform can encode up to eight byte planes together.
// One context-owned pinned vector serves both that path and residual batches.
static_assert(kMaximumSegmentsPerBatch <= kAnsSizeStagingCapacity);
static_assert(kNvcompFrameAlignment >= nvcompANSRequiredCompressionAlignment);
static_assert(kNvcompFrameAlignment >= nvcompANSRequiredDecompressionAlignment);

[[nodiscard]] std::vector<cudf::device_span<const uint8_t>> makeSegments(
    cudf::device_span<const uint8_t> input) {
  const auto count =
      (input.size() / kAnsSegmentSize) + (input.size() % kAnsSegmentSize != 0);
  CUDF_EXPECTS(count <= std::numeric_limits<uint32_t>::max(),
               "Too many nvCOMP ANS segments",
               std::overflow_error);

  std::vector<cudf::device_span<const uint8_t>> segments;
  segments.reserve(count);
  for (std::size_t offset = 0; offset < input.size();) {
    const auto size = std::min(kAnsSegmentSize, input.size() - offset);
    segments.emplace_back(input.data() + offset, size);
    offset = checkedAddSizes(offset, size, "nvCOMP ANS input offset overflow");
  }
  return segments;
}

} // namespace

AnsCodecContext::AnsCodecContext(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource)
    : stream_{stream},
      temporaryMemoryResource_{temporaryMemoryResource},
      manager_{std::make_unique<nvcomp::ANSManager>(
          kNvcompAnsChunkSize,
          nvcompBatchedANSCompressDefaultOpts,
          nvcompBatchedANSDecompressDefaultOpts,
          stream.value(),
          nvcomp::NoComputeNoVerify,
          nvcomp::BitstreamKind::NVCOMP_NATIVE)},
      sizeStaging_{cudf::detail::make_pinned_vector_async<std::size_t>(
          kAnsSizeStagingCapacity,
          stream)} {}

AnsCodecContext::~AnsCodecContext() = default;

rmm::cuda_stream_view AnsCodecContext::stream() const noexcept {
  return stream_;
}

rmm::device_async_resource_ref AnsCodecContext::temporaryMemoryResource()
    const noexcept {
  return temporaryMemoryResource_;
}

nvcomp::ANSManager& AnsCodecContext::manager() {
  return *manager_;
}

std::span<std::size_t> AnsCodecContext::sizeStaging(std::size_t count) {
  CUDF_EXPECTS(count <= sizeStaging_.size(),
               "nvCOMP size staging capacity exceeded",
               std::invalid_argument);
  return {sizeStaging_.data(), count};
}

AnsCompressedData compressAnsBatch(
    std::span<const cudf::device_span<const uint8_t>> inputs,
    AnsCodecContext& context) {
  CUDF_EXPECTS(!inputs.empty() && inputs.size() <= kAnsSizeStagingCapacity,
               "Invalid nvCOMP ANS batch size",
               std::invalid_argument);

  std::vector<std::size_t> inputSizes(inputs.size());
  std::vector<const uint8_t*> inputPointers(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    CUDF_EXPECTS(!inputs[index].empty(),
                 "nvCOMP ANS input segment is empty",
                 std::invalid_argument);
    inputSizes[index] = inputs[index].size();
    inputPointers[index] = inputs[index].data();
  }

  auto configs = context.manager().configure_compression(inputSizes);
  std::vector<uint8_t*> outputPointers(inputs.size());
  std::size_t maximumOutputSize = 0;
  for (const auto& config : configs) {
    maximumOutputSize =
        checkedAddSizes(maximumOutputSize,
                        nvcompAlignedSize(config.max_compressed_buffer_size),
                        "nvCOMP ANS scratch size overflow");
  }

  const auto stream = context.stream();
  const auto memoryResource = context.temporaryMemoryResource();
  rmm::device_buffer scratch{maximumOutputSize, stream, memoryResource};
  CUDF_CUDA_TRY(
      cudaMemsetAsync(scratch.data(), 0, scratch.size(), stream.value()));
  std::size_t outputOffset = 0;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    outputPointers[index] =
        static_cast<uint8_t*>(scratch.data()) + outputOffset;
    outputOffset = checkedAddSizes(
        outputOffset,
        nvcompAlignedSize(configs[index].max_compressed_buffer_size),
        "nvCOMP ANS scratch offset overflow");
  }

  rmm::device_buffer outputSizesDevice{
      inputs.size() * sizeof(std::size_t), stream, memoryResource};
  context.manager().compress(
      inputPointers.data(),
      outputPointers.data(),
      configs,
      static_cast<std::size_t*>(outputSizesDevice.data()));

  auto stagedSizes = context.sizeStaging(inputs.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(stagedSizes.data(),
                                outputSizesDevice.data(),
                                inputs.size() * sizeof(std::size_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  std::vector<uint32_t> segmentSizes(inputs.size());
  std::size_t compressedSize = 0;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto size = stagedSizes[index];
    CUDF_EXPECTS(*configs[index].get_status() == nvcompSuccess && size != 0 &&
                     size <= configs[index].max_compressed_buffer_size &&
                     size <= std::numeric_limits<uint32_t>::max(),
                 "nvCOMP ANS compression failed");
    segmentSizes[index] = static_cast<uint32_t>(size);
    compressedSize = checkedAddSizes(compressedSize,
                                     nvcompAlignedSize(size),
                                     "nvCOMP ANS result size overflow");
  }

  rmm::device_buffer output{compressedSize, stream, memoryResource};
  CUDF_CUDA_TRY(
      cudaMemsetAsync(output.data(), 0, output.size(), stream.value()));
  outputOffset = 0;
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto size = segmentSizes[index];
    CUDF_CUDA_TRY(
        cudaMemcpyAsync(static_cast<uint8_t*>(output.data()) + outputOffset,
                        outputPointers[index],
                        size,
                        cudaMemcpyDeviceToDevice,
                        stream.value()));
    outputOffset = checkedAddSizes(
        outputOffset, nvcompAlignedSize(size), "nvCOMP ANS offset overflow");
  }
  return AnsCompressedData{std::move(output), std::move(segmentSizes)};
}

std::optional<AnsCompressedData> compressAns(
    cudf::device_span<const uint8_t> input,
    AnsCodecContext& context) {
  if (input.size() < kNvcompAnsChunkSize) {
    return std::nullopt;
  }

  const auto stream = context.stream();
  const auto temporaryMemoryResource = context.temporaryMemoryResource();
  const auto inputSegments = makeSegments(input);
  std::vector<uint32_t> segmentSizes(inputSegments.size());

  std::size_t compressedTotal = 0;
  std::vector<rmm::device_buffer> batches;
  batches.reserve((inputSegments.size() / kMaximumSegmentsPerBatch) +
                  (inputSegments.size() % kMaximumSegmentsPerBatch != 0));

  for (std::size_t first = 0; first < inputSegments.size();
       first += kMaximumSegmentsPerBatch) {
    const auto count =
        std::min(kMaximumSegmentsPerBatch, inputSegments.size() - first);
    auto batch = compressAnsBatch(
        std::span{inputSegments}.subspan(first, count), context);
    std::copy(batch.segmentSizes.begin(),
              batch.segmentSizes.end(),
              segmentSizes.begin() + first);
    compressedTotal = checkedAddSizes(
        compressedTotal, batch.data.size(), "nvCOMP ANS result size overflow");
    batches.push_back(std::move(batch.data));
  }

  if (static_cast<long double>(compressedTotal) >
      (1.0L - static_cast<long double>(kMinimumEncodedByteReduction)) *
          static_cast<long double>(input.size())) {
    stream.synchronize();
    return std::nullopt;
  }

  rmm::device_buffer output{compressedTotal, stream, temporaryMemoryResource};
  std::size_t outputOffset = 0;
  for (const auto& batch : batches) {
    CUDF_CUDA_TRY(
        cudaMemcpyAsync(static_cast<uint8_t*>(output.data()) + outputOffset,
                        batch.data(),
                        batch.size(),
                        cudaMemcpyDeviceToDevice,
                        stream.value()));
    outputOffset = checkedAddSizes(
        outputOffset, batch.size(), "nvCOMP ANS output offset overflow");
  }
  stream.synchronize();
  return AnsCompressedData{std::move(output), std::move(segmentSizes)};
}

rmm::device_buffer decompressAns(cudf::device_span<const uint8_t> input,
                                 std::span<const uint32_t> segmentSizes,
                                 std::size_t uncompressedSize,
                                 AnsCodecContext& context) {
  CUDF_EXPECTS(!segmentSizes.empty(),
               "nvCOMP ANS descriptor has no segments",
               std::invalid_argument);

  std::size_t encodedSize = 0;
  for (const auto size : segmentSizes) {
    CUDF_EXPECTS(size != 0, "nvCOMP ANS segment is empty");
    encodedSize = checkedAddSizes(
        encodedSize, nvcompAlignedSize(size), "nvCOMP ANS input size overflow");
  }
  CUDF_EXPECTS(encodedSize == input.size(),
               "nvCOMP ANS segment sizes do not match the input span",
               std::invalid_argument);

  const auto expectedSegmentCount = (uncompressedSize / kAnsSegmentSize) +
      (uncompressedSize % kAnsSegmentSize != 0);
  CUDF_EXPECTS(expectedSegmentCount == segmentSizes.size(),
               "nvCOMP ANS segment count does not match the output size",
               std::invalid_argument);

  const auto stream = context.stream();
  rmm::device_buffer output{
      uncompressedSize, stream, context.temporaryMemoryResource()};
  std::size_t inputOffset = 0;
  std::size_t outputOffset = 0;

  for (std::size_t first = 0; first < segmentSizes.size();
       first += kMaximumSegmentsPerBatch) {
    const auto count =
        std::min(kMaximumSegmentsPerBatch, segmentSizes.size() - first);
    std::vector<const uint8_t*> inputPointers(count);
    std::vector<uint8_t*> outputPointers(count);
    std::vector<std::size_t> expectedSizes(count);

    for (std::size_t local = 0; local < count; ++local) {
      const auto index = first + local;
      inputPointers[local] = input.data() + inputOffset;
      inputOffset = checkedAddSizes(inputOffset,
                                    nvcompAlignedSize(segmentSizes[index]),
                                    "nvCOMP ANS input offset overflow");
      outputPointers[local] =
          static_cast<uint8_t*>(output.data()) + outputOffset;
      expectedSizes[local] =
          std::min(kAnsSegmentSize, uncompressedSize - outputOffset);
      outputOffset = checkedAddSizes(outputOffset,
                                     expectedSizes[local],
                                     "nvCOMP ANS output offset overflow");
    }

    auto configs =
        context.manager().configure_decompression(inputPointers.data(), count);
    for (std::size_t local = 0; local < count; ++local) {
      CUDF_EXPECTS(*configs[local].get_status() == nvcompSuccess &&
                       configs[local].decomp_data_size == expectedSizes[local],
                   "Invalid nvCOMP ANS frame");
    }

    context.manager().decompress(
        outputPointers.data(), inputPointers.data(), configs);
    stream.synchronize();
    for (const auto& config : configs) {
      CUDF_EXPECTS(*config.get_status() == nvcompSuccess,
                   "nvCOMP ANS decompression failed");
    }
  }

  CUDF_EXPECTS(inputOffset == input.size() && outputOffset == uncompressedSize,
               "nvCOMP ANS descriptor does not cover its buffers",
               std::invalid_argument);
  return output;
}

} // namespace facebook::velox::cudf_velox::compression::detail
