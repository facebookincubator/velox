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
#include "velox/experimental/cudf/compression/PackedColumnsCodec.h"
#include "velox/experimental/cudf/compression/detail/AnsCodec.h"
#include "velox/experimental/cudf/compression/detail/SizeUtils.h"

#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

// clang-format off (CudfNoDefaults must follow all cuDF headers)
#include "velox/experimental/cudf/CudfNoDefaults.h"
// clang-format on

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include <nvcomp/ans.hpp>

#include <algorithm>
#include <bit>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::compression {
namespace {

// This threshold avoids reduction and transform launches for very small
// columns. It is a codec heuristic, not a format requirement.
constexpr std::size_t kMinimumTypedElementCount = 4096;

constexpr int kThreadsPerBlock = 256;

// ASCII "VLXPCOMP". Format identity and version are separate wire fields.
constexpr int64_t kDescriptorMagic = 0x564c5850434f4d50LL;
constexpr int64_t kDescriptorVersion = 2;
constexpr std::size_t kDescriptorHeaderWordCount = 4;
constexpr std::size_t kRegionFixedWordCount = 6;
constexpr std::size_t kMaximumBytePlaneCount = sizeof(uint64_t);
static_assert(kMaximumBytePlaneCount <= detail::kAnsSizeStagingCapacity);
constexpr std::size_t kMinimumStagingIndex = 0;
constexpr std::size_t kMaximumStagingIndex = 1;
constexpr std::size_t kDeltaMaximumStagingIndex = 2;
constexpr std::size_t kFirstValueStagingIndex = 3;
constexpr std::size_t kNumericStagingValueCount = 4;

enum class RegionCodec : int64_t {
  kRaw = 0,
  kByteAns = 1,
  kFrameOfReference = 2,
  kDeltaFrameOfReference = 3,
};

struct EncodedRegion {
  std::size_t rawSize{0};
  RegionCodec codec{RegionCodec::kRaw};
  cudf::data_type logicalType{cudf::type_id::EMPTY};
  uint64_t referenceBits{0};
  std::vector<uint32_t> segmentSizes;
};

struct ParsedDescriptor {
  std::size_t uncompressedSize{0};
  std::size_t compressedSize{0};
  std::vector<EncodedRegion> regions;
};

class DescriptorReader {
 public:
  explicit DescriptorReader(std::span<const int64_t> words) : words_{words} {}

  [[nodiscard]] std::optional<int64_t> read() {
    if (position_ == words_.size()) {
      return std::nullopt;
    }
    return words_[position_++];
  }

  [[nodiscard]] std::optional<std::size_t> readSize() {
    const auto value = read();
    if (!value || *value < 0) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(*value);
  }

  [[nodiscard]] std::optional<uint32_t> readUint32() {
    const auto value = read();
    if (!value || *value < 0 ||
        static_cast<uint64_t>(*value) > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }
    return static_cast<uint32_t>(*value);
  }

  [[nodiscard]] std::size_t remaining() const noexcept {
    return words_.size() - position_;
  }

  [[nodiscard]] bool empty() const noexcept {
    return position_ == words_.size();
  }

 private:
  std::span<const int64_t> words_;
  std::size_t position_{0};
};

[[nodiscard]] std::optional<cudf::data_type> parseLogicalType(
    int64_t idValue,
    int64_t scaleValue) {
  if (idValue < static_cast<int64_t>(cudf::type_id::EMPTY) ||
      idValue >= static_cast<int64_t>(cudf::type_id::NUM_TYPE_IDS) ||
      scaleValue < std::numeric_limits<int32_t>::min() ||
      scaleValue > std::numeric_limits<int32_t>::max()) {
    return std::nullopt;
  }

  const auto id = static_cast<cudf::type_id>(idValue);
  const auto scale = static_cast<int32_t>(scaleValue);
  if (id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64 ||
      id == cudf::type_id::DECIMAL128) {
    return cudf::data_type{id, scale};
  }
  if (scale != 0) {
    return std::nullopt;
  }
  return cudf::data_type{id};
}

[[nodiscard]] std::optional<ParsedDescriptor> parseDescriptor(
    std::span<const int64_t> words) {
  DescriptorReader reader{words};
  const auto magic = reader.read();
  const auto version = reader.read();
  if (!magic || !version || *magic != kDescriptorMagic ||
      *version != kDescriptorVersion) {
    return std::nullopt;
  }

  const auto uncompressedSize = reader.readSize();
  const auto regionCount = reader.readSize();
  if (!uncompressedSize || !regionCount || *uncompressedSize == 0 ||
      *regionCount == 0) {
    return std::nullopt;
  }

  // Each region has six fixed fields before its segment-size list. This
  // bound prevents an untrusted count from causing a disproportionate reserve.
  if (*regionCount > reader.remaining() / kRegionFixedWordCount) {
    return std::nullopt;
  }

  ParsedDescriptor parsed;
  parsed.uncompressedSize = *uncompressedSize;
  parsed.regions.reserve(*regionCount);

  std::size_t rawCoverage = 0;
  std::size_t encodedCoverage = 0;
  for (std::size_t index = 0; index < *regionCount; ++index) {
    const auto rawSize = reader.readSize();
    const auto codecValue = reader.read();
    const auto typeValue = reader.read();
    const auto scaleValue = reader.read();
    const auto referenceValue = reader.read();
    const auto segmentCount = reader.readSize();
    if (!rawSize || !codecValue || !typeValue || !scaleValue ||
        !referenceValue || !segmentCount || *rawSize == 0 ||
        *segmentCount > reader.remaining()) {
      return std::nullopt;
    }

    if (*codecValue < static_cast<int64_t>(RegionCodec::kRaw) ||
        *codecValue >
            static_cast<int64_t>(RegionCodec::kDeltaFrameOfReference)) {
      return std::nullopt;
    }
    const auto codec = static_cast<RegionCodec>(*codecValue);
    const auto logicalType = parseLogicalType(*typeValue, *scaleValue);
    if (!logicalType) {
      return std::nullopt;
    }

    EncodedRegion region;
    region.rawSize = *rawSize;
    region.codec = codec;
    region.logicalType = *logicalType;
    region.referenceBits = std::bit_cast<uint64_t>(*referenceValue);
    region.segmentSizes.reserve(*segmentCount);
    for (std::size_t segment = 0; segment < *segmentCount; ++segment) {
      const auto segmentSize = reader.readUint32();
      if (!segmentSize || *segmentSize == 0) {
        return std::nullopt;
      }
      region.segmentSizes.push_back(*segmentSize);
    }

    std::size_t regionEncodedSize = 0;
    if (codec == RegionCodec::kRaw) {
      if (!region.segmentSizes.empty() ||
          region.logicalType.id() != cudf::type_id::EMPTY ||
          region.referenceBits != 0) {
        return std::nullopt;
      }
      regionEncodedSize = region.rawSize;
    } else {
      if (codec == RegionCodec::kByteAns) {
        if (region.segmentSizes.empty() ||
            region.logicalType.id() != cudf::type_id::EMPTY ||
            region.referenceBits != 0) {
          return std::nullopt;
        }
      } else {
        if (!cudf::is_fixed_width(region.logicalType)) {
          return std::nullopt;
        }
        const auto elementWidth = cudf::size_of(region.logicalType);
        if ((elementWidth != 4 && elementWidth != 8) ||
            region.rawSize % elementWidth != 0 ||
            rawCoverage % elementWidth != 0 || region.segmentSizes.empty() ||
            region.segmentSizes.size() > kMaximumBytePlaneCount ||
            (codec == RegionCodec::kFrameOfReference &&
             region.segmentSizes.size() > elementWidth)) {
          return std::nullopt;
        }
      }

      for (const auto size : region.segmentSizes) {
        std::size_t alignedSize = 0;
        if (!detail::tryNvcompAlignedSize(size, alignedSize) ||
            !detail::tryAddSizes(
                regionEncodedSize, alignedSize, regionEncodedSize)) {
          return std::nullopt;
        }
      }
    }

    std::size_t regionWireSize = 0;
    if (!detail::tryNvcompAlignedSize(regionEncodedSize, regionWireSize) ||
        !detail::tryAddSizes(rawCoverage, region.rawSize, rawCoverage) ||
        rawCoverage > parsed.uncompressedSize ||
        !detail::tryAddSizes(
            encodedCoverage, regionWireSize, encodedCoverage)) {
      return std::nullopt;
    }
    parsed.regions.push_back(std::move(region));
  }

  if (!reader.empty() || rawCoverage != parsed.uncompressedSize) {
    return std::nullopt;
  }
  parsed.compressedSize = encodedCoverage;
  return parsed;
}

[[nodiscard]] int64_t checkedDescriptorWord(std::size_t value) {
  CUDF_EXPECTS(
      value <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
      "Packed-column descriptor value exceeds int64",
      std::overflow_error);
  return static_cast<int64_t>(value);
}

[[nodiscard]] std::vector<int64_t> serializeDescriptor(
    const std::vector<EncodedRegion>& regions,
    std::size_t uncompressedSize) {
  std::size_t wordCount = kDescriptorHeaderWordCount;
  for (const auto& region : regions) {
    wordCount = detail::checkedAddSizes(
        wordCount,
        detail::checkedAddSizes(kRegionFixedWordCount,
                                region.segmentSizes.size(),
                                "Packed-column descriptor size overflow"),
        "Packed-column descriptor size overflow");
  }

  std::vector<int64_t> words;
  words.reserve(wordCount);
  words.push_back(kDescriptorMagic);
  words.push_back(kDescriptorVersion);
  words.push_back(checkedDescriptorWord(uncompressedSize));
  words.push_back(checkedDescriptorWord(regions.size()));

  for (const auto& region : regions) {
    words.push_back(checkedDescriptorWord(region.rawSize));
    words.push_back(static_cast<int64_t>(region.codec));
    words.push_back(static_cast<int64_t>(region.logicalType.id()));
    words.push_back(static_cast<int64_t>(region.logicalType.scale()));
    words.push_back(std::bit_cast<int64_t>(region.referenceBits));
    words.push_back(checkedDescriptorWord(region.segmentSizes.size()));
    for (const auto size : region.segmentSizes) {
      words.push_back(static_cast<int64_t>(size));
    }
  }
  return words;
}

[[nodiscard]] int bytePlaneCount(uint64_t range) noexcept {
  if (range == 0) {
    return 1;
  }
  return static_cast<int>((std::bit_width(range) + 7) / 8);
}

template <typename T>
[[nodiscard]] uint64_t valueBits(T value) noexcept {
  using Unsigned = std::make_unsigned_t<T>;
  return static_cast<uint64_t>(std::bit_cast<Unsigned>(value));
}

template <typename T>
[[nodiscard]] T valueFromBits(uint64_t bits) noexcept {
  using Unsigned = std::make_unsigned_t<T>;
  return std::bit_cast<T>(static_cast<Unsigned>(bits));
}

[[nodiscard]] bool usesUnsignedStorage(cudf::data_type type) noexcept {
  return type.id() == cudf::type_id::UINT32 ||
      type.id() == cudf::type_id::UINT64;
}

template <typename T>
__global__ void subtractAndSplitKernel(const T* values,
                                       T base,
                                       uint8_t* planes,
                                       uint32_t size,
                                       std::size_t planeStride,
                                       int planeCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }

  using Unsigned = std::make_unsigned_t<T>;
  const auto adjusted =
      static_cast<Unsigned>(values[index]) - static_cast<Unsigned>(base);
  for (int plane = 0; plane < planeCount; ++plane) {
    planes[static_cast<std::size_t>(plane) * planeStride + index] =
        static_cast<uint8_t>(adjusted >> (8 * plane));
  }
}

template <typename T>
__global__ void recombineAndAddKernel(const uint8_t* planes,
                                      T base,
                                      void* output,
                                      uint32_t size,
                                      std::size_t planeStride,
                                      int planeCount) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }

  using Unsigned = std::make_unsigned_t<T>;
  Unsigned adjusted = 0;
  for (int plane = 0; plane < planeCount; ++plane) {
    adjusted |=
        static_cast<Unsigned>(
            planes[static_cast<std::size_t>(plane) * planeStride + index])
        << (8 * plane);
  }
  static_cast<Unsigned*>(output)[index] =
      static_cast<Unsigned>(base) + adjusted;
}

template <typename T>
__global__ void zigzagDeltaKernel(const T* values,
                                  uint64_t* output,
                                  uint32_t size) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }

  using Unsigned = std::make_unsigned_t<T>;
  constexpr auto kBits = sizeof(Unsigned) * 8;
  const auto current = static_cast<Unsigned>(values[index]);
  const auto previous =
      index == 0 ? current : static_cast<Unsigned>(values[index - 1]);
  const auto delta = current - previous;
  const auto signMask = Unsigned{0} - (delta >> (kBits - 1));
  const auto zigzag = static_cast<Unsigned>((delta << 1) ^ signMask);
  output[index] = static_cast<uint64_t>(zigzag);
}

__global__ void unzigzagKernel(uint64_t* values, uint32_t size) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }
  const auto zigzag = values[index];
  values[index] = (zigzag >> 1) ^ (uint64_t{0} - (zigzag & 1));
}

template <typename T>
__global__ void finalizeDeltaKernel(const uint64_t* prefixSums,
                                    uint64_t firstBits,
                                    void* output,
                                    uint32_t size) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) {
    return;
  }

  using Unsigned = std::make_unsigned_t<T>;
  static_cast<Unsigned*>(output)[index] = static_cast<Unsigned>(firstBits) +
      static_cast<Unsigned>(prefixSums[index]);
}

struct TypedRegion {
  std::size_t offset;
  std::size_t elementCount;
  cudf::data_type logicalType;
};

void collectTypedRegions(const cudf::column_view& column,
                         const uint8_t* blobBase,
                         std::size_t blobSize,
                         std::vector<TypedRegion>& output) {
  const auto type = column.type();
  if (column.size() >=
          static_cast<cudf::size_type>(kMinimumTypedElementCount) &&
      cudf::is_fixed_width(type) && column.head<uint8_t>() != nullptr) {
    const auto elementWidth = cudf::size_of(type);
    if (elementWidth == 4 || elementWidth == 8) {
      CUDF_EXPECTS(column.offset() >= 0, "Negative packed-column offset");
      const auto byteOffset = detail::checkedMultiplySizes(
          static_cast<std::size_t>(column.offset()),
          elementWidth,
          "Packed-column element offset overflow");
      const auto headAddress =
          reinterpret_cast<std::uintptr_t>(column.head<uint8_t>());
      const auto dataAddress = detail::checkedAddSizes(
          headAddress, byteOffset, "Packed-column pointer overflow");
      const auto baseAddress = reinterpret_cast<std::uintptr_t>(blobBase);
      CUDF_EXPECTS(dataAddress >= baseAddress,
                   "Packed column points before its GPU allocation");
      const auto offset = dataAddress - baseAddress;
      const auto byteSize =
          detail::checkedMultiplySizes(static_cast<std::size_t>(column.size()),
                                       elementWidth,
                                       "Packed-column byte size overflow");
      CUDF_EXPECTS(offset <= blobSize && byteSize <= blobSize - offset,
                   "Packed column points outside its GPU allocation");
      output.push_back(
          TypedRegion{offset, static_cast<std::size_t>(column.size()), type});
    }
  }

  for (auto child = 0; child < column.num_children(); ++child) {
    collectTypedRegions(column.child(child), blobBase, blobSize, output);
  }
}

class DeferredAnsStatusChecks {
 public:
  void add(std::vector<nvcomp::DecompressionConfig> configs) {
    pending_.push_back(std::move(configs));
  }

  void verify() {
    for (const auto& batch : pending_) {
      for (const auto& config : batch) {
        CUDF_EXPECTS(*config.get_status() == nvcompSuccess,
                     "nvCOMP byte-plane decompression failed");
      }
    }
    pending_.clear();
  }

 private:
  std::vector<std::vector<nvcomp::DecompressionConfig>> pending_;
};

[[nodiscard]] detail::AnsCompressedData encodePlanes(
    cudf::device_span<const uint8_t> planes,
    uint32_t elementCount,
    int planeCount,
    detail::AnsCodecContext& context) {
  const auto planeStride = detail::nvcompAlignedSize(elementCount);
  CUDF_EXPECTS(
      planeCount > 0 && planeCount <= kMaximumBytePlaneCount &&
          planes.size() ==
              detail::checkedMultiplySizes(planeStride,
                                           static_cast<std::size_t>(planeCount),
                                           "Byte-plane input size overflow"),
      "Invalid byte-plane compression input",
      std::invalid_argument);

  std::vector<cudf::device_span<const uint8_t>> inputs;
  inputs.reserve(planeCount);
  for (int plane = 0; plane < planeCount; ++plane) {
    inputs.emplace_back(
        planes.data() + static_cast<std::size_t>(plane) * planeStride,
        elementCount);
  }
  return detail::compressAnsBatch(inputs, context);
}

[[nodiscard]] rmm::device_buffer decodePlanes(
    cudf::device_span<const uint8_t> input,
    std::span<const uint32_t> segmentSizes,
    uint32_t elementCount,
    DeferredAnsStatusChecks& pending,
    detail::AnsCodecContext& context) {
  CUDF_EXPECTS(
      !segmentSizes.empty() && segmentSizes.size() <= kMaximumBytePlaneCount,
      "Invalid byte-plane segment count",
      std::invalid_argument);

  std::size_t encodedSize = 0;
  for (const auto size : segmentSizes) {
    encodedSize = detail::checkedAddSizes(encodedSize,
                                          detail::nvcompAlignedSize(size),
                                          "Byte-plane input size overflow");
  }
  CUDF_EXPECTS(encodedSize == input.size(),
               "Byte-plane sizes do not match the input span",
               std::invalid_argument);

  const auto planeStride = detail::nvcompAlignedSize(elementCount);
  const auto planeBytes = detail::checkedMultiplySizes(
      planeStride, segmentSizes.size(), "Byte-plane output size overflow");
  const auto stream = context.stream();
  rmm::device_buffer planes{
      planeBytes, stream, context.temporaryMemoryResource()};
  std::vector<const uint8_t*> inputPointers(segmentSizes.size());
  std::vector<uint8_t*> outputPointers(segmentSizes.size());
  std::size_t inputOffset = 0;
  for (std::size_t plane = 0; plane < segmentSizes.size(); ++plane) {
    inputPointers[plane] = input.data() + inputOffset;
    inputOffset =
        detail::checkedAddSizes(inputOffset,
                                detail::nvcompAlignedSize(segmentSizes[plane]),
                                "Byte-plane input offset overflow");
    outputPointers[plane] =
        static_cast<uint8_t*>(planes.data()) + plane * planeStride;
  }

  auto configs = context.manager().configure_decompression(inputPointers.data(),
                                                           segmentSizes.size());
  for (const auto& config : configs) {
    CUDF_EXPECTS(*config.get_status() == nvcompSuccess &&
                     config.decomp_data_size == elementCount,
                 "Invalid nvCOMP byte-plane frame");
  }
  context.manager().decompress(
      outputPointers.data(), inputPointers.data(), configs);
  pending.add(std::move(configs));
  return planes;
}

} // namespace

namespace detail {

class PackedColumnsCodecState {
 public:
  PackedColumnsCodecState(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref temporaryMemoryResource,
      rmm::device_async_resource_ref outputMemoryResource)
      : stream{stream},
        temporaryMemoryResource{temporaryMemoryResource},
        outputMemoryResource{outputMemoryResource},
        ans{stream, temporaryMemoryResource},
        numericStaging{cudf::detail::make_pinned_vector_async<uint64_t>(
            kNumericStagingValueCount,
            stream)} {}

  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref temporaryMemoryResource;
  rmm::device_async_resource_ref outputMemoryResource;
  AnsCodecContext ans;
  cudf::detail::host_vector<uint64_t> numericStaging;
};

} // namespace detail

namespace {

struct EncodedTypedRegion {
  EncodedRegion descriptor;
  rmm::device_buffer data;
};

template <typename T>
EncodedTypedRegion encodeTypedRegion(const uint8_t* blobBase,
                                     const TypedRegion& region,
                                     detail::PackedColumnsCodecState& state) {
  CUDF_EXPECTS(region.elementCount <= std::numeric_limits<uint32_t>::max(),
               "Packed column is too large for the byte-plane codec",
               std::overflow_error);
  const auto elementCount = static_cast<uint32_t>(region.elementCount);
  const auto* values = reinterpret_cast<const T*>(blobBase + region.offset);
  const auto blocks = (elementCount / kThreadsPerBlock) +
      (elementCount % kThreadsPerBlock != 0);
  const auto stream = state.stream;
  const auto temporaryMemoryResource = state.temporaryMemoryResource;

  rmm::device_buffer minMaxOutput{
      2 * sizeof(T), stream, temporaryMemoryResource};
  auto* minimumOutput = static_cast<T*>(minMaxOutput.data());
  auto* maximumOutput = minimumOutput + 1;

  std::size_t minimumTempSize = 0;
  std::size_t maximumTempSize = 0;
  CUDF_CUDA_TRY(cub::DeviceReduce::Min(nullptr,
                                       minimumTempSize,
                                       values,
                                       minimumOutput,
                                       elementCount,
                                       stream.value()));
  CUDF_CUDA_TRY(cub::DeviceReduce::Max(nullptr,
                                       maximumTempSize,
                                       values,
                                       maximumOutput,
                                       elementCount,
                                       stream.value()));
  rmm::device_buffer reductionTemporary{
      std::max(minimumTempSize, maximumTempSize),
      stream,
      temporaryMemoryResource};
  CUDF_CUDA_TRY(cub::DeviceReduce::Min(reductionTemporary.data(),
                                       minimumTempSize,
                                       values,
                                       minimumOutput,
                                       elementCount,
                                       stream.value()));
  CUDF_CUDA_TRY(cub::DeviceReduce::Max(reductionTemporary.data(),
                                       maximumTempSize,
                                       values,
                                       maximumOutput,
                                       elementCount,
                                       stream.value()));

  auto* staged = state.numericStaging.data();
  CUDF_CUDA_TRY(cudaMemcpyAsync(staged + kMinimumStagingIndex,
                                minimumOutput,
                                sizeof(T),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(staged + kMaximumStagingIndex,
                                maximumOutput,
                                sizeof(T),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  const auto minimum = valueFromBits<T>(staged[kMinimumStagingIndex]);
  const auto maximum = valueFromBits<T>(staged[kMaximumStagingIndex]);
  using Unsigned = std::make_unsigned_t<T>;
  const auto frameRange = static_cast<uint64_t>(static_cast<Unsigned>(maximum) -
                                                static_cast<Unsigned>(minimum));
  const auto framePlaneCount = bytePlaneCount(frameRange);

  rmm::device_buffer deltas{
      detail::checkedMultiplySizes(
          elementCount, sizeof(uint64_t), "Delta buffer size overflow"),
      stream,
      temporaryMemoryResource};
  auto* deltaValues = static_cast<uint64_t*>(deltas.data());
  zigzagDeltaKernel<T><<<blocks, kThreadsPerBlock, 0, stream.value()>>>(
      values, deltaValues, elementCount);
  CUDF_CUDA_TRY(cudaGetLastError());

  rmm::device_buffer deltaMaximum{
      sizeof(uint64_t), stream, temporaryMemoryResource};
  std::size_t deltaTempSize = 0;
  CUDF_CUDA_TRY(
      cub::DeviceReduce::Max(nullptr,
                             deltaTempSize,
                             deltaValues,
                             static_cast<uint64_t*>(deltaMaximum.data()),
                             elementCount,
                             stream.value()));
  rmm::device_buffer deltaTemporary{
      deltaTempSize, stream, temporaryMemoryResource};
  CUDF_CUDA_TRY(
      cub::DeviceReduce::Max(deltaTemporary.data(),
                             deltaTempSize,
                             deltaValues,
                             static_cast<uint64_t*>(deltaMaximum.data()),
                             elementCount,
                             stream.value()));

  CUDF_CUDA_TRY(cudaMemcpyAsync(staged + kDeltaMaximumStagingIndex,
                                deltaMaximum.data(),
                                sizeof(uint64_t),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(staged + kFirstValueStagingIndex,
                                values,
                                sizeof(T),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  const auto deltaPlaneCount =
      bytePlaneCount(staged[kDeltaMaximumStagingIndex]);
  const auto firstBits =
      valueBits(valueFromBits<T>(staged[kFirstValueStagingIndex]));
  const auto useDelta = deltaPlaneCount < framePlaneCount;
  const auto planeCount = useDelta ? deltaPlaneCount : framePlaneCount;
  const auto planeStride = detail::nvcompAlignedSize(elementCount);

  rmm::device_buffer planes{
      detail::checkedMultiplySizes(planeStride,
                                   static_cast<std::size_t>(planeCount),
                                   "Byte-plane buffer size overflow"),
      stream,
      temporaryMemoryResource};

  EncodedRegion descriptor;
  descriptor.rawSize = detail::checkedMultiplySizes(
      region.elementCount, sizeof(T), "Typed region size overflow");
  descriptor.logicalType = region.logicalType;

  if (useDelta) {
    descriptor.codec = RegionCodec::kDeltaFrameOfReference;
    descriptor.referenceBits = firstBits;
    subtractAndSplitKernel<uint64_t>
        <<<blocks, kThreadsPerBlock, 0, stream.value()>>>(
            deltaValues,
            uint64_t{0},
            static_cast<uint8_t*>(planes.data()),
            elementCount,
            planeStride,
            planeCount);
  } else {
    descriptor.codec = RegionCodec::kFrameOfReference;
    descriptor.referenceBits = valueBits(minimum);
    subtractAndSplitKernel<T><<<blocks, kThreadsPerBlock, 0, stream.value()>>>(
        values,
        minimum,
        static_cast<uint8_t*>(planes.data()),
        elementCount,
        planeStride,
        planeCount);
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  auto encoded =
      encodePlanes({static_cast<const uint8_t*>(planes.data()), planes.size()},
                   elementCount,
                   planeCount,
                   state.ans);
  descriptor.segmentSizes = std::move(encoded.segmentSizes);
  return EncodedTypedRegion{std::move(descriptor), std::move(encoded.data)};
}

template <typename T>
void decodeTypedRegion(cudf::device_span<const uint8_t> input,
                       const EncodedRegion& region,
                       uint8_t* output,
                       DeferredAnsStatusChecks& pending,
                       detail::PackedColumnsCodecState& state) {
  const auto elementWidth = sizeof(T);
  CUDF_EXPECTS(
      region.rawSize % elementWidth == 0 &&
          region.rawSize / elementWidth <= std::numeric_limits<uint32_t>::max(),
      "Invalid typed-region element count",
      std::invalid_argument);
  const auto elementCount =
      static_cast<uint32_t>(region.rawSize / elementWidth);
  const auto blocks = (elementCount / kThreadsPerBlock) +
      (elementCount % kThreadsPerBlock != 0);
  const auto stream = state.stream;

  auto planes = decodePlanes(
      input, region.segmentSizes, elementCount, pending, state.ans);
  const auto planeCount = static_cast<int>(region.segmentSizes.size());
  const auto planeStride = detail::nvcompAlignedSize(elementCount);

  if (region.codec == RegionCodec::kFrameOfReference) {
    recombineAndAddKernel<T><<<blocks, kThreadsPerBlock, 0, stream.value()>>>(
        static_cast<const uint8_t*>(planes.data()),
        valueFromBits<T>(region.referenceBits),
        output,
        elementCount,
        planeStride,
        planeCount);
    CUDF_CUDA_TRY(cudaGetLastError());
    return;
  }

  rmm::device_buffer deltas{
      detail::checkedMultiplySizes(
          elementCount, sizeof(uint64_t), "Delta decode buffer size overflow"),
      stream,
      state.temporaryMemoryResource};
  auto* deltaValues = static_cast<uint64_t*>(deltas.data());
  recombineAndAddKernel<uint64_t>
      <<<blocks, kThreadsPerBlock, 0, stream.value()>>>(
          static_cast<const uint8_t*>(planes.data()),
          uint64_t{0},
          deltaValues,
          elementCount,
          planeStride,
          planeCount);
  CUDF_CUDA_TRY(cudaGetLastError());

  unzigzagKernel<<<blocks, kThreadsPerBlock, 0, stream.value()>>>(deltaValues,
                                                                  elementCount);
  CUDF_CUDA_TRY(cudaGetLastError());

  std::size_t scanTemporarySize = 0;
  CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(nullptr,
                                              scanTemporarySize,
                                              deltaValues,
                                              deltaValues,
                                              elementCount,
                                              stream.value()));
  rmm::device_buffer scanTemporary{
      scanTemporarySize, stream, state.temporaryMemoryResource};
  CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(scanTemporary.data(),
                                              scanTemporarySize,
                                              deltaValues,
                                              deltaValues,
                                              elementCount,
                                              stream.value()));

  finalizeDeltaKernel<T><<<blocks, kThreadsPerBlock, 0, stream.value()>>>(
      deltaValues, region.referenceBits, output, elementCount);
  CUDF_CUDA_TRY(cudaGetLastError());
}

[[nodiscard]] std::size_t encodedRegionSize(const EncodedRegion& region) {
  if (region.codec == RegionCodec::kRaw) {
    return region.rawSize;
  }
  std::size_t result = 0;
  for (const auto size : region.segmentSizes) {
    result = detail::checkedAddSizes(result,
                                     detail::nvcompAlignedSize(size),
                                     "Encoded region size overflow");
  }
  return result;
}

} // namespace

PackedColumnsDescriptor::PackedColumnsDescriptor(std::vector<int64_t> words)
    : words_{std::move(words)} {}

std::optional<PackedColumnsDescriptor> PackedColumnsDescriptor::deserialize(
    std::span<const int64_t> words) {
  const auto parsed = parseDescriptor(words);
  if (!parsed) {
    return std::nullopt;
  }
  return PackedColumnsDescriptor{
      std::vector<int64_t>{words.begin(), words.end()}};
}

std::vector<int64_t> PackedColumnsDescriptor::serialize() const {
  return words_;
}

PackedColumnsCodec::PackedColumnsCodec(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref temporaryMemoryResource,
    rmm::device_async_resource_ref outputMemoryResource)
    : state_{std::make_unique<detail::PackedColumnsCodecState>(
          stream,
          temporaryMemoryResource,
          outputMemoryResource)} {}

PackedColumnsCodec::~PackedColumnsCodec() = default;

std::optional<CompressedPackedColumns> PackedColumnsCodec::compress(
    const cudf::packed_columns& input) {
  CUDF_EXPECTS(input.metadata != nullptr && input.gpu_data != nullptr,
               "Cannot compress moved-from packed columns",
               std::invalid_argument);
  if (input.gpu_data->size() == 0) {
    return std::nullopt;
  }
  CUDF_EXPECTS(
      input.gpu_data->size() <=
          static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
      "Packed-column allocation is too large to serialize",
      std::overflow_error);

  const auto* blobBase = static_cast<const uint8_t*>(input.gpu_data->data());
  const auto blobSize = input.gpu_data->size();

  std::vector<TypedRegion> typedRegions;
  const auto table = cudf::unpack(input);
  for (const auto& column : table) {
    collectTypedRegions(column, blobBase, blobSize, typedRegions);
  }
  std::sort(typedRegions.begin(),
            typedRegions.end(),
            [](const auto& left, const auto& right) {
              return left.offset < right.offset;
            });

  std::vector<EncodedRegion> regions;
  std::vector<rmm::device_buffer> payloads;
  std::size_t cursor = 0;

  auto addResidual = [&](std::size_t offset, std::size_t size) {
    if (size == 0) {
      return;
    }

    EncodedRegion region;
    region.rawSize = size;
    auto compressed =
        detail::compressAns({blobBase + offset, size}, state_->ans);
    if (compressed) {
      region.codec = RegionCodec::kByteAns;
      region.segmentSizes = compressed->segmentSizes;
      payloads.push_back(std::move(compressed->data));
    } else {
      payloads.emplace_back();
    }
    regions.push_back(std::move(region));
  };

  for (const auto& typed : typedRegions) {
    const auto elementWidth = cudf::size_of(typed.logicalType);
    const auto regionSize = detail::checkedMultiplySizes(
        typed.elementCount,
        elementWidth,
        "Typed packed-column region size overflow");
    if (typed.offset < cursor) {
      continue;
    }

    addResidual(cursor, typed.offset - cursor);
    EncodedTypedRegion encoded = [&] {
      if (elementWidth == 8) {
        if (usesUnsignedStorage(typed.logicalType)) {
          return encodeTypedRegion<uint64_t>(blobBase, typed, *state_);
        }
        return encodeTypedRegion<int64_t>(blobBase, typed, *state_);
      }
      if (usesUnsignedStorage(typed.logicalType)) {
        return encodeTypedRegion<uint32_t>(blobBase, typed, *state_);
      }
      return encodeTypedRegion<int32_t>(blobBase, typed, *state_);
    }();
    regions.push_back(std::move(encoded.descriptor));
    payloads.push_back(std::move(encoded.data));
    cursor = detail::checkedAddSizes(
        typed.offset, regionSize, "Typed packed-column extent overflow");
  }
  CUDF_EXPECTS(cursor <= blobSize,
               "Typed packed-column regions exceed the allocation");
  addResidual(cursor, blobSize - cursor);

  CUDF_EXPECTS(regions.size() == payloads.size() && !regions.empty(),
               "Packed-column codec produced an invalid region list");

  std::size_t compressedSize = 0;
  for (const auto& region : regions) {
    compressedSize = detail::checkedAddSizes(
        compressedSize,
        detail::nvcompAlignedSize(encodedRegionSize(region)),
        "Packed-column compressed size overflow");
  }
  if (static_cast<long double>(compressedSize) >
      (1.0L - static_cast<long double>(detail::kMinimumEncodedByteReduction)) *
          static_cast<long double>(blobSize)) {
    state_->stream.synchronize();
    return std::nullopt;
  }

  rmm::device_buffer output{
      compressedSize, state_->stream, state_->outputMemoryResource};
  CUDF_CUDA_TRY(
      cudaMemsetAsync(output.data(), 0, output.size(), state_->stream.value()));
  std::size_t outputOffset = 0;
  std::size_t rawOffset = 0;
  for (std::size_t index = 0; index < regions.size(); ++index) {
    const auto& region = regions[index];
    const auto size = encodedRegionSize(region);
    const auto* source = region.codec == RegionCodec::kRaw
        ? blobBase + rawOffset
        : static_cast<const uint8_t*>(payloads[index].data());
    CUDF_CUDA_TRY(
        cudaMemcpyAsync(static_cast<uint8_t*>(output.data()) + outputOffset,
                        source,
                        size,
                        cudaMemcpyDeviceToDevice,
                        state_->stream.value()));
    rawOffset = detail::checkedAddSizes(
        rawOffset, region.rawSize, "Packed-column input offset overflow");
    outputOffset =
        detail::checkedAddSizes(outputOffset,
                                detail::nvcompAlignedSize(size),
                                "Packed-column output offset overflow");
  }
  state_->stream.synchronize();

  auto descriptorWords = serializeDescriptor(regions, blobSize);
  return CompressedPackedColumns{
      std::move(output), PackedColumnsDescriptor{std::move(descriptorWords)}};
}

rmm::device_buffer PackedColumnsCodec::decompress(
    cudf::device_span<const uint8_t> input,
    const PackedColumnsDescriptor& descriptor) {
  CUDF_EXPECTS(input.data() != nullptr,
               "Compressed packed-column input is null",
               std::invalid_argument);
  const auto parsed = parseDescriptor(descriptor.words_);
  CUDF_EXPECTS(parsed.has_value(),
               "Invalid packed-column compression descriptor",
               std::invalid_argument);
  CUDF_EXPECTS(input.size() == parsed->compressedSize,
               "Packed-column descriptor does not match its encoded input",
               std::invalid_argument);

  rmm::device_buffer output{
      parsed->uncompressedSize, state_->stream, state_->outputMemoryResource};
  auto* outputBase = static_cast<uint8_t*>(output.data());
  DeferredAnsStatusChecks pending;
  std::size_t inputOffset = 0;
  std::size_t outputOffset = 0;

  for (const auto& region : parsed->regions) {
    const auto regionEncodedSize = encodedRegionSize(region);
    cudf::device_span<const uint8_t> regionInput{input.data() + inputOffset,
                                                 regionEncodedSize};

    switch (region.codec) {
      case RegionCodec::kRaw:
        CUDF_CUDA_TRY(cudaMemcpyAsync(outputBase + outputOffset,
                                      regionInput.data(),
                                      regionInput.size(),
                                      cudaMemcpyDeviceToDevice,
                                      state_->stream.value()));
        break;
      case RegionCodec::kByteAns: {
        auto decoded = detail::decompressAns(
            regionInput, region.segmentSizes, region.rawSize, state_->ans);
        CUDF_CUDA_TRY(cudaMemcpyAsync(outputBase + outputOffset,
                                      decoded.data(),
                                      region.rawSize,
                                      cudaMemcpyDeviceToDevice,
                                      state_->stream.value()));
        break;
      }
      case RegionCodec::kFrameOfReference:
      case RegionCodec::kDeltaFrameOfReference:
        if (cudf::size_of(region.logicalType) == 8) {
          if (usesUnsignedStorage(region.logicalType)) {
            decodeTypedRegion<uint64_t>(regionInput,
                                        region,
                                        outputBase + outputOffset,
                                        pending,
                                        *state_);
          } else {
            decodeTypedRegion<int64_t>(regionInput,
                                       region,
                                       outputBase + outputOffset,
                                       pending,
                                       *state_);
          }
        } else if (usesUnsignedStorage(region.logicalType)) {
          decodeTypedRegion<uint32_t>(
              regionInput, region, outputBase + outputOffset, pending, *state_);
        } else {
          decodeTypedRegion<int32_t>(
              regionInput, region, outputBase + outputOffset, pending, *state_);
        }
        break;
    }
    outputOffset = detail::checkedAddSizes(
        outputOffset, region.rawSize, "Packed-column output offset overflow");
    inputOffset =
        detail::checkedAddSizes(inputOffset,
                                detail::nvcompAlignedSize(regionEncodedSize),
                                "Packed-column input offset overflow");
  }

  state_->stream.synchronize();
  pending.verify();
  return output;
}

} // namespace facebook::velox::cudf_velox::compression
