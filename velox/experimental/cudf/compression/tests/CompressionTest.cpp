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

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::compression {
namespace {

constexpr std::size_t kDescriptorVersionIndex = 1;
constexpr std::size_t kUncompressedSizeIndex = 2;
constexpr std::size_t kRegionCountIndex = 3;
constexpr std::size_t kFirstRegionIndex = 4;
constexpr std::size_t kRegionRawSizeOffset = 0;
constexpr std::size_t kRegionTransformOffset = 1;
constexpr std::size_t kRegionEncodingOffset = 2;
constexpr std::size_t kRegionTypeOffset = 3;
constexpr std::size_t kRegionScaleOffset = 4;
constexpr std::size_t kRegionReferenceOffset = 5;
constexpr std::size_t kRegionPlaneCountOffset = 6;
constexpr std::size_t kRegionSegmentCountOffset = 7;
constexpr std::size_t kRegionFixedWordCount = 8;
constexpr int64_t kNoTransform = 0;
constexpr int64_t kFrameOfReferenceTransform = 1;
constexpr int64_t kDeltaFrameOfReferenceTransform = 2;
constexpr int64_t kNoEntropyEncoding = 0;
constexpr int64_t kAnsEncoding = 1;

template <typename T>
std::unique_ptr<cudf::column> makeColumn(
    cudf::data_type type,
    const std::vector<T>& values,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource,
    bool nullable = false) {
  rmm::device_buffer data{
      values.data(), values.size() * sizeof(T), stream, memoryResource};
  rmm::device_buffer nullMask;
  cudf::size_type nullCount = 0;
  if (nullable) {
    nullMask =
        cudf::create_null_mask(static_cast<cudf::size_type>(values.size()),
                               cudf::mask_state::ALL_VALID,
                               stream,
                               memoryResource);
    const auto firstNull = static_cast<cudf::size_type>(values.size() / 3);
    const auto lastNull = static_cast<cudf::size_type>(values.size() / 2);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(nullMask.data()),
                        firstNull,
                        lastNull,
                        false,
                        stream);
    nullCount = lastNull - firstNull;
  }
  return std::make_unique<cudf::column>(
      type,
      static_cast<cudf::size_type>(values.size()),
      std::move(data),
      std::move(nullMask),
      nullCount);
}

struct RoundTripObservation {
  std::size_t uncompressedSize;
  std::size_t compressedSize;
  std::vector<int64_t> serializedDescriptor;
};

RoundTripObservation roundTrip(
    std::vector<std::unique_ptr<cudf::column>> columns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource,
    CompressionOptions options = {}) {
  cudf::table table{std::move(columns)};
  auto packed = cudf::pack(table.view(), stream, memoryResource);
  stream.synchronize();

  std::vector<uint8_t> expected(packed.gpu_data->size());
  EXPECT_EQ(cudaMemcpyAsync(expected.data(),
                            packed.gpu_data->data(),
                            expected.size(),
                            cudaMemcpyDeviceToHost,
                            stream.value()),
            cudaSuccess);
  stream.synchronize();

  PackedColumnsCodec codec{stream, memoryResource, memoryResource};
  auto compressed = codec.compress(packed, options);
  EXPECT_TRUE(compressed.has_value());
  if (!compressed) {
    return {packed.gpu_data->size(), 0, {}};
  }

  auto words = compressed->descriptor.serialize();
  auto descriptor = PackedColumnsDescriptor::deserialize(words);
  EXPECT_TRUE(descriptor.has_value());
  if (!descriptor) {
    return {packed.gpu_data->size(), compressed->data.size(), std::move(words)};
  }
  EXPECT_EQ(descriptor->serialize(), words);

  auto decoded =
      codec.decompress({static_cast<const uint8_t*>(compressed->data.data()),
                        compressed->data.size()},
                       *descriptor);
  std::vector<uint8_t> actual(decoded.size());
  EXPECT_EQ(cudaMemcpyAsync(actual.data(),
                            decoded.data(),
                            actual.size(),
                            cudaMemcpyDeviceToHost,
                            stream.value()),
            cudaSuccess);
  stream.synchronize();
  EXPECT_EQ(actual, expected);

  return {packed.gpu_data->size(), compressed->data.size(), std::move(words)};
}

std::vector<int64_t> lowCardinalityInt64(std::size_t size) {
  std::vector<int64_t> values(size);
  for (std::size_t index = 0; index < size; ++index) {
    values[index] = static_cast<int64_t>((index * 17) % 251) - 125;
  }
  return values;
}

std::vector<uint8_t> copyToHost(const rmm::device_buffer& input,
                                rmm::cuda_stream_view stream) {
  std::vector<uint8_t> output(input.size());
  EXPECT_EQ(cudaMemcpyAsync(output.data(),
                            input.data(),
                            input.size(),
                            cudaMemcpyDeviceToHost,
                            stream.value()),
            cudaSuccess);
  stream.synchronize();
  return output;
}

TEST(SizeUtilsTest, RejectsOverflowAndAlignsSafely) {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  std::size_t result = 0;

  EXPECT_TRUE(detail::tryAddSizes(7, 9, result));
  EXPECT_EQ(result, 16);
  EXPECT_FALSE(detail::tryAddSizes(maximum, 1, result));
  EXPECT_THROW(detail::checkedAddSizes(maximum, 1, "test overflow"),
               std::overflow_error);
  EXPECT_THROW(detail::checkedMultiplySizes(maximum, 2, "test overflow"),
               std::overflow_error);

  EXPECT_TRUE(detail::tryNvcompAlignedSize(17, result));
  EXPECT_EQ(result, 32);
  EXPECT_FALSE(detail::tryNvcompAlignedSize(maximum, result));
  EXPECT_THROW(detail::nvcompAlignedSize(maximum), std::overflow_error);
}

TEST(PackedColumnsCodecTest, RoundTripsLogicalTypesAndNullMask) {
  constexpr std::size_t kRows = 1u << 18;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  std::vector<int32_t> signed32(kRows);
  std::vector<uint32_t> unsigned32(kRows);
  std::vector<uint64_t> unsigned64(kRows);
  std::vector<float> floating32(kRows);
  std::vector<double> floating64(kRows);
  std::vector<int32_t> decimal32(kRows);
  std::vector<int64_t> decimal64(kRows);
  std::vector<int64_t> timestampMillis(kRows);
  for (std::size_t index = 0; index < kRows; ++index) {
    signed32[index] = static_cast<int32_t>(index % 37) - 18;
    unsigned32[index] = std::numeric_limits<uint32_t>::max() - (index % 67);
    unsigned64[index] = std::numeric_limits<uint64_t>::max() - (index % 113);
    floating32[index] = 100.0F + static_cast<float>(index % 7) / 2.0F;
    floating64[index] = 1000.0 + static_cast<double>(index % 11) / 4.0;
    decimal32[index] = 90'000 + static_cast<int32_t>(index % 73);
    decimal64[index] = 9'000'000 + static_cast<int64_t>(index % 101);
    timestampMillis[index] =
        1'700'000'000'000LL + static_cast<int64_t>(index) * 1000;
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT32},
                               signed32,
                               stream.view(),
                               memoryResource,
                               true));
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::UINT32},
                               unsigned32,
                               stream.view(),
                               memoryResource));
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::UINT64},
                               unsigned64,
                               stream.view(),
                               memoryResource));
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::FLOAT32},
                               floating32,
                               stream.view(),
                               memoryResource));
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::FLOAT64},
                               floating64,
                               stream.view(),
                               memoryResource));
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::DECIMAL32, -2},
                               decimal32,
                               stream.view(),
                               memoryResource));
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::DECIMAL64, -2},
                               decimal64,
                               stream.view(),
                               memoryResource));
  columns.push_back(
      makeColumn(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
                 timestampMillis,
                 stream.view(),
                 memoryResource));

  const auto observation =
      roundTrip(std::move(columns), stream.view(), memoryResource);
  EXPECT_LT(observation.compressedSize, observation.uncompressedSize);
}

TEST(PackedColumnsCodecTest, FrameOfReferenceWithoutAnsSupportsDirectLookup) {
  constexpr std::size_t kRows = 1u << 18;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  std::vector<int64_t> values(kRows);
  for (std::size_t index = 0; index < kRows; ++index) {
    values[index] = 5'000'000'000LL +
        static_cast<int64_t>((index * 7'919) & ((1u << 20) - 1));
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               values,
                               stream.view(),
                               memoryResource));
  cudf::table table{std::move(columns)};
  auto packed = cudf::pack(table.view(), stream.view(), memoryResource);
  stream.synchronize();
  const auto expected = copyToHost(*packed.gpu_data, stream.view());

  CompressionOptions options;
  options.numericTransform = NumericTransform::kFrameOfReference;
  options.entropyEncoding = EntropyEncoding::kNone;
  PackedColumnsCodec codec{stream.view(), memoryResource, memoryResource};
  auto compressed = codec.compress(packed, options);
  ASSERT_TRUE(compressed);
  const auto words = compressed->descriptor.serialize();
  const auto bytes = copyToHost(compressed->data, stream.view());

  bool foundTypedRegion = false;
  std::size_t descriptorPosition = kFirstRegionIndex;
  std::size_t encodedPosition = 0;
  const auto regionCount = static_cast<std::size_t>(words[kRegionCountIndex]);
  for (std::size_t region = 0; region < regionCount; ++region) {
    ASSERT_LE(descriptorPosition + kRegionFixedWordCount, words.size());
    const auto rawSize = static_cast<std::size_t>(
        words[descriptorPosition + kRegionRawSizeOffset]);
    const auto transform = words[descriptorPosition + kRegionTransformOffset];
    const auto encoding = words[descriptorPosition + kRegionEncodingOffset];
    const auto planeCount = static_cast<std::size_t>(
        words[descriptorPosition + kRegionPlaneCountOffset]);
    const auto segmentCount = static_cast<std::size_t>(
        words[descriptorPosition + kRegionSegmentCountOffset]);
    EXPECT_EQ(encoding, kNoEntropyEncoding);
    EXPECT_EQ(segmentCount, 0);

    std::size_t encodedSize = rawSize;
    if (transform != kNoTransform) {
      const auto typeId = static_cast<cudf::type_id>(
          words[descriptorPosition + kRegionTypeOffset]);
      const auto elementWidth = cudf::size_of(cudf::data_type{typeId});
      const auto elementCount = rawSize / elementWidth;
      encodedSize = elementCount * planeCount;

      ASSERT_EQ(transform, kFrameOfReferenceTransform);
      ASSERT_EQ(typeId, cudf::type_id::INT64);
      ASSERT_EQ(elementCount, kRows);
      ASSERT_EQ(planeCount, 3);
      const auto referenceBits = std::bit_cast<uint64_t>(
          words[descriptorPosition + kRegionReferenceOffset]);
      for (const auto row : {std::size_t{0}, kRows / 3, kRows - 1}) {
        uint64_t adjusted = 0;
        for (std::size_t plane = 0; plane < planeCount; ++plane) {
          adjusted |= static_cast<uint64_t>(
                          bytes[encodedPosition + plane * elementCount + row])
              << (8 * plane);
        }
        EXPECT_EQ(std::bit_cast<int64_t>(referenceBits + adjusted),
                  values[row]);
      }
      foundTypedRegion = true;
    }

    encodedPosition += detail::nvcompAlignedSize(encodedSize);
    descriptorPosition += kRegionFixedWordCount + segmentCount;
  }
  EXPECT_TRUE(foundTypedRegion);
  EXPECT_EQ(descriptorPosition, words.size());
  EXPECT_EQ(encodedPosition, bytes.size());

  auto decoded =
      codec.decompress({static_cast<const uint8_t*>(compressed->data.data()),
                        compressed->data.size()},
                       compressed->descriptor);
  EXPECT_EQ(copyToHost(decoded, stream.view()), expected);
}

TEST(PackedColumnsCodecTest, DeltaFrameOfReferenceCanSkipAns) {
  constexpr std::size_t kRows = 1u << 18;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  std::vector<int64_t> values(kRows);
  for (std::size_t index = 0; index < kRows; ++index) {
    values[index] = 1'700'000'000'000LL + static_cast<int64_t>(index) * 1'000;
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               values,
                               stream.view(),
                               memoryResource));

  CompressionOptions options;
  options.numericTransform = NumericTransform::kDeltaFrameOfReference;
  options.entropyEncoding = EntropyEncoding::kNone;
  const auto observation =
      roundTrip(std::move(columns), stream.view(), memoryResource, options);
  ASSERT_FALSE(observation.serializedDescriptor.empty());

  bool foundTypedRegion = false;
  const auto& words = observation.serializedDescriptor;
  std::size_t position = kFirstRegionIndex;
  const auto regionCount = static_cast<std::size_t>(words[kRegionCountIndex]);
  for (std::size_t region = 0; region < regionCount; ++region) {
    ASSERT_LE(position + kRegionFixedWordCount, words.size());
    EXPECT_EQ(words[position + kRegionEncodingOffset], kNoEntropyEncoding);
    const auto segmentCount =
        static_cast<std::size_t>(words[position + kRegionSegmentCountOffset]);
    EXPECT_EQ(segmentCount, 0);
    if (words[position + kRegionTransformOffset] != kNoTransform) {
      EXPECT_EQ(words[position + kRegionTransformOffset],
                kDeltaFrameOfReferenceTransform);
      foundTypedRegion = true;
    }
    position += kRegionFixedWordCount + segmentCount;
  }
  EXPECT_TRUE(foundTypedRegion);
  EXPECT_EQ(position, words.size());
}

TEST(PackedColumnsCodecTest, RoundTripsSignedAndUnsignedExtremes) {
  constexpr std::size_t kRows = 1u << 18;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  const std::vector<int64_t> signedPattern{
      std::numeric_limits<int64_t>::min(),
      std::numeric_limits<int64_t>::max(),
      0,
      -1,
      1,
      std::numeric_limits<int64_t>::min() + 1,
      std::numeric_limits<int64_t>::max() - 1};
  const std::vector<uint64_t> unsignedPattern{
      0,
      std::numeric_limits<uint64_t>::max(),
      1,
      std::numeric_limits<uint64_t>::max() - 1,
      uint64_t{1} << 63};

  std::vector<int64_t> signedValues(kRows);
  std::vector<uint64_t> unsignedValues(kRows);
  for (std::size_t index = 0; index < kRows; ++index) {
    signedValues[index] = signedPattern[index % signedPattern.size()];
    unsignedValues[index] = unsignedPattern[index % unsignedPattern.size()];
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               signedValues,
                               stream.view(),
                               memoryResource));
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::UINT64},
                               unsignedValues,
                               stream.view(),
                               memoryResource));
  roundTrip(std::move(columns), stream.view(), memoryResource);
}

TEST(PackedColumnsCodecTest, RoundTripsStringsAndNestedBuffers) {
  constexpr std::size_t kRows = 1u << 15;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  const std::string value = "packed-column-value";
  std::vector<int32_t> offsets(kRows + 1);
  std::vector<char> characters;
  characters.reserve(kRows * value.size());
  for (std::size_t index = 0; index < kRows; ++index) {
    offsets[index] = static_cast<int32_t>(characters.size());
    characters.insert(characters.end(), value.begin(), value.end());
  }
  offsets.back() = static_cast<int32_t>(characters.size());

  auto offsetsColumn = makeColumn(cudf::data_type{cudf::type_id::INT32},
                                  offsets,
                                  stream.view(),
                                  memoryResource);
  rmm::device_buffer characterBuffer{
      characters.data(), characters.size(), stream.view(), memoryResource};
  auto strings = cudf::make_strings_column(static_cast<cudf::size_type>(kRows),
                                           std::move(offsetsColumn),
                                           std::move(characterBuffer),
                                           0,
                                           rmm::device_buffer{});

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(strings));
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               lowCardinalityInt64(kRows),
                               stream.view(),
                               memoryResource));
  roundTrip(std::move(columns), stream.view(), memoryResource);
}

TEST(PackedColumnsCodecTest, DescriptorRejectsMalformedInput) {
  constexpr std::size_t kRows = 1u << 18;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               lowCardinalityInt64(kRows),
                               stream.view(),
                               memoryResource,
                               true));
  const auto observation =
      roundTrip(std::move(columns), stream.view(), memoryResource);
  ASSERT_FALSE(observation.serializedDescriptor.empty());
  const auto& valid = observation.serializedDescriptor;

  for (std::size_t size = 0; size < valid.size(); ++size) {
    EXPECT_FALSE(PackedColumnsDescriptor::deserialize(
        std::span<const int64_t>{valid.data(), size}))
        << "truncated descriptor size " << size;
  }

  auto unknownVersion = valid;
  unknownVersion[kDescriptorVersionIndex] += 1;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(unknownVersion));

  auto negativeUncompressedSize = valid;
  negativeUncompressedSize[kUncompressedSizeIndex] = -1;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(negativeUncompressedSize));

  auto excessiveRegionCount = valid;
  excessiveRegionCount[kRegionCountIndex] = std::numeric_limits<int64_t>::max();
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(excessiveRegionCount));

  auto trailingWord = valid;
  trailingWord.push_back(0);
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(trailingWord));

  std::size_t typedRegionIndex = valid.size();
  std::size_t untypedRegionIndex = valid.size();
  std::size_t position = kFirstRegionIndex;
  const auto regionCount = static_cast<std::size_t>(valid[kRegionCountIndex]);
  for (std::size_t region = 0; region < regionCount; ++region) {
    ASSERT_LE(position + kRegionFixedWordCount, valid.size());
    const auto transform = valid[position + kRegionTransformOffset];
    if (transform == kNoTransform && untypedRegionIndex == valid.size()) {
      untypedRegionIndex = position;
    } else if (transform != kNoTransform && typedRegionIndex == valid.size()) {
      typedRegionIndex = position;
    }
    const auto segmentCount =
        static_cast<std::size_t>(valid[position + kRegionSegmentCountOffset]);
    position += kRegionFixedWordCount + segmentCount;
  }
  ASSERT_EQ(position, valid.size());
  ASSERT_LT(typedRegionIndex, valid.size());
  ASSERT_LT(untypedRegionIndex, valid.size());
  ASSERT_GT(valid[typedRegionIndex + kRegionSegmentCountOffset], 0);

  auto emptySegment = valid;
  emptySegment[typedRegionIndex + kRegionFixedWordCount] = 0;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(emptySegment));

  auto invalidTransform = valid;
  invalidTransform[kFirstRegionIndex + kRegionTransformOffset] = 99;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(invalidTransform));

  auto invalidEncoding = valid;
  invalidEncoding[kFirstRegionIndex + kRegionEncodingOffset] = 99;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(invalidEncoding));

  auto invalidUntypedReference = valid;
  invalidUntypedReference[untypedRegionIndex + kRegionReferenceOffset] = 1;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(invalidUntypedReference));

  auto zeroPlaneCount = valid;
  zeroPlaneCount[typedRegionIndex + kRegionPlaneCountOffset] = 0;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(zeroPlaneCount));

  auto excessivePlaneCount = valid;
  excessivePlaneCount[typedRegionIndex + kRegionPlaneCountOffset] = 9;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(excessivePlaneCount));

  auto noEntropyWithSegments = valid;
  noEntropyWithSegments[typedRegionIndex + kRegionEncodingOffset] =
      kNoEntropyEncoding;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(noEntropyWithSegments));

  auto invalidType = valid;
  invalidType[typedRegionIndex + kRegionTypeOffset] =
      static_cast<int64_t>(cudf::type_id::NUM_TYPE_IDS);
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(invalidType));

  auto invalidScale = valid;
  invalidScale[typedRegionIndex + kRegionScaleOffset] =
      std::numeric_limits<int64_t>::max();
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(invalidScale));

  auto negativeSegmentCount = valid;
  negativeSegmentCount[typedRegionIndex + kRegionSegmentCountOffset] = -1;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(negativeSegmentCount));

  auto oversizedRawRegion = valid;
  oversizedRawRegion[kFirstRegionIndex + kRegionRawSizeOffset] =
      std::numeric_limits<int64_t>::max();
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(oversizedRawRegion));

  auto oversizedSegment = valid;
  oversizedSegment[typedRegionIndex + kRegionFixedWordCount] =
      std::numeric_limits<int64_t>::max();
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(oversizedSegment));
}

TEST(PackedColumnsCodecTest, DecompressRejectsWrongInputExtent) {
  constexpr std::size_t kRows = 1u << 18;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               lowCardinalityInt64(kRows),
                               stream.view(),
                               memoryResource));
  cudf::table table{std::move(columns)};
  auto packed = cudf::pack(table.view(), stream.view(), memoryResource);
  PackedColumnsCodec codec{stream.view(), memoryResource, memoryResource};
  auto compressed = codec.compress(packed);
  ASSERT_TRUE(compressed);
  ASSERT_GT(compressed->data.size(), 1);
  EXPECT_THROW(codec.decompress({static_cast<const uint8_t*>(nullptr),
                                 compressed->data.size()},
                                compressed->descriptor),
               std::invalid_argument);

  EXPECT_THROW(
      codec.decompress({static_cast<const uint8_t*>(compressed->data.data()),
                        compressed->data.size() - 1},
                       compressed->descriptor),
      std::invalid_argument);
  EXPECT_THROW(
      codec.decompress({static_cast<const uint8_t*>(compressed->data.data()),
                        compressed->data.size() + 1},
                       compressed->descriptor),
      std::invalid_argument);
}

TEST(PackedColumnsCodecTest, RejectsInsufficientReduction) {
  constexpr std::size_t kRows = 1u << 18;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();
  std::mt19937_64 generator{42};
  std::vector<int64_t> values(kRows);
  std::generate(values.begin(), values.end(), [&] {
    return static_cast<int64_t>(generator());
  });

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               values,
                               stream.view(),
                               memoryResource));
  cudf::table table{std::move(columns)};
  auto packed = cudf::pack(table.view(), stream.view(), memoryResource);
  PackedColumnsCodec codec{stream.view(), memoryResource, memoryResource};

  EXPECT_FALSE(codec.compress(packed));
}

TEST(PackedColumnsCodecTest, SmallInputIsNotExpanded) {
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();
  const std::vector<int64_t> values{1, 2, 3, 4};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               values,
                               stream.view(),
                               memoryResource));
  cudf::table table{std::move(columns)};
  auto packed = cudf::pack(table.view(), stream.view(), memoryResource);
  PackedColumnsCodec codec{stream.view(), memoryResource, memoryResource};
  EXPECT_FALSE(codec.compress(packed));
}

TEST(PackedColumnsCodecTest, HonorsTypedTransformThreshold) {
  constexpr std::size_t kTypedThreshold = 4096;
  constexpr int kColumnCount = 3;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  auto selectedTransforms = [&](std::size_t rowCount) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    // Keep the residual input large enough to produce a descriptor below the
    // typed-transform threshold, where each column remains untransformed.
    for (int column = 0; column < kColumnCount; ++column) {
      columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                                   std::vector<int64_t>(rowCount, 7),
                                   stream.view(),
                                   memoryResource));
    }
    cudf::table table{std::move(columns)};
    auto packed = cudf::pack(table.view(), stream.view(), memoryResource);
    PackedColumnsCodec codec{stream.view(), memoryResource, memoryResource};
    auto compressed = codec.compress(packed);
    EXPECT_TRUE(compressed);
    if (!compressed) {
      return std::vector<int64_t>{};
    }

    const auto words = compressed->descriptor.serialize();
    std::vector<int64_t> transforms;
    std::size_t position = kFirstRegionIndex;
    const auto regionCount = static_cast<std::size_t>(words[kRegionCountIndex]);
    for (std::size_t region = 0; region < regionCount; ++region) {
      EXPECT_LE(position + kRegionFixedWordCount, words.size());
      transforms.push_back(words[position + kRegionTransformOffset]);
      const auto segmentCount =
          static_cast<std::size_t>(words[position + kRegionSegmentCountOffset]);
      position += kRegionFixedWordCount + segmentCount;
    }
    EXPECT_EQ(position, words.size());
    return transforms;
  };

  const auto belowThreshold = selectedTransforms(kTypedThreshold - 1);
  EXPECT_TRUE(std::none_of(
      belowThreshold.begin(), belowThreshold.end(), [](int64_t transform) {
        return transform != kNoTransform;
      }));

  const auto atThreshold = selectedTransforms(kTypedThreshold);
  EXPECT_TRUE(std::any_of(
      atThreshold.begin(), atThreshold.end(), [](int64_t transform) {
        return transform != kNoTransform;
      }));
}

TEST(PackedColumnsCodecTest, HonorsResidualAnsThreshold) {
  constexpr std::size_t kResidualThreshold = detail::kNvcompAnsChunkSize;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();
  rmm::device_buffer input{kResidualThreshold, stream.view(), memoryResource};
  ASSERT_EQ(cudaMemsetAsync(input.data(), 0, input.size(), stream.value()),
            cudaSuccess);

  detail::AnsCodecContext context{stream.view(), memoryResource};
  EXPECT_FALSE(detail::compressAns(
      {static_cast<const uint8_t*>(input.data()), kResidualThreshold - 1},
      context));

  auto compressed = detail::compressAns(
      {static_cast<const uint8_t*>(input.data()), kResidualThreshold}, context);
  ASSERT_TRUE(compressed);
  auto decoded = detail::decompressAns(
      {static_cast<const uint8_t*>(compressed->data.data()),
       compressed->data.size()},
      compressed->segmentSizes,
      kResidualThreshold,
      context);
  const auto decodedBytes = copyToHost(decoded, stream.view());
  EXPECT_TRUE(std::all_of(decodedBytes.begin(),
                          decodedBytes.end(),
                          [](uint8_t value) { return value == 0; }));
}

TEST(PackedColumnsCodecTest, EncodedPaddingIsDeterministic) {
  constexpr std::size_t kRows = (1u << 18) + 3;
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               lowCardinalityInt64(kRows),
                               stream.view(),
                               memoryResource));
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::UINT8},
                               std::vector<uint8_t>(kRows, 3),
                               stream.view(),
                               memoryResource));
  cudf::table table{std::move(columns)};
  auto packed = cudf::pack(table.view(), stream.view(), memoryResource);
  PackedColumnsCodec codec{stream.view(), memoryResource, memoryResource};

  auto first = codec.compress(packed);
  auto second = codec.compress(packed);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  const auto descriptor = first->descriptor.serialize();
  EXPECT_EQ(descriptor, second->descriptor.serialize());
  const auto firstBytes = copyToHost(first->data, stream.view());
  const auto secondBytes = copyToHost(second->data, stream.view());

  auto expectZeroPadding = [&](const std::vector<uint8_t>& bytes) {
    std::size_t descriptorPosition = kFirstRegionIndex;
    std::size_t encodedPosition = 0;
    const auto regionCount =
        static_cast<std::size_t>(descriptor[kRegionCountIndex]);
    for (std::size_t region = 0; region < regionCount; ++region) {
      const auto rawSize = static_cast<std::size_t>(
          descriptor[descriptorPosition + kRegionRawSizeOffset]);
      const auto transform =
          descriptor[descriptorPosition + kRegionTransformOffset];
      const auto encoding =
          descriptor[descriptorPosition + kRegionEncodingOffset];
      const auto segmentCount = static_cast<std::size_t>(
          descriptor[descriptorPosition + kRegionSegmentCountOffset]);
      const auto segmentSizes = descriptorPosition + kRegionFixedWordCount;
      if (encoding == kNoEntropyEncoding) {
        std::size_t encodedSize = rawSize;
        if (transform != kNoTransform) {
          const auto type = cudf::data_type{
              static_cast<cudf::type_id>(
                  descriptor[descriptorPosition + kRegionTypeOffset]),
              static_cast<int32_t>(
                  descriptor[descriptorPosition + kRegionScaleOffset])};
          const auto planeCount = static_cast<std::size_t>(
              descriptor[descriptorPosition + kRegionPlaneCountOffset]);
          encodedSize = rawSize / cudf::size_of(type) * planeCount;
        }
        for (auto offset = encodedSize;
             offset < detail::nvcompAlignedSize(encodedSize);
             ++offset) {
          EXPECT_EQ(bytes[encodedPosition + offset], 0);
        }
        encodedPosition += detail::nvcompAlignedSize(encodedSize);
      } else {
        EXPECT_EQ(encoding, kAnsEncoding);
        for (std::size_t segment = 0; segment < segmentCount; ++segment) {
          const auto size =
              static_cast<std::size_t>(descriptor[segmentSizes + segment]);
          for (auto offset = size; offset < detail::nvcompAlignedSize(size);
               ++offset) {
            EXPECT_EQ(bytes[encodedPosition + offset], 0);
          }
          encodedPosition += detail::nvcompAlignedSize(size);
        }
      }
      descriptorPosition = segmentSizes + segmentCount;
    }
    EXPECT_EQ(descriptorPosition, descriptor.size());
    EXPECT_EQ(encodedPosition, bytes.size());
  };
  expectZeroPadding(firstBytes);
  expectZeroPadding(secondBytes);
}

TEST(PackedColumnsCodecTest, SupportsIndependentStreams) {
  constexpr std::size_t kRows = 1u << 16;
  rmm::cuda_stream firstStream;
  rmm::cuda_stream secondStream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  std::vector<std::unique_ptr<cudf::column>> firstColumns;
  firstColumns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                                    lowCardinalityInt64(kRows),
                                    firstStream.view(),
                                    memoryResource));
  std::vector<std::unique_ptr<cudf::column>> secondColumns;
  secondColumns.push_back(makeColumn(
      cudf::data_type{cudf::type_id::UINT64},
      std::vector<uint64_t>(kRows, std::numeric_limits<uint64_t>::max()),
      secondStream.view(),
      memoryResource));

  roundTrip(std::move(firstColumns), firstStream.view(), memoryResource);
  roundTrip(std::move(secondColumns), secondStream.view(), memoryResource);
}

TEST(PackedColumnsCodecTest, SupportsConcurrentCodecInstances) {
  constexpr std::size_t kRows = 1u << 16;
  auto work = [](int64_t offset) {
    if (cudaSetDevice(0) != cudaSuccess) {
      throw std::runtime_error{"Failed to select CUDA device"};
    }
    rmm::cuda_stream stream;
    const auto memoryResource = rmm::mr::get_current_device_resource_ref();
    auto values = lowCardinalityInt64(kRows);
    for (auto& value : values) {
      value += offset;
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                                 values,
                                 stream.view(),
                                 memoryResource));
    const auto result =
        roundTrip(std::move(columns), stream.view(), memoryResource);
    return result.compressedSize < result.uncompressedSize;
  };

  auto first = std::async(std::launch::async, work, 0);
  auto second = std::async(std::launch::async, work, 1000);
  EXPECT_TRUE(first.get());
  EXPECT_TRUE(second.get());
}

TEST(PackedColumnsCodecTest, SupportsIndependentDevices) {
  int originalDevice = 0;
  int deviceCount = 0;
  ASSERT_EQ(cudaGetDevice(&originalDevice), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "Two visible CUDA devices are required";
  }
  struct RestoreDevice {
    int device;
    ~RestoreDevice() {
      cudaSetDevice(device);
    }
  } restore{originalDevice};

  for (int device = 0; device < 2; ++device) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    rmm::cuda_stream stream;
    const auto memoryResource = rmm::mr::get_current_device_resource_ref();
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT32},
                                 std::vector<int32_t>(1u << 16, device + 1),
                                 stream.view(),
                                 memoryResource));
    const auto result =
        roundTrip(std::move(columns), stream.view(), memoryResource);
    EXPECT_LT(result.compressedSize, result.uncompressedSize);
  }
}

TEST(PackedColumnsCodecTest, RejectsMovedFromInputAndEmptyAllocation) {
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();
  PackedColumnsCodec codec{stream.view(), memoryResource, memoryResource};

  cudf::table emptyTable{std::vector<std::unique_ptr<cudf::column>>{}};
  auto empty = cudf::pack(emptyTable.view(), stream.view(), memoryResource);
  EXPECT_FALSE(codec.compress(empty));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                               lowCardinalityInt64(1u << 16),
                               stream.view(),
                               memoryResource));
  cudf::table table{std::move(columns)};
  auto packed = cudf::pack(table.view(), stream.view(), memoryResource);
  auto owner = std::move(packed);
  EXPECT_THROW(codec.compress(packed), std::invalid_argument);
  EXPECT_TRUE(codec.compress(owner));
}
} // namespace
} // namespace facebook::velox::cudf_velox::compression
