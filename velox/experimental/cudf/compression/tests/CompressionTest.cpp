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
constexpr std::size_t kCompressedSizeIndex = 3;
constexpr std::size_t kRegionCountIndex = 4;
constexpr std::size_t kFirstRegionIndex = 5;
constexpr std::size_t kRegionBaseOffset = 5;
constexpr std::size_t kRegionFirstOffset = 6;
constexpr std::size_t kRegionRawSizeOffset = 1;
constexpr std::size_t kRegionCodecOffset = 2;
constexpr std::size_t kRegionTypeOffset = 3;
constexpr std::size_t kRegionScaleOffset = 4;
constexpr std::size_t kRegionSegmentCountOffset = 7;
constexpr std::size_t kRegionFixedWordCount = 8;
constexpr int64_t kRawRegionCodec = 0;
constexpr int64_t kFrameOfReferenceRegionCodec = 2;
constexpr int64_t kDeltaFrameOfReferenceRegionCodec = 3;

std::size_t alignedEncodedSize(std::size_t size) {
  return detail::nvcompAlignedSize(size);
}

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
    rmm::device_async_resource_ref memoryResource) {
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
  auto compressed = codec.compress(packed);
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
                               memoryResource));
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

  auto wrongCompressedSize = valid;
  wrongCompressedSize[kCompressedSizeIndex] += 1;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(wrongCompressedSize));

  auto excessiveRegionCount = valid;
  excessiveRegionCount[kRegionCountIndex] = std::numeric_limits<int64_t>::max();
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(excessiveRegionCount));

  auto trailingWord = valid;
  trailingWord.push_back(0);
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(trailingWord));

  auto emptySegment = valid;
  emptySegment.back() = 0;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(emptySegment));

  std::size_t typedRegionIndex = kFirstRegionIndex;
  const auto regionCount = static_cast<std::size_t>(valid[kRegionCountIndex]);
  for (std::size_t region = 0; region < regionCount; ++region) {
    if (valid[typedRegionIndex + kRegionCodecOffset] >=
        kFrameOfReferenceRegionCodec) {
      break;
    }
    const auto segmentCount = static_cast<std::size_t>(
        valid[typedRegionIndex + kRegionSegmentCountOffset]);
    typedRegionIndex += kRegionFixedWordCount + segmentCount;
  }
  ASSERT_LT(typedRegionIndex, valid.size());

  auto invalidCodec = valid;
  invalidCodec[kFirstRegionIndex + kRegionCodecOffset] = 99;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(invalidCodec));

  auto nonCanonicalFrameOfReference = valid;
  nonCanonicalFrameOfReference[typedRegionIndex + kRegionCodecOffset] =
      kFrameOfReferenceRegionCodec;
  nonCanonicalFrameOfReference[typedRegionIndex + kRegionFirstOffset] = 1;
  EXPECT_FALSE(
      PackedColumnsDescriptor::deserialize(nonCanonicalFrameOfReference));

  auto nonCanonicalDeltaFrameOfReference = valid;
  nonCanonicalDeltaFrameOfReference[typedRegionIndex + kRegionCodecOffset] =
      kDeltaFrameOfReferenceRegionCodec;
  nonCanonicalDeltaFrameOfReference[typedRegionIndex + kRegionBaseOffset] = 1;
  EXPECT_FALSE(
      PackedColumnsDescriptor::deserialize(nonCanonicalDeltaFrameOfReference));

  auto invalidType = valid;
  invalidType[kFirstRegionIndex + kRegionTypeOffset] =
      static_cast<int64_t>(cudf::type_id::NUM_TYPE_IDS);
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(invalidType));

  auto invalidScale = valid;
  invalidScale[kFirstRegionIndex + kRegionScaleOffset] =
      std::numeric_limits<int64_t>::max();
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(invalidScale));

  auto negativeSegmentCount = valid;
  negativeSegmentCount[kFirstRegionIndex + kRegionSegmentCountOffset] = -1;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(negativeSegmentCount));

  auto nonContiguousRegion = valid;
  nonContiguousRegion[kFirstRegionIndex] = 1;
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(nonContiguousRegion));

  auto oversizedRawRegion = valid;
  oversizedRawRegion[kFirstRegionIndex + kRegionRawSizeOffset] =
      std::numeric_limits<int64_t>::max();
  EXPECT_FALSE(PackedColumnsDescriptor::deserialize(oversizedRawRegion));

  auto oversizedSegment = valid;
  oversizedSegment.back() = std::numeric_limits<int64_t>::max();
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
  rmm::cuda_stream stream;
  const auto memoryResource = rmm::mr::get_current_device_resource_ref();

  auto selectedCodecs = [&](std::size_t rowCount) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeColumn(cudf::data_type{cudf::type_id::INT64},
                                 std::vector<int64_t>(rowCount, 7),
                                 stream.view(),
                                 memoryResource));
    cudf::table table{std::move(columns)};
    auto packed = cudf::pack(table.view(), stream.view(), memoryResource);
    PackedColumnsCodec codec{stream.view(), memoryResource, memoryResource};
    auto compressed = codec.compress(packed);
    EXPECT_TRUE(compressed);
    if (!compressed) {
      return std::vector<int64_t>{};
    }

    const auto words = compressed->descriptor.serialize();
    std::vector<int64_t> codecs;
    std::size_t position = kFirstRegionIndex;
    const auto regionCount = static_cast<std::size_t>(words[kRegionCountIndex]);
    for (std::size_t region = 0; region < regionCount; ++region) {
      EXPECT_LE(position + kRegionFixedWordCount, words.size());
      codecs.push_back(words[position + kRegionCodecOffset]);
      const auto segmentCount =
          static_cast<std::size_t>(words[position + kRegionSegmentCountOffset]);
      position += kRegionFixedWordCount + segmentCount;
    }
    EXPECT_EQ(position, words.size());
    return codecs;
  };

  const auto belowThreshold = selectedCodecs(kTypedThreshold - 1);
  EXPECT_TRUE(std::none_of(
      belowThreshold.begin(), belowThreshold.end(), [](int64_t codec) {
        return codec >= kFrameOfReferenceRegionCodec;
      }));

  const auto atThreshold = selectedCodecs(kTypedThreshold);
  EXPECT_TRUE(
      std::any_of(atThreshold.begin(), atThreshold.end(), [](int64_t codec) {
        return codec >= kFrameOfReferenceRegionCodec;
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
      kResidualThreshold,
      context));

  auto compressed = detail::compressAns(
      {static_cast<const uint8_t*>(input.data()), kResidualThreshold},
      kResidualThreshold,
      context);
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
      const auto codec = descriptor[descriptorPosition + kRegionCodecOffset];
      const auto segmentCount = static_cast<std::size_t>(
          descriptor[descriptorPosition + kRegionSegmentCountOffset]);
      const auto segmentSizes = descriptorPosition + kRegionFixedWordCount;
      if (codec == kRawRegionCodec) {
        for (auto offset = rawSize; offset < alignedEncodedSize(rawSize);
             ++offset) {
          EXPECT_EQ(bytes[encodedPosition + offset], 0);
        }
        encodedPosition += alignedEncodedSize(rawSize);
      } else {
        for (std::size_t segment = 0; segment < segmentCount; ++segment) {
          const auto size =
              static_cast<std::size_t>(descriptor[segmentSizes + segment]);
          for (auto offset = size; offset < alignedEncodedSize(size);
               ++offset) {
            EXPECT_EQ(bytes[encodedPosition + offset], 0);
          }
          encodedPosition += alignedEncodedSize(size);
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
