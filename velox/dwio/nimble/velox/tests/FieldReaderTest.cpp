/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <map>
#include <optional>

#include "folly/coro/BlockingWait.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/velox/Decoder.h"
#include "velox/dwio/nimble/velox/FieldReader.h"
#include "velox/dwio/nimble/velox/HybridFlatMap.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble {
namespace {

HybridFlatMap makeHybridFlatMap(
    std::vector<std::vector<std::string>> explicitGroups) {
  HybridFlatMap hybridMap;
  hybridMap.groups.reserve(explicitGroups.size() + 1);
  for (uint32_t groupId = 0; groupId < explicitGroups.size(); ++groupId) {
    hybridMap.groups.push_back(
        HybridFlatMap::Group{
            .groupId = groupId,
            .groupKeys = std::move(explicitGroups[groupId]),
        });
  }
  hybridMap.groups.push_back(
      HybridFlatMap::Group{
          .groupId = HybridFlatMap::kDefaultGroupId,
          .groupKeys = {},
      });
  return hybridMap;
}

class DecoderWithoutRead final : public Decoder {
 public:
  uint32_t next(
      uint32_t,
      void*,
      std::function<void*()>,
      std::vector<velox::BufferPtr>&,
      const velox::bits::Bitmap*) override {
    return 0;
  }

  uint32_t read(
      std::span<const uint32_t>,
      DataType,
      void*,
      std::function<void*()>,
      std::vector<velox::BufferPtr>&) override {
    return 0;
  }

  uint32_t read(
      std::span<const RowRange>,
      DataType,
      void*,
      std::function<void*()>,
      std::vector<velox::BufferPtr>&) override {
    return 0;
  }

  void skip(uint32_t) override {}
  void reset() override {}
  const Encoding* encoding() const override {
    return nullptr;
  }

  void read(
      const std::function<void*(uint32_t)>&,
      std::function<void*()>,
      std::vector<velox::BufferPtr>&) override {
    NIMBLE_UNSUPPORTED("read-all is not supported by this decoder");
  }
};

template <typename T>
class TestDecoder final : public Decoder {
 public:
  explicit TestDecoder(std::vector<T> values, bool nullable = false)
      : values_{std::move(values)}, nullable_{nullable} {}

  uint32_t next(
      uint32_t count,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>&,
      const velox::bits::Bitmap*) override {
    NIMBLE_CHECK_LE(index_ + count, values_.size());
    auto* typedOutput = static_cast<T*>(output);
    std::copy_n(values_.data() + index_, count, typedOutput);
    index_ += count;
    if (nullable_) {
      NIMBLE_CHECK_GT(count, 0);
      NIMBLE_CHECK_NOT_NULL(getOutputNulls);
      auto* nulls = static_cast<uint64_t*>(getOutputNulls());
      velox::bits::fillBits(
          nulls,
          0,
          static_cast<velox::vector_size_t>(count),
          velox::bits::kNotNull);
      velox::bits::clearBit(nulls, 0);
      return count - 1;
    }
    return count;
  }

  uint32_t read(
      std::span<const uint32_t>,
      DataType,
      void*,
      std::function<void*()>,
      std::vector<velox::BufferPtr>&) override {
    NIMBLE_UNSUPPORTED("not implemented");
  }

  uint32_t read(
      std::span<const RowRange>,
      DataType,
      void*,
      std::function<void*()>,
      std::vector<velox::BufferPtr>&) override {
    NIMBLE_UNSUPPORTED("not implemented");
  }

  void skip(uint32_t count) override {
    index_ += count;
  }

  void reset() override {
    index_ = 0;
  }

  const Encoding* encoding() const override {
    return nullptr;
  }

  void read(
      const std::function<void*(uint32_t)>& prepareOutput,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers) override {
    const auto rowCount = static_cast<uint32_t>(values_.size() - index_);
    if (rowCount == 0) {
      return;
    }
    auto* output = prepareOutput(rowCount);
    NIMBLE_CHECK_NOT_NULL(output);
    const auto nonNullCount = next(
        rowCount,
        output,
        std::move(getOutputNulls),
        stringBuffers,
        /*scatterOutputBitmap=*/nullptr);
    if (!nullable_) {
      NIMBLE_CHECK_EQ(
          nonNullCount, rowCount, "Test decoder values must be non-null.");
    }
  }

 private:
  std::vector<T> values_;
  uint32_t index_{0};
  const bool nullable_;
};

class BoolDecoder final : public Decoder {
 public:
  explicit BoolDecoder(std::vector<uint8_t> values, bool nullable = false)
      : values_{std::move(values)}, nullable_{nullable} {}

  uint32_t next(
      uint32_t count,
      void* output,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>&,
      const velox::bits::Bitmap*) override {
    NIMBLE_CHECK_LE(index_ + count, values_.size());
    auto* typedOutput = static_cast<bool*>(output);
    for (uint32_t i = 0; i < count; ++i) {
      typedOutput[i] = values_[index_++];
    }
    if (nullable_) {
      NIMBLE_CHECK_GT(count, 0);
      NIMBLE_CHECK_NOT_NULL(getOutputNulls);
      auto* nulls = static_cast<uint64_t*>(getOutputNulls());
      velox::bits::fillBits(
          nulls,
          0,
          static_cast<velox::vector_size_t>(count),
          velox::bits::kNotNull);
      velox::bits::clearBit(nulls, 0);
      return count - 1;
    }
    return count;
  }

  uint32_t read(
      std::span<const uint32_t>,
      DataType,
      void*,
      std::function<void*()>,
      std::vector<velox::BufferPtr>&) override {
    NIMBLE_UNSUPPORTED("not implemented");
  }

  uint32_t read(
      std::span<const RowRange>,
      DataType,
      void*,
      std::function<void*()>,
      std::vector<velox::BufferPtr>&) override {
    NIMBLE_UNSUPPORTED("not implemented");
  }

  void skip(uint32_t count) override {
    index_ += count;
  }

  void reset() override {
    index_ = 0;
  }

  const Encoding* encoding() const override {
    return nullptr;
  }

  void read(
      const std::function<void*(uint32_t)>& prepareOutput,
      std::function<void*()> getOutputNulls,
      std::vector<velox::BufferPtr>& stringBuffers) override {
    const auto rowCount = static_cast<uint32_t>(values_.size() - index_);
    if (rowCount == 0) {
      return;
    }
    auto* output = prepareOutput(rowCount);
    NIMBLE_CHECK_NOT_NULL(output);
    const auto nonNullCount = next(
        rowCount,
        output,
        std::move(getOutputNulls),
        stringBuffers,
        /*scatterOutputBitmap=*/nullptr);
    if (!nullable_) {
      NIMBLE_CHECK_EQ(
          nonNullCount, rowCount, "Test decoder values must be non-null.");
    }
  }

 private:
  std::vector<uint8_t> values_;
  uint32_t index_{0};
  const bool nullable_;
};

class FieldReaderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("field_reader_test");
    pool_ = rootPool_->addLeafChild("leaf");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(pool_.get());
  }

  template <typename T>
  void verifyHybridKeyType(
      const velox::TypePtr& keyType,
      const std::array<T, 4>& keyData,
      std::string groupedKey) {
    constexpr velox::vector_size_t kRows{2};
    constexpr velox::vector_size_t kEntries{4};
    const auto mapType = velox::MAP(keyType, velox::BIGINT());
    auto keys = velox::BaseVector::create(keyType, kEntries, pool_.get());
    auto values =
        velox::BaseVector::create(velox::BIGINT(), kEntries, pool_.get());
    for (velox::vector_size_t i = 0; i < kEntries; ++i) {
      keys->template asFlatVector<T>()->set(i, keyData[i]);
      values->template asFlatVector<int64_t>()->set(i, 10 + i);
    }
    auto map = std::make_shared<velox::MapVector>(
        pool_.get(),
        mapType,
        nullptr,
        kRows,
        velox::allocateOffsets(kRows, pool_.get()),
        velox::allocateSizes(kRows, pool_.get()),
        keys,
        values);
    const std::array<velox::vector_size_t, kRows> offsets{0, 2};
    const std::array<velox::vector_size_t, kRows> sizes{2, 2};
    std::copy(
        offsets.begin(),
        offsets.end(),
        map->mutableOffsets(kRows)->template asMutable<velox::vector_size_t>());
    std::copy(
        sizes.begin(),
        sizes.end(),
        map->mutableSizes(kRows)->template asMutable<velox::vector_size_t>());

    const auto input = vectorMaker_->rowVector({"features"}, {map});
    Serializer serializer{
        SerializerOptions{
            .version = SerializationVersion::kSerialization,
            .hybridFlatMapColumns =
                {{"features", makeHybridFlatMap({{std::move(groupedKey)}})}},
        },
        input->type(),
        pool_.get()};
    const std::string serialized{
        serializer.serialize(input, OrderedRanges::of(0, kRows))};
    const auto schema =
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
    Deserializer deserializer{schema, pool_.get()};
    velox::VectorPtr output;
    deserializer.deserialize(serialized, output);
    for (velox::vector_size_t row = 0; row < kRows; ++row) {
      EXPECT_TRUE(input->equalValueAt(output.get(), row, row));
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

TEST_F(FieldReaderTest, decoderReadAllCanBeUnsupported) {
  DecoderWithoutRead decoder;
  std::vector<velox::BufferPtr> stringBuffers;
  bool preparedOutput{false};
  NIMBLE_ASSERT_THROW(
      decoder.read(
          [&](uint32_t) -> void* {
            preparedOutput = true;
            return nullptr;
          },
          /*getOutputNulls=*/nullptr,
          stringBuffers),
      "read-all is not supported by this decoder");
  EXPECT_FALSE(preparedOutput);
}

TEST_F(FieldReaderTest, roundTripsEveryHybridKeyType) {
  verifyHybridKeyType<int8_t>(velox::TINYINT(), {1, 9, 9, 1}, "1");
  verifyHybridKeyType<int16_t>(velox::SMALLINT(), {1, 9, 9, 1}, "1");
  verifyHybridKeyType<int32_t>(velox::INTEGER(), {1, 9, 9, 1}, "1");
  verifyHybridKeyType<int64_t>(velox::BIGINT(), {1, 9, 9, 1}, "1");
  verifyHybridKeyType<velox::StringView>(
      velox::VARCHAR(),
      {velox::StringView{"a"},
       velox::StringView{"z"},
       velox::StringView{"z"},
       velox::StringView{"a"}},
      "a");
}

TEST_F(FieldReaderTest, createsRegularFlatMapReaderWithoutNullStream) {
  const auto type =
      velox::ROW({{"features", velox::MAP(velox::INTEGER(), velox::BIGINT())}});
  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .flatMapColumns = {{"features", {"1"}}},
      },
      type,
      pool_.get()};
  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(convertToVeloxType(*schema));
  std::vector<uint32_t> streamOffsets;
  auto factory = FieldReaderFactory::create(
      {},
      schema,
      typeWithId,
      streamOffsets,
      [](uint32_t) { return true; },
      pool_.get());
  folly::F14FastMap<offset_size, std::unique_ptr<Decoder>> decoders;
  EXPECT_NE(factory->createReader(decoders), nullptr);
}

TEST_F(FieldReaderTest, handlesHybridGroupsWithNoObservedKeys) {
  using Entry = std::pair<int32_t, std::optional<int64_t>>;
  const auto input = vectorMaker_->rowVector(
      {"features"},
      {vectorMaker_->mapVector<int32_t, int64_t>(
          std::vector<std::vector<Entry>>{{}, {}})});
  Serializer serializer{
      SerializerOptions{
          .hybridFlatMapColumns = {{"features", makeHybridFlatMap({{"1"}})}},
      },
      input->type(),
      pool_.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, input->size()))};
  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  Deserializer deserializer{schema, pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  const auto* map =
      output->as<velox::RowVector>()->childAt(0)->as<velox::MapVector>();
  ASSERT_NE(map, nullptr);
  EXPECT_EQ(map->sizeAt(0), 0);
  EXPECT_EQ(map->sizeAt(1), 0);
}

TEST_F(FieldReaderTest, rejectsMalformedHybridGroupStreams) {
  SchemaBuilder builder;
  auto root = builder.createRowTypeBuilder(1);
  auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::Int32);
  auto groupValue = builder.createScalarTypeBuilder(ScalarKind::Int64);
  const auto groupValueOffset = groupValue->scalarDescriptor().offset();
  const auto group = hybridMap->addGroup(7, {"1"}, std::move(groupValue));
  const auto defaultGroup = hybridMap->addGroup(
      HybridFlatMap::kDefaultGroupId,
      {},
      builder.createScalarTypeBuilder(ScalarKind::Int64));
  root->addChild("features", hybridMap);
  const auto schema = SchemaReader::getSchema(builder.schemaNodes());
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(convertToVeloxType(*schema));
  FieldReaderParams params;
  params.flatMapFeatureSelector["features"] = {
      .features = {"1"}, .mode = SelectionMode::Include};
  std::vector<uint32_t> streamOffsets;
  auto factory = FieldReaderFactory::create(
      params,
      schema,
      typeWithId,
      streamOffsets,
      [](uint32_t) { return true; },
      pool_.get());

  const auto expectFailure = [&](std::optional<std::vector<int32_t>> keys,
                                 std::optional<std::vector<uint8_t>> inMap,
                                 std::string_view message,
                                 bool isCorruptedFile = false) {
    folly::F14FastMap<offset_size, std::unique_ptr<Decoder>> decoders;
    if (keys.has_value()) {
      decoders[group.keyDescriptor.offset()] =
          std::make_unique<TestDecoder<int32_t>>(std::move(*keys));
    }
    if (inMap.has_value()) {
      decoders[group.inMapDescriptor.offset()] =
          std::make_unique<BoolDecoder>(std::move(*inMap));
    }
    auto reader = factory->createReader(decoders);
    velox::VectorPtr output;
    if (isCorruptedFile) {
      NIMBLE_ASSERT_FILE_THROW(
          folly::coro::blockingWait(reader->co_next(1, output)), message);
    } else {
      NIMBLE_ASSERT_THROW(
          folly::coro::blockingWait(reader->co_next(1, output)), message);
    }
  };

  expectFailure(
      std::vector<int32_t>{1, 2}, std::vector<uint8_t>{true}, "not divisible");
  expectFailure(
      std::vector<int32_t>{1, 1},
      std::vector<uint8_t>{true, true},
      "Duplicate key",
      /*isCorruptedFile=*/true);
  expectFailure(
      std::vector<int32_t>{2},
      std::vector<uint8_t>{true},
      "does not belong",
      /*isCorruptedFile=*/true);
  expectFailure(
      std::vector<int32_t>{1},
      std::vector<uint8_t>{false},
      "has no present rows",
      /*isCorruptedFile=*/true);
  expectFailure(
      std::nullopt,
      std::vector<uint8_t>{true},
      "key and in-map streams must be present together");
  expectFailure(
      std::vector<int32_t>{1},
      std::nullopt,
      "key and in-map streams must be present together");
  expectFailure(
      std::vector<int32_t>{},
      std::vector<uint8_t>{true},
      "in-map rows without keys");

  const auto expectNullableMetadataFailure = [&](bool nullableKey,
                                                 std::string_view message) {
    folly::F14FastMap<offset_size, std::unique_ptr<Decoder>> decoders;
    decoders[group.keyDescriptor.offset()] =
        std::make_unique<TestDecoder<int32_t>>(
            std::vector<int32_t>{1}, nullableKey);
    decoders[group.inMapDescriptor.offset()] =
        std::make_unique<BoolDecoder>(std::vector<uint8_t>{true}, !nullableKey);
    auto reader = factory->createReader(decoders);
    velox::VectorPtr output;
    NIMBLE_ASSERT_FILE_THROW(
        folly::coro::blockingWait(reader->co_next(1, output)), message);
  };
  expectNullableMetadataFailure(
      /*nullableKey=*/true, "key stream must not contain nulls");
  expectNullableMetadataFailure(
      /*nullableKey=*/false, "in-map stream must not contain nulls");

  folly::F14FastMap<offset_size, std::unique_ptr<Decoder>> partialDecoders;
  partialDecoders[group.keyDescriptor.offset()] =
      std::make_unique<TestDecoder<int32_t>>(std::vector<int32_t>{1});
  partialDecoders[group.inMapDescriptor.offset()] =
      std::make_unique<BoolDecoder>(std::vector<uint8_t>{true, true});
  partialDecoders[groupValueOffset] =
      std::make_unique<TestDecoder<int64_t>>(std::vector<int64_t>{10, 11});
  auto partialReader = factory->createReader(partialDecoders);
  velox::VectorPtr partialOutput;
  folly::coro::blockingWait(partialReader->co_next(1, partialOutput));
  const auto* partialMap =
      partialOutput->as<velox::RowVector>()->childAt(0)->as<velox::MapVector>();
  ASSERT_NE(partialMap, nullptr);
  EXPECT_EQ(partialMap->sizeAt(0), 1);
  EXPECT_NO_THROW(partialReader->reset());

  std::vector<uint32_t> allStreamOffsets;
  auto allGroupsFactory = FieldReaderFactory::create(
      {},
      schema,
      typeWithId,
      allStreamOffsets,
      [](uint32_t) { return true; },
      pool_.get());
  folly::F14FastMap<offset_size, std::unique_ptr<Decoder>> defaultDecoders;
  defaultDecoders[defaultGroup.keyDescriptor.offset()] =
      std::make_unique<TestDecoder<int32_t>>(std::vector<int32_t>{1});
  defaultDecoders[defaultGroup.inMapDescriptor.offset()] =
      std::make_unique<BoolDecoder>(std::vector<uint8_t>{true});
  auto defaultReader = allGroupsFactory->createReader(defaultDecoders);
  velox::VectorPtr output;
  NIMBLE_ASSERT_FILE_THROW(
      folly::coro::blockingWait(defaultReader->co_next(1, output)),
      "is configured in a non-Default group but appears in the Default "
      "group's key stream");
}

TEST_F(FieldReaderTest, filtersDecodedHybridGroupsToRequestedFeatures) {
  using Entry = std::pair<int32_t, std::optional<int64_t>>;
  const std::vector<std::vector<Entry>> maps{
      {{1, 10}, {2, 20}, {9, 90}, {10, 100}},
      {{2, 21}, {10, 101}},
      {{1, 12}, {9, 92}},
      {{1, 13}, {9, 93}},
  };
  const auto input = vectorMaker_->rowVector(
      {"features"}, {vectorMaker_->mapVector<int32_t, int64_t>(maps)});
  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .hybridFlatMapColumns =
              {{"features", makeHybridFlatMap({{"1", "2"}})}},
      },
      input->type(),
      pool_.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, input->size()))};
  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  Deserializer wholeGroupDeserializer{schema, pool_.get()};
  velox::VectorPtr wholeGroupOutput;
  wholeGroupDeserializer.deserialize(serialized, wholeGroupOutput);
  for (velox::vector_size_t row = 0; row < input->size(); ++row) {
    EXPECT_TRUE(input->equalValueAt(wholeGroupOutput.get(), row, row));
  }

  std::vector<Deserializer::Subfield> selected;
  selected.emplace_back("features[1]");
  selected.emplace_back("features[9]");
  Deserializer deserializer{
      schema, selected, pool_.get(), DeserializerOptions{}};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  const auto* outputMap =
      output->as<velox::RowVector>()->childAt(0)->as<velox::MapVector>();
  ASSERT_NE(outputMap, nullptr);
  const auto* keys = outputMap->mapKeys()->as<velox::FlatVector<int32_t>>();
  const auto* values = outputMap->mapValues()->as<velox::FlatVector<int64_t>>();
  ASSERT_NE(keys, nullptr);
  ASSERT_NE(values, nullptr);
  const std::array<std::map<int32_t, int64_t>, 4> expected{
      std::map<int32_t, int64_t>{{1, 10}, {9, 90}},
      std::map<int32_t, int64_t>{},
      std::map<int32_t, int64_t>{{1, 12}, {9, 92}},
      std::map<int32_t, int64_t>{{1, 13}, {9, 93}},
  };
  for (velox::vector_size_t row = 0; row < outputMap->size(); ++row) {
    std::map<int32_t, int64_t> actual;
    for (velox::vector_size_t entry = 0; entry < outputMap->sizeAt(row);
         ++entry) {
      const auto index = outputMap->offsetAt(row) + entry;
      actual.emplace(keys->valueAt(index), values->valueAt(index));
    }
    EXPECT_EQ(actual, expected[row]);
  }

  deserializer.deserialize(serialized, std::vector<RowRange>{{1, 2}}, output);
  ASSERT_EQ(output->size(), 1);
  EXPECT_EQ(
      output->as<velox::RowVector>()
          ->childAt(0)
          ->as<velox::MapVector>()
          ->sizeAt(0),
      0);

  deserializer.deserialize(
      serialized, std::vector<RowRange>{{1, 2}, {3, 4}}, output);
  ASSERT_EQ(output->size(), 2);
  EXPECT_EQ(
      output->as<velox::RowVector>()
          ->childAt(0)
          ->as<velox::MapVector>()
          ->sizeAt(0),
      0);
  EXPECT_EQ(
      output->as<velox::RowVector>()
          ->childAt(0)
          ->as<velox::MapVector>()
          ->sizeAt(1),
      2);

  deserializer.deserialize(serialized, std::vector<RowRange>{{0, 0}}, output);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(), 0);
}

TEST_F(FieldReaderTest, excludesRequestedKeysAfterWholeGroupDecode) {
  SchemaBuilder builder;
  auto root = builder.createRowTypeBuilder(1);
  auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::Int32);
  auto groupValue = builder.createScalarTypeBuilder(ScalarKind::Int64);
  const auto groupValueOffset = groupValue->scalarDescriptor().offset();
  const auto group = hybridMap->addGroup(0, {"1", "2"}, std::move(groupValue));
  auto defaultValue = builder.createScalarTypeBuilder(ScalarKind::Int64);
  const auto defaultValueOffset = defaultValue->scalarDescriptor().offset();
  const auto defaultGroup = hybridMap->addGroup(
      HybridFlatMap::kDefaultGroupId, {}, std::move(defaultValue));
  root->addChild("features", hybridMap);
  const auto schema = SchemaReader::getSchema(builder.schemaNodes());
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(convertToVeloxType(*schema));
  FieldReaderParams params;
  params.flatMapFeatureSelector["features"] = {
      .features = {"2", "9"}, .mode = SelectionMode::Exclude};
  std::vector<uint32_t> streamOffsets;
  auto factory = FieldReaderFactory::create(
      params,
      schema,
      typeWithId,
      streamOffsets,
      [](uint32_t) { return true; },
      pool_.get());

  folly::F14FastMap<offset_size, std::unique_ptr<Decoder>> decoders;
  decoders[group.keyDescriptor.offset()] =
      std::make_unique<TestDecoder<int32_t>>(std::vector<int32_t>{1, 2});
  decoders[group.inMapDescriptor.offset()] =
      std::make_unique<BoolDecoder>(std::vector<uint8_t>{true, true});
  decoders[groupValueOffset] =
      std::make_unique<TestDecoder<int64_t>>(std::vector<int64_t>{10, 20});
  decoders[defaultGroup.keyDescriptor.offset()] =
      std::make_unique<TestDecoder<int32_t>>(std::vector<int32_t>{9, 10});
  decoders[defaultGroup.inMapDescriptor.offset()] =
      std::make_unique<BoolDecoder>(std::vector<uint8_t>{true, true});
  decoders[defaultValueOffset] =
      std::make_unique<TestDecoder<int64_t>>(std::vector<int64_t>{90, 100});

  auto reader = factory->createReader(decoders);
  velox::VectorPtr output;
  folly::coro::blockingWait(reader->co_next(1, output));
  const auto* map =
      output->as<velox::RowVector>()->childAt(0)->as<velox::MapVector>();
  ASSERT_NE(map, nullptr);
  ASSERT_EQ(map->sizeAt(0), 2);
  const auto* keys = map->mapKeys()->as<velox::FlatVector<int32_t>>();
  ASSERT_NE(keys, nullptr);
  EXPECT_EQ(keys->valueAt(map->offsetAt(0)), 1);
  EXPECT_EQ(keys->valueAt(map->offsetAt(0) + 1), 10);
}

TEST_F(FieldReaderTest, skipsNullRowsWithoutAdvancingHybridGroups) {
  using Entry = std::pair<int32_t, std::optional<int64_t>>;
  const auto map = vectorMaker_->mapVector<int32_t, int64_t>(
      std::vector<std::vector<Entry>>{{{1, 10}}, {{1, 20}}});
  map->setNull(0, true);
  const auto input = vectorMaker_->rowVector({"features"}, {map});
  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .hybridFlatMapColumns = {{"features", makeHybridFlatMap({{"1"}})}},
      },
      input->type(),
      pool_.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, input->size()))};
  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  Deserializer deserializer{schema, pool_.get()};

  velox::VectorPtr output;
  deserializer.deserialize(serialized, std::vector<RowRange>{{1, 2}}, output);

  ASSERT_EQ(output->size(), 1);
  EXPECT_TRUE(input->equalValueAt(output.get(), 1, 0));
}

TEST_F(FieldReaderTest, hybridFlatMapRoundTripsNestedValues) {
  constexpr velox::vector_size_t kRows{3};
  constexpr velox::vector_size_t kMapEntries{6};
  constexpr velox::vector_size_t kArrayElements{7};
  const auto valueType = velox::ARRAY(velox::BIGINT());
  const auto mapType = velox::MAP(velox::INTEGER(), valueType);

  auto keys =
      velox::BaseVector::create(velox::INTEGER(), kMapEntries, pool_.get());
  const std::array<int32_t, kMapEntries> keyData{1, 99, 2, 3, 99, 1};
  for (velox::vector_size_t i = 0; i < kMapEntries; ++i) {
    keys->asFlatVector<int32_t>()->set(i, keyData[i]);
  }

  auto elements =
      velox::BaseVector::create(velox::BIGINT(), kArrayElements, pool_.get());
  const std::array<int64_t, kArrayElements> elementData{
      10, 11, 90, 20, 30, 31, 12};
  for (velox::vector_size_t i = 0; i < kArrayElements; ++i) {
    elements->asFlatVector<int64_t>()->set(i, elementData[i]);
  }

  auto arrays = std::make_shared<velox::ArrayVector>(
      pool_.get(),
      valueType,
      nullptr,
      kMapEntries,
      velox::allocateOffsets(kMapEntries, pool_.get()),
      velox::allocateSizes(kMapEntries, pool_.get()),
      elements);
  const std::array<velox::vector_size_t, kMapEntries> arrayOffsets{
      0, 2, 3, 4, 6, 6};
  const std::array<velox::vector_size_t, kMapEntries> arraySizes{
      2, 1, 1, 2, 0, 1};
  std::copy(
      arrayOffsets.begin(),
      arrayOffsets.end(),
      arrays->mutableOffsets(kMapEntries)->asMutable<velox::vector_size_t>());
  std::copy(
      arraySizes.begin(),
      arraySizes.end(),
      arrays->mutableSizes(kMapEntries)->asMutable<velox::vector_size_t>());

  auto map = std::make_shared<velox::MapVector>(
      pool_.get(),
      mapType,
      nullptr,
      kRows,
      velox::allocateOffsets(kRows, pool_.get()),
      velox::allocateSizes(kRows, pool_.get()),
      keys,
      arrays);
  const std::array<velox::vector_size_t, kRows> mapOffsets{0, 2, 4};
  const std::array<velox::vector_size_t, kRows> mapSizes{2, 2, 2};
  std::copy(
      mapOffsets.begin(),
      mapOffsets.end(),
      map->mutableOffsets(kRows)->asMutable<velox::vector_size_t>());
  std::copy(
      mapSizes.begin(),
      mapSizes.end(),
      map->mutableSizes(kRows)->asMutable<velox::vector_size_t>());

  const auto type = velox::ROW({{"features", mapType}});
  const auto input = std::make_shared<velox::RowVector>(
      pool_.get(), type, nullptr, kRows, std::vector<velox::VectorPtr>{map});
  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .hybridFlatMapColumns =
              {{"features", makeHybridFlatMap({{"1", "2"}, {"3"}})}},
      },
      type,
      pool_.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, kRows))};
  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const auto& hybridMap = schema->asRow().childAt(0)->asHybridFlatMap();
  ASSERT_EQ(hybridMap.groupCount(), 3);
  EXPECT_TRUE(hybridMap.groupAt(0).valueType->isArray());
  EXPECT_EQ(hybridMap.groupAt(2).groupId, HybridFlatMap::kDefaultGroupId);

  Deserializer deserializer{schema, pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);
  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    EXPECT_TRUE(input->equalValueAt(output.get(), row, row))
        << "row " << row << " expected " << input->toString(row) << " actual "
        << output->toString(row);
  }
}

} // namespace
} // namespace facebook::nimble
