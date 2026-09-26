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
#include <cstring>
#include <optional>
#include <string_view>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/velox/FieldWriter.h"
#include "velox/dwio/nimble/velox/HybridFlatMap.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
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

class FieldWriterTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("field_writer_test");
    pool_ = rootPool_->addLeafChild("leaf");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(pool_.get());
  }

  static const StreamData* findStream(
      FieldWriterContext& context,
      uint32_t offset) {
    for (const auto& [_, stream] : context.streams()) {
      if (stream->descriptor().offset() == offset) {
        return stream.get();
      }
    }
    return nullptr;
  }

  template <typename T>
  static std::vector<T> readStream(const StreamData& stream) {
    const auto data = stream.data();
    EXPECT_EQ(data.size() % sizeof(T), 0);
    std::vector<T> values(data.size() / sizeof(T));
    if (!data.empty()) {
      std::memcpy(values.data(), data.data(), data.size());
    }
    return values;
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

TEST_F(FieldWriterTest, writesHybridGroupKeysInMapsAndValues) {
  using Entry = std::pair<int32_t, std::optional<int64_t>>;
  const std::vector<std::vector<Entry>> maps{
      {{1, 10}, {9, 90}},
      {{2, 20}},
      {{1, 11}, {2, 21}},
  };
  const auto input = vectorMaker_->rowVector(
      {"features"}, {vectorMaker_->mapVector<int32_t, int64_t>(maps)});
  const std::shared_ptr<const velox::dwio::common::TypeWithId> schemaWithId =
      velox::dwio::common::TypeWithId::create(input->type());
  const auto& mapNode = schemaWithId->childAt(0);

  FieldWriterContext context{*pool_};
  context.addHybridFlatMapNode(
      mapNode->id(), makeHybridFlatMap({{"1", "2"}, {"3"}}));
  EXPECT_FALSE(context.hasFlatMapNodeId(mapNode->id()));
  EXPECT_TRUE(context.hasHybridFlatMapNodeId(mapNode->id()));
  auto writer = FieldWriter::create(context, schemaWithId);
  writer->write(input, OrderedRanges::of(0, input->size()));

  const auto schema =
      SchemaReader::getSchema(context.schemaBuilder().schemaNodes());
  const auto& hybridMap = schema->asRow().childAt(0)->asHybridFlatMap();
  ASSERT_EQ(hybridMap.groupCount(), 3);
  EXPECT_EQ(hybridMap.groupAt(0).groupId, 0);
  EXPECT_EQ(
      hybridMap.groupAt(0).groupKeys, (std::vector<std::string>{"1", "2"}));
  EXPECT_EQ(hybridMap.groupAt(1).groupId, 1);
  EXPECT_EQ(hybridMap.groupAt(1).groupKeys, (std::vector<std::string>{"3"}));
  EXPECT_EQ(hybridMap.groupAt(2).groupId, HybridFlatMap::kDefaultGroupId);
  // Default's observed keys ride in its data stream, not in the schema.
  EXPECT_TRUE(hybridMap.groupAt(2).groupKeys.empty());

  const auto& explicitGroup = hybridMap.groupAt(0);
  const auto* explicitKeys =
      findStream(context, explicitGroup.keyDescriptor.offset());
  const auto* explicitInMap =
      findStream(context, explicitGroup.inMapDescriptor.offset());
  const auto* explicitValues = findStream(
      context, explicitGroup.valueType->asScalar().scalarDescriptor().offset());
  ASSERT_NE(explicitKeys, nullptr);
  ASSERT_NE(explicitInMap, nullptr);
  ASSERT_NE(explicitValues, nullptr);
  EXPECT_EQ(readStream<int32_t>(*explicitKeys), (std::vector<int32_t>{1, 2}));
  EXPECT_EQ(
      readStream<uint8_t>(*explicitInMap),
      (std::vector<uint8_t>{1, 0, 1, 0, 1, 1}));
  EXPECT_EQ(
      readStream<int64_t>(*explicitValues),
      (std::vector<int64_t>{10, 11, 20, 21}));

  const auto& defaultGroup = hybridMap.groupAt(2);
  EXPECT_EQ(defaultGroup.groupId, HybridFlatMap::kDefaultGroupId);
  const auto* defaultKeys =
      findStream(context, defaultGroup.keyDescriptor.offset());
  const auto* defaultInMap =
      findStream(context, defaultGroup.inMapDescriptor.offset());
  const auto* defaultValues = findStream(
      context, defaultGroup.valueType->asScalar().scalarDescriptor().offset());
  ASSERT_NE(defaultKeys, nullptr);
  ASSERT_NE(defaultInMap, nullptr);
  ASSERT_NE(defaultValues, nullptr);
  EXPECT_EQ(readStream<int32_t>(*defaultKeys), (std::vector<int32_t>{9}));
  EXPECT_EQ(
      readStream<uint8_t>(*defaultInMap), (std::vector<uint8_t>{1, 0, 0}));
  EXPECT_EQ(readStream<int64_t>(*defaultValues), (std::vector<int64_t>{90}));
}

TEST_F(FieldWriterTest, omitsAbsentHybridKeysAndEmptyGroups) {
  using Entry = std::pair<int32_t, std::optional<int64_t>>;
  const auto map = vectorMaker_->mapVector<int32_t, int64_t>(
      std::vector<std::vector<Entry>>{{{1, 10}}});
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(map->type());

  FieldWriterContext context{*pool_};
  context.addHybridFlatMapNode(
      typeWithId->id(), makeHybridFlatMap({{"1", "2"}}));
  auto writer = FieldWriter::create(context, typeWithId);
  writer->write(map, OrderedRanges::of(0, map->size()));

  const auto schema =
      SchemaReader::getSchema(context.schemaBuilder().schemaNodes());
  const auto& writtenMap = schema->asHybridFlatMap();
  ASSERT_EQ(writtenMap.groupCount(), 2);

  const auto& explicitGroup = writtenMap.groupAt(0);
  const auto* explicitKeys =
      findStream(context, explicitGroup.keyDescriptor.offset());
  ASSERT_NE(explicitKeys, nullptr);
  EXPECT_EQ(readStream<int32_t>(*explicitKeys), (std::vector<int32_t>{1}));

  const auto& defaultGroup = writtenMap.groupAt(1);
  const auto* defaultKeys =
      findStream(context, defaultGroup.keyDescriptor.offset());
  const auto* defaultInMap =
      findStream(context, defaultGroup.inMapDescriptor.offset());
  const auto* defaultValues = findStream(
      context, defaultGroup.valueType->asScalar().scalarDescriptor().offset());
  ASSERT_NE(defaultKeys, nullptr);
  ASSERT_NE(defaultInMap, nullptr);
  ASSERT_NE(defaultValues, nullptr);
  EXPECT_EQ(defaultKeys->rowCount(), 0);
  EXPECT_EQ(defaultInMap->rowCount(), 0);
  EXPECT_EQ(defaultValues->rowCount(), 0);
}

TEST_F(FieldWriterTest, writesStringKeysInGroupOrder) {
  using Entry = std::pair<velox::StringView, std::optional<int64_t>>;
  const auto map = vectorMaker_->mapVector<velox::StringView, int64_t>(
      std::vector<std::vector<Entry>>{{{"b", 20}, {"x", 90}, {"a", 10}}});
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(map->type());

  FieldWriterContext context{*pool_};
  context.addHybridFlatMapNode(
      typeWithId->id(), makeHybridFlatMap({{"b", "a"}}));
  auto writer = FieldWriter::create(context, typeWithId);
  const auto* configuredHybridMap = context.hybridFlatMapNode(typeWithId->id());
  ASSERT_NE(configuredHybridMap, nullptr);
  EXPECT_EQ(
      configuredHybridMap->groups[0].groupKeys,
      (std::vector<std::string>{"a", "b"}));
  writer->write(map, OrderedRanges::of(0, map->size()));

  const auto schema =
      SchemaReader::getSchema(context.schemaBuilder().schemaNodes());
  const auto& hybridMap = schema->asHybridFlatMap();
  EXPECT_EQ(
      hybridMap.groupAt(0).groupKeys, (std::vector<std::string>{"a", "b"}));

  const auto* configuredKeys =
      findStream(context, hybridMap.groupAt(0).keyDescriptor.offset());
  const auto* defaultKeys =
      findStream(context, hybridMap.defaultGroup().keyDescriptor.offset());
  ASSERT_NE(configuredKeys, nullptr);
  ASSERT_NE(defaultKeys, nullptr);
  EXPECT_EQ(
      readStream<std::string_view>(*configuredKeys),
      (std::vector<std::string_view>{"a", "b"}));
  EXPECT_EQ(
      readStream<std::string_view>(*defaultKeys),
      (std::vector<std::string_view>{"x"}));
}

TEST_F(FieldWriterTest, defaultKeysAreBatchLocalAndDoNotChangeSchema) {
  using Entry = std::pair<int32_t, std::optional<int64_t>>;
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(
          velox::MAP(velox::INTEGER(), velox::BIGINT()));
  FieldWriterContext context{*pool_};
  context.addHybridFlatMapNode(typeWithId->id(), makeHybridFlatMap({{"1"}}));
  auto writer = FieldWriter::create(context, typeWithId);

  const auto writeAndReadDefaultKeys =
      [&](const std::vector<std::vector<Entry>>& maps) {
        const auto input = vectorMaker_->mapVector<int32_t, int64_t>(maps);
        writer->write(input, OrderedRanges::of(0, input->size()));
        const auto schema =
            SchemaReader::getSchema(context.schemaBuilder().schemaNodes());
        const auto& defaultGroup = schema->asHybridFlatMap().defaultGroup();
        EXPECT_TRUE(defaultGroup.groupKeys.empty());
        const auto* keys =
            findStream(context, defaultGroup.keyDescriptor.offset());
        NIMBLE_CHECK_NOT_NULL(keys);
        return readStream<int32_t>(*keys);
      };

  EXPECT_EQ(
      writeAndReadDefaultKeys({{{1, 10}, {9, 90}, {8, 80}}}),
      (std::vector<int32_t>{9, 8}));
  writer->reset();
  EXPECT_EQ(
      writeAndReadDefaultKeys({{{1, 11}, {10, 100}, {9, 99}}}),
      (std::vector<int32_t>{10, 9}));
}

TEST_F(FieldWriterTest, rejectsEmptyConfiguredKey) {
  const auto type =
      velox::ROW({{"features", velox::MAP(velox::VARCHAR(), velox::DOUBLE())}});
  const SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .hybridFlatMapColumns = {{"features", makeHybridFlatMap({{""}})}},
  };

  NIMBLE_ASSERT_THROW(
      std::make_unique<Serializer>(options, type, pool_.get()),
      "Hybrid FlatMap key cannot be empty");
}

TEST_F(FieldWriterTest, rejectsInvalidConfiguredNumericKeys) {
  const auto type =
      velox::ROW({{"features", velox::MAP(velox::INTEGER(), velox::BIGINT())}});
  const auto expectRejected = [&](std::string key) {
    const SerializerOptions options{
        .version = SerializationVersion::kSerialization,
        .hybridFlatMapColumns =
            {{"features", makeHybridFlatMap({{std::move(key)}})}},
    };
    NIMBLE_ASSERT_THROW(
        std::make_unique<Serializer>(options, type, pool_.get()),
        "cannot be parsed");
  };

  expectRejected("not-a-number");
  expectRejected("2147483648");
}

TEST_F(FieldWriterTest, rejectsUnsupportedHybridKeyType) {
  const auto expectRejected = [&](const velox::TypePtr& keyType,
                                  std::string key) {
    const auto type =
        velox::ROW({{"features", velox::MAP(keyType, velox::BIGINT())}});
    const SerializerOptions options{
        .version = SerializationVersion::kSerialization,
        .hybridFlatMapColumns =
            {{"features", makeHybridFlatMap({{std::move(key)}})}},
    };
    NIMBLE_ASSERT_THROW(
        std::make_unique<Serializer>(options, type, pool_.get()),
        "Unsupported hybrid FlatMap key type");
  };

  expectRejected(velox::BOOLEAN(), "true");
  expectRejected(velox::VARBINARY(), "binary");
}

TEST_F(FieldWriterTest, rejectsDuplicateDefaultGroupsAfterWrite) {
  const auto type = velox::MAP(velox::INTEGER(), velox::BIGINT());
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(type);
  FieldWriterContext context{*pool_};
  context.addHybridFlatMapNode(
      typeWithId->id(),
      HybridFlatMap{
          .groups =
              {
                  HybridFlatMap::Group{
                      .groupId = HybridFlatMap::kDefaultGroupId,
                      .groupKeys = {},
                  },
                  HybridFlatMap::Group{
                      .groupId = HybridFlatMap::kDefaultGroupId,
                      .groupKeys = {},
                  },
              },
      });

  const auto writer = FieldWriter::create(context, typeWithId);
  ASSERT_NE(writer, nullptr);
  NIMBLE_ASSERT_THROW(
      context.schemaBuilder().schemaNodes(),
      "Duplicate Hybrid FlatMap group ID: 4294967295");
}

TEST_F(FieldWriterTest, rejectsConfiguredKeysInDefaultGroup) {
  const auto type = velox::MAP(velox::INTEGER(), velox::BIGINT());
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(type);
  FieldWriterContext context{*pool_};
  context.addHybridFlatMapNode(
      typeWithId->id(),
      HybridFlatMap{
          .groups =
              {
                  HybridFlatMap::Group{
                      .groupId = 0,
                      .groupKeys = {"1"},
                  },
                  HybridFlatMap::Group{
                      .groupId = HybridFlatMap::kDefaultGroupId,
                      .groupKeys = {"2"},
                  },
              },
      });

  EXPECT_THROW(FieldWriter::create(context, typeWithId), NimbleInternalError);
}

TEST_F(FieldWriterTest, initializesEverySupportedHybridValueType) {
  const auto valueType = velox::ROW(
      {"boolean",
       "tinyint",
       "smallint",
       "integer",
       "bigint",
       "real",
       "double",
       "varchar",
       "varbinary",
       "timestamp",
       "array",
       "offset_array",
       "map",
       "sliding_map"},
      {velox::BOOLEAN(),
       velox::TINYINT(),
       velox::SMALLINT(),
       velox::INTEGER(),
       velox::BIGINT(),
       velox::REAL(),
       velox::DOUBLE(),
       velox::VARCHAR(),
       velox::VARBINARY(),
       velox::TIMESTAMP(),
       velox::ARRAY(velox::INTEGER()),
       velox::ARRAY(velox::SMALLINT()),
       velox::MAP(velox::VARCHAR(), velox::DOUBLE()),
       velox::MAP(velox::TINYINT(), velox::BOOLEAN())});
  const auto type = velox::MAP(velox::INTEGER(), valueType);
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(type);
  const auto& valueNode = typeWithId->childAt(1);

  FieldWriterContext context{*pool_};
  context.addHybridFlatMapNode(typeWithId->id(), makeHybridFlatMap({{"1"}}));
  context.dictionaryArrayNodeIds().insert(
      valueNode->childByName("offset_array")->id());
  context.deduplicatedMapNodeIds().insert(
      valueNode->childByName("sliding_map")->id());

  auto writer = FieldWriter::create(context, typeWithId);
  writer->reset();
  writer->close();

  const auto schema =
      SchemaReader::getSchema(context.schemaBuilder().schemaNodes());
  const auto& hybridMap = schema->asHybridFlatMap();
  ASSERT_EQ(hybridMap.groupCount(), 2);
  EXPECT_EQ(hybridMap.groupAt(1).groupId, HybridFlatMap::kDefaultGroupId);
  // Each group owns a complete copy of the value subtree.
  const auto countValueStreams = [](const Type& valueType) {
    size_t count{0};
    visitValueStreamLeaves(valueType, [&](offset_size) {
      ++count;
      return false;
    });
    return count;
  };
  // Counts value-stream leaves; the pre-V15 assertion counted every stream
  // descriptor in the subtree, including Row nulls and Array lengths.
  EXPECT_EQ(countValueStreams(*hybridMap.groupAt(0).valueType), 14);
  EXPECT_EQ(countValueStreams(*hybridMap.groupAt(1).valueType), 14);
}

TEST_F(FieldWriterTest, delegatesUnsupportedHybridValueTypeValidation) {
  const auto type = velox::MAP(velox::INTEGER(), velox::UNKNOWN());
  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(type);
  FieldWriterContext context{*pool_};
  context.addHybridFlatMapNode(typeWithId->id(), makeHybridFlatMap({{"1"}}));

  NIMBLE_ASSERT_THROW(
      FieldWriter::create(context, typeWithId), "Unsupported kind: UNKNOWN");
}

TEST_F(FieldWriterTest, writesDecodedMapAndKeyVectors) {
  using Entry = std::pair<int32_t, std::optional<int64_t>>;
  const auto baseMap = vectorMaker_->mapVector<int32_t, int64_t>(
      std::vector<std::vector<Entry>>{{{1, 10}, {9, 90}}, {{1, 11}}});

  auto keyIndices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(3, pool_.get());
  auto* rawKeyIndices = keyIndices->asMutable<velox::vector_size_t>();
  rawKeyIndices[0] = 0;
  rawKeyIndices[1] = 1;
  rawKeyIndices[2] = 2;
  auto dictionaryKeys = velox::BaseVector::wrapInDictionary(
      nullptr, std::move(keyIndices), 3, baseMap->mapKeys());
  auto map = std::make_shared<velox::MapVector>(
      pool_.get(),
      baseMap->type(),
      nullptr,
      baseMap->size(),
      baseMap->offsets(),
      baseMap->sizes(),
      std::move(dictionaryKeys),
      baseMap->mapValues());

  auto rowIndices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(2, pool_.get());
  rowIndices->asMutable<velox::vector_size_t>()[0] = 1;
  rowIndices->asMutable<velox::vector_size_t>()[1] = 0;
  const auto dictionaryMap = velox::BaseVector::wrapInDictionary(
      nullptr, std::move(rowIndices), 2, map);

  const std::shared_ptr<const velox::dwio::common::TypeWithId> typeWithId =
      velox::dwio::common::TypeWithId::create(baseMap->type());
  FieldWriterContext context{*pool_};
  context.addHybridFlatMapNode(typeWithId->id(), makeHybridFlatMap({{"1"}}));
  auto writer = FieldWriter::create(context, typeWithId);
  writer->write(dictionaryMap, OrderedRanges::of(0, dictionaryMap->size()));

  const auto schema =
      SchemaReader::getSchema(context.schemaBuilder().schemaNodes());
  const auto& writtenMap = schema->asHybridFlatMap();
  ASSERT_EQ(writtenMap.groupCount(), 2);
  const auto* explicitValues = findStream(
      context,
      writtenMap.groupAt(0).valueType->asScalar().scalarDescriptor().offset());
  const auto* defaultValues = findStream(
      context,
      writtenMap.groupAt(1).valueType->asScalar().scalarDescriptor().offset());
  ASSERT_NE(explicitValues, nullptr);
  ASSERT_NE(defaultValues, nullptr);
  const auto readValues = [](const StreamData& stream) {
    const auto data = stream.data();
    EXPECT_EQ(data.size() % sizeof(int64_t), 0);
    std::vector<int64_t> values(data.size() / sizeof(int64_t));
    if (!data.empty()) {
      std::memcpy(values.data(), data.data(), data.size());
    }
    return values;
  };
  EXPECT_EQ(readValues(*explicitValues), (std::vector<int64_t>{11, 10}));
  EXPECT_EQ(readValues(*defaultValues), (std::vector<int64_t>{90}));

  writer->reset();
  EXPECT_EQ(explicitValues->rowCount(), 0);
  EXPECT_EQ(defaultValues->rowCount(), 0);
  writer->write(baseMap, OrderedRanges::of(0, baseMap->size()));
  EXPECT_EQ(readValues(*explicitValues), (std::vector<int64_t>{10, 11}));
  EXPECT_EQ(readValues(*defaultValues), (std::vector<int64_t>{90}));
  writer->close();
}

} // namespace
} // namespace facebook::nimble
