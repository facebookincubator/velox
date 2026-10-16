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
#include <set>
#include <string>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/velox/HybridFlatMap.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

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

std::shared_ptr<TypeBuilder> makeComplexValueType(SchemaBuilder& builder) {
  auto row = builder.createRowTypeBuilder(3);
  row->addChild("scalar", builder.createScalarTypeBuilder(ScalarKind::Int64));
  auto array = builder.createArrayTypeBuilder();
  array->setChildren(builder.createScalarTypeBuilder(ScalarKind::Int32));
  row->addChild("array", array);
  auto map = builder.createMapTypeBuilder();
  map->setChildren(
      builder.createScalarTypeBuilder(ScalarKind::String),
      builder.createScalarTypeBuilder(ScalarKind::Double));
  row->addChild("map", map);
  return row;
}

class DeserializerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("deserializer_test");
    pool_ = rootPool_->addLeafChild("leaf");
  }

  // Maps one selected key subfield to the JSON maps it decodes to.
  struct KeySelection {
    std::string subfield;
    std::vector<std::string> expectedMaps;
  };

  // Serializes `inputMaps` as a `features` column whose Hybrid FlatMap has
  // only the Default group, then checks a full decode and one decode per entry
  // of `selections` against the expected maps.
  template <typename TKey>
  void verifyDefaultOnlyRoundTrip(
      const std::vector<std::string>& inputMaps,
      const std::vector<std::string>& expectedGroupKeys,
      const std::vector<KeySelection>& selections) {
    velox::test::VectorMaker vectorMaker{pool_.get()};
    const auto makeInput = [&](const std::vector<std::string>& maps) {
      return vectorMaker.rowVector(
          {"features"}, {vectorMaker.mapVectorFromJson<TKey, double>(maps)});
    };
    const auto input = makeInput(inputMaps);
    Serializer serializer{
        SerializerOptions{
            .version = SerializationVersion::kSerialization,
            .hybridFlatMapColumns = {{"features", makeHybridFlatMap({})}},
        },
        input->type(),
        pool_.get()};
    const std::string serialized{
        serializer.serialize(input, OrderedRanges::of(0, input->size()))};

    // A batch of only empty and null maps observes no Default keys.
    const auto emptyInput = makeInput({"{}", "null"});
    const std::string emptySerialized{serializer.serialize(
        emptyInput, OrderedRanges::of(0, emptyInput->size()))};
    for (const auto* slice : {&serialized, &emptySerialized}) {
      const auto* pos = slice->data();
      const auto header =
          serde::readSerializationHeader(pos, slice->data() + slice->size());
      EXPECT_TRUE(header.flags.requiredBarrier);
    }

    const auto schema =
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
    SchemaSerializer schemaSerializer;
    const auto deserializerSchema =
        SchemaDeserializer::deserialize(schemaSerializer.serialize(*schema));
    for (const auto* roundTripSchema :
         {schema.get(), deserializerSchema.get()}) {
      const auto& hybridMap =
          roundTripSchema->asRow().childAt(0)->asHybridFlatMap();
      ASSERT_EQ(hybridMap.groupCount(), 1);
      EXPECT_EQ(hybridMap.groupAt(0).groupId, HybridFlatMap::kDefaultGroupId);
      EXPECT_EQ(hybridMap.groupAt(0).groupKeys, expectedGroupKeys);
    }

    Deserializer deserializer{deserializerSchema, pool_.get()};
    velox::VectorPtr output;
    deserializer.deserialize(serialized, output);
    velox::test::assertEqualVectors(input, output);
    deserializer.deserialize(emptySerialized, output);
    velox::test::assertEqualVectors(emptyInput, output);

    for (const auto& selection : selections) {
      SCOPED_TRACE(selection.subfield);
      std::vector<Deserializer::Subfield> subfields;
      subfields.emplace_back(selection.subfield);
      Deserializer selectedDeserializer{
          deserializerSchema, subfields, pool_.get(), DeserializerOptions{}};
      selectedDeserializer.deserialize(serialized, output);
      velox::test::assertEqualVectors(
          makeInput(selection.expectedMaps), output);
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(DeserializerTest, acceptsRegularFlatMapSchema) {
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

  EXPECT_NO_THROW(Deserializer(schema, pool_.get()));
}

TEST_F(DeserializerTest, traversesComplexHybridValueSubtrees) {
  SchemaBuilder builder;
  auto root = builder.createRowTypeBuilder(1);
  auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::Int32);
  hybridMap->addGroup(0, {"1"}, makeComplexValueType(builder));
  hybridMap->addGroup(
      HybridFlatMap::kDefaultGroupId, {}, makeComplexValueType(builder));
  root->addChild("features", hybridMap);
  const auto schema = SchemaReader::getSchema(builder.schemaNodes());

  EXPECT_NO_THROW(Deserializer(schema, pool_.get()));
}

TEST_F(DeserializerTest, hybridFlatMapOmitsConstantGroupMetadata) {
  velox::test::VectorMaker vectorMaker{pool_.get()};
  const auto makeInput = [&](const std::vector<std::string>& maps) {
    return vectorMaker.rowVector(
        {"features"}, {vectorMaker.mapVectorFromJson<int64_t, double>(maps)});
  };

  struct Batch {
    std::string_view name;
    std::vector<std::string> maps;
    bool keyPresenceExpected;
    bool inMapExpected;
    bool valueExpected;
    std::string serialized;
  };
  std::vector<Batch> batches{
      {
          .name = "allPresent",
          .maps = {"{1: 10, 2: 20}", "{1: 11, 2: 21}"},
          .keyPresenceExpected = false,
          .inMapExpected = true,
          .valueExpected = true,
          .serialized = {},
      },
      {
          .name = "presentKeySubset",
          .maps = {"{1: 10}", "{1: 11}"},
          .keyPresenceExpected = true,
          .inMapExpected = true,
          .valueExpected = true,
          .serialized = {},
      },
      {
          .name = "mixedRowPresence",
          .maps = {"{1: 10}", "{2: 20}"},
          .keyPresenceExpected = false,
          .inMapExpected = true,
          .valueExpected = true,
          .serialized = {},
      },
      {
          .name = "allAbsent",
          .maps = {"{}", "{}"},
          .keyPresenceExpected = false,
          .inMapExpected = false,
          .valueExpected = false,
          .serialized = {},
      },
      {
          .name = "allNullValues",
          .maps = {"{1: null, 2: null}", "{1: null, 2: null}"},
          .keyPresenceExpected = false,
          .inMapExpected = true,
          .valueExpected = true,
          .serialized = {},
      },
  };

  const auto type = makeInput(batches.front().maps)->type();
  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .hybridFlatMapColumns =
              {{"features", makeHybridFlatMap({{"1", "2"}})}},
          .skipConstantHybridFlatMapMetadataStreams = true,
      },
      type,
      pool_.get()};
  std::vector<std::string> expectedMaps;
  for (auto& batch : batches) {
    const auto input = makeInput(batch.maps);
    batch.serialized =
        serializer.serialize(input, OrderedRanges::of(0, input->size()));
    expectedMaps.insert(
        expectedMaps.end(), batch.maps.begin(), batch.maps.end());
  }

  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const auto& group = schema->asRow().childAt(0)->asHybridFlatMap().groupAt(0);
  const auto keyPresenceOffset = group.keyPresenceDescriptor.offset();
  const auto inMapOffset = group.inMapDescriptor.offset();
  const auto valueOffset =
      group.valueType->asScalar().scalarDescriptor().offset();
  for (const auto& batch : batches) {
    SCOPED_TRACE(batch.name);
    serde::StreamDataParser parser{pool_.get()};
    parser.initialize(batch.serialized);
    std::set<uint32_t> offsets;
    parser.iterateStreams(
        [&](uint32_t offset, std::string_view) { offsets.insert(offset); });
    EXPECT_EQ(offsets.contains(keyPresenceOffset), batch.keyPresenceExpected);
    EXPECT_EQ(offsets.contains(inMapOffset), batch.inMapExpected);
    EXPECT_EQ(offsets.contains(valueOffset), batch.valueExpected);
  }

  std::vector<std::string_view> serializedBatches;
  serializedBatches.reserve(batches.size());
  for (const auto& batch : batches) {
    serializedBatches.push_back(batch.serialized);
  }
  Deserializer deserializer{schema, pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(serializedBatches, output);
  velox::test::assertEqualVectors(makeInput(expectedMaps), output);
}

TEST_F(DeserializerTest, hybridFlatMapOmitsAllTrueKeyPresenceForFixedGroups) {
  velox::test::VectorMaker vectorMaker{pool_.get()};
  const auto makeInput = [&](const std::vector<std::string>& maps) {
    return vectorMaker.rowVector(
        {"features"}, {vectorMaker.mapVectorFromJson<int64_t, double>(maps)});
  };

  // The Default catalog grows from {99} to {99, 100} after the first batch is
  // serialized. Its first all-true key-presence stream must remain explicit so
  // key 100 is not inferred as present in that earlier batch.
  const std::array<std::vector<std::string>, 2> batchMaps{
      std::vector<std::string>{"{1: 10, 99: 90}"},
      std::vector<std::string>{"{1: 11, 100: 100}"},
  };
  const auto type = makeInput(batchMaps.front())->type();
  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .hybridFlatMapColumns = {{"features", makeHybridFlatMap({{"1"}})}},
          .skipConstantHybridFlatMapMetadataStreams = true,
      },
      type,
      pool_.get()};

  std::vector<std::string> serialized;
  std::vector<std::string> expectedMaps;
  for (const auto& maps : batchMaps) {
    const auto input = makeInput(maps);
    const auto encoded =
        serializer.serialize(input, OrderedRanges::of(0, input->size()));
    serialized.emplace_back(encoded.data(), encoded.size());
    expectedMaps.insert(expectedMaps.end(), maps.begin(), maps.end());
  }

  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const auto& hybridMap = schema->asRow().childAt(0)->asHybridFlatMap();
  const HybridFlatMapType::Group* fixedGroup{nullptr};
  const HybridFlatMapType::Group* defaultGroup{nullptr};
  for (size_t i = 0; i < hybridMap.groupCount(); ++i) {
    const auto& group = hybridMap.groupAt(i);
    if (HybridFlatMap::isDefaultGroup(group.groupId)) {
      defaultGroup = &group;
    } else {
      fixedGroup = &group;
    }
  }
  ASSERT_NE(fixedGroup, nullptr);
  ASSERT_NE(defaultGroup, nullptr);
  ASSERT_EQ(defaultGroup->groupKeys.size(), 2);

  const auto streamOffsets = [&](const std::string& batch) {
    serde::StreamDataParser parser{pool_.get()};
    parser.initialize(batch);
    std::set<uint32_t> offsets;
    parser.iterateStreams(
        [&](uint32_t offset, std::string_view) { offsets.insert(offset); });
    return offsets;
  };
  const auto firstBatchOffsets = streamOffsets(serialized[0]);
  EXPECT_FALSE(
      firstBatchOffsets.contains(fixedGroup->keyPresenceDescriptor.offset()));
  EXPECT_TRUE(
      firstBatchOffsets.contains(defaultGroup->keyPresenceDescriptor.offset()));
  EXPECT_TRUE(streamOffsets(serialized[1])
                  .contains(defaultGroup->keyPresenceDescriptor.offset()));

  std::vector<std::string_view> serializedBatches;
  serializedBatches.reserve(serialized.size());
  for (const auto& batch : serialized) {
    serializedBatches.push_back(batch);
  }
  Deserializer deserializer{schema, pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(serializedBatches, output);
  velox::test::assertEqualVectors(makeInput(expectedMaps), output);
}

TEST_F(DeserializerTest, hybridFlatMapOmitsAllFalseDefaultGroupMetadata) {
  velox::test::VectorMaker vectorMaker{pool_.get()};
  const auto makeInput = [&](const std::vector<std::string>& maps) {
    return vectorMaker.rowVector(
        {"features"}, {vectorMaker.mapVectorFromJson<int64_t, double>(maps)});
  };

  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .hybridFlatMapColumns = {{"features", makeHybridFlatMap({{"1"}})}},
          .skipConstantHybridFlatMapMetadataStreams = true,
      },
      makeInput({"{99: 90}"})->type(),
      pool_.get()};
  const std::array<std::vector<std::string>, 3> batchMaps{
      std::vector<std::string>{"{99: 90}"},
      std::vector<std::string>{"{1: 10}"},
      std::vector<std::string>{"{100: 100}"},
  };
  std::vector<std::string> serialized;
  std::vector<std::string> expectedMaps;
  serialized.reserve(batchMaps.size());
  for (size_t i = 0; i < batchMaps.size(); ++i) {
    SCOPED_TRACE(i);
    const auto input = makeInput(batchMaps[i]);
    const auto encoded =
        serializer.serialize(input, OrderedRanges::of(0, input->size()));
    serialized.emplace_back(encoded.data(), encoded.size());
    expectedMaps.insert(
        expectedMaps.end(), batchMaps[i].begin(), batchMaps[i].end());
  }

  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const auto& hybridMap = schema->asRow().childAt(0)->asHybridFlatMap();
  const HybridFlatMapType::Group* defaultGroup{nullptr};
  for (size_t i = 0; i < hybridMap.groupCount(); ++i) {
    const auto& group = hybridMap.groupAt(i);
    if (HybridFlatMap::isDefaultGroup(group.groupId)) {
      defaultGroup = &group;
      break;
    }
  }
  ASSERT_NE(defaultGroup, nullptr);
  const auto keyPresenceOffset = defaultGroup->keyPresenceDescriptor.offset();
  const auto inMapOffset = defaultGroup->inMapDescriptor.offset();
  const auto valueOffset =
      defaultGroup->valueType->asScalar().scalarDescriptor().offset();

  const auto streamOffsets = [&](const std::string& batch) {
    serde::StreamDataParser parser{pool_.get()};
    parser.initialize(batch);
    std::set<uint32_t> offsets;
    parser.iterateStreams(
        [&](uint32_t offset, std::string_view) { offsets.insert(offset); });
    return offsets;
  };
  EXPECT_TRUE(streamOffsets(serialized[0]).contains(keyPresenceOffset));
  EXPECT_TRUE(streamOffsets(serialized[0]).contains(inMapOffset));
  EXPECT_FALSE(streamOffsets(serialized[1]).contains(keyPresenceOffset));
  EXPECT_FALSE(streamOffsets(serialized[1]).contains(inMapOffset));
  EXPECT_FALSE(streamOffsets(serialized[1]).contains(valueOffset));

  std::vector<std::string_view> serializedBatches;
  serializedBatches.reserve(serialized.size());
  for (const auto& batch : serialized) {
    serializedBatches.push_back(batch);
  }
  Deserializer deserializer{schema, pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(serializedBatches, output);
  velox::test::assertEqualVectors(makeInput(expectedMaps), output);
}

TEST_F(DeserializerTest, hybridFlatMapMetadataIsRetainedByDefault) {
  velox::test::VectorMaker vectorMaker{pool_.get()};
  const auto input = vectorMaker.rowVector(
      {"features"},
      {vectorMaker.mapVectorFromJson<int64_t, double>(
          {"{1: 10, 2: 20}", "{1: 11, 2: 21}"})});
  Serializer serializer{
      SerializerOptions{
          .hybridFlatMapColumns =
              {{"features", makeHybridFlatMap({{"1", "2"}})}},
      },
      input->type(),
      pool_.get()};
  const auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const auto& group = schema->asRow().childAt(0)->asHybridFlatMap().groupAt(0);
  serde::StreamDataParser parser{pool_.get()};
  parser.initialize(serialized);
  std::set<uint32_t> offsets;
  parser.iterateStreams(
      [&](uint32_t offset, std::string_view) { offsets.insert(offset); });

  EXPECT_TRUE(offsets.contains(group.keyPresenceDescriptor.offset()));
  EXPECT_TRUE(offsets.contains(group.inMapDescriptor.offset()));
}

TEST_F(DeserializerTest, hybridFlatMapRoundTripsMixedDefault) {
  constexpr velox::vector_size_t kRows = 4;
  auto mapType = velox::MAP(velox::INTEGER(), velox::DOUBLE());
  auto keys = velox::BaseVector::create(velox::INTEGER(), 8, pool_.get());
  auto values = velox::BaseVector::create(velox::DOUBLE(), 8, pool_.get());
  const std::array<int32_t, 8> keyData{1, 99, 100, 2, 1, 3, 99, 100};
  const std::array<double, 8> valueData{
      1.0, 99.0, 100.0, 2.0, 10.0, 3.0, 999.0, 1000.0};
  for (velox::vector_size_t i = 0; i < keyData.size(); ++i) {
    keys->asFlatVector<int32_t>()->set(i, keyData[i]);
    values->asFlatVector<double>()->set(i, valueData[i]);
  }
  values->setNull(3, true);
  auto mapNulls =
      velox::AlignedBuffer::allocate<bool>(kRows, pool_.get(), true);
  velox::bits::setNull(mapNulls->asMutable<uint64_t>(), 1);
  auto map = std::make_shared<velox::MapVector>(
      pool_.get(),
      mapType,
      mapNulls,
      kRows,
      velox::allocateOffsets(kRows, pool_.get()),
      velox::allocateSizes(kRows, pool_.get()),
      keys,
      values);
  const std::array<velox::vector_size_t, kRows> mapOffsets{0, 3, 3, 5};
  const std::array<velox::vector_size_t, kRows> mapSizes{3, 0, 2, 3};
  std::copy(
      mapOffsets.begin(),
      mapOffsets.end(),
      map->mutableOffsets(kRows)->asMutable<velox::vector_size_t>());
  std::copy(
      mapSizes.begin(),
      mapSizes.end(),
      map->mutableSizes(kRows)->asMutable<velox::vector_size_t>());

  auto type = velox::ROW({{"features", mapType}});
  auto input = std::make_shared<velox::RowVector>(
      pool_.get(), type, nullptr, kRows, std::vector<velox::VectorPtr>{map});
  Serializer serializer{
      SerializerOptions{
          .version = SerializationVersion::kSerialization,
          .hybridFlatMapColumns =
              {{"features", makeHybridFlatMap({{"1", "2", "4"}, {"3"}})}},
          .skipConstantHybridFlatMapMetadataStreams = true,
      },
      type,
      pool_.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, kRows))};
  auto secondIndices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(1, pool_.get());
  secondIndices->asMutable<velox::vector_size_t>()[0] = 3;
  const auto secondInput = velox::BaseVector::wrapInDictionary(
      nullptr, std::move(secondIndices), 1, input);
  const std::string secondSerialized{
      serializer.serialize(secondInput, OrderedRanges::of(0, 1))};

  auto newDefaultKeys =
      velox::BaseVector::create(velox::INTEGER(), 1, pool_.get());
  auto newDefaultValues =
      velox::BaseVector::create(velox::DOUBLE(), 1, pool_.get());
  newDefaultKeys->asFlatVector<int32_t>()->set(0, 101);
  newDefaultValues->asFlatVector<double>()->set(0, 1010.0);
  auto newDefaultMap = std::make_shared<velox::MapVector>(
      pool_.get(),
      mapType,
      nullptr,
      1,
      velox::allocateOffsets(1, pool_.get()),
      velox::allocateSizes(1, pool_.get()),
      newDefaultKeys,
      newDefaultValues);
  newDefaultMap->mutableOffsets(1)->asMutable<velox::vector_size_t>()[0] = 0;
  newDefaultMap->mutableSizes(1)->asMutable<velox::vector_size_t>()[0] = 1;
  auto newDefaultInput = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      1,
      std::vector<velox::VectorPtr>{newDefaultMap});
  const std::string newDefaultSerialized{
      serializer.serialize(newDefaultInput, OrderedRanges::of(0, 1))};

  // Every serialized Hybrid FlatMap slice requires an independent decode run,
  // including later calls on the same Serializer.
  for (const auto* slice :
       {&serialized, &newDefaultSerialized, &secondSerialized}) {
    const auto* pos = slice->data();
    const auto header =
        serde::readSerializationHeader(pos, slice->data() + slice->size());
    EXPECT_TRUE(header.flags.requiredBarrier);
  }

  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const auto& hybridMap = schema->asRow().childAt(0)->asHybridFlatMap();
  ASSERT_EQ(hybridMap.groupCount(), 3);
  EXPECT_TRUE(hybridMap.valueType().isScalar());
  EXPECT_EQ(hybridMap.groupAt(0).groupId, 0);
  EXPECT_EQ(
      hybridMap.groupAt(0).groupKeys,
      (std::vector<std::string>{"1", "2", "4"}));
  EXPECT_EQ(hybridMap.groupAt(1).groupId, 1);
  EXPECT_EQ(hybridMap.groupAt(1).groupKeys, (std::vector<std::string>{"3"}));
  EXPECT_EQ(hybridMap.groupAt(2).groupId, HybridFlatMap::kDefaultGroupId);
  EXPECT_EQ(
      hybridMap.groupAt(2).groupKeys,
      (std::vector<std::string>{"99", "100", "101"}));

  SchemaSerializer schemaSerializer;
  const auto deserializerSchema =
      SchemaDeserializer::deserialize(schemaSerializer.serialize(*schema));
  Deserializer deserializer{deserializerSchema, pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);
  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    EXPECT_TRUE(input->equalValueAt(output.get(), row, row))
        << "row " << row << " expected " << input->toString(row) << " actual "
        << output->toString(row);
  }

  deserializer.deserialize(newDefaultSerialized, output);
  ASSERT_EQ(output->size(), newDefaultInput->size());
  EXPECT_TRUE(newDefaultInput->equalValueAt(output.get(), 0, 0));

  std::vector<Deserializer::Subfield> defaultSubfields;
  defaultSubfields.emplace_back("features[99]");
  Deserializer defaultDeserializer{
      deserializerSchema, defaultSubfields, pool_.get(), DeserializerOptions{}};
  defaultDeserializer.deserialize(serialized, output);
  const auto* defaultMap =
      output->as<velox::RowVector>()->childAt(0)->as<velox::MapVector>();
  ASSERT_NE(defaultMap, nullptr);
  const std::array<std::map<int32_t, double>, kRows> expectedDefault{
      std::map<int32_t, double>{{99, 99.0}},
      std::map<int32_t, double>{},
      std::map<int32_t, double>{},
      std::map<int32_t, double>{{99, 999.0}},
  };
  const auto expectDefaultRows = [&](const velox::MapVector& actualMap) {
    const auto* actualKeys =
        actualMap.mapKeys()->as<velox::FlatVector<int32_t>>();
    const auto* actualValues =
        actualMap.mapValues()->as<velox::FlatVector<double>>();
    ASSERT_NE(actualKeys, nullptr);
    ASSERT_NE(actualValues, nullptr);
    for (velox::vector_size_t row = 0; row < kRows; ++row) {
      std::map<int32_t, double> actual;
      for (velox::vector_size_t entry = 0; entry < actualMap.sizeAt(row);
           ++entry) {
        const auto index = actualMap.offsetAt(row) + entry;
        actual.emplace(
            actualKeys->valueAt(index), actualValues->valueAt(index));
      }
      EXPECT_EQ(actual, expectedDefault[row]);
    }
  };
  expectDefaultRows(*defaultMap);

  std::vector<Deserializer::Subfield> unknownSubfields;
  unknownSubfields.emplace_back("features[102]");
  Deserializer unknownDeserializer{
      deserializerSchema, unknownSubfields, pool_.get(), DeserializerOptions{}};
  unknownDeserializer.deserialize(serialized, output);
  const auto* unknownMap =
      output->as<velox::RowVector>()->childAt(0)->as<velox::MapVector>();
  ASSERT_NE(unknownMap, nullptr);
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    EXPECT_EQ(unknownMap->sizeAt(row), 0);
  }
  unknownDeserializer.deserialize(newDefaultSerialized, output);
  ASSERT_EQ(output->size(), newDefaultInput->size());
  EXPECT_EQ(
      output->as<velox::RowVector>()
          ->childAt(0)
          ->as<velox::MapVector>()
          ->sizeAt(0),
      0);

  deserializer.deserialize(
      std::vector<std::string_view>{
          serialized, newDefaultSerialized, secondSerialized},
      output);
  // Every batch decodes against the final accumulated schema. Earlier
  // key-presence bitmaps omit later trailing Default keys.
  ASSERT_EQ(
      output->size(),
      input->size() + newDefaultInput->size() + secondInput->size());
  for (velox::vector_size_t row = 0; row < output->size(); ++row) {
    if (row < kRows) {
      EXPECT_TRUE(input->equalValueAt(output.get(), row, row))
          << "row " << row << " expected " << input->toString(row) << " actual "
          << output->toString(row);
    } else if (row == kRows) {
      EXPECT_TRUE(newDefaultInput->equalValueAt(output.get(), 0, row));
    } else {
      EXPECT_TRUE(secondInput->equalValueAt(output.get(), 0, row))
          << "row " << row << " expected " << secondInput->toString(0)
          << " actual " << output->toString(row);
    }
  }

  // A strict interior range leaves decoded group rows unmaterialized. Reset
  // discards them without requiring a trailing skip of the parent reader.
  deserializer.deserialize(serialized, std::vector<RowRange>{{1, 3}}, output);
  ASSERT_EQ(output->size(), 2);
  EXPECT_TRUE(input->equalValueAt(output.get(), 1, 0));
  EXPECT_TRUE(input->equalValueAt(output.get(), 2, 1));
}

TEST_F(DeserializerTest, hybridFlatMapRoundTripsDefaultOnly) {
  // The rows cover a non-empty map, a null map, an empty map and a null value,
  // and one key repeats across rows.
  verifyDefaultOnlyRoundTrip<int64_t>(
      {"{1: 10, 9: 90}", "null", "{}", "{2: null, 1: 11}"},
      {"9", "1", "2"},
      {
          {.subfield = "features[1]",
           .expectedMaps = {"{1: 10}", "null", "{}", "{1: 11}"}},
          {.subfield = "features[99]",
           .expectedMaps = {"{}", "null", "{}", "{}"}},
      });
  verifyDefaultOnlyRoundTrip<std::string>(
      {R"({"a": 10, "i": 90})", "null", "{}", R"({"b": null, "a": 11})"},
      {"i", "a", "b"},
      {
          {.subfield = R"(features["a"])",
           .expectedMaps = {R"({"a": 10})", "null", "{}", R"({"a": 11})"}},
          {.subfield = R"(features["missing"])",
           .expectedMaps = {"{}", "null", "{}", "{}"}},
      });
}

} // namespace
} // namespace facebook::nimble
