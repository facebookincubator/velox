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
#include <folly/Random.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <string_view>
#include <thread>

#include "folly/container/F14Set.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/serializer/Deserializer.h"
#include "velox/dwio/nimble/serializer/SerializationHeader.h"
#include "velox/dwio/nimble/serializer/Serializer.h"
#include "velox/dwio/nimble/serializer/StreamDataParser.h"
#include "velox/dwio/nimble/serializer/StreamDataWriter.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/dwio/nimble/writer/Writer.h"

#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook;
using namespace facebook::nimble;
using facebook::nimble::test::makeTestTabletOptions;

// Test parameters for parameterized tests.
// For kSerialization mode, we test both with and without compression to
// exercise the compressionOptions path in ReplayedEncodingSelectionPolicy.
struct TestParams {
  std::optional<SerializationVersion> version;
  // Compression options used with encodingLayoutTree.
  // nullopt (default) means no compression.
  std::optional<CompressionOptions> compressionOptions{};
  // Encoding type for stream sizes in compact trailer.
  EncodingType streamSizesEncodingType{EncodingType::Trivial};
  // Max cached buffers per per-stream BufferPool. 0 disables pooling.
  size_t bufferPoolCapacity{velox::BufferPool::kDefaultCapacity};
};

class SerializationTest : public ::testing::TestWithParam<TestParams> {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("default_root");
    pool_ = velox::memory::memoryManager()->addLeafPool("default_leaf");
  }

  std::optional<SerializationVersion> version() const {
    return GetParam().version;
  }

  bool hasHeader() const {
    return version().has_value();
  }

  EncodingType streamSizesEncodingType() const {
    return GetParam().streamSizesEncodingType;
  }

  const std::optional<CompressionOptions>& compressionOptions() const {
    return GetParam().compressionOptions;
  }

  size_t bufferPoolCapacity() const {
    return GetParam().bufferPoolCapacity;
  }

  DeserializerOptions deserializerOptions() const {
    return DeserializerOptions{
        .hasHeader = hasHeader(),
        .bufferPoolCapacity = bufferPoolCapacity(),
    };
  }

  // Result of serializing input vectors.
  struct SerializeResult {
    // One serialized buffer per input vector (or per stripe for kTablet).
    std::vector<std::string> serialized;
    // Nimble schema for deserialization.
    std::shared_ptr<const Type> schema;
    // Populated only for kTablet test assemblies.
    std::vector<size_t> numTabletStreams;
    std::vector<size_t> numUniqueTabletStreams;
    std::vector<uint64_t> tabletBodyBytes;
    std::vector<uint64_t> tabletUniqueBodyBytes;
  };

  // Serializes input vectors using the current test parameter's version.
  // For kTablet, writes a Nimble file and assembles kTablet buffers.
  // For other versions, uses the Serializer directly.
  SerializeResult serialize(
      const velox::TypePtr& type,
      const std::vector<velox::VectorPtr>& inputs);

  // Writes vectors to a Nimble file, reads raw tablet streams, and assembles
  // kTablet buffers. Each input vector becomes part of a single-stripe file.
  // When enableChunking is true, streams include chunk headers (the actual
  // kTablet format); when false, streams are raw encoded data without chunk
  // headers.
  SerializeResult serializeTablet(
      const velox::TypePtr& type,
      const std::vector<velox::VectorPtr>& inputs,
      bool enableChunking);

  SerializeResult serializeTablet(
      const velox::TypePtr& type,
      const std::vector<velox::VectorPtr>& inputs,
      bool enableChunking,
      EncodingType streamIdsEncodingType,
      EncodingType sizeIndicesEncodingType,
      EncodingType uniqueSizesEncodingType,
      bool forceDuplicateFirstTwoStreams = false);

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;

  static bool vectorEquals(
      const velox::VectorPtr& expected,
      const velox::VectorPtr& actual,
      velox::vector_size_t index) {
    return expected->equalValueAt(actual.get(), index, index);
  }

  /// Builds a Velox RowType from a nimble schema, converting FlatMap columns to
  /// ROW types so the deserializer reads them as structs.
  static velox::RowTypePtr buildOutputTypeForFlatMapAsStruct(
      const Type& schema) {
    const auto& root = schema.asRow();
    std::vector<std::string> names;
    std::vector<velox::TypePtr> types;
    names.reserve(root.childrenCount());
    types.reserve(root.childrenCount());
    for (size_t i = 0; i < root.childrenCount(); ++i) {
      names.push_back(root.nameAt(i));
      const auto* child = root.childAt(i).get();
      if (child->isFlatMap()) {
        const auto& flatMap = child->asFlatMap();
        std::vector<std::string> fieldNames;
        std::vector<velox::TypePtr> fieldTypes;
        fieldNames.reserve(flatMap.childrenCount());
        fieldTypes.reserve(flatMap.childrenCount());
        for (size_t j = 0; j < flatMap.childrenCount(); ++j) {
          fieldNames.push_back(flatMap.nameAt(j));
          fieldTypes.push_back(convertToVeloxType(*flatMap.childAt(j)));
        }
        types.push_back(
            std::make_shared<const velox::RowType>(
                std::move(fieldNames), std::move(fieldTypes)));
      } else {
        types.push_back(convertToVeloxType(*child));
      }
    }
    return std::make_shared<const velox::RowType>(
        std::move(names), std::move(types));
  }

  /// Verifies that flatmap data serialized as map can be correctly deserialized
  /// as struct (ROW). For each flatmap column, the struct fields should match
  /// the corresponding map key values. Rows where a key is absent should have
  /// null struct field values.
  void verifyFlatMapAsStruct(
      const Serializer& serializer,
      const std::vector<std::string>& serializedData,
      const std::vector<velox::VectorPtr>& inputs) {
    auto schema =
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
    auto outputType = buildOutputTypeForFlatMapAsStruct(*schema);

    auto opts = deserializerOptions();
    opts.outputType = outputType;
    Deserializer structDeserializer{schema, pool_.get(), opts};

    velox::VectorPtr structOutput;
    for (size_t i = 0; i < inputs.size(); ++i) {
      SCOPED_TRACE(fmt::format("flatMapAsStruct batch {}", i));
      structDeserializer.deserialize(serializedData[i], structOutput);
      ASSERT_EQ(structOutput->size(), inputs[i]->size());

      auto* outputRow = structOutput->as<velox::RowVector>();
      auto* inputRow = inputs[i]->as<velox::RowVector>();
      ASSERT_NE(outputRow, nullptr);
      ASSERT_NE(inputRow, nullptr);

      // For each column, verify struct fields match map values.
      for (size_t col = 0; col < outputRow->childrenSize(); ++col) {
        if (outputType->childAt(col)->kind() != velox::TypeKind::ROW) {
          // Non-flatmap column: direct comparison.
          for (velox::vector_size_t row = 0; row < inputRow->size(); ++row) {
            ASSERT_TRUE(outputRow->childAt(col)->equalValueAt(
                inputRow->childAt(col).get(), row, row))
                << "Non-flatmap column " << col << " mismatch at row " << row;
          }
          continue;
        }

        // FlatMap column read as struct: verify each struct field matches the
        // corresponding map key value.
        auto* structVec = outputRow->childAt(col)->as<velox::RowVector>();
        auto* mapVec = inputRow->childAt(col)->as<velox::MapVector>();
        ASSERT_NE(structVec, nullptr);
        ASSERT_NE(mapVec, nullptr);

        const auto& structType = outputType->childAt(col)->asRow();
        for (size_t field = 0; field < structType.size(); ++field) {
          const auto& keyName = structType.nameOf(field);
          auto* fieldVec = structVec->childAt(field).get();

          for (velox::vector_size_t row = 0; row < mapVec->size(); ++row) {
            // Find the key in the map for this row.
            bool found = false;
            auto mapOffset = mapVec->offsetAt(row);
            auto mapSize = mapVec->sizeAt(row);
            for (velox::vector_size_t entry = 0; entry < mapSize; ++entry) {
              auto keyIdx = mapOffset + entry;
              std::string mapKey;
              auto* mapKeys = mapVec->mapKeys().get();
              if (mapKeys->type()->kind() == velox::TypeKind::INTEGER) {
                mapKey = std::to_string(
                    mapKeys->asFlatVector<int32_t>()->valueAt(keyIdx));
              } else if (mapKeys->type()->kind() == velox::TypeKind::VARCHAR) {
                mapKey = std::string(
                    mapKeys->asFlatVector<velox::StringView>()->valueAt(
                        keyIdx));
              } else {
                FAIL() << "Unsupported map key type: "
                       << mapKeys->type()->toString();
              }

              if (mapKey == keyName) {
                found = true;
                ASSERT_TRUE(fieldVec->equalValueAt(
                    mapVec->mapValues().get(), row, keyIdx))
                    << "FlatMap column " << col << " key " << keyName
                    << " value mismatch at row " << row;
                break;
              }
            }
            if (!found) {
              ASSERT_TRUE(fieldVec->isNullAt(row))
                  << "FlatMap column " << col << " key " << keyName
                  << " should be null at row " << row;
            }
          }
        }
      }
    }
  }

  template <typename T = int32_t>
  void writeAndVerify(
      folly::detail::DefaultGenerator& rng,
      velox::memory::MemoryPool* pool,
      const velox::TypePtr& type,
      std::function<velox::VectorPtr(const velox::TypePtr&)> generator,
      std::function<bool(
          const velox::VectorPtr&,
          const velox::VectorPtr&,
          velox::vector_size_t)> validator,
      const folly::F14FastMap<std::string, std::set<std::string>>&
          flatMapColumns,
      size_t count) {
    SerializerOptions options{
        .compressionType = CompressionType::Zstd,
        .compressionThreshold = 32,
        .compressionLevel = 3,
        .version = version(),
        .flatMapColumns = flatMapColumns,
        .streamIndicesEncodingType = streamSizesEncodingType(),
        .streamSizesEncodingType = streamSizesEncodingType(),
    };
    Serializer serializer{options, type, pool};

    velox::VectorPtr output;
    velox::VectorPtr expected;
    std::vector<std::string> serialized;
    for (auto i = 0; i < count; ++i) {
      auto input = generator(type);
      serialized.emplace_back(
          serializer.serialize(input, OrderedRanges::of(0, input->size())));
      if (!expected) {
        expected = input;
      } else {
        auto oldSize = expected->size();
        expected->resize(oldSize + input->size());
        expected->copy(input.get(), oldSize, 0, input->size());
      }
      // FlatMap output type construction needs the final schema after key
      // discovery across all serialized batches.
      if (flatMapColumns.empty() && i < count - 1 &&
          folly::Random::oneIn(3, rng)) {
        velox::BaseVector::ensureWritable(
            velox::SelectivityVector::empty(),
            expected->type(),
            expected->pool(),
            expected);
        continue;
      }
      std::vector<std::string_view> serializedSVs;
      serializedSVs.reserve(serialized.size());
      for (auto& s : serialized) {
        serializedSVs.push_back(s);
      }
      Deserializer deserializer{
          SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
          pool,
          deserializerOptions()};
      deserializer.deserialize(serializedSVs, output);

      ASSERT_EQ(output->size(), expected->size());
      for (auto j = 0; j < expected->size(); ++j) {
        ASSERT_TRUE(validator(output, expected, j))
            << "Content mismatch at index " << j
            << "\nReference: " << expected->toString(j)
            << "\nResult: " << output->toString(j);
      }
      expected.reset();
      serialized.clear();
    }
  }
};

TEST_P(SerializationTest, fuzzSimple) {
  auto type = velox::ROW({
      {"bool_val", velox::BOOLEAN()},
      {"byte_val", velox::TINYINT()},
      {"short_val", velox::SMALLINT()},
      {"int_val", velox::INTEGER()},
      {"long_val", velox::BIGINT()},
      {"float_val", velox::REAL()},
      {"double_val", velox::DOUBLE()},
      {"string_val", velox::VARCHAR()},
      {"binary_val", velox::VARBINARY()},
      // {"ts_val", velox::TIMESTAMP()},
  });
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;

  // Small batches creates more edge cases.
  size_t batchSize = 10;
  velox::VectorFuzzer noNulls(
      {
          .vectorSize = batchSize,
          .nullRatio = 0,
          .stringLength = 20,
          .stringVariableLength = true,
      },
      pool_.get(),
      seed);

  folly::detail::DefaultGenerator rng{seed};
  auto iterations = 20;
  auto batches = 20;
  for (auto i = 0; i < iterations; ++i) {
    writeAndVerify(
        rng,
        pool_.get(),
        type,
        [&](auto& type) {
          return noNulls.fuzzInputRow(
              std::dynamic_pointer_cast<const velox::RowType>(type));
        },
        vectorEquals,
        {},
        batches);
  }
}

TEST_P(SerializationTest, fuzzNullableStreams) {
  if (!hasSerializationHeaderFlags(version())) {
    GTEST_SKIP()
        << "Non-null-capable formats do not support Row/FlatMap null streams";
  }

  const char* seedEnv = std::getenv("FUZZ_SEED");
  auto seed = seedEnv != nullptr ? static_cast<uint32_t>(std::stoul(seedEnv))
                                 : folly::Random::rand32();
  LOG(INFO) << "fuzzNullableStreams seed: " << seed;

  const std::vector<velox::TypePtr> supportedScalarTypes = {
      velox::BOOLEAN(),
      velox::TINYINT(),
      velox::SMALLINT(),
      velox::INTEGER(),
      velox::BIGINT(),
      velox::REAL(),
      velox::DOUBLE(),
      velox::VARCHAR(),
      velox::VARBINARY(),
  };

  folly::detail::DefaultGenerator rng{seed};
  {
    constexpr velox::vector_size_t kRows = 30;
    const auto regularMapType = velox::MAP(velox::INTEGER(), velox::DOUBLE());
    const auto intFeaturesType = velox::MAP(velox::INTEGER(), velox::DOUBLE());
    const auto stringFeaturesType =
        velox::MAP(velox::VARCHAR(), velox::VARCHAR());
    const auto nestedRowType = velox::ROW({
        {"child",
         velox::ROW({
             {"grandchild",
              velox::ROW({
                  {"flag", velox::BOOLEAN()},
                  {"score", velox::DOUBLE()},
              })},
             {"labels", velox::ARRAY(velox::VARCHAR())},
         })},
    });
    auto type = velox::ROW({
        {"id", velox::BIGINT()},
        {"regular_map", regularMapType},
        {"int_features", intFeaturesType},
        {"string_features", stringFeaturesType},
        {"nested_row", nestedRowType},
    });
    velox::VectorFuzzer hasNulls(
        {
            .vectorSize = kRows,
            .nullRatio = 0.25,
            .useRandomNullPattern = true,
            .stringLength = 8,
            .stringVariableLength = false,
            .containerLength = 5,
            .containerVariableLength = true,
            .normalizeMapKeys = true,
        },
        pool_.get(),
        folly::Random::rand32(rng));
    auto makeIntFeatures = [&]() -> velox::VectorPtr {
      constexpr std::array<int32_t, 2> kKeys = {1, 2};
      constexpr velox::vector_size_t kKeysPerRow = kKeys.size();
      auto keys = velox::BaseVector::create(
          velox::INTEGER(), kRows * kKeysPerRow, pool_.get());
      auto values = velox::BaseVector::create(
          velox::DOUBLE(), kRows * kKeysPerRow, pool_.get());
      for (velox::vector_size_t row = 0; row < kRows; ++row) {
        for (velox::vector_size_t keyIndex = 0; keyIndex < kKeysPerRow;
             ++keyIndex) {
          const auto entry = row * kKeysPerRow + keyIndex;
          keys->asFlatVector<int32_t>()->set(entry, kKeys[keyIndex]);
          values->asFlatVector<double>()->set(entry, row * 10 + keyIndex);
          if ((row + keyIndex) % 7 == 0) {
            values->setNull(entry, true);
          }
        }
      }

      auto offsets = velox::allocateOffsets(kRows, pool_.get());
      auto sizes = velox::allocateSizes(kRows, pool_.get());
      auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
      auto* rawSizes = sizes->asMutable<velox::vector_size_t>();
      for (velox::vector_size_t row = 0; row < kRows; ++row) {
        rawOffsets[row] = row * kKeysPerRow;
        rawSizes[row] = kKeysPerRow;
      }
      auto map = std::make_shared<velox::MapVector>(
          pool_.get(),
          intFeaturesType,
          nullptr,
          kRows,
          offsets,
          sizes,
          keys,
          values);
      map->setNull(2, true);
      map->setNull(11, true);
      return map;
    };
    auto makeStringFeatures = [&]() -> velox::VectorPtr {
      constexpr std::array<const char*, 2> kKeys = {"a", "b"};
      constexpr std::array<const char*, 2> kValues = {"red", "blue"};
      constexpr velox::vector_size_t kKeysPerRow = kKeys.size();
      auto keys = velox::BaseVector::create(
          velox::VARCHAR(), kRows * kKeysPerRow, pool_.get());
      auto values = velox::BaseVector::create(
          velox::VARCHAR(), kRows * kKeysPerRow, pool_.get());
      for (velox::vector_size_t row = 0; row < kRows; ++row) {
        for (velox::vector_size_t keyIndex = 0; keyIndex < kKeysPerRow;
             ++keyIndex) {
          const auto entry = row * kKeysPerRow + keyIndex;
          keys->asFlatVector<velox::StringView>()->set(
              entry, velox::StringView(kKeys[keyIndex]));
          values->asFlatVector<velox::StringView>()->set(
              entry, velox::StringView(kValues[(row + keyIndex) % 2]));
          if ((row + keyIndex) % 11 == 0) {
            values->setNull(entry, true);
          }
        }
      }

      auto offsets = velox::allocateOffsets(kRows, pool_.get());
      auto sizes = velox::allocateSizes(kRows, pool_.get());
      auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
      auto* rawSizes = sizes->asMutable<velox::vector_size_t>();
      for (velox::vector_size_t row = 0; row < kRows; ++row) {
        rawOffsets[row] = row * kKeysPerRow;
        rawSizes[row] = kKeysPerRow;
      }
      auto map = std::make_shared<velox::MapVector>(
          pool_.get(),
          stringFeaturesType,
          nullptr,
          kRows,
          offsets,
          sizes,
          keys,
          values);
      map->setNull(5, true);
      map->setNull(17, true);
      return map;
    };
    auto generateDeterministicInput = [&]() -> velox::VectorPtr {
      auto ids = velox::BaseVector::create(velox::BIGINT(), kRows, pool_.get());
      for (velox::vector_size_t row = 0; row < kRows; ++row) {
        ids->asFlatVector<int64_t>()->set(row, row);
      }
      ids->setNull(3, true);

      auto input = std::make_shared<velox::RowVector>(
          pool_.get(),
          type,
          nullptr,
          kRows,
          std::vector<velox::VectorPtr>{
              ids,
              hasNulls.fuzz(regularMapType),
              makeIntFeatures(),
              makeStringFeatures(),
              hasNulls.fuzz(nestedRowType),
          });
      return input;
    };
    SCOPED_TRACE("deterministic regular map, FlatMap, and nested row");

    writeAndVerify(
        rng,
        pool_.get(),
        type,
        [&](const auto&) { return generateDeterministicInput(); },
        vectorEquals,
        {{"int_features", {}}, {"string_features", {}}},
        /*count=*/10);
  }

  {
    auto nestedType = velox::ROW({
        {"score", velox::BIGINT()},
        {"label", velox::VARCHAR()},
    });
    auto type = velox::ROW({
        {"id", velox::INTEGER()},
        {"nested", nestedType},
        {"events", velox::ARRAY(nestedType)},
    });

    SerializerOptions options{
        .compressionType = CompressionType::Zstd,
        .compressionThreshold = 32,
        .compressionLevel = 3,
        .version = version(),
        .streamIndicesEncodingType = streamSizesEncodingType(),
        .streamSizesEncodingType = streamSizesEncodingType(),
    };
    Serializer serializer{options, type, pool_.get()};
    Deserializer deserializer{
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
        pool_.get(),
        deserializerOptions()};

    constexpr int kMultiBatchIterations = 8;
    constexpr int kBatches = 6;
    for (int iteration = 0; iteration < kMultiBatchIterations; ++iteration) {
      SCOPED_TRACE(fmt::format("multi-batch iteration {}", iteration));
      std::vector<velox::VectorPtr> inputs;
      std::vector<std::string> serialized;
      inputs.reserve(kBatches);
      serialized.reserve(kBatches);

      for (int batch = 0; batch < kBatches; ++batch) {
        velox::VectorFuzzer fuzzer(
            {
                .vectorSize = 5,
                .nullRatio = batch % 2 == 0 ? 0.0 : 0.35,
                .useRandomNullPattern = true,
                .stringLength = 12,
                .stringVariableLength = true,
                .containerLength = 3,
                .containerVariableLength = true,
            },
            pool_.get(),
            folly::Random::rand32(rng));
        auto input = fuzzer.fuzzInputRow(
            std::dynamic_pointer_cast<const velox::RowType>(type));
        serialized.emplace_back(
            serializer.serialize(input, OrderedRanges::of(0, input->size())));
        inputs.emplace_back(std::move(input));
      }

      std::vector<std::string_view> serializedViews;
      serializedViews.reserve(serialized.size());
      for (const auto& data : serialized) {
        serializedViews.push_back(data);
      }

      auto expected = inputs.front();
      for (size_t i = 1; i < inputs.size(); ++i) {
        const auto oldSize = expected->size();
        expected->resize(oldSize + inputs[i]->size());
        expected->copy(inputs[i].get(), oldSize, 0, inputs[i]->size());
      }

      velox::VectorPtr output;
      deserializer.deserialize(serializedViews, output);

      ASSERT_EQ(output->size(), expected->size());
      for (velox::vector_size_t row = 0; row < expected->size(); ++row) {
        EXPECT_TRUE(vectorEquals(expected, output, row))
            << "Content mismatch at row " << row
            << "\nExpected: " << expected->toString(row)
            << "\nActual: " << output->toString(row);
      }
    }
  }

  constexpr int kIterations = 8;
  for (int i = 0; i < kIterations; ++i) {
    velox::VectorFuzzer hasNulls(
        {
            .vectorSize = 12,
            .nullRatio = 0.2,
            .useRandomNullPattern = true,
            .stringLength = 20,
            .stringVariableLength = true,
            .containerLength = 5,
            .containerVariableLength = true,
            .normalizeMapKeys = true,
        },
        pool_.get(),
        folly::Random::rand32(rng));
    auto type = hasNulls.randRowType(
        supportedScalarTypes,
        /*maxDepth=*/3);
    SCOPED_TRACE(fmt::format("iteration {}, type {}", i, type->toString()));

    writeAndVerify(
        rng,
        pool_.get(),
        type,
        [&](auto& type) {
          return hasNulls.fuzzInputRow(
              std::dynamic_pointer_cast<const velox::RowType>(type));
        },
        vectorEquals,
        {},
        /*count=*/4);
  }
}

TEST_P(SerializationTest, flatMapRejectsTopLevelNullRows) {
  if (!hasSerializationHeaderFlags(version())) {
    GTEST_SKIP()
        << "Non-null-capable formats do not support Row/FlatMap null streams";
  }

  constexpr velox::vector_size_t kRows = 4;
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  auto ids = velox::BaseVector::create(velox::BIGINT(), kRows, pool_.get());
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    ids->asFlatVector<int64_t>()->set(row, row);
  }

  auto offsets = velox::allocateOffsets(kRows, pool_.get());
  auto sizes = velox::allocateSizes(kRows, pool_.get());
  auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
  auto* rawSizes = sizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    rawOffsets[row] = row;
    rawSizes[row] = 1;
  }
  auto keys = velox::BaseVector::create(velox::INTEGER(), kRows, pool_.get());
  auto values = velox::BaseVector::create(velox::DOUBLE(), kRows, pool_.get());
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    keys->asFlatVector<int32_t>()->set(row, 10 + row);
    values->asFlatVector<double>()->set(row, row * 1.5);
  }
  auto features = std::make_shared<velox::MapVector>(
      pool_.get(),
      type->childAt(1),
      nullptr,
      kRows,
      offsets,
      sizes,
      keys,
      values);

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      kRows,
      std::vector<velox::VectorPtr>{ids, features});
  input->setNull(2, true);

  SerializerOptions options{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"features", {}}},
      .streamIndicesEncodingType = streamSizesEncodingType(),
      .streamSizesEncodingType = streamSizesEncodingType(),
  };
  Serializer serializer{options, type, pool_.get()};

  NIMBLE_ASSERT_THROW(
      serializer.serialize(input, OrderedRanges::of(0, kRows)),
      "Top-level row nulls are not supported when serializing FlatMap columns.");
}

TEST_P(SerializationTest, flatMapWithNestedNullsRoundTrips) {
  if (!hasSerializationHeaderFlags(version())) {
    GTEST_SKIP()
        << "Non-null-capable formats do not support Row/FlatMap null streams";
  }

  constexpr velox::vector_size_t kRows = 4;
  constexpr velox::vector_size_t kKeysPerRow = 2;
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
      {"nested", velox::ROW({{"score", velox::DOUBLE()}})},
  });

  auto ids = velox::BaseVector::create(velox::BIGINT(), kRows, pool_.get());
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    ids->asFlatVector<int64_t>()->set(row, row);
  }

  auto offsets = velox::allocateOffsets(kRows, pool_.get());
  auto sizes = velox::allocateSizes(kRows, pool_.get());
  auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
  auto* rawSizes = sizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    rawOffsets[row] = row * kKeysPerRow;
    rawSizes[row] = kKeysPerRow;
  }

  auto keys = velox::BaseVector::create(
      velox::INTEGER(), kRows * kKeysPerRow, pool_.get());
  auto values = velox::BaseVector::create(
      velox::DOUBLE(), kRows * kKeysPerRow, pool_.get());
  for (velox::vector_size_t entry = 0; entry < kRows * kKeysPerRow; ++entry) {
    keys->asFlatVector<int32_t>()->set(entry, entry % kKeysPerRow);
    values->asFlatVector<double>()->set(entry, entry * 1.25);
  }
  values->setNull(3, true);
  auto features = std::make_shared<velox::MapVector>(
      pool_.get(),
      type->childAt(1),
      nullptr,
      kRows,
      offsets,
      sizes,
      keys,
      values);
  features->setNull(1, true);

  auto scores = velox::BaseVector::create(velox::DOUBLE(), kRows, pool_.get());
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    scores->asFlatVector<double>()->set(row, row * 2.0);
  }
  auto nested = std::make_shared<velox::RowVector>(
      pool_.get(),
      type->childAt(2),
      nullptr,
      kRows,
      std::vector<velox::VectorPtr>{scores});
  nested->setNull(2, true);

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      kRows,
      std::vector<velox::VectorPtr>{ids, features, nested});

  SerializerOptions options{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"features", {}}},
      .streamIndicesEncodingType = streamSizesEncodingType(),
      .streamSizesEncodingType = streamSizesEncodingType(),
  };
  Serializer serializer{options, type, pool_.get()};
  auto serialized = std::string(
      serializer.serialize(input, OrderedRanges::of(0, input->size())));

  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t row = 0; row < input->size(); ++row) {
    ASSERT_TRUE(vectorEquals(output, input, row))
        << "Content mismatch at row " << row
        << "\nReference: " << input->toString(row)
        << "\nResult: " << output->toString(row);
  }
}

namespace {

// References one logical stream body while assembling a kTablet test blob.
struct TabletStream {
  uint32_t id{0};
  std::string_view data;
};

// Reports whether kTablet assembly found exact-duplicate stream bodies.
struct TabletTrailerWriteResult {
  size_t numStreams{0};
  size_t numUniqueStreams{0};
  uint64_t numBodyBytes{0};
  uint64_t numUniqueBodyBytes{0};

  bool hasSharedStreamBodies() const {
    return numUniqueStreams < numStreams;
  }
};

std::vector<TabletStream> collectTabletStreams(
    const std::vector<uint32_t>& streamOffsets,
    const std::vector<std::unique_ptr<StreamLoader>>& streamLoaders) {
  std::vector<TabletStream> streams;
  streams.reserve(streamLoaders.size());
  for (size_t i = 0; i < streamLoaders.size(); ++i) {
    if (streamLoaders[i] == nullptr) {
      continue;
    }
    streams.push_back({streamOffsets[i], streamLoaders[i]->getStream()});
  }
  return streams;
}

TabletTrailerWriteResult writeTabletStreamsAndTrailer(
    const std::vector<TabletStream>& streams,
    EncodingType streamIdsEncodingType,
    EncodingType sizeIndicesEncodingType,
    EncodingType uniqueSizesEncodingType,
    std::string& buffer) {
  TabletTrailerWriteResult result;
  std::vector<uint32_t> streamIds;
  std::vector<uint32_t> streamSizeIndices;
  std::vector<uint32_t> uniqueStreamSizes;
  std::vector<std::string_view> uniqueStreams;
  streamIds.reserve(streams.size());
  streamSizeIndices.reserve(streams.size());
  uniqueStreams.reserve(streams.size());
  uniqueStreamSizes.reserve(streams.size());

  for (const auto& stream : streams) {
    if (stream.data.empty()) {
      continue;
    }
    streamIds.emplace_back(stream.id);
    result.numBodyBytes += stream.data.size();

    size_t uniqueStreamIndex{0};
    for (; uniqueStreamIndex < uniqueStreams.size(); ++uniqueStreamIndex) {
      if (uniqueStreams[uniqueStreamIndex] == stream.data) {
        break;
      }
    }

    if (uniqueStreamIndex == uniqueStreams.size()) {
      uniqueStreams.emplace_back(stream.data);
      uniqueStreamSizes.emplace_back(static_cast<uint32_t>(stream.data.size()));
      result.numUniqueBodyBytes += stream.data.size();
      buffer.append(stream.data.data(), stream.data.size());
    }
    streamSizeIndices.emplace_back(static_cast<uint32_t>(uniqueStreamIndex));
  }

  result.numStreams = streamIds.size();
  result.numUniqueStreams = uniqueStreams.size();
  serde::detail::writeTrailer(
      streamIds,
      streamSizeIndices,
      uniqueStreamSizes,
      streamIdsEncodingType,
      sizeIndicesEncodingType,
      uniqueSizesEncodingType,
      buffer);
  return result;
}

void collectStreamOffsets(
    const nimble::Type& type,
    std::set<uint32_t>& offsets) {
  switch (type.kind()) {
    case nimble::Kind::Scalar:
      offsets.insert(type.asScalar().scalarDescriptor().offset());
      break;
    case nimble::Kind::TimestampMicroNano: {
      const auto& ts = type.asTimestampMicroNano();
      offsets.insert(ts.microsDescriptor().offset());
      offsets.insert(ts.nanosDescriptor().offset());
      break;
    }
    case nimble::Kind::Row: {
      const auto& row = type.asRow();
      offsets.insert(row.nullsDescriptor().offset());
      for (size_t i = 0; i < row.childrenCount(); ++i) {
        collectStreamOffsets(*row.childAt(i), offsets);
      }
      break;
    }
    case nimble::Kind::Array:
      offsets.insert(type.asArray().lengthsDescriptor().offset());
      collectStreamOffsets(*type.asArray().elements(), offsets);
      break;
    case nimble::Kind::ArrayWithOffsets: {
      const auto& arr = type.asArrayWithOffsets();
      offsets.insert(arr.offsetsDescriptor().offset());
      offsets.insert(arr.lengthsDescriptor().offset());
      collectStreamOffsets(*arr.elements(), offsets);
      break;
    }
    case nimble::Kind::Map: {
      const auto& map = type.asMap();
      offsets.insert(map.lengthsDescriptor().offset());
      collectStreamOffsets(*map.keys(), offsets);
      collectStreamOffsets(*map.values(), offsets);
      break;
    }
    case nimble::Kind::FlatMap: {
      const auto& flatMap = type.asFlatMap();
      offsets.insert(flatMap.nullsDescriptor().offset());
      for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
        offsets.insert(flatMap.inMapDescriptorAt(i).offset());
        collectStreamOffsets(*flatMap.childAt(i), offsets);
      }
      break;
    }
    case nimble::Kind::SlidingWindowMap: {
      const auto& map = type.asSlidingWindowMap();
      offsets.insert(map.offsetsDescriptor().offset());
      offsets.insert(map.lengthsDescriptor().offset());
      collectStreamOffsets(*map.keys(), offsets);
      collectStreamOffsets(*map.values(), offsets);
      break;
    }
  }
}

std::string formatName(const ::testing::TestParamInfo<TestParams>& info) {
  std::string name;
  if (!info.param.version.has_value()) {
    name = "LegacyFormat";
  } else {
    switch (*info.param.version) {
      case SerializationVersion::kLegacy:
        name = "LegacyFormat";
        break;
      case SerializationVersion::kLegacyCompact:
        name = "CompactRawFormat";
        break;
      case SerializationVersion::kTablet:
        name = "TabletFormat";
        break;
      case SerializationVersion::kSerialization:
        name = "SerializationFormat";
        break;
      case SerializationVersion::kProjection:
        name = "ProjectionFormat";
        break;
      case SerializationVersion::kLegacySerialization:
        name = "LegacySerializationFormat";
        break;
    }
  }
  // Add compression suffix for encoding modes.
  if (info.param.compressionOptions.has_value()) {
    name += "_Compressed";
  }
  if (info.param.streamSizesEncodingType != EncodingType::Trivial) {
    name += "_" + toString(info.param.streamSizesEncodingType) + "StreamSizes";
  }
  if (info.param.bufferPoolCapacity == 0) {
    name += "_NoBufferPool";
  } else if (
      info.param.bufferPoolCapacity != velox::BufferPool::kDefaultCapacity) {
    name += "_BufferPool" + std::to_string(info.param.bufferPoolCapacity);
  }
  return name;
}

} // namespace

SerializationTest::SerializeResult SerializationTest::serialize(
    const velox::TypePtr& type,
    const std::vector<velox::VectorPtr>& inputs) {
  if (version() == SerializationVersion::kTablet) {
    return serializeTablet(type, inputs, /*enableChunking=*/true);
  }

  SerializerOptions options{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .streamIndicesEncodingType = streamSizesEncodingType(),
      .streamSizesEncodingType = streamSizesEncodingType(),
  };
  Serializer serializer{options, type, pool_.get()};

  SerializeResult result;
  for (const auto& input : inputs) {
    result.serialized.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
  }
  result.schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  return result;
}

SerializationTest::SerializeResult SerializationTest::serializeTablet(
    const velox::TypePtr& type,
    const std::vector<velox::VectorPtr>& inputs,
    bool enableChunking) {
  return serializeTablet(
      type,
      inputs,
      enableChunking,
      EncodingType::Trivial,
      EncodingType::Trivial,
      EncodingType::Trivial);
}

SerializationTest::SerializeResult SerializationTest::serializeTablet(
    const velox::TypePtr& type,
    const std::vector<velox::VectorPtr>& inputs,
    bool enableChunking,
    EncodingType streamIdsEncodingType,
    EncodingType sizeIndicesEncodingType,
    EncodingType uniqueSizesEncodingType,
    bool forceDuplicateFirstTwoStreams) {
  // Write all inputs to a single Nimble file.
  std::string fileData;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&fileData);
    nimble::WriterOptions options;
    options.enableChunking = enableChunking;
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));
    for (const auto& input : inputs) {
      writer.write(input);
    }
    writer.close();
  }

  // Read the tablet to get raw stream bytes.
  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string_view(fileData));
  auto tablet = nimble::TabletReader::create(
      readFile, pool_.get(), makeTestTabletOptions(pool_.get()));
  EXPECT_GT(tablet->stripeCount(), 0);

  // Load the nimble schema from the tablet file.
  auto schemaSection =
      tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
  EXPECT_TRUE(schemaSection.has_value());
  auto nimbleSchema =
      nimble::SchemaDeserializer::deserialize(schemaSection->content().data());

  // Collect stream offsets to load.
  std::set<uint32_t> offsetSet;
  collectStreamOffsets(*nimbleSchema, offsetSet);
  std::vector<uint32_t> streamOffsets(offsetSet.begin(), offsetSet.end());

  // Assemble kTablet buffer for each stripe.
  SerializeResult result;
  result.schema = nimbleSchema;
  for (uint32_t stripeIdx = 0; stripeIdx < tablet->stripeCount(); ++stripeIdx) {
    const auto stripeId = tablet->stripeIdentifier(stripeIdx);
    auto streamLoaders = tablet->load(stripeId, streamOffsets);
    const auto stripeRows = tablet->stripeRowCount(stripeIdx);

    // Build kTablet: [header][raw stream data...][trailer].
    auto headerIOBuf = serde::createTabletChunkHeader(
        {.rowCount = stripeRows, .rowRange = RowRange{0, stripeRows}});
    std::string headerBuf(
        reinterpret_cast<const char*>(headerIOBuf.data()),
        headerIOBuf.length());

    auto streams = collectTabletStreams(streamOffsets, streamLoaders);
    if (forceDuplicateFirstTwoStreams) {
      std::optional<std::string_view> firstNonEmptyStream;
      bool duplicated = false;
      for (auto& stream : streams) {
        if (stream.data.empty()) {
          continue;
        }
        if (!firstNonEmptyStream.has_value()) {
          firstNonEmptyStream = stream.data;
          continue;
        }
        stream.data = *firstNonEmptyStream;
        duplicated = true;
        break;
      }
      EXPECT_TRUE(duplicated) << "Expected at least two non-empty streams";
    }

    std::string assembled;
    assembled.reserve(headerBuf.size());
    assembled.append(headerBuf);
    auto writeResult = writeTabletStreamsAndTrailer(
        streams,
        streamIdsEncodingType,
        sizeIndicesEncodingType,
        uniqueSizesEncodingType,
        assembled);
    result.numTabletStreams.push_back(writeResult.numStreams);
    result.numUniqueTabletStreams.push_back(writeResult.numUniqueStreams);
    result.tabletBodyBytes.push_back(writeResult.numBodyBytes);
    result.tabletUniqueBodyBytes.push_back(writeResult.numUniqueBodyBytes);
    result.serialized.push_back(std::move(assembled));
  }
  return result;
}

TEST_P(SerializationTest, flatMapEncodingFuzz) {
  // Test flat map encoding with fuzzer-generated data.
  // Iterates over different schema variations:
  // - Scalar value types
  // - Nested FlatMap value types (Map<K, Map<K2, V>>)
  // - Mix of scalar and nested

  auto seed = folly::Random::rand32();
  LOG(INFO) << "flatMapEncodingFuzz seed: " << seed;

  // Schema 1: Scalar value types only.
  auto scalarType = velox::ROW({
      {"id", velox::BIGINT()},
      {"int_features", velox::MAP(velox::INTEGER(), velox::REAL())},
      {"string_features", velox::MAP(velox::VARCHAR(), velox::DOUBLE())},
  });

  // Schema 2: Nested FlatMap value types (Map<K, Map<K2, V>>).
  auto nestedType = velox::ROW({
      {"id", velox::BIGINT()},
      {"nested_features",
       velox::MAP(
           velox::INTEGER(), velox::MAP(velox::INTEGER(), velox::REAL()))},
  });

  // Schema 3: Mix of scalar and nested FlatMap value types.
  auto mixedType = velox::ROW({
      {"id", velox::BIGINT()},
      {"scalar_features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
      {"nested_features",
       velox::MAP(
           velox::VARCHAR(), velox::MAP(velox::INTEGER(), velox::REAL()))},
  });

  // Schema 4: Map of Array (Map<K, Array<V>>).
  auto mapOfArrayType = velox::ROW({
      {"id", velox::BIGINT()},
      {"array_features",
       velox::MAP(velox::INTEGER(), velox::ARRAY(velox::BIGINT()))},
  });

  // Schema 5: Map of Array of Array (Map<K, Array<Array<V>>>).
  auto mapOfNestedArrayType = velox::ROW({
      {"id", velox::BIGINT()},
      {"nested_array_features",
       velox::MAP(
           velox::INTEGER(), velox::ARRAY(velox::ARRAY(velox::BIGINT())))},
  });

  struct TestCase {
    std::string name;
    velox::TypePtr type;
    folly::F14FastMap<std::string, std::set<std::string>> flatMapColumns;
  };

  std::vector<TestCase> testCases = {
      {"scalar", scalarType, {{"int_features", {}}, {"string_features", {}}}},
      {"nested", nestedType, {{"nested_features", {}}}},
      {"mixed", mixedType, {{"scalar_features", {}}, {"nested_features", {}}}},
      {"mapOfArray", mapOfArrayType, {{"array_features", {}}}},
      {"mapOfNestedArray",
       mapOfNestedArrayType,
       {{"nested_array_features", {}}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);

    size_t batchSize = 50;
    velox::VectorFuzzer fuzzer(
        {
            .vectorSize = batchSize,
            .nullRatio = 0,
            // Use fixed non-zero string length to avoid empty string keys.
            .stringLength = 8,
            .stringVariableLength = false,
            .containerLength = 5,
            .containerVariableLength = true,
            // Remove duplicate keys from maps.
            .normalizeMapKeys = true,
        },
        pool_.get(),
        seed);

    const SerializerOptions options{
        .compressionType = CompressionType::Zstd,
        .compressionThreshold = 32,
        .compressionLevel = 3,
        .version = version(),
        .flatMapColumns = testCase.flatMapColumns,
        .streamIndicesEncodingType = streamSizesEncodingType(),
        .streamSizesEncodingType = streamSizesEncodingType(),
    };
    Serializer serializer{options, testCase.type, pool_.get()};

    // Generate and serialize multiple batches.
    auto iterations = 20;
    std::vector<velox::VectorPtr> inputs;
    std::vector<std::string> serializedData;
    inputs.reserve(iterations);
    serializedData.reserve(iterations);

    for (auto i = 0; i < iterations; ++i) {
      auto input = fuzzer.fuzzInputRow(
          std::dynamic_pointer_cast<const velox::RowType>(testCase.type));
      inputs.push_back(input);
      serializedData.emplace_back(
          serializer.serialize(input, OrderedRanges::of(0, input->size())));
    }

    // Create deserializer with final schema containing all discovered keys.
    Deserializer deserializer{
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
        pool_.get(),
        deserializerOptions()};

    // Deserialize and verify each batch.
    velox::VectorPtr output;
    for (auto i = 0; i < iterations; ++i) {
      deserializer.deserialize(serializedData[i], output);

      ASSERT_EQ(output->size(), inputs[i]->size());
      for (velox::vector_size_t j = 0; j < inputs[i]->size(); ++j) {
        ASSERT_TRUE(vectorEquals(output, inputs[i], j))
            << "Content mismatch at index " << j
            << "\nReference: " << inputs[i]->toString(j)
            << "\nResult: " << output->toString(j);
      }
    }
  }
}

TEST_P(SerializationTest, fuzzComplex) {
  auto type = velox::ROW({
      {"array", velox::ARRAY(velox::REAL())},
      {"map", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
      {"row",
       velox::ROW({
           {"a", velox::REAL()},
           {"b", velox::INTEGER()},
       })},
      {"nested",
       velox::ARRAY(
           velox::ROW({
               {"a", velox::INTEGER()},
               {"b", velox::MAP(velox::REAL(), velox::REAL())},
           }))},
  });
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;

  // Small batches creates more edge cases.
  size_t batchSize = 10;
  velox::VectorFuzzer noNulls(
      {
          .vectorSize = batchSize,
          .nullRatio = 0,
          .stringLength = 20,
          .stringVariableLength = true,
          .containerLength = 5,
          .containerVariableLength = true,
      },
      pool_.get(),
      seed);

  folly::detail::DefaultGenerator rng{seed};
  auto iterations = 20;
  auto batches = 20;
  for (auto i = 0; i < iterations; ++i) {
    writeAndVerify(
        rng,
        pool_.get(),
        type,
        [&](auto& type) {
          return noNulls.fuzzInputRow(
              std::dynamic_pointer_cast<const velox::RowType>(type));
        },
        vectorEquals,
        {},
        batches);
  }
}

TEST_P(SerializationTest, rootNotRow) {
  const auto type = velox::MAP(velox::INTEGER(), velox::ARRAY(velox::DOUBLE()));
  const auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;

  // Small batches creates more edge cases.
  size_t batchSize = 10;
  velox::VectorFuzzer noNulls(
      {
          .vectorSize = batchSize,
          .nullRatio = 0,
          .stringLength = 20,
          .stringVariableLength = true,
          .containerLength = 5,
          .containerVariableLength = true,
      },
      pool_.get(),
      seed);

  folly::detail::DefaultGenerator rng{seed};
  const auto iterations = 20;
  const auto batches = 20;
  for (auto i = 0; i < iterations; ++i) {
    writeAndVerify(
        rng,
        pool_.get(),
        type,
        [&](auto& type) { return noNulls.fuzz(type); },
        vectorEquals,
        {},
        batches);
  }
}

TEST_P(SerializationTest, flatMapEncoding) {
  // Test flat map encoding with fixed keys.
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"float_features", velox::MAP(velox::INTEGER(), velox::REAL())},
      {"string_features", velox::MAP(velox::VARCHAR(), velox::DOUBLE())},
  });

  auto seed = folly::Random::rand32();
  LOG(INFO) << "flatMapEncoding seed: " << seed;

  // Fixed keys for FlatMap columns - every row has the same keys.
  const std::vector<int32_t> intKeys = {1, 2, 3, 4, 5};
  const std::vector<std::string> stringKeys = {"key_a", "key_b", "key_c"};

  size_t batchSize = 100;
  folly::Random::DefaultGenerator rng(seed);

  // Generate input with fixed keys.
  auto generateInput = [&]() -> velox::VectorPtr {
    // Generate id column
    auto ids =
        velox::BaseVector::create(velox::BIGINT(), batchSize, pool_.get());
    const auto batchSizeInt = static_cast<velox::vector_size_t>(batchSize);
    const auto intKeysCount = static_cast<velox::vector_size_t>(intKeys.size());
    const auto stringKeysCount =
        static_cast<velox::vector_size_t>(stringKeys.size());

    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      ids->asFlatVector<int64_t>()->set(i, folly::Random::rand64(rng));
    }

    // Generate float_features with fixed integer keys
    std::vector<velox::VectorPtr> intMapKeys;
    std::vector<velox::VectorPtr> intMapValues;
    std::vector<velox::vector_size_t> intMapOffsets;
    intMapOffsets.push_back(0);

    auto intKeysFlat = velox::BaseVector::create(
        velox::INTEGER(), batchSizeInt * intKeysCount, pool_.get());
    auto intValuesFlat = velox::BaseVector::create(
        velox::REAL(), batchSizeInt * intKeysCount, pool_.get());
    velox::vector_size_t intOffset = 0;
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      for (auto key : intKeys) {
        intKeysFlat->asFlatVector<int32_t>()->set(intOffset, key);
        intValuesFlat->asFlatVector<float>()->set(
            intOffset, folly::Random::randDouble01(rng));
        ++intOffset;
      }
      intMapOffsets.push_back(intOffset);
    }

    auto intMapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::REAL()),
        nullptr,
        batchSizeInt,
        velox::allocateOffsets(batchSizeInt, pool_.get()),
        velox::allocateSizes(batchSizeInt, pool_.get()),
        intKeysFlat,
        intValuesFlat);

    auto* intRawOffsets = intMapVector->mutableOffsets(batchSizeInt)
                              ->asMutable<velox::vector_size_t>();
    auto* intRawSizes = intMapVector->mutableSizes(batchSizeInt)
                            ->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      intRawOffsets[i] = intMapOffsets[i];
      intRawSizes[i] = intKeysCount;
    }

    // Generate string_features with fixed string keys
    auto strKeysFlat = velox::BaseVector::create(
        velox::VARCHAR(), batchSizeInt * stringKeysCount, pool_.get());
    auto strValuesFlat = velox::BaseVector::create(
        velox::DOUBLE(), batchSizeInt * stringKeysCount, pool_.get());
    velox::vector_size_t strOffset = 0;
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      for (const auto& key : stringKeys) {
        strKeysFlat->asFlatVector<velox::StringView>()->set(
            strOffset, velox::StringView(key));
        strValuesFlat->asFlatVector<double>()->set(
            strOffset, folly::Random::randDouble01(rng));
        ++strOffset;
      }
    }

    auto strMapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::VARCHAR(), velox::DOUBLE()),
        nullptr,
        batchSizeInt,
        velox::allocateOffsets(batchSizeInt, pool_.get()),
        velox::allocateSizes(batchSizeInt, pool_.get()),
        strKeysFlat,
        strValuesFlat);

    auto* strRawOffsets = strMapVector->mutableOffsets(batchSizeInt)
                              ->asMutable<velox::vector_size_t>();
    auto* strRawSizes = strMapVector->mutableSizes(batchSizeInt)
                            ->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      strRawOffsets[i] = i * stringKeysCount;
      strRawSizes[i] = stringKeysCount;
    }

    // Create row vector
    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        batchSizeInt,
        std::vector<velox::VectorPtr>{ids, intMapVector, strMapVector});
  };

  const SerializerOptions options{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"float_features", {}}, {"string_features", {}}},
  };
  Serializer serializer{options, type, pool_.get()};

  // Serialize multiple batches with the same fixed keys.
  auto iterations = 10;
  std::vector<velox::VectorPtr> inputs;
  std::vector<std::string> serializedData;
  inputs.reserve(iterations);
  serializedData.reserve(iterations);

  for (auto i = 0; i < iterations; ++i) {
    auto input = generateInput();
    inputs.push_back(input);
    serializedData.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
  }

  // Create deserializer with final schema.
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  // Deserialize and verify each batch.
  velox::VectorPtr output;
  for (auto i = 0; i < iterations; ++i) {
    deserializer.deserialize(serializedData[i], output);

    ASSERT_EQ(output->size(), inputs[i]->size());
    for (velox::vector_size_t j = 0; j < inputs[i]->size(); ++j) {
      ASSERT_TRUE(vectorEquals(output, inputs[i], j))
          << "Content mismatch at index " << j
          << "\nReference: " << inputs[i]->toString(j)
          << "\nResult: " << output->toString(j);
    }
  }

  // Also verify reading flatmap as struct.
  verifyFlatMapAsStruct(serializer, serializedData, inputs);
}

TEST_P(SerializationTest, flatMapEncodingWithVaryingKeys) {
  // Test flat map encoding where each row has a varying number of keys.
  // This tests the realistic scenario where different rows have different
  // subsets of the possible FlatMap keys.
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  auto seed = folly::Random::rand32();
  LOG(INFO) << "flatMapEncodingWithVaryingKeys seed: " << seed;

  // All possible keys - each row will have a subset of these.
  const std::vector<int32_t> allKeys = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  size_t batchSize = 50;
  folly::Random::DefaultGenerator rng(seed);

  // Generate input where each row has a varying subset of keys.
  auto generateInput = [&]() -> velox::VectorPtr {
    const auto batchSizeInt = static_cast<velox::vector_size_t>(batchSize);
    const auto allKeysCount = static_cast<velox::vector_size_t>(allKeys.size());

    // Generate id column
    auto ids =
        velox::BaseVector::create(velox::BIGINT(), batchSizeInt, pool_.get());
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      ids->asFlatVector<int64_t>()->set(i, folly::Random::rand64(rng));
    }

    // For each row, randomly select a subset of keys (1 to allKeys.size()).
    // To ensure determinism and reproducibility, each row gets keys based on
    // its index.
    std::vector<velox::vector_size_t> mapOffsets;
    std::vector<int32_t> keysData;
    std::vector<double> valuesData;

    mapOffsets.push_back(0);
    for (velox::vector_size_t row = 0; row < batchSizeInt; ++row) {
      // Each row gets a different number of keys (1 to allKeys.size()).
      // Use a deterministic pattern: row 0 gets 1 key, row 1 gets 2 keys, etc.
      velox::vector_size_t numKeysForRow = (row % allKeysCount) + 1;

      for (velox::vector_size_t k = 0; k < numKeysForRow; ++k) {
        keysData.push_back(allKeys[k]);
        valuesData.push_back(folly::Random::randDouble01(rng));
      }
      mapOffsets.push_back(static_cast<velox::vector_size_t>(keysData.size()));
    }

    // Create keys vector
    const auto keysCount = static_cast<velox::vector_size_t>(keysData.size());
    auto keysFlat =
        velox::BaseVector::create(velox::INTEGER(), keysCount, pool_.get());
    for (velox::vector_size_t i = 0; i < keysCount; ++i) {
      keysFlat->asFlatVector<int32_t>()->set(i, keysData[i]);
    }

    // Create values vector
    const auto valuesCount =
        static_cast<velox::vector_size_t>(valuesData.size());
    auto valuesFlat =
        velox::BaseVector::create(velox::DOUBLE(), valuesCount, pool_.get());
    for (velox::vector_size_t i = 0; i < valuesCount; ++i) {
      valuesFlat->asFlatVector<double>()->set(i, valuesData[i]);
    }

    // Create map vector
    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        batchSizeInt,
        velox::allocateOffsets(batchSizeInt, pool_.get()),
        velox::allocateSizes(batchSizeInt, pool_.get()),
        keysFlat,
        valuesFlat);

    auto* rawOffsets = mapVector->mutableOffsets(batchSizeInt)
                           ->asMutable<velox::vector_size_t>();
    auto* rawSizes = mapVector->mutableSizes(batchSizeInt)
                         ->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      rawOffsets[i] = mapOffsets[i];
      rawSizes[i] = mapOffsets[i + 1] - mapOffsets[i];
    }

    // Create row vector
    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        batchSizeInt,
        std::vector<velox::VectorPtr>{ids, mapVector});
  };

  const SerializerOptions options{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{options, type, pool_.get()};

  // Serialize multiple batches.
  auto iterations = 10;
  std::vector<velox::VectorPtr> inputs;
  std::vector<std::string> serializedData;
  inputs.reserve(iterations);
  serializedData.reserve(iterations);

  for (auto i = 0; i < iterations; ++i) {
    auto input = generateInput();
    inputs.push_back(input);
    serializedData.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
  }

  // Create deserializer with final schema containing all discovered keys.
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  // Deserialize and verify each batch.
  velox::VectorPtr output;
  for (auto i = 0; i < iterations; ++i) {
    deserializer.deserialize(serializedData[i], output);

    ASSERT_EQ(output->size(), inputs[i]->size());
    for (velox::vector_size_t j = 0; j < inputs[i]->size(); ++j) {
      ASSERT_TRUE(vectorEquals(output, inputs[i], j))
          << "Content mismatch at index " << j
          << "\nReference: " << inputs[i]->toString(j)
          << "\nResult: " << output->toString(j);
    }
  }
}

TEST_P(SerializationTest, flatMapEncodingWithNestedTypes) {
  // Test flat map encoding with nested value types using fixed keys.
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"nested_features",
       velox::MAP(velox::INTEGER(), velox::ARRAY(velox::REAL()))},
      {"struct_features",
       velox::MAP(
           velox::VARCHAR(),
           velox::ROW({{"a", velox::INTEGER()}, {"b", velox::DOUBLE()}}))},
  });

  auto seed = folly::Random::rand32();
  LOG(INFO) << "flatMapEncodingWithNestedTypes seed: " << seed;

  // Fixed keys for FlatMap columns - every row has the same keys.
  const std::vector<int32_t> intKeys = {10, 20, 30};
  const std::vector<std::string> stringKeys = {"struct_a", "struct_b"};

  size_t batchSize = 50;
  folly::Random::DefaultGenerator rng(seed);

  // Generate input with fixed keys.
  auto generateInput = [&]() -> velox::VectorPtr {
    const auto batchSizeInt = static_cast<velox::vector_size_t>(batchSize);
    const auto intKeysCount = static_cast<velox::vector_size_t>(intKeys.size());
    const auto stringKeysCount =
        static_cast<velox::vector_size_t>(stringKeys.size());
    const velox::vector_size_t arraySize = 3;

    // Generate id column
    auto ids =
        velox::BaseVector::create(velox::BIGINT(), batchSizeInt, pool_.get());
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      ids->asFlatVector<int64_t>()->set(i, folly::Random::rand64(rng));
    }

    // Generate nested_features: MAP(INTEGER, ARRAY(REAL)) with fixed keys
    const auto totalArrayElements = batchSizeInt * intKeysCount * arraySize;
    const auto totalMapEntries = batchSizeInt * intKeysCount;

    auto intKeysFlat = velox::BaseVector::create(
        velox::INTEGER(), totalMapEntries, pool_.get());
    auto arrayElements = velox::BaseVector::create(
        velox::REAL(), totalArrayElements, pool_.get());

    // Fill array elements with random values
    for (velox::vector_size_t i = 0; i < totalArrayElements; ++i) {
      arrayElements->asFlatVector<float>()->set(
          i, folly::Random::randDouble01(rng));
    }

    // Create the nested array vector
    auto arrayVector = std::make_shared<velox::ArrayVector>(
        pool_.get(),
        velox::ARRAY(velox::REAL()),
        nullptr,
        totalMapEntries,
        velox::allocateOffsets(totalMapEntries, pool_.get()),
        velox::allocateSizes(totalMapEntries, pool_.get()),
        arrayElements);

    auto* arrayOffsets = arrayVector->mutableOffsets(totalMapEntries)
                             ->asMutable<velox::vector_size_t>();
    auto* arraySizes = arrayVector->mutableSizes(totalMapEntries)
                           ->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < totalMapEntries; ++i) {
      arrayOffsets[i] = i * arraySize;
      arraySizes[i] = arraySize;
    }

    // Fill integer keys
    velox::vector_size_t keyIdx = 0;
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      for (auto key : intKeys) {
        intKeysFlat->asFlatVector<int32_t>()->set(keyIdx++, key);
      }
    }

    auto intMapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::ARRAY(velox::REAL())),
        nullptr,
        batchSizeInt,
        velox::allocateOffsets(batchSizeInt, pool_.get()),
        velox::allocateSizes(batchSizeInt, pool_.get()),
        intKeysFlat,
        arrayVector);

    auto* intMapOffsets = intMapVector->mutableOffsets(batchSizeInt)
                              ->asMutable<velox::vector_size_t>();
    auto* intMapSizes = intMapVector->mutableSizes(batchSizeInt)
                            ->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      intMapOffsets[i] = i * intKeysCount;
      intMapSizes[i] = intKeysCount;
    }

    // Generate struct_features: MAP(VARCHAR, ROW({a: INT, b: DOUBLE}))
    const auto totalStrMapEntries = batchSizeInt * stringKeysCount;
    auto strKeysFlat = velox::BaseVector::create(
        velox::VARCHAR(), totalStrMapEntries, pool_.get());
    auto structFieldA = velox::BaseVector::create(
        velox::INTEGER(), totalStrMapEntries, pool_.get());
    auto structFieldB = velox::BaseVector::create(
        velox::DOUBLE(), totalStrMapEntries, pool_.get());

    velox::vector_size_t strIdx = 0;
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      for (const auto& key : stringKeys) {
        strKeysFlat->asFlatVector<velox::StringView>()->set(
            strIdx, velox::StringView(key));
        structFieldA->asFlatVector<int32_t>()->set(
            strIdx, folly::Random::rand32(rng));
        structFieldB->asFlatVector<double>()->set(
            strIdx, folly::Random::randDouble01(rng));
        ++strIdx;
      }
    }

    auto structVector = std::make_shared<velox::RowVector>(
        pool_.get(),
        velox::ROW({{"a", velox::INTEGER()}, {"b", velox::DOUBLE()}}),
        nullptr,
        totalStrMapEntries,
        std::vector<velox::VectorPtr>{structFieldA, structFieldB});

    auto strMapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(
            velox::VARCHAR(),
            velox::ROW({{"a", velox::INTEGER()}, {"b", velox::DOUBLE()}})),
        nullptr,
        batchSizeInt,
        velox::allocateOffsets(batchSizeInt, pool_.get()),
        velox::allocateSizes(batchSizeInt, pool_.get()),
        strKeysFlat,
        structVector);

    auto* strMapOffsets = strMapVector->mutableOffsets(batchSizeInt)
                              ->asMutable<velox::vector_size_t>();
    auto* strMapSizes = strMapVector->mutableSizes(batchSizeInt)
                            ->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < batchSizeInt; ++i) {
      strMapOffsets[i] = i * stringKeysCount;
      strMapSizes[i] = stringKeysCount;
    }

    // Create row vector
    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        batchSizeInt,
        std::vector<velox::VectorPtr>{ids, intMapVector, strMapVector});
  };

  SerializerOptions options{
      .compressionType = CompressionType::Uncompressed,
      .version = version(),
      .flatMapColumns = {{"nested_features", {}}, {"struct_features", {}}},
  };
  Serializer serializer{options, type, pool_.get()};

  // Serialize multiple batches with the same fixed keys.
  auto iterations = 10;
  std::vector<velox::VectorPtr> inputs;
  std::vector<std::string> serializedData;
  inputs.reserve(iterations);
  serializedData.reserve(iterations);

  for (auto i = 0; i < iterations; ++i) {
    auto input = generateInput();
    inputs.push_back(input);
    serializedData.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
  }

  // Create deserializer with final schema.
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  // Deserialize and verify each batch.
  velox::VectorPtr output;
  for (auto i = 0; i < iterations; ++i) {
    deserializer.deserialize(serializedData[i], output);

    ASSERT_EQ(output->size(), inputs[i]->size());
    for (velox::vector_size_t j = 0; j < inputs[i]->size(); ++j) {
      ASSERT_TRUE(vectorEquals(output, inputs[i], j))
          << "Content mismatch at index " << j
          << "\nReference: " << inputs[i]->toString(j)
          << "\nResult: " << output->toString(j);
    }
  }

  // Also verify reading flatmap as struct.
  verifyFlatMapAsStruct(serializer, serializedData, inputs);
}

TEST_P(SerializationTest, nestedFlatMapWithVaryingInnerKeys) {
  // Test nested FlatMap where inner FlatMap has varying keys across batches.
  // This tests the scenario where:
  // - Batch 1: outer key "a" has inner keys [1, 2]
  // - Batch 2: outer key "a" has inner keys [2, 3] (key 1 missing, key 3 new)
  // The deserialization should correctly handle the missing inner keys.
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"nested_map",
       velox::MAP(
           velox::VARCHAR(), velox::MAP(velox::INTEGER(), velox::DOUBLE()))},
  });

  auto seed = folly::Random::rand32();
  LOG(INFO) << "nestedFlatMapWithVaryingInnerKeys seed: " << seed;

  folly::Random::DefaultGenerator rng(seed);
  const velox::vector_size_t batchSize = 10;

  // Helper to create a batch with specific outer and inner keys.
  auto generateBatch =
      [&](const std::vector<std::string>& outerKeys,
          const std::vector<std::vector<int32_t>>& innerKeysPerOuterKey)
      -> velox::VectorPtr {
    VELOX_CHECK_EQ(outerKeys.size(), innerKeysPerOuterKey.size());

    // Generate id column
    auto ids =
        velox::BaseVector::create(velox::BIGINT(), batchSize, pool_.get());
    for (velox::vector_size_t i = 0; i < batchSize; ++i) {
      ids->asFlatVector<int64_t>()->set(i, folly::Random::rand64(rng));
    }

    // Calculate total sizes
    velox::vector_size_t totalOuterEntries = 0;
    velox::vector_size_t totalInnerEntries = 0;
    for (size_t k = 0; k < outerKeys.size(); ++k) {
      totalOuterEntries += batchSize; // Each row has each outer key
      totalInnerEntries += batchSize *
          static_cast<velox::vector_size_t>(innerKeysPerOuterKey[k].size());
    }

    // Create inner map keys and values
    auto innerKeys = velox::BaseVector::create(
        velox::INTEGER(), totalInnerEntries, pool_.get());
    auto innerValues = velox::BaseVector::create(
        velox::DOUBLE(), totalInnerEntries, pool_.get());

    // Create inner map vector
    auto innerMap = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        totalOuterEntries,
        velox::allocateOffsets(totalOuterEntries, pool_.get()),
        velox::allocateSizes(totalOuterEntries, pool_.get()),
        innerKeys,
        innerValues);

    auto* innerMapOffsets = innerMap->mutableOffsets(totalOuterEntries)
                                ->asMutable<velox::vector_size_t>();
    auto* innerMapSizes = innerMap->mutableSizes(totalOuterEntries)
                              ->asMutable<velox::vector_size_t>();

    // Create outer map keys
    auto outerKeysVec = velox::BaseVector::create(
        velox::VARCHAR(), totalOuterEntries, pool_.get());

    // Fill data: for each row, add all outer keys with their inner maps
    velox::vector_size_t outerIdx = 0;
    velox::vector_size_t innerIdx = 0;
    for (velox::vector_size_t row = 0; row < batchSize; ++row) {
      for (size_t k = 0; k < outerKeys.size(); ++k) {
        outerKeysVec->asFlatVector<velox::StringView>()->set(
            outerIdx, velox::StringView(outerKeys[k]));

        innerMapOffsets[outerIdx] = innerIdx;
        const auto& innerKeysForThis = innerKeysPerOuterKey[k];
        innerMapSizes[outerIdx] =
            static_cast<velox::vector_size_t>(innerKeysForThis.size());

        for (auto innerKey : innerKeysForThis) {
          innerKeys->asFlatVector<int32_t>()->set(innerIdx, innerKey);
          innerValues->asFlatVector<double>()->set(
              innerIdx, folly::Random::randDouble01(rng));
          ++innerIdx;
        }
        ++outerIdx;
      }
    }

    // Create outer map vector
    auto outerMap = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(
            velox::VARCHAR(), velox::MAP(velox::INTEGER(), velox::DOUBLE())),
        nullptr,
        batchSize,
        velox::allocateOffsets(batchSize, pool_.get()),
        velox::allocateSizes(batchSize, pool_.get()),
        outerKeysVec,
        innerMap);

    auto* outerMapOffsets =
        outerMap->mutableOffsets(batchSize)->asMutable<velox::vector_size_t>();
    auto* outerMapSizes =
        outerMap->mutableSizes(batchSize)->asMutable<velox::vector_size_t>();

    const auto outerKeysCount =
        static_cast<velox::vector_size_t>(outerKeys.size());
    for (velox::vector_size_t i = 0; i < batchSize; ++i) {
      outerMapOffsets[i] = i * outerKeysCount;
      outerMapSizes[i] = outerKeysCount;
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        batchSize,
        std::vector<velox::VectorPtr>{ids, outerMap});
  };

  SerializerOptions options{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"nested_map", {}}},
  };
  Serializer serializer{options, type, pool_.get()};

  // Create batches with varying inner keys:
  // Batch 1: outer key "a" has inner keys [1, 2], outer key "b" has [10]
  // Batch 2: outer key "a" has inner keys [2, 3], outer key "b" has [20]
  // (inner key 1 missing in batch 2, inner key 3 new in batch 2)
  std::vector<velox::VectorPtr> inputs;
  std::vector<std::string> serializedData;

  // Batch 1
  auto batch1 = generateBatch({"a", "b"}, {{1, 2}, {10}});
  inputs.push_back(batch1);
  serializedData.emplace_back(
      serializer.serialize(batch1, OrderedRanges::of(0, batch1->size())));

  // Batch 2 - different inner keys
  auto batch2 = generateBatch({"a", "b"}, {{2, 3}, {20}});
  inputs.push_back(batch2);
  serializedData.emplace_back(
      serializer.serialize(batch2, OrderedRanges::of(0, batch2->size())));

  // Batch 3 - outer key "c" is new, "a" has different inner keys again
  auto batch3 = generateBatch({"a", "c"}, {{1, 3}, {100}});
  inputs.push_back(batch3);
  serializedData.emplace_back(
      serializer.serialize(batch3, OrderedRanges::of(0, batch3->size())));

  // Create deserializer with final schema containing all discovered keys.
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  // Test 1: Deserialize each batch individually - should work.
  velox::VectorPtr output;
  for (size_t i = 0; i < inputs.size(); ++i) {
    deserializer.deserialize(serializedData[i], output);
    ASSERT_EQ(output->size(), inputs[i]->size())
        << "Size mismatch for batch " << i;
    for (velox::vector_size_t j = 0; j < inputs[i]->size(); ++j) {
      ASSERT_TRUE(vectorEquals(output, inputs[i], j))
          << "Content mismatch at batch " << i << " index " << j
          << "\nReference: " << inputs[i]->toString(j)
          << "\nResult: " << output->toString(j);
    }
  }

  // Test 2: Deserialize multiple batches together - this is the key test.
  // This tests gap detection for nested FlatMaps with varying inner keys.
  std::vector<std::string_view> allBatches;
  allBatches.reserve(serializedData.size());
  for (const auto& s : serializedData) {
    allBatches.push_back(s);
  }

  deserializer.deserialize(allBatches, output);

  // Build expected output by concatenating inputs.
  velox::VectorPtr expected = inputs[0];
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto oldSize = expected->size();
    expected->resize(oldSize + inputs[i]->size());
    expected->copy(inputs[i].get(), oldSize, 0, inputs[i]->size());
  }

  ASSERT_EQ(output->size(), expected->size()) << "Multi-batch size mismatch";

  for (velox::vector_size_t j = 0; j < expected->size(); ++j) {
    ASSERT_TRUE(vectorEquals(output, expected, j))
        << "Multi-batch content mismatch at index " << j
        << "\nReference: " << expected->toString(j)
        << "\nResult: " << output->toString(j);
  }
}

TEST_P(SerializationTest, nullableStreamsRoundTrip) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "nullableStreamsRoundTrip seed: " << seed;

  const folly::F14FastMap<std::string, std::set<std::string>> noFlatMaps;
  auto roundTrip =
      [&](const velox::TypePtr& type,
          const velox::VectorPtr& row,
          const folly::F14FastMap<std::string, std::set<std::string>>&
              flatMapColumns) {
        SerializerOptions options{
            .compressionType = CompressionType::Zstd,
            .compressionThreshold = 32,
            .compressionLevel = 3,
            .version = version(),
            .flatMapColumns = flatMapColumns,
            .streamIndicesEncodingType = streamSizesEncodingType(),
            .streamSizesEncodingType = streamSizesEncodingType(),
        };
        Serializer serializer{options, type, pool_.get()};
        auto serialized =
            serializer.serialize(row, OrderedRanges::of(0, row->size()));

        Deserializer deserializer{
            SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
            pool_.get(),
            deserializerOptions()};

        velox::VectorPtr output;
        deserializer.deserialize(serialized, output);
        ASSERT_EQ(output->size(), row->size());
        for (velox::vector_size_t i = 0; i < row->size(); ++i) {
          ASSERT_TRUE(vectorEquals(output, row, i))
              << "Content mismatch at row " << i
              << "\nExpected: " << row->toString(i)
              << "\nActual: " << output->toString(i);
        }
      };

  const bool encodingEnabled =
      SerializerOptions{.version = version()}.enableEncoding();

  // Nested scalar content nulls require the encoded serializer format.
  {
    auto type =
        velox::ROW({{"id", velox::BIGINT()}, {"value", velox::DOUBLE()}});
    auto ids = velox::BaseVector::create(velox::BIGINT(), 10, pool_.get());
    auto values = velox::BaseVector::create(velox::DOUBLE(), 10, pool_.get());
    for (velox::vector_size_t i = 0; i < 10; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
      values->asFlatVector<double>()->set(i, i * 1.5);
    }
    values->setNull(3, true); // Set one value as null

    auto row = std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        10,
        std::vector<velox::VectorPtr>{ids, values});

    if (encodingEnabled) {
      roundTrip(type, row, noFlatMaps);
    } else {
      SerializerOptions options{.version = version()};
      Serializer serializer{options, type, pool_.get()};
      NIMBLE_ASSERT_THROW(
          serializer.serialize(row, OrderedRanges::of(0, row->size())),
          "nullable content streams require an encoded serializer format");
    }
  }

  if (!encodingEnabled) {
    return;
  }

  // Map value nulls are encoded as nullable content streams.
  {
    auto type = velox::ROW(
        {{"id", velox::BIGINT()},
         {"map_col", velox::MAP(velox::INTEGER(), velox::DOUBLE())}});
    const velox::vector_size_t numRows = 5;
    const velox::vector_size_t entriesPerRow = 3;
    const velox::vector_size_t totalEntries = numRows * entriesPerRow;

    auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
    auto mapKeys =
        velox::BaseVector::create(velox::INTEGER(), totalEntries, pool_.get());
    auto mapValues =
        velox::BaseVector::create(velox::DOUBLE(), totalEntries, pool_.get());

    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
    }
    for (velox::vector_size_t i = 0; i < totalEntries; ++i) {
      mapKeys->asFlatVector<int32_t>()->set(i, i);
      mapValues->asFlatVector<double>()->set(i, i * 0.5);
    }
    mapValues->setNull(5, true); // Set one map value as null

    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool_.get()),
        velox::allocateSizes(numRows, pool_.get()),
        mapKeys,
        mapValues);

    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = i * entriesPerRow;
      rawSizes[i] = entriesPerRow;
    }

    auto row = std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{ids, mapVector});

    roundTrip(type, row, noFlatMaps);
  }

  // FlatMap value nulls exercise nullable decoding through scatterBitmap.
  {
    auto type = velox::ROW(
        {{"id", velox::BIGINT()},
         {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())}});
    const velox::vector_size_t numRows = 5;
    const velox::vector_size_t entriesPerRow = 2;
    const velox::vector_size_t totalEntries = numRows * entriesPerRow;

    auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
    auto mapKeys =
        velox::BaseVector::create(velox::INTEGER(), totalEntries, pool_.get());
    auto mapValues =
        velox::BaseVector::create(velox::DOUBLE(), totalEntries, pool_.get());

    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
    }
    for (velox::vector_size_t i = 0; i < totalEntries; ++i) {
      mapKeys->asFlatVector<int32_t>()->set(i, i % entriesPerRow);
      mapValues->asFlatVector<double>()->set(i, i * 0.5);
    }
    mapValues->setNull(3, true);

    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool_.get()),
        velox::allocateSizes(numRows, pool_.get()),
        mapKeys,
        mapValues);

    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = i * entriesPerRow;
      rawSizes[i] = entriesPerRow;
    }

    auto row = std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{ids, mapVector});

    roundTrip(type, row, {{"features", {}}});
  }
}

TEST_F(SerializationTest, rowNullStreamOmissionRoundTrips) {
  constexpr velox::vector_size_t kRows = 8;
  auto nestedType = velox::ROW({{"value", velox::INTEGER()}});
  auto type = velox::ROW({{"nested", nestedType}});

  auto makeRow = [&](bool allNull) -> velox::VectorPtr {
    auto values =
        velox::BaseVector::create(velox::INTEGER(), kRows, pool_.get());
    for (velox::vector_size_t i = 0; i < kRows; ++i) {
      values->asFlatVector<int32_t>()->set(i, i);
    }

    auto nested = std::make_shared<velox::RowVector>(
        pool_.get(),
        nestedType,
        nullptr,
        kRows,
        std::vector<velox::VectorPtr>{values});
    if (allNull) {
      for (velox::vector_size_t i = 0; i < kRows; ++i) {
        nested->setNull(i, true);
      }
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRows,
        std::vector<velox::VectorPtr>{nested});
  };

  auto serializeAndVerify = [&](bool allNull, bool expectNestedNullStream) {
    SerializerOptions options{
        .version = SerializationVersion::kSerialization,
        .streamIndicesEncodingType = EncodingType::Trivial,
        .streamSizesEncodingType = EncodingType::Trivial,
    };
    Serializer serializer{options, type, pool_.get()};
    auto row = makeRow(allNull);
    const std::string serialized{
        serializer.serialize(row, OrderedRanges::of(0, row->size()))};
    const auto schema =
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
    const auto rootNullOffset = schema->asRow().nullsDescriptor().offset();
    const auto nestedNullOffset =
        schema->asRow().childAt(0)->asRow().nullsDescriptor().offset();

    DeserializerOptions deserializerOptions{.hasHeader = true};
    serde::StreamDataParser reader{pool_.get(), deserializerOptions};
    reader.initialize(serialized);

    bool sawRootNullStream = false;
    bool sawNestedNullStream = false;
    reader.iterateStreams([&](uint32_t offset, std::string_view streamData) {
      if (offset == rootNullOffset) {
        sawRootNullStream = true;
        EXPECT_FALSE(streamData.empty());
        return;
      }
      if (offset == nestedNullOffset) {
        sawNestedNullStream = true;
        EXPECT_FALSE(streamData.empty());
      }
    });

    EXPECT_FALSE(sawRootNullStream);
    EXPECT_EQ(sawNestedNullStream, expectNestedNullStream);

    Deserializer deserializer{schema, pool_.get(), deserializerOptions};
    velox::VectorPtr output;
    deserializer.deserialize(serialized, output);

    ASSERT_EQ(output->size(), row->size());
    for (velox::vector_size_t i = 0; i < kRows; ++i) {
      EXPECT_TRUE(vectorEquals(row, output, i));
    }
  };
  serializeAndVerify(/*allNull=*/false, /*expectNestedNullStream=*/false);
  serializeAndVerify(/*allNull=*/true, /*expectNestedNullStream=*/true);
}

TEST_F(SerializationTest, nestedEncodingBufferCacheSizeConfigured) {
  constexpr velox::vector_size_t kRows = 16;
  auto type = velox::ROW({{"value", velox::BIGINT()}});

  auto makeInput = [&](int64_t base) -> velox::VectorPtr {
    auto values =
        velox::BaseVector::create(velox::BIGINT(), kRows, pool_.get());
    auto* flatValues = values->asFlatVector<int64_t>();
    for (velox::vector_size_t i = 0; i < kRows; ++i) {
      flatValues->set(i, base + i);
      if (i % 5 == 0) {
        values->setNull(i, true);
      }
    }
    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRows,
        std::vector<velox::VectorPtr>{values});
  };

  struct TestCase {
    const char* name;
    bool setMaxCachedNestedEncodingBuffers;
    uint32_t maxCachedNestedEncodingBuffers;
  };

  for (const auto& testCase : std::vector<TestCase>{
           {
               .name = "default",
               .setMaxCachedNestedEncodingBuffers = false,
               .maxCachedNestedEncodingBuffers = 0,
           },
           {
               .name = "cache disabled",
               .setMaxCachedNestedEncodingBuffers = true,
               .maxCachedNestedEncodingBuffers = 0,
           },
           {
               .name = "cache enabled",
               .setMaxCachedNestedEncodingBuffers = true,
               .maxCachedNestedEncodingBuffers = 1,
           },
       }) {
    SCOPED_TRACE(testCase.name);

    SerializerOptions options{
        .version = SerializationVersion::kSerialization,
    };
    if (testCase.setMaxCachedNestedEncodingBuffers) {
      options.maxCachedNestedEncodingBuffers =
          testCase.maxCachedNestedEncodingBuffers;
    }
    Serializer serializer{options, type, pool_.get()};
    Deserializer deserializer{
        SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
        pool_.get(),
        DeserializerOptions{.hasHeader = true}};

    for (const auto base : {0, 100}) {
      auto input = makeInput(base);
      const std::string serialized{
          serializer.serialize(input, OrderedRanges::of(0, input->size()))};

      velox::VectorPtr output;
      deserializer.deserialize(serialized, output);

      ASSERT_EQ(output->size(), input->size());
      for (velox::vector_size_t i = 0; i < kRows; ++i) {
        EXPECT_TRUE(vectorEquals(input, output, i));
      }
    }
  }
}

TEST_F(SerializationTest, encodedSerializerRoundTripsTopLevelRowNulls) {
  constexpr velox::vector_size_t kRows = 6;
  auto type = velox::ROW({
      {"id", velox::INTEGER()},
      {"score", velox::DOUBLE()},
  });

  auto makeInput = [&]() -> velox::VectorPtr {
    auto ids = velox::BaseVector::create(velox::INTEGER(), kRows, pool_.get());
    auto scores =
        velox::BaseVector::create(velox::DOUBLE(), kRows, pool_.get());
    for (velox::vector_size_t i = 0; i < kRows; ++i) {
      ids->asFlatVector<int32_t>()->set(i, 100 + i);
      scores->asFlatVector<double>()->set(i, i * 1.25);
    }

    auto input = std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRows,
        std::vector<velox::VectorPtr>{ids, scores});
    input->setNull(1, true);
    input->setNull(4, true);
    return input;
  };

  SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .streamIndicesEncodingType = EncodingType::Trivial,
      .streamSizesEncodingType = EncodingType::Trivial,
  };
  Serializer serializer{options, type, pool_.get()};
  auto input = makeInput();
  const auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const auto rootNullOffset = schema->asRow().nullsDescriptor().offset();
  DeserializerOptions deserializerOptions{.hasHeader = true};
  serde::StreamDataParser reader{pool_.get(), deserializerOptions};
  reader.initialize(serialized);
  bool sawRootNullStream = false;
  reader.iterateStreams([&](uint32_t offset, std::string_view streamData) {
    if (offset == rootNullOffset) {
      sawRootNullStream = true;
      EXPECT_FALSE(streamData.empty());
    }
  });
  EXPECT_TRUE(sawRootNullStream);

  Deserializer deserializer{schema, pool_.get(), deserializerOptions};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < kRows; ++i) {
    EXPECT_TRUE(vectorEquals(input, output, i));
  }
}

TEST_F(SerializationTest, regularDataNullsDoNotRequireNullBarrier) {
  constexpr velox::vector_size_t kRows = 3;
  auto type = velox::ROW({
      {"scalar", velox::INTEGER()},
      {"items", velox::ARRAY(velox::INTEGER())},
      {"attrs", velox::MAP(velox::VARCHAR(), velox::INTEGER())},
  });

  auto scalar = velox::BaseVector::create(velox::INTEGER(), kRows, pool_.get());
  for (velox::vector_size_t i = 0; i < kRows; ++i) {
    scalar->asFlatVector<int32_t>()->set(i, 10 + i);
  }
  scalar->setNull(0, true);

  auto arrayOffsets = velox::allocateOffsets(kRows, pool_.get());
  auto arraySizes = velox::allocateSizes(kRows, pool_.get());
  auto* rawArrayOffsets = arrayOffsets->asMutable<velox::vector_size_t>();
  auto* rawArraySizes = arraySizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < kRows; ++i) {
    rawArrayOffsets[i] = i * 2;
    rawArraySizes[i] = 2;
  }
  auto arrayElements =
      velox::BaseVector::create(velox::INTEGER(), kRows * 2, pool_.get());
  for (velox::vector_size_t i = 0; i < kRows * 2; ++i) {
    arrayElements->asFlatVector<int32_t>()->set(i, 100 + i);
  }
  auto items = std::make_shared<velox::ArrayVector>(
      pool_.get(),
      velox::ARRAY(velox::INTEGER()),
      nullptr,
      kRows,
      arrayOffsets,
      arraySizes,
      arrayElements);
  items->setNull(1, true);

  auto mapOffsets = velox::allocateOffsets(kRows, pool_.get());
  auto mapSizes = velox::allocateSizes(kRows, pool_.get());
  auto* rawMapOffsets = mapOffsets->asMutable<velox::vector_size_t>();
  auto* rawMapSizes = mapSizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < kRows; ++i) {
    rawMapOffsets[i] = i;
    rawMapSizes[i] = 1;
  }
  auto mapKeys =
      velox::BaseVector::create(velox::VARCHAR(), kRows, pool_.get());
  auto mapValues =
      velox::BaseVector::create(velox::INTEGER(), kRows, pool_.get());
  const std::array<const char*, kRows> mapKeyValues = {"a", "b", "c"};
  for (velox::vector_size_t i = 0; i < kRows; ++i) {
    mapKeys->asFlatVector<velox::StringView>()->set(
        i, velox::StringView(mapKeyValues[i]));
    mapValues->asFlatVector<int32_t>()->set(i, 200 + i);
  }
  auto attrs = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(velox::VARCHAR(), velox::INTEGER()),
      nullptr,
      kRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);
  attrs->setNull(2, true);

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      kRows,
      std::vector<velox::VectorPtr>{scalar, items, attrs});

  SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .streamIndicesEncodingType = EncodingType::Trivial,
      .streamSizesEncodingType = EncodingType::Trivial,
  };
  Serializer serializer{options, type, pool_.get()};
  const std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, input->size()))};

  const char* pos = serialized.data();
  const auto header = serde::readSerializationHeader(
      pos, serialized.data() + serialized.size(), true);
  EXPECT_FALSE(header.flags.requiresNullBarrier);

  const auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  Deserializer deserializer{
      schema, pool_.get(), DeserializerOptions{.hasHeader = true}};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < kRows; ++i) {
    EXPECT_TRUE(vectorEquals(input, output, i));
  }
}

TEST_F(SerializationTest, serializerRejectsFixedStreamRowCountEncoding) {
  constexpr velox::vector_size_t kRows{4};
  auto type = velox::ROW({{"value", velox::INTEGER()}});

  auto values = velox::BaseVector::create(velox::INTEGER(), kRows, pool_.get());
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    values->asFlatVector<int32_t>()->set(row, 10 + row);
  }
  auto input = std::make_shared<velox::RowVector>(
      pool_.get(), type, nullptr, kRows, std::vector<velox::VectorPtr>{values});

  SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .streamIndicesEncodingType = EncodingType::Trivial,
      .streamSizesEncodingType = EncodingType::Trivial,
      .encodingOptions = Encoding::Options{.useVarintRowCount = false},
  };
  Serializer serializer{options, type, pool_.get()};
  NIMBLE_ASSERT_THROW(
      serializer.serialize(input, OrderedRanges::of(0, input->size())),
      "Non-tablet writers must use varint stream row counts");
}

TEST_F(SerializationTest, deserializesInputWithoutStreamVarintRowCountFlag) {
  constexpr velox::vector_size_t kRows{4};
  auto type = velox::ROW({{"value", velox::INTEGER()}});

  auto values = velox::BaseVector::create(velox::INTEGER(), kRows, pool_.get());
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    values->asFlatVector<int32_t>()->set(row, 10 + row);
  }
  auto input = std::make_shared<velox::RowVector>(
      pool_.get(), type, nullptr, kRows, std::vector<velox::VectorPtr>{values});

  SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .streamIndicesEncodingType = EncodingType::Trivial,
      .streamSizesEncodingType = EncodingType::Trivial,
  };
  Serializer serializer{options, type, pool_.get()};
  std::string serialized{
      serializer.serialize(input, OrderedRanges::of(0, input->size()))};
  serialized[sizeof(uint8_t) + varint::varintSize(kRows)] =
      static_cast<char>(serde::detail::makeFlagsByte(
          /*requiresNullBarrier=*/false,
          /*streamEncodingUsesVarintRowCount=*/false));

  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      DeserializerOptions{.hasHeader = true}};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t row = 0; row < kRows; ++row) {
    EXPECT_TRUE(vectorEquals(input, output, row));
  }
}

TEST_F(SerializationTest, deserializerReadsPhysicalRootNullStream) {
  constexpr size_t kRows = 3;
  auto type = velox::ROW({{"value", velox::INTEGER()}});
  auto schema = convertToNimbleType(*type);
  const auto rootNullOffset = schema->asRow().nullsDescriptor().offset();
  const auto valueOffset =
      schema->asRow().childAt(0)->asScalar().scalarDescriptor().offset();

  const std::array<int32_t, kRows> allValues = {11, 22, 33};
  const std::array<int32_t, 2> nonNullValues = {11, 33};
  const SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .streamIndicesEncodingType = EncodingType::Trivial,
      .streamSizesEncodingType = EncodingType::Trivial,
  };

  auto makeSerialized = [&](const std::array<bool, kRows>& rootNonNulls,
                            std::string_view valueData) -> std::string {
    const Encoding::Options encodingOptions{.useVarintRowCount = true};
    Buffer rootBuffer{*pool_};
    const std::string_view rootData{
        reinterpret_cast<const char*>(rootNonNulls.data()),
        rootNonNulls.size() * sizeof(bool)};
    std::string rootEncoded{serde::detail::encodeScalar<std::string>(
        options,
        ScalarKind::Bool,
        rootData,
        *pool_,
        rootBuffer,
        /*encodingLayout=*/nullptr,
        encodingOptions)};

    Buffer valueBuffer{*pool_};
    std::string valueEncoded{serde::detail::encodeScalar<std::string>(
        options,
        ScalarKind::Int32,
        valueData,
        *pool_,
        valueBuffer,
        /*encodingLayout=*/nullptr,
        encodingOptions)};

    std::vector<std::pair<uint32_t, std::string>> streams;
    streams.emplace_back(rootNullOffset, std::move(rootEncoded));
    streams.emplace_back(valueOffset, std::move(valueEncoded));
    std::sort(streams.begin(), streams.end());

    std::string serialized;
    const auto flagsOffset = serde::writeSerializationHeader(
        serialized, SerializationVersion::kSerialization, kRows);
    const bool requiresNullBarrier = !std::all_of(
        rootNonNulls.begin(), rootNonNulls.end(), [](bool nonNull) {
          return nonNull;
        });
    NIMBLE_CHECK_LT(
        flagsOffset, serialized.size(), "Invalid flags byte offset");
    serialized[flagsOffset] = static_cast<char>(
        requiresNullBarrier
            ? serde::SerializationHeader::kNullBarrierRequiredFlag
            : 0);
    std::vector<uint32_t> streamSizes(streams.back().first + 1, 0);
    for (const auto& [offset, stream] : streams) {
      streamSizes[offset] = static_cast<uint32_t>(stream.size());
      serialized.append(stream);
    }
    serde::detail::writeTrailer(
        streamSizes, EncodingType::Trivial, EncodingType::Trivial, serialized);
    return serialized;
  };

  DeserializerOptions deserializerOptions{.hasHeader = true};
  Deserializer deserializer{schema, pool_.get(), deserializerOptions};

  velox::VectorPtr output;
  const std::string_view allValueData{
      reinterpret_cast<const char*>(allValues.data()),
      allValues.size() * sizeof(int32_t)};
  deserializer.deserialize(
      makeSerialized({true, true, true}, allValueData), output);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->size(), kRows);
  auto* row = output->as<velox::RowVector>();
  ASSERT_NE(row, nullptr);
  auto* valueVector = row->childAt(0)->as<velox::FlatVector<int32_t>>();
  ASSERT_NE(valueVector, nullptr);
  for (velox::vector_size_t i = 0; i < kRows; ++i) {
    EXPECT_EQ(valueVector->valueAt(i), allValues[i]);
  }

  velox::VectorPtr nullableOutput;
  const std::string_view nonNullValueData{
      reinterpret_cast<const char*>(nonNullValues.data()),
      nonNullValues.size() * sizeof(int32_t)};
  deserializer.deserialize(
      makeSerialized({true, false, true}, nonNullValueData), nullableOutput);
  ASSERT_NE(nullableOutput, nullptr);
  ASSERT_EQ(nullableOutput->size(), kRows);
  EXPECT_FALSE(nullableOutput->isNullAt(0));
  EXPECT_TRUE(nullableOutput->isNullAt(1));
  EXPECT_FALSE(nullableOutput->isNullAt(2));
  auto* nullableRow = nullableOutput->as<velox::RowVector>();
  ASSERT_NE(nullableRow, nullptr);
  auto* nullableValues =
      nullableRow->childAt(0)->as<velox::FlatVector<int32_t>>();
  ASSERT_NE(nullableValues, nullptr);
  EXPECT_EQ(nullableValues->valueAt(0), nonNullValues[0]);
  EXPECT_EQ(nullableValues->valueAt(2), nonNullValues[1]);
}

TEST_F(
    SerializationTest,
    encodedSerializerNullableContentNullBitmapAcrossBatches) {
  constexpr velox::vector_size_t kRows = 4;
  auto type = velox::ROW({{"value", velox::INTEGER()}});

  SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .streamIndicesEncodingType = EncodingType::Trivial,
      .streamSizesEncodingType = EncodingType::Trivial,
  };
  Serializer serializer{options, type, pool_.get()};
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      DeserializerOptions{.hasHeader = true}};

  auto makeInput = [&](int32_t base,
                       const std::vector<velox::vector_size_t>& nullRows)
      -> velox::VectorPtr {
    auto values =
        velox::BaseVector::create(velox::INTEGER(), kRows, pool_.get());
    for (velox::vector_size_t i = 0; i < kRows; ++i) {
      values->asFlatVector<int32_t>()->set(i, base + i);
    }
    for (auto row : nullRows) {
      values->setNull(row, true);
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRows,
        std::vector<velox::VectorPtr>{values});
  };

  std::vector<velox::VectorPtr> inputs = {
      makeInput(100, {}),
      makeInput(200, {1, 3}),
      makeInput(300, {}),
      makeInput(400, {0}),
      makeInput(500, {}),
      makeInput(600, {0, 1, 2, 3}),
      makeInput(700, {}),
  };

  std::vector<std::string> serialized;
  serialized.reserve(inputs.size());
  for (const auto& input : inputs) {
    serialized.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
  }
  std::vector<std::string_view> serializedViews;
  serializedViews.reserve(serialized.size());
  for (const auto& data : serialized) {
    serializedViews.push_back(data);
  }

  auto expected = inputs.front();
  for (size_t i = 1; i < inputs.size(); ++i) {
    const auto oldSize = expected->size();
    expected->resize(oldSize + inputs[i]->size());
    expected->copy(inputs[i].get(), oldSize, 0, inputs[i]->size());
  }

  velox::VectorPtr output;
  deserializer.deserialize(serializedViews, output);

  ASSERT_EQ(output->size(), expected->size());
  for (velox::vector_size_t i = 0; i < expected->size(); ++i) {
    EXPECT_TRUE(vectorEquals(expected, output, i))
        << "Content mismatch at row " << i
        << "\nExpected: " << expected->toString(i)
        << "\nActual: " << output->toString(i);
  }
}

TEST_F(SerializationTest, encodedSerializerNestedRowNullsAcrossBatches) {
  constexpr velox::vector_size_t kRows = 5;
  auto nestedType = velox::ROW({{"score", velox::BIGINT()}});
  auto type = velox::ROW({
      {"id", velox::INTEGER()},
      {"nested", nestedType},
  });

  SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .streamIndicesEncodingType = EncodingType::Trivial,
      .streamSizesEncodingType = EncodingType::Trivial,
  };
  Serializer serializer{options, type, pool_.get()};
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      DeserializerOptions{.hasHeader = true}};

  auto makeInput = [&](int32_t idBase,
                       const std::vector<velox::vector_size_t>& nestedNullRows)
      -> velox::VectorPtr {
    auto ids = velox::BaseVector::create(velox::INTEGER(), kRows, pool_.get());
    auto scores =
        velox::BaseVector::create(velox::BIGINT(), kRows, pool_.get());
    for (velox::vector_size_t i = 0; i < kRows; ++i) {
      ids->asFlatVector<int32_t>()->set(i, idBase + i);
      scores->asFlatVector<int64_t>()->set(i, (idBase + i) * 10);
    }

    auto nested = std::make_shared<velox::RowVector>(
        pool_.get(),
        nestedType,
        nullptr,
        kRows,
        std::vector<velox::VectorPtr>{scores});
    for (auto row : nestedNullRows) {
      nested->setNull(row, true);
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRows,
        std::vector<velox::VectorPtr>{ids, nested});
  };

  struct TestCase {
    std::string name;
    std::vector<velox::vector_size_t> firstBatchNullRows;
    std::vector<velox::vector_size_t> secondBatchNullRows;
  };

  const std::vector<TestCase> testCases = {
      {
          .name = "gap before nested null stream",
          .firstBatchNullRows = {},
          .secondBatchNullRows = {1, 3},
      },
      {
          .name = "gap after nested null stream",
          .firstBatchNullRows = {0, 2},
          .secondBatchNullRows = {},
      },
      {
          .name = "nested null stream missing from all batches",
          .firstBatchNullRows = {},
          .secondBatchNullRows = {},
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    auto first = makeInput(100, testCase.firstBatchNullRows);
    auto second = makeInput(200, testCase.secondBatchNullRows);
    std::vector<std::string> serialized;
    serialized.emplace_back(
        serializer.serialize(first, OrderedRanges::of(0, first->size())));
    serialized.emplace_back(
        serializer.serialize(second, OrderedRanges::of(0, second->size())));
    std::vector<std::string_view> serializedViews;
    serializedViews.reserve(serialized.size());
    for (const auto& data : serialized) {
      serializedViews.push_back(data);
    }

    auto expected = first;
    const auto firstRows = first->size();
    expected->resize(firstRows + second->size());
    expected->copy(second.get(), firstRows, 0, second->size());

    velox::VectorPtr output;
    deserializer.deserialize(serializedViews, output);

    ASSERT_EQ(output->size(), expected->size());
    for (velox::vector_size_t i = 0; i < expected->size(); ++i) {
      EXPECT_TRUE(vectorEquals(expected, output, i))
          << "Content mismatch at row " << i
          << "\nExpected: " << expected->toString(i)
          << "\nActual: " << output->toString(i);
    }
  }
}

TEST_F(
    SerializationTest,
    encodedSerializerNestedRowNullsUnderArrayAcrossBatches) {
  constexpr velox::vector_size_t kRows = 2;
  auto nestedType = velox::ROW({{"score", velox::BIGINT()}});
  auto type = velox::ROW({
      {"events", velox::ARRAY(nestedType)},
  });

  SerializerOptions options{
      .version = SerializationVersion::kSerialization,
      .streamIndicesEncodingType = EncodingType::Trivial,
      .streamSizesEncodingType = EncodingType::Trivial,
  };
  Serializer serializer{options, type, pool_.get()};
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      DeserializerOptions{.hasHeader = true}};

  auto makeInput = [&](int64_t scoreBase,
                       const std::array<velox::vector_size_t, kRows>& sizes,
                       const std::vector<velox::vector_size_t>& nullElements)
      -> velox::VectorPtr {
    const velox::vector_size_t totalElements =
        std::accumulate(sizes.begin(), sizes.end(), velox::vector_size_t{0});
    auto scores =
        velox::BaseVector::create(velox::BIGINT(), totalElements, pool_.get());
    for (velox::vector_size_t i = 0; i < totalElements; ++i) {
      scores->asFlatVector<int64_t>()->set(i, scoreBase + i);
    }

    auto nested = std::make_shared<velox::RowVector>(
        pool_.get(),
        nestedType,
        nullptr,
        totalElements,
        std::vector<velox::VectorPtr>{scores});
    for (auto element : nullElements) {
      nested->setNull(element, true);
    }

    auto offsets = velox::allocateOffsets(kRows, pool_.get());
    auto arraySizes = velox::allocateSizes(kRows, pool_.get());
    auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
    auto* rawSizes = arraySizes->asMutable<velox::vector_size_t>();
    velox::vector_size_t runningOffset = 0;
    for (velox::vector_size_t i = 0; i < kRows; ++i) {
      rawOffsets[i] = runningOffset;
      rawSizes[i] = sizes[i];
      runningOffset += sizes[i];
    }

    auto events = std::make_shared<velox::ArrayVector>(
        pool_.get(),
        velox::ARRAY(nestedType),
        nullptr,
        kRows,
        offsets,
        arraySizes,
        nested);
    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRows,
        std::vector<velox::VectorPtr>{events});
  };

  struct TestCase {
    std::string name;
    std::vector<velox::vector_size_t> firstBatchNullElements;
    std::vector<velox::vector_size_t> secondBatchNullElements;
  };

  const std::array<velox::vector_size_t, kRows> firstBatchSizes = {1, 1};
  const std::array<velox::vector_size_t, kRows> secondBatchSizes = {2, 1};
  const std::vector<TestCase> testCases = {
      {
          .name = "gap before nested array row null stream",
          .firstBatchNullElements = {},
          .secondBatchNullElements = {1},
      },
      {
          .name = "gap after nested array row null stream",
          .firstBatchNullElements = {0},
          .secondBatchNullElements = {},
      },
      {
          .name = "nested array row null stream all true in all batches",
          .firstBatchNullElements = {},
          .secondBatchNullElements = {},
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    auto first =
        makeInput(100, firstBatchSizes, testCase.firstBatchNullElements);
    auto second =
        makeInput(200, secondBatchSizes, testCase.secondBatchNullElements);

    std::vector<std::string> serialized;
    serialized.emplace_back(
        serializer.serialize(first, OrderedRanges::of(0, first->size())));
    serialized.emplace_back(
        serializer.serialize(second, OrderedRanges::of(0, second->size())));
    std::vector<std::string_view> serializedViews;
    serializedViews.reserve(serialized.size());
    for (const auto& data : serialized) {
      serializedViews.push_back(data);
    }

    auto expected = first;
    const auto firstRows = first->size();
    expected->resize(firstRows + second->size());
    expected->copy(second.get(), firstRows, 0, second->size());

    velox::VectorPtr output;
    deserializer.deserialize(serializedViews, output);

    ASSERT_EQ(output->size(), expected->size());
    for (velox::vector_size_t i = 0; i < expected->size(); ++i) {
      EXPECT_TRUE(vectorEquals(expected, output, i))
          << "Content mismatch at row " << i
          << "\nExpected: " << expected->toString(i)
          << "\nActual: " << output->toString(i);
    }
  }
}

TEST_P(SerializationTest, flatMapSparseKeysScatterBitmap) {
  // Test that FlatMap values are placed at correct positions when some rows
  // don't have certain keys. This exercises the scatterBitmap code path in
  // deserialization - values must be scattered to correct positions based on
  // the inMap bitmap rather than written contiguously.
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  // Create a batch where:
  // - Row 0: has key 1 only
  // - Row 1: has keys 1 and 2
  // - Row 2: has key 1 only
  // - Row 3: has keys 1 and 2
  // - Row 4: has key 1 only
  //
  // For key 2, inMap = [false, true, false, true, false]
  // Values for key 2: [v1, v2] (only 2 values in stream)
  // Expected output positions: row 1 gets v1, row 3 gets v2
  // Without scatterBitmap: row 0 gets v1, row 1 gets v2 (WRONG!)

  const velox::vector_size_t numRows = 5;

  auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ids->asFlatVector<int64_t>()->set(i, i * 100);
  }

  // Build map data: rows 0,2,4 have 1 key; rows 1,3 have 2 keys.
  // Total entries: 3*1 + 2*2 = 7
  std::vector<int32_t> keysData;
  std::vector<double> valuesData;
  std::vector<velox::vector_size_t> offsets;
  offsets.push_back(0);

  // Row 0: key 1 only
  keysData.push_back(1);
  valuesData.push_back(10.0);
  offsets.push_back(keysData.size());

  // Row 1: keys 1 and 2
  keysData.push_back(1);
  valuesData.push_back(11.0);
  keysData.push_back(2);
  valuesData.push_back(21.0); // This should end up at row 1 position for key 2
  offsets.push_back(keysData.size());

  // Row 2: key 1 only
  keysData.push_back(1);
  valuesData.push_back(12.0);
  offsets.push_back(keysData.size());

  // Row 3: keys 1 and 2
  keysData.push_back(1);
  valuesData.push_back(13.0);
  keysData.push_back(2);
  valuesData.push_back(23.0); // This should end up at row 3 position for key 2
  offsets.push_back(keysData.size());

  // Row 4: key 1 only
  keysData.push_back(1);
  valuesData.push_back(14.0);
  offsets.push_back(keysData.size());

  const auto totalEntries = static_cast<velox::vector_size_t>(keysData.size());

  auto keysFlat =
      velox::BaseVector::create(velox::INTEGER(), totalEntries, pool_.get());
  auto valuesFlat =
      velox::BaseVector::create(velox::DOUBLE(), totalEntries, pool_.get());

  for (velox::vector_size_t i = 0; i < totalEntries; ++i) {
    keysFlat->asFlatVector<int32_t>()->set(i, keysData[i]);
    valuesFlat->asFlatVector<double>()->set(i, valuesData[i]);
  }

  auto mapVector = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(velox::INTEGER(), velox::DOUBLE()),
      nullptr,
      numRows,
      velox::allocateOffsets(numRows, pool_.get()),
      velox::allocateSizes(numRows, pool_.get()),
      keysFlat,
      valuesFlat);

  auto* rawOffsets =
      mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
  auto* rawSizes =
      mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = offsets[i];
    rawSizes[i] = offsets[i + 1] - offsets[i];
  }

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{ids, mapVector});

  // Serialize with FlatMap encoding
  const SerializerOptions options{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{options, type, pool_.get()};
  auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  // Deserialize
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  // Verify output matches input
  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ASSERT_TRUE(vectorEquals(output, input, i))
        << "Content mismatch at row " << i
        << "\nExpected: " << input->toString(i)
        << "\nActual: " << output->toString(i)
        << "\nThis may indicate scatterBitmap is not being handled correctly.";
  }
}

TEST_P(SerializationTest, deserializeRejectsEmptyBatchList) {
  auto type = velox::ROW({{"int_val", velox::INTEGER()}});
  SerializerOptions serializerOptions{};
  Serializer serializer{serializerOptions, type, pool_.get()};
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get()};

  velox::VectorPtr output;
  NIMBLE_ASSERT_THROW(
      deserializer.deserialize(std::vector<std::string_view>{}, output),
      "Expected at least one serialized batch");
}

TEST_P(SerializationTest, unsupportedWriterVersionsRejected) {
  auto type = velox::ROW({{"int_val", velox::INTEGER()}});
  struct TestCase {
    SerializationVersion version;
    std::string_view expectedMessage;
  };

  const std::vector<TestCase> testCases{
      TestCase{
          .version = SerializationVersion::kProjection,
          .expectedMessage =
              "Serializer writes must use kLegacy or kSerialization"},
      TestCase{
          .version = SerializationVersion::kTablet,
          .expectedMessage =
              "Serializer writes must use kLegacy or kSerialization"},
  };
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(fmt::format("version={}", toString(testCase.version)));
    SerializerOptions options{.version = testCase.version};
    NIMBLE_ASSERT_THROW(
        Serializer(options, type, pool_.get()), testCase.expectedMessage);
  }
}

TEST_F(
    SerializationTest,
    serializerNormalizesLegacyWriterVersionsToSerialization) {
  auto type = velox::ROW({{"x", velox::INTEGER()}});
  auto col = velox::BaseVector::create(velox::INTEGER(), 1, pool_.get());
  col->asFlatVector<int32_t>()->set(0, 42);
  auto row = std::make_shared<velox::RowVector>(
      pool_.get(), type, nullptr, 1, std::vector<velox::VectorPtr>{col});

  const std::vector<SerializationVersion> versions{
      SerializationVersion::kLegacy,
      SerializationVersion::kLegacyCompact,
      SerializationVersion::kLegacySerialization};
  for (const auto version : versions) {
    SCOPED_TRACE(toString(version));
    SerializerOptions options{.version = version};
    Serializer serializer{options, type, pool_.get()};
    std::string blob;
    serializer.serialize(row, OrderedRanges::of(0, 1), blob);
    ASSERT_FALSE(blob.empty());
    const char* pos = blob.data();
    const auto header =
        serde::readSerializationHeader(pos, blob.data() + blob.size(), true);
    EXPECT_EQ(header.version, SerializationVersion::kSerialization);
    EXPECT_EQ(header.rowCount, 1);
  }
}

TEST_F(SerializationTest, serializerDefaultWritesNoHeaderLegacy) {
  auto type = velox::ROW({{"x", velox::INTEGER()}});
  auto col = velox::BaseVector::create(velox::INTEGER(), 1, pool_.get());
  col->asFlatVector<int32_t>()->set(0, 42);
  auto row = std::make_shared<velox::RowVector>(
      pool_.get(), type, nullptr, 1, std::vector<velox::VectorPtr>{col});

  Serializer serializer{SerializerOptions{}, type, pool_.get()};
  std::string blob;
  serializer.serialize(row, OrderedRanges::of(0, 1), blob);
  ASSERT_FALSE(blob.empty());

  const char* pos = blob.data();
  const auto header =
      serde::readSerializationHeader(pos, blob.data() + blob.size(), false);
  EXPECT_EQ(header.version, SerializationVersion::kLegacy);
  EXPECT_EQ(header.rowCount, 1);
  EXPECT_EQ(pos - blob.data(), sizeof(uint32_t));

  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get()};
  velox::VectorPtr output;
  deserializer.deserialize(blob, output);
  ASSERT_EQ(output->size(), row->size());
  EXPECT_TRUE(vectorEquals(output, row, 0));
}

// Test encoding layout tree for non-FlatMap types.
TEST_P(SerializationTest, encodingLayoutTree) {
  // ROW type with scalar, string, and nested Map(Array(Array(Int))) children.
  auto type = velox::ROW({
      {"int_val", velox::INTEGER()},
      {"double_val", velox::DOUBLE()},
      {"string_val", velox::VARCHAR()},
      {"nested_map",
       velox::MAP(
           velox::VARCHAR(), velox::ARRAY(velox::ARRAY(velox::INTEGER())))},
  });

  // Create encoding layout tree with specific encodings.
  EncodingLayoutTree layoutTree{
      Kind::Row,
      {{EncodingLayoutTree::StreamIdentifiers::Row::NullsStream,
        EncodingLayout{
            EncodingType::SparseBool, {}, CompressionType::Uncompressed}}},
      "",
      {
          // int_val child
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::FixedBitWidth,
                 {},
                 CompressionType::Uncompressed}}},
           ""},
          // double_val child
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::Trivial, {}, CompressionType::Uncompressed}}},
           ""},
          // string_val child
          // Trivial encoding for strings needs a child encoding for lengths.
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::Trivial,
                 {},
                 CompressionType::Uncompressed,
                 {EncodingLayout{
                     EncodingType::Trivial,
                     {},
                     CompressionType::Uncompressed}}}}},
           ""},
          // nested_map: MAP(VARCHAR, ARRAY(ARRAY(INTEGER)))
          {Kind::Map,
           {{EncodingLayoutTree::StreamIdentifiers::Map::LengthsStream,
             EncodingLayout{
                 EncodingType::FixedBitWidth,
                 {},
                 CompressionType::Uncompressed}}},
           "",
           {
               // Key child (VARCHAR)
               // Trivial encoding for strings needs a child encoding for
               // lengths.
               {Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::Trivial,
                      {},
                      CompressionType::Uncompressed,
                      {EncodingLayout{
                          EncodingType::Trivial,
                          {},
                          CompressionType::Uncompressed}}}}},
                ""},
               // Value child: ARRAY(ARRAY(INTEGER))
               {Kind::Array,
                {{EncodingLayoutTree::StreamIdentifiers::Array::LengthsStream,
                  EncodingLayout{
                      EncodingType::FixedBitWidth,
                      {},
                      CompressionType::Uncompressed}}},
                "",
                {
                    // Inner ARRAY(INTEGER)
                    {Kind::Array,
                     {{EncodingLayoutTree::StreamIdentifiers::Array::
                           LengthsStream,
                       EncodingLayout{
                           EncodingType::FixedBitWidth,
                           {},
                           CompressionType::Uncompressed}}},
                     "",
                     {
                         // INTEGER element
                         {Kind::Scalar,
                          {{EncodingLayoutTree::StreamIdentifiers::Scalar::
                                ScalarStream,
                            EncodingLayout{
                                EncodingType::FixedBitWidth,
                                {},
                                CompressionType::Uncompressed}}},
                          ""},
                     }},
                }},
           }},
      }};

  SerializerOptions options{
      .version = version(),
      .encodingLayoutTree = layoutTree,
      .compressionOptions = compressionOptions(),
  };

  Serializer serializer{options, type, pool_.get()};

  // Generate test data.
  const velox::vector_size_t numRows = 5;

  // Pre-generate string values to avoid dangling StringViews.
  std::vector<std::string> stringData;
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    stringData.push_back("str" + std::to_string(i));
  }

  auto intVals =
      velox::BaseVector::create(velox::INTEGER(), numRows, pool_.get());
  auto doubleVals =
      velox::BaseVector::create(velox::DOUBLE(), numRows, pool_.get());
  auto stringVals =
      velox::BaseVector::create(velox::VARCHAR(), numRows, pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    intVals->asFlatVector<int32_t>()->set(i, i * 100);
    doubleVals->asFlatVector<double>()->set(i, i * 1.5);
    stringVals->asFlatVector<velox::StringView>()->set(
        i, velox::StringView(stringData[i]));
  }

  // Build nested_map: MAP(VARCHAR, ARRAY(ARRAY(INTEGER)))
  // Each row has 2 map entries, each entry has outer array of size 2,
  // each inner array has 3 integers.
  const velox::vector_size_t mapEntriesPerRow = 2;
  const velox::vector_size_t outerArraySize = 2;
  const velox::vector_size_t innerArraySize = 3;
  const velox::vector_size_t totalMapEntries = numRows * mapEntriesPerRow;
  const velox::vector_size_t totalOuterArrays =
      totalMapEntries * outerArraySize;
  const velox::vector_size_t totalInnerInts = totalOuterArrays * innerArraySize;

  // Build innermost integers
  auto innerInts =
      velox::BaseVector::create(velox::INTEGER(), totalInnerInts, pool_.get());
  for (velox::vector_size_t i = 0; i < totalInnerInts; ++i) {
    innerInts->asFlatVector<int32_t>()->set(i, i);
  }

  // Build inner arrays (ARRAY(INTEGER))
  auto innerArrayOffsets =
      velox::allocateOffsets(totalOuterArrays, pool_.get());
  auto innerArraySizes = velox::allocateSizes(totalOuterArrays, pool_.get());
  auto* innerOffsets = innerArrayOffsets->asMutable<velox::vector_size_t>();
  auto* innerSizes = innerArraySizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < totalOuterArrays; ++i) {
    innerOffsets[i] = i * innerArraySize;
    innerSizes[i] = innerArraySize;
  }
  auto innerArrayVector = std::make_shared<velox::ArrayVector>(
      pool_.get(),
      velox::ARRAY(velox::INTEGER()),
      nullptr,
      totalOuterArrays,
      innerArrayOffsets,
      innerArraySizes,
      innerInts);

  // Build outer arrays (ARRAY(ARRAY(INTEGER)))
  auto outerArrayOffsets = velox::allocateOffsets(totalMapEntries, pool_.get());
  auto outerArraySizes = velox::allocateSizes(totalMapEntries, pool_.get());
  auto* outerOffsets = outerArrayOffsets->asMutable<velox::vector_size_t>();
  auto* outerSizes = outerArraySizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < totalMapEntries; ++i) {
    outerOffsets[i] = i * outerArraySize;
    outerSizes[i] = outerArraySize;
  }
  auto outerArrayVector = std::make_shared<velox::ArrayVector>(
      pool_.get(),
      velox::ARRAY(velox::ARRAY(velox::INTEGER())),
      nullptr,
      totalMapEntries,
      outerArrayOffsets,
      outerArraySizes,
      innerArrayVector);

  // Build map keys (VARCHAR)
  std::vector<std::string> mapKeyData;
  for (velox::vector_size_t i = 0; i < totalMapEntries; ++i) {
    mapKeyData.push_back("key" + std::to_string(i));
  }
  auto mapKeys =
      velox::BaseVector::create(velox::VARCHAR(), totalMapEntries, pool_.get());
  for (velox::vector_size_t i = 0; i < totalMapEntries; ++i) {
    mapKeys->asFlatVector<velox::StringView>()->set(
        i, velox::StringView(mapKeyData[i]));
  }

  // Build map vector
  auto mapOffsets = velox::allocateOffsets(numRows, pool_.get());
  auto mapSizes = velox::allocateSizes(numRows, pool_.get());
  auto* rawMapOffsets = mapOffsets->asMutable<velox::vector_size_t>();
  auto* rawMapSizes = mapSizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawMapOffsets[i] = i * mapEntriesPerRow;
    rawMapSizes[i] = mapEntriesPerRow;
  }
  auto mapVector = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(
          velox::VARCHAR(), velox::ARRAY(velox::ARRAY(velox::INTEGER()))),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      outerArrayVector);

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{
          intVals, doubleVals, stringVals, mapVector});

  // Serialize.
  auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  // Deserialize and verify.
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < input->size(); ++i) {
    ASSERT_TRUE(vectorEquals(output, input, i))
        << "Content mismatch at index " << i;
  }
}

// Columnar reader-level round-trip for PFOREncoding. Forces PFOR via
// EncodingLayoutTree and verifies data integrity through the full
// Serializer → Deserializer path.
TEST_P(SerializationTest, pforColumnarRoundTrip) {
  auto type = velox::ROW({
      {"int_val", velox::INTEGER()},
      {"long_val", velox::BIGINT()},
  });

  EncodingLayoutTree layoutTree{
      Kind::Row,
      {},
      "",
      {
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::PFOR,
                 {},
                 CompressionType::Uncompressed,
                 {std::nullopt, std::nullopt}}}},
           ""},
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::PFOR,
                 {},
                 CompressionType::Uncompressed,
                 {std::nullopt, std::nullopt}}}},
           ""},
      }};

  SerializerOptions options{
      .version = version(),
      .encodingLayoutTree = layoutTree,
      .compressionOptions = compressionOptions(),
  };

  Serializer serializer{options, type, pool_.get()};

  const velox::vector_size_t numRows = 100;
  auto intVals =
      velox::BaseVector::create(velox::INTEGER(), numRows, pool_.get());
  auto longVals =
      velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    // ~90% narrow values, ~10% outliers — Pfor's sweet spot.
    intVals->asFlatVector<int32_t>()->set(
        i, i % 10 == 0 ? 1'000'000 + i : i % 100);
    longVals->asFlatVector<int64_t>()->set(
        i, i % 10 == 0 ? 9'000'000'000LL + i : i * 60);
  }

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{intVals, longVals});

  auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < input->size(); ++i) {
    ASSERT_TRUE(vectorEquals(output, input, i))
        << "Content mismatch at index " << i;
  }
}

// Columnar reader-level round-trip for BlockBitPackingEncoding. Forces
// BlockBitPacking via EncodingLayoutTree on integer columns and verifies data
// integrity through the full Serializer → Deserializer path, exercising the
// nested per-block metadata sub-streams (baselines, bit widths, data offsets).
TEST_P(SerializationTest, blockBitPackingColumnarRoundTrip) {
  auto type = velox::ROW({
      {"int_val", velox::INTEGER()},
      {"long_val", velox::BIGINT()},
  });

  EncodingLayoutTree layoutTree{
      Kind::Row,
      {},
      "",
      {
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::BlockBitPacking,
                 {},
                 CompressionType::Uncompressed,
                 {std::nullopt, std::nullopt, std::nullopt}}}},
           ""},
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::BlockBitPacking,
                 {},
                 CompressionType::Uncompressed,
                 {std::nullopt, std::nullopt, std::nullopt}}}},
           ""},
      }};

  SerializerOptions options{
      .version = version(),
      .encodingLayoutTree = layoutTree,
      .compressionOptions = compressionOptions(),
  };

  Serializer serializer{options, type, pool_.get()};

  // Enough rows to span multiple blocks; per-block-varying ranges so blocks get
  // distinct baselines and bit widths.
  const velox::vector_size_t numRows = 3000;
  auto intVals =
      velox::BaseVector::create(velox::INTEGER(), numRows, pool_.get());
  auto longVals =
      velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    intVals->asFlatVector<int32_t>()->set(i, (i / 256) * 1000 + (i % 37));
    longVals->asFlatVector<int64_t>()->set(
        i, (i / 256) * 1'000'000LL + (i % 37));
  }

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{intVals, longVals});

  auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < input->size(); ++i) {
    ASSERT_TRUE(vectorEquals(output, input, i))
        << "Content mismatch at index " << i;
  }
}

// Columnar reader-level round-trip for SimdForBitpackEncoding. Forces
// SimdForBitpack via EncodingLayoutTree on integer columns and verifies data
// integrity through the full Serializer → Deserializer path.
TEST_P(SerializationTest, simdForBitpackColumnarRoundTrip) {
  auto type = velox::ROW({
      {"int_val", velox::INTEGER()},
      {"long_val", velox::BIGINT()},
  });

  EncodingLayoutTree layoutTree{
      Kind::Row,
      {},
      "",
      {
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::SimdForBitpack,
                 {},
                 CompressionType::Uncompressed}}},
           ""},
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::SimdForBitpack,
                 {},
                 CompressionType::Uncompressed}}},
           ""},
      }};

  SerializerOptions options{
      .version = version(),
      .encodingLayoutTree = layoutTree,
      .compressionOptions = compressionOptions(),
  };

  Serializer serializer{options, type, pool_.get()};

  const velox::vector_size_t numRows = 100;
  auto intVals =
      velox::BaseVector::create(velox::INTEGER(), numRows, pool_.get());
  auto longVals =
      velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    intVals->asFlatVector<int32_t>()->set(i, 1000 + i % 50);
    longVals->asFlatVector<int64_t>()->set(i, 5000LL + i * 3);
  }

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{intVals, longVals});

  auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < input->size(); ++i) {
    ASSERT_TRUE(vectorEquals(output, input, i))
        << "Content mismatch at index " << i;
  }
}

// Test encoding layout tree for FlatMap with dynamic key discovery and nested
// types.
TEST_P(SerializationTest, encodingLayoutTreeFlatMap) {
  // Use nested types to test:
  // - features: MAP(INTEGER, ARRAY(DOUBLE)) - FlatMap with nested array values
  // - tags: MAP(VARCHAR, DOUBLE) - FlatMap with string keys and scalar values
  // - metadata: MAP(INTEGER, VARCHAR) - Regular Map (not FlatMap)
  const auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::ARRAY(velox::DOUBLE()))},
      {"tags", velox::MAP(velox::VARCHAR(), velox::DOUBLE())},
      {"metadata", velox::MAP(velox::INTEGER(), velox::VARCHAR())},
  });

  // Create encoding layout tree with FlatMap encodings.
  // The FlatMap children are keyed by the key value as string.
  EncodingLayoutTree layoutTree{
      Kind::Row,
      {{EncodingLayoutTree::StreamIdentifiers::Row::NullsStream,
        EncodingLayout{
            EncodingType::SparseBool, {}, CompressionType::Uncompressed}}},
      "",
      {
          // id child (scalar)
          {Kind::Scalar,
           {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
             EncodingLayout{
                 EncodingType::FixedBitWidth,
                 {},
                 CompressionType::Uncompressed}}},
           ""},
          // features child (FlatMap with ARRAY(DOUBLE) values)
          {Kind::FlatMap,
           {{EncodingLayoutTree::StreamIdentifiers::FlatMap::NullsStream,
             EncodingLayout{
                 EncodingType::SparseBool, {}, CompressionType::Uncompressed}}},
           "",
           {
               // Key "1" - ARRAY(DOUBLE) value encoding
               {Kind::Array,
                {{EncodingLayoutTree::StreamIdentifiers::Array::LengthsStream,
                  EncodingLayout{
                      EncodingType::FixedBitWidth,
                      {},
                      CompressionType::Uncompressed}}},
                "1",
                {
                    // Array element (DOUBLE)
                    {Kind::Scalar,
                     {{EncodingLayoutTree::StreamIdentifiers::Scalar::
                           ScalarStream,
                       EncodingLayout{
                           EncodingType::Trivial,
                           {},
                           CompressionType::Uncompressed}}},
                     ""},
                }},
               // Key "2" - ARRAY(DOUBLE) value encoding
               {Kind::Array,
                {{EncodingLayoutTree::StreamIdentifiers::Array::LengthsStream,
                  EncodingLayout{
                      EncodingType::FixedBitWidth,
                      {},
                      CompressionType::Uncompressed}}},
                "2",
                {
                    // Array element (DOUBLE)
                    {Kind::Scalar,
                     {{EncodingLayoutTree::StreamIdentifiers::Scalar::
                           ScalarStream,
                       EncodingLayout{
                           EncodingType::FixedBitWidth,
                           {},
                           CompressionType::Uncompressed}}},
                     ""},
                }},
           }},
          // tags child (FlatMap with VARCHAR keys and DOUBLE values)
          {Kind::FlatMap,
           {{EncodingLayoutTree::StreamIdentifiers::FlatMap::NullsStream,
             EncodingLayout{
                 EncodingType::SparseBool, {}, CompressionType::Uncompressed}}},
           "",
           {
               // Key "tag_a" - DOUBLE value encoding
               {Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::Trivial,
                      {},
                      CompressionType::Uncompressed}}},
                "tag_a"},
               // Key "tag_b" - DOUBLE value encoding
               {Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::FixedBitWidth,
                      {},
                      CompressionType::Uncompressed}}},
                "tag_b"},
           }},
          // metadata child (regular Map, not FlatMap)
          {Kind::Map,
           {{EncodingLayoutTree::StreamIdentifiers::Map::LengthsStream,
             EncodingLayout{
                 EncodingType::FixedBitWidth,
                 {},
                 CompressionType::Uncompressed}}},
           "",
           {
               // Key child (INTEGER)
               {Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::FixedBitWidth,
                      {},
                      CompressionType::Uncompressed}}},
                ""},
               // Value child (VARCHAR)
               // Trivial encoding for strings needs a child encoding for
               // lengths.
               {Kind::Scalar,
                {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                  EncodingLayout{
                      EncodingType::Trivial,
                      {},
                      CompressionType::Uncompressed,
                      {EncodingLayout{
                          EncodingType::Trivial,
                          {},
                          CompressionType::Uncompressed}}}}},
                ""},
           }},
      }};

  SerializerOptions options{
      .version = version(),
      .flatMapColumns = {{"features", {}}, {"tags", {}}},
      .encodingLayoutTree = layoutTree,
      .compressionOptions = compressionOptions(),
  };

  Serializer serializer{options, type, pool_.get()};

  // Generate test data with FlatMap keys 1 and 2, each with ARRAY(DOUBLE)
  // values.
  const velox::vector_size_t numRows = 5;

  auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ids->asFlatVector<int64_t>()->set(i, i * 100);
  }

  // Build map: each row has keys 1 and 2, each with an array of 2 doubles.
  // Map structure: key -> ARRAY(DOUBLE)
  const velox::vector_size_t entriesPerRow = 2; // keys 1 and 2
  const velox::vector_size_t elementsPerArray = 2;
  const velox::vector_size_t totalMapEntries = numRows * entriesPerRow;
  const velox::vector_size_t totalArrayElements =
      totalMapEntries * elementsPerArray;

  // Build map keys
  auto mapKeys =
      velox::BaseVector::create(velox::INTEGER(), totalMapEntries, pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    mapKeys->asFlatVector<int32_t>()->set(i * 2, 1); // key 1
    mapKeys->asFlatVector<int32_t>()->set(i * 2 + 1, 2); // key 2
  }

  // Build array elements (doubles)
  auto arrayElements = velox::BaseVector::create(
      velox::DOUBLE(), totalArrayElements, pool_.get());
  for (velox::vector_size_t i = 0; i < totalArrayElements; ++i) {
    arrayElements->asFlatVector<double>()->set(i, i * 1.5);
  }

  // Build array offsets and sizes (each array has elementsPerArray elements)
  auto arrayOffsetsBuffer =
      velox::allocateOffsets(totalMapEntries, pool_.get());
  auto arraySizesBuffer = velox::allocateSizes(totalMapEntries, pool_.get());
  auto* arrayOffsets = arrayOffsetsBuffer->asMutable<velox::vector_size_t>();
  auto* arraySizes = arraySizesBuffer->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < totalMapEntries; ++i) {
    arrayOffsets[i] = i * elementsPerArray;
    arraySizes[i] = elementsPerArray;
  }

  // Build array vector for map values
  auto mapValues = std::make_shared<velox::ArrayVector>(
      pool_.get(),
      velox::ARRAY(velox::DOUBLE()),
      nullptr,
      totalMapEntries,
      arrayOffsetsBuffer,
      arraySizesBuffer,
      arrayElements);

  // Build map offsets and sizes (each row has entriesPerRow map entries)
  auto mapOffsetsBuffer = velox::allocateOffsets(numRows, pool_.get());
  auto mapSizesBuffer = velox::allocateSizes(numRows, pool_.get());
  auto* mapOffsets = mapOffsetsBuffer->asMutable<velox::vector_size_t>();
  auto* mapSizes = mapSizesBuffer->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    mapOffsets[i] = i * entriesPerRow;
    mapSizes[i] = entriesPerRow;
  }

  auto featuresMap = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(velox::INTEGER(), velox::ARRAY(velox::DOUBLE())),
      nullptr,
      numRows,
      mapOffsetsBuffer,
      mapSizesBuffer,
      mapKeys,
      mapValues);

  // Build tags FlatMap: MAP(VARCHAR, DOUBLE) with keys "tag_a" and "tag_b"
  std::vector<std::string> tagKeyStrings;
  tagKeyStrings.reserve(totalMapEntries);
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    tagKeyStrings.push_back("tag_a");
    tagKeyStrings.push_back("tag_b");
  }
  auto tagKeys =
      velox::BaseVector::create(velox::VARCHAR(), totalMapEntries, pool_.get());
  for (size_t i = 0; i < tagKeyStrings.size(); ++i) {
    tagKeys->asFlatVector<velox::StringView>()->set(
        i, velox::StringView(tagKeyStrings[i]));
  }
  auto tagValues =
      velox::BaseVector::create(velox::DOUBLE(), totalMapEntries, pool_.get());
  for (velox::vector_size_t i = 0; i < totalMapEntries; ++i) {
    tagValues->asFlatVector<double>()->set(i, i * 2.5);
  }
  auto tagOffsetsBuffer = velox::allocateOffsets(numRows, pool_.get());
  auto tagSizesBuffer = velox::allocateSizes(numRows, pool_.get());
  auto* tagOffsets = tagOffsetsBuffer->asMutable<velox::vector_size_t>();
  auto* tagSizes = tagSizesBuffer->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    tagOffsets[i] = i * entriesPerRow;
    tagSizes[i] = entriesPerRow;
  }
  auto tagsMap = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(velox::VARCHAR(), velox::DOUBLE()),
      nullptr,
      numRows,
      tagOffsetsBuffer,
      tagSizesBuffer,
      tagKeys,
      tagValues);

  // Build metadata regular Map: MAP(INTEGER, VARCHAR)
  std::vector<std::string> metaValueStrings;
  metaValueStrings.reserve(totalMapEntries);
  for (velox::vector_size_t i = 0; i < totalMapEntries; ++i) {
    metaValueStrings.push_back("meta_" + std::to_string(i));
  }
  auto metaKeys =
      velox::BaseVector::create(velox::INTEGER(), totalMapEntries, pool_.get());
  auto metaValues =
      velox::BaseVector::create(velox::VARCHAR(), totalMapEntries, pool_.get());
  for (velox::vector_size_t i = 0; i < totalMapEntries; ++i) {
    metaKeys->asFlatVector<int32_t>()->set(i, i);
    metaValues->asFlatVector<velox::StringView>()->set(
        i, velox::StringView(metaValueStrings[i]));
  }
  auto metaOffsetsBuffer = velox::allocateOffsets(numRows, pool_.get());
  auto metaSizesBuffer = velox::allocateSizes(numRows, pool_.get());
  auto* metaOffsets = metaOffsetsBuffer->asMutable<velox::vector_size_t>();
  auto* metaSizes = metaSizesBuffer->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    metaOffsets[i] = i * entriesPerRow;
    metaSizes[i] = entriesPerRow;
  }
  auto metadataMap = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(velox::INTEGER(), velox::VARCHAR()),
      nullptr,
      numRows,
      metaOffsetsBuffer,
      metaSizesBuffer,
      metaKeys,
      metaValues);

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{ids, featuresMap, tagsMap, metadataMap});

  // Serialize.
  auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  // Deserialize and verify.
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserializerOptions()};

  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < input->size(); ++i) {
    ASSERT_TRUE(vectorEquals(output, input, i))
        << "Content mismatch at index " << i;
  }
}

// Test that wrong encoding layout tree kind throws an error.
TEST_P(SerializationTest, encodingLayoutTreeWrongKind) {
  // Test: Pass Row layout for a Scalar column.
  auto type = velox::ROW({{"int_val", velox::INTEGER()}});

  // Create encoding layout tree with wrong kind (Row instead of Scalar).
  EncodingLayoutTree layoutTree{
      Kind::Row,
      {{EncodingLayoutTree::StreamIdentifiers::Row::NullsStream,
        EncodingLayout{
            EncodingType::SparseBool, {}, CompressionType::Uncompressed}}},
      "",
      {
          // int_val child - WRONG: using Row layout instead of Scalar
          {Kind::Row,
           {{EncodingLayoutTree::StreamIdentifiers::Row::NullsStream,
             EncodingLayout{
                 EncodingType::SparseBool, {}, CompressionType::Uncompressed}}},
           ""},
      }};

  SerializerOptions options{
      .version = version(),
      .encodingLayoutTree = layoutTree,
  };

  NIMBLE_ASSERT_THROW(
      Serializer(options, type, pool_.get()),
      "Incompatible encoding layout node. Expecting scalar node.");
}

// Test that FlatMap vs Map mismatch throws an error.
TEST_P(SerializationTest, encodingLayoutTreeFlatMapForMap) {
  // Test: Pass FlatMap layout for a Map column.
  auto type =
      velox::ROW({{"map_val", velox::MAP(velox::INTEGER(), velox::DOUBLE())}});

  // Create encoding layout tree with FlatMap kind for a Map column.
  EncodingLayoutTree layoutTree{
      Kind::Row,
      {{EncodingLayoutTree::StreamIdentifiers::Row::NullsStream,
        EncodingLayout{
            EncodingType::SparseBool, {}, CompressionType::Uncompressed}}},
      "",
      {
          // map_val child - WRONG: using FlatMap layout for Map column
          {Kind::FlatMap,
           {{EncodingLayoutTree::StreamIdentifiers::FlatMap::NullsStream,
             EncodingLayout{
                 EncodingType::SparseBool, {}, CompressionType::Uncompressed}}},
           ""},
      }};

  SerializerOptions options{
      .version = version(),
      .encodingLayoutTree = layoutTree,
  };

  NIMBLE_ASSERT_THROW(
      Serializer(options, type, pool_.get()),
      "Incompatible encoding layout node. Expecting map node.");
}

// Test that Map vs FlatMap mismatch throws an error.
TEST_P(SerializationTest, encodingLayoutTreeMapForFlatMap) {
  // Test: Pass Map layout for a FlatMap column.
  const auto type =
      velox::ROW({{"map_val", velox::MAP(velox::INTEGER(), velox::DOUBLE())}});

  // Create encoding layout tree with Map kind for a FlatMap column.
  EncodingLayoutTree layoutTree{
      Kind::Row,
      {{EncodingLayoutTree::StreamIdentifiers::Row::NullsStream,
        EncodingLayout{
            EncodingType::SparseBool, {}, CompressionType::Uncompressed}}},
      "",
      {
          // map_val child - WRONG: using Map layout for FlatMap column
          {Kind::Map,
           {{EncodingLayoutTree::StreamIdentifiers::Map::LengthsStream,
             EncodingLayout{
                 EncodingType::FixedBitWidth,
                 {},
                 CompressionType::Uncompressed}}},
           ""},
      }};

  SerializerOptions options{
      .version = version(),
      .flatMapColumns = {{"map_val", {}}},
      .encodingLayoutTree = layoutTree,
  };

  NIMBLE_ASSERT_THROW(
      Serializer(options, type, pool_.get()),
      "Incompatible encoding layout node. Expecting flatmap node.");
}

TEST_P(SerializationTest, flatMapInMapStreamsForDiscoveredKeys) {
  // Test that serializer skips constant in-map boolean streams for discovered
  // FlatMap keys while preserving mixed in-map streams.
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"flat_map", velox::MAP(velox::VARCHAR(), velox::DOUBLE())},
  });

  const velox::vector_size_t batchSize = 10;

  auto generateRowsBatch =
      [&](const std::vector<std::vector<std::string>>& keysByRow)
      -> velox::VectorPtr {
    const auto numRows = static_cast<velox::vector_size_t>(keysByRow.size());
    velox::vector_size_t totalEntries = 0;
    for (const auto& keys : keysByRow) {
      totalEntries += static_cast<velox::vector_size_t>(keys.size());
    }

    auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
    auto mapKeys =
        velox::BaseVector::create(velox::VARCHAR(), totalEntries, pool_.get());
    auto mapValues =
        velox::BaseVector::create(velox::DOUBLE(), totalEntries, pool_.get());

    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
    }

    velox::vector_size_t idx = 0;
    for (velox::vector_size_t row = 0; row < numRows; ++row) {
      for (const auto& key : keysByRow[row]) {
        mapKeys->asFlatVector<velox::StringView>()->set(
            idx, velox::StringView(key));
        mapValues->asFlatVector<double>()->set(idx, row * 10.0 + idx);
        ++idx;
      }
    }

    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::VARCHAR(), velox::DOUBLE()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool_.get()),
        velox::allocateSizes(numRows, pool_.get()),
        mapKeys,
        mapValues);

    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    velox::vector_size_t offset = 0;
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = offset;
      rawSizes[i] = static_cast<velox::vector_size_t>(keysByRow[i].size());
      offset += rawSizes[i];
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{ids, mapVector});
  };
  auto generateBatch =
      [&](const std::vector<std::string>& keys) -> velox::VectorPtr {
    return generateRowsBatch(
        std::vector<std::vector<std::string>>(batchSize, keys));
  };

  SerializerOptions options{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"flat_map", {}}},
  };
  Serializer serializer{options, type, pool_.get()};

  // Batch 1: keys "a" and "b" (both in-map streams are all-true).
  auto batch1 = generateBatch({"a", "b"});
  auto serialized1 = std::string(
      serializer.serialize(batch1, OrderedRanges::of(0, batch1->size())));

  // Batch 2: only key "a" (key "b" absent -> all-false in-map).
  auto batch2 = generateBatch({"a"});
  auto serialized2 = std::string(
      serializer.serialize(batch2, OrderedRanges::of(0, batch2->size())));

  // Batch 3: keys "a", "b", and "c" (new key "c" discovered)
  auto batch3 = generateBatch({"a", "b", "c"});
  auto serialized3 = std::string(
      serializer.serialize(batch3, OrderedRanges::of(0, batch3->size())));

  // Batch 4: keys "a" and "b" again (key "a" and "b" all-true in-map, key "c"
  // all-false in-map).
  auto batch4 = generateBatch({"a", "b"});
  auto serialized4 = std::string(
      serializer.serialize(batch4, OrderedRanges::of(0, batch4->size())));

  std::vector<std::vector<std::string>> mixedKeys(batchSize);
  for (velox::vector_size_t i = 0; i < batchSize; ++i) {
    mixedKeys[i].emplace_back("a");
    if (i % 2 == 0) {
      mixedKeys[i].emplace_back("b");
    }
  }
  auto batch5 = generateRowsBatch(mixedKeys);
  auto serialized5 = std::string(
      serializer.serialize(batch5, OrderedRanges::of(0, batch5->size())));

  // Helper to collect stream offsets present in serialized data.
  auto collectStreamOffsets =
      [&](std::string_view data) -> folly::F14FastSet<uint32_t> {
    auto desOpts = deserializerOptions();
    serde::StreamDataParser reader{pool_.get(), desOpts};
    reader.initialize(data);
    folly::F14FastSet<uint32_t> offsets;
    reader.iterateStreams(
        [&](uint32_t offset, std::string_view) { offsets.insert(offset); });
    return offsets;
  };

  // Get in-map stream offsets from the schema.
  auto nimbleSchema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const auto& flatMap = nimbleSchema->asRow().childAt(1)->asFlatMap();
  std::vector<uint32_t> inMapOffsets;
  inMapOffsets.reserve(flatMap.childrenCount());
  for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
    inMapOffsets.push_back(flatMap.inMapDescriptorAt(i).offset());
  }
  // After batch 3, schema has keys "a", "b", "c": 3 in-map streams.
  ASSERT_EQ(inMapOffsets.size(), 3);

  // Verify constant in-map streams are omitted once their keys are discovered.
  {
    // Batch 1: keys "a" and "b" all-true. Key "c" is not discovered yet.
    auto offsets1 = collectStreamOffsets(serialized1);
    EXPECT_FALSE(offsets1.contains(inMapOffsets[0]))
        << "in-map 'a' should be absent (all-true)";
    EXPECT_FALSE(offsets1.contains(inMapOffsets[1]))
        << "in-map 'b' should be absent (all-true)";
    EXPECT_FALSE(offsets1.contains(inMapOffsets[2]))
        << "in-map 'c' should be absent before discovery";

    // Batch 2: key "a" all-true, key "b" all-false. Key "c" is not
    // discovered yet.
    auto offsets2 = collectStreamOffsets(serialized2);
    EXPECT_FALSE(offsets2.contains(inMapOffsets[0]))
        << "in-map 'a' should be absent (all-true)";
    EXPECT_FALSE(offsets2.contains(inMapOffsets[1]))
        << "in-map 'b' should be absent (all-false)";
    EXPECT_FALSE(offsets2.contains(inMapOffsets[2]))
        << "in-map 'c' should be absent before discovery";

    // Batch 3: keys "a", "b", "c" all present.
    auto offsets3 = collectStreamOffsets(serialized3);
    EXPECT_FALSE(offsets3.contains(inMapOffsets[0]))
        << "in-map 'a' should be absent (all-true)";
    EXPECT_FALSE(offsets3.contains(inMapOffsets[1]))
        << "in-map 'b' should be absent (all-true)";
    EXPECT_FALSE(offsets3.contains(inMapOffsets[2]))
        << "in-map 'c' should be absent (all-true)";

    // Batch 4: keys "a" and "b" all-true, key "c" all-false.
    auto offsets4 = collectStreamOffsets(serialized4);
    EXPECT_FALSE(offsets4.contains(inMapOffsets[0]))
        << "in-map 'a' should be absent (all-true)";
    EXPECT_FALSE(offsets4.contains(inMapOffsets[1]))
        << "in-map 'b' should be absent (all-true)";
    EXPECT_FALSE(offsets4.contains(inMapOffsets[2]))
        << "in-map 'c' should be absent (all-false)";

    // Batch 5: key "b" is mixed and must remain physically present. Key "a"
    // is all-true and key "c" is all-false.
    auto offsets5 = collectStreamOffsets(serialized5);
    EXPECT_FALSE(offsets5.contains(inMapOffsets[0]))
        << "in-map 'a' should be absent (all-true)";
    EXPECT_TRUE(offsets5.contains(inMapOffsets[1]))
        << "in-map 'b' should be written (mixed)";
    EXPECT_FALSE(offsets5.contains(inMapOffsets[2]))
        << "in-map 'c' should be absent (all-false)";
  }

  // Deserialize and verify correctness. Keys that were not discovered yet in
  // earlier batches are decoded as absent in those batches.
  Deserializer deserializer{nimbleSchema, pool_.get(), deserializerOptions()};

  // Verify each batch individually.
  std::vector<velox::VectorPtr> inputs = {
      batch1, batch2, batch3, batch4, batch5};
  std::vector<std::string> serializedData = {
      serialized1, serialized2, serialized3, serialized4, serialized5};

  velox::VectorPtr output;
  for (size_t i = 0; i < inputs.size(); ++i) {
    SCOPED_TRACE(fmt::format("batch {}", i));
    deserializer.deserialize(serializedData[i], output);
    ASSERT_EQ(output->size(), inputs[i]->size());
    for (velox::vector_size_t j = 0; j < inputs[i]->size(); ++j) {
      ASSERT_TRUE(vectorEquals(output, inputs[i], j))
          << "index " << j << "\nExpected: " << inputs[i]->toString(j)
          << "\nActual: " << output->toString(j);
    }
  }

  // Verify multi-batch deserialization.
  std::vector<std::string_view> allBatches;
  allBatches.reserve(serializedData.size());
  for (const auto& s : serializedData) {
    allBatches.push_back(s);
  }
  deserializer.deserialize(allBatches, output);

  velox::VectorPtr expected = inputs[0];
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto oldSize = expected->size();
    expected->resize(oldSize + inputs[i]->size());
    expected->copy(inputs[i].get(), oldSize, 0, inputs[i]->size());
  }

  ASSERT_EQ(output->size(), expected->size());
  for (velox::vector_size_t j = 0; j < expected->size(); ++j) {
    ASSERT_TRUE(vectorEquals(output, expected, j))
        << "Multi-batch index " << j << "\nExpected: " << expected->toString(j)
        << "\nActual: " << output->toString(j);
  }
}

TEST_P(SerializationTest, flatMapAsStruct) {
  // Test deserializing a flatmap column as a struct (ROW) instead of a map.
  // The serializer writes MAP data as FlatMap encoding, and the deserializer
  // reads it back as a ROW vector using outputType to control the conversion.
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  const size_t batchSize = 20;
  const std::vector<int32_t> allKeys = {1, 2, 3};

  auto generateInput = [&]() -> velox::VectorPtr {
    const auto numRows = static_cast<velox::vector_size_t>(batchSize);
    const auto numKeys = static_cast<velox::vector_size_t>(allKeys.size());

    auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
    }

    auto keysFlat = velox::BaseVector::create(
        velox::INTEGER(), numRows * numKeys, pool_.get());
    auto valuesFlat = velox::BaseVector::create(
        velox::DOUBLE(), numRows * numKeys, pool_.get());
    velox::vector_size_t offset = 0;
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      for (auto key : allKeys) {
        keysFlat->asFlatVector<int32_t>()->set(offset, key);
        valuesFlat->asFlatVector<double>()->set(offset, i * 10.0 + key);
        ++offset;
      }
    }

    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool_.get()),
        velox::allocateSizes(numRows, pool_.get()),
        keysFlat,
        valuesFlat);
    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = i * numKeys;
      rawSizes[i] = numKeys;
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{ids, mapVector});
  };

  // Serialize with FlatMap encoding.
  const SerializerOptions serOptions{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{serOptions, type, pool_.get()};

  auto input = generateInput();
  auto serializedData =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  auto nimbleSchema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  // Deserialize as struct: select keys "1" and "3" (skip "2").
  auto outputType = velox::ROW({
      {"id", velox::BIGINT()},
      {"features",
       velox::ROW({{"1", velox::DOUBLE()}, {"3", velox::DOUBLE()}})},
  });

  auto desOpts = deserializerOptions();
  desOpts.outputType = outputType;
  Deserializer deserializer{nimbleSchema, pool_.get(), desOpts};

  velox::VectorPtr output;
  deserializer.deserialize(std::string{serializedData}, output);

  ASSERT_EQ(output->size(), batchSize);
  auto* outputRow = output->as<velox::RowVector>();
  ASSERT_EQ(outputRow->childrenSize(), 2);

  // Verify id column.
  auto* idVector = outputRow->childAt(0)->asFlatVector<int64_t>();
  for (velox::vector_size_t i = 0; i < output->size(); ++i) {
    EXPECT_EQ(idVector->valueAt(i), i);
  }

  // Verify features column is ROW with 2 children (keys "1" and "3").
  auto* featuresRow = outputRow->childAt(1)->as<velox::RowVector>();
  ASSERT_NE(featuresRow, nullptr);
  ASSERT_EQ(featuresRow->childrenSize(), 2);

  auto* key1Values = featuresRow->childAt(0)->asFlatVector<double>();
  auto* key3Values = featuresRow->childAt(1)->asFlatVector<double>();
  for (velox::vector_size_t i = 0; i < output->size(); ++i) {
    EXPECT_FALSE(key1Values->isNullAt(i));
    EXPECT_DOUBLE_EQ(key1Values->valueAt(i), i * 10.0 + 1);
    EXPECT_FALSE(key3Values->isNullAt(i));
    EXPECT_DOUBLE_EQ(key3Values->valueAt(i), i * 10.0 + 3);
  }
}

TEST_P(SerializationTest, flatMapAsStructWithMissingKeys) {
  // Test deserializing a flatmap as struct when some requested keys don't
  // exist in the flatmap schema. Missing keys should be filled with nulls.
  auto type = velox::ROW({
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  const size_t batchSize = 10;
  const std::vector<int32_t> allKeys = {1, 2};

  auto generateInput = [&]() -> velox::VectorPtr {
    const auto numRows = static_cast<velox::vector_size_t>(batchSize);
    const auto numKeys = static_cast<velox::vector_size_t>(allKeys.size());

    auto keysFlat = velox::BaseVector::create(
        velox::INTEGER(), numRows * numKeys, pool_.get());
    auto valuesFlat = velox::BaseVector::create(
        velox::DOUBLE(), numRows * numKeys, pool_.get());
    velox::vector_size_t offset = 0;
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      for (auto key : allKeys) {
        keysFlat->asFlatVector<int32_t>()->set(offset, key);
        valuesFlat->asFlatVector<double>()->set(offset, key * 100.0 + i);
        ++offset;
      }
    }

    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool_.get()),
        velox::allocateSizes(numRows, pool_.get()),
        keysFlat,
        valuesFlat);
    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = i * numKeys;
      rawSizes[i] = numKeys;
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{mapVector});
  };

  const SerializerOptions serOptions{
      .version = version(),
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{serOptions, type, pool_.get()};

  auto input = generateInput();
  auto serializedData =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  auto nimbleSchema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  // Request keys "1", "999" (missing), and "2".
  auto outputType = velox::ROW({
      {"features",
       velox::ROW(
           {{"1", velox::DOUBLE()},
            {"999", velox::DOUBLE()},
            {"2", velox::DOUBLE()}})},
  });

  auto desOpts = deserializerOptions();
  desOpts.outputType = outputType;
  Deserializer deserializer{nimbleSchema, pool_.get(), desOpts};

  velox::VectorPtr output;
  deserializer.deserialize(std::string{serializedData}, output);

  ASSERT_EQ(output->size(), batchSize);
  auto* outputRow = output->as<velox::RowVector>();
  auto* featuresRow = outputRow->childAt(0)->as<velox::RowVector>();
  ASSERT_NE(featuresRow, nullptr);
  ASSERT_EQ(featuresRow->childrenSize(), 3);

  auto* key1Values = featuresRow->childAt(0)->asFlatVector<double>();
  auto* key999Values = featuresRow->childAt(1).get();
  auto* key2Values = featuresRow->childAt(2)->asFlatVector<double>();

  for (velox::vector_size_t i = 0; i < output->size(); ++i) {
    EXPECT_FALSE(key1Values->isNullAt(i));
    EXPECT_DOUBLE_EQ(key1Values->valueAt(i), 100.0 + i);

    // Key "999" doesn't exist in the flatmap - should be all nulls.
    EXPECT_TRUE(key999Values->isNullAt(i));

    EXPECT_FALSE(key2Values->isNullAt(i));
    EXPECT_DOUBLE_EQ(key2Values->valueAt(i), 200.0 + i);
  }
}

TEST_P(SerializationTest, flatMapAsStructWithPreviouslyDiscoveredMissingKey) {
  auto type = velox::ROW({
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  auto makeInput = [&](const std::vector<std::vector<int32_t>>& keysByRow)
      -> velox::VectorPtr {
    const auto numRows = static_cast<velox::vector_size_t>(keysByRow.size());
    velox::vector_size_t totalEntries = 0;
    for (const auto& keys : keysByRow) {
      totalEntries += static_cast<velox::vector_size_t>(keys.size());
    }

    auto keysFlat =
        velox::BaseVector::create(velox::INTEGER(), totalEntries, pool_.get());
    auto valuesFlat =
        velox::BaseVector::create(velox::DOUBLE(), totalEntries, pool_.get());

    velox::vector_size_t offset = 0;
    for (velox::vector_size_t row = 0; row < numRows; ++row) {
      for (const auto key : keysByRow[row]) {
        keysFlat->asFlatVector<int32_t>()->set(offset, key);
        valuesFlat->asFlatVector<double>()->set(offset, row * 10.0 + key);
        ++offset;
      }
    }

    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool_.get()),
        velox::allocateSizes(numRows, pool_.get()),
        keysFlat,
        valuesFlat);

    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    offset = 0;
    for (velox::vector_size_t row = 0; row < numRows; ++row) {
      rawOffsets[row] = offset;
      rawSizes[row] = static_cast<velox::vector_size_t>(keysByRow[row].size());
      offset += rawSizes[row];
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{mapVector});
  };

  const SerializerOptions serOptions{
      .version = version(),
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{serOptions, type, pool_.get()};

  std::vector<velox::VectorPtr> inputs{
      makeInput({{1, 2}, {1}, {2}, {}, {1, 2}}),
      makeInput({{2}, {2, 3}, {3}, {}, {2}}),
      makeInput({{1, 3}, {1}, {3}, {}, {1, 3}}),
      makeInput({{1, 2, 3}, {2}, {1}, {}, {3}}),
  };

  std::vector<std::string> serializedData;
  serializedData.reserve(inputs.size());
  for (const auto& input : inputs) {
    serializedData.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
  }

  verifyFlatMapAsStruct(serializer, serializedData, inputs);
}

// Standalone test for kTablet deserialization.
// kTablet is not produced by the Serializer — it's constructed by reading
// raw tablet stream bytes from a Nimble file and assembling them with a header
// and trailer. This test writes a Nimble file, reads raw streams, assembles
// kTablet format, and verifies that Deserializer correctly handles it.
TEST_F(SerializationTest, tabletDeserialization) {
  auto rowType = velox::ROW(
      {{"col_a", velox::INTEGER()},
       {"col_b", velox::BIGINT()},
       {"col_c", velox::VARCHAR()}});

  const int numRows = 100;
  velox::VectorFuzzer fuzzer(
      {.vectorSize = static_cast<size_t>(numRows),
       .nullRatio = 0,
       .stringLength = 20,
       .stringVariableLength = true},
      pool_.get());
  auto input = fuzzer.fuzzInputRow(rowType);

  auto tablet = serializeTablet(rowType, {input}, /*enableChunking=*/true);

  nimble::Deserializer deserializer(
      tablet.schema, pool_.get(), DeserializerOptions{.hasHeader = true});

  velox::VectorPtr deserialized;
  for (const auto& assembled : tablet.serialized) {
    velox::VectorPtr stripeOutput;
    deserializer.deserialize(std::string_view(assembled), stripeOutput);
    ASSERT_NE(stripeOutput, nullptr);

    if (deserialized == nullptr) {
      deserialized = stripeOutput;
    } else {
      auto oldSize = deserialized->size();
      deserialized->resize(oldSize + stripeOutput->size());
      deserialized->copy(stripeOutput.get(), oldSize, 0, stripeOutput->size());
    }
  }

  ASSERT_NE(deserialized, nullptr);
  ASSERT_EQ(deserialized->size(), numRows);

  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ASSERT_TRUE(input->equalValueAt(deserialized.get(), i, i))
        << "Mismatch at row " << i << "\nExpected: " << input->toString(i)
        << "\nActual: " << deserialized->toString(i);
  }
}

TEST_F(SerializationTest, tabletTrailerDeserialization) {
  auto rowType =
      velox::ROW({{"dup_a", velox::INTEGER()}, {"dup_b", velox::INTEGER()}});
  const std::vector<EncodingType> encodings = {
      EncodingType::Trivial,
      EncodingType::Varint,
      EncodingType::Delta,
      EncodingType::FixedBitWidth,
  };

  constexpr int numRows = 128;
  auto dupA = velox::BaseVector::create(
      velox::INTEGER(),
      static_cast<velox::vector_size_t>(numRows),
      pool_.get());
  auto dupB = velox::BaseVector::create(
      velox::INTEGER(),
      static_cast<velox::vector_size_t>(numRows),
      pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    const auto value = static_cast<int32_t>(i * 7);
    dupA->asFlatVector<int32_t>()->set(i, value);
    dupB->asFlatVector<int32_t>()->set(i, value);
  }
  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      rowType,
      nullptr,
      static_cast<velox::vector_size_t>(numRows),
      std::vector<velox::VectorPtr>{dupA, dupB});

  for (const auto streamIdsEncodingType : encodings) {
    for (const auto sizeIndicesEncodingType : encodings) {
      for (const auto uniqueSizesEncodingType : encodings) {
        SCOPED_TRACE(
            fmt::format(
                "streamIdsEncodingType={} sizeIndicesEncodingType={} uniqueSizesEncodingType={}",
                toString(streamIdsEncodingType),
                toString(sizeIndicesEncodingType),
                toString(uniqueSizesEncodingType)));
        auto tablet = serializeTablet(
            rowType,
            {input},
            /*enableChunking=*/true,
            streamIdsEncodingType,
            sizeIndicesEncodingType,
            uniqueSizesEncodingType,
            /*forceDuplicateFirstTwoStreams=*/true);
        ASSERT_EQ(tablet.serialized.size(), 1);
        ASSERT_EQ(tablet.numTabletStreams.size(), 1);
        ASSERT_GT(tablet.numTabletStreams[0], tablet.numUniqueTabletStreams[0]);
        ASSERT_LT(tablet.tabletUniqueBodyBytes[0], tablet.tabletBodyBytes[0]);

        nimble::Deserializer deserializer{
            tablet.schema, pool_.get(), DeserializerOptions{.hasHeader = true}};
        velox::VectorPtr output;
        deserializer.deserialize(
            std::string_view(tablet.serialized[0]), output);

        ASSERT_NE(output, nullptr);
        ASSERT_EQ(output->size(), numRows);
        for (velox::vector_size_t i = 0; i < numRows; ++i) {
          ASSERT_TRUE(input->equalValueAt(output.get(), i, i))
              << "Mismatch at row " << i << "\nExpected: " << input->toString(i)
              << "\nActual: " << output->toString(i);
        }
      }
    }
  }
}

// Verifies ZSTD decompression round-trips correctly with the thread-local
// ZSTD_DCtx reuse (always-on). Exercises the StreamDataParser and StreamData
// decompress paths with kTablet chunked format.
TEST_F(SerializationTest, zstdThreadLocalDCtxRoundTrip) {
  auto rowType = velox::ROW(
      {{"col_a", velox::INTEGER()},
       {"col_b", velox::BIGINT()},
       {"col_c", velox::VARCHAR()}});

  const int numRows = 100;
  velox::VectorFuzzer fuzzer(
      {.vectorSize = static_cast<size_t>(numRows),
       .nullRatio = 0,
       .stringLength = 20,
       .stringVariableLength = true},
      pool_.get());
  auto input = fuzzer.fuzzInputRow(rowType);

  auto tablet = serializeTablet(rowType, {input}, /*enableChunking=*/true);

  nimble::Deserializer deserializer(
      tablet.schema, pool_.get(), DeserializerOptions{.hasHeader = true});

  velox::VectorPtr deserialized;
  for (const auto& assembled : tablet.serialized) {
    velox::VectorPtr stripeOutput;
    deserializer.deserialize(std::string_view(assembled), stripeOutput);
    ASSERT_NE(stripeOutput, nullptr);

    if (deserialized == nullptr) {
      deserialized = stripeOutput;
    } else {
      auto oldSize = deserialized->size();
      deserialized->resize(oldSize + stripeOutput->size());
      deserialized->copy(stripeOutput.get(), oldSize, 0, stripeOutput->size());
    }
  }

  ASSERT_NE(deserialized, nullptr);
  ASSERT_EQ(deserialized->size(), numRows);

  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ASSERT_TRUE(input->equalValueAt(deserialized.get(), i, i))
        << "Mismatch at row " << i << "\nExpected: " << input->toString(i)
        << "\nActual: " << deserialized->toString(i);
  }
}

// Verifies ZSTD thread-local DCtx works correctly with parallel decode
// (multiple threads each get their own context).
TEST_F(SerializationTest, zstdThreadLocalDCtxWithParallelDecode) {
  auto rowType = velox::ROW(
      {{"col_a", velox::INTEGER()},
       {"col_b", velox::BIGINT()},
       {"col_c", velox::VARCHAR()}});

  const int numRows = 100;
  velox::VectorFuzzer fuzzer(
      {.vectorSize = static_cast<size_t>(numRows),
       .nullRatio = 0,
       .stringLength = 20,
       .stringVariableLength = true},
      pool_.get());
  auto input = fuzzer.fuzzInputRow(rowType);

  auto tablet = serializeTablet(rowType, {input}, /*enableChunking=*/true);

  folly::CPUThreadPoolExecutor executor(2);
  nimble::Deserializer deserializer(
      tablet.schema,
      pool_.get(),
      DeserializerOptions{
          .hasHeader = true,
          .decodeExecutor = &executor,
          .maxDecodeParallelism = 2,
      });

  velox::VectorPtr deserialized;
  for (const auto& assembled : tablet.serialized) {
    velox::VectorPtr stripeOutput;
    deserializer.deserialize(std::string_view(assembled), stripeOutput);
    ASSERT_NE(stripeOutput, nullptr);

    if (deserialized == nullptr) {
      deserialized = stripeOutput;
    } else {
      auto oldSize = deserialized->size();
      deserialized->resize(oldSize + stripeOutput->size());
      deserialized->copy(stripeOutput.get(), oldSize, 0, stripeOutput->size());
    }
  }

  ASSERT_NE(deserialized, nullptr);
  ASSERT_EQ(deserialized->size(), numRows);

  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ASSERT_TRUE(input->equalValueAt(deserialized.get(), i, i))
        << "Mismatch at row " << i << "\nExpected: " << input->toString(i)
        << "\nActual: " << deserialized->toString(i);
  }
}

// Stress-tests thread-local ZSTD_DCtx under high parallelism with many
// columns and forced ZSTD compression, ensuring each thread's DCtx produces
// correct results without cross-thread interference.
TEST_F(SerializationTest, zstdThreadLocalDCtxHighParallelism) {
  auto rowType = velox::ROW(
      {{"col_a", velox::INTEGER()},
       {"col_b", velox::BIGINT()},
       {"col_c", velox::VARCHAR()},
       {"col_d", velox::DOUBLE()},
       {"col_e", velox::INTEGER()},
       {"col_f", velox::BIGINT()},
       {"col_g", velox::VARCHAR()},
       {"col_h", velox::DOUBLE()}});

  const int numRows = 1000;
  velox::VectorFuzzer fuzzer(
      {.vectorSize = static_cast<size_t>(numRows),
       .nullRatio = 0,
       .stringLength = 50,
       .stringVariableLength = true},
      pool_.get());
  auto input = fuzzer.fuzzInputRow(rowType);

  std::string fileData;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&fileData);
    nimble::WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.compressionOptions.compressionType =
        nimble::CompressionType::Zstd;
    writerOptions.compressionOptions.zstdMinCompressionSize = 0;
    nimble::Writer writer(
        rowType, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(input);
    writer.close();
  }

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string_view(fileData));
  auto tablet = nimble::TabletReader::create(
      readFile, pool_.get(), makeTestTabletOptions(pool_.get()));
  auto schemaSection =
      tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
  auto nimbleSchema =
      nimble::SchemaDeserializer::deserialize(schemaSection->content().data());

  std::set<uint32_t> offsetSet;
  collectStreamOffsets(*nimbleSchema, offsetSet);
  std::vector<uint32_t> streamOffsets(offsetSet.begin(), offsetSet.end());

  std::vector<std::string> assembledBuffers;
  for (uint32_t stripeIdx = 0; stripeIdx < tablet->stripeCount(); ++stripeIdx) {
    const auto stripeId = tablet->stripeIdentifier(stripeIdx);
    auto streamLoaders = tablet->load(stripeId, streamOffsets);
    const auto stripeRows = tablet->stripeRowCount(stripeIdx);

    auto headerIOBuf = serde::createTabletChunkHeader(
        {.rowCount = stripeRows, .rowRange = RowRange{0, stripeRows}});
    std::string headerBuf(
        reinterpret_cast<const char*>(headerIOBuf.data()),
        headerIOBuf.length());

    auto streams = collectTabletStreams(streamOffsets, streamLoaders);
    std::string assembled;
    assembled.reserve(headerBuf.size());
    assembled.append(headerBuf);
    writeTabletStreamsAndTrailer(
        streams,
        EncodingType::Trivial,
        EncodingType::Trivial,
        EncodingType::Trivial,
        assembled);
    assembledBuffers.push_back(std::move(assembled));
  }

  folly::CPUThreadPoolExecutor executor(8);
  nimble::Deserializer deserializer(
      nimbleSchema,
      pool_.get(),
      DeserializerOptions{
          .hasHeader = true,
          .decodeExecutor = &executor,
          .maxDecodeParallelism = 8,
      });

  velox::VectorPtr deserialized;
  for (const auto& assembled : assembledBuffers) {
    velox::VectorPtr stripeOutput;
    deserializer.deserialize(std::string_view(assembled), stripeOutput);
    ASSERT_NE(stripeOutput, nullptr);

    if (deserialized == nullptr) {
      deserialized = stripeOutput;
    } else {
      auto oldSize = deserialized->size();
      deserialized->resize(oldSize + stripeOutput->size());
      deserialized->copy(stripeOutput.get(), oldSize, 0, stripeOutput->size());
    }
  }

  ASSERT_NE(deserialized, nullptr);
  ASSERT_EQ(deserialized->size(), numRows);

  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ASSERT_TRUE(input->equalValueAt(deserialized.get(), i, i))
        << "Mismatch at row " << i << "\nExpected: " << input->toString(i)
        << "\nActual: " << deserialized->toString(i);
  }
}

// Verifies thread-local ZSTD_DCtx correctness when multiple Deserializers
// run concurrently on separate threads, each with their own data.
TEST_F(SerializationTest, zstdThreadLocalDCtxConcurrentDeserializers) {
  auto rowType = velox::ROW(
      {{"col_a", velox::INTEGER()},
       {"col_b", velox::BIGINT()},
       {"col_c", velox::VARCHAR()}});

  const int numRows = 500;
  const int numThreads = 4;

  std::vector<velox::VectorPtr> inputs;
  for (int t = 0; t < numThreads; ++t) {
    velox::VectorFuzzer fuzzer(
        {.vectorSize = static_cast<size_t>(numRows),
         .nullRatio = 0,
         .stringLength = 30,
         .stringVariableLength = true},
        pool_.get(),
        t);
    inputs.push_back(fuzzer.fuzzInputRow(rowType));
  }

  std::vector<std::vector<std::string>> allAssembled(numThreads);
  std::vector<std::shared_ptr<const Type>> schemas(numThreads);
  for (int t = 0; t < numThreads; ++t) {
    std::string fileData;
    {
      auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&fileData);
      nimble::WriterOptions writerOptions;
      writerOptions.enableChunking = true;
      writerOptions.compressionOptions.compressionType =
          nimble::CompressionType::Zstd;
      writerOptions.compressionOptions.zstdMinCompressionSize = 0;
      nimble::Writer writer(
          rowType, std::move(writeFile), *rootPool_, std::move(writerOptions));
      writer.write(inputs[t]);
      writer.close();
    }

    auto readFile =
        std::make_shared<velox::InMemoryReadFile>(std::string_view(fileData));
    auto tablet = nimble::TabletReader::create(
        readFile, pool_.get(), makeTestTabletOptions(pool_.get()));
    auto schemaSection =
        tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
    schemas[t] = nimble::SchemaDeserializer::deserialize(
        schemaSection->content().data());

    std::set<uint32_t> offsetSet;
    collectStreamOffsets(*schemas[t], offsetSet);
    std::vector<uint32_t> streamOffsets(offsetSet.begin(), offsetSet.end());

    for (uint32_t si = 0; si < tablet->stripeCount(); ++si) {
      const auto stripeId = tablet->stripeIdentifier(si);
      auto streamLoaders = tablet->load(stripeId, streamOffsets);
      const auto stripeRows = tablet->stripeRowCount(si);

      auto headerIOBuf = serde::createTabletChunkHeader(
          {.rowCount = stripeRows, .rowRange = RowRange{0, stripeRows}});
      std::string headerBuf(
          reinterpret_cast<const char*>(headerIOBuf.data()),
          headerIOBuf.length());

      auto streams = collectTabletStreams(streamOffsets, streamLoaders);
      std::string assembled;
      assembled.reserve(headerBuf.size());
      assembled.append(headerBuf);
      writeTabletStreamsAndTrailer(
          streams,
          EncodingType::Trivial,
          EncodingType::Trivial,
          EncodingType::Trivial,
          assembled);
      allAssembled[t].push_back(std::move(assembled));
    }
  }

  std::vector<std::thread> threads;
  std::vector<std::atomic<bool>> success(numThreads);
  for (int t = 0; t < numThreads; ++t) {
    success[t].store(false);
  }

  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([&, t]() {
      auto threadPool = velox::memory::memoryManager()->addLeafPool(
          "thread_" + std::to_string(t));
      nimble::Deserializer deserializer(
          schemas[t], threadPool.get(), DeserializerOptions{.hasHeader = true});

      velox::VectorPtr deserialized;
      for (const auto& assembled : allAssembled[t]) {
        velox::VectorPtr stripeOutput;
        deserializer.deserialize(std::string_view(assembled), stripeOutput);
        if (stripeOutput == nullptr) {
          return;
        }

        if (deserialized == nullptr) {
          deserialized = stripeOutput;
        } else {
          auto oldSize = deserialized->size();
          deserialized->resize(oldSize + stripeOutput->size());
          deserialized->copy(
              stripeOutput.get(), oldSize, 0, stripeOutput->size());
        }
      }

      if (deserialized == nullptr ||
          deserialized->size() != static_cast<size_t>(numRows)) {
        return;
      }
      for (velox::vector_size_t i = 0; i < numRows; ++i) {
        if (!inputs[t]->equalValueAt(deserialized.get(), i, i)) {
          return;
        }
      }
      success[t].store(true);
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
  for (int t = 0; t < numThreads; ++t) {
    ASSERT_TRUE(success[t].load()) << "Thread " << t << " failed";
  }
}

// Verifies thread-local ZSTD_DCtx works correctly with a flat-map schema
// that produces many ZSTD-compressed key streams, combined with parallel
// decode.
TEST_F(SerializationTest, zstdThreadLocalDCtxFlatMapWithParallelDecode) {
  auto rowType = velox::ROW(
      {{"id", velox::BIGINT()},
       {"features", velox::MAP(velox::INTEGER(), velox::REAL())}});

  const int numRows = 500;
  const int numKeys = 20;
  const int totalEntries = numRows * numKeys;

  auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ids->asFlatVector<int64_t>()->set(i, i);
  }

  auto keys =
      velox::BaseVector::create(velox::INTEGER(), totalEntries, pool_.get());
  auto values =
      velox::BaseVector::create(velox::REAL(), totalEntries, pool_.get());
  for (velox::vector_size_t i = 0; i < totalEntries; ++i) {
    keys->asFlatVector<int32_t>()->set(i, i % numKeys);
    values->asFlatVector<float>()->set(i, static_cast<float>(i) * 0.1f);
  }

  auto offsets = velox::AlignedBuffer::allocate<velox::vector_size_t>(
      numRows, pool_.get());
  auto sizes = velox::AlignedBuffer::allocate<velox::vector_size_t>(
      numRows, pool_.get());
  auto rawOffsets = offsets->asMutable<velox::vector_size_t>();
  auto rawSizes = sizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * numKeys;
    rawSizes[i] = numKeys;
  }

  auto mapVector = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(velox::INTEGER(), velox::REAL()),
      nullptr,
      numRows,
      offsets,
      sizes,
      keys,
      values);

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      rowType,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{ids, mapVector});

  std::string fileData;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&fileData);
    nimble::WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.compressionOptions.compressionType =
        nimble::CompressionType::Zstd;
    writerOptions.compressionOptions.zstdMinCompressionSize = 0;
    writerOptions.flatMapColumns = {{"features", {}}};
    nimble::Writer writer(
        rowType, std::move(writeFile), *rootPool_, std::move(writerOptions));
    writer.write(input);
    writer.close();
  }

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string_view(fileData));
  auto tablet = nimble::TabletReader::create(
      readFile, pool_.get(), makeTestTabletOptions(pool_.get()));
  auto schemaSection =
      tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
  auto nimbleSchema =
      nimble::SchemaDeserializer::deserialize(schemaSection->content().data());

  std::set<uint32_t> offsetSet;
  collectStreamOffsets(*nimbleSchema, offsetSet);
  std::vector<uint32_t> streamOffsets(offsetSet.begin(), offsetSet.end());

  std::vector<std::string> assembledBuffers;
  for (uint32_t stripeIdx = 0; stripeIdx < tablet->stripeCount(); ++stripeIdx) {
    const auto stripeId = tablet->stripeIdentifier(stripeIdx);
    auto streamLoaders = tablet->load(stripeId, streamOffsets);
    const auto stripeRows = tablet->stripeRowCount(stripeIdx);

    auto headerIOBuf = serde::createTabletChunkHeader(
        {.rowCount = stripeRows, .rowRange = RowRange{0, stripeRows}});
    std::string headerBuf(
        reinterpret_cast<const char*>(headerIOBuf.data()),
        headerIOBuf.length());

    auto streams = collectTabletStreams(streamOffsets, streamLoaders);
    std::string assembled;
    assembled.reserve(headerBuf.size());
    assembled.append(headerBuf);
    writeTabletStreamsAndTrailer(
        streams,
        EncodingType::Trivial,
        EncodingType::Trivial,
        EncodingType::Trivial,
        assembled);
    assembledBuffers.push_back(std::move(assembled));
  }

  folly::CPUThreadPoolExecutor executor(4);
  nimble::Deserializer deserializer(
      nimbleSchema,
      pool_.get(),
      DeserializerOptions{
          .hasHeader = true,
          .decodeExecutor = &executor,
          .maxDecodeParallelism = 4,
      });

  velox::VectorPtr deserialized;
  for (const auto& assembled : assembledBuffers) {
    velox::VectorPtr stripeOutput;
    deserializer.deserialize(std::string_view(assembled), stripeOutput);
    ASSERT_NE(stripeOutput, nullptr);

    if (deserialized == nullptr) {
      deserialized = stripeOutput;
    } else {
      auto oldSize = deserialized->size();
      deserialized->resize(oldSize + stripeOutput->size());
      deserialized->copy(stripeOutput.get(), oldSize, 0, stripeOutput->size());
    }
  }

  ASSERT_NE(deserialized, nullptr);
  ASSERT_EQ(deserialized->size(), numRows);

  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ASSERT_TRUE(input->equalValueAt(deserialized.get(), i, i))
        << "Mismatch at row " << i << "\nExpected: " << input->toString(i)
        << "\nActual: " << deserialized->toString(i);
  }
}

// Verifies thread-local ZSTD_DCtx correctness across repeated batches
// deserialized sequentially, ensuring the DCtx state resets properly.
TEST_F(SerializationTest, zstdThreadLocalDCtxRepeatedBatches) {
  auto rowType =
      velox::ROW({{"col_a", velox::INTEGER()}, {"col_b", velox::VARCHAR()}});

  const int numBatches = 20;
  const int rowsPerBatch = 200;

  std::vector<velox::VectorPtr> inputs;
  for (int b = 0; b < numBatches; ++b) {
    velox::VectorFuzzer fuzzer(
        {.vectorSize = static_cast<size_t>(rowsPerBatch),
         .nullRatio = 0,
         .stringLength = 40,
         .stringVariableLength = true},
        pool_.get(),
        b);
    inputs.push_back(fuzzer.fuzzInputRow(rowType));
  }

  std::string fileData;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&fileData);
    nimble::WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.compressionOptions.compressionType =
        nimble::CompressionType::Zstd;
    writerOptions.compressionOptions.zstdMinCompressionSize = 0;
    nimble::Writer writer(
        rowType, std::move(writeFile), *rootPool_, std::move(writerOptions));
    for (const auto& input : inputs) {
      writer.write(input);
    }
    writer.close();
  }

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string_view(fileData));
  auto tablet = nimble::TabletReader::create(
      readFile, pool_.get(), makeTestTabletOptions(pool_.get()));
  auto schemaSection =
      tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
  auto nimbleSchema =
      nimble::SchemaDeserializer::deserialize(schemaSection->content().data());

  std::set<uint32_t> offsetSet;
  collectStreamOffsets(*nimbleSchema, offsetSet);
  std::vector<uint32_t> streamOffsets(offsetSet.begin(), offsetSet.end());

  std::vector<std::string> assembledBuffers;
  for (uint32_t stripeIdx = 0; stripeIdx < tablet->stripeCount(); ++stripeIdx) {
    const auto stripeId = tablet->stripeIdentifier(stripeIdx);
    auto streamLoaders = tablet->load(stripeId, streamOffsets);
    const auto stripeRows = tablet->stripeRowCount(stripeIdx);

    auto headerIOBuf = serde::createTabletChunkHeader(
        {.rowCount = stripeRows, .rowRange = RowRange{0, stripeRows}});
    std::string headerBuf(
        reinterpret_cast<const char*>(headerIOBuf.data()),
        headerIOBuf.length());

    auto streams = collectTabletStreams(streamOffsets, streamLoaders);
    std::string assembled;
    assembled.reserve(headerBuf.size());
    assembled.append(headerBuf);
    writeTabletStreamsAndTrailer(
        streams,
        EncodingType::Trivial,
        EncodingType::Trivial,
        EncodingType::Trivial,
        assembled);
    assembledBuffers.push_back(std::move(assembled));
  }

  folly::CPUThreadPoolExecutor executor(4);
  nimble::Deserializer deserializer(
      nimbleSchema,
      pool_.get(),
      DeserializerOptions{
          .hasHeader = true,
          .decodeExecutor = &executor,
          .maxDecodeParallelism = 4,
      });

  velox::VectorPtr deserialized;
  for (const auto& assembled : assembledBuffers) {
    velox::VectorPtr stripeOutput;
    deserializer.deserialize(std::string_view(assembled), stripeOutput);
    ASSERT_NE(stripeOutput, nullptr);

    if (deserialized == nullptr) {
      deserialized = stripeOutput;
    } else {
      auto oldSize = deserialized->size();
      deserialized->resize(oldSize + stripeOutput->size());
      deserialized->copy(stripeOutput.get(), oldSize, 0, stripeOutput->size());
    }
  }

  ASSERT_NE(deserialized, nullptr);
  const int totalRows = numBatches * rowsPerBatch;
  ASSERT_EQ(deserialized->size(), totalRows);
}

TEST_P(SerializationTest, flatMapDeserializeWithFlatMapSchema) {
  // Verifies that FlatMap-encoded data round-trips correctly when deserialized
  // with the FlatMap schema from the serializer (no projection).
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  const velox::vector_size_t numRows = 10;
  const int numKeys = 3;
  const int totalEntries = numRows * numKeys;

  auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ids->asFlatVector<int64_t>()->set(i, i * 100);
  }

  auto mapKeys =
      velox::BaseVector::create(velox::INTEGER(), totalEntries, pool_.get());
  auto mapValues =
      velox::BaseVector::create(velox::DOUBLE(), totalEntries, pool_.get());
  for (int i = 0; i < totalEntries; ++i) {
    mapKeys->asFlatVector<int32_t>()->set(i, (i % numKeys) + 1);
    mapValues->asFlatVector<double>()->set(i, i * 1.5);
  }

  auto mapOffsets = velox::allocateOffsets(numRows, pool_.get());
  auto mapSizes = velox::allocateSizes(numRows, pool_.get());
  auto* rawOffsets = mapOffsets->asMutable<velox::vector_size_t>();
  auto* rawSizes = mapSizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i * numKeys;
    rawSizes[i] = numKeys;
  }

  auto mapVector = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(velox::INTEGER(), velox::DOUBLE()),
      nullptr,
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{ids, mapVector});

  // Serialize with FlatMap encoding.
  const SerializerOptions options{
      .version = version(),
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{options, type, pool_.get()};
  auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  // Get the schema from the serializer — should be FlatMap.
  auto flatMapSchema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  ASSERT_TRUE(flatMapSchema->isRow());
  ASSERT_EQ(flatMapSchema->asRow().childrenCount(), 2);
  const auto& featuresType = *flatMapSchema->asRow().childAt(1);
  EXPECT_TRUE(featuresType.isFlatMap())
      << "Expected FlatMap, got " << toString(featuresType.kind());

  // convertToNimbleType creates a regular Map schema (not FlatMap).
  auto regularMapSchema = convertToNimbleType(*type);
  ASSERT_TRUE(regularMapSchema->isRow());
  const auto& regularFeaturesType = *regularMapSchema->asRow().childAt(1);
  EXPECT_TRUE(regularFeaturesType.isMap())
      << "Expected Map, got " << toString(regularFeaturesType.kind());

  // Deserialize with the FlatMap schema — should work.
  {
    Deserializer deserializer{
        flatMapSchema, pool_.get(), deserializerOptions()};
    velox::VectorPtr output;
    deserializer.deserialize(serialized, output);
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->size(), numRows);

    auto* resultRow = output->as<velox::RowVector>();
    auto* resultMap = resultRow->childAt(1)->as<velox::MapVector>();
    ASSERT_NE(resultMap, nullptr);
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      EXPECT_EQ(resultMap->sizeAt(i), numKeys)
          << "Row " << i << " should have " << numKeys << " entries";
    }
  }
}

// Exercises the scatteredRead path in Deserializer when flat-map keys are
// sparse (absent in some rows). Deserializing with outputType (struct mode)
// triggers scatter bitmaps for each key. Without the nulls initialization fix
// in BatchedStreamDecoder::next(), this crashes in loadOffsets() accessing
// rawNulls() on an unallocated buffer.
TEST_P(SerializationTest, flatMapScatteredReadWithSparseKeys) {
  if (!version().has_value() || version() == SerializationVersion::kLegacy) {
    GTEST_SKIP() << "Legacy format does not support struct flat-map reads";
  }
  // Map<int, Array<bigint>> — similar to EBF's data column.
  auto type = velox::ROW({
      {"data", velox::MAP(velox::INTEGER(), velox::ARRAY(velox::BIGINT()))},
  });

  const velox::vector_size_t numRows = 6;

  // Sparse keys: key 1 in rows 0,2,4; key 2 in rows 1,3,5.
  // Each key is absent in half the rows → exercises scattered read.
  // Row 0: {1: [10,20]}    Row 1: {2: [30,40]}
  // Row 2: {1: [50,60]}    Row 3: {2: [70,80]}
  // Row 4: {1: [90,100]}   Row 5: {2: [110,120]}
  const int totalMapEntries = 6; // 1 entry per row
  const int totalArrayElements = 12; // 2 elements per entry

  auto mapOffsetsBuffer = velox::allocateOffsets(numRows, pool_.get());
  auto mapSizesBuffer = velox::allocateSizes(numRows, pool_.get());
  auto* rawMapOffsets = mapOffsetsBuffer->asMutable<velox::vector_size_t>();
  auto* rawMapSizes = mapSizesBuffer->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawMapOffsets[i] = i;
    rawMapSizes[i] = 1; // every row has exactly 1 map entry
  }

  auto mapKeys =
      velox::BaseVector::create(velox::INTEGER(), totalMapEntries, pool_.get());
  for (int i = 0; i < totalMapEntries; ++i) {
    // Even rows get key=1, odd rows get key=2
    mapKeys->asFlatVector<int32_t>()->set(i, (i % 2 == 0) ? 1 : 2);
  }

  auto arrayElements = velox::BaseVector::create(
      velox::BIGINT(), totalArrayElements, pool_.get());
  for (int i = 0; i < totalArrayElements; ++i) {
    arrayElements->asFlatVector<int64_t>()->set(i, (i + 1) * 10);
  }
  auto arrayOffsetsBuffer =
      velox::allocateOffsets(totalMapEntries, pool_.get());
  auto arraySizesBuffer = velox::allocateSizes(totalMapEntries, pool_.get());
  for (int i = 0; i < totalMapEntries; ++i) {
    arrayOffsetsBuffer->asMutable<velox::vector_size_t>()[i] = i * 2;
    arraySizesBuffer->asMutable<velox::vector_size_t>()[i] = 2;
  }
  auto mapValues = std::make_shared<velox::ArrayVector>(
      pool_.get(),
      velox::ARRAY(velox::BIGINT()),
      nullptr,
      totalMapEntries,
      arrayOffsetsBuffer,
      arraySizesBuffer,
      arrayElements);

  auto mapVector = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(velox::INTEGER(), velox::ARRAY(velox::BIGINT())),
      nullptr, // no nulls on map itself
      numRows,
      mapOffsetsBuffer,
      mapSizesBuffer,
      mapKeys,
      mapValues);

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      type,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{mapVector});

  const SerializerOptions options{
      .version = version(),
      .flatMapColumns = {{"data", {}}},
  };
  Serializer serializer{options, type, pool_.get()};
  auto serialized =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  // Request both keys in struct mode. Key 1 is absent in odd rows, key 2 in
  // even rows → each key triggers scattered read with ~50% nulls.
  auto structType = velox::ROW(
      {{"1", velox::ARRAY(velox::BIGINT())},
       {"2", velox::ARRAY(velox::BIGINT())}});
  auto outputType = velox::ROW({{"data", structType}});

  DeserializerOptions deserOptions{.hasHeader = true};
  deserOptions.outputType = outputType;
  Deserializer deserializer{schema, pool_.get(), deserOptions};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->size(), numRows);

  auto* resultRow = output->as<velox::RowVector>();
  auto* dataStruct = resultRow->childAt(0)->as<velox::RowVector>();
  ASSERT_NE(dataStruct, nullptr);
  ASSERT_EQ(dataStruct->childrenSize(), 2);

  // Key "1": present in even rows, null in odd rows
  auto* key1 = dataStruct->childAt(0).get();
  // Key "2": present in odd rows, null in even rows
  auto* key2 = dataStruct->childAt(1).get();
  ASSERT_EQ(key1->size(), numRows);
  ASSERT_EQ(key2->size(), numRows);

  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    if (i % 2 == 0) {
      EXPECT_FALSE(key1->isNullAt(i)) << "Key 1 row " << i;
      EXPECT_TRUE(key2->isNullAt(i)) << "Key 2 row " << i;
    } else {
      EXPECT_TRUE(key1->isNullAt(i)) << "Key 1 row " << i;
      EXPECT_FALSE(key2->isNullAt(i)) << "Key 2 row " << i;
    }
  }
}

// Tests serialization of arrays and maps with non-trivial offsets.
// "Sliding window" means each row's elements overlap with the previous row's:
// row 0 → elements[0..2], row 1 → elements[1..3], row 2 → elements[2..4], etc.
// This exercises the serializer's handling of shared underlying buffers with
// per-row offset/size indirection.
TEST_P(SerializationTest, arrayWithOffsetsAndSlidingMapWindows) {
  const velox::vector_size_t numRows = 10;
  const velox::vector_size_t windowSize = 3;
  const velox::vector_size_t numElements = numRows + windowSize - 1;

  // Build array with sliding window offsets: each row sees a window of
  // 'windowSize' elements shifted by 1 from the previous row.
  auto arrayElements =
      velox::BaseVector::create(velox::INTEGER(), numElements, pool_.get());
  for (velox::vector_size_t i = 0; i < numElements; ++i) {
    arrayElements->asFlatVector<int32_t>()->set(i, i * 10);
  }

  auto arraySizes = velox::AlignedBuffer::allocate<velox::vector_size_t>(
      numRows, pool_.get());
  auto arrayOffsets = velox::AlignedBuffer::allocate<velox::vector_size_t>(
      numRows, pool_.get());
  auto* rawArraySizes = arraySizes->asMutable<velox::vector_size_t>();
  auto* rawArrayOffsets = arrayOffsets->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawArrayOffsets[i] = i;
    rawArraySizes[i] = windowSize;
  }

  auto arrayVector = std::make_shared<velox::ArrayVector>(
      pool_.get(),
      velox::ARRAY(velox::INTEGER()),
      nullptr, // no nulls
      numRows,
      arrayOffsets,
      arraySizes,
      arrayElements);

  // Build map with sliding window offsets over shared keys/values buffers.
  auto mapKeys =
      velox::BaseVector::create(velox::VARCHAR(), numElements, pool_.get());
  auto mapValues =
      velox::BaseVector::create(velox::DOUBLE(), numElements, pool_.get());
  for (velox::vector_size_t i = 0; i < numElements; ++i) {
    auto keyStr = fmt::format("key_{}", i);
    mapKeys->asFlatVector<velox::StringView>()->set(
        i, velox::StringView(keyStr));
    mapValues->asFlatVector<double>()->set(i, i * 1.5);
  }

  auto mapSizes = velox::AlignedBuffer::allocate<velox::vector_size_t>(
      numRows, pool_.get());
  auto mapOffsets = velox::AlignedBuffer::allocate<velox::vector_size_t>(
      numRows, pool_.get());
  auto* rawMapSizes = mapSizes->asMutable<velox::vector_size_t>();
  auto* rawMapOffsets = mapOffsets->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawMapOffsets[i] = i;
    rawMapSizes[i] = windowSize;
  }

  auto mapVector = std::make_shared<velox::MapVector>(
      pool_.get(),
      velox::MAP(velox::VARCHAR(), velox::DOUBLE()),
      nullptr, // no nulls
      numRows,
      mapOffsets,
      mapSizes,
      mapKeys,
      mapValues);

  auto rowType = velox::ROW(
      {{"sliding_array", velox::ARRAY(velox::INTEGER())},
       {"sliding_map", velox::MAP(velox::VARCHAR(), velox::DOUBLE())}});

  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      rowType,
      nullptr,
      numRows,
      std::vector<velox::VectorPtr>{arrayVector, mapVector});

  // Serialize and deserialize.
  SerializerOptions serOptions{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
  };
  Serializer serializer{serOptions, rowType, pool_.get()};
  auto serialized = serializer.serialize(input, OrderedRanges::of(0, numRows));

  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  Deserializer deserializer{schema, pool_.get(), deserializerOptions()};
  velox::VectorPtr output;
  deserializer.deserialize(serialized, output);

  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->size(), numRows);

  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    ASSERT_TRUE(input->equalValueAt(output.get(), i, i))
        << "Mismatch at row " << i << "\nExpected: " << input->toString(i)
        << "\nActual: " << output->toString(i);
  }
}

namespace {

// Helper to create a MapVector with given integer keys and float values.
velox::VectorPtr createMapVector(
    velox::memory::MemoryPool* pool,
    const velox::TypePtr& type,
    const std::vector<std::vector<int32_t>>& rowKeys,
    const std::vector<std::vector<float>>& rowValues) {
  const auto numRows = static_cast<velox::vector_size_t>(rowKeys.size());
  velox::vector_size_t totalEntries = 0;
  for (const auto& keys : rowKeys) {
    totalEntries += keys.size();
  }

  auto keysFlat =
      velox::BaseVector::create(velox::INTEGER(), totalEntries, pool);
  auto valuesFlat =
      velox::BaseVector::create(velox::REAL(), totalEntries, pool);
  auto offsets = velox::allocateOffsets(numRows, pool);
  auto sizes = velox::allocateSizes(numRows, pool);
  auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
  auto* rawSizes = sizes->asMutable<velox::vector_size_t>();

  velox::vector_size_t offset = 0;
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = offset;
    rawSizes[i] = static_cast<velox::vector_size_t>(rowKeys[i].size());
    for (size_t j = 0; j < rowKeys[i].size(); ++j) {
      keysFlat->asFlatVector<int32_t>()->set(offset, rowKeys[i][j]);
      valuesFlat->asFlatVector<float>()->set(offset, rowValues[i][j]);
      ++offset;
    }
  }

  return std::make_shared<velox::MapVector>(
      pool,
      type,
      nullptr,
      numRows,
      std::move(offsets),
      std::move(sizes),
      keysFlat,
      valuesFlat);
}

} // namespace

TEST_P(SerializationTest, flatmapColumnsKeysSchemaConsistency) {
  // Two serializers with the same 8 predefined keys, 200 rows each, should
  // produce identical schemas regardless of data arrival order.
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"m", velox::MAP(velox::INTEGER(), velox::REAL())},
  });

  const auto mapType = velox::MAP(velox::INTEGER(), velox::REAL());
  const std::vector<std::string> keysList = {
      "20", "5", "13", "8", "17", "2", "11", "19"};
  const std::set<std::string> predefinedKeys(keysList.begin(), keysList.end());
  constexpr int32_t kNumKeys = 8;
  constexpr int32_t kNumRows = 200;

  auto buildInput = [&](bool reverseKeys) {
    // Build keys and values arrays for kNumRows rows.
    std::vector<std::vector<int32_t>> rowKeys(kNumRows);
    std::vector<std::vector<float>> rowValues(kNumRows);
    for (int32_t r = 0; r < kNumRows; ++r) {
      int32_t numKeysInRow = (r % kNumKeys) + 1;
      for (int32_t k = 0; k < numKeysInRow; ++k) {
        int32_t keyIdx =
            reverseKeys ? (kNumKeys - 1 - k) % kNumKeys : k % kNumKeys;
        rowKeys[r].push_back(folly::to<int32_t>(keysList[keyIdx]));
        rowValues[r].push_back(static_cast<float>(r * 100 + k));
      }
    }
    auto ids =
        velox::BaseVector::create(velox::BIGINT(), kNumRows, pool_.get());
    for (int32_t r = 0; r < kNumRows; ++r) {
      ids->asFlatVector<int64_t>()->set(r, r);
    }
    auto map = createMapVector(pool_.get(), mapType, rowKeys, rowValues);
    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        kNumRows,
        std::vector<velox::VectorPtr>{ids, map});
  };

  // Serializer 1: keys in forward order
  SerializerOptions options1{
      .version = version(),
      .flatMapColumns = {{"m", predefinedKeys}},
  };
  Serializer serializer1{options1, type, pool_.get()};
  serializer1.serialize(buildInput(false), OrderedRanges::of(0, kNumRows));

  // Serializer 2: keys in reverse order
  SerializerOptions options2{
      .version = version(),
      .flatMapColumns = {{"m", predefinedKeys}},
  };
  Serializer serializer2{options2, type, pool_.get()};
  serializer2.serialize(buildInput(true), OrderedRanges::of(0, kNumRows));

  // Both serializers must produce identical schemas.
  auto schema1 = serializer1.schemaBuilder().schemaNodes();
  auto schema2 = serializer2.schemaBuilder().schemaNodes();
  ASSERT_EQ(schema1.size(), schema2.size());
  for (size_t i = 0; i < schema1.size(); ++i) {
    EXPECT_EQ(schema1[i].kind(), schema2[i].kind()) << "Mismatch at node " << i;
    EXPECT_EQ(schema1[i].childrenCount(), schema2[i].childrenCount())
        << "Mismatch at node " << i;
    EXPECT_EQ(schema1[i].name(), schema2[i].name()) << "Mismatch at node " << i;
  }
}

TEST_P(SerializationTest, flatmapColumnsKeysRoundtrip) {
  // Serialize 500 rows with 6 predefined keys across 5 serialize calls,
  // deserialize and verify correctness.
  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"m", velox::MAP(velox::INTEGER(), velox::REAL())},
  });

  const auto mapType = velox::MAP(velox::INTEGER(), velox::REAL());
  const std::vector<std::string> keysList = {"5", "3", "7", "11", "2", "9"};
  const std::set<std::string> predefinedKeys(keysList.begin(), keysList.end());
  constexpr int32_t kNumKeys = 6;
  constexpr int32_t kRowsPerBatch = 100;
  constexpr int32_t kBatches = 5;

  SerializerOptions options{
      .version = version(),
      .flatMapColumns = {{"m", predefinedKeys}},
  };
  Serializer serializer{options, type, pool_.get()};

  std::shared_ptr<const nimble::Type> schema;
  std::unique_ptr<Deserializer> deserializer;

  for (int32_t b = 0; b < kBatches; ++b) {
    // Build input batch.
    std::vector<std::vector<int32_t>> rowKeys(kRowsPerBatch);
    std::vector<std::vector<float>> rowValues(kRowsPerBatch);
    auto ids =
        velox::BaseVector::create(velox::BIGINT(), kRowsPerBatch, pool_.get());
    for (int32_t r = 0; r < kRowsPerBatch; ++r) {
      ids->asFlatVector<int64_t>()->set(r, b * kRowsPerBatch + r);
      int32_t numKeysInRow = (r % kNumKeys) + 1;
      for (int32_t k = 0; k < numKeysInRow; ++k) {
        // Use different key subsets per batch by rotating starting offset.
        rowKeys[r].push_back(folly::to<int32_t>(keysList[(b + k) % kNumKeys]));
        rowValues[r].push_back(static_cast<float>(b * 1000 + r * 10 + k));
      }
    }
    auto map = createMapVector(pool_.get(), mapType, rowKeys, rowValues);
    auto input = std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRowsPerBatch,
        std::vector<velox::VectorPtr>{ids, map});

    auto serialized =
        serializer.serialize(input, OrderedRanges::of(0, kRowsPerBatch));

    // Create deserializer after first serialize so schema includes flatmap
    // keys.
    if (!deserializer) {
      schema =
          SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
      deserializer = std::make_unique<Deserializer>(
          schema, pool_.get(), deserializerOptions());
    }

    velox::VectorPtr output;
    deserializer->deserialize(serialized, output);

    ASSERT_EQ(output->size(), input->size());
    for (velox::vector_size_t i = 0; i < input->size(); ++i) {
      ASSERT_TRUE(input->equalValueAt(output.get(), i, i))
          << "Mismatch at batch " << b << " row " << i
          << "\nExpected: " << input->toString(i)
          << "\nActual: " << output->toString(i);
    }
  }
}

TEST_P(SerializationTest, flatmapColumnsKeysUnknownKeyRejection) {
  // Pre-register 5 keys, feed 100 rows where each row includes unknown key 99.
  auto type = velox::ROW({
      {"m", velox::MAP(velox::INTEGER(), velox::REAL())},
  });

  const auto mapType = velox::MAP(velox::INTEGER(), velox::REAL());
  constexpr int32_t kNumRows = 100;

  SerializerOptions options{
      .version = version(),
      .flatMapColumns = {{"m", {"11", "2", "3", "5", "7"}}},
  };
  Serializer serializer{options, type, pool_.get()};

  // Each row has one valid key and one unknown key (99).
  std::vector<std::vector<int32_t>> rowKeys(kNumRows);
  std::vector<std::vector<float>> rowValues(kNumRows);
  for (int32_t r = 0; r < kNumRows; ++r) {
    rowKeys[r] = {(r % 5) * 2 + 1, 99}; // valid key + unknown key
    rowValues[r] = {static_cast<float>(r), static_cast<float>(r + 1)};
  }
  // Fix valid keys to match predefined set.
  for (int32_t r = 0; r < kNumRows; ++r) {
    const std::vector<int32_t> validKeys = {5, 3, 7, 11, 2};
    rowKeys[r][0] = validKeys[r % 5];
  }
  auto map = createMapVector(pool_.get(), mapType, rowKeys, rowValues);
  auto input = std::make_shared<velox::RowVector>(
      pool_.get(), type, nullptr, kNumRows, std::vector<velox::VectorPtr>{map});

  EXPECT_THROW(
      serializer.serialize(input, OrderedRanges::of(0, kNumRows)),
      NimbleUserError);
}

TEST_P(SerializationTest, flatmapColumnsKeysEmptyData) {
  // Pre-register 8 keys, serialize 200 rows of empty maps across 4 serialize
  // calls. Schema should still contain all predefined keys.
  auto type = velox::ROW({
      {"m", velox::MAP(velox::INTEGER(), velox::REAL())},
  });

  const auto mapType = velox::MAP(velox::INTEGER(), velox::REAL());
  const std::set<std::string> predefinedKeys = {
      "5", "3", "7", "11", "2", "9", "14", "1"};
  const std::vector<std::string> sortedKeys(
      predefinedKeys.begin(), predefinedKeys.end());
  constexpr int32_t kNumKeys = 8;
  constexpr int32_t kRowsPerBatch = 50;
  constexpr int32_t kBatches = 4;

  SerializerOptions options{
      .version = version(),
      .flatMapColumns = {{"m", predefinedKeys}},
  };
  Serializer serializer{options, type, pool_.get()};

  for (int32_t b = 0; b < kBatches; ++b) {
    // All empty maps.
    std::vector<std::vector<int32_t>> rowKeys(kRowsPerBatch);
    std::vector<std::vector<float>> rowValues(kRowsPerBatch);
    auto map = createMapVector(pool_.get(), mapType, rowKeys, rowValues);
    auto input = std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        kRowsPerBatch,
        std::vector<velox::VectorPtr>{map});
    serializer.serialize(input, OrderedRanges::of(0, kRowsPerBatch));
  }

  // Verify schema contains all 8 predefined keys in sorted order.
  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  const auto& root = schema->asRow();
  ASSERT_EQ(root.childrenCount(), 1);
  const auto& flatMap = root.childAt(0)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), kNumKeys);
  for (int i = 0; i < kNumKeys; ++i) {
    EXPECT_EQ(flatMap.nameAt(i), sortedKeys[i]);
  }
}

TEST_P(SerializationTest, flatmapColumnsKeysRejectsRowIngestion) {
  // When flatMapColumns keys are configured, passing a ROW vector (instead of
  // MAP) should throw. Test with 100 rows and 5 predefined keys.
  auto mapType = velox::MAP(velox::INTEGER(), velox::REAL());
  auto type = velox::ROW({{"m", mapType}});

  constexpr int32_t kNumRows = 100;

  SerializerOptions options{
      .version = version(),
      .flatMapColumns = {{"m", {"11", "2", "3", "5", "7"}}},
  };
  Serializer serializer{options, type, pool_.get()};

  // Build a ROW vector to pass as the flatmap child (triggers ingestRow).
  auto rowChild = velox::BaseVector::create(
      velox::ROW({{"a", velox::REAL()}}), kNumRows, pool_.get());
  auto input = std::make_shared<velox::RowVector>(
      pool_.get(),
      velox::ROW({{"m", rowChild->type()}}),
      nullptr,
      kNumRows,
      std::vector<velox::VectorPtr>{rowChild});

  EXPECT_THROW(
      serializer.serialize(input, OrderedRanges::of(0, kNumRows)),
      NimbleUserError);
}

// Fuzz test that serializes batches with different versions (cycling through
// kLegacyCompact, kLegacyCompact+Delta), deserializes each batch, and
// verifies round-trip correctness.
TEST_F(SerializationTest, fuzzMixedVersionSerialization) {
  auto type = velox::ROW({
      {"bool_val", velox::BOOLEAN()},
      {"int_val", velox::INTEGER()},
      {"long_val", velox::BIGINT()},
      {"double_val", velox::DOUBLE()},
      {"string_val", velox::VARCHAR()},
  });

  const auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;

  const size_t batchSize = 20;
  velox::VectorFuzzer fuzzer(
      {
          .vectorSize = batchSize,
          .nullRatio = 0,
          .stringLength = 20,
          .stringVariableLength = true,
      },
      pool_.get(),
      seed);

  // Versions to cycle through for each batch.
  const std::vector<SerializerOptions> serializerVersions = {
      {.version = SerializationVersion::kSerialization},
      {.version = SerializationVersion::kSerialization,
       .streamIndicesEncodingType = EncodingType::Delta,
       .streamSizesEncodingType = EncodingType::Delta},
  };

  const int iterations = 10;
  const int batchesPerIteration = 6;
  for (int iter = 0; iter < iterations; ++iter) {
    SCOPED_TRACE(fmt::format("iteration {}", iter));

    // Serialize batches with cycling versions and collect buffers + expected
    // vectors.
    std::vector<std::string> serializedBuffers;
    std::shared_ptr<const Type> schema;
    velox::VectorPtr expected;

    for (int i = 0; i < batchesPerIteration; ++i) {
      const auto& opts = serializerVersions[i % serializerVersions.size()];

      auto input = fuzzer.fuzzInputRow(
          std::dynamic_pointer_cast<const velox::RowType>(type));

      Serializer serializer{opts, type, pool_.get()};
      serializedBuffers.emplace_back(
          serializer.serialize(input, OrderedRanges::of(0, input->size())));
      if (schema == nullptr) {
        schema =
            SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
      }

      if (expected == nullptr) {
        expected = input;
      } else {
        const auto oldSize = expected->size();
        expected->resize(oldSize + input->size());
        expected->copy(input.get(), oldSize, 0, input->size());
      }
    }

    // Deserialize all batches together.
    std::vector<std::string_view> views;
    views.reserve(serializedBuffers.size());
    for (const auto& buf : serializedBuffers) {
      views.emplace_back(buf);
    }

    Deserializer deserializer(
        schema, pool_.get(), DeserializerOptions{.hasHeader = true});
    velox::VectorPtr output;
    deserializer.deserialize(views, output);

    ASSERT_EQ(output->size(), expected->size());
    for (velox::vector_size_t i = 0; i < expected->size(); ++i) {
      ASSERT_TRUE(output->equalValueAt(expected.get(), i, i))
          << "Mismatch at row " << i << "\nExpected: " << expected->toString(i)
          << "\nActual: " << output->toString(i);
    }
  }
}

// Verifies that parallel decode is triggered via coroutine-based batched
// dispatch and produces correct results for RowFieldReader.
DEBUG_ONLY_TEST_P(SerializationTest, parallelDecodeRow) {
  velox::common::testutil::TestValue::enable();

  auto type = velox::ROW({
      {"a", velox::BIGINT()},
      {"b", velox::INTEGER()},
      {"c", velox::DOUBLE()},
      {"d", velox::REAL()},
      {"e", velox::VARCHAR()},
      {"f", velox::BIGINT()},
      {"g", velox::INTEGER()},
      {"h", velox::DOUBLE()},
  });

  const size_t batchSize = 50;
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  velox::VectorFuzzer fuzzer(
      {.vectorSize = batchSize, .nullRatio = 0, .stringLength = 10},
      pool_.get(),
      seed);

  SerializerOptions serOptions{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
  };
  Serializer serializer{serOptions, type, pool_.get()};

  const size_t numBatches = 5;
  std::vector<velox::VectorPtr> inputs;
  std::vector<std::string> serialized;
  for (size_t i = 0; i < numBatches; ++i) {
    auto input = fuzzer.fuzzInputRow(
        std::dynamic_pointer_cast<const velox::RowType>(type));
    serialized.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
    inputs.emplace_back(std::move(input));
  }

  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  struct ParallelDecodeParam {
    uint32_t maxDecodeParallelism;
    uint32_t minStreamsPerDecodeUnit;
    uint32_t expectedTaskCount;

    std::string debugString() const {
      return fmt::format(
          "maxParallel={}, minStreams={}, expectedTasks={}",
          maxDecodeParallelism,
          minStreamsPerDecodeUnit,
          expectedTaskCount);
    }
  };

  // 8 children: test different parallelism combinations.
  // parallelDecodeEnabled requires numChildren >= minStreamsPerDecodeUnit * 2,
  // so all test cases must satisfy 8 >= minStreams * 2.
  const std::vector<ParallelDecodeParam> testCases = {
      {2, 1, 2}, // 2 tasks, 4 children each
      {4, 1, 4}, // 4 tasks, 2 children each
      {8, 1, 8}, // 8 tasks, 1 child each
      {4, 4, 2}, // min 4 streams/task -> 8/4=2 tasks
      {8, 4, 2}, // min 4 streams/task -> 8/4=2 tasks
      {100, 1, 8}, // clamped to numChildren=8
  };

  folly::CPUThreadPoolExecutor executor(4);

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    auto opts = deserializerOptions();
    opts.decodeExecutor = &executor;
    opts.maxDecodeParallelism = testCase.maxDecodeParallelism;
    opts.minStreamsPerDecodeUnit = testCase.minStreamsPerDecodeUnit;

    Deserializer deserializer{schema, pool_.get(), opts};

    uint32_t parallelDecodeCount = 0;
    std::vector<uint32_t> observedTaskCounts;
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::RowFieldReader::co_next",
        std::function<void(const uint32_t*)>([&](const uint32_t* taskCount) {
          ++parallelDecodeCount;
          observedTaskCounts.emplace_back(*taskCount);
        }));

    velox::VectorPtr output;
    for (size_t i = 0; i < numBatches; ++i) {
      deserializer.deserialize(serialized[i], output);
      ASSERT_EQ(output->size(), inputs[i]->size());
      for (velox::vector_size_t row = 0; row < output->size(); ++row) {
        ASSERT_TRUE(output->equalValueAt(inputs[i].get(), row, row))
            << "Mismatch at batch " << i << " row " << row;
      }
    }

    EXPECT_EQ(parallelDecodeCount, numBatches);
    for (const auto taskCount : observedTaskCounts) {
      EXPECT_EQ(taskCount, testCase.expectedTaskCount);
    }
  }
}

// Verifies that parallel decode is triggered for StructFlatMapFieldReader
// when deserializing flatmap-as-struct with parallel decode options.
DEBUG_ONLY_TEST_P(SerializationTest, parallelDecodeFlatMapAsStruct) {
  velox::common::testutil::TestValue::enable();

  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  const size_t batchSize = 50;
  const std::vector<int32_t> allKeys = {1, 2, 3, 4, 5, 6, 7, 8};

  auto generateInput = [&]() -> velox::VectorPtr {
    const auto numRows = static_cast<velox::vector_size_t>(batchSize);
    const auto numKeys = static_cast<velox::vector_size_t>(allKeys.size());

    auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
    }

    auto keysFlat = velox::BaseVector::create(
        velox::INTEGER(), numRows * numKeys, pool_.get());
    auto valuesFlat = velox::BaseVector::create(
        velox::DOUBLE(), numRows * numKeys, pool_.get());
    velox::vector_size_t offset = 0;
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      for (auto key : allKeys) {
        keysFlat->asFlatVector<int32_t>()->set(offset, key);
        valuesFlat->asFlatVector<double>()->set(offset, i * 10.0 + key);
        ++offset;
      }
    }

    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool_.get()),
        velox::allocateSizes(numRows, pool_.get()),
        keysFlat,
        valuesFlat);
    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = i * numKeys;
      rawSizes[i] = numKeys;
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{ids, mapVector});
  };

  const SerializerOptions serOptions{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{serOptions, type, pool_.get()};

  const size_t numBatches = 5;
  std::vector<velox::VectorPtr> inputs;
  std::vector<std::string> serialized;
  for (size_t i = 0; i < numBatches; ++i) {
    auto input = generateInput();
    serialized.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
    inputs.emplace_back(std::move(input));
  }

  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  // Build struct output type from flatmap keys.
  auto outputType = buildOutputTypeForFlatMapAsStruct(*schema);

  struct ParallelDecodeParam {
    uint32_t maxDecodeParallelism;
    uint32_t minStreamsPerDecodeUnit;
    // Expected task count for the StructFlatMapFieldReader (8 keys).
    uint32_t expectedFlatMapTaskCount;
    // Whether parallel decode is expected for the Row (2 children: id +
    // features). Depends on whether 2 >= minStreamsPerDecodeUnit * 2.
    bool rowParallelExpected;

    std::string debugString() const {
      return fmt::format(
          "maxParallel={}, minStreams={}, expectedFlatMapTasks={}, rowParallel={}",
          maxDecodeParallelism,
          minStreamsPerDecodeUnit,
          expectedFlatMapTaskCount,
          rowParallelExpected);
    }
  };

  // 8 flatmap keys -> 8 children in StructFlatMapFieldReader.
  // parallelDecodeEnabled requires numChildren >= minStreamsPerDecodeUnit * 2.
  const std::vector<ParallelDecodeParam> testCases = {
      {2, 1, 2, true},
      {4, 1, 4, true},
      {4, 4, 2, false}, // Row: 2 < 4*2=8
      {100, 1, 8, true},
  };

  folly::CPUThreadPoolExecutor executor(4);

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    auto opts = deserializerOptions();
    opts.outputType = outputType;
    opts.decodeExecutor = &executor;
    opts.maxDecodeParallelism = testCase.maxDecodeParallelism;
    opts.minStreamsPerDecodeUnit = testCase.minStreamsPerDecodeUnit;

    Deserializer deserializer{schema, pool_.get(), opts};

    uint32_t rowParallelCount = 0;
    uint32_t flatMapParallelCount = 0;
    std::vector<uint32_t> flatMapTaskCounts;
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::RowFieldReader::co_next",
        std::function<void(const uint32_t*)>(
            [&](const uint32_t*) { ++rowParallelCount; }));
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::StructFlatMapFieldReader::co_next",
        std::function<void(const uint32_t*)>([&](const uint32_t* taskCount) {
          ++flatMapParallelCount;
          flatMapTaskCounts.emplace_back(*taskCount);
        }));

    velox::VectorPtr output;
    for (size_t i = 0; i < numBatches; ++i) {
      deserializer.deserialize(serialized[i], output);
      ASSERT_EQ(output->size(), batchSize);

      auto* outputRow = output->as<velox::RowVector>();
      ASSERT_NE(outputRow, nullptr);

      // Verify id column.
      auto* idVector = outputRow->childAt(0)->asFlatVector<int64_t>();
      for (velox::vector_size_t row = 0; row < output->size(); ++row) {
        EXPECT_EQ(idVector->valueAt(row), row);
      }

      // Verify features struct.
      auto* featuresRow = outputRow->childAt(1)->as<velox::RowVector>();
      ASSERT_NE(featuresRow, nullptr);
      ASSERT_EQ(featuresRow->childrenSize(), allKeys.size());
      for (size_t k = 0; k < allKeys.size(); ++k) {
        auto* values = featuresRow->childAt(k)->asFlatVector<double>();
        for (velox::vector_size_t row = 0; row < output->size(); ++row) {
          EXPECT_DOUBLE_EQ(values->valueAt(row), row * 10.0 + allKeys[k]);
        }
      }
    }

    EXPECT_EQ(rowParallelCount, testCase.rowParallelExpected ? numBatches : 0);
    EXPECT_EQ(flatMapParallelCount, numBatches);
    for (const auto taskCount : flatMapTaskCounts) {
      EXPECT_EQ(taskCount, testCase.expectedFlatMapTaskCount);
    }
  }
}

// Verifies that parallel decode is not triggered for StructFlatMapFieldReader
// when only 1 key is present (not enough non-null children for parallelism).
DEBUG_ONLY_TEST_P(SerializationTest, parallelDecodeSkippedFewKeys) {
  velox::common::testutil::TestValue::enable();

  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  const size_t batchSize = 50;
  // Only 1 key in the data.
  const std::vector<int32_t> dataKeys = {1};

  auto generateInput = [&]() -> velox::VectorPtr {
    const auto numRows = static_cast<velox::vector_size_t>(batchSize);
    const auto numKeys = static_cast<velox::vector_size_t>(dataKeys.size());

    auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
    }

    auto keysFlat = velox::BaseVector::create(
        velox::INTEGER(), numRows * numKeys, pool_.get());
    auto valuesFlat = velox::BaseVector::create(
        velox::DOUBLE(), numRows * numKeys, pool_.get());
    velox::vector_size_t offset = 0;
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      for (auto key : dataKeys) {
        keysFlat->asFlatVector<int32_t>()->set(offset, key);
        valuesFlat->asFlatVector<double>()->set(offset, i * 10.0 + key);
        ++offset;
      }
    }

    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool_.get()),
        velox::allocateSizes(numRows, pool_.get()),
        keysFlat,
        valuesFlat);
    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = i * numKeys;
      rawSizes[i] = numKeys;
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{ids, mapVector});
  };

  const SerializerOptions serOptions{
      .compressionType = CompressionType::Zstd,
      .compressionThreshold = 32,
      .compressionLevel = 3,
      .version = version(),
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{serOptions, type, pool_.get()};

  const size_t numBatches = 3;
  std::vector<velox::VectorPtr> inputs;
  std::vector<std::string> serialized;
  for (size_t i = 0; i < numBatches; ++i) {
    auto input = generateInput();
    serialized.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
    inputs.emplace_back(std::move(input));
  }

  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  auto outputType = buildOutputTypeForFlatMapAsStruct(*schema);

  folly::CPUThreadPoolExecutor executor(4);

  auto opts = deserializerOptions();
  opts.outputType = outputType;
  opts.decodeExecutor = &executor;
  opts.maxDecodeParallelism = 4;
  opts.minStreamsPerDecodeUnit = 1;

  Deserializer deserializer{schema, pool_.get(), opts};

  std::vector<uint32_t> flatMapTaskCounts;
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::StructFlatMapFieldReader::co_next",
      std::function<void(const uint32_t*)>([&](const uint32_t* taskCount) {
        flatMapTaskCounts.emplace_back(*taskCount);
      }));

  velox::VectorPtr output;
  for (size_t i = 0; i < numBatches; ++i) {
    deserializer.deserialize(serialized[i], output);
    ASSERT_EQ(output->size(), batchSize);

    auto* outputRow = output->as<velox::RowVector>();
    auto* featuresRow = outputRow->childAt(1)->as<velox::RowVector>();
    ASSERT_EQ(featuresRow->childrenSize(), 1);
    auto* values = featuresRow->childAt(0)->asFlatVector<double>();
    for (velox::vector_size_t row = 0; row < output->size(); ++row) {
      EXPECT_DOUBLE_EQ(values->valueAt(row), row * 10.0 + dataKeys[0]);
    }
  }

  // Only 1 non-null key node -> taskCount should be 1 (single coroutine task,
  // no parallelism benefit).
  for (const auto taskCount : flatMapTaskCounts) {
    EXPECT_EQ(taskCount, 1);
  }
}

// Verifies that parallel decode is NOT triggered when the executor is not set
// or maxDecodeParallelism <= 1, even when the data has enough children to be
// eligible for parallel decode.
DEBUG_ONLY_TEST_P(SerializationTest, parallelDecodeDisabled) {
  velox::common::testutil::TestValue::enable();

  // 8 children — enough for parallel decode if settings were enabled.
  auto type = velox::ROW({
      {"a", velox::BIGINT()},
      {"b", velox::INTEGER()},
      {"c", velox::DOUBLE()},
      {"d", velox::REAL()},
      {"e", velox::VARCHAR()},
      {"f", velox::BIGINT()},
      {"g", velox::INTEGER()},
      {"h", velox::DOUBLE()},
  });

  const size_t batchSize = 50;
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  velox::VectorFuzzer fuzzer(
      {.vectorSize = batchSize, .nullRatio = 0, .stringLength = 10},
      pool_.get(),
      seed);

  SerializerOptions serOptions{.version = version()};
  Serializer serializer{serOptions, type, pool_.get()};

  const size_t numBatches = 3;
  std::vector<velox::VectorPtr> inputs;
  std::vector<std::string> serialized;
  for (size_t i = 0; i < numBatches; ++i) {
    auto input = fuzzer.fuzzInputRow(
        std::dynamic_pointer_cast<const velox::RowType>(type));
    serialized.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
    inputs.emplace_back(std::move(input));
  }

  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());

  struct TestParam {
    std::string label;
    bool hasExecutor;
    uint32_t maxDecodeParallelism;

    std::string debugString() const {
      return fmt::format(
          "{}: hasExecutor={}, maxParallel={}",
          label,
          hasExecutor,
          maxDecodeParallelism);
    }
  };

  const std::vector<TestParam> testCases = {
      {"no executor", false, 4},
      {"maxParallel=0", true, 0},
      {"maxParallel=1", true, 1},
  };

  folly::CPUThreadPoolExecutor executor(4);

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    auto opts = deserializerOptions();
    opts.decodeExecutor = testCase.hasExecutor ? &executor : nullptr;
    opts.maxDecodeParallelism = testCase.maxDecodeParallelism;
    opts.minStreamsPerDecodeUnit = 1;

    Deserializer deserializer{schema, pool_.get(), opts};

    uint32_t rowParallelCount = 0;
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::RowFieldReader::co_next",
        std::function<void(const uint32_t*)>(
            [&](const uint32_t*) { ++rowParallelCount; }));

    velox::VectorPtr output;
    for (size_t i = 0; i < numBatches; ++i) {
      deserializer.deserialize(serialized[i], output);
      ASSERT_EQ(output->size(), inputs[i]->size());
      for (velox::vector_size_t row = 0; row < output->size(); ++row) {
        ASSERT_TRUE(output->equalValueAt(inputs[i].get(), row, row))
            << "Mismatch at batch " << i << " row " << row;
      }
    }

    EXPECT_EQ(rowParallelCount, 0)
        << "Parallel decode should NOT be triggered for: "
        << testCase.debugString();
  }
}

// Verifies that parallel decode is NOT triggered for flatmap-as-struct when
// the executor is not set or maxDecodeParallelism <= 1, even when the data
// has enough keys (8) to be eligible.
DEBUG_ONLY_TEST_P(SerializationTest, parallelDecodeDisabledFlatMap) {
  velox::common::testutil::TestValue::enable();

  auto type = velox::ROW({
      {"id", velox::BIGINT()},
      {"features", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
  });

  const size_t batchSize = 50;
  const std::vector<int32_t> allKeys = {1, 2, 3, 4, 5, 6, 7, 8};

  auto generateInput = [&]() -> velox::VectorPtr {
    const auto numRows = static_cast<velox::vector_size_t>(batchSize);
    const auto numKeys = static_cast<velox::vector_size_t>(allKeys.size());

    auto ids = velox::BaseVector::create(velox::BIGINT(), numRows, pool_.get());
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      ids->asFlatVector<int64_t>()->set(i, i);
    }

    auto keysFlat = velox::BaseVector::create(
        velox::INTEGER(), numRows * numKeys, pool_.get());
    auto valuesFlat = velox::BaseVector::create(
        velox::DOUBLE(), numRows * numKeys, pool_.get());
    velox::vector_size_t offset = 0;
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      for (auto key : allKeys) {
        keysFlat->asFlatVector<int32_t>()->set(offset, key);
        valuesFlat->asFlatVector<double>()->set(offset, i * 10.0 + key);
        ++offset;
      }
    }

    auto mapVector = std::make_shared<velox::MapVector>(
        pool_.get(),
        velox::MAP(velox::INTEGER(), velox::DOUBLE()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool_.get()),
        velox::allocateSizes(numRows, pool_.get()),
        keysFlat,
        valuesFlat);
    auto* rawOffsets =
        mapVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        mapVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    for (velox::vector_size_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = i * numKeys;
      rawSizes[i] = numKeys;
    }

    return std::make_shared<velox::RowVector>(
        pool_.get(),
        type,
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{ids, mapVector});
  };

  const SerializerOptions serOptions{
      .version = version(),
      .flatMapColumns = {{"features", {}}},
  };
  Serializer serializer{serOptions, type, pool_.get()};

  const size_t numBatches = 3;
  std::vector<velox::VectorPtr> inputs;
  std::vector<std::string> serialized;
  for (size_t i = 0; i < numBatches; ++i) {
    auto input = generateInput();
    serialized.emplace_back(
        serializer.serialize(input, OrderedRanges::of(0, input->size())));
    inputs.emplace_back(std::move(input));
  }

  auto schema =
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes());
  auto outputType = buildOutputTypeForFlatMapAsStruct(*schema);

  struct TestParam {
    std::string label;
    bool hasExecutor;
    uint32_t maxDecodeParallelism;

    std::string debugString() const {
      return fmt::format(
          "{}: hasExecutor={}, maxParallel={}",
          label,
          hasExecutor,
          maxDecodeParallelism);
    }
  };

  const std::vector<TestParam> testCases = {
      {"no executor", false, 4},
      {"maxParallel=0", true, 0},
      {"maxParallel=1", true, 1},
  };

  folly::CPUThreadPoolExecutor executor(4);

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());

    auto opts = deserializerOptions();
    opts.outputType = outputType;
    opts.decodeExecutor = testCase.hasExecutor ? &executor : nullptr;
    opts.maxDecodeParallelism = testCase.maxDecodeParallelism;
    opts.minStreamsPerDecodeUnit = 1;

    Deserializer deserializer{schema, pool_.get(), opts};

    uint32_t rowParallelCount = 0;
    uint32_t flatMapParallelCount = 0;
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::RowFieldReader::co_next",
        std::function<void(const uint32_t*)>(
            [&](const uint32_t*) { ++rowParallelCount; }));
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::StructFlatMapFieldReader::co_next",
        std::function<void(const uint32_t*)>(
            [&](const uint32_t*) { ++flatMapParallelCount; }));

    velox::VectorPtr output;
    for (size_t i = 0; i < numBatches; ++i) {
      deserializer.deserialize(serialized[i], output);
      ASSERT_EQ(output->size(), batchSize);

      auto* outputRow = output->as<velox::RowVector>();
      auto* idVector = outputRow->childAt(0)->asFlatVector<int64_t>();
      for (velox::vector_size_t row = 0; row < output->size(); ++row) {
        EXPECT_EQ(idVector->valueAt(row), row);
      }

      auto* featuresRow = outputRow->childAt(1)->as<velox::RowVector>();
      ASSERT_EQ(featuresRow->childrenSize(), allKeys.size());
      for (size_t k = 0; k < allKeys.size(); ++k) {
        auto* values = featuresRow->childAt(k)->asFlatVector<double>();
        for (velox::vector_size_t row = 0; row < output->size(); ++row) {
          EXPECT_DOUBLE_EQ(values->valueAt(row), row * 10.0 + allKeys[k]);
        }
      }
    }

    EXPECT_EQ(rowParallelCount, 0)
        << "Row parallel decode should NOT be triggered for: "
        << testCase.debugString();
    EXPECT_EQ(flatMapParallelCount, 0)
        << "FlatMap parallel decode should NOT be triggered for: "
        << testCase.debugString();
  }
}

namespace {

// Master-format legacy kLegacyCompact Trivial trailer.
// Wire: [encodingByte:Trivial][denseSizes:u32[]][trailerSize:u32]
std::string buildLegacyTrivialTrailer(const std::vector<uint32_t>& denseSizes) {
  std::string trailer;
  trailer.push_back(static_cast<char>(EncodingType::Trivial));
  trailer.append(
      reinterpret_cast<const char*>(denseSizes.data()),
      denseSizes.size() * sizeof(uint32_t));
  const auto trailerSize = static_cast<uint32_t>(trailer.size());
  trailer.append(reinterpret_cast<const char*>(&trailerSize), sizeof(uint32_t));
  return trailer;
}

// Rewrites a kSerialization blob to bytes a master-format kLegacyCompact writer
// would have produced. The stream payloads are byte-identical across the two
// versions; this strips the kSerialization flags byte and rewrites the trailer.
std::string rewriteAsLegacyCompactRaw(std::string_view kSerializationBlob) {
  const char* end = kSerializationBlob.data() + kSerializationBlob.size();
  const char* headerEnd = kSerializationBlob.data();
  const auto header = serde::readSerializationHeader(headerEnd, end, true);
  NIMBLE_CHECK_EQ(header.version, SerializationVersion::kSerialization);
  const auto bodyStartOffset =
      static_cast<size_t>(headerEnd - kSerializationBlob.data());
  const auto flagsOffset = bodyStartOffset - 1;

  auto [sparseIndices, sparseSizes] =
      serde::detail::readTrailerStreamMetadata(end);
  const auto newTrailerSize = serde::detail::readTrailerSize(end);
  const auto bodyEndOffset =
      kSerializationBlob.size() - sizeof(uint32_t) - newTrailerSize;
  NIMBLE_CHECK_GE(
      bodyEndOffset, bodyStartOffset, "Truncated kSerialization body");

  std::vector<uint32_t> denseSizes;
  if (!sparseIndices.empty()) {
    denseSizes.assign(sparseIndices.back() + 1, 0);
    for (size_t i = 0; i < sparseIndices.size(); ++i) {
      denseSizes[sparseIndices[i]] = sparseSizes[i];
    }
  }

  std::string rewritten;
  rewritten.reserve(
      bodyEndOffset - 1 + sizeof(uint8_t) +
      denseSizes.size() * sizeof(uint32_t) + sizeof(uint32_t));
  rewritten.push_back(static_cast<char>(SerializationVersion::kLegacyCompact));
  rewritten.append(kSerializationBlob.substr(1, flagsOffset - 1));
  rewritten.append(kSerializationBlob.substr(
      bodyStartOffset, bodyEndOffset - bodyStartOffset));
  rewritten.append(buildLegacyTrivialTrailer(denseSizes));
  return rewritten;
}

std::string rewriteAsLegacySerialization(std::string_view kSerializationBlob) {
  const char* end = kSerializationBlob.data() + kSerializationBlob.size();
  const char* headerEnd = kSerializationBlob.data();
  const auto header = serde::readSerializationHeader(headerEnd, end, true);
  NIMBLE_CHECK_EQ(header.version, SerializationVersion::kSerialization);
  const auto bodyStartOffset =
      static_cast<size_t>(headerEnd - kSerializationBlob.data());
  const auto flagsOffset = bodyStartOffset - 1;

  std::string rewritten;
  rewritten.reserve(kSerializationBlob.size() - 1);
  rewritten.push_back(
      static_cast<char>(SerializationVersion::kLegacySerialization));
  rewritten.append(kSerializationBlob.substr(1, flagsOffset - 1));
  rewritten.append(kSerializationBlob.substr(bodyStartOffset));
  return rewritten;
}

} // namespace

// End-to-end safety net: prove that production kLegacyCompact blobs (version=2,
// legacy Trivial trailer) flow through the post-D108024182 dispatch path
// (StreamDataParser → legacy::readLegacyTrailerStreamMetadata) and decode to
// the same vector the writer produced. This is the load-bearing test for the
// "kLegacyCompact stays readable" guarantee — without it, the legacy reader's
// integration into the dispatch path is unverified.
TEST_F(SerializationTest, compactRawProductionBlobRoundtrip) {
  auto type = velox::ROW({
      {"bool_val", velox::BOOLEAN()},
      {"int_val", velox::INTEGER()},
      {"long_val", velox::BIGINT()},
      {"double_val", velox::DOUBLE()},
      {"string_val", velox::VARCHAR()},
  });

  velox::VectorFuzzer fuzzer(
      {
          .vectorSize = 64,
          .nullRatio = 0,
          .stringLength = 16,
          .stringVariableLength = true,
      },
      pool_.get(),
      /*seed=*/12345);
  auto input = fuzzer.fuzzInputRow(type);

  SerializerOptions writerOptions{
      .version = SerializationVersion::kSerialization,
  };
  Serializer serializer{writerOptions, type, pool_.get()};
  const auto kSerializationBlob =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  const std::string kLegacyCompactBlob =
      rewriteAsLegacyCompactRaw(kSerializationBlob);

  ASSERT_FALSE(kLegacyCompactBlob.empty());
  ASSERT_EQ(
      static_cast<uint8_t>(kLegacyCompactBlob[0]),
      static_cast<uint8_t>(SerializationVersion::kLegacyCompact));

  DeserializerOptions deserOptions{.hasHeader = true};
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserOptions};

  velox::VectorPtr output;
  deserializer.deserialize(kLegacyCompactBlob, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < input->size(); ++i) {
    EXPECT_TRUE(vectorEquals(input, output, i))
        << "Row " << i << " mismatch:\n  expected: " << input->toString(i)
        << "\n  actual:   " << output->toString(i);
  }
}

TEST_F(SerializationTest, legacySerializationProductionBlobRoundtrip) {
  auto type = velox::ROW({
      {"bool_val", velox::BOOLEAN()},
      {"int_val", velox::INTEGER()},
      {"long_val", velox::BIGINT()},
      {"double_val", velox::DOUBLE()},
      {"string_val", velox::VARCHAR()},
  });

  velox::VectorFuzzer fuzzer(
      {
          .vectorSize = 64,
          .nullRatio = 0,
          .stringLength = 16,
          .stringVariableLength = true,
      },
      pool_.get(),
      /*seed=*/12345);
  auto input = fuzzer.fuzzInputRow(type);

  SerializerOptions writerOptions{
      .version = SerializationVersion::kSerialization,
  };
  Serializer serializer{writerOptions, type, pool_.get()};
  const auto kSerializationBlob =
      serializer.serialize(input, OrderedRanges::of(0, input->size()));

  const std::string kLegacySerializationBlob =
      rewriteAsLegacySerialization(kSerializationBlob);

  ASSERT_FALSE(kLegacySerializationBlob.empty());
  ASSERT_EQ(
      static_cast<uint8_t>(kLegacySerializationBlob[0]),
      static_cast<uint8_t>(SerializationVersion::kLegacySerialization));

  DeserializerOptions deserOptions{.hasHeader = true};
  Deserializer deserializer{
      SchemaReader::getSchema(serializer.schemaBuilder().schemaNodes()),
      pool_.get(),
      deserOptions};

  velox::VectorPtr output;
  deserializer.deserialize(kLegacySerializationBlob, output);

  ASSERT_EQ(output->size(), input->size());
  for (velox::vector_size_t i = 0; i < input->size(); ++i) {
    EXPECT_TRUE(vectorEquals(input, output, i))
        << "Row " << i << " mismatch:\n  expected: " << input->toString(i)
        << "\n  actual:   " << output->toString(i);
  }
}

TEST_F(SerializationTest, serializerHonorsRequestedEncodedVersion) {
  auto type = velox::ROW({{"x", velox::INTEGER()}});
  auto col = velox::BaseVector::create(velox::INTEGER(), 1, pool_.get());
  col->asFlatVector<int32_t>()->set(0, 42);
  auto row = std::make_shared<velox::RowVector>(
      pool_.get(), type, nullptr, 1, std::vector<velox::VectorPtr>{col});

  SerializerOptions options{.version = SerializationVersion::kSerialization};
  Serializer serializer{options, type, pool_.get()};
  std::string blob;
  serializer.serialize(row, OrderedRanges::of(0, 1), blob);
  ASSERT_FALSE(blob.empty());
  EXPECT_EQ(
      static_cast<uint8_t>(blob[0]),
      static_cast<uint8_t>(SerializationVersion::kSerialization));
}

INSTANTIATE_TEST_SUITE_P(
    AllFormats,
    SerializationTest,
    ::testing::Values(
        // kSerialization format without compression.
        TestParams{.version = SerializationVersion::kSerialization},
        // kSerialization format with compression enabled (for
        // encodingLayoutTree tests).
        TestParams{
            .version = SerializationVersion::kSerialization,
            .compressionOptions =
                CompressionOptions{
                    .compressionAcceptRatio = 1.0f,
                    .zstdMinCompressionSize = 0}},
        // kSerialization format with Delta stream sizes encoding.
        TestParams{
            .version = SerializationVersion::kSerialization,
            .streamSizesEncodingType = EncodingType::Delta},
        // kSerialization format with FixedBitWidth stream sizes encoding.
        TestParams{
            .version = SerializationVersion::kSerialization,
            .streamSizesEncodingType = EncodingType::FixedBitWidth},
        // kSerialization format with Varint stream sizes encoding.
        TestParams{
            .version = SerializationVersion::kSerialization,
            .streamSizesEncodingType = EncodingType::Varint},
        // Buffer pool disabled variants.
        TestParams{
            .version = SerializationVersion::kSerialization,
            .bufferPoolCapacity = 0},
        TestParams{
            .version = SerializationVersion::kSerialization,
            .compressionOptions =
                CompressionOptions{
                    .compressionAcceptRatio = 1.0f,
                    .zstdMinCompressionSize = 0},
            .bufferPoolCapacity = 0},
        TestParams{
            .version = SerializationVersion::kSerialization,
            .streamSizesEncodingType = EncodingType::Delta,
            .bufferPoolCapacity = 0},
        TestParams{
            .version = SerializationVersion::kSerialization,
            .streamSizesEncodingType = EncodingType::FixedBitWidth,
            .bufferPoolCapacity = 0},
        TestParams{
            .version = SerializationVersion::kSerialization,
            .streamSizesEncodingType = EncodingType::Varint,
            .bufferPoolCapacity = 0}),
    formatName);
