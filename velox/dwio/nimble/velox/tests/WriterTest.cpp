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
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/system/HardwareConcurrency.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <unordered_map>

#include "velox/common/testutil/TestValue.h"

#include <utility>
#include "folly/FileUtil.h"
#include "folly/Random.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/common/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/PrefixEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/tests/RandomEncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/KeyEncoding.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/FileLayout.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/velox/BatchReader.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/SchemaUtils.h"
#include "velox/dwio/nimble/velox/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/velox/StatsGenerated.h"
#include "velox/dwio/nimble/velox/stats/VectorizedStatistics.h"
#include "velox/dwio/nimble/velox/tests/WriterTestUtils.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/serializers/KeyEncoder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook {
DEFINE_uint32(
    writer_tests_seed,
    0,
    "If provided, this seed will be used when executing tests. "
    "Otherwise, a random seed will be used.");

using nimble::test::makeTestTabletOptions;

// The Meta internal compressor has no OSS implementation. The replay policy
// redirects a request for it to Zstd, which the accept ratio may in turn reject
// in favour of leaving the data uncompressed, so both outcomes are valid there.
void expectMetaInternalCompression(nimble::CompressionType actual) {
#ifdef DISABLE_META_INTERNAL_COMPRESSOR
  EXPECT_TRUE(
      actual == nimble::CompressionType::Zstd ||
      actual == nimble::CompressionType::Uncompressed)
      << "Expected MetaInternal (redirected to Zstd or Uncompressed), but got "
      << nimble::toString(actual);
#else
  EXPECT_EQ(nimble::CompressionType::MetaInternal, actual);
#endif
}

class WriterTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::SharedArbitrator::registerFactory();
    velox::memory::MemoryManager::Options options;
    options.arbitratorKind = "SHARED";
    velox::memory::MemoryManager::testingSetInstance(options);
  }

  static void TearDownTestCase() {
    // registerFactory() throws if the kind is already registered, and every
    // suite in this binary runs SetUpTestCase. Pairing them leaves the
    // process-global registry as each suite found it, matching
    // OperatorTestBase::TearDownTestCase.
    velox::memory::SharedArbitrator::unregisterFactory();
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("default_root");
    leafPool_ = rootPool_->addLeafChild("default_leaf");
  }

  // Builds chunkCount chunks of rowsPerChunk int64 values where chunk 0 is
  // high-entropy random data (which selection encodes as a flat layout that can
  // re-encode any later chunk on replay) and every later chunk repeats a single
  // value (a constant layout). Without caching the chunks select different
  // encodings, so the cached-replay assertions are not vacuous. The fixed seed
  // makes the cached and control writers see identical data.
  static std::vector<std::vector<int64_t>>
  makeDivergentInt64Chunks(int chunkCount, int rowsPerChunk, uint32_t seed) {
    std::mt19937 rng{seed};
    std::vector<std::vector<int64_t>> chunkData(chunkCount);
    for (int chunk = 0; chunk < chunkCount; ++chunk) {
      if (chunk == 0) {
        chunkData[chunk].reserve(rowsPerChunk);
        for (int row = 0; row < rowsPerChunk; ++row) {
          chunkData[chunk].push_back(static_cast<int64_t>(rng()));
        }
      } else {
        chunkData[chunk].assign(rowsPerChunk, static_cast<int64_t>(rng()));
      }
    }
    return chunkData;
  }

  // Captures, in file order across all stripes, the EncodingLayout of every
  // chunk of the given top-level scalar column, asserting the file holds
  // expectedStripeCount stripes.
  std::vector<nimble::EncodingLayout> captureColumnChunkLayouts(
      const std::shared_ptr<velox::InMemoryReadFile>& readFile,
      int columnIndex,
      uint32_t expectedStripeCount) {
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    auto section =
        tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
    NIMBLE_CHECK(section.has_value(), "Schema not found.");
    auto schema =
        nimble::SchemaDeserializer::deserialize(section->content().data());
    auto offset = schema->asRow()
                      .childAt(columnIndex)
                      ->asScalar()
                      .scalarDescriptor()
                      .offset();

    EXPECT_EQ(tablet->stripeCount(), expectedStripeCount);
    std::vector<nimble::EncodingLayout> chunkLayouts;
    for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
      auto streams = tablet->load(
          tablet->stripeIdentifier(stripe), std::vector<uint32_t>{offset});
      nimble::InMemoryChunkedStream chunkedStream{
          *leafPool_, std::move(streams[0])};
      while (chunkedStream.hasNext()) {
        chunkLayouts.push_back(
            nimble::EncodingLayoutCapture::capture(
                chunkedStream.nextChunk(), nimble::Encoding::Options{}));
      }
    }
    return chunkLayouts;
  }

  std::vector<nimble::EncodingLayout> captureFlatMapKeyChunkLayouts(
      const std::shared_ptr<velox::InMemoryReadFile>& readFile,
      int columnIndex,
      std::string_view key,
      uint32_t expectedStripeCount) {
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    auto section =
        tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
    NIMBLE_CHECK(section.has_value(), "Schema not found.");
    auto schema =
        nimble::SchemaDeserializer::deserialize(section->content().data());
    const auto& flatMap = schema->asRow().childAt(columnIndex)->asFlatMap();
    auto child = flatMap.findChild(key);
    NIMBLE_CHECK(child.has_value());
    auto offset =
        flatMap.childAt(child.value())->asScalar().scalarDescriptor().offset();

    EXPECT_EQ(tablet->stripeCount(), expectedStripeCount);
    std::vector<nimble::EncodingLayout> chunkLayouts;
    for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
      auto streams = tablet->load(
          tablet->stripeIdentifier(stripe), std::vector<uint32_t>{offset});
      nimble::InMemoryChunkedStream chunkedStream{
          *leafPool_, std::move(streams[0])};
      while (chunkedStream.hasNext()) {
        chunkLayouts.push_back(
            nimble::EncodingLayoutCapture::capture(
                chunkedStream.nextChunk(), nimble::Encoding::Options{}));
      }
    }
    return chunkLayouts;
  }

  static std::unique_ptr<nimble::EncodingSelectionPolicyBase>
  sharedDictionaryFeatureSelectionPolicy(nimble::DataType dataType) {
    std::vector<std::pair<nimble::EncodingType, float>> readFactors;
    switch (dataType) {
      case nimble::DataType::Int32:
        readFactors = {{nimble::EncodingType::Dictionary, 1.0}};
        break;
      case nimble::DataType::Uint32:
        readFactors = {{nimble::EncodingType::FixedBitWidth, 1.0}};
        break;
      case nimble::DataType::Undefined:
      case nimble::DataType::Int8:
      case nimble::DataType::Uint8:
      case nimble::DataType::Int16:
      case nimble::DataType::Uint16:
      case nimble::DataType::Int64:
      case nimble::DataType::Uint64:
      case nimble::DataType::Float:
      case nimble::DataType::Double:
      case nimble::DataType::Bool:
      case nimble::DataType::String:
        readFactors = {{nimble::EncodingType::Trivial, 1.0}};
        break;
    }
    nimble::ManualEncodingSelectionPolicyFactory factory{
        readFactors, /*compressionOptions=*/std::nullopt};
    return factory.createPolicy(dataType);
  }

  // Structurally compares two captured EncodingLayouts, recursing into every
  // nested sub-encoding (e.g. a Dictionary's Alphabet/Indices), so a match
  // means the full encoding tree is identical, not just the top-level encoding
  // type.
  static bool encodingLayoutsEqual(
      const nimble::EncodingLayout& a,
      const nimble::EncodingLayout& b) {
    if (a.encodingType() != b.encodingType() ||
        a.compressionType() != b.compressionType() ||
        a.config().values() != b.config().values() ||
        a.childrenCount() != b.childrenCount()) {
      return false;
    }
    for (nimble::NestedEncodingIdentifier i = 0; i < a.childrenCount(); ++i) {
      const auto& childA = a.child(i);
      const auto& childB = b.child(i);
      if (childA.has_value() != childB.has_value()) {
        return false;
      }
      if (childA.has_value() && !encodingLayoutsEqual(*childA, *childB)) {
        return false;
      }
    }
    return true;
  }

  // Wraps each inner vector of chunkData (one per chunk) into a single BIGINT
  // column "c0" RowVector batch, for the BIGINT-specific cached-encoding tests.
  std::vector<velox::RowVectorPtr> bigintBatches(
      const std::vector<std::vector<int64_t>>& chunkData) {
    velox::test::VectorMaker vectorMaker{leafPool_.get()};
    std::vector<velox::RowVectorPtr> batches;
    batches.reserve(chunkData.size());
    for (const auto& chunkValues : chunkData) {
      batches.push_back(vectorMaker.rowVector(
          {"c0"}, {vectorMaker.flatVector<int64_t>(chunkValues)}));
    }
    return batches;
  }

  // Flush-policy factory for the cache tests: one chunk per batch, with a
  // stripe closed after every chunksPerStripe batches, so a single write
  // exercises the cache across both chunk and stripe boundaries (the default
  // expectation). With N batches this yields ceil(N / chunksPerStripe) stripes,
  // each holding multiple chunks. The batch counter lives in the returned
  // factory's closure because the writer invokes the factory afresh on every
  // write() (flush policies are stateful, see
  // Writer::evaluateFlushPolicy).
  static std::function<std::unique_ptr<nimble::FlushPolicy>()>
  chunkAndStripeFlushPolicyFactory(int chunksPerStripe) {
    auto batchesSinceFlush = std::make_shared<int>(0);
    return [batchesSinceFlush, chunksPerStripe]() {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /*flushLambda=*/
          [batchesSinceFlush, chunksPerStripe](const nimble::StripeProgress&) {
            if (++(*batchesSinceFlush) >= chunksPerStripe) {
              *batchesSinceFlush = 0;
              return true;
            }
            return false;
          },
          /*chunkLambda=*/[](auto&) { return true; });
    };
  }

  // A row of every scalar type Nimble encodes as a top-level scalar stream, for
  // the AllDataTypes variants. (TIMESTAMP is intentionally excluded: Nimble
  // does not represent it as a scalar node, so the scalar-stream capture idiom
  // in captureColumnChunkLayouts does not apply to it.)
  static velox::RowTypePtr allScalarTypesRow() {
    return velox::ROW({
        {"c_bool", velox::BOOLEAN()},
        {"c_tinyint", velox::TINYINT()},
        {"c_smallint", velox::SMALLINT()},
        {"c_int", velox::INTEGER()},
        {"c_bigint", velox::BIGINT()},
        {"c_real", velox::REAL()},
        {"c_double", velox::DOUBLE()},
        {"c_varchar", velox::VARCHAR()},
        {"c_varbinary", velox::VARBINARY()},
    });
  }

  // Builds chunkCount divergent batches of the given scalar-typed row schema:
  // chunk 0 is fuzzed random data (with some nulls, exercising the nullable
  // path), and every later chunk is constant (all rows identical, no nulls), so
  // without caching the chunks select different encodings per column.
  std::vector<velox::RowVectorPtr> makeDivergentBatches(
      const velox::RowTypePtr& type,
      int chunkCount,
      int rowsPerChunk,
      uint32_t seed) {
    velox::VectorFuzzer randomFuzzer(
        {.vectorSize = static_cast<size_t>(rowsPerChunk), .nullRatio = 0.1},
        leafPool_.get(),
        seed);
    velox::VectorFuzzer constantFuzzer(
        {.vectorSize = 1, .nullRatio = 0.0}, leafPool_.get(), seed + 1);

    std::vector<velox::RowVectorPtr> batches;
    batches.reserve(chunkCount);
    batches.push_back(randomFuzzer.fuzzInputFlatRow(type));

    const auto constantSeed = constantFuzzer.fuzzInputFlatRow(type);
    for (int chunk = 1; chunk < chunkCount; ++chunk) {
      std::vector<velox::VectorPtr> constantChildren;
      constantChildren.reserve(type->size());
      for (size_t column = 0; column < type->size(); ++column) {
        constantChildren.push_back(
            velox::BaseVector::wrapInConstant(
                rowsPerChunk,
                /*index=*/0,
                constantSeed->childAt(
                    static_cast<velox::column_index_t>(column))));
      }
      batches.push_back(
          std::make_shared<velox::RowVector>(
              leafPool_.get(),
              type,
              /*nulls=*/nullptr,
              rowsPerChunk,
              std::move(constantChildren)));
    }
    return batches;
  }

  // Writes each RowVector batch (one per chunk) of the given schema using
  // options, verifies every row round-trips via the Velox vector comparator
  // (replaying a layout onto divergent data must not corrupt values), and
  // returns the per-chunk EncodingLayouts captured for the given scalar column
  // across all stripes.
  std::vector<nimble::EncodingLayout> writeAndCaptureChunkLayouts(
      const velox::RowTypePtr& type,
      const std::vector<velox::RowVectorPtr>& batches,
      nimble::WriterOptions options,
      uint32_t expectedStripeCount,
      int columnIndex = 0) {
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));
    for (const auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();

    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);

    // Every output row must equal the corresponding input row, across all
    // columns (BaseVector::equalValueAt is type-agnostic and null-aware).
    {
      nimble::BatchReader reader(readFile.get(), *leafPool_);
      velox::VectorPtr result;
      size_t batchIndex = 0;
      velox::vector_size_t rowInBatch = 0;
      while (reader.next(/*rowCount=*/1024, result)) {
        for (velox::vector_size_t row = 0; row < result->size(); ++row) {
          while (batchIndex < batches.size() &&
                 rowInBatch == batches[batchIndex]->size()) {
            ++batchIndex;
            rowInBatch = 0;
          }
          NIMBLE_CHECK(
              batchIndex < batches.size(), "More rows read than written.");
          EXPECT_TRUE(
              result->equalValueAt(batches[batchIndex].get(), row, rowInBatch))
              << "row mismatch: chunk " << batchIndex << " row " << rowInBatch;
          ++rowInBatch;
        }
      }
      while (batchIndex < batches.size() &&
             rowInBatch == batches[batchIndex]->size()) {
        ++batchIndex;
        rowInBatch = 0;
      }
      EXPECT_EQ(batchIndex, batches.size()) << "Fewer rows read than written.";
    }

    return captureColumnChunkLayouts(
        readFile, columnIndex, expectedStripeCount);
  }

  // Convenience overload for the single-BIGINT-column tests: wraps raw int64
  // chunk data into RowVector batches and delegates to the vector-based helper.
  std::vector<nimble::EncodingLayout> writeAndCaptureChunkLayouts(
      const std::vector<std::vector<int64_t>>& chunkData,
      nimble::WriterOptions options,
      uint32_t expectedStripeCount) {
    return writeAndCaptureChunkLayouts(
        velox::ROW({{"c0", velox::BIGINT()}}),
        bigintBatches(chunkData),
        std::move(options),
        expectedStripeCount);
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

nimble::EncodingSelectionPolicyCreator makeEncodingSelectionPolicyCreator(
    nimble::EncodingType encodingType) {
  nimble::ManualEncodingSelectionPolicyFactory factory{
      {{encodingType, 1.0}}, std::nullopt};
  return [factory = std::move(factory)](nimble::DataType dataType)
             -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return factory.createPolicy(dataType);
  };
}

nimble::EncodingSelectionPolicyCreator makeRandomEncodingSelectionPolicyCreator(
    uint64_t seed,
    std::vector<nimble::EncodingType> candidateEncodingTypes = nimble::testing::
        RandomEncodingSelectionPolicyFactory::defaultEncodingChoices()) {
  nimble::testing::RandomEncodingSelectionPolicyFactory factory{
      seed, std::move(candidateEncodingTypes)};
  return [factory = std::move(factory)](nimble::DataType dataType)
             -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return factory.createPolicy(dataType);
  };
}

std::string writeWithWriterOptions(
    velox::memory::MemoryPool& rootPool,
    const velox::RowVectorPtr& vector,
    nimble::WriterOptions options) {
  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(), std::move(writeFile), rootPool, std::move(options));
  writer.write(vector);
  writer.close();
  return file;
}

// Writes |vector| to an in-memory Nimble file using |creator| for encoding
// selection.
std::string writeWithEncodingSelectionCreator(
    velox::memory::MemoryPool& rootPool,
    const velox::RowVectorPtr& vector,
    nimble::EncodingSelectionPolicyCreator creator) {
  nimble::WriterOptions options;
  options.encodingSelectionPolicyCreator = std::move(creator);
  return writeWithWriterOptions(rootPool, vector, std::move(options));
}

// A schema spanning the scalar physical types plus array/map/row nesting, so
// the random policy is exercised across every per-type compatible encoding set.
velox::RowTypePtr randomEncodingSelectionTestType() {
  return velox::ROW({
      {"i8", velox::TINYINT()},
      {"i32", velox::INTEGER()},
      {"i64", velox::BIGINT()},
      {"f32", velox::REAL()},
      {"f64", velox::DOUBLE()},
      {"b", velox::BOOLEAN()},
      {"s", velox::VARCHAR()},
      {"arr", velox::ARRAY(velox::INTEGER())},
      {"map", velox::MAP(velox::VARCHAR(), velox::BIGINT())},
      {"nested",
       velox::ROW({{"n1", velox::INTEGER()}, {"n2", velox::VARCHAR()}})},
  });
}

nimble::EncodingLayout captureFirstColumnEncoding(
    const std::string& file,
    velox::memory::MemoryPool* pool) {
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet =
      nimble::TabletReader::create(readFile, pool, makeTestTabletOptions(pool));
  auto section =
      tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
  NIMBLE_CHECK(section.has_value(), "Schema not found.");
  auto schema =
      nimble::SchemaDeserializer::deserialize(section->content().data());
  const auto& scalarNode = schema->asRow().childAt(0)->asScalar();

  std::vector<uint32_t> streamIdentifiers{
      scalarNode.scalarDescriptor().offset()};
  auto streams = tablet->load(tablet->stripeIdentifier(0), streamIdentifiers);
  nimble::InMemoryChunkedStream chunkedStream{*pool, std::move(streams[0])};
  NIMBLE_CHECK(chunkedStream.hasNext(), "Expected at least one chunk.");
  return nimble::EncodingLayoutCapture::capture(
      chunkedStream.nextChunk(), nimble::Encoding::Options{});
}

template <typename T>
void testAlpE2EWriterSelection(
    velox::memory::MemoryPool& rootPool,
    velox::memory::MemoryPool* leafPool) {
  SCOPED_TRACE(fmt::format("type={}", nimble::TypeTraits<T>::dataType));

  velox::test::VectorMaker vectorMaker{leafPool};
  auto vector = vectorMaker.rowVector(
      {"c0"}, {vectorMaker.flatVector<T>(512, [](auto row) {
        return static_cast<T>(static_cast<int32_t>(row % 101) - 50) /
            static_cast<T>(4);
      })});

  {
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.encodingSelectionPolicyCreator =
        makeEncodingSelectionPolicyCreator(nimble::EncodingType::ALP);
    nimble::Writer writer(
        vector->type(), std::move(writeFile), rootPool, std::move(options));
    writer.write(vector);
    writer.close();

    const auto captured = captureFirstColumnEncoding(file, leafPool);
    EXPECT_EQ(captured.encodingType(), nimble::EncodingType::ALP);
  }

  {
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.allowNestedAlpSelection = true;
    options.encodingSelectionPolicyCreator =
        makeEncodingSelectionPolicyCreator(nimble::EncodingType::Dictionary);
    nimble::Writer writer(
        vector->type(), std::move(writeFile), rootPool, std::move(options));
    writer.write(vector);
    writer.close();

    const auto captured = captureFirstColumnEncoding(file, leafPool);
    ASSERT_EQ(captured.encodingType(), nimble::EncodingType::Dictionary);
    const auto& alphabet =
        captured.child(nimble::EncodingIdentifiers::Dictionary::Alphabet);
    ASSERT_TRUE(alphabet.has_value());
    EXPECT_EQ(alphabet->encodingType(), nimble::EncodingType::ALP);
  }
}

TEST_F(WriterTest, emptyFile) {
  auto type = velox::ROW({{"simple", velox::INTEGER()}});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(type, std::move(writeFile), *rootPool_, {});
  writer.close();

  // Verify FileLayout for empty file using FileLayout::create()
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto layout = nimble::FileLayout::create(readFile, leafPool_.get());
  EXPECT_EQ(layout.fileSize, file.size());
  EXPECT_EQ(layout.postscript.majorVersion(), nimble::kVersionMajor);
  EXPECT_EQ(layout.postscript.minorVersion(), nimble::kVersionMinor);
  EXPECT_GT(layout.footer.size(), 0);
  EXPECT_TRUE(layout.stripeGroups.empty());
  EXPECT_TRUE(layout.indexPartitions.empty());
  EXPECT_TRUE(layout.stripesInfo.empty());

  nimble::BatchReader reader(readFile.get(), *leafPool_);
  velox::VectorPtr result;
  ASSERT_FALSE(reader.next(1, result));
}

TEST_F(WriterTest, buildEncodingOptionsPropagatesEncodingOptions) {
  {
    const nimble::WriterOptions options;
    const auto encodingOptions = options.buildEncodingOptions();
    EXPECT_FALSE(encodingOptions.fixedBitWidthUseExactBits);
    EXPECT_FALSE(encodingOptions.allowNestedAlpSelection);
  }

  for (const auto useExactBits : {false, true}) {
    SCOPED_TRACE(fmt::format("useExactBits={}", useExactBits));
    for (const auto allowNestedAlpSelection : {false, true}) {
      SCOPED_TRACE(
          fmt::format("allowNestedAlpSelection={}", allowNestedAlpSelection));
      nimble::WriterOptions options;
      options.fsstCompressionTargetRatio = 0.42;
      options.fixedBitWidthUseExactBits = useExactBits;
      options.allowNestedAlpSelection = allowNestedAlpSelection;

      const auto encodingOptions = options.buildEncodingOptions();

      EXPECT_DOUBLE_EQ(encodingOptions.fsstCompressionTargetRatio, 0.42);
      EXPECT_EQ(encodingOptions.fixedBitWidthUseExactBits, useExactBits);
      EXPECT_EQ(
          encodingOptions.allowNestedAlpSelection, allowNestedAlpSelection);
    }
  }
}

TEST_F(WriterTest, alpEncodingSelectionControlsWriterEncoding) {
  testAlpE2EWriterSelection<float>(*rootPool_, leafPool_.get());
  testAlpE2EWriterSelection<double>(*rootPool_, leafPool_.get());
}

// End-to-end check for the simplified MainlyConstant encode-selection: a dense
// mainly-constant column must still be selected as MainlyConstant and,
// alongside a high-cardinality column, round-trip byte-for-byte through the
// reader.
TEST_F(WriterTest, mainlyConstantRoundTrip) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  // Column 0: ~99% a single value with a handful of distinct uncommon values
  // -> should still be selected as MainlyConstant.
  auto dense = vectorMaker.flatVector<int64_t>(
      4096, [](auto row) { return row % 128 == 0 ? 1000 + row : 7; });
  // Column 1: all distinct (high cardinality) -> other-values priced via the
  // FixedBitWidth over-estimate.
  auto diverse = vectorMaker.flatVector<int64_t>(
      4096, [](auto row) { return static_cast<int64_t>(row); });
  auto vector = vectorMaker.rowVector({"dense", "diverse"}, {dense, diverse});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(vector->type(), std::move(writeFile), *rootPool_, {});
  writer.write(vector);
  writer.close();

  // The dense column is still encoded as MainlyConstant.
  const auto captured = captureFirstColumnEncoding(file, leafPool_.get());
  EXPECT_EQ(captured.encodingType(), nimble::EncodingType::MainlyConstant);

  // Both columns must read back exactly.
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  nimble::BatchReader reader(readFile.get(), *leafPool_);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(vector->size(), result));
  ASSERT_EQ(result->size(), vector->size());
  for (velox::vector_size_t i = 0; i < vector->size(); ++i) {
    EXPECT_TRUE(vector->equalValueAt(result.get(), i, i))
        << "mismatch at row " << i;
  }
  ASSERT_FALSE(reader.next(1, result));
}

DEBUG_ONLY_TEST_F(WriterTest, encodingPoolsPassedToEncodeOptions) {
  velox::common::testutil::TestValue::enable();

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {"a", "b", "c", "d"},
      {vectorMaker.flatVector<int64_t>(
           128,
           [](auto row) { return static_cast<int64_t>(row % 8); },
           [](auto row) { return row % 7 == 0; }),
       vectorMaker.flatVector<int64_t>(
           128, [](auto row) { return static_cast<int64_t>(row); }),
       vectorMaker.flatVector<int32_t>(
           128, [](auto row) { return static_cast<int32_t>(row % 11); }),
       vectorMaker.flatVector<double>(
           128, [](auto row) { return static_cast<double>(row) * 0.5; })});

  struct TestCase {
    const char* name;
    uint32_t maxCachedEncodingScratchBuffers;
    bool setMaxCachedNestedEncodingBuffers;
    uint32_t maxCachedNestedEncodingBuffers;
    uint32_t maxEncodeParallelism;
    bool expectEncodingScratchBufferPool;
    bool expectEncodingBufferPool;
    size_t expectedScratchPoolCount;
    size_t expectedEncodingPoolCount;
  };

  for (const auto& testCase : std::vector<TestCase>{
           {
               .name = "default",
               .maxCachedEncodingScratchBuffers = 0,
               .setMaxCachedNestedEncodingBuffers = false,
               .maxCachedNestedEncodingBuffers = 0,
               .maxEncodeParallelism = 0,
               .expectEncodingScratchBufferPool = false,
               .expectEncodingBufferPool = false,
               .expectedScratchPoolCount = 0,
               .expectedEncodingPoolCount = 0,
           },
           {
               .name = "nested cache disabled",
               .maxCachedEncodingScratchBuffers = 0,
               .setMaxCachedNestedEncodingBuffers = true,
               .maxCachedNestedEncodingBuffers = 0,
               .maxEncodeParallelism = 0,
               .expectEncodingScratchBufferPool = false,
               .expectEncodingBufferPool = false,
               .expectedScratchPoolCount = 0,
               .expectedEncodingPoolCount = 0,
           },
           {
               .name = "encoding scratch cache enabled",
               .maxCachedEncodingScratchBuffers = 1,
               .setMaxCachedNestedEncodingBuffers = false,
               .maxCachedNestedEncodingBuffers = 0,
               .maxEncodeParallelism = 0,
               .expectEncodingScratchBufferPool = true,
               .expectEncodingBufferPool = false,
               .expectedScratchPoolCount = 1,
               .expectedEncodingPoolCount = 0,
           },
           {
               .name = "nested cache enabled",
               .maxCachedEncodingScratchBuffers = 0,
               .setMaxCachedNestedEncodingBuffers = true,
               .maxCachedNestedEncodingBuffers = 1,
               .maxEncodeParallelism = 0,
               .expectEncodingScratchBufferPool = false,
               .expectEncodingBufferPool = true,
               .expectedScratchPoolCount = 0,
               .expectedEncodingPoolCount = 1,
           },
           {
               .name = "parallel encoding scratch cache enabled",
               .maxCachedEncodingScratchBuffers = 1,
               .setMaxCachedNestedEncodingBuffers = false,
               .maxCachedNestedEncodingBuffers = 0,
               .maxEncodeParallelism = 2,
               .expectEncodingScratchBufferPool = true,
               .expectEncodingBufferPool = false,
               .expectedScratchPoolCount = 2,
               .expectedEncodingPoolCount = 0,
           },
           {
               .name = "parallel nested cache enabled",
               .maxCachedEncodingScratchBuffers = 0,
               .setMaxCachedNestedEncodingBuffers = true,
               .maxCachedNestedEncodingBuffers = 1,
               .maxEncodeParallelism = 2,
               .expectEncodingScratchBufferPool = false,
               .expectEncodingBufferPool = true,
               .expectedScratchPoolCount = 0,
               .expectedEncodingPoolCount = 2,
           },
           {
               .name = "parallel caches enabled",
               .maxCachedEncodingScratchBuffers = 1,
               .setMaxCachedNestedEncodingBuffers = true,
               .maxCachedNestedEncodingBuffers = 1,
               .maxEncodeParallelism = 2,
               .expectEncodingScratchBufferPool = true,
               .expectEncodingBufferPool = true,
               .expectedScratchPoolCount = 2,
               .expectedEncodingPoolCount = 2,
           },
       }) {
    SCOPED_TRACE(testCase.name);

    uint32_t encodeCount{0};
    uint32_t pooledScratchEncodeCount{0};
    uint32_t pooledEncodingBufferCount{0};
    std::set<const void*> observedScratchPools;
    std::set<const void*> observedEncodingBufferPools;
    // Parallel encoding can invoke the callback concurrently.
    std::mutex statsMutex;
    std::condition_variable poolsObserved;
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::Writer::encode",
        std::function<void(nimble::Encoding::Options*)>(
            [&](nimble::Encoding::Options* encodingOptions) {
              std::unique_lock lock{statsMutex};
              ++encodeCount;
              if (encodingOptions->bufferPool != nullptr) {
                ++pooledScratchEncodeCount;
                observedScratchPools.insert(encodingOptions->bufferPool);
              }
              if (encodingOptions->encodingBufferPool != nullptr) {
                ++pooledEncodingBufferCount;
                observedEncodingBufferPools.insert(
                    encodingOptions->encodingBufferPool);
              }
              poolsObserved.notify_all();
              // Make the pool-count check deterministic. Without this wait,
              // one encoding task may consume every stream before the other
              // task is scheduled, especially on a single CPU.
              poolsObserved.wait(lock, [&] {
                return observedScratchPools.size() >=
                    testCase.expectedScratchPoolCount &&
                    observedEncodingBufferPools.size() >=
                    testCase.expectedEncodingPoolCount;
              });
            }));

    std::shared_ptr<folly::CPUThreadPoolExecutor> executor;
    nimble::WriterOptions options;
    options.maxCachedEncodingScratchBuffers =
        testCase.maxCachedEncodingScratchBuffers;
    if (testCase.setMaxCachedNestedEncodingBuffers) {
      options.maxCachedNestedEncodingBuffers =
          testCase.maxCachedNestedEncodingBuffers;
    }
    if (testCase.maxEncodeParallelism > 0) {
      executor = std::make_shared<folly::CPUThreadPoolExecutor>(
          testCase.maxEncodeParallelism);
      options.encodingExecutor = folly::getKeepAliveToken(*executor);
      options.maxEncodeParallelism = testCase.maxEncodeParallelism;
    }

    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::Writer writer(
        vector->type(), std::move(writeFile), *rootPool_, std::move(options));
    writer.write(vector);
    writer.close();

    EXPECT_GT(encodeCount, 0);
    EXPECT_EQ(
        pooledScratchEncodeCount,
        testCase.expectEncodingScratchBufferPool ? encodeCount : 0);
    EXPECT_EQ(
        pooledEncodingBufferCount,
        testCase.expectEncodingBufferPool ? encodeCount : 0);
    EXPECT_EQ(observedScratchPools.size(), testCase.expectedScratchPoolCount);
    EXPECT_EQ(
        observedEncodingBufferPools.size(), testCase.expectedEncodingPoolCount);
  }
}

TEST_F(WriterTest, fsstEncodingTargetControlsWriterEncoding) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {vectorMaker.flatVector<std::string>({std::string(2'000, 'x')})});

  struct TestCase {
    std::string name;
    double targetRatio;
    nimble::CompressionType fsstCompressionType;
    nimble::CompressionOptions compressionOptions{};
    nimble::EncodingType expectedEncodingType;
    nimble::CompressionType expectedCompressionType;
  };

  nimble::CompressionOptions zstdCompressionOptions;
  zstdCompressionOptions.compressionType = nimble::CompressionType::Zstd;
  zstdCompressionOptions.zstdMinCompressionSize = 0;

  for (const auto& testCase : std::vector<TestCase>{
           {
               .name = "permissive_target_uses_fsst",
               .targetRatio = 10.0,
               .fsstCompressionType = nimble::CompressionType::Uncompressed,
               .expectedEncodingType = nimble::EncodingType::Fsst,
               .expectedCompressionType = nimble::CompressionType::Uncompressed,
           },
           {
               .name = "strict_target_falls_back_to_compressed_trivial",
               .targetRatio = 0.0,
               .fsstCompressionType = nimble::CompressionType::Zstd,
               .compressionOptions = zstdCompressionOptions,
               .expectedEncodingType = nimble::EncodingType::Trivial,
               .expectedCompressionType = nimble::CompressionType::Zstd,
           },
       }) {
    SCOPED_TRACE(testCase.name);

    nimble::EncodingLayout fsstLayout{
        nimble::EncodingType::Fsst,
        {},
        testCase.fsstCompressionType,
        {nimble::EncodingLayout{
            nimble::EncodingType::Trivial,
            {},
            nimble::CompressionType::Uncompressed}}};

    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::Writer writer(
        vector->type(),
        std::move(writeFile),
        *rootPool_,
        {
            .encodingLayoutTree =
                nimble::EncodingLayoutTree{
                    nimble::Kind::Row,
                    {},
                    "",
                    {nimble::EncodingLayoutTree{
                        nimble::Kind::Scalar,
                        {{nimble::EncodingLayoutTree::StreamIdentifiers::
                              Scalar::ScalarStream,
                          std::move(fsstLayout)}},
                        "c0"}}},
            .compressionOptions = testCase.compressionOptions,
            .fsstCompressionTargetRatio = testCase.targetRatio,
        });
    writer.write(vector);
    writer.close();

    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    auto section =
        tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
    NIMBLE_CHECK(section.has_value(), "Schema not found.");
    auto schema =
        nimble::SchemaDeserializer::deserialize(section->content().data());
    const auto& scalarNode = schema->asRow().childAt(0)->asScalar();

    ASSERT_EQ(tablet->stripeCount(), 1);
    std::vector<uint32_t> streamIdentifiers{
        scalarNode.scalarDescriptor().offset()};
    auto streams = tablet->load(tablet->stripeIdentifier(0), streamIdentifiers);
    nimble::InMemoryChunkedStream chunkedStream{
        *leafPool_, std::move(streams[0])};
    ASSERT_TRUE(chunkedStream.hasNext());
    const auto capture = nimble::EncodingLayoutCapture::capture(
        chunkedStream.nextChunk(), nimble::Encoding::Options{});

    EXPECT_EQ(capture.encodingType(), testCase.expectedEncodingType);
    EXPECT_EQ(capture.compressionType(), testCase.expectedCompressionType);
    if (testCase.expectedEncodingType == nimble::EncodingType::Fsst) {
      EXPECT_EQ(
          capture.child(nimble::EncodingIdentifiers::Fsst::Lengths)
              ->encodingType(),
          nimble::EncodingType::Trivial);
    }
  }
}

TEST_F(WriterTest, emptyFileWithIndexEnabled) {
  auto type = velox::ROW({
      {"key_col", velox::INTEGER()},
      {"value_col", velox::VARCHAR()},
  });

  auto clusterIndexConfig =
      nimble::index::ClusterIndexConfigBuilder{}
          .withKeyColumns({"key_col"})
          .withSortOrders({nimble::SortOrder{.ascending = true}})
          .withEnforceKeyOrder(true)
          .build();

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      type,
      std::move(writeFile),
      *rootPool_,
      {.clusterIndexConfig = std::move(clusterIndexConfig)});
  writer.close();

  // Verify FileLayout for empty file with index enabled using
  // FileLayout::create()
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto layout = nimble::FileLayout::create(readFile, leafPool_.get());
  EXPECT_EQ(layout.fileSize, file.size());
  EXPECT_EQ(layout.stripesInfo.size(), 0);
  EXPECT_EQ(layout.stripeGroups.size(), 0);
  EXPECT_TRUE(layout.stripeGroups.empty());
  // Index groups should be empty for empty file (no stripes to index)
  EXPECT_TRUE(layout.indexPartitions.empty());
  EXPECT_TRUE(layout.stripesInfo.empty());

  nimble::BatchReader reader(readFile.get(), *leafPool_);
  velox::VectorPtr result;
  ASSERT_FALSE(reader.next(1, result));
}

TEST_F(WriterTest, exceptionOnClose) {
  class ThrowingWriteFile final : public velox::WriteFile {
   public:
    void append(std::string_view /* data */) final {
      throw std::runtime_error(uniqueErrorMessage());
    }
    void flush() final {
      throw std::runtime_error(uniqueErrorMessage());
    }
    void close() final {
      throw std::runtime_error(uniqueErrorMessage());
    }
    uint64_t size() const final {
      throw std::runtime_error(uniqueErrorMessage());
    }

   private:
    std::string uniqueErrorMessage() const {
      return "error/" + folly::to<std::string>(folly::Random::rand32());
    }
  };

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int32_t>({1, 2, 3})});

  std::string file;
  auto writeFile = std::make_unique<ThrowingWriteFile>();

  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.flushPolicyFactory = [&]() {
        return std::make_unique<nimble::LambdaFlushPolicy>(
            /*flushLambda=*/[&](auto&) { return true; });
      }});
  std::string error;
  try {
    writer.write(vector);
    FAIL() << "Expecting exception";
  } catch (const std::runtime_error& e) {
    EXPECT_TRUE(std::string{e.what()}.starts_with("error/"));
    error = e.what();
  }

  try {
    writer.write(vector);
    FAIL() << "Expecting exception";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(error, e.what());
  }

  try {
    writer.flush();
    FAIL() << "Expecting exception";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(error, e.what());
  }

  try {
    writer.close();
    FAIL() << "Expecting exception";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(error, e.what());
  }

  try {
    writer.close();
    FAIL() << "Expecting exception";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(error, e.what());
  }
}

TEST_F(WriterTest, emptyFileNoSchema) {
  const uint32_t batchSize = 10;
  auto type = velox::ROW({{"simple", velox::INTEGER()}});
  nimble::WriterOptions writerOptions;

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(writerOptions));
  writer.close();

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);

  velox::VectorPtr result;
  ASSERT_FALSE(reader.next(batchSize, result));
}

TEST_F(WriterTest, rootHasNulls) {
  auto batchSize = 5;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int32_t>(batchSize, [](auto row) {
        return row;
      })});

  // add nulls
  for (auto i = 0; i < batchSize; ++i) {
    vector->setNull(i, i % 2 == 0);
  }

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(vector->type(), std::move(writeFile), *rootPool_, {});

  writer.write(vector);
  writer.close();

  // Verify FileLayout for non-empty file without index using
  // FileLayout::create()
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto layout = nimble::FileLayout::create(readFile, leafPool_.get());
  EXPECT_EQ(layout.fileSize, file.size());
  EXPECT_EQ(layout.stripesInfo.size(), 1);
  EXPECT_EQ(layout.stripeGroups.size(), 1);
  EXPECT_EQ(layout.stripeGroups.size(), 1);
  // No index configured
  EXPECT_TRUE(layout.indexPartitions.empty());
  // Stripes metadata should be valid (stripeGroups not empty)
  EXPECT_GT(layout.stripes.size(), 0);
  EXPECT_LT(layout.stripes.offset(), layout.footer.offset());
  // Per-stripe info
  EXPECT_EQ(layout.stripesInfo.size(), 1);
  EXPECT_EQ(layout.stripesInfo[0].stripeGroupIndex, 0);
  EXPECT_GT(layout.stripesInfo[0].size, 0);

  nimble::BatchReader reader(readFile.get(), *leafPool_);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(batchSize, result));
  ASSERT_EQ(result->size(), batchSize);
  for (auto i = 0; i < batchSize; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i));
  }
}

TEST_F(WriterTest, schemaGrowthExtraColumn) {
  // File type has a single column
  const auto type = velox::ROW({"c0"}, {velox::BIGINT()});
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  // Expect the written values to match the file type
  velox::RowVectorPtr expectedVector = vectorMaker.rowVector(
      {"c0"}, {vectorMaker.flatVector<int64_t>({1, 2, 3})});

  // Add extra column into the written vector
  velox::RowVectorPtr vector = vectorMaker.rowVector(
      {"c0", "c1"},
      {vectorMaker.flatVector<int64_t>({1, 2, 3}),
       vectorMaker.flatVector<int64_t>({10, 20, 30})});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(type, std::move(writeFile), *rootPool_, {});
  writer.write(vector);
  writer.close();

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);

  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(3, result));
  ASSERT_EQ(result->size(), 3);
  ASSERT_EQ(*result->type(), *type);
  for (auto i = 0; i < 3; ++i) {
    ASSERT_TRUE(result->equalValueAt(expectedVector.get(), i, i));
  }
}

TEST_F(WriterTest, schemaGrowthExtraSubField) {
  // File type has a single column of type struct<f1>
  const auto type = velox::ROW({"c0"}, {velox::ROW({"f1"}, {velox::BIGINT()})});
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  // Expect the written values to match the file type
  velox::RowVectorPtr expectedVector = vectorMaker.rowVector(
      {"c0"},
      {vectorMaker.rowVector(
          {"f1"}, {vectorMaker.flatVector<int64_t>({1, 2, 3})})});

  // Add extra sub-field into the column: struct<f1> -> struct<f1, f2>
  velox::RowVectorPtr vector = vectorMaker.rowVector(
      {"c0"},
      {vectorMaker.rowVector(
          {"f1", "f2"},
          {vectorMaker.flatVector<int64_t>({1, 2, 3}),
           vectorMaker.flatVector<int64_t>({10, 20, 30})})});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(type, std::move(writeFile), *rootPool_, {});
  writer.write(vector);
  writer.close();

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);

  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(3, result));
  ASSERT_EQ(result->size(), 3);
  ASSERT_EQ(*result->type(), *type);
  for (auto i = 0; i < 3; ++i) {
    ASSERT_TRUE(result->equalValueAt(expectedVector.get(), i, i));
  }
}

TEST_F(WriterTest, featureReorderingNonFlatmapColumnIgnoresMismatchedConfig) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {"map", "flatmap"},
      {vectorMaker.mapVector<int32_t, int32_t>(
           5,
           /* sizeAt */ [](auto row) { return row % 3; },
           /* keyAt */ [](auto /* row */, auto mapIndex) { return mapIndex; },
           /* valueAt */ [](auto row, auto /* mapIndex */) { return row; },
           /* isNullAt */ [](auto /* row */) { return false; }),
       vectorMaker.mapVector<int32_t, int32_t>(
           5,
           /* sizeAt */ [](auto row) { return row % 3; },
           /* keyAt */ [](auto /* row */, auto mapIndex) { return mapIndex; },
           /* valueAt */ [](auto row, auto /* mapIndex */) { return row; },
           /* isNullAt */ [](auto /* row */) { return false; })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.flatMapColumns = {{"flatmap", {}}},
       .featureReordering =
           std::vector<std::tuple<size_t, std::vector<int64_t>>>{
               {0, {1, 2}}, {1, {3, 4}}}});
  writer.write(vector);
  writer.close();
}

// A single write whose map has more distinct keys than maxFlatMapKeys fails.
TEST_F(WriterTest, flatMapKeyLimitExceededInSingleWriteThrows) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  // One row with 10 distinct BIGINT keys against a limit of 4.
  auto vector = vectorMaker.rowVector(
      {"flatmap"},
      {vectorMaker.mapVector<int64_t, int64_t>(
          /* size */
          1,
          /* sizeAt */ [](auto) { return 10; },
          /* keyAt */ [](auto, auto mapIndex) -> int64_t { return mapIndex; },
          /* valueAt */ [](auto, auto mapIndex) -> int64_t { return mapIndex; },
          /* isNullAt */ [](auto) { return false; })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.flatMapColumns = {{"flatmap", {}}}, .maxFlatMapKeys = 4});

  NIMBLE_ASSERT_USER_THROW(
      writer.write(vector), "Too many flatmap keys for node");
}

// A map whose distinct key count is at/under maxFlatMapKeys writes fine.
TEST_F(WriterTest, flatMapKeyLimitWithinLimitSucceeds) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  // 5 distinct keys, well under the limit of 100.
  auto vector = vectorMaker.rowVector(
      {"flatmap"},
      {vectorMaker.mapVector<int64_t, int64_t>(
          /* size */
          10,
          /* sizeAt */ [](auto) { return 5; },
          /* keyAt */ [](auto, auto mapIndex) -> int64_t { return mapIndex; },
          /* valueAt */
          [](auto row, auto mapIndex) -> int64_t {
            return row * 10 + mapIndex;
          },
          /* isNullAt */ [](auto) { return false; })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.flatMapColumns = {{"flatmap", {}}}, .maxFlatMapKeys = 100});

  EXPECT_NO_THROW(writer.write(vector));
  EXPECT_NO_THROW(writer.close());
}

// The limit is file-wide: distinct keys accumulate across write() calls even
// when no single write exceeds it.
TEST_F(WriterTest, flatMapKeyLimitAccumulatesAcrossWrites) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  // Each batch has 8 distinct keys (< limit 10), but the two batches use
  // disjoint key ranges, so the file-wide count (16) exceeds the limit.
  auto makeBatch = [&](int64_t keyOffset) {
    return vectorMaker.rowVector(
        {"flatmap"},
        {vectorMaker.mapVector<int64_t, int64_t>(
            /* size */
            1,
            /* sizeAt */ [](auto) { return 8; },
            /* keyAt */
            [keyOffset](auto, auto mapIndex) -> int64_t {
              return keyOffset + mapIndex;
            },
            /* valueAt */
            [](auto, auto mapIndex) -> int64_t { return mapIndex; },
            /* isNullAt */ [](auto) { return false; })});
  };

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      makeBatch(0)->type(),
      std::move(writeFile),
      *rootPool_,
      {.flatMapColumns = {{"flatmap", {}}}, .maxFlatMapKeys = 10});

  // First batch: 8 distinct keys, under the limit.
  EXPECT_NO_THROW(writer.write(makeBatch(0)));
  // Second batch: 8 new keys push the file-wide total to 16, over the limit.
  NIMBLE_ASSERT_USER_THROW(
      writer.write(makeBatch(100)), "Too many flatmap keys for node");
}

// A maxFlatMapKeys of 0 disables the cap (unlimited keys).
TEST_F(WriterTest, flatMapKeyLimitZeroMeansUnlimited) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  // 50 distinct keys, which would exceed any small positive limit.
  auto vector = vectorMaker.rowVector(
      {"flatmap"},
      {vectorMaker.mapVector<int64_t, int64_t>(
          /* size */
          1,
          /* sizeAt */ [](auto) { return 50; },
          /* keyAt */ [](auto, auto mapIndex) -> int64_t { return mapIndex; },
          /* valueAt */ [](auto, auto mapIndex) -> int64_t { return mapIndex; },
          /* isNullAt */ [](auto) { return false; })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.flatMapColumns = {{"flatmap", {}}}, .maxFlatMapKeys = 0});

  EXPECT_NO_THROW(writer.write(vector));
  EXPECT_NO_THROW(writer.close());
}

TEST_F(WriterTest, featureReorderingStreamCollocation) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  const std::vector<int64_t> reorderedKeys = {4, 2, 0};
  const int numRows = 1'000;

  for (bool enableIndex : {false, true}) {
    SCOPED_TRACE(fmt::format("enableIndex={}", enableIndex));

    // With index: schema is (key, flatmap); without: just (flatmap).
    // The flatmap column ordinal differs accordingly.
    auto vector = enableIndex
        ? vectorMaker.rowVector(
              {"key", "flatmap"},
              {vectorMaker.flatVector<int64_t>(
                   numRows, [](auto row) { return static_cast<int64_t>(row); }),
               vectorMaker.mapVector<int32_t, int32_t>(
                   numRows,
                   [](auto) { return 5; },
                   [](auto, auto mapIndex) { return mapIndex; },
                   [](auto row, auto mapIndex) { return row * 10 + mapIndex; },
                   [](auto) { return false; })})
        : vectorMaker.rowVector(
              {"flatmap"},
              {vectorMaker.mapVector<int32_t, int32_t>(
                  numRows,
                  [](auto) { return 5; },
                  [](auto, auto mapIndex) { return mapIndex; },
                  [](auto row, auto mapIndex) { return row * 10 + mapIndex; },
                  [](auto) { return false; })});

    const size_t flatmapOrdinal = enableIndex ? 1 : 0;

    std::string file;
    {
      auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
      nimble::WriterOptions options;
      options.flatMapColumns = {{"flatmap", {}}};
      options.featureReordering =
          std::vector<std::tuple<size_t, std::vector<int64_t>>>{
              {flatmapOrdinal, reorderedKeys}};

      if (enableIndex) {
        options.clusterIndexConfig =
            nimble::index::ClusterIndexConfigBuilder{}
                .withKeyColumns({"key"})
                .withSortOrders({nimble::SortOrder{.ascending = true}})
                .withEnforceKeyOrder(true)
                .build();
      }

      nimble::Writer writer(
          vector->type(), std::move(writeFile), *rootPool_, std::move(options));
      writer.write(vector);
      writer.close();
    }

    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    ASSERT_GE(tablet->stripeCount(), 1);
    if (enableIndex) {
      ASSERT_NE(tablet->clusterIndex(), nullptr) << "Cluster index must exist";
    }

    auto stripeId = tablet->stripeIdentifier(0);
    const auto streamCount = tablet->streamCount(stripeId);
    std::vector<nimble::TabletReader::StreamLocation> streamLocations(
        streamCount);
    tablet->streamLocations(stripeId, streamLocations);

    nimble::BatchReader reader(readFile.get(), *leafPool_);
    const auto& flatMap =
        reader.schema()->asRow().childAt(flatmapOrdinal)->asFlatMap();

    std::unordered_map<std::string, uint32_t> keyToValueStreamId;
    for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
      keyToValueStreamId[flatMap.nameAt(i)] =
          flatMap.childAt(i)->asScalar().scalarDescriptor().offset();
    }

    // Use value stream offsets for ordering since inMap streams may be
    // constant-encoded and deduplicated when all keys are present in every
    // row.
    auto diskPosition = [&](const std::string& key) -> uint32_t {
      return streamLocations[keyToValueStreamId.at(key)].offset;
    };

    // Verify reordered keys appear in the specified order on disk.
    for (size_t i = 1; i < reorderedKeys.size(); ++i) {
      auto prevKey = folly::to<std::string>(reorderedKeys[i - 1]);
      auto currKey = folly::to<std::string>(reorderedKeys[i]);
      EXPECT_LT(diskPosition(prevKey), diskPosition(currKey))
          << "Key " << prevKey << " should appear before key " << currKey
          << " on disk";
    }

    // Verify reordered keys' value streams are contiguous (adjacent on disk).
    for (size_t i = 1; i < reorderedKeys.size(); ++i) {
      auto prevKey = folly::to<std::string>(reorderedKeys[i - 1]);
      auto currKey = folly::to<std::string>(reorderedKeys[i]);
      auto prevStreamId = keyToValueStreamId.at(prevKey);
      auto currStreamId = keyToValueStreamId.at(currKey);
      EXPECT_EQ(
          streamLocations[prevStreamId].offset +
              streamLocations[prevStreamId].size,
          streamLocations[currStreamId].offset)
          << "Key " << prevKey << " value stream should be adjacent to key "
          << currKey;
    }

    // Verify leftover keys (1, 3) appear after all reordered keys.
    auto lastReorderedKey = folly::to<std::string>(reorderedKeys.back());
    auto lastReorderedPos = diskPosition(lastReorderedKey);
    for (const auto& leftoverKey : {"1", "3"}) {
      EXPECT_GT(diskPosition(leftoverKey), lastReorderedPos)
          << "Leftover key " << leftoverKey
          << " should appear after last reordered key " << lastReorderedKey;
    }
  }
}

TEST_F(WriterTest, featureReorderingSharedDictionaryStreamCollocation) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  const std::vector<int64_t> reorderedKeys = {4, 2, 0};
  constexpr int kNumRows = 1'000;

  for (bool enableIndex : {false, true}) {
    SCOPED_TRACE(fmt::format("enableIndex={}", enableIndex));

    auto keyAt = [](auto, auto mapIndex) {
      return static_cast<int64_t>(mapIndex);
    };
    // Keep each configured feature's value stream physically distinct so
    // stream deduplication does not mask the layout being checked below.
    auto valueAt = [](auto row, auto mapIndex) {
      const auto cardinality = static_cast<int32_t>(mapIndex + 2);
      return static_cast<int32_t>(mapIndex * 100 + (row + 1) % cardinality);
    };
    auto vector = enableIndex
        ? vectorMaker.rowVector(
              {"key", "flatmap"},
              {vectorMaker.flatVector<int64_t>(
                   kNumRows,
                   [](auto row) { return static_cast<int64_t>(row); }),
               vectorMaker.mapVector<int64_t, int32_t>(
                   kNumRows,
                   [](auto) { return 5; },
                   keyAt,
                   valueAt,
                   [](auto) { return false; })})
        : vectorMaker.rowVector(
              {"flatmap"},
              {vectorMaker.mapVector<int64_t, int32_t>(
                  kNumRows,
                  [](auto) { return 5; },
                  keyAt,
                  valueAt,
                  [](auto) { return false; })});

    const size_t flatmapOrdinal = enableIndex ? 1 : 0;

    std::string file;
    {
      auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
      nimble::WriterOptions options;
      options.flatMapColumns = {{"flatmap", {}}};
      options.featureReordering =
          std::vector<std::tuple<size_t, std::vector<int64_t>>>{
              {flatmapOrdinal, reorderedKeys}};
      options.encodingSelectionPolicyCreator = [](nimble::DataType dataType) {
        return sharedDictionaryFeatureSelectionPolicy(dataType);
      };
      options.experimentalSharedDictionaryEncoding =
          nimble::SharedDictionaryEncodingConfig::builder(
              std::move(options.experimentalSharedDictionaryEncoding))
              .addFlatmapValueDictionary(
                  "flatmap",
                  4,
                  nimble::SharedDictionaryConfig{
                      .scope = nimble::SharedDictionaryScope::Stripe})
              .addFlatmapValueDictionary(
                  "flatmap",
                  2,
                  nimble::SharedDictionaryConfig{
                      .scope = nimble::SharedDictionaryScope::Stripe})
              .addFlatmapValueDictionary(
                  "flatmap",
                  0,
                  nimble::SharedDictionaryConfig{
                      .scope = nimble::SharedDictionaryScope::Stripe})
              .build();

      if (enableIndex) {
        options.clusterIndexConfig =
            nimble::index::ClusterIndexConfigBuilder{}
                .withKeyColumns({"key"})
                .withSortOrders({nimble::SortOrder{.ascending = true}})
                .withEnforceKeyOrder(true)
                .build();
      }

      nimble::Writer writer(
          vector->type(), std::move(writeFile), *rootPool_, std::move(options));
      writer.write(vector);
      writer.close();
    }

    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    {
      nimble::BatchReader reader(readFile.get(), *leafPool_);
      velox::VectorPtr result;
      ASSERT_TRUE(reader.next(kNumRows, result));
      ASSERT_EQ(result->size(), vector->size());
      for (velox::vector_size_t row = 0; row < vector->size(); ++row) {
        EXPECT_TRUE(vector->equalValueAt(result.get(), row, row))
            << "Content mismatch at row " << row;
      }
      ASSERT_FALSE(reader.next(1, result));
    }

    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    ASSERT_GE(tablet->stripeCount(), 1);
    if (enableIndex) {
      ASSERT_NE(tablet->clusterIndex(), nullptr) << "Cluster index must exist";
    }

    const auto stripeId = tablet->stripeIdentifier(0);
    const auto streamCount = tablet->streamCount(stripeId);
    std::vector<nimble::TabletReader::StreamLocation> streamLocations(
        streamCount);
    tablet->streamLocations(stripeId, streamLocations);

    nimble::BatchReader reader(readFile.get(), *leafPool_);
    const auto& flatMap =
        reader.schema()->asRow().childAt(flatmapOrdinal)->asFlatMap();

    std::unordered_map<std::string, uint32_t> keyToValueStreamId;
    for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
      keyToValueStreamId[flatMap.nameAt(i)] =
          flatMap.childAt(i)->asScalar().scalarDescriptor().offset();
    }

    struct FeatureStreamSpan {
      std::string key;
      uint32_t begin;
      uint32_t end;
    };

    std::vector<FeatureStreamSpan> featureStreamSpans;
    featureStreamSpans.reserve(reorderedKeys.size());
    for (const auto reorderedKey : reorderedKeys) {
      const auto key = folly::to<std::string>(reorderedKey);
      const auto valueStreamId = keyToValueStreamId.at(key);
      const auto dictionaryStreamId =
          tablet->stripeDictionaryStreamId(valueStreamId);
      ASSERT_TRUE(dictionaryStreamId.has_value())
          << "Missing shared dictionary stream for key " << key;

      ASSERT_LT(valueStreamId, streamLocations.size());
      ASSERT_LT(dictionaryStreamId.value(), streamLocations.size());
      EXPECT_GT(streamLocations[valueStreamId].size, 0);
      EXPECT_GT(streamLocations[dictionaryStreamId.value()].size, 0);

      auto streams =
          tablet->load(stripeId, std::vector<uint32_t>{valueStreamId});
      nimble::InMemoryChunkedStream chunkedStream{
          *leafPool_, std::move(streams[0])};
      ASSERT_TRUE(chunkedStream.hasNext());
      EXPECT_EQ(
          nimble::EncodingPrefix::encodingType(chunkedStream.nextChunk()),
          nimble::EncodingType::SharedDictionary)
          << "Key " << key << " value stream should use SharedDictionary";

      const auto valueBegin = streamLocations[valueStreamId].offset;
      const auto valueEnd = valueBegin + streamLocations[valueStreamId].size;
      const auto dictionaryBegin =
          streamLocations[dictionaryStreamId.value()].offset;
      const auto dictionaryEnd =
          dictionaryBegin + streamLocations[dictionaryStreamId.value()].size;
      EXPECT_TRUE(valueEnd == dictionaryBegin || dictionaryEnd == valueBegin)
          << "Key " << key
          << " dictionary stream should be adjacent to its value stream";

      featureStreamSpans.push_back({
          key,
          std::min(valueBegin, dictionaryBegin),
          std::max(valueEnd, dictionaryEnd),
      });
    }

    for (size_t i = 1; i < featureStreamSpans.size(); ++i) {
      EXPECT_EQ(featureStreamSpans[i - 1].end, featureStreamSpans[i].begin)
          << "Key " << featureStreamSpans[i - 1].key << " and key "
          << featureStreamSpans[i].key
          << " value/dictionary streams should be adjacent on disk";
    }
  }
}

TEST_F(
    WriterTest,
    featureReorderingUsesInputOrdinalWithOmittedClusterIndexKey) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  const std::vector<int64_t> reorderedKeys = {4, 2, 0};
  constexpr int kNumRows = 1'000;
  auto vector = vectorMaker.rowVector(
      {"key", "flatmap"},
      {
          vectorMaker.flatVector<int64_t>(
              kNumRows, [](auto row) { return static_cast<int64_t>(row); }),
          vectorMaker.mapVector<int32_t, int32_t>(
              kNumRows,
              [](auto) { return 5; },
              [](auto, auto mapIndex) { return mapIndex; },
              [](auto row, auto mapIndex) { return row * 10 + mapIndex; },
              [](auto) { return false; }),
      });

  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.flatMapColumns = {{"flatmap", {}}};
    options.clusterIndexConfig =
        nimble::index::ClusterIndexConfigBuilder{}
            .withKeyColumns({"key"})
            .withSortOrders({nimble::SortOrder{.ascending = true}})
            .withEnforceKeyOrder(true)
            .build();
    options.experimentalOmitClusterIndexKeyColumnStorage = true;
    // The option is expressed against the input schema where key=0 and
    // flatmap=1. The stored schema omits key, so the writer must remap this to
    // stored ordinal 0 before handing it to the layout planner.
    options.featureReordering =
        std::vector<std::tuple<size_t, std::vector<int64_t>>>{
            {1, reorderedKeys}};

    nimble::Writer writer(
        vector->type(), std::move(writeFile), *rootPool_, std::move(options));
    writer.write(vector);
    writer.close();
  }

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = nimble::TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  ASSERT_GE(tablet->stripeCount(), 1);
  ASSERT_NE(tablet->clusterIndex(), nullptr);

  auto stripeId = tablet->stripeIdentifier(0);
  const auto streamCount = tablet->streamCount(stripeId);
  std::vector<nimble::TabletReader::StreamLocation> streamLocations(
      streamCount);
  tablet->streamLocations(stripeId, streamLocations);

  nimble::BatchReader reader(readFile.get(), *leafPool_);
  const auto& rowSchema = reader.schema()->asRow();
  ASSERT_EQ(1, rowSchema.childrenCount());
  EXPECT_EQ("flatmap", rowSchema.nameAt(0));
  const auto& flatMap = rowSchema.childAt(0)->asFlatMap();

  std::unordered_map<std::string, uint32_t> keyToValueStreamId;
  for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
    keyToValueStreamId[flatMap.nameAt(i)] =
        flatMap.childAt(i)->asScalar().scalarDescriptor().offset();
  }

  for (size_t i = 1; i < reorderedKeys.size(); ++i) {
    const auto prevKey = folly::to<std::string>(reorderedKeys[i - 1]);
    const auto currKey = folly::to<std::string>(reorderedKeys[i]);
    const auto prevStreamId = keyToValueStreamId.at(prevKey);
    const auto currStreamId = keyToValueStreamId.at(currKey);
    EXPECT_EQ(
        streamLocations[prevStreamId].offset +
            streamLocations[prevStreamId].size,
        streamLocations[currStreamId].offset)
        << "Key " << prevKey << " value stream should be adjacent to key "
        << currKey;
  }
}

TEST_F(WriterTest, encodingLayoutTreeWithOmittedClusterIndexKey) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {"key", "flatmap"},
      {
          vectorMaker.flatVector<int64_t>({1}),
          vectorMaker.mapVector<int32_t, int32_t>(
              1,
              [](auto) { return 1; },
              [](auto, auto) { return 1; },
              [](auto, auto) { return 10; },
              [](auto) { return false; }),
      });

  auto makeOptions = [](nimble::EncodingLayoutTree encodingLayoutTree) {
    nimble::WriterOptions options;
    options.flatMapColumns = {{"flatmap", {}}};
    options.clusterIndexConfig =
        nimble::index::ClusterIndexConfigBuilder{}
            .withKeyColumns({"key"})
            .withSortOrders({nimble::SortOrder{.ascending = true}})
            .withEnforceKeyOrder(true)
            .build();
    options.experimentalOmitClusterIndexKeyColumnStorage = true;
    options.encodingLayoutTree.emplace(std::move(encodingLayoutTree));
    return options;
  };

  std::string file;
  {
    nimble::Writer writer(
        vector->type(),
        std::make_unique<velox::InMemoryWriteFile>(&file),
        *rootPool_,
        makeOptions(
            nimble::EncodingLayoutTree{
                nimble::Kind::Row,
                {},
                "",
                {
                    {nimble::Kind::Scalar, {}, "key"},
                    {nimble::Kind::FlatMap, {}, "flatmap"},
                }}));
    writer.write(vector);
    writer.close();
  }

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  nimble::BatchReader reader(readFile.get(), *leafPool_);
  const auto& schema = reader.schema()->asRow();
  ASSERT_EQ(schema.childrenCount(), 1);
  EXPECT_EQ(schema.nameAt(0), "flatmap");

  std::string invalidFile;
  NIMBLE_ASSERT_THROW(
      nimble::Writer(
          vector->type(),
          std::make_unique<velox::InMemoryWriteFile>(&invalidFile),
          *rootPool_,
          makeOptions(
              nimble::EncodingLayoutTree{
                  nimble::Kind::Row,
                  {},
                  "",
                  {
                      {nimble::Kind::Scalar, {}, "key"},
                      {nimble::Kind::Scalar, {}, "flatmap"},
                  }})),
      "Incompatible encoding layout node. Expecting flatmap node.");
}

TEST_F(WriterTest, duplicateFlatmapKey) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  // Vector with constant but duplicate key set. Potentially omitting in map
  // stream in the future.
  {
    auto vec = vectorMaker.rowVector(
        {"flatmap"},
        {vectorMaker.mapVector<int32_t, int32_t>(
            10,
            /* sizeAt */ [](auto row) { return 6; },
            /* keyAt */
            [](auto /* row */, auto mapIndex) { return mapIndex / 2; },
            /* valueAt */ [](auto row, auto /* mapIndex */) { return row; },
            /* isNullAt */ [](auto /* row */) { return false; })});
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

    nimble::Writer writer(
        vec->type(),
        std::move(writeFile),
        *rootPool_,
        {.flatMapColumns = {{"flatmap", {}}}});
    EXPECT_THROW(writer.write(vec), nimble::NimbleInternalError);
    EXPECT_ANY_THROW(writer.close());
  }
  // Vector with a rotating duplicate key set. The more typical layout
  // requiring in map stream to represent.
  {
    auto vec = vectorMaker.rowVector(
        {"flatmap"},
        {vectorMaker.mapVector<int32_t, int32_t>(
            10,
            /* sizeAt */ [](auto row) { return 6; },
            /* keyAt */
            [](auto row, auto mapIndex) { return (row + mapIndex / 2) % 6; },
            /* valueAt */ [](auto row, auto /* mapIndex */) { return row; },
            /* isNullAt */ [](auto /* row */) { return false; })});

    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

    nimble::Writer writer(
        vec->type(),
        std::move(writeFile),
        *rootPool_,
        {.flatMapColumns = {{"flatmap", {}}}});
    EXPECT_THROW(writer.write(vec), nimble::NimbleInternalError);
    EXPECT_ANY_THROW(writer.close());
  }
}

struct StripeRawSizeFlushPolicyTestCase {
  const size_t batchCount;
  const uint32_t rawStripeSize;
  const uint32_t stripeCount;
};

class StripeRawSizeFlushPolicyTest
    : public WriterTest,
      public ::testing::WithParamInterface<StripeRawSizeFlushPolicyTestCase> {};

TEST_P(StripeRawSizeFlushPolicyTest, stripeRawSizeFlushPolicy) {
  auto type = velox::ROW({{"simple", velox::INTEGER()}});
  nimble::WriterOptions writerOptions{.flushPolicyFactory = []() {
    // Buffering 256MB data before encoding stripes.
    return std::make_unique<nimble::StripeRawSizeFlushPolicy>(
        GetParam().rawStripeSize);
  }};

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(writerOptions));
  auto batches =
      generateBatches(type, GetParam().batchCount, 4000, 20221110, *leafPool_);

  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();

  velox::InMemoryReadFile readFile(file);
  auto selector = std::make_shared<velox::dwio::common::ColumnSelector>(type);
  nimble::BatchReader reader(&readFile, *leafPool_, std::move(selector));

  EXPECT_EQ(GetParam().stripeCount, reader.tabletReader().stripeCount());
}

namespace {
class MockReclaimer : public velox::memory::MemoryReclaimer {
 public:
  explicit MockReclaimer() : velox::memory::MemoryReclaimer(0) {}
  void setEnterArbitrationFunc(std::function<void()>&& func) {
    enterArbitrationFunc_ = func;
  }
  void enterArbitration() override {
    if (enterArbitrationFunc_) {
      enterArbitrationFunc_();
    }
  }

 private:
  std::function<void()> enterArbitrationFunc_;
};
} // namespace

TEST_F(WriterTest, memoryReclaimPath) {
  auto rootPool = velox::memory::memoryManager()->addRootPool(
      "root", 4L << 20, velox::memory::MemoryReclaimer::create());
  auto writerPool = rootPool->addAggregateChild(
      "writer", velox::memory::MemoryReclaimer::create());

  auto type = velox::ROW(
      {{"simple_int", velox::INTEGER()}, {"simple_double", velox::DOUBLE()}});
  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  std::atomic_bool reclaimEntered = false;
  nimble::WriterOptions writerOptions{.reclaimerFactory = [&]() {
    auto reclaimer = std::make_unique<MockReclaimer>();
    reclaimer->setEnterArbitrationFunc([&]() { reclaimEntered = true; });
    return reclaimer;
  }};
  nimble::Writer writer(
      type, std::move(writeFile), *writerPool, std::move(writerOptions));
  auto batches = generateBatches(type, 100, 4000, 20221110, *leafPool_);

  EXPECT_THROW(
      {
        for (const auto& batch : batches) {
          writer.write(batch);
        }
      },
      velox::VeloxException);
  ASSERT_TRUE(reclaimEntered.load());
}

namespace {
// A spill config is what makes the writer reclaimable at all, so the tests
// below set one rather than short-circuiting on the null check.
velox::common::SpillConfig makeReclaimSpillConfig() {
  velox::common::SpillConfig spillConfig;
  spillConfig.writerFlushThresholdSize = 1 << 20;
  return spillConfig;
}

// Mirrors what SharedArbitrator does when it walks a task's pools: ask every
// child that carries a reclaimer how much it could free. Returns the number of
// reclaimers queried.
int32_t queryChildReclaimers(velox::memory::MemoryPool& parent) {
  int32_t queried = 0;
  parent.visitChildren([&](velox::memory::MemoryPool* child) {
    auto* reclaimer = child->reclaimer();
    if (reclaimer != nullptr) {
      uint64_t reclaimableBytes{1};
      EXPECT_FALSE(reclaimer->reclaimableBytes(*child, reclaimableBytes));
      EXPECT_EQ(reclaimableBytes, 0);
      ++queried;
    }
    return true;
  });
  return queried;
}
} // namespace

// The writer installs its reclaimer while `pool_` is initialized, several
// members before the state that reclaimer reads. Arbitration triggered by any
// allocation in between -- the writer's own context allocates a 1MB Buffer, and
// a concurrent driver can arbitrate at any moment -- reaches the reclaimer
// first. Drives that through the reclaimer factory, which the writer invokes
// for `encodingMemoryPool_` and for the context: after the reclaimer is live on
// the pool, before the writer is built.
TEST_F(WriterTest, reclaimableBytesDuringConstruction) {
  auto rootPool = velox::memory::memoryManager()->addRootPool(
      "reclaimDuringConstruction",
      64L << 20,
      velox::memory::MemoryReclaimer::create());
  auto writerPool = rootPool->addAggregateChild(
      "writer", velox::memory::MemoryReclaimer::create());

  auto type = velox::ROW({{"simple_int", velox::INTEGER()}});
  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  const auto spillConfig = makeReclaimSpillConfig();
  int32_t queriedDuringConstruction = 0;
  nimble::WriterOptions writerOptions{
      .reclaimerFactory =
          [&]() -> std::unique_ptr<velox::memory::MemoryReclaimer> {
        queriedDuringConstruction += queryChildReclaimers(*writerPool);
        return nullptr;
      },
      .spillConfig = &spillConfig};

  nimble::Writer writer(
      type, std::move(writeFile), *writerPool, std::move(writerOptions));

  // The first factory call happens before the pool exists; the later ones see
  // it, and are the ones that matter.
  EXPECT_GT(queriedDuringConstruction, 0);
  writer.close();
}

// A closed writer has no stripe left to flush, so it must stop advertising its
// reserved bytes as reclaimable -- otherwise the arbitrator picks the pool as a
// candidate, suspends the driver, and reclaimBytes() turns it away. Also covers
// the teardown ordering: `context_` is destroyed ahead of `pool_`, so anything
// the reclaimer reads has to outlive the writer's members.
TEST_F(WriterTest, reclaimableBytesAfterClose) {
  auto rootPool = velox::memory::memoryManager()->addRootPool(
      "reclaimAfterClose", 64L << 20, velox::memory::MemoryReclaimer::create());
  auto writerPool = rootPool->addAggregateChild(
      "writer", velox::memory::MemoryReclaimer::create());

  auto type = velox::ROW({{"simple_int", velox::INTEGER()}});
  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  const auto spillConfig = makeReclaimSpillConfig();
  nimble::Writer writer(
      type,
      std::move(writeFile),
      *writerPool,
      nimble::WriterOptions{.spillConfig = &spillConfig});
  for (const auto& batch :
       generateBatches(type, 2, 100, 20221110, *leafPool_)) {
    writer.write(batch);
  }
  writer.close();

  EXPECT_GT(queryChildReclaimers(*writerPool), 0);
}

TEST_F(WriterTest, flushHugeStrings) {
  nimble::WriterOptions writerOptions{.flushPolicyFactory = []() {
    return std::make_unique<nimble::StripeRawSizeFlushPolicy>(1 * 1024 * 1024);
  }};

  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  // Each vector contains 99 strings with 36 characters each (36*99=3564) +
  // 100 bytes for null vector + 99 string_views (99*16=1584) for a total of
  // 5248 bytes, so writing 200 batches should exceed the flush theshold of
  // 1MB
  auto vector = vectorMaker.rowVector(
      {"string"},
      {
          vectorMaker.flatVector<std::string>(
              100,
              [](auto /* row */) {
                return std::string("abcdefghijklmnopqrstuvwxyz0123456789");
              },
              [](auto row) { return row == 6; }),
      });

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      std::move(writerOptions));

  // Writing 500 batches should produce 3 stripes, as each 200 vectors will
  // exceed the flush threshold.
  for (auto i = 0; i < 500; ++i) {
    writer.write(vector);
  }
  writer.close();

  velox::InMemoryReadFile readFile(file);
  auto selector = std::make_shared<velox::dwio::common::ColumnSelector>(
      std::dynamic_pointer_cast<const velox::RowType>(vector->type()));
  nimble::BatchReader reader(&readFile, *leafPool_, std::move(selector));

  EXPECT_EQ(3, reader.tabletReader().stripeCount());
}

TEST_F(WriterTest, encodingLayout) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {},
      "",
      {
          {nimble::Kind::Map,
           {
               {
                   0,
                   nimble::EncodingLayout{
                       nimble::EncodingType::Dictionary,
                       {},
                       nimble::CompressionType::Uncompressed,
                       {
                           nimble::EncodingLayout{
                               nimble::EncodingType::FixedBitWidth,
                               {},
                               nimble::CompressionType::MetaInternal},
                           std::nullopt,
                       }},
               },
           },
           "",
           {
               // Map keys
               {nimble::Kind::Scalar, {}, ""},
               // Map Values
               {nimble::Kind::Scalar,
                {
                    {
                        0,
                        nimble::EncodingLayout{
                            nimble::EncodingType::MainlyConstant,
                            {},
                            nimble::CompressionType::Uncompressed,
                            {
                                std::nullopt,
                                nimble::EncodingLayout{
                                    nimble::EncodingType::Trivial,
                                    {},
                                    nimble::CompressionType::MetaInternal},
                            }},
                    },
                },
                ""},
           }},
          {nimble::Kind::FlatMap,
           {},
           "",
           {
               {
                   nimble::Kind::Scalar,
                   {
                       {
                           0,
                           nimble::EncodingLayout{
                               nimble::EncodingType::MainlyConstant,
                               {},
                               nimble::CompressionType::Uncompressed,
                               {
                                   nimble::EncodingLayout{
                                       nimble::EncodingType::Trivial,
                                       {},
                                       nimble::CompressionType::Uncompressed},
                                   nimble::EncodingLayout{
                                       nimble::EncodingType::FixedBitWidth,
                                       {},
                                       nimble::CompressionType::Uncompressed},
                               }},
                       },
                   },
                   "1",
               },
               {
                   nimble::Kind::Scalar,
                   {
                       {
                           0,
                           nimble::EncodingLayout{
                               nimble::EncodingType::Constant,
                               {},
                               nimble::CompressionType::Uncompressed,
                           },
                       },
                   },
                   "2",
               },
           }},
      }};

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {"map", "flatmap"},
      {vectorMaker.mapVector<int32_t, int32_t>(
           5,
           /* sizeAt */ [](auto row) { return row % 3; },
           /* keyAt */
           [](auto /* row */, auto mapIndex) { return mapIndex; },
           /* valueAt */ [](auto row, auto /* mapIndex */) { return row; },
           /* isNullAt */ [](auto /* row */) { return false; }),
       vectorMaker.mapVector(
           std::vector<std::optional<
               std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
               std::vector<std::pair<int32_t, std::optional<int32_t>>>{
                   {0, 2},
                   {2, 3},
               },
               std::nullopt,
               {},
               std::vector<std::pair<int32_t, std::optional<int32_t>>>{
                   {1, 4},
                   {0, std::nullopt},
               },
               std::vector<std::pair<int32_t, std::optional<int32_t>>>{
                   {1, std::nullopt},
               },
           })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {
          .flatMapColumns = {{"flatmap", {}}},
          .encodingLayoutTree = std::move(expected),
          // Boosting acceptance ratio by 100x to make sure it is always
          // accepted (even if compressed size if bigger than uncompressed
          // size)
          .compressionOptions =
              {.compressionAcceptRatio = 100, .internalMinCompressionSize = 0},
      });

  writer.write(vector);
  writer.close();

  for (auto useChainedBuffers : {false, true}) {
    auto readFile =
        std::make_shared<nimble::testing::InMemoryTrackableReadFile>(
            file, useChainedBuffers);
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    auto section =
        tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
    NIMBLE_CHECK(section.has_value(), "Schema not found.");
    auto schema =
        nimble::SchemaDeserializer::deserialize(section->content().data());
    auto& mapNode = schema->asRow().childAt(0)->asMap();
    auto& mapValuesNode = mapNode.values()->asScalar();
    auto& flatMapNode = schema->asRow().childAt(1)->asFlatMap();
    ASSERT_EQ(3, flatMapNode.childrenCount());

    auto findChild =
        [](const facebook::nimble::FlatMapType& map,
           std::string_view key) -> std::shared_ptr<const nimble::Type> {
      for (auto i = 0; i < map.childrenCount(); ++i) {
        if (map.nameAt(i) == key) {
          return map.childAt(i);
        }
      }
      return nullptr;
    };
    const auto& flatMapKey1Node = findChild(flatMapNode, "1")->asScalar();
    const auto& flatMapKey2Node = findChild(flatMapNode, "2")->asScalar();

    for (auto i = 0; i < tablet->stripeCount(); ++i) {
      auto stripeIdentifier = tablet->stripeIdentifier(i);
      std::vector<uint32_t> identifiers{
          mapNode.lengthsDescriptor().offset(),
          mapValuesNode.scalarDescriptor().offset(),
          flatMapKey1Node.scalarDescriptor().offset(),
          flatMapKey2Node.scalarDescriptor().offset()};
      auto streams = tablet->load(stripeIdentifier, identifiers);
      {
        nimble::InMemoryChunkedStream chunkedStream{
            *leafPool_, std::move(streams[0])};
        ASSERT_TRUE(chunkedStream.hasNext());
        // Verify Map stream
        auto capture = nimble::EncodingLayoutCapture::capture(
            chunkedStream.nextChunk(), nimble::Encoding::Options{});
        EXPECT_EQ(nimble::EncodingType::Dictionary, capture.encodingType());
        EXPECT_EQ(
            nimble::EncodingType::FixedBitWidth,
            capture.child(nimble::EncodingIdentifiers::Dictionary::Alphabet)
                ->encodingType());
        expectMetaInternalCompression(
            capture.child(nimble::EncodingIdentifiers::Dictionary::Alphabet)
                ->compressionType());
      }

      {
        nimble::InMemoryChunkedStream chunkedStream{
            *leafPool_, std::move(streams[1])};
        ASSERT_TRUE(chunkedStream.hasNext());
        // Verify Map Values stream
        auto capture = nimble::EncodingLayoutCapture::capture(
            chunkedStream.nextChunk(), nimble::Encoding::Options{});
        EXPECT_EQ(nimble::EncodingType::MainlyConstant, capture.encodingType());
        EXPECT_EQ(
            nimble::EncodingType::Trivial,
            capture
                .child(nimble::EncodingIdentifiers::MainlyConstant::OtherValues)
                ->encodingType());
        expectMetaInternalCompression(
            capture
                .child(nimble::EncodingIdentifiers::MainlyConstant::OtherValues)
                ->compressionType());
      }

      {
        nimble::InMemoryChunkedStream chunkedStream{
            *leafPool_, std::move(streams[2])};
        ASSERT_TRUE(chunkedStream.hasNext());
        // Verify FlatMap Kay "1" stream
        auto capture = nimble::EncodingLayoutCapture::capture(
            chunkedStream.nextChunk(), nimble::Encoding::Options{});
        EXPECT_EQ(nimble::EncodingType::MainlyConstant, capture.encodingType());
        EXPECT_EQ(
            nimble::EncodingType::Trivial,
            capture
                .child(nimble::EncodingIdentifiers::MainlyConstant::IsCommon)
                ->encodingType());
        EXPECT_EQ(
            nimble::CompressionType::Uncompressed,
            capture
                .child(nimble::EncodingIdentifiers::MainlyConstant::IsCommon)
                ->compressionType());
        EXPECT_EQ(
            nimble::EncodingType::FixedBitWidth,
            capture
                .child(nimble::EncodingIdentifiers::MainlyConstant::OtherValues)
                ->encodingType());
        EXPECT_EQ(
            nimble::CompressionType::Uncompressed,
            capture
                .child(nimble::EncodingIdentifiers::MainlyConstant::OtherValues)
                ->compressionType());
      }

      {
        nimble::InMemoryChunkedStream chunkedStream{
            *leafPool_, std::move(streams[3])};
        ASSERT_TRUE(chunkedStream.hasNext());
        // Verify FlatMap Kay "2" stream
        auto capture = nimble::EncodingLayoutCapture::capture(
            chunkedStream.nextChunk(), nimble::Encoding::Options{});
        EXPECT_EQ(nimble::EncodingType::Constant, capture.encodingType());
      }
    }
  }
}

TEST_F(WriterTest, openZLCompressionNumericRoundTrip) {
  // E2E: force the OpenZL codec, write compressible numeric columns, and assert
  // (a) at least one numeric stream is actually OpenZL-compressed and (b) the
  // data round-trips byte-for-byte through the reader.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  constexpr velox::vector_size_t kRowCount = 1000;
  auto vector = vectorMaker.rowVector(
      {"i32", "i64", "f32"},
      {
          vectorMaker.flatVector<int32_t>(
              kRowCount,
              [](auto row) { return static_cast<int32_t>(1000 + row % 50); }),
          vectorMaker.flatVector<int64_t>(
              kRowCount,
              [](auto row) { return static_cast<int64_t>(row / 4); }),
          vectorMaker.flatVector<float>(
              kRowCount, [](auto row) { return static_cast<float>(row % 16); }),
      });

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::WriterOptions writerOptions;
  // The default write path drives compression off the encoding selection policy
  // (not WriterOptions::compressionOptions), so force OpenZL via the
  // factory. Accept ratio 100 + min size 0 ensure the codec is actually applied
  // even to tiny streams.
  writerOptions.encodingSelectionPolicyCreator =
      [factory =
           nimble::ManualEncodingSelectionPolicyFactory{
               nimble::ManualEncodingSelectionPolicyFactory::
                   defaultEncodingReadFactors(),
               nimble::CompressionOptions{
                   .compressionAcceptRatio = 100,
                   .compressionType = nimble::CompressionType::OpenZL,
                   .openzlMinCompressionSize = 0,
               }}](nimble::DataType dataType)
      -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
    return factory.createPolicy(dataType);
  };
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      std::move(writerOptions));
  writer.write(vector);
  writer.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = nimble::TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  auto section =
      tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
  ASSERT_TRUE(section.has_value());
  auto schema =
      nimble::SchemaDeserializer::deserialize(section->content().data());

  // Recursively checks whether any node in the encoding tree was compressed
  // with the given codec, so the assertion does not depend on which encoding
  // the writer happened to pick.
  std::function<bool(const nimble::EncodingLayout&, nimble::CompressionType)>
      usesCompression = [&](const nimble::EncodingLayout& layout,
                            nimble::CompressionType compressionType) {
        if (layout.compressionType() == compressionType) {
          return true;
        }
        for (uint8_t i = 0; i < layout.childrenCount(); ++i) {
          const auto& child = layout.child(i);
          if (child.has_value() &&
              usesCompression(child.value(), compressionType)) {
            return true;
          }
        }
        return false;
      };

  bool anyOpenZL = false;
  for (auto stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
    auto stripeIdentifier = tablet->stripeIdentifier(stripe);
    for (auto column = 0; column < schema->asRow().childrenCount(); ++column) {
      auto offset = schema->asRow()
                        .childAt(column)
                        ->asScalar()
                        .scalarDescriptor()
                        .offset();
      auto streams =
          tablet->load(stripeIdentifier, std::vector<uint32_t>{offset});
      if (streams.empty() || streams[0] == nullptr) {
        continue;
      }
      nimble::InMemoryChunkedStream chunkedStream{
          *leafPool_, std::move(streams[0])};
      while (chunkedStream.hasNext()) {
        auto capture = nimble::EncodingLayoutCapture::capture(
            chunkedStream.nextChunk(), nimble::Encoding::Options{});
        if (usesCompression(capture, nimble::CompressionType::OpenZL)) {
          anyOpenZL = true;
        }
      }
    }
  }
  EXPECT_TRUE(anyOpenZL)
      << "Expected at least one numeric stream to be OpenZL-compressed";

  nimble::BatchReader reader(readFile.get(), *leafPool_);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(kRowCount, result->size());
  for (velox::vector_size_t i = 0; i < kRowCount; ++i) {
    EXPECT_TRUE(vector->equalValueAt(result.get(), i, i))
        << "Content mismatch at row " << i;
  }
  ASSERT_FALSE(reader.next(1, result));
}

TEST_F(WriterTest, rejectsReadOnlyEncodingLayout) {
  const auto makeOptions = [](nimble::EncodingType encodingType,
                              uint32_t numChildren) {
    nimble::WriterOptions options;
    options.encodingLayoutTree.emplace(
        nimble::Kind::Row,
        std::unordered_map<
            nimble::EncodingLayoutTree::StreamIdentifier,
            nimble::EncodingLayout>{},
        "",
        std::vector<nimble::EncodingLayoutTree>{nimble::EncodingLayoutTree{
            nimble::Kind::Scalar,
            {{nimble::EncodingLayoutTree::StreamIdentifiers::Scalar::
                  ScalarStream,
              nimble::EncodingLayout{
                  encodingType,
                  {},
                  nimble::CompressionType::Uncompressed,
                  std::vector<std::optional<const nimble::EncodingLayout>>(
                      numChildren)}}},
            "c0"}});
    return options;
  };

  std::string file;
  NIMBLE_ASSERT_THROW(
      nimble::Writer(
          velox::ROW({{"c0", velox::BIGINT()}}),
          std::make_unique<velox::InMemoryWriteFile>(&file),
          *rootPool_,
          makeOptions(nimble::EncodingType::FOR, 3)),
      "Encoding is read-only and cannot be used for new writes: FOR");

  std::string pforFile;
  EXPECT_NO_THROW(
      nimble::Writer(
          velox::ROW({{"c0", velox::BIGINT()}}),
          std::make_unique<velox::InMemoryWriteFile>(&pforFile),
          *rootPool_,
          makeOptions(nimble::EncodingType::PFOR, 2)));
}

TEST_F(WriterTest, encodingLayoutSchemaMismatch) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {},
      "",
      {
          {
              nimble::Kind::Scalar,
              {
                  {
                      0,
                      nimble::EncodingLayout{
                          nimble::EncodingType::Dictionary,
                          {},
                          nimble::CompressionType::Uncompressed,
                          {
                              nimble::EncodingLayout{
                                  nimble::EncodingType::FixedBitWidth,
                                  {},
                                  nimble::CompressionType::MetaInternal},
                              std::nullopt,
                          }},
                  },
              },
              "",
          },
      }};

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {"map"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              5,
              /* sizeAt */ [](auto row) { return row % 3; },
              /* keyAt */
              [](auto /* row */, auto mapIndex) { return mapIndex; },
              /* valueAt */ [](auto row, auto /* mapIndex */) { return row; },
              /* isNullAt */ [](auto /* row */) { return false; }),
      });

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  try {
    nimble::Writer writer(
        vector->type(),
        std::move(writeFile),
        *rootPool_,
        {
            .encodingLayoutTree = std::move(expected),
            .compressionOptions = {.compressionAcceptRatio = 100},
        });
    FAIL() << "Writer should fail on incompatible encoding layout node";
  } catch (const nimble::NimbleInternalError& e) {
    EXPECT_NE(
        std::string(e.what()).find(
            "Incompatible encoding layout node. Expecting map node"),
        std::string::npos);
  }
}

TEST_F(WriterTest, encodingLayoutSchemaEvolutionMapToFlatmap) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {},
      "",
      {
          {nimble::Kind::Map,
           {
               {
                   0,
                   nimble::EncodingLayout{
                       nimble::EncodingType::Dictionary,
                       {},
                       nimble::CompressionType::Uncompressed,
                       {
                           nimble::EncodingLayout{
                               nimble::EncodingType::FixedBitWidth,
                               {},
                               nimble::CompressionType::MetaInternal},
                           std::nullopt,
                       }},
               },
           },
           "",
           {
               // Map keys
               {nimble::Kind::Scalar, {}, ""},
               // Map Values
               {nimble::Kind::Scalar,
                {
                    {
                        0,
                        nimble::EncodingLayout{
                            nimble::EncodingType::MainlyConstant,
                            {},
                            nimble::CompressionType::Uncompressed,
                            {
                                std::nullopt,
                                nimble::EncodingLayout{
                                    nimble::EncodingType::Trivial,
                                    {},
                                    nimble::CompressionType::MetaInternal},
                            }},
                    },
                },
                ""},
           }},
      }};

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {"map"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              5,
              /* sizeAt */ [](auto row) { return row % 3; },
              /* keyAt */
              [](auto /* row */, auto mapIndex) { return mapIndex; },
              /* valueAt */ [](auto row, auto /* mapIndex */) { return row; },
              /* isNullAt */ [](auto /* row */) { return false; }),
      });

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {
          .flatMapColumns = {{"map", {}}},
          .encodingLayoutTree = std::move(expected),
          .compressionOptions = {.compressionAcceptRatio = 100},
      });

  writer.write(vector);
  writer.close();

  // Getting here is good enough for now (as it means we didn't fail on node
  // type mismatch). Once we add metric collection, we can use these to verify
  // that no captured encoding was used.
}

TEST_F(WriterTest, encodingLayoutSchemaEvolutionFlamapToMap) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {},
      "",
      {
          {nimble::Kind::FlatMap,
           {},
           "",
           {
               {
                   nimble::Kind::Scalar,
                   {
                       {
                           0,
                           nimble::EncodingLayout{
                               nimble::EncodingType::MainlyConstant,
                               {},
                               nimble::CompressionType::Uncompressed,
                               {
                                   nimble::EncodingLayout{
                                       nimble::EncodingType::Trivial,
                                       {},
                                       nimble::CompressionType::Uncompressed},
                                   nimble::EncodingLayout{
                                       nimble::EncodingType::FixedBitWidth,
                                       {},
                                       nimble::CompressionType::Uncompressed},
                               }},
                       },
                   },
                   "1",
               },
               {
                   nimble::Kind::Scalar,
                   {
                       {
                           0,
                           nimble::EncodingLayout{
                               nimble::EncodingType::Constant,
                               {},
                               nimble::CompressionType::Uncompressed,
                           },
                       },
                   },
                   "2",
               },
           }},
      }};

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vector = vectorMaker.rowVector(
      {"flatmap"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              5,
              /* sizeAt */ [](auto row) { return row % 3; },
              /* keyAt */
              [](auto /* row */, auto mapIndex) { return mapIndex; },
              /* valueAt */ [](auto row, auto /* mapIndex */) { return row; },
              /* isNullAt */ [](auto /* row */) { return false; }),
      });

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {
          .encodingLayoutTree = std::move(expected),
          .compressionOptions = {.compressionAcceptRatio = 100},
      });

  writer.write(vector);
  writer.close();

  // Getting here is good enough for now (as it means we didn't fail on node
  // type mismatch). Once we add metric collection, we can use these to verify
  // that no captured encoding was used.
}

TEST_F(WriterTest, encodingLayoutSchemaEvolutionExpandingRow) {
  nimble::EncodingLayoutTree expected{
      nimble::Kind::Row,
      {},
      "",
      {
          {nimble::Kind::Row,
           {
               {
                   0,
                   nimble::EncodingLayout{
                       nimble::EncodingType::Trivial,
                       {},
                       nimble::CompressionType::Uncompressed},
               },
           },
           "",
           {
               {
                   nimble::Kind::Scalar,
                   {
                       {
                           0,
                           nimble::EncodingLayout{
                               nimble::EncodingType::Trivial,
                               {},
                               nimble::CompressionType::Uncompressed},
                       },
                   },
                   "",
               },
           }},
      }};

  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  // We are adding new top level column and also nested column
  auto vector = vectorMaker.rowVector(
      {"row1", "row2"},
      {
          vectorMaker.rowVector({
              vectorMaker.flatVector<int32_t>({1, 2, 3, 4, 5}),
              vectorMaker.flatVector<int32_t>({1, 2, 3, 4, 5}),
          }),
          vectorMaker.rowVector({
              vectorMaker.flatVector<int32_t>({1, 2, 3, 4, 5}),
              vectorMaker.flatVector<int32_t>({1, 2, 3, 4, 5}),
          }),
      });

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {
          .encodingLayoutTree = std::move(expected),
          .compressionOptions = {.compressionAcceptRatio = 100},
      });

  writer.write(vector);
  writer.close();

  // Getting here is good enough for now (as it means we didn't fail on node
  // type mismatch). Once we add metric collection, we can use these to verify
  // that no captured encoding was used.
}

TEST_F(WriterTest, combineMultipleLayersOfDictionaries) {
  using namespace facebook::velox;
  test::VectorMaker vectorMaker{leafPool_.get()};
  auto wrapInDictionary = [&](const std::vector<vector_size_t>& indices,
                              const VectorPtr& values) {
    auto buf =
        AlignedBuffer::allocate<vector_size_t>(indices.size(), leafPool_.get());
    memcpy(
        buf->asMutable<vector_size_t>(),
        indices.data(),
        sizeof(vector_size_t) * indices.size());
    return BaseVector::wrapInDictionary(nullptr, buf, indices.size(), values);
  };
  auto vector = vectorMaker.rowVector({
      wrapInDictionary(
          {0, 0, 1, 1},
          vectorMaker.rowVector({
              wrapInDictionary(
                  {0, 0}, vectorMaker.arrayVector<int64_t>({{1, 2, 3}})),
          })),
  });
  nimble::WriterOptions options;
  options.flatMapColumns = {{"c0", {}}};
  options.dictionaryArrayColumns = {"c0"};
  std::string file;
  auto writeFile = std::make_unique<InMemoryWriteFile>(&file);
  nimble::Writer writer(
      ROW({"c0"}, {MAP(VARCHAR(), ARRAY(BIGINT()))}),
      std::move(writeFile),
      *rootPool_,
      std::move(options));
  writer.write(vector);
  writer.close();
  InMemoryReadFile readFile(file);
  nimble::BatchReadParams params;
  params.readFlatMapFieldAsStruct = {"c0"};
  params.flatMapFeatureSelector["c0"].features = {"c0"};
  nimble::BatchReader reader(&readFile, *leafPool_, nullptr, params);
  VectorPtr result;
  ASSERT_TRUE(reader.next(4, result));
  ASSERT_EQ(result->size(), 4);
  auto* c0 = result->asChecked<RowVector>()->childAt(0)->asChecked<RowVector>();
  auto& dict = c0->childAt(0);
  ASSERT_EQ(dict->encoding(), VectorEncoding::Simple::DICTIONARY);
  ASSERT_EQ(dict->size(), 4);
  auto* indices = dict->wrapInfo()->as<vector_size_t>();
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(indices[i], 0);
  }
  auto* values = dict->valueVector()->asChecked<ArrayVector>();
  ASSERT_EQ(values->size(), 1);
  auto* elements = values->elements()->asChecked<SimpleVector<int64_t>>();
  ASSERT_EQ(values->sizeAt(0), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(elements->valueAt(i + values->offsetAt(0)), 1 + i);
  }
}

#define ASSERT_CHUNK_COUNT(count, chunked) \
  for (auto __i = 0; __i < count; ++__i) { \
    ASSERT_TRUE(chunked.hasNext());        \
    auto chunk = chunked.nextChunk();      \
    EXPECT_LT(0, chunk.size());            \
  }                                        \
  ASSERT_FALSE(chunked.hasNext());

void testChunks(
    velox::memory::MemoryPool& rootPool,
    uint32_t minStreamChunkRawSize,
    uint32_t maxStreamChunkRawSize,
    std::vector<std::tuple<velox::VectorPtr, bool>> vectors,
    std::function<void(const nimble::TabletReader&)> verifier,
    folly::F14FastMap<std::string, std::set<std::string>> flatMapColumns = {}) {
  ASSERT_LT(0, vectors.size());
  auto& type = std::get<0>(vectors[0])->type();

  auto leafPool = rootPool.addLeafChild("chunk_leaf");
  auto expected = velox::BaseVector::create(type, 0, leafPool.get());

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  auto flushDecision = false;
  nimble::Writer writer(
      type,
      std::move(writeFile),
      rootPool,
      {
          .flatMapColumns = std::move(flatMapColumns),
          .minStreamChunkRawSize = minStreamChunkRawSize,
          .flushPolicyFactory =
              [&]() {
                return std::make_unique<nimble::LambdaFlushPolicy>(
                    /*flushLambda=*/[&](auto&) { return false; },
                    /*chunkLambda=*/[&](auto&) { return flushDecision; });
              },
          .enableChunking = true,
      });

  for (const auto& vector : vectors) {
    flushDecision = std::get<1>(vector);
    writer.write(std::get<0>(vector));
    expected->append(std::get<0>(vector).get());
  }

  writer.close();

  folly::writeFile(file, "/tmp/afile");

  auto tabletReadFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = nimble::TabletReader::create(
      tabletReadFile, leafPool.get(), makeTestTabletOptions(leafPool.get()));
  verifier(*tablet);

  nimble::BatchReader reader(
      std::make_shared<velox::InMemoryReadFile>(file), *leafPool);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(expected->size(), result));
  ASSERT_EQ(expected->size(), result->size());
  for (auto i = 0; i < expected->size(); ++i) {
    ASSERT_TRUE(expected->equalValueAt(result.get(), i, i));
  }
  ASSERT_FALSE(reader.next(1, result));

  validateChunkSize(reader, minStreamChunkRawSize, maxStreamChunkRawSize);
}

TEST_F(WriterTest, omitsAllNonNullRowNullStreams) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto makeVector = [&]() {
    return vectorMaker.rowVector(
        {"nested"},
        {vectorMaker.rowVector(
            {"c1"}, {vectorMaker.flatVector<int32_t>({1, 2, 3})})});
  };
  auto makeVectorWithAllocatedAllNonNullNulls = [&]() {
    constexpr velox::vector_size_t kSize = 3;
    auto nestedType = velox::ROW({"c1"}, {velox::INTEGER()});
    auto nestedNulls = velox::AlignedBuffer::allocate<bool>(
        kSize, leafPool_.get(), velox::bits::kNotNull);
    auto nested = std::make_shared<velox::RowVector>(
        leafPool_.get(),
        nestedType,
        nestedNulls,
        kSize,
        std::vector<velox::VectorPtr>{
            vectorMaker.flatVector<int32_t>({1, 2, 3})});

    auto rootNulls = velox::AlignedBuffer::allocate<bool>(
        kSize, leafPool_.get(), velox::bits::kNotNull);
    return std::make_shared<velox::RowVector>(
        leafPool_.get(),
        velox::ROW({"nested"}, {nestedType}),
        rootNulls,
        kSize,
        std::vector<velox::VectorPtr>{nested});
  };

  auto allNonNullVector = makeVector();
  auto allocatedAllNonNullVector = makeVectorWithAllocatedAllNonNullNulls();
  auto topLevelNullVector = makeVector();
  topLevelNullVector->setNull(1, true);

  struct TestCase {
    std::string_view name;
    velox::RowVectorPtr input;
    velox::RowVectorPtr expected;
    bool ignoreTopLevelNulls;
    bool enableChunking;
    bool expectRootNullStreamOmitted;
  };

  const std::vector<TestCase> testCases{
      {
          .name = "allNonNull",
          .input = allNonNullVector,
          .expected = allNonNullVector,
          .ignoreTopLevelNulls = false,
          .enableChunking = false,
          .expectRootNullStreamOmitted = true,
      },
      {
          .name = "allNonNull",
          .input = allNonNullVector,
          .expected = allNonNullVector,
          .ignoreTopLevelNulls = false,
          .enableChunking = true,
          .expectRootNullStreamOmitted = true,
      },
      {
          .name = "allocatedAllNonNullNulls",
          .input = allocatedAllNonNullVector,
          .expected = allNonNullVector,
          .ignoreTopLevelNulls = false,
          .enableChunking = false,
          .expectRootNullStreamOmitted = true,
      },
      {
          .name = "allocatedAllNonNullNulls",
          .input = allocatedAllNonNullVector,
          .expected = allNonNullVector,
          .ignoreTopLevelNulls = false,
          .enableChunking = true,
          .expectRootNullStreamOmitted = true,
      },
      {
          .name = "preservedTopLevelNulls",
          .input = topLevelNullVector,
          .expected = topLevelNullVector,
          .ignoreTopLevelNulls = false,
          .enableChunking = false,
          .expectRootNullStreamOmitted = false,
      },
      {
          .name = "preservedTopLevelNulls",
          .input = topLevelNullVector,
          .expected = topLevelNullVector,
          .ignoreTopLevelNulls = false,
          .enableChunking = true,
          .expectRootNullStreamOmitted = false,
      },
      {
          .name = "ignoredTopLevelNulls",
          .input = topLevelNullVector,
          .expected = allNonNullVector,
          .ignoreTopLevelNulls = true,
          .enableChunking = false,
          .expectRootNullStreamOmitted = true,
      },
      {
          .name = "ignoredTopLevelNulls",
          .input = topLevelNullVector,
          .expected = allNonNullVector,
          .ignoreTopLevelNulls = true,
          .enableChunking = true,
          .expectRootNullStreamOmitted = true,
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "case={}, enableChunking={}",
            testCase.name,
            testCase.enableChunking));

    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

    nimble::WriterOptions options;
    options.enableChunking = testCase.enableChunking;
    options.ignoreTopLevelNulls = testCase.ignoreTopLevelNulls;

    nimble::Writer writer(
        testCase.input->type(),
        std::move(writeFile),
        *rootPool_,
        std::move(options));
    writer.write(testCase.input);
    writer.close();

    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    ASSERT_EQ(1, tablet->stripeCount());

    nimble::BatchReader reader(readFile.get(), *leafPool_);
    const auto& root = reader.schema()->asRow();
    const auto& nested = root.childAt(0)->asRow();
    const auto rootNullOffset = root.nullsDescriptor().offset();
    const auto nestedNullOffset = nested.nullsDescriptor().offset();
    const auto stripeIdentifier = tablet->stripeIdentifier(0);
    const auto streamCount = tablet->streamCount(stripeIdentifier);
    ASSERT_GT(streamCount, static_cast<uint32_t>(rootNullOffset));
    ASSERT_GT(streamCount, static_cast<uint32_t>(nestedNullOffset));

    std::array<uint32_t, 2> nullStreamOffsets{rootNullOffset, nestedNullOffset};
    auto streamLoaders = tablet->load(stripeIdentifier, nullStreamOffsets);
    ASSERT_EQ(2, streamLoaders.size());

    // All-non-null Row null streams are omitted on the wire. The root null
    // stream is omitted only when the root is logically all-non-null (no nulls,
    // or top-level nulls ignored); otherwise it is written and round-trips the
    // top-level nulls. The always-non-null nested row null stream is omitted.
    if (testCase.expectRootNullStreamOmitted) {
      EXPECT_EQ(0, tablet->streamSize(stripeIdentifier, rootNullOffset));
      EXPECT_EQ(nullptr, streamLoaders[0]);
    } else {
      EXPECT_GT(tablet->streamSize(stripeIdentifier, rootNullOffset), 0);
      EXPECT_NE(nullptr, streamLoaders[0]);
    }
    EXPECT_EQ(0, tablet->streamSize(stripeIdentifier, nestedNullOffset));
    EXPECT_EQ(nullptr, streamLoaders[1]);

    velox::VectorPtr result;
    ASSERT_TRUE(reader.next(testCase.expected->size(), result));
    ASSERT_EQ(result->size(), testCase.expected->size());
    for (velox::vector_size_t i = 0; i < testCase.expected->size(); ++i) {
      ASSERT_TRUE(result->equalValueAt(testCase.expected.get(), i, i));
    }
  }
}

TEST_F(WriterTest, chunkedStreamsRowAllNullsNoChunks) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({}),
      });
  vector->appendNulls(5);

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{vector, false}, {vector, false}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // Logically, there should be two streams in the tablet.
        // However, when writing stripes, we do not write empty streams.
        // In this case, the integer column is empty, and therefore, omitted.
        ASSERT_EQ(1, tablet.streamCount(stripeIdentifier));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 0));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 1>{0});
        ASSERT_EQ(1, streamLoaders.size());

        // No chunks used, so expecting single chunk
        nimble::InMemoryChunkedStream chunked{
            *leafPool_, std::move(streamLoaders[0])};
        ASSERT_CHUNK_COUNT(1, chunked);
      });
}

TEST_F(WriterTest, chunkedStreamsRowAllNullsWithChunksMinSizeBig) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({}),
      });
  vector->appendNulls(5);

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 1024,
      /* maxStreamChunkRawSize */ 1024,
      {{vector, true}, {vector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // Logically, there should be two streams in the tablet.
        // However, when writing stripes, we do not write empty streams.
        // In this case, the integer column is empty, and therefore, omitted.
        ASSERT_EQ(1, tablet.streamCount(stripeIdentifier));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 0));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 1>{0});
        ASSERT_EQ(1, streamLoaders.size());

        // Chunks requested, but min chunk size is too big, so expecting one
        // merged chunk
        nimble::InMemoryChunkedStream chunked{
            *leafPool_, std::move(streamLoaders[0])};
        ASSERT_CHUNK_COUNT(1, chunked);
      });
}

TEST_F(WriterTest, chunkedStreamsRowAllNullsWithChunksMinSizeZero) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({}),
      });
  vector->appendNulls(5);

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{vector, true}, {vector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // Logically, there should be two streams in the tablet.
        // However, when writing stripes, we do not write empty streams.
        // In this case, the integer column is empty, and therefore, omitted.
        ASSERT_EQ(1, tablet.streamCount(stripeIdentifier));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 0));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 1>{0});
        ASSERT_EQ(1, streamLoaders.size());

        // Chunks requested, and min chunk size is zero, so expecting two
        // separate chunks.
        nimble::InMemoryChunkedStream chunked{
            *leafPool_, std::move(streamLoaders[0])};
        ASSERT_CHUNK_COUNT(2, chunked);
      });
}

TEST_F(WriterTest, chunkedStreamsRowSomeNullsNoChunks) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto nullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({}),
      });
  nullsVector->appendNulls(5);

  auto nonNullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({1, 2, 3}),
      });
  nonNullsVector->setNull(1, /* isNull */ true);

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{nullsVector, false}, {nonNullsVector, false}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // We have values in stream 2, so it is not optimized away.
        ASSERT_EQ(2, tablet.streamCount(stripeIdentifier));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 2>{0, 1});
        ASSERT_EQ(2, streamLoaders.size());
        {
          // No chunks requested, so expecting single chunk.
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[0])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
        {
          // No chunks requested, so expecting single chunk.
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[1])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
      });
}

TEST_F(WriterTest, chunkedStreamsRowSomeNullsWithChunksMinSizeBig) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto nullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({}),
      });
  nullsVector->appendNulls(5);

  auto nonNullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({1, 2, 3}),
      });
  nonNullsVector->setNull(1, /* isNull */ true);

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 1024,
      /* maxStreamChunkRawSize */ 1024,
      {{nullsVector, true}, {nonNullsVector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        ASSERT_EQ(2, tablet.streamCount(stripeIdentifier));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 2>{0, 1});
        ASSERT_EQ(2, streamLoaders.size());
        {
          // Chunks requested, but min chunk size is too big, so expecting one
          // merged chunk.
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[0])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
        {
          // Chunks requested, but min chunk size is too big, so expecting one
          // merged chunk.
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[1])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
      });
}

TEST_F(WriterTest, chunkedStreamsRowSomeNullsWithChunksMinSizeZero) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto nullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({}),
      });
  nullsVector->appendNulls(5);

  auto nonNullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({1, 2, 3}),
      });
  nonNullsVector->setNull(1, /* isNull */ true);

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{nullsVector, true}, {nonNullsVector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        ASSERT_EQ(2, tablet.streamCount(stripeIdentifier));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 2>{0, 1});
        ASSERT_EQ(2, streamLoaders.size());
        {
          // Chunks requested, and min chunk size is zero, so expecting two
          // separate chunks.
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[0])};
          ASSERT_CHUNK_COUNT(2, chunked);
        }
        {
          // Chunks requested, and min chunk size is zero. However, first
          // write didn't have any data, so no chunk was written.
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[1])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
      });
}

// Regression guard for per-chunk null counts on null-only (struct/ROW) streams.
// A nullable ROW writes a dedicated null stream whose validity is promoted to
// boolean bytes at chunk time -- there is no separate validity bitmap. The
// per-chunk null count must count those bytes; before the fix numNulls()
// returned 0 for such streams, silently mis-reporting struct/ROW nulls as "no
// nulls" in columnar.chunk.stats (a false "no nulls" assertion that would
// over-prune once struct pruning consumes it).
TEST_F(WriterTest, chunkNullCountsForStructNullStream) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  // Batch 1: 5 rows, the entire struct is null.
  auto nullsVector =
      vectorMaker.rowVector({"c1"}, {vectorMaker.flatVector<int32_t>({})});
  nullsVector->appendNulls(5);

  // Batch 2: 3 rows, struct null at row 1 (child values otherwise present).
  auto someNullsVector = vectorMaker.rowVector(
      {"c1"}, {vectorMaker.flatVector<int32_t>({1, 2, 3})});
  someNullsVector->setNull(1, /*isNull=*/true);

  const auto& type = nullsVector->type();

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  bool flushDecision = false;
  nimble::Writer writer(
      type,
      std::move(writeFile),
      *rootPool_,
      {
          .enableChunkIndex = true,
          // Never skip the stripe group, so the section is always written.
          .chunkStatsMinAvgChunks = 0,
          .minStreamChunkRawSize = 0,
          .flushPolicyFactory =
              [&]() {
                return std::make_unique<nimble::LambdaFlushPolicy>(
                    /*flushLambda=*/[&](auto&) { return false; },
                    /*chunkLambda=*/[&](auto&) { return flushDecision; });
              },
          .enableChunking = true,
      });

  // Force a chunk boundary between the two writes so the null stream is split
  // into two chunks (5 nulls, then 1 null).
  flushDecision = true;
  writer.write(nullsVector);
  writer.write(someNullsVector);
  writer.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = nimble::TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  ASSERT_EQ(1, tablet->stripeCount());

  auto stripeIdentifier = tablet->stripeIdentifier(0);
  auto chunkStats = stripeIdentifier.chunkStats();
  ASSERT_NE(chunkStats, nullptr)
      << "columnar.chunk.stats section should be present";

  // The root struct null stream is the only source of nulls: 5 (batch 1) + 1
  // (batch 2) = 6. Sum per-chunk null counts across every indexed stream; the
  // int child stream has no nulls of its own and is single-chunk (not indexed).
  const uint32_t streamCount = tablet->streamCount(stripeIdentifier);
  uint64_t totalChunkNullCount = 0;
  bool sawIndexedStream = false;
  bool sawNonZeroChunk = false;
  for (uint32_t streamId = 0; streamId < streamCount; ++streamId) {
    auto streamIndex = chunkStats->createStreamIndex(
        0, streamId, tablet->streamSize(stripeIdentifier, streamId));
    if (streamIndex == nullptr) {
      continue; // single-chunk (<=1) streams are not indexed
    }
    const uint32_t rows = streamIndex->rowCount();
    if (rows == 0) {
      continue;
    }
    sawIndexedStream = true;
    const uint32_t firstChunk = streamIndex->lookupChunk(0).chunkIndex;
    const uint32_t lastChunk = streamIndex->lookupChunk(rows - 1).chunkIndex;
    for (uint32_t ci = firstChunk; ci <= lastChunk; ++ci) {
      auto nullCount = streamIndex->chunkNullCount(ci);
      ASSERT_TRUE(nullCount.has_value())
          << "chunk " << ci << " of stream " << streamId
          << " should carry a null count";
      totalChunkNullCount += *nullCount;
      if (*nullCount > 0) {
        sawNonZeroChunk = true;
      }
    }
  }

  EXPECT_TRUE(sawIndexedStream)
      << "expected an indexed (multi-chunk) null stream";
  // Before the fix, every per-chunk null count for the struct null stream was
  // 0.
  EXPECT_TRUE(sawNonZeroChunk)
      << "per-chunk null counts for the struct null stream must be non-zero";
  EXPECT_EQ(6, totalChunkNullCount);
}

TEST_F(WriterTest, chunkStatsAbsentWhenChunkIndexDisabled) {
  // encodeChunk sets chunk.nullCount unconditionally; the chunk stats section
  // (and its null counts) must still be written only when the chunk index is
  // enabled. With the index off, no section should be produced.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto vec = vectorMaker.rowVector(
      {"c1"}, {vectorMaker.flatVector<int32_t>({1, 2, 3})});
  vec->setNull(1, /*isNull=*/true);

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vec->type(),
      std::move(writeFile),
      *rootPool_,
      {.enableChunkIndex = false});
  writer.write(vec);
  writer.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = nimble::TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  ASSERT_EQ(1, tablet->stripeCount());
  auto stripeIdentifier = tablet->stripeIdentifier(0);
  EXPECT_EQ(stripeIdentifier.chunkStats(), nullptr)
      << "no chunk stats section should be written when the index is disabled";
}

TEST_F(WriterTest, chunkIndexRequiresChunking) {
  auto type = velox::ROW({{"c1", velox::INTEGER()}});
  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  NIMBLE_ASSERT_USER_THROW(
      nimble::Writer(
          type,
          std::move(writeFile),
          *rootPool_,
          {.enableChunkIndex = true, .enableChunking = false}),
      "Chunk stats require chunking to be enabled.");
}

TEST_F(WriterTest, chunkedStreamsRowNoNullsNoChunks) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({1, 2, 3}),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{vector, false}, {vector, false}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        ASSERT_EQ(2, tablet.streamCount(stripeIdentifier));

        // When there are no nulls, the nulls stream is omitted.
        EXPECT_EQ(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 2>{0, 1});
        ASSERT_EQ(2, streamLoaders.size());
        {
          // Nulls stream should be missing, as all values are non-null
          EXPECT_FALSE(streamLoaders[0]);
        }
        {
          // No chunks requested, so expecting one chunk.
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[1])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
      });
}

TEST_F(WriterTest, chunkedStreamsRowNoNullsWithChunksMinSizeBig) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({1, 2, 3}),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 1024,
      /* maxStreamChunkRawSize */ 2048,
      {{vector, true}, {vector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        ASSERT_EQ(2, tablet.streamCount(stripeIdentifier));

        // When there are no nulls, the nulls stream is omitted.
        EXPECT_EQ(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 2>{0, 1});
        ASSERT_EQ(2, streamLoaders.size());
        {
          // Nulls stream should be missing, as all values are non-null
          EXPECT_FALSE(streamLoaders[0]);
        }
        {
          // Chunks requested, but min size is too big, so expecting one
          // merged chunk.
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[1])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
      });
}

TEST_F(WriterTest, chunkedStreamsRowNoNullsWithChunksMinSizeZero) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVector<int32_t>({1, 2, 3}),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{vector, true}, {vector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        ASSERT_EQ(2, tablet.streamCount(stripeIdentifier));

        // When there are no nulls, the nulls stream is omitted.
        EXPECT_EQ(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 2>{0, 1});
        ASSERT_EQ(2, streamLoaders.size());
        {
          // Nulls stream should be missing, as all values are non-null
          EXPECT_FALSE(streamLoaders[0]);
        }
        {
          // Chunks requested, with min size zero, so expecting two chunks.
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[1])};
          ASSERT_CHUNK_COUNT(2, chunked);
        }
      });
}

TEST_F(WriterTest, chunkedStreamsChildAllNullsNoChunks) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVectorNullable<int32_t>(
              {std::nullopt, std::nullopt, std::nullopt}),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{vector, false}, {vector, false}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // When all rows are not null, the nulls stream is omitted.
        // When all values are null, the values stream is omitted.
        // Since these are the last two stream, they are optimized away.
        ASSERT_EQ(0, tablet.streamCount(stripeIdentifier));
      });
}

TEST_F(WriterTest, chunkedStreamsChildAllNullsWithChunksMinSizeBig) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVectorNullable<int32_t>(
              {std::nullopt, std::nullopt, std::nullopt}),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 1024,
      /* maxStreamChunkRawSize */ 2048,
      {{vector, true}, {vector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // When all rows are not null, the nulls stream is omitted.
        // When all values are null, the values stream is omitted.
        // Since these are the last two stream, they are optimized away.
        ASSERT_EQ(0, tablet.streamCount(stripeIdentifier));
      });
}

TEST_F(WriterTest, chunkedStreamsChildAllNullsWithChunksMinSizeZero) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.flatVectorNullable<int32_t>(
              {std::nullopt, std::nullopt, std::nullopt}),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{vector, true}, {vector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // When all rows are not null, the nulls stream is omitted.
        // When all values are null, the values stream is omitted.
        // Since these are the last two stream, they are optimized away.
        ASSERT_EQ(0, tablet.streamCount(stripeIdentifier));
      });
}

TEST_F(WriterTest, chunkedStreamsFlatmapAllNullsNoChunks) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              std::vector<std::optional<
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
                  std::nullopt,
                  std::nullopt,
                  //   std::vector<std::pair<int32_t,
                  //   std::optional<int32_t>>>{
                  //       {5, 6}},
                  std::nullopt,
              }),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{vector, false}, {vector, false}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // Expected streams:
        // 0: Row nulls stream (expected empty, as all values are not null)
        // 1: Flatmap nulls stream
        ASSERT_EQ(2, tablet.streamCount(stripeIdentifier));
        EXPECT_EQ(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 2>{0, 1});
        ASSERT_EQ(2, streamLoaders.size());

        // No chunks used, so expecting single chunk
        nimble::InMemoryChunkedStream chunked{
            *leafPool_, std::move(streamLoaders[1])};
        ASSERT_CHUNK_COUNT(1, chunked);
      },
      /* flatmapColumns */
      (folly::F14FastMap<std::string, std::set<std::string>>){{"c1", {}}});
}

TEST_F(WriterTest, chunkedStreamsFlatmapAllNullsWithChunksMinSizeBig) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              std::vector<std::optional<
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
                  std::nullopt,
                  std::nullopt,
                  std::nullopt,
              }),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 1024,
      /* maxStreamChunkRawSize */ 2048,
      {{vector, true}, {vector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // Expected streams:
        // 0: Row nulls stream (expected empty, as all values are not null)
        // 1: Flatmap nulls stream
        ASSERT_EQ(2, tablet.streamCount(stripeIdentifier));
        EXPECT_EQ(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 2>{0, 1});
        ASSERT_EQ(2, streamLoaders.size());

        // Chunks requested, but min size is too big, so expecting single
        // merged chunk
        nimble::InMemoryChunkedStream chunked{
            *leafPool_, std::move(streamLoaders[1])};
        ASSERT_CHUNK_COUNT(1, chunked);
      },
      /* flatmapColumns */
      (folly::F14FastMap<std::string, std::set<std::string>>){{"c1", {}}});
}

TEST_F(WriterTest, chunkedStreamsFlatmapAllNullsWithChunksMinSizeZero) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              std::vector<std::optional<
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
                  std::nullopt,
                  std::nullopt,
                  std::nullopt,
              }),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{vector, true}, {vector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // Expected streams:
        // 0: Row nulls stream (expected empty, as all values are not null)
        // 1: Flatmap nulls stream
        ASSERT_EQ(2, tablet.streamCount(stripeIdentifier));
        EXPECT_EQ(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 2>{0, 1});
        ASSERT_EQ(2, streamLoaders.size());

        // Chunks requested, with min size zero, so expecting two chunks
        nimble::InMemoryChunkedStream chunked{
            *leafPool_, std::move(streamLoaders[1])};
        ASSERT_CHUNK_COUNT(2, chunked);
      },
      /* flatmapColumns */
      (folly::F14FastMap<std::string, std::set<std::string>>){{"c1", {}}});
}

TEST_F(WriterTest, chunkedStreamsFlatmapSomeNullsNoChunks) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto nullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              std::vector<std::optional<
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
                  std::nullopt,
                  std::nullopt,
                  std::nullopt,
              }),
      });
  auto nonNullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              std::vector<std::optional<
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
                  std::nullopt,
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>{
                      {5, 6}},
                  std::nullopt,
              }),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{nullsVector, false}, {nonNullsVector, false}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // Expected streams:
        // 0: Row nulls stream (expected empty, as all values are not null)
        // 1: Flatmap nulls stream
        // 2: Scalar stream (flatmap value for key 5)
        // 3: Scalar stream (flatmap in-map for key 5)
        ASSERT_EQ(4, tablet.streamCount(stripeIdentifier));
        EXPECT_EQ(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 2));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 3));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 4>{0, 1, 2, 3});
        ASSERT_EQ(4, streamLoaders.size());

        EXPECT_FALSE(streamLoaders[0]);
        EXPECT_TRUE(streamLoaders[1]);
        EXPECT_TRUE(streamLoaders[2]);
        EXPECT_TRUE(streamLoaders[3]);

        {
          // No chunks used, so expecting single chunk
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[1])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
        {
          // No chunks used, so expecting single chunk
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[2])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
        {
          // No chunks used, so expecting single chunk
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[3])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
      },
      /* flatmapColumns */
      (folly::F14FastMap<std::string, std::set<std::string>>){{"c1", {}}});
}

TEST_F(WriterTest, chunkedStreamsFlatmapSomeNullsWithChunksMinSizeBig) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto nullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              std::vector<std::optional<
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
                  std::nullopt,
                  std::nullopt,
                  std::nullopt,
              }),
      });
  auto nonNullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              std::vector<std::optional<
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
                  std::nullopt,
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>{
                      {5, 6}},
                  std::nullopt,
              }),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 1024,
      /* maxStreamChunkRawSize */ 2048,
      {{nullsVector, true}, {nonNullsVector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // Expected streams:
        // 0: Row nulls stream (expected empty, as all values are not null)
        // 1: Flatmap nulls stream
        // 2: Scalar stream (flatmap value for key 5)
        // 3: Scalar stream (flatmap in-map for key 5)
        ASSERT_EQ(4, tablet.streamCount(stripeIdentifier));
        EXPECT_EQ(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 2));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 3));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 4>{0, 1, 2, 3});
        ASSERT_EQ(4, streamLoaders.size());

        EXPECT_FALSE(streamLoaders[0]);
        EXPECT_TRUE(streamLoaders[1]);
        EXPECT_TRUE(streamLoaders[2]);
        EXPECT_TRUE(streamLoaders[3]);

        {
          // Chunks requested, but min size is big, so expecting single merged
          // chunk
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[1])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
        {
          // Chunks requested, but min size is big, so expecting single merged
          // chunk
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[2])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
        {
          // Chunks requested, but min size is big, so expecting single merged
          // chunk
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[3])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
      },
      /* flatmapColumns */
      (folly::F14FastMap<std::string, std::set<std::string>>){{"c1", {}}});
}

TEST_F(WriterTest, chunkedStreamsFlatmapSomeNullsWithChunksMinSizeZero) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto nullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              std::vector<std::optional<
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
                  std::nullopt,
                  std::nullopt,
                  std::nullopt,
              }),
      });
  auto nonNullsVector = vectorMaker.rowVector(
      {"c1"},
      {
          vectorMaker.mapVector<int32_t, int32_t>(
              std::vector<std::optional<
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>>>{
                  std::nullopt,
                  std::vector<std::pair<int32_t, std::optional<int32_t>>>{
                      {5, 6}},
                  std::nullopt,
              }),
      });

  testChunks(
      *rootPool_,
      /* minStreamChunkRawSize */ 0,
      /* maxStreamChunkRawSize */ 1024,
      {{nullsVector, true}, {nonNullsVector, true}},
      [&](const auto& tablet) {
        auto stripeIdentifier = tablet.stripeIdentifier(0);
        ASSERT_EQ(1, tablet.stripeCount());

        // Expected streams:
        // 0: Row nulls stream (expected empty, as all values are not null)
        // 1: Flatmap nulls stream
        // 2: Scalar stream (flatmap value for key 5)
        // 3: Scalar stream (flatmap in-map for key 5)
        ASSERT_EQ(4, tablet.streamCount(stripeIdentifier));
        EXPECT_EQ(0, tablet.streamSize(stripeIdentifier, 0));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 1));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 2));
        EXPECT_LT(0, tablet.streamSize(stripeIdentifier, 3));

        auto streamLoaders =
            tablet.load(stripeIdentifier, std::array<uint32_t, 4>{0, 1, 2, 3});
        ASSERT_EQ(4, streamLoaders.size());

        EXPECT_FALSE(streamLoaders[0]);
        EXPECT_TRUE(streamLoaders[1]);
        EXPECT_TRUE(streamLoaders[2]);
        EXPECT_TRUE(streamLoaders[3]);

        {
          // Chunks requested, with min size zero, so expecting two chunks
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[1])};
          ASSERT_CHUNK_COUNT(2, chunked);
        }
        {
          // Chunks requested, with min size zero, but first write didn't
          // contain any values, so expecting single merged chunk
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[2])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
        {
          // Chunks requested, with min size zero, but first write didn't
          // have any items in the map, so expecting single merged chunk
          nimble::InMemoryChunkedStream chunked{
              *leafPool_, std::move(streamLoaders[3])};
          ASSERT_CHUNK_COUNT(1, chunked);
        }
      },
      /* flatmapColumns */
      (folly::F14FastMap<std::string, std::set<std::string>>){{"c1", {}}});
}

TEST_F(WriterTest, rawSizeWritten) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  constexpr uint64_t expectedRawSize = sizeof(int32_t) * 20;
  auto vector = vectorMaker.rowVector(
      {"row1", "row2"},
      {
          vectorMaker.rowVector({
              vectorMaker.flatVector<int32_t>({1, 2, 3, 4, 5}),
              vectorMaker.flatVector<int32_t>({1, 2, 3, 4, 5}),
          }),
          vectorMaker.rowVector({
              vectorMaker.flatVector<int32_t>({1, 2, 3, 4, 5}),
              vectorMaker.flatVector<int32_t>({1, 2, 3, 4, 5}),
          }),
      });

  // Test with enableVectorizedStats = false (default, uses kStatsSection)
  {
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options{};
    options.enableVectorizedStats = false;
    nimble::Writer writer(
        vector->type(), std::move(writeFile), *rootPool_, options);
    writer.write(vector);
    writer.close();

    auto statsReadFile = std::make_shared<velox::InMemoryReadFile>(file);
    nimble::TabletReader::Options readerOptions =
        makeTestTabletOptions(leafPool_.get());
    readerOptions.preloadOptionalSections = {
        std::string(facebook::nimble::kStatsSection)};
    auto tablet = facebook::nimble::TabletReader::create(
        statsReadFile, leafPool_.get(), readerOptions);
    auto statsSection =
        tablet->loadOptionalSection(readerOptions.preloadOptionalSections[0]);
    ASSERT_TRUE(statsSection.has_value());

    // Use flatbuffers to deserialize the stats payload
    auto rawSize = flatbuffers::GetRoot<nimble::serialization::Stats>(
                       statsSection->content().data())
                       ->raw_size();
    ASSERT_EQ(expectedRawSize, rawSize);
  }

  // Test with enableVectorizedStats = true (uses kVectorizedStatsSection)
  {
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions writerOptions{};
    writerOptions.enableVectorizedStats = true;
    nimble::Writer writer(
        vector->type(), std::move(writeFile), *rootPool_, writerOptions);
    writer.write(vector);
    writer.close();

    auto vecStatsReadFile = std::make_shared<velox::InMemoryReadFile>(file);
    nimble::TabletReader::Options readerOptions =
        makeTestTabletOptions(leafPool_.get());
    readerOptions.preloadOptionalSections = {
        std::string(facebook::nimble::kVectorizedStatsSection)};
    auto tablet = facebook::nimble::TabletReader::create(
        vecStatsReadFile, leafPool_.get(), readerOptions);
    auto statsSection =
        tablet->loadOptionalSection(readerOptions.preloadOptionalSections[0]);
    ASSERT_TRUE(statsSection.has_value());

    // Use VectorizedFileStats to deserialize the stats payload
    auto fileStats = nimble::VectorizedFileStats::deserialize(
        statsSection->content(), *leafPool_);
    ASSERT_NE(fileStats, nullptr);

    // Convert to column statistics using schema and nimbleType
    auto nimbleType = nimble::convertToNimbleType(*vector->type());
    auto columnStats =
        fileStats->toColumnStatistics(vector->type(), nimbleType);
    ASSERT_FALSE(columnStats.empty());

    // The root column statistics contains the raw size (logical size)
    auto rawSize = columnStats.front()->getLogicalSize();
    ASSERT_EQ(expectedRawSize, rawSize);
  }
}

struct ChunkFlushPolicyTestCase {
  const size_t batchCount{20};
  const bool enableChunking{true};
  const uint64_t targetStripeSizeBytes{250 << 10};
  const uint64_t writerMemoryHighThresholdBytes{80 << 10};
  const uint64_t writerMemoryLowThresholdBytes{75 << 10};
  const double estimatedCompressionFactor{1.3};
  const uint32_t minStreamChunkRawSize{100};
  const uint32_t maxStreamChunkRawSize{128 << 10};
  const uint32_t expectedStripeCount{0};
  const uint32_t expectedMaxChunkCount{0};
  const uint32_t expectedMinChunkCount{0};
  const uint32_t chunkedStreamBatchSize{2};
};

class ChunkFlushPolicyTest
    : public WriterTest,
      public ::testing::WithParamInterface<ChunkFlushPolicyTestCase> {};

TEST_P(ChunkFlushPolicyTest, chunkFlushPolicyIntegration) {
  const auto type = velox::ROW(
      {{"BIGINT", velox::BIGINT()}, {"SMALLINT", velox::SMALLINT()}});
  nimble::WriterOptions writerOptions{
      .minStreamChunkRawSize = GetParam().minStreamChunkRawSize,
      .maxStreamChunkRawSize = GetParam().maxStreamChunkRawSize,
      .chunkedStreamBatchSize = GetParam().chunkedStreamBatchSize,
      .flushPolicyFactory = GetParam().enableChunking
          ? []() -> std::unique_ptr<nimble::FlushPolicy> {
              return std::make_unique<nimble::ChunkFlushPolicy>(
                      nimble::ChunkFlushPolicyConfig{
                          .writerMemoryHighThresholdBytes = GetParam().writerMemoryHighThresholdBytes,
                          .writerMemoryLowThresholdBytes = GetParam().writerMemoryLowThresholdBytes,
                          .targetStripeSizeBytes = GetParam().targetStripeSizeBytes,
                          .estimatedCompressionFactor =
                              GetParam().estimatedCompressionFactor,
                      });
            }
          : []() -> std::unique_ptr<nimble::FlushPolicy> {
              return std::make_unique<nimble::StripeRawSizeFlushPolicy>(
                  GetParam().targetStripeSizeBytes);
            },
      .enableChunking = GetParam().enableChunking,
  };

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(writerOptions));
  const auto batches = generateBatches(
      type,
      GetParam().batchCount,
      /*size=*/4000,
      /*seed=*/20221110,
      *leafPool_);

  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();
  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);
  ChunkSizeResults result = validateChunkSize(
      reader,
      GetParam().minStreamChunkRawSize,
      GetParam().maxStreamChunkRawSize);

  EXPECT_EQ(GetParam().expectedStripeCount, result.stripeCount);
  EXPECT_EQ(GetParam().expectedMaxChunkCount, result.maxChunkCount);
  EXPECT_EQ(GetParam().expectedMinChunkCount, result.minChunkCount);
}

TEST_F(WriterTest, batchedChunkingRelievesMemoryPressure) {
  // Verify we stop chunking early when chunking relieves memory pressure.
  const uint32_t seed = FLAGS_writer_tests_seed > 0 ? FLAGS_writer_tests_seed
                                                    : folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng{seed};
  const uint32_t rowCount =
      std::uniform_int_distribution<uint32_t>(1, 4096)(rng);

  velox::VectorFuzzer fuzzer({.vectorSize = rowCount}, leafPool_.get(), seed);
  const auto stringColumn = fuzzer.fuzzFlat(velox::VARCHAR());
  const auto intColumn = fuzzer.fuzzFlat(velox::INTEGER());

  for (bool disableSharedStringBuffers : {true, false}) {
    LOG(INFO) << "disableSharedStringBuffers: " << disableSharedStringBuffers;
    nimble::RawSizeContext context;
    nimble::OrderedRanges ranges;
    ranges.add(0, rowCount);
    // When shared string buffers are disabled, each row contributes the size
    // of a uint64_t offset; otherwise it contributes the size of a
    // std::string_view.
    const uint64_t perRowOverhead = disableSharedStringBuffers
        ? sizeof(uint64_t)
        : sizeof(std::string_view);
    const uint64_t stringColumnRawSize =
        nimble::getRawSizeFromVector(stringColumn, ranges, context) +
        perRowOverhead * rowCount;
    const uint64_t intColumnRawSize =
        nimble::getRawSizeFromVector(intColumn, ranges, context);

    constexpr size_t kColumnCount = 20;
    constexpr size_t kBatchSize = 4;
    std::vector<velox::VectorPtr> children(kColumnCount);
    std::vector<std::string> columnNames(kColumnCount);
    uint64_t totalRawSize = 0;
    for (size_t i = 0; i < kColumnCount; i += 2) {
      columnNames[i] = fmt::format("string_column_{}", i);
      columnNames[i + 1] = fmt::format("int_column_{}", i);
      children[i] = stringColumn;
      children[i + 1] = intColumn;
      totalRawSize += intColumnRawSize + stringColumnRawSize;
    }

    velox::test::VectorMaker vectorMaker{leafPool_.get()};
    const auto rowVector = vectorMaker.rowVector(columnNames, children);

    // Ensure we can chunk the integer streams into multiple chunks
    const uint64_t minChunkSize = intColumnRawSize / 2;
    // In the aggresive stage, we chunk large streams in the first batch. We set
    // the memoryPressureThreshold with the assumption of at least once max
    // chunk being produced for each stream in that batch.
    const uint64_t maxChunkSize = stringColumnRawSize / 2;
    uint64_t memoryPressureThreshold =
        totalRawSize - (kBatchSize * maxChunkSize);

    std::vector<bool> actualChunkingDecisions;
    nimble::WriterOptions writerOptions;
    writerOptions.chunkedStreamBatchSize = kBatchSize;
    writerOptions.enableChunking = true;
    writerOptions.disableSharedStringBuffers = disableSharedStringBuffers;
    writerOptions.minStreamChunkRawSize = minChunkSize;
    writerOptions.maxStreamChunkRawSize = maxChunkSize;
    writerOptions.flushPolicyFactory =
        [&]() -> std::unique_ptr<nimble::FlushPolicy> {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /* shouldFlush */ [](const auto&) { return true; },
          /* shouldChunk */
          [&](const nimble::StripeProgress& stripeProgress) {
            bool shouldChunk =
                stripeProgress.stripeRawSize > memoryPressureThreshold;
            actualChunkingDecisions.push_back(shouldChunk);
            if (!shouldChunk) {
              // Force memory pressure after the initial aggressive stage.
              memoryPressureThreshold = 0;
              // Force beginning of the non-aggressive chunking stage.
              shouldChunk = true;
            }
            return shouldChunk;
          });
    };

    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::Writer writer(
        rowVector->type(), std::move(writeFile), *rootPool_, writerOptions);
    writer.write(rowVector);
    writer.close();

    // We expect true and then false for the aggressive stage.
    EXPECT_GE(actualChunkingDecisions.size(), 2);
    EXPECT_TRUE(actualChunkingDecisions[0]);
    EXPECT_FALSE(actualChunkingDecisions[1]);

    velox::InMemoryReadFile readFile(file);
    nimble::BatchReader reader(&readFile, *leafPool_);
    validateChunkSize(
        reader,
        writerOptions.minStreamChunkRawSize,
        writerOptions.maxStreamChunkRawSize);
  }
}

TEST_F(WriterTest, ignoreTopLevelNulls) {
  auto seed = folly::randomNumberSeed();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng{seed};
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  constexpr velox::vector_size_t kSize = 10;
  auto type = velox::ROW({"c1"}, {velox::INTEGER()});
  auto nulls = velox::AlignedBuffer::allocate<bool>(kSize, leafPool_.get());
  auto rawNulls = nulls->asMutable<uint64_t>();
  for (auto i = 0; i < kSize; ++i) {
    velox::bits::setBit(rawNulls, i, folly::Random::oneIn(2, rng));
  }
  auto vector =
      vectorMaker.flatVector<int32_t>(kSize, [](auto row) { return row; });
  velox::VectorPtr rowVector = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      type,
      nulls,
      kSize,
      std::vector<velox::VectorPtr>{vector});

  auto verify =
      [](auto& pool, const auto& input, const auto& expected, auto options) {
        std::string file;
        auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
        nimble::Writer writer(
            input->type(), std::move(writeFile), pool, std::move(options));
        writer.write(input);
        writer.close();

        velox::InMemoryReadFile readFile(file);
        nimble::BatchReader reader(&readFile, pool);
        velox::VectorPtr output;
        reader.next(expected->size(), output);
        ASSERT_EQ(output->size(), expected->size());
        for (auto i = 0; i < kSize; ++i) {
          ASSERT_TRUE(output->equalValueAt(expected.get(), i, i));
        }
      };

  // Write with top level nulls
  verify(*leafPool_, rowVector, rowVector, nimble::WriterOptions{});

  // Write ignoring top level nulls with flat input
  {
    auto expected = std::make_shared<velox::RowVector>(
        leafPool_.get(),
        type,
        nullptr,
        kSize,
        std::vector<velox::VectorPtr>{vector});
    verify(
        *leafPool_,
        rowVector,
        expected,
        nimble::WriterOptions{
            .ignoreTopLevelNulls = true,
        });
  }

  // Write ignoring top level nulls with encoded input
  {
    auto indices = velox::AlignedBuffer::allocate<velox::vector_size_t>(
        kSize, leafPool_.get());
    auto rawIndices = indices->asMutable<velox::vector_size_t>();
    for (auto i = 0; i < kSize; ++i) {
      rawIndices[i] = i;
    }
    rowVector =
        velox::BaseVector::wrapInDictionary(nullptr, indices, kSize, rowVector);
    auto expected = std::make_shared<velox::RowVector>(
        leafPool_.get(),
        type,
        nullptr,
        kSize,
        std::vector<velox::VectorPtr>{vector});
    verify(
        *leafPool_,
        rowVector,
        expected,
        nimble::WriterOptions{
            .ignoreTopLevelNulls = true,
        });
  }
}

struct TimestampTestCase {
  std::string name;
  velox::Timestamp timestamp;
};

class TimestampEdgeCaseTest
    : public WriterTest,
      public ::testing::WithParamInterface<TimestampTestCase> {};

// We rely on fuzz tests in BatchReaderTests for more complex data shapes.
TEST_P(TimestampEdgeCaseTest, roundTrip) {
  auto testCase = GetParam();
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"timestamp"},
      {vectorMaker.flatVector<velox::Timestamp>(
          std::vector<velox::Timestamp>{testCase.timestamp})});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(vector->type(), std::move(writeFile), *rootPool_, {});
  writer.write(vector);
  writer.close();

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(1, result));

  auto actual = result->as<velox::RowVector>()
                    ->childAt(0)
                    ->asFlatVector<velox::Timestamp>()
                    ->valueAt(0);

  EXPECT_EQ(testCase.timestamp.getSeconds(), actual.getSeconds());
  EXPECT_EQ(testCase.timestamp.getNanos(), actual.getNanos());
}

// Test that timestamps exceeding Nimble's microsecond range cause overflow.
// Nimble stores timestamps as int64 microseconds, so seconds values beyond
// INT64_MAX/1'000'000 or INT64_MIN/1'000'000 will overflow during conversion.
TEST_F(WriterTest, timestampOverflowMax) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto overflowVector = vectorMaker.rowVector(
      {"timestamp"},
      {vectorMaker.flatVector<velox::Timestamp>(std::vector<velox::Timestamp>{
          velox::Timestamp(INT64_MAX / 1'000'000 + 1, 0)})});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      overflowVector->type(), std::move(writeFile), *rootPool_, {});
  EXPECT_THROW(writer.write(overflowVector), nimble::NimbleUserError);
}

TEST_F(WriterTest, timestampOverflowMin) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto overflowVector = vectorMaker.rowVector(
      {"timestamp"},
      {vectorMaker.flatVector<velox::Timestamp>(std::vector<velox::Timestamp>{
          velox::Timestamp(INT64_MIN / 1'000'000 - 1, 0)})});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      overflowVector->type(), std::move(writeFile), *rootPool_, {});
  EXPECT_THROW(writer.write(overflowVector), nimble::NimbleUserError);
}

INSTANTIATE_TEST_CASE_P(
    TimestampEdgeCaseTestSuite,
    TimestampEdgeCaseTest,
    ::testing::Values(
        TimestampTestCase{"Epoch", velox::Timestamp(0, 0)},
        TimestampTestCase{"EpochPlus1Nano", velox::Timestamp::fromNanos(1)},
        TimestampTestCase{"EpochMinus1Nano", velox::Timestamp::fromNanos(-1)},

        TimestampTestCase{"ExactMicrosecond", velox::Timestamp(0, 1'000)},
        TimestampTestCase{"ExactMillisecond", velox::Timestamp(0, 1'000'000)},

        TimestampTestCase{"Negative1ms", velox::Timestamp::fromMillis(-1)},
        TimestampTestCase{"Negative999ms", velox::Timestamp::fromMillis(-999)},
        TimestampTestCase{
            "Negative1000ms",
            velox::Timestamp::fromMillis(-1000)},
        TimestampTestCase{
            "Negative1500ms",
            velox::Timestamp::fromMillis(-1500)},

        TimestampTestCase{
            "SubMicroPrecisionConversion",
            velox::Timestamp(100, 999'999'999)},

        TimestampTestCase{
            "MaxMicros",
            velox::Timestamp(INT64_MAX / 1'000'000, 999)},
        TimestampTestCase{
            "MinMicros",
            velox::Timestamp(INT64_MIN / 1'000'000, 0)}));

INSTANTIATE_TEST_CASE_P(
    StripeRawSizeFlushPolicyTestSuite,
    StripeRawSizeFlushPolicyTest,
    ::testing::Values(
        StripeRawSizeFlushPolicyTestCase{
            .batchCount = 50,
            .rawStripeSize = 256 << 10,
            .stripeCount = 4},
        StripeRawSizeFlushPolicyTestCase{
            .batchCount = 100,
            .rawStripeSize = 256 << 10,
            .stripeCount = 7},
        StripeRawSizeFlushPolicyTestCase{
            .batchCount = 100,
            .rawStripeSize = 256 << 11,
            .stripeCount = 4},
        StripeRawSizeFlushPolicyTestCase{
            .batchCount = 100,
            .rawStripeSize = 256 << 12,
            .stripeCount = 2},
        StripeRawSizeFlushPolicyTestCase{
            .batchCount = 100,
            .rawStripeSize = 256 << 20,
            .stripeCount = 1}));

INSTANTIATE_TEST_CASE_P(
    ChunkFlushPolicyTestSuite,
    ChunkFlushPolicyTest,
    ::testing::Values(
        // Base case (no chunking, RawStripeSizeFlushPolicy)
        ChunkFlushPolicyTestCase{
            .batchCount = 20,
            .enableChunking = false,
            .targetStripeSizeBytes = 250 << 10, // 250KB
            .writerMemoryHighThresholdBytes = 80 << 10,
            .writerMemoryLowThresholdBytes = 75 << 10,
            .estimatedCompressionFactor = 1.3,
            .minStreamChunkRawSize = 100,
            .maxStreamChunkRawSize =
                std::numeric_limits<uint32_t>::max(), // no limit
            .expectedStripeCount = 4,
            .expectedMaxChunkCount = 1,
            .expectedMinChunkCount = 1,
            .chunkedStreamBatchSize = 2,
        },
        // Baseline with default settings (has chunking)
        ChunkFlushPolicyTestCase{
            .batchCount = 20,
            .enableChunking = true,
            .targetStripeSizeBytes = 250 << 10, // 250KB
            .writerMemoryHighThresholdBytes = 80 << 10,
            .writerMemoryLowThresholdBytes = 75 << 10,
            .estimatedCompressionFactor = 1.3,
            .minStreamChunkRawSize = 100,
            .maxStreamChunkRawSize = 128 << 10,
            .expectedStripeCount = 7,
            .expectedMaxChunkCount = 2,
            .expectedMinChunkCount = 1,
            .chunkedStreamBatchSize = 2,
        },
        // Reducing maxStreamChunkRawSize produces more chunks
        ChunkFlushPolicyTestCase{
            .batchCount = 20,
            .enableChunking = true,
            .targetStripeSizeBytes = 250 << 10, // 250KB
            .writerMemoryHighThresholdBytes = 80 << 10,
            .writerMemoryLowThresholdBytes = 75 << 10,
            .estimatedCompressionFactor = 1.0,
            .minStreamChunkRawSize = 100,
            .maxStreamChunkRawSize = 12
                << 10, // -126KB (as opposed to 128KB in other cases)
            .expectedStripeCount = 7,
            .expectedMaxChunkCount = 9, // +7 chunks
            .expectedMinChunkCount = 2, // +1 chunk
            .chunkedStreamBatchSize = 10,
        },
        // High memory regression threshold and no compression
        // Stripe count identical to RawStripeSizeFlushPolicy
        ChunkFlushPolicyTestCase{
            .batchCount = 20,
            .enableChunking = true,
            .targetStripeSizeBytes = 250 << 10, // 250KB
            .writerMemoryHighThresholdBytes = 500
                << 10, // 500KB (as opposed to 80 KB in other cases)
            .writerMemoryLowThresholdBytes = 75 << 10,
            .estimatedCompressionFactor =
                1.0, // No compression (as opposed to 1.3 in other cases)
            .minStreamChunkRawSize = 100,
            .maxStreamChunkRawSize = 128 << 10,
            .expectedStripeCount = 4,
            .expectedMaxChunkCount = 2,
            .expectedMinChunkCount = 1,
            .chunkedStreamBatchSize = 2,
        },
        // Low memory regression threshold
        // Produces file with more min chunks per stripe
        ChunkFlushPolicyTestCase{
            .batchCount = 20,
            .enableChunking = true,
            .targetStripeSizeBytes = 250 << 10,
            .writerMemoryHighThresholdBytes = 40
                << 10, // 40KB (as opposed to 80KB in other cases)
            .writerMemoryLowThresholdBytes = 35
                << 10, // 35KB (as opposed to 75KB in other cases)
            .estimatedCompressionFactor = 1.3,
            .minStreamChunkRawSize = 100,
            .maxStreamChunkRawSize = 128 << 10,
            .expectedStripeCount = 10,
            .expectedMaxChunkCount = 2,
            .expectedMinChunkCount = 2, // +1 chunk
            .chunkedStreamBatchSize = 2,
        },
        // High target stripe size bytes (with disabled memory pressure
        // optimization) produces fewer stripes.
        ChunkFlushPolicyTestCase{
            .batchCount = 20,
            .enableChunking = true,
            .targetStripeSizeBytes = 900
                << 10, // 900KB (as opposed to 250KB in other cases)
            .writerMemoryHighThresholdBytes = 2
                << 20, // 2MB (as opposed to 80KB in other cases)
            .writerMemoryLowThresholdBytes = 1
                << 20, // 1MB (as opposed to 75KB in other cases)
            .estimatedCompressionFactor = 1.3,
            .minStreamChunkRawSize = 100,
            .maxStreamChunkRawSize = 128 << 10,
            .expectedStripeCount = 1,
            .expectedMaxChunkCount = 5,
            .expectedMinChunkCount = 2,
            .chunkedStreamBatchSize = 2,

        },
        // Low target stripe size bytes (with disabled memory pressure
        // optimization) produces more stripes. Single chunks.
        ChunkFlushPolicyTestCase{
            .batchCount = 20,
            .enableChunking = true,
            .targetStripeSizeBytes = 90
                << 10, // 90KB (as opposed to 250KB in other cases)
            .writerMemoryHighThresholdBytes = 2
                << 20, // 2MB (as opposed to 80KB in other cases)
            .writerMemoryLowThresholdBytes = 1
                << 20, // 1MB (as opposed to 75KB in other cases)
            .estimatedCompressionFactor = 1.3,
            .minStreamChunkRawSize = 100,
            .maxStreamChunkRawSize = 128 << 10,
            .expectedStripeCount = 7,
            .expectedMaxChunkCount = 1,
            .expectedMinChunkCount = 1,
            .chunkedStreamBatchSize = 2,

        },
        // Higher chunked stream batch size (no change in policy)
        ChunkFlushPolicyTestCase{
            .batchCount = 20,
            .enableChunking = true,
            .targetStripeSizeBytes = 250 << 10, // 250KB
            .writerMemoryHighThresholdBytes = 80 << 10,
            .writerMemoryLowThresholdBytes = 75 << 10,
            .estimatedCompressionFactor = 1.0,
            .minStreamChunkRawSize = 100,
            .maxStreamChunkRawSize = 128 << 10,
            .expectedStripeCount = 7,
            .expectedMaxChunkCount = 2,
            .expectedMinChunkCount = 1,
            .chunkedStreamBatchSize = 10}));

TEST_F(WriterTest, chunkSizeStatsPopulatedWhenChunkingTriggered) {
  const auto type = velox::ROW({{"c0", velox::BIGINT()}});

  nimble::WriterOptions options{
      .minStreamChunkRawSize = 100,
      .maxStreamChunkRawSize = 128 << 10,
      .flushPolicyFactory = []() -> std::unique_ptr<nimble::FlushPolicy> {
        return std::make_unique<nimble::ChunkFlushPolicy>(
            nimble::ChunkFlushPolicyConfig{
                .writerMemoryHighThresholdBytes = 80 << 10,
                .writerMemoryLowThresholdBytes = 75 << 10,
                .targetStripeSizeBytes = 250 << 10,
                .estimatedCompressionFactor = 1.3,
            });
      },
      .enableChunking = true,
  };

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(options));

  const auto batches = generateBatches(
      type, /*batchCount=*/20, /*size=*/4000, /*seed=*/42, *leafPool_);
  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();

  const auto chunkSizeBytes = nimble::runtimeStat(
      writer.runtimeStats(), nimble::Writer::RuntimeStats::kChunkSizeBytes);
  EXPECT_GT(chunkSizeBytes.count, 0);
  EXPECT_GT(chunkSizeBytes.sum, 0);
  EXPECT_LE(chunkSizeBytes.min, chunkSizeBytes.max);
}

TEST_F(WriterTest, smallMaxChunkSizeProducesMultipleChunks) {
  const auto type =
      velox::ROW({{"c0", velox::BIGINT()}, {"c1", velox::VARCHAR()}});

  nimble::WriterOptions options{
      .enableChunkIndex = true,
      .minStreamChunkRawSize = 0,
      .maxStreamChunkRawSize = 2048,
      .flushPolicyFactory = []() -> std::unique_ptr<nimble::FlushPolicy> {
        return std::make_unique<nimble::StripeRawSizeFlushPolicy>(256ULL << 20);
      },
      .enableChunking = true,
  };

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(options));

  velox::VectorFuzzer fuzzer(
      {.vectorSize = 5000, .nullRatio = 0.1, .stringLength = 50},
      leafPool_.get(),
      42);
  for (int i = 0; i < 4; ++i) {
    writer.write(fuzzer.fuzzInputRow(type));
  }
  writer.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = nimble::TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  ASSERT_EQ(1, tablet->stripeCount());

  auto stripeId = tablet->stripeIdentifier(0);
  const auto streamCount = tablet->streamCount(stripeId);
  bool foundMultiChunkStream = false;
  for (uint32_t s = 0; s < streamCount; ++s) {
    auto streamLoaders = tablet->load(stripeId, std::array<uint32_t, 1>{s});
    if (streamLoaders.empty() || !streamLoaders[0]) {
      continue;
    }
    nimble::InMemoryChunkedStream chunked{
        *leafPool_, std::move(streamLoaders[0])};
    uint32_t chunkCount = 0;
    while (chunked.hasNext()) {
      chunked.nextChunk();
      ++chunkCount;
    }
    if (chunkCount > 1) {
      foundMultiChunkStream = true;
      LOG(INFO) << "stream " << s << ": " << chunkCount << " chunks";
    }
  }
  EXPECT_TRUE(foundMultiChunkStream)
      << "Expected at least one stream with multiple chunks at maxStreamChunkRawSize=2048";
}

TEST_F(WriterTest, chunkSizeStatsEmptyWhenChunkingNotTriggered) {
  const auto type = velox::ROW({{"c0", velox::BIGINT()}});

  nimble::WriterOptions options{
      .flushPolicyFactory = []() -> std::unique_ptr<nimble::FlushPolicy> {
        return std::make_unique<nimble::StripeRawSizeFlushPolicy>(256ULL << 20);
      },
      .enableChunking = false,
  };

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(options));

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  writer.write(vectorMaker.rowVector(
      {"c0"}, {vectorMaker.flatVector<int64_t>({1, 2, 3, 4, 5})}));
  writer.close();

  EXPECT_EQ(
      nimble::runtimeStat(
          writer.runtimeStats(), nimble::Writer::RuntimeStats::kChunkSizeBytes)
          .count,
      0);
}

TEST_F(WriterTest, runtimeStatsPublishesEveryCounter) {
  using NimbleStats = nimble::Writer::RuntimeStats;
  const auto type = velox::ROW({{"c0", velox::BIGINT()}});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, nimble::WriterOptions{});

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  writer.write(vectorMaker.rowVector(
      {"c0"}, {vectorMaker.flatVector<int64_t>({1, 2, 3, 4, 5})}));
  writer.close();

  // The connector collects the counters after close, so they have to survive
  // it. Spelling the keys out here rather than reusing the constants keeps the
  // published names themselves under test.
  const auto metrics = writer.runtimeStats();
  EXPECT_THAT(
      metrics,
      ::testing::UnorderedElementsAre(
          ::testing::Key("nimble.writtenBytes"),
          ::testing::Key("nimble.inputBytes"),
          ::testing::Key("nimble.writeCpuNanos"),
          ::testing::Key("nimble.writeWallNanos"),
          ::testing::Key("nimble.ingestionCpuNanos"),
          ::testing::Key("nimble.ingestionWallNanos"),
          ::testing::Key("nimble.encodingCpuNanos"),
          ::testing::Key("nimble.encodingWallNanos"),
          ::testing::Key("nimble.encodingSelectionCpuNanos"),
          ::testing::Key("nimble.rowsPerStripe"),
          ::testing::Key("nimble.chunkSizeBytes"),
          ::testing::Key("nimble.duplicateStreamCount"),
          ::testing::Key("nimble.duplicateStreamBytes")));

  // Each constant has to resolve to one of those keys, so renaming a constant
  // without renaming what the writer publishes fails here.
  EXPECT_EQ(nimble::runtimeStat(metrics, NimbleStats::kRowsPerStripe).count, 1);
  EXPECT_GT(nimble::runtimeStat(metrics, NimbleStats::kWrittenBytes).sum, 0);
  EXPECT_GT(nimble::runtimeStat(metrics, NimbleStats::kInputBytes).sum, 0);
  EXPECT_EQ(nimble::runtimeStat(metrics, NimbleStats::kRowsPerStripe).sum, 5);

  // An unknown key reads as a zeroed metric rather than throwing, which is what
  // lets a consumer read a map merged from writers that never published it.
  EXPECT_EQ(nimble::runtimeStat(metrics, "nimble.notAKey").count, 0);

  // The connector merges these maps across the writers of one sink and rejects
  // a key whose unit disagrees, so units are part of the contract.
  EXPECT_EQ(
      nimble::runtimeStat(metrics, NimbleStats::kWrittenBytes).unit,
      velox::RuntimeCounter::Unit::kBytes);
  EXPECT_EQ(
      nimble::runtimeStat(metrics, NimbleStats::kEncodingCpuNanos).unit,
      velox::RuntimeCounter::Unit::kNanos);

  // Column statistics have no RuntimeMetric form, so they are collected by
  // default but read through their own accessor rather than the map.
  EXPECT_FALSE(writer.columnStats().empty());
}

TEST_F(WriterTest, cachedEncodingLayoutAcrossChunks) {
  // Isolation test: a single stripe holding multiple chunks, so a failure here
  // points at the chunk-boundary replay path specifically. The default
  // multi-stripe-and-multi-chunk coverage lives in the other cache tests.
  constexpr int kChunkCount = 5;
  constexpr int kRowsPerChunk = 1000;
  // Fixed seed so the cached and control writers generate identical data and
  // the test is reproducible across runs.
  constexpr uint32_t kSeed = 0xC0FFEE;
  auto chunkData = makeDivergentInt64Chunks(kChunkCount, kRowsPerChunk, kSeed);

  // One chunk per batch (chunkLambda true) with no stripe flush, so the file is
  // a single stripe holding kChunkCount chunks.
  auto makeOptions = [](bool enableEncodingSelectionCache) {
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = []() {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /*flushLambda=*/[](auto&) { return false; },
          /*chunkLambda=*/[](auto&) { return true; });
    };
    return options;
  };

  // Precondition: without caching, the divergent data selects different
  // encodings across chunks. If this regresses, the cached assertion below
  // would be vacuous, so fail hard here.
  auto uncached = writeAndCaptureChunkLayouts(
      chunkData,
      makeOptions(/*enableEncodingSelectionCache=*/false),
      /*expectedStripeCount=*/1);
  ASSERT_EQ(uncached.size(), static_cast<size_t>(kChunkCount));
  ASSERT_NE(uncached.front().encodingType(), uncached[1].encodingType());

  // With caching, every chunk replays the first chunk's encoding.
  auto cached = writeAndCaptureChunkLayouts(
      chunkData,
      makeOptions(/*enableEncodingSelectionCache=*/true),
      /*expectedStripeCount=*/1);
  ASSERT_EQ(cached.size(), static_cast<size_t>(kChunkCount));
  for (const auto& layout : cached) {
    EXPECT_EQ(layout.encodingType(), cached.front().encodingType());
  }
}

TEST_F(WriterTest, cachedEncodingLayoutMultiType) {
  // Applies the divergent-data replay invariant to every column of a multi-type
  // schema (BIGINT, VARCHAR, REAL, BIGINT) simultaneously, across both chunk
  // and stripe boundaries. All columns are non-nullable so the captured
  // top-level EncodingType of each column's stream is directly comparable
  // across chunks (a nullable column would wrap the data encoding in a Nullable
  // encoding).
  auto type = velox::ROW({
      {"c0", velox::BIGINT()},
      {"c1", velox::VARCHAR()},
      {"c2", velox::REAL()},
      {"c3", velox::BIGINT()},
  });
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  constexpr int kColumnCount = 4;
  constexpr int kChunkCount = 8;
  constexpr int kChunksPerStripe = 3;
  constexpr uint32_t kExpectedStripeCount = 3; // ceil(8 / 3)
  constexpr int kRowsPerChunk = 1000;
  constexpr uint32_t kSeed = 0xC0FFEE;

  // Returns, per column, the top-level EncodingType captured for each chunk of
  // that column's stream. Outer index is column, inner index is chunk.
  auto writeAndCaptureChunkEncodings = [&](bool enableEncodingSelectionCache)
      -> std::vector<std::vector<nimble::EncodingType>> {
    // Re-seeded identically each call. For every column, chunk 0 holds
    // high-entropy random values (a flat layout that re-encodes anything on
    // replay) and each later chunk repeats a single seeded value (a constant
    // layout), guaranteeing divergent encodings without caching.
    std::mt19937 rng{kSeed};
    std::vector<std::vector<int64_t>> c0Data(kChunkCount);
    std::vector<std::vector<std::string>> c1Data(kChunkCount);
    std::vector<std::vector<float>> c2Data(kChunkCount);
    std::vector<std::vector<int64_t>> c3Data(kChunkCount);
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      if (chunk == 0) {
        for (int row = 0; row < kRowsPerChunk; ++row) {
          c0Data[chunk].push_back(static_cast<int64_t>(rng()));
          c1Data[chunk].push_back(fmt::format("s_{}", rng()));
          c2Data[chunk].push_back(static_cast<float>(rng()));
          c3Data[chunk].push_back(static_cast<int64_t>(rng()));
        }
      } else {
        c0Data[chunk].assign(kRowsPerChunk, static_cast<int64_t>(rng()));
        c1Data[chunk].assign(kRowsPerChunk, fmt::format("s_{}", rng()));
        c2Data[chunk].assign(kRowsPerChunk, static_cast<float>(rng()));
        c3Data[chunk].assign(kRowsPerChunk, static_cast<int64_t>(rng()));
      }
    }

    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory =
        chunkAndStripeFlushPolicyFactory(kChunksPerStripe);

    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      std::vector<velox::StringView> c1Views;
      c1Views.reserve(kRowsPerChunk);
      for (const auto& value : c1Data[chunk]) {
        c1Views.push_back(velox::StringView{value});
      }
      writer.write(vectorMaker.rowVector(
          {"c0", "c1", "c2", "c3"},
          {vectorMaker.flatVector<int64_t>(c0Data[chunk]),
           vectorMaker.flatVector<velox::StringView>(c1Views),
           vectorMaker.flatVector<float>(c2Data[chunk]),
           vectorMaker.flatVector<int64_t>(c3Data[chunk])}));
    }
    writer.close();

    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);

    // Verify every column round-trips: replaying a cached layout onto divergent
    // data must not corrupt values.
    std::vector<int64_t> actualC0;
    std::vector<std::string> actualC1;
    std::vector<float> actualC2;
    std::vector<int64_t> actualC3;
    {
      nimble::BatchReader reader(readFile.get(), *leafPool_);
      velox::VectorPtr result;
      while (reader.next(kRowsPerChunk, result)) {
        auto* row = result->as<velox::RowVector>();
        auto* c0 = row->childAt(0)->asFlatVector<int64_t>();
        auto* c1 = row->childAt(1)->asFlatVector<velox::StringView>();
        auto* c2 = row->childAt(2)->asFlatVector<float>();
        auto* c3 = row->childAt(3)->asFlatVector<int64_t>();
        for (auto i = 0; i < result->size(); ++i) {
          actualC0.push_back(c0->valueAt(i));
          actualC1.emplace_back(c1->valueAt(i));
          actualC2.push_back(c2->valueAt(i));
          actualC3.push_back(c3->valueAt(i));
        }
      }
    }
    std::vector<int64_t> expectedC0;
    std::vector<std::string> expectedC1;
    std::vector<float> expectedC2;
    std::vector<int64_t> expectedC3;
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      expectedC0.insert(
          expectedC0.end(), c0Data[chunk].begin(), c0Data[chunk].end());
      expectedC1.insert(
          expectedC1.end(), c1Data[chunk].begin(), c1Data[chunk].end());
      expectedC2.insert(
          expectedC2.end(), c2Data[chunk].begin(), c2Data[chunk].end());
      expectedC3.insert(
          expectedC3.end(), c3Data[chunk].begin(), c3Data[chunk].end());
    }
    EXPECT_EQ(actualC0, expectedC0);
    EXPECT_EQ(actualC1, expectedC1);
    EXPECT_EQ(actualC2, expectedC2);
    EXPECT_EQ(actualC3, expectedC3);

    std::vector<std::vector<nimble::EncodingType>> columnChunkEncodings(
        kColumnCount);
    for (int column = 0; column < kColumnCount; ++column) {
      for (const auto& layout :
           captureColumnChunkLayouts(readFile, column, kExpectedStripeCount)) {
        columnChunkEncodings[column].push_back(layout.encodingType());
      }
    }
    return columnChunkEncodings;
  };

  // Precondition: without caching, every column's divergent data selects
  // different encodings across chunks, so the cached assertions are not
  // vacuous.
  auto uncachedEncodings =
      writeAndCaptureChunkEncodings(/*enableEncodingSelectionCache=*/false);
  ASSERT_EQ(uncachedEncodings.size(), static_cast<size_t>(kColumnCount));
  for (int column = 0; column < kColumnCount; ++column) {
    ASSERT_EQ(
        uncachedEncodings[column].size(), static_cast<size_t>(kChunkCount));
    ASSERT_NE(uncachedEncodings[column].front(), uncachedEncodings[column][1]);
  }

  // With caching, every chunk of every column replays that column's first
  // chunk encoding.
  auto cachedEncodings =
      writeAndCaptureChunkEncodings(/*enableEncodingSelectionCache=*/true);
  ASSERT_EQ(cachedEncodings.size(), static_cast<size_t>(kColumnCount));
  for (int column = 0; column < kColumnCount; ++column) {
    ASSERT_EQ(cachedEncodings[column].size(), static_cast<size_t>(kChunkCount));
    for (auto encoding : cachedEncodings[column]) {
      EXPECT_EQ(encoding, cachedEncodings[column].front());
    }
  }
}

TEST_F(WriterTest, cachedEncodingLayoutNullableEncoding) {
  // Reviewer @783: when a chunk has nulls the writer wraps its data encoding in
  // a Nullable encoding (EncodingType::Nullable), and the cache stores only the
  // data encoding -- EncodingLayoutCapture::capture() strips the Nullable
  // wrapper. So a chunk whose nullability differs from the chunk that populated
  // the cache must (a) replay the cached data encoding and (b) re-apply the
  // Nullable wrapper based on its own nulls, decided per chunk by encode<T> in
  // Writer.cpp, not baked into the cache.
  //
  // capture() always strips the Nullable wrapper, so the wrapper is observed by
  // reading byte 0 (the EncodingType) of the raw encoded chunk, while the
  // stripped data encoding comes from capture().
  auto type = velox::ROW({{"c0", velox::BIGINT()}});
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  constexpr int kChunkCount = 4;
  constexpr int kRowsPerChunk = 1000;
  constexpr uint32_t kSeed = 0xF00D;

  // Chunk 0: high-entropy random values (a flat data layout that replays onto
  // any data) with nulls. Later chunks: a single repeated constant (a Constant
  // data layout when freshly selected), alternating nullability. So without
  // caching the data encodings diverge across chunks, and the null-bearing
  // chunks are Nullable-wrapped while the others are bare.
  auto makeChunkData = [&] {
    std::mt19937 rng{kSeed};
    std::vector<std::vector<std::optional<int64_t>>> data(kChunkCount);
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      const bool withNulls = (chunk % 2 == 0); // chunks 0, 2 have nulls
      for (int row = 0; row < kRowsPerChunk; ++row) {
        if (withNulls && row % 5 == 0) {
          data[chunk].push_back(std::nullopt);
        } else {
          data[chunk].push_back(
              chunk == 0 ? static_cast<int64_t>(rng()) : int64_t{42});
        }
      }
    }
    return data;
  };

  struct ChunkEncoding {
    // The on-disk wrapper: byte 0 (EncodingType) of the raw chunk. Nullable for
    // a chunk with nulls, otherwise the bare data encoding.
    nimble::EncodingType wrapper;
    // The data encoding, with any Nullable wrapper stripped by capture().
    nimble::EncodingLayout dataLayout;
  };

  auto writeAndCapture =
      [&](bool enableEncodingSelectionCache) -> std::vector<ChunkEncoding> {
    const auto data = makeChunkData();
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = [] {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /*flushLambda=*/[](auto&) { return false; },
          /*chunkLambda=*/[](auto&) { return true; });
    };

    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      writer.write(vectorMaker.rowVector(
          {"c0"}, {vectorMaker.flatVectorNullable<int64_t>(data[chunk])}));
    }
    writer.close();

    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);

    // Replaying a cached layout onto divergent data/nullability must not
    // corrupt values or nulls.
    std::vector<std::optional<int64_t>> actual;
    {
      nimble::BatchReader reader(readFile.get(), *leafPool_);
      velox::VectorPtr result;
      while (reader.next(kRowsPerChunk, result)) {
        auto* c0 =
            result->as<velox::RowVector>()->childAt(0)->asFlatVector<int64_t>();
        for (velox::vector_size_t i = 0; i < result->size(); ++i) {
          actual.push_back(
              c0->isNullAt(i) ? std::nullopt
                              : std::optional<int64_t>{c0->valueAt(i)});
        }
      }
    }
    std::vector<std::optional<int64_t>> expected;
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      expected.insert(expected.end(), data[chunk].begin(), data[chunk].end());
    }
    EXPECT_EQ(actual, expected);

    // Capture, per chunk, both the raw on-disk wrapper (byte 0) and the
    // Nullable-stripped data encoding.
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    auto section =
        tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
    NIMBLE_CHECK(section.has_value(), "Schema not found.");
    auto schema =
        nimble::SchemaDeserializer::deserialize(section->content().data());
    const auto offset =
        schema->asRow().childAt(0)->asScalar().scalarDescriptor().offset();

    EXPECT_EQ(tablet->stripeCount(), 1U);
    std::vector<ChunkEncoding> chunkEncodings;
    auto streams = tablet->load(
        tablet->stripeIdentifier(0), std::vector<uint32_t>{offset});
    nimble::InMemoryChunkedStream chunkedStream{
        *leafPool_, std::move(streams[0])};
    while (chunkedStream.hasNext()) {
      const auto rawChunk = chunkedStream.nextChunk();
      auto dataLayout = nimble::EncodingLayoutCapture::capture(
          rawChunk, nimble::Encoding::Options{});
      const auto wrapper =
          static_cast<nimble::EncodingType>(static_cast<uint8_t>(rawChunk[0]));
      chunkEncodings.push_back(ChunkEncoding{wrapper, std::move(dataLayout)});
    }
    return chunkEncodings;
  };

  // Precondition: without caching the data encodings diverge across chunks (so
  // the cached data-replay assertion is not vacuous), and the null-bearing
  // chunks are Nullable-wrapped on disk while the others are bare.
  const auto uncached = writeAndCapture(/*enableEncodingSelectionCache=*/false);
  ASSERT_EQ(uncached.size(), static_cast<size_t>(kChunkCount));
  EXPECT_NE(
      uncached[0].dataLayout.encodingType(),
      uncached[1].dataLayout.encodingType());
  EXPECT_EQ(uncached[0].wrapper, nimble::EncodingType::Nullable);
  EXPECT_NE(uncached[1].wrapper, nimble::EncodingType::Nullable);

  // With caching, every chunk replays chunk 0's data encoding regardless of its
  // own nullability...
  const auto cached = writeAndCapture(/*enableEncodingSelectionCache=*/true);
  ASSERT_EQ(cached.size(), static_cast<size_t>(kChunkCount));
  for (const auto& chunk : cached) {
    EXPECT_EQ(
        chunk.dataLayout.encodingType(), cached[0].dataLayout.encodingType());
  }

  // ...while the on-disk Nullable wrapper is still re-applied per chunk from
  // its own nulls (decided by encode<T>, not baked into the cached data
  // layout): a null-bearing chunk wraps the replayed data encoding in a
  // Nullable, a null-free chunk uses it bare.
  EXPECT_EQ(cached[0].wrapper, nimble::EncodingType::Nullable);
  EXPECT_NE(cached[1].wrapper, nimble::EncodingType::Nullable);
  EXPECT_EQ(cached[2].wrapper, nimble::EncodingType::Nullable);
  EXPECT_NE(cached[3].wrapper, nimble::EncodingType::Nullable);
}

TEST_F(WriterTest, cachedEncodingLayoutWithEncodingExecutor) {
  // Same divergent-data replay invariant across chunks and stripes, but with a
  // parallel encoding executor wired in to exercise the encoding selection
  // cache on encoding-executor pool threads.
  constexpr int kChunkCount = 8;
  constexpr int kChunksPerStripe = 3;
  constexpr uint32_t kExpectedStripeCount = 3; // ceil(8 / 3)
  constexpr int kRowsPerChunk = 1000;
  constexpr uint32_t kSeed = 0xC0FFEE;
  auto chunkData = makeDivergentInt64Chunks(kChunkCount, kRowsPerChunk, kSeed);

  folly::CPUThreadPoolExecutor executor{4};
  auto makeOptions = [&](bool enableEncodingSelectionCache) {
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.encodingExecutor = folly::getKeepAliveToken(executor);
    options.flushPolicyFactory =
        chunkAndStripeFlushPolicyFactory(kChunksPerStripe);
    return options;
  };

  // Precondition: without caching, the divergent data selects different
  // encodings across chunks, so the cached assertion is not vacuous.
  auto uncached = writeAndCaptureChunkLayouts(
      chunkData,
      makeOptions(/*enableEncodingSelectionCache=*/false),
      kExpectedStripeCount);
  ASSERT_EQ(uncached.size(), static_cast<size_t>(kChunkCount));
  ASSERT_NE(uncached.front().encodingType(), uncached[1].encodingType());

  // With caching, every chunk across all stripes replays the first chunk's
  // encoding.
  auto cached = writeAndCaptureChunkLayouts(
      chunkData,
      makeOptions(/*enableEncodingSelectionCache=*/true),
      kExpectedStripeCount);
  ASSERT_EQ(cached.size(), static_cast<size_t>(kChunkCount));
  for (const auto& layout : cached) {
    EXPECT_EQ(layout.encodingType(), cached.front().encodingType());
  }
}

TEST_F(WriterTest, cachedEncodingLayoutReplay) {
  // The default cache expectation: divergent data written as organically-formed
  // stripes that each hold multiple chunks (one chunk per batch, a stripe
  // closed after every kChunksPerStripe batches). The cached layout from the
  // very first chunk of the file must be replayed onto every later chunk,
  // across all chunk and stripe boundaries.
  constexpr int kChunkCount = 20;
  constexpr int kRowsPerChunk = 1000;
  // Close a stripe after every 6 batches: stripes close after batches 6, 12, 18
  // and the trailing 2 batches form the final stripe at close, giving 4 stripes
  // (sizes 6, 6, 6, 2), each with multiple chunks.
  constexpr int kChunksPerStripe = 6;
  constexpr uint32_t kSeed = 0xC0FFEE;
  constexpr uint32_t kExpectedStripeCount = 4;
  auto chunkData = makeDivergentInt64Chunks(kChunkCount, kRowsPerChunk, kSeed);

  auto makeOptions = [](bool enableEncodingSelectionCache) {
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory =
        chunkAndStripeFlushPolicyFactory(kChunksPerStripe);
    return options;
  };

  // Precondition: without caching, the divergent data selects different
  // encodings across chunks, so the cached assertion is not vacuous.
  auto uncached = writeAndCaptureChunkLayouts(
      chunkData,
      makeOptions(/*enableEncodingSelectionCache=*/false),
      kExpectedStripeCount);
  ASSERT_EQ(uncached.size(), static_cast<size_t>(kChunkCount));
  ASSERT_NE(uncached.front().encodingType(), uncached[1].encodingType());

  // With caching, every chunk across all stripes replays the first chunk's
  // encoding.
  auto cached = writeAndCaptureChunkLayouts(
      chunkData,
      makeOptions(/*enableEncodingSelectionCache=*/true),
      kExpectedStripeCount);
  ASSERT_EQ(cached.size(), static_cast<size_t>(kChunkCount));
  for (const auto& layout : cached) {
    EXPECT_EQ(layout.encodingType(), cached.front().encodingType());
  }
}

TEST_F(WriterTest, cachedEncodingLayoutFuzz) {
  // Seeds that previously triggered a failure are replayed first as fixed
  // regressions; afterwards a single random seed is drawn per process. Drive
  // repetition with `--stress-runs N`, never an in-process loop.
  static constexpr std::array<uint32_t, 0> kRegressionSeeds{};

  constexpr int kChunkCount = 8;
  constexpr int kRowsPerChunk = 1000;

  // One chunk per batch, single stripe.
  auto makeOptions = [](bool enableEncodingSelectionCache) {
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = []() {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /*flushLambda=*/[](auto&) { return false; },
          /*chunkLambda=*/[](auto&) { return true; });
    };
    return options;
  };

  auto runOneSeed = [&](uint32_t seed) {
    LOG(INFO) << "cachedEncodingLayoutFuzz seed: " << seed;
    std::mt19937 rng{seed};

    // Generate random per-chunk data with varied shapes: each chunk is either
    // constant, low-cardinality, or high-cardinality random. This mixes data
    // that selection encodes very differently per chunk.
    std::vector<std::vector<int64_t>> chunkData(kChunkCount);
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      const int shape = std::uniform_int_distribution<int>(0, 2)(rng);
      chunkData[chunk].reserve(kRowsPerChunk);
      if (shape == 0) {
        // Constant.
        const int64_t value = static_cast<int64_t>(rng());
        chunkData[chunk].assign(kRowsPerChunk, value);
      } else if (shape == 1) {
        // Low cardinality (small set of repeated values).
        const int cardinality = std::uniform_int_distribution<int>(2, 8)(rng);
        std::vector<int64_t> alphabet(cardinality);
        for (auto& value : alphabet) {
          value = static_cast<int64_t>(rng());
        }
        for (int row = 0; row < kRowsPerChunk; ++row) {
          chunkData[chunk].push_back(
              alphabet[std::uniform_int_distribution<int>(
                  0, cardinality - 1)(rng)]);
        }
      } else {
        // High cardinality / random.
        for (int row = 0; row < kRowsPerChunk; ++row) {
          chunkData[chunk].push_back(static_cast<int64_t>(rng()));
        }
      }
    }

    // The cached write must not throw for valid input.
    std::vector<nimble::EncodingLayout> controlLayouts;
    std::vector<nimble::EncodingLayout> cachedLayouts;
    ASSERT_NO_THROW(
        controlLayouts = writeAndCaptureChunkLayouts(
            chunkData,
            makeOptions(/*enableEncodingSelectionCache=*/false),
            /*expectedStripeCount=*/1));
    ASSERT_NO_THROW(
        cachedLayouts = writeAndCaptureChunkLayouts(
            chunkData,
            makeOptions(/*enableEncodingSelectionCache=*/true),
            /*expectedStripeCount=*/1));
    ASSERT_EQ(controlLayouts.size(), static_cast<size_t>(kChunkCount));
    ASSERT_EQ(cachedLayouts.size(), static_cast<size_t>(kChunkCount));

    // The encoding selection cache is best-effort: each chunk either
    // successfully replays chunk 0's cached encoding, or the cached encoding
    // was incompatible with the chunk's data, raising IncompatibleEncoding
    // which the writer catches and falls back to a fresh selection — the same
    // encoding the uncached writer chose for that chunk (see
    // encodeWithFallback).
    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      EXPECT_TRUE(
          cachedLayouts[chunk].encodingType() ==
              cachedLayouts.front().encodingType() ||
          cachedLayouts[chunk].encodingType() ==
              controlLayouts[chunk].encodingType())
          << "seed=" << seed << " chunk=" << chunk
          << " cached=" << static_cast<int>(cachedLayouts[chunk].encodingType())
          << " cached0="
          << static_cast<int>(cachedLayouts.front().encodingType())
          << " control="
          << static_cast<int>(controlLayouts[chunk].encodingType());
    }
  };

  for (uint32_t seed : kRegressionSeeds) {
    runOneSeed(seed);
  }
  const uint32_t seed = FLAGS_writer_tests_seed > 0 ? FLAGS_writer_tests_seed
                                                    : folly::Random::rand32();
  runOneSeed(seed);
}

TEST_F(WriterTest, cachedEncodingLayoutIncompatibleFallback) {
  // Exercises the best-effort fallback when a cached encoding is incompatible
  // with a later chunk's data (see encodeWithFallback). Chunk 0 is fully
  // constant -> Constant, which gets cached. Chunk 1 is mainly constant (not
  // constant); replaying the cached Constant onto it raises
  // IncompatibleEncoding (ConstantEncoding requires constant data), which the
  // writer catches and falls back to a fresh selection for that chunk. The
  // fallback does not re-cache (it only caches when no layout is cached yet),
  // so chunks 2-4 are fully constant again and still replay the original cached
  // Constant.
  constexpr int kRowsPerChunk = 1000;
  constexpr int64_t kDominantValue = 7;
  constexpr uint32_t kSeed = 0xC0FFEE;

  // Fully constant: selection picks Constant, which gets cached; reused for
  // chunks 0, 2, 3, 4.
  const std::vector<int64_t> fullyConstant(kRowsPerChunk, kDominantValue);
  // Mainly constant: the dominant value with ~1% distinct exceptions, generated
  // once with a fixed seed. Replaying the cached Constant onto this
  // non-constant chunk trips IncompatibleEncoding.
  std::mt19937 rng{kSeed};
  std::vector<int64_t> mainlyConstant(kRowsPerChunk, kDominantValue);
  for (int i = 0; i < kRowsPerChunk / 100; ++i) {
    mainlyConstant[std::uniform_int_distribution<int>(
        0, kRowsPerChunk - 1)(rng)] = static_cast<int64_t>(rng());
  }
  const std::vector<std::vector<int64_t>> chunkData = {
      fullyConstant,
      mainlyConstant,
      fullyConstant,
      fullyConstant,
      fullyConstant};

  auto makeOptions = [](bool enableEncodingSelectionCache) {
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = []() {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /*flushLambda=*/[](auto&) { return false; },
          /*chunkLambda=*/[](auto&) { return true; });
    };
    return options;
  };

  // The incompatible second chunk must be handled gracefully, not thrown.
  std::vector<nimble::EncodingLayout> controlLayouts;
  std::vector<nimble::EncodingLayout> cachedLayouts;
  ASSERT_NO_THROW(
      controlLayouts = writeAndCaptureChunkLayouts(
          chunkData,
          makeOptions(/*enableEncodingSelectionCache=*/false),
          /*expectedStripeCount=*/1));
  ASSERT_NO_THROW(
      cachedLayouts = writeAndCaptureChunkLayouts(
          chunkData,
          makeOptions(/*enableEncodingSelectionCache=*/true),
          /*expectedStripeCount=*/1));
  ASSERT_EQ(controlLayouts.size(), chunkData.size());
  ASSERT_EQ(cachedLayouts.size(), chunkData.size());

  // Sanity: freshly selected, the mainly-constant and fully-constant chunks
  // pick different encodings.
  ASSERT_NE(
      controlLayouts.front().encodingType(), controlLayouts[1].encodingType());

  // Chunk 1 (mainly constant) is incompatible with the cached Constant, so it
  // falls back to the same fresh encoding the uncached writer chose.
  EXPECT_NE(
      cachedLayouts[1].encodingType(), cachedLayouts.front().encodingType());
  EXPECT_EQ(cachedLayouts[1].encodingType(), controlLayouts[1].encodingType());

  // The incompatible chunk does not disrupt the cache: the fully-constant
  // chunks all replay the first chunk's cached encoding.
  EXPECT_EQ(
      cachedLayouts.front().encodingType(), cachedLayouts[2].encodingType());
  EXPECT_EQ(
      cachedLayouts.front().encodingType(), cachedLayouts[3].encodingType());
  EXPECT_EQ(
      cachedLayouts.front().encodingType(), cachedLayouts[4].encodingType());
}

TEST_F(WriterTest, cachedEncodingLayoutNestedDictionaryIncompatibleFallback) {
  // Exercises the best-effort fallback when a *nested* cached encoding is
  // incompatible with a later chunk. Chunk 0 is mainly constant with a few
  // distinct, large exception values, so selection picks MainlyConstant whose
  // OtherValues sub-stream is a Dictionary; this layout is cached. Chunk 1 is
  // fully constant, so on replay the MainlyConstant's OtherValues stream is
  // empty -- the nested Dictionary replay then has 0 rows and raises the
  // catchable IncompatibleEncoding (this is the exact shape that used to
  // DCHECK-abort the process before the empty-Dictionary replay was made
  // catchable). The writer catches it and falls back to a fresh selection for
  // that chunk. Chunks 2-4 repeat chunk 0's data and replay the cached
  // MainlyConstant successfully.
  constexpr int kRowsPerChunk = 1000;
  constexpr int64_t kDominantValue = 7;
  constexpr uint32_t kSeed = 0xC0FFEE;

  // Mainly constant: kDominantValue for most rows, with a small number of rows
  // set to one of a few large, distinct exception values. Few distinct + large
  // magnitude biases the OtherValues sub-stream toward a Dictionary encoding.
  constexpr std::array<int64_t, 3> kExceptions = {
      int64_t{1} << 40, int64_t{2} << 40, int64_t{3} << 40};
  std::mt19937 rng{kSeed};
  std::vector<int64_t> mainlyConstant(kRowsPerChunk, kDominantValue);
  for (int i = 0; i < kRowsPerChunk / 20; ++i) {
    mainlyConstant[std::uniform_int_distribution<int>(
        0, kRowsPerChunk - 1)(rng)] = kExceptions[i % kExceptions.size()];
  }
  // Fully constant: replaying MainlyConstant onto this empties OtherValues.
  const std::vector<int64_t> fullyConstant(kRowsPerChunk, kDominantValue);
  const std::vector<std::vector<int64_t>> chunkData = {
      mainlyConstant,
      fullyConstant,
      mainlyConstant,
      mainlyConstant,
      mainlyConstant};

  auto makeOptions = [](bool enableEncodingSelectionCache) {
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = []() {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /*flushLambda=*/[](auto&) { return false; },
          /*chunkLambda=*/[](auto&) { return true; });
    };
    return options;
  };

  std::vector<nimble::EncodingLayout> controlLayouts;
  std::vector<nimble::EncodingLayout> cachedLayouts;
  // The fully-constant chunk that empties the nested Dictionary must be handled
  // gracefully (catchable IncompatibleEncoding), not abort the process.
  ASSERT_NO_THROW(
      controlLayouts = writeAndCaptureChunkLayouts(
          chunkData,
          makeOptions(/*enableEncodingSelectionCache=*/false),
          /*expectedStripeCount=*/1));
  ASSERT_NO_THROW(
      cachedLayouts = writeAndCaptureChunkLayouts(
          chunkData,
          makeOptions(/*enableEncodingSelectionCache=*/true),
          /*expectedStripeCount=*/1));
  ASSERT_EQ(controlLayouts.size(), chunkData.size());
  ASSERT_EQ(cachedLayouts.size(), chunkData.size());

  // Precondition: chunk 0 is MainlyConstant with a nested Dictionary
  // OtherValues -- the exact shape that empties on the fully-constant replay.
  // Without this the test would not exercise the nested-Dictionary path.
  ASSERT_EQ(
      controlLayouts.front().encodingType(),
      nimble::EncodingType::MainlyConstant);
  const auto& otherValues = controlLayouts.front().child(
      nimble::EncodingIdentifiers::MainlyConstant::OtherValues);
  ASSERT_TRUE(otherValues.has_value());
  ASSERT_EQ(otherValues->encodingType(), nimble::EncodingType::Dictionary);

  // Sanity: freshly selected, the mainly-constant and fully-constant chunks
  // pick different top-level encodings.
  ASSERT_NE(
      controlLayouts.front().encodingType(), controlLayouts[1].encodingType());

  // Chunk 1 (fully constant) empties the cached MainlyConstant's nested
  // Dictionary on replay -> incompatible -> falls back to the same fresh
  // encoding the uncached writer chose.
  EXPECT_NE(
      cachedLayouts[1].encodingType(), cachedLayouts.front().encodingType());
  EXPECT_EQ(cachedLayouts[1].encodingType(), controlLayouts[1].encodingType());

  // The incompatible chunk does not disrupt the cache: the mainly-constant
  // chunks all replay the first chunk's cached MainlyConstant.
  EXPECT_EQ(
      cachedLayouts.front().encodingType(), cachedLayouts[2].encodingType());
  EXPECT_EQ(
      cachedLayouts.front().encodingType(), cachedLayouts[3].encodingType());
  EXPECT_EQ(
      cachedLayouts.front().encodingType(), cachedLayouts[4].encodingType());
}

TEST_F(WriterTest, cachedEncodingLayoutNestedDictionaryReplay) {
  // Strict nested-encoding coverage: the cache must replay the full nested
  // encoding tree, not just the top-level type. Chunk 0 is many distinct,
  // randomly interleaved large values -> Dictionary(Alphabet, Indices). Later
  // chunks arrange the same values in contiguous runs, which a fresh selection
  // encodes differently (a run-length family), but the cached Dictionary
  // replays onto them compatibly (16-entry alphabet, never empty). Every cached
  // chunk's full layout tree — including the nested Alphabet/Indices
  // sub-encodings — must equal chunk 0's.
  constexpr int kRowsPerChunk = 1024;
  constexpr int kCardinality = 16;
  constexpr int kRunLength = kRowsPerChunk / kCardinality;
  constexpr uint32_t kSeed = 0xD1C7;

  std::mt19937 rng{kSeed};
  // Large, distinct values so Dictionary (16-entry alphabet + 4-bit indices)
  // beats a flat 64-bit FixedBitWidth.
  std::vector<int64_t> alphabet(kCardinality);
  for (auto& value : alphabet) {
    value = (static_cast<int64_t>(rng()) << 20) | (int64_t{1} << 40);
  }

  // Chunk 0: randomly interleaved (no runs) -> Dictionary.
  std::vector<int64_t> interleaved(kRowsPerChunk);
  for (auto& value : interleaved) {
    value =
        alphabet[std::uniform_int_distribution<int>(0, kCardinality - 1)(rng)];
  }
  // Later chunks: the same values in contiguous runs -> a run-length family
  // when freshly selected, but Dictionary-replay compatible.
  auto makeRuns = [&](int shift) {
    std::vector<int64_t> runs(kRowsPerChunk);
    for (int i = 0; i < kRowsPerChunk; ++i) {
      runs[i] = alphabet[((i / kRunLength) + shift) % kCardinality];
    }
    return runs;
  };
  const std::vector<std::vector<int64_t>> chunkData = {
      interleaved, makeRuns(0), makeRuns(3)};

  auto makeOptions = [](bool enableEncodingSelectionCache) {
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = []() {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /*flushLambda=*/[](auto&) { return false; },
          /*chunkLambda=*/[](auto&) { return true; });
    };
    return options;
  };

  const auto control = writeAndCaptureChunkLayouts(
      chunkData,
      makeOptions(/*enableEncodingSelectionCache=*/false),
      /*expectedStripeCount=*/1);
  const auto cached = writeAndCaptureChunkLayouts(
      chunkData,
      makeOptions(/*enableEncodingSelectionCache=*/true),
      /*expectedStripeCount=*/1);
  ASSERT_EQ(control.size(), chunkData.size());
  ASSERT_EQ(cached.size(), chunkData.size());

  // Precondition: chunk 0 is a Dictionary with populated Alphabet + Indices
  // sub-encodings (so nested replay is genuinely exercised)...
  ASSERT_EQ(control.front().encodingType(), nimble::EncodingType::Dictionary);
  ASSERT_TRUE(control.front()
                  .child(nimble::EncodingIdentifiers::Dictionary::Alphabet)
                  .has_value());
  ASSERT_TRUE(control.front()
                  .child(nimble::EncodingIdentifiers::Dictionary::Indices)
                  .has_value());
  // ...and a fresh selection of a later (run-shaped) chunk diverges from it, so
  // the replay assertion below is not vacuous.
  ASSERT_FALSE(encodingLayoutsEqual(control.front(), control[1]));

  // Cached chunk 0 is itself a fresh selection (it populates the cache), so it
  // matches the uncached chunk 0.
  EXPECT_TRUE(encodingLayoutsEqual(cached.front(), control.front()));

  // Core assertion: every cached chunk replays chunk 0's full nested tree
  // (Dictionary + Alphabet/Indices), not merely the top-level Dictionary type.
  for (const auto& layout : cached) {
    EXPECT_TRUE(encodingLayoutsEqual(layout, cached.front()));
  }
}

TEST_F(WriterTest, cachedEncodingLayoutAllDataTypes) {
  // Interface-level coverage across chunk and stripe boundaries: caching must
  // round-trip correctly and produce sane encodings for every scalar column
  // type, not just BIGINT. Chunk 0 is fuzzed (with nulls); later chunks are
  // constant. Caching is best-effort, so per column each cached chunk either
  // replays chunk 0's encoding or falls back to the same fresh encoding the
  // uncached writer chose; the round-trip (verified inside the helper) is the
  // hard correctness guarantee.
  const auto type = allScalarTypesRow();
  constexpr int kChunkCount = 8;
  constexpr int kChunksPerStripe = 3;
  constexpr uint32_t kExpectedStripeCount = 3; // ceil(8 / 3)
  constexpr int kRowsPerChunk = 1000;
  constexpr uint32_t kSeed = 0xC0FFEE;
  const auto batches =
      makeDivergentBatches(type, kChunkCount, kRowsPerChunk, kSeed);

  auto makeOptions = [](bool enableEncodingSelectionCache) {
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory =
        chunkAndStripeFlushPolicyFactory(kChunksPerStripe);
    return options;
  };

  for (int column = 0; column < static_cast<int>(type->size()); ++column) {
    auto control = writeAndCaptureChunkLayouts(
        type,
        batches,
        makeOptions(/*enableEncodingSelectionCache=*/false),
        kExpectedStripeCount,
        column);
    auto cached = writeAndCaptureChunkLayouts(
        type,
        batches,
        makeOptions(/*enableEncodingSelectionCache=*/true),
        kExpectedStripeCount,
        column);
    ASSERT_EQ(control.size(), static_cast<size_t>(kChunkCount));
    ASSERT_EQ(cached.size(), static_cast<size_t>(kChunkCount));

    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      EXPECT_TRUE(
          encodingLayoutsEqual(cached[chunk], cached.front()) ||
          encodingLayoutsEqual(cached[chunk], control[chunk]))
          << "column " << column << " (" << type->childAt(column)->toString()
          << ") chunk " << chunk;
    }
  }
}

DEBUG_ONLY_TEST_F(WriterTest, cachedEncodingLayoutReplayRetry) {
  // With the cache on, chunk 0 runs a fresh selection and caches its layout;
  // chunk 1 replays it through encodeWithFallback. A TestValue hook at the
  // encode entry point counts fresh vs replay attempts and can throw to force
  // the replay to fail. encodeWithFallback catches ANY exception (not just
  // IncompatibleEncoding) and retries once with a fresh selection.
  velox::common::testutil::TestValue::enable();

  const auto type = velox::ROW({{"c0", velox::BIGINT()}});
  // Two non-nullable chunks: chunk 0 seeds the cache, chunk 1 replays it.
  const auto batches = bigintBatches(makeDivergentInt64Chunks(
      /*chunkCount=*/2, /*rowsPerChunk=*/1000, /*seed=*/0xC0FFEE));

  // Runs a 2-chunk cached write with the given encode hook; returns whether the
  // write threw.
  auto runScenario = [&](std::function<void(const bool*)> onEncode) {
    SCOPED_TESTVALUE_SET("facebook::nimble::encode", std::move(onEncode));
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = true;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = [] {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /*flushLambda=*/[](auto&) { return false; },
          /*chunkLambda=*/[](auto&) { return true; });
    };
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));
    try {
      for (const auto& batch : batches) {
        writer.write(batch);
      }
      writer.close();
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };

  // Case 1: the replay succeeds, so there is no retry. Chunk 0 is the only
  // fresh selection; chunk 1 replays exactly once.
  {
    int fullSelectionCount = 0;
    int replayCount = 0;
    const bool threw = runScenario([&](const bool* hasEncodingLayout) {
      if (*hasEncodingLayout) {
        ++replayCount;
      } else {
        ++fullSelectionCount;
      }
    });
    EXPECT_FALSE(threw);
    EXPECT_EQ(fullSelectionCount, 1);
    EXPECT_EQ(replayCount, 1);
  }

  // Case 2: the replay throws a non-IncompatibleEncoding error; the catch-all
  // retries with a fresh selection, which succeeds.
  {
    int fullSelectionCount = 0;
    int replayCount = 0;
    const bool threw = runScenario([&](const bool* hasEncodingLayout) {
      if (*hasEncodingLayout) {
        ++replayCount;
        throw std::runtime_error("injected replay failure");
      }
      ++fullSelectionCount;
    });
    EXPECT_FALSE(threw);
    EXPECT_EQ(replayCount, 1); // one replay attempt, which threw
    EXPECT_EQ(fullSelectionCount, 2); // chunk 0 seed + chunk 1 retry
  }

  // Case 3: the replay and the fresh-selection retry both throw, so the error
  // propagates out of the writer.
  {
    int fullSelectionCount = 0;
    int replayCount = 0;
    bool replayed = false;
    const bool threw = runScenario([&](const bool* hasEncodingLayout) {
      if (*hasEncodingLayout) {
        ++replayCount;
        replayed = true;
        throw std::runtime_error("injected replay failure");
      }
      ++fullSelectionCount;
      if (replayed) {
        throw std::runtime_error("injected retry failure");
      }
    });
    EXPECT_TRUE(threw);
    EXPECT_EQ(replayCount, 1);
    EXPECT_EQ(
        fullSelectionCount, 2); // chunk 0 seed (ok) + chunk 1 retry (threw)
  }
}

DEBUG_ONLY_TEST_F(WriterTest, cachedEncodingLayoutSelectionCount) {
  // With the cache on, the full encoding selection runs only once per stream
  // (chunk 0) and every later chunk across all stripes replays chunk 0's
  // encoding. With it off, a fresh selection runs for every chunk, and because
  // the data is divergent each chunk selects a different encoding. A TestValue
  // hook at the encode entry point counts the full selections
  // (hasEncodingLayout
  // == false).
  velox::common::testutil::TestValue::enable();

  const auto type = velox::ROW({{"c0", velox::BIGINT()}});
  constexpr int kChunkCount = 8;
  constexpr int kChunksPerStripe = 3;
  constexpr uint32_t kExpectedStripeCount = 3; // ceil(8 / 3)
  // Chunk 0 is high-entropy random (a flat layout that replays onto anything);
  // later chunks are constant, so replay succeeds and never falls back to a
  // fresh selection.
  const auto batches = bigintBatches(makeDivergentInt64Chunks(
      kChunkCount, /*rowsPerChunk=*/1000, /*seed=*/0xC0FFEE));

  struct RunResult {
    int fullSelectionCount;
    std::vector<nimble::EncodingLayout> chunkLayouts;
  };
  // Writes the batches with the given cache flag and returns the number of full
  // encoding selections (counted via the TestValue hook) alongside the
  // per-chunk encoding layouts captured from the written file.
  auto run = [&](bool enableEncodingSelectionCache) -> RunResult {
    int fullSelectionCount = 0;
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::encode",
        std::function<void(const bool*)>([&](const bool* hasEncodingLayout) {
          if (!*hasEncodingLayout) {
            ++fullSelectionCount;
          }
        }));
    nimble::WriterOptions options;
    options.enableEncodingSelectionCache = enableEncodingSelectionCache;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory =
        chunkAndStripeFlushPolicyFactory(kChunksPerStripe);
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));
    for (const auto& batch : batches) {
      writer.write(batch);
    }
    writer.close();
    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    return {
        fullSelectionCount,
        captureColumnChunkLayouts(
            readFile, /*columnIndex=*/0, kExpectedStripeCount)};
  };

  // Cache off: the full selection runs for every chunk, and the divergent data
  // makes the chunks select different encodings.
  const auto uncached = run(/*enableEncodingSelectionCache=*/false);
  EXPECT_EQ(uncached.fullSelectionCount, kChunkCount);
  ASSERT_EQ(uncached.chunkLayouts.size(), static_cast<size_t>(kChunkCount));
  EXPECT_NE(
      uncached.chunkLayouts.front().encodingType(),
      uncached.chunkLayouts[1].encodingType());

  // Cache on: the full selection runs once, and every chunk replays chunk 0's
  // encoding.
  const auto cached = run(/*enableEncodingSelectionCache=*/true);
  EXPECT_EQ(cached.fullSelectionCount, 1);
  ASSERT_EQ(cached.chunkLayouts.size(), static_cast<size_t>(kChunkCount));
  for (const auto& layout : cached.chunkLayouts) {
    EXPECT_EQ(
        layout.encodingType(), cached.chunkLayouts.front().encodingType());
  }
}

DEBUG_ONLY_TEST_F(WriterTest, cachedEncodingLayoutSkipsSharedDictionary) {
  velox::common::testutil::TestValue::enable();

  constexpr int kChunkCount = 4;
  constexpr int kRowsPerChunk = 512;
  const auto type =
      velox::ROW({{"c0", velox::MAP(velox::BIGINT(), velox::INTEGER())}});

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  std::vector<velox::RowVectorPtr> batches;
  batches.reserve(kChunkCount);
  for (int chunk = 0; chunk < kChunkCount; ++chunk) {
    batches.push_back(vectorMaker.rowVector(
        {"c0"},
        {vectorMaker.mapVector<int64_t, int32_t>(
            kRowsPerChunk,
            [](auto /*row*/) { return 1; },
            [](auto /*idx*/) { return 10; },
            [chunk](auto row) {
              return static_cast<int32_t>((row + chunk) % 4);
            })}));
  }

  int fullSelectionCount = 0;
  int replayCount = 0;
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::encode",
      std::function<void(const bool*)>([&](const bool* hasEncodingLayout) {
        if (*hasEncodingLayout) {
          ++replayCount;
        } else {
          ++fullSelectionCount;
        }
      }));

  nimble::WriterOptions options;
  options.enableEncodingSelectionCache = true;
  options.enableChunking = true;
  options.minStreamChunkRawSize = 0;
  options.flushPolicyFactory = [] {
    return std::make_unique<nimble::LambdaFlushPolicy>(
        /*flushLambda=*/[](auto&) { return false; },
        /*chunkLambda=*/[](auto&) { return true; });
  };
  options.flatMapColumns.emplace("c0", std::set<std::string>{});
  options.experimentalSharedDictionaryEncoding =
      nimble::SharedDictionaryEncodingConfig::builder()
          .addFlatmapValueDictionary(
              "c0",
              /*key=*/10,
              nimble::SharedDictionaryConfig{
                  .scope = nimble::SharedDictionaryScope::File,
                  .dictionaryId = 17})
          .build();

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer{
      type, std::move(writeFile), *rootPool_, std::move(options)};
  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();

  // The shared-dictionary value stream still runs fresh selection for every
  // chunk. The flat-map in-map bool stream is not dictionary-configured, so it
  // captures once and replays for the remaining chunks.
  EXPECT_EQ(fullSelectionCount, kChunkCount + 1);
  EXPECT_EQ(replayCount, kChunkCount - 1);

  auto layouts = captureFlatMapKeyChunkLayouts(
      std::make_shared<velox::InMemoryReadFile>(file),
      /*columnIndex=*/0,
      /*key=*/"10",
      /*expectedStripeCount=*/1);
  ASSERT_EQ(layouts.size(), static_cast<size_t>(kChunkCount));
  for (const auto& layout : layouts) {
    EXPECT_EQ(layout.encodingType(), nimble::EncodingType::SharedDictionary);
  }
}

// Parameterized test fixture for index tests.
// When enableChunking is false, sets very high minChunkRawSize and
// maxChunkRawSize to prevent chunking (one chunk per stream).
// When enableChunking is true, sets very low values to force aggressive
// chunking.
struct IndexTestParams {
  bool enableChunking;
  nimble::EncodingType encodingType;
  std::optional<uint32_t> prefixRestartInterval;

  std::string toString() const {
    std::string name =
        encodingType == nimble::EncodingType::Prefix ? "Prefix" : "Trivial";
    if (prefixRestartInterval.has_value()) {
      name += "_restart" + std::to_string(prefixRestartInterval.value());
    }
    name += enableChunking ? "_WithKeyChunking" : "_NoKeyChunking";
    return name;
  }
};

class WriterIndexTest : public WriterTest,
                        public ::testing::WithParamInterface<IndexTestParams> {
 protected:
  bool enableChunking() const {
    return GetParam().enableChunking;
  }

  nimble::EncodingType encodingType() const {
    return GetParam().encodingType;
  }

  std::optional<uint32_t> prefixRestartInterval() const {
    return GetParam().prefixRestartInterval;
  }

  // Default type with integer key column and multiple scalar/complex non-key
  // columns for comprehensive index testing.
  static velox::RowTypePtr defaultType() {
    return velox::ROW({
        {"key_col", velox::BIGINT()}, // Integer key column for indexing
        // Scalar types
        {"int_col", velox::INTEGER()},
        {"double_col", velox::DOUBLE()},
        {"string_col1", velox::VARCHAR()},
        {"string_col2", velox::VARCHAR()},
        {"bool_col", velox::BOOLEAN()},
        // Complex types
        {"array_col", velox::ARRAY(velox::INTEGER())},
        {"map_col", velox::MAP(velox::VARCHAR(), velox::BIGINT())},
        {"row_col",
         velox::ROW(
             {{"nested_int", velox::INTEGER()},
              {"nested_string", velox::VARCHAR()}})},
    });
  }

  // Generate pre-sorted batches with sorted key column and fuzzed non-key
  // columns. Uses VectorFuzzer to generate non-key columns with various
  // encodings (flat, constant, dictionary) but not lazy vectors.
  std::vector<velox::RowVectorPtr> generateFuzzedSortedBatches(
      const velox::RowTypePtr& type,
      size_t batchCount,
      size_t batchSize,
      uint32_t seed = 12345) {
    folly::Random::DefaultGenerator rng(seed);
    velox::test::VectorMaker vectorMaker{leafPool_.get()};

    // Configure fuzzer to generate various encodings but not lazy vectors
    velox::VectorFuzzer::Options fuzzerOpts;
    fuzzerOpts.vectorSize = batchSize;
    fuzzerOpts.nullRatio = 0.1;
    fuzzerOpts.containerLength = 5;
    fuzzerOpts.stringLength = 20;
    fuzzerOpts.containerVariableLength = true;
    // Note: VectorFuzzer generates flat, constant, and dictionary encodings
    // by default. Lazy vectors are not generated unless explicitly requested.
    velox::VectorFuzzer fuzzer(fuzzerOpts, leafPool_.get(), seed);

    std::vector<velox::RowVectorPtr> batches;
    batches.reserve(batchCount);

    int64_t currentKey = 0;
    for (size_t i = 0; i < batchCount; ++i) {
      // Generate sorted key column
      std::vector<int64_t> keys(batchSize);
      for (size_t j = 0; j < batchSize; ++j) {
        currentKey += folly::Random::rand32(1, 10, rng);
        keys[j] = currentKey;
      }
      auto keyVector = vectorMaker.flatVector<int64_t>(keys);

      // Generate fuzzed non-key columns
      std::vector<velox::VectorPtr> children;
      children.push_back(keyVector);

      for (size_t colIdx = 1; colIdx < type->size(); ++colIdx) {
        auto childType = type->childAt(colIdx);
        children.push_back(fuzzer.fuzz(childType));
      }

      batches.push_back(
          std::make_shared<velox::RowVector>(
              leafPool_.get(),
              type,
              nullptr, // no nulls at top level
              batchSize,
              std::move(children)));
    }

    return batches;
  }

  std::shared_ptr<const nimble::index::IndexConfig> createIndexConfig(
      const std::vector<std::string>& columns,
      bool enforceKeyOrder = true) {
    nimble::EncodingLayout encodingLayout{
        nimble::EncodingType::Prefix,
        {},
        nimble::CompressionType::Uncompressed};
    if (encodingType() == nimble::EncodingType::Trivial) {
      // Trivial encoding for string data needs a child encoding for lengths.
      encodingLayout = nimble::EncodingLayout{
          nimble::EncodingType::Trivial,
          {},
          nimble::CompressionType::Zstd,
          {nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed}}};
    } else {
      // Prefix encoding - optionally set restart interval config
      nimble::EncodingLayout::Config layoutConfig;
      if (prefixRestartInterval().has_value()) {
        layoutConfig = nimble::EncodingLayout::Config{
            {{std::string(nimble::PrefixEncoding::kRestartIntervalConfigKey),
              std::to_string(prefixRestartInterval().value())}}};
      }
      encodingLayout = nimble::EncodingLayout{
          encodingType(),
          std::move(layoutConfig),
          nimble::CompressionType::Zstd};
    }

    return nimble::index::ClusterIndexConfigBuilder{}
        .withKeyColumns(columns)
        .withEnforceKeyOrder(enforceKeyOrder)
        .withEncodingLayout(std::move(encodingLayout))
        .withMaxRowsPerKeyChunk(enableChunking() ? 100 : 10'000)
        .build();
  }

  nimble::WriterOptions createWriterOptions(
      const std::shared_ptr<const nimble::index::IndexConfig>&
          clusterIndexConfig,
      const std::function<std::unique_ptr<nimble::FlushPolicy>()>&
          flushPolicyFactory = nullptr) {
    nimble::WriterOptions options{
        .clusterIndexConfig = clusterIndexConfig,
    };
    if (enableChunking()) {
      options.minStreamChunkRawSize = 1 << 10; // 1KB
      options.maxStreamChunkRawSize = 4 << 10; // 4KB
      options.enableChunking = true;
    }
    if (flushPolicyFactory) {
      options.flushPolicyFactory = std::move(flushPolicyFactory);
    }
    options.enableStreamDeduplication = true;
    return options;
  }

  // Verifies that file data matches the expected batches row-by-row.
  void verifyFileData(
      const std::string& file,
      const velox::TypePtr& type,
      const std::vector<velox::RowVectorPtr>& expectedBatches,
      uint32_t readBatchSize = 100) {
    velox::InMemoryReadFile readFile(file);
    nimble::BatchReader reader(&readFile, *leafPool_);

    auto expected = velox::BaseVector::create(type, 0, leafPool_.get());
    for (const auto& batch : expectedBatches) {
      expected->append(batch.get());
    }

    velox::VectorPtr result;
    uint64_t rowOffset = 0;
    while (reader.next(readBatchSize, result)) {
      for (velox::vector_size_t i = 0; i < result->size(); ++i) {
        ASSERT_TRUE(result->equalValueAt(expected.get(), i, rowOffset + i))
            << "Mismatch at row " << (rowOffset + i);
      }
      rowOffset += result->size();
    }
    EXPECT_EQ(rowOffset, expected->size()) << "All rows should be verified";
  }

  // Verifies that the position index correctly stores chunk row counts.
  // For each stripe and stream:
  // 1. Gets the expected chunk row counts from the position index
  // 2. Verifies using two methods:
  //    a. Sequential iteration using InMemoryChunkedStream
  //    b. Direct chunk access using offset/size from index with
  //    SingleChunkDecoder
  void verifyPositionIndex(const nimble::TabletReader& tablet) {
    for (uint32_t stripeIdx = 0; stripeIdx < tablet.stripeCount();
         ++stripeIdx) {
      const auto stripeId = tablet.stripeIdentifier(stripeIdx);

      if (stripeId.chunkStats() == nullptr) {
        // Group was skipped (no streams with >1 chunk). Skip verification.
        continue;
      }

      nimble::index::test::ChunkStatsTestHelper chunkHelper(
          stripeId.chunkStats().get());

      const uint32_t stripeOffsetInGroup =
          stripeIdx - chunkHelper.firstStripe();

      // Load all streams for this stripe
      const uint32_t streamCount = tablet.streamCount(stripeId);
      std::vector<uint32_t> streamIds(streamCount);
      std::iota(streamIds.begin(), streamIds.end(), 0);
      auto streamLoaders = tablet.load(stripeId, streamIds);

      for (uint32_t streamId = 0; streamId < streamLoaders.size(); ++streamId) {
        if (!streamLoaders[streamId]) {
          continue;
        }

        auto streamStats = chunkHelper.streamStats(streamId);
        if (streamStats.chunkCounts.empty()) {
          // Stream not indexed (0 or 1 chunk). Skip verification.
          continue;
        }

        // Get chunk count for this stripe from accumulated values
        const uint32_t prevChunkCount = (stripeOffsetInGroup == 0)
            ? 0
            : streamStats.chunkCounts[stripeOffsetInGroup - 1];
        const uint32_t currChunkCount =
            streamStats.chunkCounts[stripeOffsetInGroup];
        const uint32_t numChunksInStripe = currChunkCount - prevChunkCount;

        // Extract per-chunk row counts from accumulated row counts
        std::vector<uint32_t> expectedChunkRowCounts;
        expectedChunkRowCounts.reserve(numChunksInStripe);
        uint32_t prevRowCount = 0;
        for (uint32_t i = 0; i < numChunksInStripe; ++i) {
          const uint32_t accumulatedRows =
              streamStats.chunkRows[prevChunkCount + i];
          expectedChunkRowCounts.push_back(accumulatedRows - prevRowCount);
          prevRowCount = accumulatedRows;
        }

        // Get the raw stream data
        const std::string_view rawStreamData =
            streamLoaders[streamId]->getStream();

        // Method 1: Verify using sequential InMemoryChunkedStream iteration
        {
          nimble::InMemoryChunkedStream chunkedStream{
              *leafPool_,
              std::make_unique<nimble::index::test::TestStreamLoader>(
                  rawStreamData)};
          uint32_t chunkIndex = 0;
          while (chunkedStream.hasNext()) {
            const auto chunkData = chunkedStream.nextChunk();
            std::vector<velox::BufferPtr> stringBuffers;
            auto encoding = nimble::EncodingFactory().create(
                *leafPool_,
                chunkData,
                [&](uint32_t totalLength) {
                  auto& buffer = stringBuffers.emplace_back(
                      velox::AlignedBuffer::allocate<char>(
                          totalLength, leafPool_.get()));
                  return buffer->asMutable<void>();
                },
                nimble::Encoding::Options{});
            EXPECT_EQ(encoding->rowCount(), expectedChunkRowCounts[chunkIndex])
                << "Stripe " << stripeIdx << " stream " << streamId << " chunk "
                << chunkIndex << " row count mismatch (sequential)";
            ++chunkIndex;
          }

          EXPECT_EQ(chunkIndex, numChunksInStripe)
              << "Stripe " << stripeIdx << " stream " << streamId
              << " chunk count mismatch";
        }

        // Method 2: Verify using chunk offset/size from index with
        // SingleChunkDecoder
        for (uint32_t i = 0; i < numChunksInStripe; ++i) {
          const uint32_t chunkOffset =
              streamStats.chunkOffsets[prevChunkCount + i];

          // Calculate chunk size from consecutive offsets or to end of stream
          uint32_t chunkSize;
          if (i + 1 < numChunksInStripe) {
            chunkSize =
                streamStats.chunkOffsets[prevChunkCount + i + 1] - chunkOffset;
          } else {
            chunkSize = rawStreamData.size() - chunkOffset;
          }

          // Extract chunk data using offset and size from index
          std::string_view chunkStreamData(
              rawStreamData.data() + chunkOffset, chunkSize);

          // Decode using SingleChunkDecoder (same pattern as
          // verifyValueIndex)
          nimble::index::test::SingleChunkDecoder chunkDecoder(
              *leafPool_, chunkStreamData);
          auto encodingData = chunkDecoder.decode();
          std::vector<velox::BufferPtr> stringBuffers;
          auto encoding = nimble::EncodingFactory().create(
              *leafPool_,
              encodingData,
              [&](uint32_t totalLength) {
                auto& buffer = stringBuffers.emplace_back(
                    velox::AlignedBuffer::allocate<char>(
                        totalLength, leafPool_.get()));
                return buffer->asMutable<void>();
              },
              nimble::Encoding::Options{});

          EXPECT_EQ(encoding->rowCount(), expectedChunkRowCounts[i])
              << "Stripe " << stripeIdx << " stream " << streamId << " chunk "
              << i << " row count mismatch (using chunk offset from index)";
        }
      }
    }
  }

  nimble::TabletWriter::Stats duplicateStreamStatsFromLayout(
      const nimble::TabletReader& tablet) {
    nimble::TabletWriter::Stats stats;
    for (uint32_t stripeIndex = 0; stripeIndex < tablet.stripeCount();
         ++stripeIndex) {
      const auto stripeIdentifier = tablet.stripeIdentifier(stripeIndex);
      const auto streamCount = tablet.streamCount(stripeIdentifier);
      std::vector<nimble::TabletReader::StreamLocation> streamLocations(
          streamCount);
      tablet.streamLocations(stripeIdentifier, streamLocations);

      std::map<std::pair<uint32_t, uint32_t>, uint64_t> duplicateGroups;
      for (const auto& streamLocation : streamLocations) {
        if (streamLocation.size == 0) {
          continue;
        }
        ++duplicateGroups[{streamLocation.offset, streamLocation.size}];
      }

      for (const auto& [offsetAndSize, count] : duplicateGroups) {
        if (count > 1) {
          const auto duplicateCount = count - 1;
          stats.duplicateStreamCount += duplicateCount;
          stats.duplicateStreamBytes += offsetAndSize.second * duplicateCount;
        }
      }
    }
    return stats;
  }

  // Verifies that the value index correctly maps each key to its row
  // position. For each row in the input batches:
  // 1. Encodes the key using KeyEncoder
  // 2. Looks up the stripe via ClusterIndex
  // 3. Gets chunk location within stripe via ClusterIndex::lookupChunk
  // 4. Loads the key stream chunk and decodes it
  // 5. Uses seek to find the exact row within the chunk
  // 6. For duplicate keys, verifies the found row id matches the earliest row
  //    with the same key value
  void verifyValueIndex(
      const nimble::TabletReader& tablet,
      velox::ReadFile* file,
      const velox::RowTypePtr& type,
      const std::vector<velox::RowVectorPtr>& batches,
      const std::vector<std::string>& indexColumns) {
    const auto* index = tablet.clusterIndex();
    ASSERT_NE(index, nullptr) << "Index must exist";

    // Create sort orders for KeyEncoder (all ascending, nulls last)
    std::vector<velox::core::SortOrder> sortOrders(
        indexColumns.size(),
        velox::core::SortOrder(true, false)); // ascending, nulls last

    // Create KeyEncoder for encoding row keys
    auto keyEncoder = velox::serializer::KeyEncoder::create(
        indexColumns, type, sortOrders, leafPool_.get());

    // Pre-encode all keys and build a map from encoded key to earliest row
    // id. This handles duplicate keys where seek returns the first
    // occurrence.
    std::map<std::string, uint64_t> keyToEarliestRowId;
    std::vector<std::string> allEncodedKeys;
    {
      uint64_t rowId = 0;
      for (const auto& batch : batches) {
        velox::HashStringAllocator allocator(leafPool_.get());
        std::vector<std::string_view> encodedKeys;
        keyEncoder->encode(batch, encodedKeys, [&allocator](size_t size) {
          return allocator.allocate(size)->begin();
        });

        for (velox::vector_size_t i = 0; i < batch->size(); ++i) {
          std::string keyStr(encodedKeys[i]);
          allEncodedKeys.push_back(keyStr);
          // Only store the first occurrence (earliest row id) for each key
          if (keyToEarliestRowId.find(keyStr) == keyToEarliestRowId.end()) {
            keyToEarliestRowId[keyStr] = rowId;
          }
          ++rowId;
        }
      }
    }

    // Cache for loaded and decoded key stream chunks
    // Key: chunk file offset (unique across all stripes), Value: decoded
    // key encoding
    std::unordered_map<uint64_t, std::unique_ptr<nimble::index::KeyEncoding>>
        keyStreamCache;
    // Buffer for loaded key stream data (must outlive the encoding)
    std::unordered_map<uint64_t, std::string> keyStreamBufferCache;
    std::vector<velox::BufferPtr> stringBuffers;

    // Verify each row's index lookup
    uint64_t currentRowId = 0;
    for (const auto& batch : batches) {
      // Verify each row
      for (velox::vector_size_t i = 0; i < batch->size(); ++i) {
        const std::string& encodedKey = allEncodedKeys[currentRowId];
        std::string_view encodedKeyView(encodedKey);

        // Look up the row location for this key
        velox::serializer::EncodedKeyBounds bounds{
            .lowerKey = std::string(encodedKeyView), .upperKey = std::nullopt};
        auto lookupResult = index->lookup(
            nimble::index::IndexLookup::LookupRequest::rangeScan({bounds}));
        const auto rowRanges = lookupResult[0];
        ASSERT_EQ(rowRanges.size(), 1)
            << "Key at row " << currentRowId << " should be found in index";

        const auto& rowRange = rowRanges[0];
        ASSERT_LT(rowRange.startRow, tablet.tabletRowCount())
            << "Row range start out of range";

        // Look up chunk by encoded key to get chunk location within the
        // partition. Single partition in this test.
        constexpr uint32_t kPartitionIndex = 0;
        nimble::index::test::ClusterIndexTestHelper indexHelper(index);
        const auto chunkLocation =
            indexHelper.lookupChunk(kPartitionIndex, encodedKeyView);

        // Get key stream region for this partition
        const auto keyStreamRegion =
            indexHelper.keyStreamRegion(kPartitionIndex);

        // Cache key: chunk file offset (unique across all stripes)
        const uint64_t chunkFileOffset =
            keyStreamRegion.offset + chunkLocation.chunkOffset;

        // Load and decode key chunk if not cached
        if (keyStreamCache.find(chunkFileOffset) == keyStreamCache.end()) {
          const uint32_t chunkLength = chunkLocation.chunkSize;

          // Load the key chunk data from file
          velox::common::Region region{
              chunkFileOffset, chunkLength, "keyChunk"};
          folly::IOBuf iobuf;
          file->preadv({&region, 1}, {&iobuf, 1});

          // Copy data to a persistent buffer
          std::string& buffer = keyStreamBufferCache[chunkFileOffset];
          buffer.resize(iobuf.computeChainDataLength());
          size_t offset = 0;
          for (auto range : iobuf) {
            std::memcpy(buffer.data() + offset, range.data(), range.size());
            offset += range.size();
          }

          // Decode the single chunk to get the raw encoding data
          nimble::index::test::SingleChunkDecoder chunkDecoder(
              *leafPool_, buffer);
          auto encodingData = chunkDecoder.decode();

          // Decode the key encoding from the chunk data

          keyStreamCache[chunkFileOffset] = nimble::index::KeyEncoding::create(
              *leafPool_, encodingData, [&](uint32_t totalLength) {
                auto& buf = stringBuffers.emplace_back(
                    velox::AlignedBuffer::allocate<char>(
                        totalLength, leafPool_.get()));
                return buf->asMutable<void>();
              });
        }

        auto* keyEncoding = keyStreamCache[chunkFileOffset].get();
        ASSERT_NE(keyEncoding, nullptr);

        auto seekResult = keyEncoding->seek(encodedKeyView, /*inclusive=*/true);
        ASSERT_TRUE(seekResult.has_value())
            << "seek should find key at row " << currentRowId;

        // Calculate the actual file row id.
        // lookupChunk returns partition-wide row offsets. For single-partition
        // files, the partition start row is 0.
        const uint64_t fileRowId = chunkLocation.rowOffset + seekResult.value();

        // For duplicate keys, seek returns the first occurrence.
        // Verify that the found row id matches the earliest row with the same
        // key.
        const uint64_t expectedFileRowId = keyToEarliestRowId.at(encodedKey);
        EXPECT_EQ(fileRowId, expectedFileRowId)
            << "Row " << currentRowId << " key lookup mismatch: "
            << "expected earliest row " << expectedFileRowId << ", got "
            << fileRowId << " (partition " << kPartitionIndex
            << ", chunk row offset " << chunkLocation.rowOffset
            << ", seek offset " << seekResult.value() << ")";

        ++currentRowId;
      }
    }
  }
};

TEST_P(WriterIndexTest, singleGroup) {
  // Test writing a file with index using real pre-sorted data with complex
  // types. Uses a large flush threshold to ensure all stripes stay in one
  // group.
  auto type = defaultType();

  auto clusterIndexConfig = createIndexConfig({"key_col"});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  constexpr int kNumBatches = 5;
  constexpr int kBatchSize = 100;
  nimble::Writer writer(
      type,
      std::move(writeFile),
      *rootPool_,
      createWriterOptions(clusterIndexConfig, []() {
        // Flush after every batch to create one stripe per batch
        return std::make_unique<nimble::LambdaFlushPolicy>(
            [](auto) { return true; }, [](auto) { return false; });
      }));

  // Generate pre-sorted batches with fuzzed non-key columns
  auto batches = generateFuzzedSortedBatches(type, kNumBatches, kBatchSize);

  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();

  // Read and verify
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tabletOptions = makeTestTabletOptions(leafPool_.get());
  auto tablet =
      nimble::TabletReader::create(readFile, leafPool_.get(), tabletOptions);

  // Verify each batch triggered exactly one stripe
  EXPECT_EQ(tablet->stripeCount(), kNumBatches)
      << "Each batch should trigger exactly one stripe";

  // Verify all stripes are in the same stripe group (index 0)
  // Default metadataFlushThreshold is large, so all stripes stay in one group
  for (uint32_t i = 0; i < tablet->stripeCount(); ++i) {
    auto stripeId = tablet->stripeIdentifier(i);
    EXPECT_EQ(stripeId.stripeGroup()->index(), 0)
        << "Stripe " << i << " should be in stripe group 0";
  }

  // Verify index section exists
  EXPECT_TRUE(tablet->hasOptionalSection(std::string(nimble::kIndexSection)));

  // Verify index is available
  const auto* index = tablet->clusterIndex();
  ASSERT_NE(index, nullptr);

  // Verify index columns
  EXPECT_EQ(index->indexColumns().size(), 1);
  EXPECT_EQ(index->indexColumns()[0], "key_col");

  // Verify sub-partition keys are monotonically increasing
  nimble::index::test::ClusterIndexTestHelper indexHelper(index);
  const auto& partitionKeys = indexHelper.partitionKeys();
  for (size_t i = 1; i < partitionKeys.size(); ++i) {
    EXPECT_LE(partitionKeys[i - 1], partitionKeys[i])
        << "Sub-partition keys should be in non-descending order";
  }

  // Verify lookups work correctly
  // Keys should be found in the correct partitions
  ASSERT_FALSE(partitionKeys.empty());
  auto makeLookup = [](std::string_view key) {
    velox::serializer::EncodedKeyBounds bounds{
        .lowerKey = std::string(key), .upperKey = std::nullopt};
    return nimble::index::IndexLookup::LookupRequest::rangeScan({bounds});
  };
  {
    auto r = index->lookup(makeLookup(partitionKeys.front()));
    EXPECT_FALSE(r[0].empty());
  }
  {
    auto r = index->lookup(makeLookup(partitionKeys.back()));
    EXPECT_FALSE(r[0].empty());
  }

  // Empty key with range scan (no upper bound) matches all rows since
  // "" is less than any encoded key.
  {
    auto r = index->lookup(makeLookup(""));
    EXPECT_FALSE(r[0].empty());
  }

  // Read back all data and verify row-by-row match with written batches
  verifyFileData(file, type, batches, kBatchSize);

  // Verify position index chunk row counts
  verifyPositionIndex(*tablet);

  // Verify value index maps each key to correct row position
  verifyValueIndex(*tablet, readFile.get(), type, batches, {"key_col"});
}

TEST_P(WriterIndexTest, multipleGroups) {
  // Test writing a file with index and multiple stripe groups.
  // Uses small batches with flush-per-batch policy to create multiple
  // stripes.
  auto type = defaultType();

  auto clusterIndexConfig = createIndexConfig({"key_col"});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  constexpr int kNumBatches = 5;
  constexpr int kBatchSize = 100;
  nimble::Writer writer(
      type,
      std::move(writeFile),
      *rootPool_,
      createWriterOptions(clusterIndexConfig, []() {
        // Flush stripe after each batch
        return std::make_unique<nimble::LambdaFlushPolicy>(
            [](auto) { return true; }, [](auto) { return false; });
      }));

  // Generate pre-sorted batches with fuzzed non-key columns
  auto batches = generateFuzzedSortedBatches(type, kNumBatches, kBatchSize);

  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();

  // Read and verify
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tabletOptions = makeTestTabletOptions(leafPool_.get());
  auto tablet =
      nimble::TabletReader::create(readFile, leafPool_.get(), tabletOptions);

  // Verify each batch triggered exactly one stripe
  EXPECT_EQ(tablet->stripeCount(), kNumBatches)
      << "Each batch should trigger exactly one stripe";

  // Note: Without controlling metadataFlushThreshold from WriterOptions,
  // all stripes end up in the same group (default threshold is 8MB).
  // This test verifies the default behavior where all stripes share group 0.
  for (uint32_t i = 0; i < tablet->stripeCount(); ++i) {
    auto stripeId = tablet->stripeIdentifier(i);
    EXPECT_EQ(stripeId.stripeGroup()->index(), 0)
        << "Stripe " << i << " should be in stripe group 0";
  }

  // Verify index section exists
  EXPECT_TRUE(tablet->hasOptionalSection(std::string(nimble::kIndexSection)));

  // Verify index is available
  const auto* index = tablet->clusterIndex();
  ASSERT_NE(index, nullptr);

  // Verify index columns
  EXPECT_EQ(index->indexColumns().size(), 1);
  EXPECT_EQ(index->indexColumns()[0], "key_col");

  // Verify sub-partition keys are monotonically increasing
  nimble::index::test::ClusterIndexTestHelper indexHelper(index);
  const auto& partitionKeys = indexHelper.partitionKeys();
  for (size_t i = 1; i < partitionKeys.size(); ++i) {
    EXPECT_LE(partitionKeys[i - 1], partitionKeys[i])
        << "Sub-partition keys should be in non-descending order";
  }

  // Verify we can look up by key
  // Get the first key and verify lookup returns a valid location
  ASSERT_FALSE(partitionKeys.empty());
  auto makeLookup2 = [](std::string_view key) {
    velox::serializer::EncodedKeyBounds bounds{
        .lowerKey = std::string(key), .upperKey = std::nullopt};
    return nimble::index::IndexLookup::LookupRequest::rangeScan({bounds});
  };
  auto firstResult = index->lookup(makeLookup2(index->minKey()));
  auto firstRanges = firstResult[0];
  ASSERT_EQ(firstRanges.size(), 1);
  EXPECT_EQ(firstRanges[0].startRow, 0);

  // Get the last key and verify lookup returns a valid location
  auto lastResult = index->lookup(makeLookup2(index->maxKey()));
  auto lastRanges = lastResult[0];
  ASSERT_EQ(lastRanges.size(), 1);
  EXPECT_GT(lastRanges[0].endRow, 0);

  // Verify stripeIdentifier returns index group
  for (uint32_t i = 0; i < tablet->stripeCount(); ++i) {
    auto stripeId = tablet->stripeIdentifier(i);
    EXPECT_NE(stripeId.stripeGroup(), nullptr);
  }

  // Read back all data and verify row-by-row match with written batches
  verifyFileData(file, type, batches);

  // Verify position index chunk row counts
  verifyPositionIndex(*tablet);

  // Verify value index maps each key to correct row position
  verifyValueIndex(*tablet, readFile.get(), type, batches, {"key_col"});
}

TEST_F(WriterTest, compactRowCountEncodingIsPersistedAsFileFeature) {
  auto type = velox::ROW({{"c0", velox::BIGINT()}});
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto batch = vectorMaker.rowVector(
      {"c0"}, {vectorMaker.flatVector<int64_t>({1, 2, 3, 4})});

  auto writeWithCompactRowCountEncoding = [&](bool enabled) {
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.experimentalCompactRowCountEncoding = enabled;

    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));
    writer.write(batch);
    writer.close();
    return file;
  };

  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(
        testing::Message() << "experimentalCompactRowCountEncoding="
                           << enabled);
    auto readFile = std::make_shared<velox::InMemoryReadFile>(
        writeWithCompactRowCountEncoding(enabled));
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    EXPECT_EQ(tablet->properties().compactRowCountEncoding(), enabled);
  }
}

TEST_P(WriterIndexTest, omitClusterIndexKeyColumnStorage) {
  auto type = velox::ROW({
      {"key_col", velox::BIGINT()},
      {"value_col", velox::INTEGER()},
      {"payload", velox::VARCHAR()},
  });
  auto storedType = velox::ROW({
      {"value_col", velox::INTEGER()},
      {"payload", velox::VARCHAR()},
  });

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto batch = vectorMaker.rowVector(
      {"key_col", "value_col", "payload"},
      {
          vectorMaker.flatVector<int64_t>({10, 20, 30, 40}),
          vectorMaker.flatVector<int32_t>({1, 2, 3, 4}),
          vectorMaker.flatVector<std::string>(
              {"first", "second", "third", "fourth"}),
      });

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  auto options = createWriterOptions(createIndexConfig({"key_col"}));
  options.experimentalOmitClusterIndexKeyColumnStorage = true;

  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(options));
  writer.write(batch);
  writer.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = nimble::TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  ASSERT_NE(tablet->clusterIndex(), nullptr);
  EXPECT_EQ(
      tablet->clusterIndex()->indexColumns(),
      (std::vector<std::string>{"key_col"}));
  EXPECT_TRUE(tablet->properties().clusterIndexKeyColumnStorageOmitted());
  EXPECT_EQ(
      tablet->properties().clusterIndexKeyColumnsWithOmittedStorage(),
      (std::vector<std::string>{"key_col"}));

  auto tabletOptions = makeTestTabletOptions(leafPool_.get());
  tabletOptions.loadClusterIndex = false;
  auto tabletWithoutIndex =
      nimble::TabletReader::create(readFile, leafPool_.get(), tabletOptions);
  EXPECT_EQ(tabletWithoutIndex->clusterIndex(), nullptr);
  EXPECT_TRUE(
      tabletWithoutIndex->properties().clusterIndexKeyColumnStorageOmitted());
  EXPECT_EQ(
      tabletWithoutIndex->properties()
          .clusterIndexKeyColumnsWithOmittedStorage(),
      (std::vector<std::string>{"key_col"}));

  nimble::BatchReader reader(readFile.get(), *leafPool_);
  auto actualStoredType = nimble::convertToVeloxType(*reader.schema());
  EXPECT_EQ(*storedType, *actualStoredType);

  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(10, result));
  ASSERT_EQ(result->size(), batch->size());
  auto expected = vectorMaker.rowVector(
      {"value_col", "payload"},
      {
          batch->childAt(1),
          batch->childAt(2),
      });
  for (velox::vector_size_t i = 0; i < result->size(); ++i) {
    EXPECT_TRUE(result->equalValueAt(expected.get(), i, i));
  }
  EXPECT_FALSE(reader.next(10, result));

  verifyValueIndex(*tablet, readFile.get(), type, {batch}, {"key_col"});
}

TEST_P(
    WriterIndexTest,
    omittedClusterIndexKeyColumnStorageRemapsSchemaAttributes) {
  auto nestedType = velox::ROW({{"score", velox::INTEGER()}});
  auto type = velox::ROW({
      {"key_col", velox::BIGINT()},
      {"value_col", velox::VARCHAR()},
      {"nested_col", nestedType},
  });

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto batch = vectorMaker.rowVector(
      {"key_col", "value_col", "nested_col"},
      {
          vectorMaker.flatVector<int64_t>({10, 20}),
          vectorMaker.flatVector<std::string>({"first", "second"}),
          vectorMaker.rowVector(
              {"score"}, {vectorMaker.flatVector<int32_t>({1, 2})}),
      });

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  auto options = createWriterOptions(createIndexConfig({"key_col"}));
  options.experimentalOmitClusterIndexKeyColumnStorage = true;
  // Input TypeWithId ids: key_col=1, value_col=2, nested_col=3, score=4.
  // key_col is omitted from the stored schema; remaining ids must be remapped
  // onto the stored TypeWithId tree while preserving the field-id values.
  options.schemaAttributes[1] = {{"iceberg.id", "101"}};
  options.schemaAttributes[2] = {{"iceberg.id", "102"}};
  options.schemaAttributes[3] = {{"iceberg.id", "103"}};
  options.schemaAttributes[4] = {{"iceberg.id", "104"}};

  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(options));
  writer.write(batch);
  writer.close();

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);
  const auto& rowSchema = reader.schema()->asRow();
  ASSERT_EQ(2, rowSchema.childrenCount());
  EXPECT_EQ("value_col", rowSchema.nameAt(0));
  EXPECT_EQ("nested_col", rowSchema.nameAt(1));

  EXPECT_EQ(
      rowSchema.childAt(0)->attributes(),
      (std::vector<std::pair<std::string, std::string>>{
          {"iceberg.id", "102"}}));

  const auto& nestedSchema = rowSchema.childAt(1);
  ASSERT_EQ(nimble::Kind::Row, nestedSchema->kind());
  EXPECT_EQ(
      nestedSchema->attributes(),
      (std::vector<std::pair<std::string, std::string>>{
          {"iceberg.id", "103"}}));
  EXPECT_EQ(
      nestedSchema->asRow().childAt(0)->attributes(),
      (std::vector<std::pair<std::string, std::string>>{
          {"iceberg.id", "104"}}));
}

TEST_P(WriterIndexTest, multipleIndexColumns) {
  // Test index with multiple index columns (composite key).
  auto type = velox::ROW({
      {"key1", velox::BIGINT()},
      {"key2", velox::INTEGER()},
      {"data_col", velox::VARCHAR()},
  });

  auto clusterIndexConfig = createIndexConfig({"key1", "key2"});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  nimble::Writer writer(
      type,
      std::move(writeFile),
      *rootPool_,
      createWriterOptions(clusterIndexConfig));

  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  // Create pre-sorted data with composite key (key1, key2)
  // key1 increases slowly, key2 varies within same key1 value
  int64_t key1Val = 0;
  int32_t key2Val = 0;
  for (int batch = 0; batch < 10; ++batch) {
    std::vector<int64_t> key1Values;
    std::vector<int32_t> key2Values;
    std::vector<std::string> dataValues;

    for (int i = 0; i < 100; ++i) {
      key1Values.push_back(key1Val);
      key2Values.push_back(key2Val);
      dataValues.push_back(
          "data_" + std::to_string(key1Val) + "_" + std::to_string(key2Val));

      // Increment keys in sorted order
      ++key2Val;
      if (key2Val >= 5) {
        key2Val = 0;
        ++key1Val;
      }
    }

    auto vector = vectorMaker.rowVector(
        {"key1", "key2", "data_col"},
        {
            vectorMaker.flatVector<int64_t>(key1Values),
            vectorMaker.flatVector<int32_t>(key2Values),
            vectorMaker.flatVector<std::string>(dataValues),
        });
    writer.write(vector);
  }
  writer.close();

  // Read and verify
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tabletOptions = makeTestTabletOptions(leafPool_.get());
  auto tablet =
      nimble::TabletReader::create(readFile, leafPool_.get(), tabletOptions);
  const auto* index = tablet->clusterIndex();
  ASSERT_NE(index, nullptr);

  // Verify both columns are indexed
  EXPECT_EQ(index->indexColumns().size(), 2);
  EXPECT_EQ(index->indexColumns()[0], "key1");
  EXPECT_EQ(index->indexColumns()[1], "key2");

  // Verify sub-partition keys exist
  nimble::index::test::ClusterIndexTestHelper multiKeyHelper(index);
  const auto& multiKeySubPartitionKeys = multiKeyHelper.partitionKeys();
  ASSERT_FALSE(multiKeySubPartitionKeys.empty());
  EXPECT_FALSE(multiKeySubPartitionKeys.front().empty());
  EXPECT_FALSE(multiKeySubPartitionKeys.back().empty());

  // Verify value index maps each key to correct row position
  // Need to collect batches to verify
  velox::test::VectorMaker vectorMaker2{leafPool_.get()};
  std::vector<velox::RowVectorPtr> batches;
  key1Val = 0;
  key2Val = 0;
  for (int batch = 0; batch < 10; ++batch) {
    std::vector<int64_t> key1Values;
    std::vector<int32_t> key2Values;
    std::vector<std::string> dataValues;

    for (int i = 0; i < 100; ++i) {
      key1Values.push_back(key1Val);
      key2Values.push_back(key2Val);
      dataValues.push_back(
          "data_" + std::to_string(key1Val) + "_" + std::to_string(key2Val));

      ++key2Val;
      if (key2Val >= 5) {
        key2Val = 0;
        ++key1Val;
      }
    }

    batches.push_back(vectorMaker2.rowVector(
        {"key1", "key2", "data_col"},
        {
            vectorMaker2.flatVector<int64_t>(key1Values),
            vectorMaker2.flatVector<int32_t>(key2Values),
            vectorMaker2.flatVector<std::string>(dataValues),
        }));
  }
  // Verify position index chunk row counts
  verifyPositionIndex(*tablet);

  verifyValueIndex(*tablet, readFile.get(), type, batches, {"key1", "key2"});
}

TEST_F(WriterTest, indexEnforceKeyOrder) {
  // Test both enforceKeyOrder=true (detects out-of-order and duplicate keys)
  // and enforceKeyOrder=false (allows out-of-order and duplicate keys)
  auto type = velox::ROW(
      {{"key_col", velox::BIGINT()}, {"data_col", velox::INTEGER()}});

  enum class TestCase { kOutOfOrder, kDuplicateKeys };

  for (bool enforceKeyOrder : {true, false}) {
    for (auto testCase : {TestCase::kOutOfOrder, TestCase::kDuplicateKeys}) {
      SCOPED_TRACE(
          fmt::format(
              "enforceKeyOrder={}, testCase={}",
              enforceKeyOrder,
              testCase == TestCase::kOutOfOrder ? "OutOfOrder"
                                                : "DuplicateKeys"));

      auto clusterIndexConfig = nimble::index::ClusterIndexConfigBuilder{}
                                    .withKeyColumns({"key_col"})
                                    .withEnforceKeyOrder(enforceKeyOrder)
                                    .withNoDuplicateKey(enforceKeyOrder)
                                    .build();

      std::string file;
      auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

      nimble::Writer writer(
          type,
          std::move(writeFile),
          *rootPool_,
          {.clusterIndexConfig = std::move(clusterIndexConfig)});

      velox::test::VectorMaker vectorMaker{leafPool_.get()};

      // Write first batch with ascending keys
      auto batch1 = vectorMaker.rowVector(
          {"key_col", "data_col"},
          {
              vectorMaker.flatVector<int64_t>({100, 200, 300}),
              vectorMaker.flatVector<int32_t>({1, 2, 3}),
          });
      writer.write(batch1);

      // Write second batch with invalid key ordering based on test case:
      // - OutOfOrder: keys LESS than previous batch
      // - DuplicateKeys: keys starting with same value as previous batch's
      // last key
      auto batch2 = testCase == TestCase::kOutOfOrder
          ? vectorMaker.rowVector(
                {"key_col", "data_col"},
                {
                    vectorMaker.flatVector<int64_t>({50, 60, 70}),
                    vectorMaker.flatVector<int32_t>({4, 5, 6}),
                })
          : vectorMaker.rowVector(
                {"key_col", "data_col"},
                {
                    vectorMaker.flatVector<int64_t>({300, 400, 500}),
                    vectorMaker.flatVector<int32_t>({4, 5, 6}),
                });

      if (enforceKeyOrder) {
        // Should fail when enforceKeyOrder is true
        NIMBLE_ASSERT_USER_THROW(
            writer.write(batch2),
            "Encoded keys must be in strictly ascending order (duplicates are not allowed)");
      } else {
        // Should succeed when enforceKeyOrder is false
        EXPECT_NO_THROW(writer.write(batch2));
        writer.close();

        // Verify file was written successfully
        velox::InMemoryReadFile readFile(file);
        nimble::BatchReader reader(&readFile, *leafPool_);
        EXPECT_TRUE(reader.tabletReader().hasOptionalSection(
            std::string(nimble::kIndexSection)));
      }
    }
  }
}

TEST_P(WriterIndexTest, duplicateKeys) {
  // Test index with duplicate key values (non-unique keys).
  // This is a valid scenario where multiple rows have the same key value.
  auto type = defaultType();

  auto clusterIndexConfig = createIndexConfig({"key_col"});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  // constexpr int kNumBatches = 10;
  constexpr int kNumBatches = 1;
  // constexpr int kBatchSize = 100;
  constexpr int kBatchSize = 32;
  constexpr uint32_t kSeed = 12345;
  // Use flush-per-batch to create multiple stripes
  nimble::Writer writer(
      type,
      std::move(writeFile),
      *rootPool_,
      createWriterOptions(clusterIndexConfig, []() {
        return std::make_unique<nimble::LambdaFlushPolicy>(
            [](auto) { return true; }, [](auto) { return false; });
      }));

  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  // Configure fuzzer to generate various encodings for non-key columns
  velox::VectorFuzzer::Options fuzzerOpts;
  fuzzerOpts.vectorSize = kBatchSize;
  fuzzerOpts.nullRatio = 0.1;
  fuzzerOpts.containerLength = 5;
  fuzzerOpts.stringLength = 20;
  fuzzerOpts.containerVariableLength = true;
  velox::VectorFuzzer fuzzer(fuzzerOpts, leafPool_.get(), kSeed);

  // Generate pre-sorted batches with duplicate keys.
  // Each key value repeats 5 times before incrementing.
  std::vector<velox::RowVectorPtr> batches;
  int64_t keyVal = 0;
  for (int batch = 0; batch < kNumBatches; ++batch) {
    std::vector<int64_t> keyValues;

    for (int i = 0; i < kBatchSize; ++i) {
      keyValues.push_back(keyVal);
      // Increment key every 5 rows to create duplicates
      if (((batch * kBatchSize + i + 1) % 5) == 0) {
        ++keyVal;
      }
    }

    // Build children: key column + fuzzed non-key columns
    std::vector<velox::VectorPtr> children;
    children.push_back(vectorMaker.flatVector<int64_t>(keyValues));
    for (size_t colIdx = 1; colIdx < type->size(); ++colIdx) {
      children.push_back(fuzzer.fuzz(type->childAt(colIdx)));
    }

    auto batchVec = std::make_shared<velox::RowVector>(
        leafPool_.get(),
        type,
        nullptr, // no nulls at top level
        kBatchSize,
        std::move(children));
    batches.push_back(batchVec);
    writer.write(batchVec);
  }
  writer.close();

  // Read and verify
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);

  // Verify FileLayout for non-empty file with index using
  // FileLayout::create()
  {
    auto layout = nimble::FileLayout::create(readFile, leafPool_.get());
    EXPECT_EQ(layout.fileSize, file.size());
    EXPECT_EQ(layout.stripesInfo.size(), kNumBatches);
    EXPECT_EQ(layout.stripeGroups.size(), 1);
    EXPECT_EQ(layout.stripeGroups.size(), 1);
    // With index enabled, should have index groups
    EXPECT_EQ(layout.indexPartitions.size(), 1);
    // Stripes metadata should be valid
    EXPECT_GT(layout.stripes.size(), 0);
    EXPECT_LT(layout.stripes.offset(), layout.footer.offset());
    // Index group should be before stripes section
    EXPECT_LT(layout.indexPartitions[0].offset(), layout.stripes.offset());
    // Per-stripe info
    EXPECT_EQ(layout.stripesInfo.size(), kNumBatches);
    for (size_t i = 0; i < layout.stripesInfo.size(); ++i) {
      EXPECT_EQ(layout.stripesInfo[i].stripeGroupIndex, 0);
      EXPECT_GT(layout.stripesInfo[i].size, 0);
    }
  }
  auto tabletOptions = makeTestTabletOptions(leafPool_.get());
  auto tablet =
      nimble::TabletReader::create(readFile, leafPool_.get(), tabletOptions);

  // Verify index exists
  const auto* index = tablet->clusterIndex();
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(index->indexColumns().size(), 1);
  EXPECT_EQ(index->indexColumns()[0], "key_col");

  // Verify sub-partition keys are monotonically increasing (even with
  // duplicates)
  nimble::index::test::ClusterIndexTestHelper dupKeyHelper(index);
  const auto& dupSubPartitionKeys = dupKeyHelper.partitionKeys();
  for (size_t i = 1; i < dupSubPartitionKeys.size(); ++i) {
    EXPECT_LE(dupSubPartitionKeys[i - 1], dupSubPartitionKeys[i])
        << "Sub-partition keys should be in non-descending order";
  }

  // Verify we can look up by key
  ASSERT_FALSE(dupSubPartitionKeys.empty());
  auto makeLookup3 = [](std::string_view key) {
    velox::serializer::EncodedKeyBounds bounds{
        .lowerKey = std::string(key), .upperKey = std::nullopt};
    return nimble::index::IndexLookup::LookupRequest::rangeScan({bounds});
  };
  auto firstResult = index->lookup(makeLookup3(index->minKey()));
  auto firstRanges = firstResult[0];
  ASSERT_EQ(firstRanges.size(), 1);
  EXPECT_EQ(firstRanges[0].startRow, 0);

  auto lastResult = index->lookup(makeLookup3(index->maxKey()));
  auto lastRanges = lastResult[0];
  ASSERT_EQ(lastRanges.size(), 1);
  EXPECT_GT(lastRanges[0].endRow, 0);

  // Read back all data and verify row-by-row match with written batches
  verifyFileData(file, type, batches);

  // Verify position index chunk row counts
  verifyPositionIndex(*tablet);

  // Verify value index maps each key to correct row position
  // With duplicate keys, lookup should find the first occurrence
  verifyValueIndex(*tablet, readFile.get(), type, batches, {"key_col"});
}

TEST_P(WriterIndexTest, chunking) {
  // Test index with chunking enabled
  auto type = defaultType();

  auto clusterIndexConfig = createIndexConfig({"key_col"});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  constexpr int kNumBatches = 50;
  constexpr int kBatchSize = 500;
  auto options = createWriterOptions(clusterIndexConfig, []() {
    return std::make_unique<nimble::StripeRawSizeFlushPolicy>(128 << 10);
  });
  // Always enable chunking for this test
  options.minStreamChunkRawSize = 1 << 10;
  options.maxStreamChunkRawSize = 4 << 10;
  options.enableChunking = true;

  nimble::Writer writer(type, std::move(writeFile), *rootPool_, options);

  // Generate pre-sorted batches with fuzzed non-key columns
  auto batches = generateFuzzedSortedBatches(type, kNumBatches, kBatchSize);

  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();

  // Read and verify
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tabletOptions = makeTestTabletOptions(leafPool_.get());
  auto tablet =
      nimble::TabletReader::create(readFile, leafPool_.get(), tabletOptions);

  // Verify index exists and works
  const auto* index = tablet->clusterIndex();
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(index->indexColumns().size(), 1);
  EXPECT_EQ(index->indexColumns()[0], "key_col");

  // Verify sub-partition keys are in order
  nimble::index::test::ClusterIndexTestHelper indexHelper(index);
  const auto& partitionKeys = indexHelper.partitionKeys();
  for (size_t i = 1; i < partitionKeys.size(); ++i) {
    EXPECT_LE(partitionKeys[i - 1], partitionKeys[i]);
  }

  // Read back all data and verify row-by-row match with written batches
  verifyFileData(file, type, batches);

  // Verify position index chunk row counts
  verifyPositionIndex(*tablet);

  // Verify value index maps each key to correct row position
  verifyValueIndex(*tablet, readFile.get(), type, batches, {"key_col"});
}

TEST_P(WriterIndexTest, streamDeduplication) {
  // Test that streams with identical content are deduplicated by the tablet
  // writer. We create batches where string_col1 and string_col2 reference the
  // same underlying vector, which should result in identical stream content
  // that gets deduplicated.
  auto type = velox::ROW({
      {"key_col", velox::BIGINT()},
      {"string_col1", velox::VARCHAR()},
      {"string_col2", velox::VARCHAR()},
  });

  auto clusterIndexConfig = createIndexConfig({"key_col"});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  constexpr int kNumBatches = 5;
  constexpr int kBatchSize = 100;

  nimble::Writer writer(
      type,
      std::move(writeFile),
      *rootPool_,
      createWriterOptions(clusterIndexConfig, []() {
        // Flush after every batch to create multiple stripes
        return std::make_unique<nimble::LambdaFlushPolicy>(
            [](auto) { return true; }, [](auto) { return false; });
      }));

  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  // Generate pre-sorted batches where string_col1 and string_col2 share the
  // same underlying vector (to trigger stream deduplication)
  std::vector<velox::RowVectorPtr> batches;
  int64_t keyVal = 0;
  for (int batch = 0; batch < kNumBatches; ++batch) {
    std::vector<int64_t> keyValues;
    keyValues.reserve(kBatchSize);
    for (int i = 0; i < kBatchSize; ++i) {
      keyValues.push_back(keyVal++);
    }

    // Create the shared string vector that will be used for both string
    // columns.
    auto sharedStringVector =
        vectorMaker.flatVector<std::string>(kBatchSize, [batch](auto row) {
          return fmt::format("batch-{}-row-{}", batch, row);
        });

    std::vector<velox::VectorPtr> children;
    children.push_back(vectorMaker.flatVector<int64_t>(keyValues));
    children.push_back(sharedStringVector);
    children.push_back(sharedStringVector);

    auto batchVec = std::make_shared<velox::RowVector>(
        leafPool_.get(),
        type,
        nullptr, // no nulls at top level
        kBatchSize,
        std::move(children));
    batches.push_back(batchVec);
    writer.write(batchVec);
  }
  writer.close();

  // Read and verify
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tabletOptions = makeTestTabletOptions(leafPool_.get());
  auto tablet =
      nimble::TabletReader::create(readFile, leafPool_.get(), tabletOptions);
  const auto expectedStats = duplicateStreamStatsFromLayout(*tablet);
  const auto writerStats = writer.runtimeStats();
  EXPECT_EQ(
      nimble::runtimeStat(
          writerStats, nimble::Writer::RuntimeStats::kDuplicateStreamCount)
          .sum,
      expectedStats.duplicateStreamCount);
  EXPECT_EQ(
      nimble::runtimeStat(
          writerStats, nimble::Writer::RuntimeStats::kDuplicateStreamBytes)
          .sum,
      expectedStats.duplicateStreamBytes);
  EXPECT_EQ(expectedStats.duplicateStreamCount, kNumBatches);
  EXPECT_GT(expectedStats.duplicateStreamBytes, 0);

  // Verify index exists
  const auto* index = tablet->clusterIndex();
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(index->indexColumns().size(), 1);
  EXPECT_EQ(index->indexColumns()[0], "key_col");

  // Verify sub-partition keys are monotonically increasing
  nimble::index::test::ClusterIndexTestHelper indexHelper(index);
  const auto& partitionKeys = indexHelper.partitionKeys();
  for (size_t i = 1; i < partitionKeys.size(); ++i) {
    EXPECT_LE(partitionKeys[i - 1], partitionKeys[i])
        << "Sub-partition keys should be in non-descending order";
  }

  // Verify we can look up by key
  ASSERT_FALSE(partitionKeys.empty());
  auto makeLookup4 = [](std::string_view key) {
    velox::serializer::EncodedKeyBounds bounds{
        .lowerKey = std::string(key), .upperKey = std::nullopt};
    return nimble::index::IndexLookup::LookupRequest::rangeScan({bounds});
  };
  auto firstResult = index->lookup(makeLookup4(index->minKey()));
  auto firstRanges = firstResult[0];
  ASSERT_EQ(firstRanges.size(), 1);
  EXPECT_EQ(firstRanges[0].startRow, 0);

  auto lastResult = index->lookup(makeLookup4(index->maxKey()));
  auto lastRanges = lastResult[0];
  ASSERT_EQ(lastRanges.size(), 1);
  EXPECT_GT(lastRanges[0].endRow, 0);

  // Read back all data and verify row-by-row match with written batches
  // Note: When reading back, string_col1 and string_col2 will have identical
  // content since they were written from the same source vector
  verifyFileData(file, type, batches);

  // Verify position index chunk row counts
  verifyPositionIndex(*tablet);

  // Verify value index maps each key to correct row position
  verifyValueIndex(*tablet, readFile.get(), type, batches, {"key_col"});
}

TEST_P(WriterIndexTest, streamStatsNoDuplicates) {
  auto type = velox::ROW({
      {"key_col", velox::BIGINT()},
      {"string_col1", velox::VARCHAR()},
      {"string_col2", velox::VARCHAR()},
  });

  auto clusterIndexConfig = createIndexConfig({"key_col"});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

  constexpr int kNumBatches = 5;
  constexpr int kBatchSize = 100;

  nimble::Writer writer(
      type,
      std::move(writeFile),
      *rootPool_,
      createWriterOptions(clusterIndexConfig, []() {
        return std::make_unique<nimble::LambdaFlushPolicy>(
            [](auto) { return true; }, [](auto) { return false; });
      }));

  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  std::vector<velox::RowVectorPtr> batches;
  int64_t keyVal = 0;
  for (int batch = 0; batch < kNumBatches; ++batch) {
    std::vector<int64_t> keyValues;
    keyValues.reserve(kBatchSize);
    for (int i = 0; i < kBatchSize; ++i) {
      keyValues.push_back(keyVal++);
    }

    std::vector<velox::VectorPtr> children;
    children.push_back(vectorMaker.flatVector<int64_t>(keyValues));
    children.push_back(vectorMaker.flatVector<std::string>(
        kBatchSize,
        [batch](auto row) { return fmt::format("left-{}-{}", batch, row); }));
    children.push_back(vectorMaker.flatVector<std::string>(
        kBatchSize,
        [batch](auto row) { return fmt::format("right-{}-{}", batch, row); }));

    auto batchVec = std::make_shared<velox::RowVector>(
        leafPool_.get(),
        type,
        nullptr, // no nulls at top level
        kBatchSize,
        std::move(children));
    batches.push_back(batchVec);
    writer.write(batchVec);
  }
  writer.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = nimble::TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  const auto expectedStats = duplicateStreamStatsFromLayout(*tablet);
  const auto writerStats = writer.runtimeStats();
  EXPECT_EQ(
      nimble::runtimeStat(
          writerStats, nimble::Writer::RuntimeStats::kDuplicateStreamCount)
          .sum,
      0);
  EXPECT_EQ(
      nimble::runtimeStat(
          writerStats, nimble::Writer::RuntimeStats::kDuplicateStreamBytes)
          .sum,
      0);
  EXPECT_EQ(expectedStats.duplicateStreamCount, 0);
  EXPECT_EQ(expectedStats.duplicateStreamBytes, 0);

  verifyFileData(file, type, batches);
  verifyPositionIndex(*tablet);
  verifyValueIndex(*tablet, readFile.get(), type, batches, {"key_col"});
}

// Test that custom prefixRestartInterval in ClusterIndexConfig flows through
// the entire E2E pipeline and is correctly applied to the PrefixEncoding.
TEST_F(WriterTest, customPrefixRestartInterval) {
  auto type = velox::ROW({
      {"key_col", velox::VARCHAR()},
      {"data_col", velox::INTEGER()},
  });

  // Test with various restart interval values including custom and default
  const std::vector<std::optional<uint32_t>> testRestartIntervals = {
      std::nullopt, // Test default (16)
      1,
      4,
      32,
  };

  for (const auto& customRestartInterval : testRestartIntervals) {
    SCOPED_TRACE(
        fmt::format(
            "prefixRestartInterval={}",
            customRestartInterval.has_value()
                ? std::to_string(customRestartInterval.value())
                : "default"));

    nimble::EncodingLayout::Config encodingConfig;
    if (customRestartInterval.has_value()) {
      encodingConfig = nimble::EncodingLayout::Config{
          {{std::string(nimble::PrefixEncoding::kRestartIntervalConfigKey),
            std::to_string(customRestartInterval.value())}}};
    }

    auto clusterIndexConfig =
        nimble::index::ClusterIndexConfigBuilder{}
            .withKeyColumns({"key_col"})
            .withEnforceKeyOrder(true)
            .withEncodingLayout(
                nimble::EncodingLayout{
                    nimble::EncodingType::Prefix,
                    std::move(encodingConfig),
                    nimble::CompressionType::Uncompressed})
            .build();

    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);

    nimble::Writer writer(
        type,
        std::move(writeFile),
        *rootPool_,
        {.clusterIndexConfig = std::move(clusterIndexConfig)});

    velox::test::VectorMaker vectorMaker{leafPool_.get()};

    // Create sorted key values
    auto batch = vectorMaker.rowVector(
        {"key_col", "data_col"},
        {vectorMaker.flatVector<std::string>(
             {"aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"}),
         vectorMaker.flatVector<int32_t>({1, 2, 3, 4, 5, 6, 7, 8})});

    writer.write(batch);
    writer.close();

    // Read back and verify the restart interval in the key encoding
    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));

    const auto* index = tablet->clusterIndex();
    ASSERT_NE(index, nullptr) << "Index must exist";

    // Get key stream region from partition 0
    nimble::index::test::ClusterIndexTestHelper indexHelper(index);
    const auto keyStreamRegion = indexHelper.keyStreamRegion(0);

    // Load the key stream data
    velox::common::Region region{
        keyStreamRegion.offset, keyStreamRegion.length, "keyStream"};
    folly::IOBuf iobuf;
    readFile->preadv({&region, 1}, {&iobuf, 1});

    std::string buffer;
    buffer.resize(iobuf.computeChainDataLength());
    size_t offset = 0;
    for (auto range : iobuf) {
      std::memcpy(buffer.data() + offset, range.data(), range.size());
      offset += range.size();
    }

    // Decode the chunk to get raw encoding data
    nimble::index::test::SingleChunkDecoder chunkDecoder(*leafPool_, buffer);
    auto encodingData = chunkDecoder.decode();

    // Decode the key encoding
    std::vector<velox::BufferPtr> stringBuffers;
    auto keyEncoding = nimble::EncodingFactory().create(
        *leafPool_,
        encodingData,
        [&](uint32_t totalLength) {
          auto& buf = stringBuffers.emplace_back(
              velox::AlignedBuffer::allocate<char>(
                  totalLength, leafPool_.get()));
          return buf->asMutable<void>();
        },
        nimble::Encoding::Options{});

    // Verify the restart interval through debugString()
    const std::string debug = keyEncoding->debugString(0);
    const uint32_t expectedInterval = customRestartInterval.value_or(
        nimble::PrefixEncoding::kDefaultRestartInterval);
    EXPECT_TRUE(
        debug.find("restart_interval=" + std::to_string(expectedInterval)) !=
        std::string::npos)
        << "Expected restart_interval=" << expectedInterval
        << " in debugString: " << debug;
  }
}

INSTANTIATE_TEST_SUITE_P(
    WriterIndexTestSuite,
    WriterIndexTest,
    ::testing::Values(
        IndexTestParams{false, nimble::EncodingType::Prefix, std::nullopt},
        IndexTestParams{false, nimble::EncodingType::Prefix, 1},
        IndexTestParams{false, nimble::EncodingType::Prefix, 1024},
        IndexTestParams{true, nimble::EncodingType::Prefix, std::nullopt},
        IndexTestParams{true, nimble::EncodingType::Prefix, 1},
        IndexTestParams{true, nimble::EncodingType::Prefix, 1024},
        IndexTestParams{false, nimble::EncodingType::Trivial, std::nullopt},
        IndexTestParams{true, nimble::EncodingType::Trivial, std::nullopt}),
    [](const ::testing::TestParamInfo<IndexTestParams>& info) {
      return info.param.toString();
    });

TEST_F(WriterTest, disableStatsCollection) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  // Build a schema covering major field writer types:
  // scalar, string, timestamp, array, map, flatmap, row.
  auto type = velox::ROW(
      {{"int_col", velox::INTEGER()},
       {"string_col", velox::VARCHAR()},
       {"ts_col", velox::TIMESTAMP()},
       {"array_col", velox::ARRAY(velox::BIGINT())},
       {"map_col", velox::MAP(velox::INTEGER(), velox::VARCHAR())},
       {"flatmap_col", velox::MAP(velox::VARCHAR(), velox::BIGINT())}});

  auto vector = vectorMaker.rowVector(
      {"int_col",
       "string_col",
       "ts_col",
       "array_col",
       "map_col",
       "flatmap_col"},
      {vectorMaker.flatVector<int32_t>({1, 2, 3}),
       vectorMaker.flatVector<velox::StringView>({"a", "bb", "ccc"}),
       vectorMaker.flatVector<velox::Timestamp>(
           {velox::Timestamp(1, 0),
            velox::Timestamp(2, 0),
            velox::Timestamp(3, 0)}),
       vectorMaker.arrayVector<int64_t>({{1, 2}, {3}, {4, 5, 6}}),
       vectorMaker.mapVector<int32_t, velox::StringView>(
           {{{1, "x"}}, {{2, "y"}, {3, "z"}}, {{4, "w"}}}),
       vectorMaker.mapVector<velox::StringView, int64_t>(
           {{{"k1", 10}}, {{"k2", 20}}, {{"k1", 30}, {"k3", 40}}})});

  // Write with stats collection disabled.
  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.enableStatsCollection = false;
    options.flatMapColumns = {{"flatmap_col", {}}};
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));
    writer.write(vector);

    // Poll stats mid-write, before flush. columnStats must be empty.
    EXPECT_TRUE(writer.columnStats().empty());

    writer.flush();

    // Poll stats after flush, before close.
    {
      EXPECT_TRUE(writer.columnStats().empty());
      EXPECT_EQ(
          nimble::runtimeStat(
              writer.runtimeStats(),
              nimble::Writer::RuntimeStats::kRowsPerStripe)
              .count,
          1);
    }

    // Write a second batch to exercise multi-stripe path.
    writer.write(vector);
    writer.close();

    // Poll stats after close.
    EXPECT_TRUE(writer.columnStats().empty());
    EXPECT_EQ(
        nimble::runtimeStat(
            writer.runtimeStats(), nimble::Writer::RuntimeStats::kRowsPerStripe)
            .count,
        2);
  }

  // Verify the file is readable and data round-trips.
  {
    velox::InMemoryReadFile readFile(file);
    nimble::BatchReader reader(&readFile, *leafPool_);
    velox::VectorPtr result;
    uint64_t totalRows = 0;
    while (reader.next(100, result)) {
      totalRows += result->size();
    }
    EXPECT_EQ(totalRows, 6);
  }
}

TEST_F(WriterTest, disableStatsCollectionWithChunking) {
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  auto type = velox::ROW({{"col0", velox::INTEGER()}});
  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int32_t>({1, 2, 3})});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::WriterOptions options;
  options.enableStatsCollection = false;
  options.enableChunking = true;
  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(options));
  writer.write(vector);
  writer.close();

  using NimbleStats = nimble::Writer::RuntimeStats;
  const auto stats = writer.runtimeStats();
  EXPECT_TRUE(writer.columnStats().empty());
  EXPECT_EQ(nimble::runtimeStat(stats, NimbleStats::kRowsPerStripe).count, 1);
  EXPECT_GT(nimble::runtimeStat(stats, NimbleStats::kWrittenBytes).sum, 0);
}

namespace {

nimble::EncodingLayout deltaEncodingLayout() {
  return nimble::EncodingLayout{
      nimble::EncodingType::Delta,
      {},
      nimble::CompressionType::Uncompressed,
      {
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
          nimble::EncodingLayout{
              nimble::EncodingType::Trivial,
              {},
              nimble::CompressionType::Uncompressed},
      }};
}

nimble::EncodingLayoutTree singleScalarLayoutTree(
    nimble::EncodingLayout layout) {
  return nimble::EncodingLayoutTree{
      nimble::Kind::Row,
      {},
      "",
      {{nimble::Kind::Scalar, {{0, std::move(layout)}}, ""}}};
}

nimble::BatchReadParams nonLegacyReadParams() {
  nimble::BatchReadParams params;
  params.encodingFactory =
      [](velox::memory::MemoryPool& pool,
         std::string_view data,
         std::function<void*(uint32_t)> stringBufferFactory)
      -> std::unique_ptr<nimble::Encoding> {
    return nimble::EncodingFactory().create(
        pool,
        data,
        std::move(stringBufferFactory),
        nimble::Encoding::Options{});
  };
  return params;
}

void verifyDeltaEncoding(
    const std::string& file,
    velox::memory::MemoryPool& pool,
    bool checkChildren = false) {
  auto readFile =
      std::make_shared<nimble::testing::InMemoryTrackableReadFile>(file, false);
  auto tablet = nimble::TabletReader::create(
      readFile, &pool, makeTestTabletOptions(&pool));
  auto section =
      tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
  NIMBLE_CHECK(section.has_value(), "Schema not found.");
  auto schema =
      nimble::SchemaDeserializer::deserialize(section->content().data());
  auto& scalarNode = schema->asRow().childAt(0)->asScalar();

  for (auto i = 0; i < tablet->stripeCount(); ++i) {
    auto stripeIdentifier = tablet->stripeIdentifier(i);
    std::vector<uint32_t> identifiers{scalarNode.scalarDescriptor().offset()};
    auto streams = tablet->load(stripeIdentifier, identifiers);

    nimble::InMemoryChunkedStream chunkedStream{pool, std::move(streams[0])};
    ASSERT_TRUE(chunkedStream.hasNext());
    auto capture = nimble::EncodingLayoutCapture::capture(
        chunkedStream.nextChunk(), nimble::Encoding::Options{});
    EXPECT_EQ(nimble::EncodingType::Delta, capture.encodingType())
        << "Stripe " << i;
    if (checkChildren) {
      EXPECT_EQ(
          nimble::EncodingType::Trivial,
          capture.child(nimble::EncodingIdentifiers::Delta::Deltas)
              ->encodingType());
      EXPECT_EQ(
          nimble::EncodingType::Trivial,
          capture.child(nimble::EncodingIdentifiers::Delta::Restatements)
              ->encodingType());
      EXPECT_EQ(
          nimble::EncodingType::Trivial,
          capture.child(nimble::EncodingIdentifiers::Delta::IsRestatements)
              ->encodingType());
    }
  }
}

} // namespace

// Monotonically increasing int64 data. Verifies Delta encoding structure
// (including child encodings) and round-trip data correctness.
TEST_F(WriterTest, encodingLayoutDelta) {
  constexpr int32_t kRowCount = 100;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int64_t>(kRowCount, [](auto row) {
        return 10 + row * 3;
      })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  for (auto useChainedBuffers : {false, true}) {
    auto readFile =
        std::make_shared<nimble::testing::InMemoryTrackableReadFile>(
            file, useChainedBuffers);
    auto tablet = nimble::TabletReader::create(
        readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
    auto section =
        tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
    NIMBLE_CHECK(section.has_value(), "Schema not found.");
    auto schema =
        nimble::SchemaDeserializer::deserialize(section->content().data());
    auto& scalarNode = schema->asRow().childAt(0)->asScalar();

    for (auto i = 0; i < tablet->stripeCount(); ++i) {
      auto stripeIdentifier = tablet->stripeIdentifier(i);
      std::vector<uint32_t> identifiers{scalarNode.scalarDescriptor().offset()};
      auto streams = tablet->load(stripeIdentifier, identifiers);

      nimble::InMemoryChunkedStream chunkedStream{
          *leafPool_, std::move(streams[0])};
      ASSERT_TRUE(chunkedStream.hasNext());
      auto capture = nimble::EncodingLayoutCapture::capture(
          chunkedStream.nextChunk(), nimble::Encoding::Options{});
      EXPECT_EQ(nimble::EncodingType::Delta, capture.encodingType());
      EXPECT_EQ(
          nimble::EncodingType::Trivial,
          capture.child(nimble::EncodingIdentifiers::Delta::Deltas)
              ->encodingType());
      EXPECT_EQ(
          nimble::EncodingType::Trivial,
          capture.child(nimble::EncodingIdentifiers::Delta::Restatements)
              ->encodingType());
      EXPECT_EQ(
          nimble::EncodingType::Trivial,
          capture.child(nimble::EncodingIdentifiers::Delta::IsRestatements)
              ->encodingType());
    }
  }

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(
      &readFile, *leafPool_, nullptr, nonLegacyReadParams());
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i));
  }
}

// Periodic large jumps trigger restatements in delta encoding.
TEST_F(WriterTest, encodingLayoutDeltaWithRestatements) {
  constexpr int32_t kRowCount = 100;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int64_t>(kRowCount, [](auto row) {
        int64_t segment = row / 10;
        int64_t offset = row % 10;
        return segment * 1000 + offset * 3;
      })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  verifyDeltaEncoding(file, *leafPool_);

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(
      &readFile, *leafPool_, nullptr, nonLegacyReadParams());
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i));
  }
}

// Verifies Delta encoding works with int32_t values.
TEST_F(WriterTest, encodingLayoutDeltaInt32) {
  constexpr int32_t kRowCount = 200;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int32_t>(kRowCount, [](auto row) {
        return 5 + row * 2;
      })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  verifyDeltaEncoding(file, *leafPool_, true);

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(
      &readFile, *leafPool_, nullptr, nonLegacyReadParams());
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i));
  }
}

// Negative values with zero crossings exercise the signed overflow
// restatement protection in DeltaEncoding::computeDeltas.
TEST_F(WriterTest, encodingLayoutDeltaNegativeValues) {
  constexpr int32_t kRowCount = 200;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int64_t>(kRowCount, [](auto row) {
        // Oscillate across zero with periodic jumps.
        int64_t segment = row / 50;
        int64_t offset = row % 50;
        return static_cast<int64_t>(-100 + offset * 3 + segment * 20);
      })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  verifyDeltaEncoding(file, *leafPool_);

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(
      &readFile, *leafPool_, nullptr, nonLegacyReadParams());
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i));
  }
}

// Nullable column with Delta encoding nested inside a Nullable encoding.
TEST_F(WriterTest, encodingLayoutDeltaNullable) {
  constexpr int32_t kRowCount = 150;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  std::vector<std::optional<int64_t>> data(kRowCount);
  for (int32_t i = 0; i < kRowCount; ++i) {
    if (i % 7 == 0) {
      data[i] = std::nullopt;
    } else {
      data[i] = 10 + static_cast<int64_t>(i) * 3;
    }
  }

  auto vector =
      vectorMaker.rowVector({"col0"}, {vectorMaker.flatVectorNullable(data)});

  // The layout specifies the data encoding only. The writer automatically
  // applies Nullable wrapping when nulls are present.
  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  // Nimble stores nulls as a separate stream, so the data stream should
  // contain only the non-null values encoded with Delta.
  verifyDeltaEncoding(file, *leafPool_, true);

  velox::InMemoryReadFile readFile2(file);
  nimble::BatchReader reader(
      &readFile2, *leafPool_, nullptr, nonLegacyReadParams());
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i));
  }
}

// Multiple stripes with a small flush threshold. Verifies Delta encoding
// is consistent across all stripes.
TEST_F(WriterTest, encodingLayoutDeltaMultiStripe) {
  constexpr int32_t kBatchSize = 500;
  constexpr int32_t kBatchCount = 4;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  std::vector<velox::RowVectorPtr> batches;
  for (int32_t b = 0; b < kBatchCount; ++b) {
    batches.push_back(vectorMaker.rowVector(
        {"col0"}, {vectorMaker.flatVector<int64_t>(kBatchSize, [b](auto row) {
          return static_cast<int64_t>(b) * 10000 + row * 3;
        })}));
  }

  auto layoutTree = singleScalarLayoutTree(deltaEncodingLayout());

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      batches[0]->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = std::move(layoutTree), .flushPolicyFactory = []() {
         return std::make_unique<nimble::StripeRawSizeFlushPolicy>(1024);
       }});

  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();

  auto readFile =
      std::make_shared<nimble::testing::InMemoryTrackableReadFile>(file, false);
  auto tablet = nimble::TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  ASSERT_GT(tablet->stripeCount(), 1);

  verifyDeltaEncoding(file, *leafPool_, true);

  // Read back all rows across stripes.
  velox::InMemoryReadFile readFile2(file);
  nimble::BatchReader reader(
      &readFile2, *leafPool_, nullptr, nonLegacyReadParams());
  int32_t totalRows = 0;
  int32_t batchIdx = 0;
  int32_t batchOffset = 0;
  velox::VectorPtr result;
  while (reader.next(kBatchSize, result)) {
    for (int32_t i = 0; i < result->size(); ++i) {
      ASSERT_TRUE(result->equalValueAt(batches[batchIdx].get(), i, batchOffset))
          << "Mismatch at global row " << totalRows;
      ++batchOffset;
      ++totalRows;
      if (batchOffset == kBatchSize) {
        ++batchIdx;
        batchOffset = 0;
      }
    }
  }
  EXPECT_EQ(totalRows, kBatchSize * kBatchCount);
}

// Two columns: col0 uses Delta, col1 uses MainlyConstant. Verifies mixed
// encoding layouts in the same file work correctly.
TEST_F(WriterTest, encodingLayoutDeltaMultiColumn) {
  constexpr int32_t kRowCount = 100;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0", "col1"},
      {vectorMaker.flatVector<int64_t>(
           kRowCount, [](auto row) { return row * 5; }),
       vectorMaker.flatVector<int32_t>(
           kRowCount, [](auto row) { return row % 20 == 0 ? row : 42; })});

  nimble::EncodingLayoutTree layoutTree{
      nimble::Kind::Row,
      {},
      "",
      {
          {nimble::Kind::Scalar, {{0, deltaEncodingLayout()}}, ""},
          {nimble::Kind::Scalar,
           {{0,
             nimble::EncodingLayout{
                 nimble::EncodingType::MainlyConstant,
                 {},
                 nimble::CompressionType::Uncompressed,
                 {
                     nimble::EncodingLayout{
                         nimble::EncodingType::Trivial,
                         {},
                         nimble::CompressionType::Uncompressed},
                     nimble::EncodingLayout{
                         nimble::EncodingType::Trivial,
                         {},
                         nimble::CompressionType::Uncompressed},
                 }}}},
           ""},
      }};

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = std::move(layoutTree)});
  writer.write(vector);
  writer.close();

  // Verify both column encodings.
  auto readFile =
      std::make_shared<nimble::testing::InMemoryTrackableReadFile>(file, false);
  auto tablet = nimble::TabletReader::create(
      readFile, leafPool_.get(), makeTestTabletOptions(leafPool_.get()));
  auto section =
      tablet->loadOptionalSection(std::string(nimble::kSchemaSection));
  NIMBLE_CHECK(section.has_value(), "Schema not found.");
  auto schema =
      nimble::SchemaDeserializer::deserialize(section->content().data());
  auto& col0Node = schema->asRow().childAt(0)->asScalar();
  auto& col1Node = schema->asRow().childAt(1)->asScalar();

  for (auto i = 0; i < tablet->stripeCount(); ++i) {
    auto stripeIdentifier = tablet->stripeIdentifier(i);
    std::vector<uint32_t> identifiers{
        col0Node.scalarDescriptor().offset(),
        col1Node.scalarDescriptor().offset()};
    auto streams = tablet->load(stripeIdentifier, identifiers);

    {
      nimble::InMemoryChunkedStream chunkedStream{
          *leafPool_, std::move(streams[0])};
      ASSERT_TRUE(chunkedStream.hasNext());
      auto capture = nimble::EncodingLayoutCapture::capture(
          chunkedStream.nextChunk(), nimble::Encoding::Options{});
      EXPECT_EQ(nimble::EncodingType::Delta, capture.encodingType());
    }
    {
      nimble::InMemoryChunkedStream chunkedStream{
          *leafPool_, std::move(streams[1])};
      ASSERT_TRUE(chunkedStream.hasNext());
      auto capture = nimble::EncodingLayoutCapture::capture(
          chunkedStream.nextChunk(), nimble::Encoding::Options{});
      EXPECT_EQ(nimble::EncodingType::MainlyConstant, capture.encodingType());
    }
  }

  velox::InMemoryReadFile readFile2(file);
  nimble::BatchReader reader(
      &readFile2, *leafPool_, nullptr, nonLegacyReadParams());
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i));
  }
}

// Multiple write() calls produce a single stripe, verifying that Delta
// encoding handles data accumulated across batches.
TEST_F(WriterTest, encodingLayoutDeltaMultipleBatches) {
  constexpr int32_t kBatchSize = 50;
  constexpr int32_t kBatchCount = 5;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  std::vector<velox::RowVectorPtr> batches;
  for (int32_t b = 0; b < kBatchCount; ++b) {
    batches.push_back(vectorMaker.rowVector(
        {"col0"}, {vectorMaker.flatVector<int64_t>(kBatchSize, [b](auto row) {
          return static_cast<int64_t>(b * kBatchSize + row) * 7;
        })}));
  }

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      batches[0]->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});

  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();

  verifyDeltaEncoding(file, *leafPool_, true);

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(
      &readFile, *leafPool_, nullptr, nonLegacyReadParams());
  velox::VectorPtr result;
  int32_t totalRows = 0;
  int32_t batchIdx = 0;
  int32_t batchOffset = 0;
  while (reader.next(kBatchSize, result)) {
    for (int32_t i = 0; i < result->size(); ++i) {
      ASSERT_TRUE(result->equalValueAt(batches[batchIdx].get(), i, batchOffset))
          << "Mismatch at global row " << totalRows;
      ++batchOffset;
      ++totalRows;
      if (batchOffset == kBatchSize) {
        ++batchIdx;
        batchOffset = 0;
      }
    }
  }
  EXPECT_EQ(totalRows, kBatchSize * kBatchCount);
}

// Writes delta-encoded data and reads it back using the default (legacy)
// encoding factory, verifying the legacy DeltaEncoding decoder works e2e.
TEST_F(WriterTest, encodingLayoutDeltaLegacyRead) {
  constexpr int32_t kRowCount = 100;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int64_t>(kRowCount, [](auto row) {
        return 10 + row * 3;
      })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  verifyDeltaEncoding(file, *leafPool_, true);

  // Read using default BatchReader (legacy encoding factory).
  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i))
        << "Mismatch at row " << i;
  }
}

// Writes delta-encoded data with restatements (non-monotonic) and reads it
// back using the default (legacy) encoding factory.
TEST_F(WriterTest, encodingLayoutDeltaWithRestatementsLegacyRead) {
  constexpr int32_t kRowCount = 100;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int64_t>(kRowCount, [](auto row) {
        int64_t segment = row / 10;
        int64_t offset = row % 10;
        return segment * 1000 + offset * 3;
      })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  verifyDeltaEncoding(file, *leafPool_);

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i))
        << "Mismatch at row " << i;
  }
}

// Writes delta-encoded int32 data across multiple stripes and reads back
// using the default (legacy) encoding factory.
TEST_F(WriterTest, encodingLayoutDeltaMultiStripeLegacyRead) {
  constexpr int32_t kBatchSize = 500;
  constexpr int32_t kBatchCount = 4;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  std::vector<velox::RowVectorPtr> batches;
  for (int32_t b = 0; b < kBatchCount; ++b) {
    batches.push_back(vectorMaker.rowVector(
        {"col0"}, {vectorMaker.flatVector<int32_t>(kBatchSize, [b](auto row) {
          return static_cast<int32_t>(b) * 10000 + row * 2;
        })}));
  }

  auto layoutTree = singleScalarLayoutTree(deltaEncodingLayout());

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      batches[0]->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = std::move(layoutTree), .flushPolicyFactory = []() {
         return std::make_unique<nimble::StripeRawSizeFlushPolicy>(1024);
       }});

  for (const auto& batch : batches) {
    writer.write(batch);
  }
  writer.close();

  auto readFileTrackable =
      std::make_shared<nimble::testing::InMemoryTrackableReadFile>(file, false);
  auto tablet = nimble::TabletReader::create(
      readFileTrackable,
      leafPool_.get(),
      makeTestTabletOptions(leafPool_.get()));
  ASSERT_GT(tablet->stripeCount(), 1);

  verifyDeltaEncoding(file, *leafPool_, true);

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);
  int32_t totalRows = 0;
  int32_t batchIdx = 0;
  int32_t batchOffset = 0;
  velox::VectorPtr result;
  while (reader.next(kBatchSize, result)) {
    for (int32_t i = 0; i < result->size(); ++i) {
      ASSERT_TRUE(result->equalValueAt(batches[batchIdx].get(), i, batchOffset))
          << "Mismatch at global row " << totalRows;
      ++batchOffset;
      ++totalRows;
      if (batchOffset == kBatchSize) {
        ++batchIdx;
        batchOffset = 0;
      }
    }
  }
  EXPECT_EQ(totalRows, kBatchSize * kBatchCount);
}

// Delta encoding with int16_t (smallint) values.
TEST_F(WriterTest, encodingLayoutDeltaInt16) {
  constexpr int32_t kRowCount = 300;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int16_t>(kRowCount, [](auto row) {
        return static_cast<int16_t>(row * 3);
      })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  verifyDeltaEncoding(file, *leafPool_, true);

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(
      &readFile, *leafPool_, nullptr, nonLegacyReadParams());
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i));
  }
}

// Delta encoding with seekToRow: write, seek to middle, read remainder.
TEST_F(WriterTest, encodingLayoutDeltaSeekToRow) {
  constexpr int32_t kRowCount = 200;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int64_t>(kRowCount, [](auto row) {
        return 100 + row * 5;
      })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  verifyDeltaEncoding(file, *leafPool_);

  // Read using default (legacy) BatchReader with seekToRow.
  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);

  constexpr int32_t kSeekRow = 100;
  reader.seekToRow(kSeekRow);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount - kSeekRow, result));
  ASSERT_EQ(result->size(), kRowCount - kSeekRow);
  for (auto i = 0; i < result->size(); ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, kSeekRow + i))
        << "Mismatch at row " << i << " (global " << kSeekRow + i << ")";
  }
}

// Delta encoding with nullable data read using legacy reader.
TEST_F(WriterTest, encodingLayoutDeltaNullableLegacyRead) {
  constexpr int32_t kRowCount = 150;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  std::vector<std::optional<int64_t>> data(kRowCount);
  for (int32_t i = 0; i < kRowCount; ++i) {
    if (i % 7 == 0) {
      data[i] = std::nullopt;
    } else {
      data[i] = 10 + static_cast<int64_t>(i) * 3;
    }
  }

  auto vector =
      vectorMaker.rowVector({"col0"}, {vectorMaker.flatVectorNullable(data)});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  verifyDeltaEncoding(file, *leafPool_, true);

  // Read using default BatchReader (legacy encoding factory).
  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(&readFile, *leafPool_);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i))
        << "Mismatch at row " << i;
  }
}

// Delta encoding: sawtooth pattern triggers frequent restatements.
TEST_F(WriterTest, encodingLayoutDeltaSawtooth) {
  constexpr int32_t kRowCount = 500;
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto vector = vectorMaker.rowVector(
      {"col0"}, {vectorMaker.flatVector<int64_t>(kRowCount, [](auto row) {
        // Sawtooth: rises to 100 then resets every 20 rows.
        return static_cast<int64_t>((row % 20) * 5);
      })});

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(
      vector->type(),
      std::move(writeFile),
      *rootPool_,
      {.encodingLayoutTree = singleScalarLayoutTree(deltaEncodingLayout())});
  writer.write(vector);
  writer.close();

  verifyDeltaEncoding(file, *leafPool_, true);

  velox::InMemoryReadFile readFile(file);
  nimble::BatchReader reader(
      &readFile, *leafPool_, nullptr, nonLegacyReadParams());
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kRowCount, result));
  ASSERT_EQ(result->size(), kRowCount);
  for (auto i = 0; i < kRowCount; ++i) {
    ASSERT_TRUE(result->equalValueAt(vector.get(), i, i));
  }
}

TEST_F(WriterTest, flatmapColumnsKeysSchemaConsistency) {
  // Two writers with the same 10 predefined keys but different data arrival
  // order should produce identical schemas. 500 rows each, 5 batches.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto type = velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::REAL())}});

  const std::vector<std::string> predefinedKeysList = {
      "20", "5", "13", "8", "17", "2", "11", "19", "7", "15"};
  const std::set<std::string> predefinedKeys(
      predefinedKeysList.begin(), predefinedKeysList.end());
  constexpr int32_t kNumRows = 500;
  constexpr int32_t kNumKeys = 10;
  constexpr int32_t kBatches = 5;
  constexpr int32_t kRowsPerBatch = kNumRows / kBatches;

  auto writeWithKeyOrder = [&](bool reverseKeys) -> std::string {
    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.flatMapColumns = {{"m", predefinedKeys}};
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));

    for (int32_t b = 0; b < kBatches; ++b) {
      auto mapVec = vectorMaker.mapVector<int32_t, float>(
          kRowsPerBatch,
          /* sizeAt */ [](auto row) { return (row % kNumKeys) + 1; },
          /* keyAt */
          [&](auto /*row*/, auto mapIndex) {
            int32_t keyIdx = reverseKeys ? (kNumKeys - 1 - mapIndex) % kNumKeys
                                         : mapIndex % kNumKeys;
            return folly::to<int32_t>(predefinedKeysList[keyIdx]);
          },
          /* valueAt */
          [](auto /*row*/, auto mapIndex) {
            return static_cast<float>(mapIndex);
          },
          /* isNullAt */ [](auto row) { return row % 50 == 0; });
      writer.write(vectorMaker.rowVector({"m"}, {mapVec}));
    }
    writer.close();
    return file;
  };

  auto file1 = writeWithKeyOrder(false);
  auto file2 = writeWithKeyOrder(true);

  nimble::BatchReader reader1(
      std::make_shared<velox::InMemoryReadFile>(file1), *leafPool_);
  nimble::BatchReader reader2(
      std::make_shared<velox::InMemoryReadFile>(file2), *leafPool_);

  const auto& flatMap1 = reader1.schema()->asRow().childAt(0)->asFlatMap();
  const auto& flatMap2 = reader2.schema()->asRow().childAt(0)->asFlatMap();

  ASSERT_EQ(flatMap1.childrenCount(), flatMap2.childrenCount());
  ASSERT_EQ(flatMap1.childrenCount(), kNumKeys);
  std::vector<std::string> sortedKeys(
      predefinedKeys.begin(), predefinedKeys.end());
  for (uint32_t i = 0; i < flatMap1.childrenCount(); ++i) {
    EXPECT_EQ(flatMap1.nameAt(i), flatMap2.nameAt(i));
    EXPECT_EQ(flatMap1.nameAt(i), sortedKeys[i]);
  }
}

TEST_F(WriterTest, flatmapColumnsKeysRoundtrip) {
  // Write 1000 rows with 8 predefined keys across 4 batches, verify roundtrip.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::BIGINT())}});

  const std::vector<std::string> predefinedKeysList = {
      "10", "3", "7", "15", "1", "20", "8", "12"};
  const std::set<std::string> predefinedKeys(
      predefinedKeysList.begin(), predefinedKeysList.end());
  constexpr int32_t kNumKeys = 8;
  constexpr int32_t kRowsPerBatch = 250;
  constexpr int32_t kBatches = 4;
  constexpr int32_t kTotalRows = kRowsPerBatch * kBatches;

  // Collect all written batches for verification.
  auto expected = velox::BaseVector::create(type, 0, leafPool_.get());

  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.flatMapColumns = {{"m", predefinedKeys}};
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));

    for (int32_t b = 0; b < kBatches; ++b) {
      auto mapVec = vectorMaker.mapVector<int32_t, int64_t>(
          kRowsPerBatch,
          /* sizeAt */
          [](auto row) { return (row % kNumKeys) + 1; },
          /* keyAt */
          [&](auto /*row*/, auto mapIndex) {
            return folly::to<int32_t>(predefinedKeysList[mapIndex % kNumKeys]);
          },
          /* valueAt */
          [&](auto row, auto mapIndex) {
            return static_cast<int64_t>(
                (b * kRowsPerBatch + row) * 100 + mapIndex);
          },
          /* isNullAt */ [](auto row) { return row % 100 == 0; });
      auto batch = vectorMaker.rowVector({"m"}, {mapVec});
      writer.write(batch);
      expected->append(batch.get());
    }
    writer.close();
  }

  nimble::BatchReader reader(
      std::make_shared<velox::InMemoryReadFile>(file), *leafPool_);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kTotalRows, result));
  ASSERT_EQ(result->size(), kTotalRows);
  for (auto i = 0; i < kTotalRows; ++i) {
    ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
        << "Mismatch at row " << i;
  }
}

TEST_F(WriterTest, flatmapColumnsKeysVarchar) {
  // Test with 6 VARCHAR keys, 500 rows, 3 batches to verify StringView
  // lifetime correctness across multiple batches.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto type =
      velox::ROW({{"m", velox::MAP(velox::VARCHAR(), velox::INTEGER())}});

  const std::vector<std::string> predefinedKeysList = {
      "alpha", "beta", "gamma", "delta", "epsilon", "zeta"};
  const std::set<std::string> predefinedKeys(
      predefinedKeysList.begin(), predefinedKeysList.end());
  constexpr int32_t kNumKeys = 6;
  constexpr int32_t kRowsPerBatch = 500;
  constexpr int32_t kBatches = 3;
  constexpr int32_t kTotalRows = kRowsPerBatch * kBatches;

  auto expected = velox::BaseVector::create(type, 0, leafPool_.get());

  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.flatMapColumns = {{"m", predefinedKeys}};
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));

    for (int32_t b = 0; b < kBatches; ++b) {
      auto mapVec = vectorMaker.mapVector<velox::StringView, int32_t>(
          kRowsPerBatch,
          /* sizeAt */ [](auto row) { return (row % kNumKeys) + 1; },
          /* keyAt */
          [&](auto /*row*/, auto mapIndex) {
            return velox::StringView(predefinedKeysList[mapIndex % kNumKeys]);
          },
          /* valueAt */
          [&](auto /*row*/, auto mapIndex) {
            return static_cast<int32_t>(b * kRowsPerBatch + mapIndex);
          },
          /* isNullAt */ [](auto row) { return row % 80 == 0; });
      auto batch = vectorMaker.rowVector({"m"}, {mapVec});
      writer.write(batch);
      expected->append(batch.get());
    }
    writer.close();
  }

  nimble::BatchReader reader(
      std::make_shared<velox::InMemoryReadFile>(file), *leafPool_);

  // Verify schema key order.
  const auto& flatMap = reader.schema()->asRow().childAt(0)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), kNumKeys);
  {
    std::vector<std::string> sortedKeys(
        predefinedKeys.begin(), predefinedKeys.end());
    for (int i = 0; i < kNumKeys; ++i) {
      EXPECT_EQ(flatMap.nameAt(i), sortedKeys[i]);
    }
  }

  // Verify data roundtrip.
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kTotalRows, result));
  ASSERT_EQ(result->size(), kTotalRows);
  for (auto i = 0; i < kTotalRows; ++i) {
    ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
        << "Mismatch at row " << i;
  }
}

TEST_F(WriterTest, flatmapColumnsKeysMultipleBatches) {
  // Write 10 batches of 200 rows each with 8 predefined keys. Each batch
  // uses a different subset of keys to exercise the pre-registration path.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::INTEGER())}});

  const std::vector<std::string> predefinedKeysList = {
      "1", "2", "3", "4", "5", "6", "7", "8"};
  const std::set<std::string> predefinedKeys(
      predefinedKeysList.begin(), predefinedKeysList.end());
  constexpr int32_t kNumKeys = 8;
  constexpr int32_t kRowsPerBatch = 200;
  constexpr int32_t kBatches = 10;
  constexpr int32_t kTotalRows = kRowsPerBatch * kBatches;

  auto expected = velox::BaseVector::create(type, 0, leafPool_.get());

  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.flatMapColumns = {{"m", predefinedKeys}};
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));

    for (int32_t b = 0; b < kBatches; ++b) {
      // Each batch uses a sliding window of keys starting at offset b.
      auto mapVec = vectorMaker.mapVector<int32_t, int32_t>(
          kRowsPerBatch,
          /* sizeAt */
          [&](auto row) {
            // Vary map size: 1 to (b % kNumKeys + 1)
            return (row % (b % kNumKeys + 1)) + 1;
          },
          /* keyAt */
          [&](auto /*row*/, auto mapIndex) {
            return folly::to<int32_t>(
                predefinedKeysList[(b + mapIndex) % kNumKeys]);
          },
          /* valueAt */
          [&](auto row, auto mapIndex) {
            return static_cast<int32_t>(b * 1000 + row * 10 + mapIndex);
          },
          /* isNullAt */ [](auto row) { return row % 40 == 0; });
      auto batch = vectorMaker.rowVector({"m"}, {mapVec});
      writer.write(batch);
      expected->append(batch.get());
    }
    writer.close();
  }

  nimble::BatchReader reader(
      std::make_shared<velox::InMemoryReadFile>(file), *leafPool_);

  // Schema should have all 8 keys in predefined (sorted) order.
  const auto& flatMap = reader.schema()->asRow().childAt(0)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), kNumKeys);
  {
    std::vector<std::string> sortedKeys(
        predefinedKeys.begin(), predefinedKeys.end());
    for (int i = 0; i < kNumKeys; ++i) {
      EXPECT_EQ(flatMap.nameAt(i), sortedKeys[i]);
    }
  }

  // Verify all rows roundtrip.
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kTotalRows, result));
  ASSERT_EQ(result->size(), kTotalRows);
  for (auto i = 0; i < kTotalRows; ++i) {
    ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
        << "Mismatch at row " << i;
  }
}

TEST_F(WriterTest, flatmapColumnsKeysEmptyData) {
  // Pre-register 10 keys, write 200 rows of empty maps across 4 batches.
  // Schema should still contain all predefined keys.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto type = velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::REAL())}});

  const std::set<std::string> predefinedKeys = {
      "5", "3", "7", "11", "2", "9", "14", "1", "18", "6"};
  constexpr int32_t kNumKeys = 10;
  constexpr int32_t kRowsPerBatch = 50;
  constexpr int32_t kBatches = 4;

  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    options.flatMapColumns = {{"m", predefinedKeys}};
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));

    for (int32_t b = 0; b < kBatches; ++b) {
      auto mapVec = vectorMaker.mapVector<int32_t, float>(
          kRowsPerBatch,
          /* sizeAt */ [](auto) { return 0; }, // all empty maps
          /* keyAt */ [](auto, auto) { return 0; },
          /* valueAt */ [](auto, auto) { return 0.0f; },
          /* isNullAt */ [](auto row) { return row % 10 == 0; });
      writer.write(vectorMaker.rowVector({"m"}, {mapVec}));
    }
    writer.close();
  }

  nimble::BatchReader reader(
      std::make_shared<velox::InMemoryReadFile>(file), *leafPool_);
  const auto& flatMap = reader.schema()->asRow().childAt(0)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), kNumKeys);
  {
    std::vector<std::string> sortedKeys(
        predefinedKeys.begin(), predefinedKeys.end());
    for (int i = 0; i < kNumKeys; ++i) {
      EXPECT_EQ(flatMap.nameAt(i), sortedKeys[i]);
    }
  }
}

TEST_F(WriterTest, flatmapColumnsKeysUnknownKeyRejection) {
  // Pre-register 5 keys, write 100 rows. Last row includes an unknown key.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::INTEGER())}});

  const std::set<std::string> predefinedKeys = {"1", "2", "3", "4", "5"};
  constexpr int32_t kNumRows = 100;
  constexpr int32_t kNumKeys = 5;

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::WriterOptions options;
  options.flatMapColumns = {{"m", predefinedKeys}};
  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(options));

  // Include unknown key 99 in every row.
  auto input = vectorMaker.rowVector(
      {"m"},
      {vectorMaker.mapVector<int32_t, int32_t>(
          kNumRows,
          /* sizeAt */ [](auto) { return 2; },
          /* keyAt */
          [](auto row, auto mapIndex) {
            return mapIndex == 0 ? (row % kNumKeys) + 1 : 99;
          },
          /* valueAt */
          [](auto row, auto mapIndex) { return row * 10 + mapIndex; })});

  EXPECT_THROW(writer.write(input), nimble::NimbleUserError);
}

TEST_F(WriterTest, flatmapColumnsKeysRejectsRowIngestion) {
  // When flatMapColumns with predefined keys is configured, passing a ROW
  // vector (instead of MAP) should throw. Test with 100 rows.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto type = velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::REAL())}});

  constexpr int32_t kNumRows = 100;

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::WriterOptions options;
  options.flatMapColumns = {{"m", {"1", "2", "3", "4", "5"}}};
  nimble::Writer writer(
      type, std::move(writeFile), *rootPool_, std::move(options));

  // Build a RowVector child to trigger ingestRow path.
  auto rowChild = vectorMaker.rowVector(
      {"a"}, {vectorMaker.flatVector<float>(kNumRows, [](auto row) {
        return static_cast<float>(row);
      })});
  auto input = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      velox::ROW({{"m", rowChild->type()}}),
      nullptr,
      kNumRows,
      std::vector<velox::VectorPtr>{rowChild});

  EXPECT_THROW(writer.write(input), nimble::NimbleUserError);
}

TEST_F(WriterTest, flatmapColumnsKeysImplicitFlatMapColumn) {
  // Columns listed in flatMapColumns with predefined keys are implicitly
  // treated as flat map columns. Test with 8 keys and 500 rows across 3
  // batches.
  velox::test::VectorMaker vectorMaker{leafPool_.get()};

  auto type =
      velox::ROW({{"m", velox::MAP(velox::INTEGER(), velox::INTEGER())}});

  const std::vector<std::string> predefinedKeysList = {
      "10", "3", "7", "15", "1", "20", "8", "12"};
  const std::set<std::string> predefinedKeys(
      predefinedKeysList.begin(), predefinedKeysList.end());
  constexpr int32_t kNumKeys = 8;
  constexpr int32_t kRowsPerBatch = 500;
  constexpr int32_t kBatches = 3;
  constexpr int32_t kTotalRows = kRowsPerBatch * kBatches;

  auto expected = velox::BaseVector::create(type, 0, leafPool_.get());

  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::WriterOptions options;
    // Only set flatMapColumns with predefined keys.
    options.flatMapColumns = {{"m", predefinedKeys}};
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, std::move(options));

    for (int32_t b = 0; b < kBatches; ++b) {
      auto mapVec = vectorMaker.mapVector<int32_t, int32_t>(
          kRowsPerBatch,
          /* sizeAt */ [](auto row) { return (row % kNumKeys) + 1; },
          /* keyAt */
          [&](auto /*row*/, auto mapIndex) {
            return folly::to<int32_t>(predefinedKeysList[mapIndex % kNumKeys]);
          },
          /* valueAt */
          [&](auto row, auto mapIndex) {
            return static_cast<int32_t>(b * 1000 + row * 10 + mapIndex);
          },
          /* isNullAt */ [](auto row) { return row % 50 == 0; });
      auto batch = vectorMaker.rowVector({"m"}, {mapVec});
      writer.write(batch);
      expected->append(batch.get());
    }
    writer.close();
  }

  // Verify it was written as a flatmap (schema has FlatMap kind).
  nimble::BatchReader reader(
      std::make_shared<velox::InMemoryReadFile>(file), *leafPool_);
  const auto& child = reader.schema()->asRow().childAt(0);
  EXPECT_EQ(child->kind(), nimble::Kind::FlatMap);
  const auto& flatMap = child->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), kNumKeys);
  {
    std::vector<std::string> sortedKeys(
        predefinedKeys.begin(), predefinedKeys.end());
    for (int i = 0; i < kNumKeys; ++i) {
      EXPECT_EQ(flatMap.nameAt(i), sortedKeys[i]);
    }
  }

  // Verify data roundtrip.
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(kTotalRows, result));
  ASSERT_EQ(result->size(), kTotalRows);
  for (auto i = 0; i < kTotalRows; ++i) {
    ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
        << "Mismatch at row " << i;
  }
}

struct ParallelEncodeParam {
  uint32_t maxEncodeParallelism;
  uint32_t minStreamsPerEncodingTask;

  std::string debugString() const {
    return fmt::format(
        "maxParallel_{}_minStreams_{}",
        maxEncodeParallelism,
        minStreamsPerEncodingTask);
  }
};

class ParallelEncodeWriterTest
    : public WriterTest,
      public ::testing::WithParamInterface<ParallelEncodeParam> {
 protected:
  nimble::WriterOptions parallelWriterOptions() {
    nimble::WriterOptions options;
    executor_ = std::make_shared<folly::CPUThreadPoolExecutor>(4);
    options.encodingExecutor = folly::getKeepAliveToken(*executor_);
    options.maxEncodeParallelism = GetParam().maxEncodeParallelism;
    options.minStreamsPerEncodingTask = GetParam().minStreamsPerEncodingTask;
    return options;
  }

  std::shared_ptr<folly::CPUThreadPoolExecutor> executor_;
};

TEST_P(ParallelEncodeWriterTest, rowParallelEncode) {
  auto type = velox::ROW({
      {"a", velox::BIGINT()},
      {"b", velox::DOUBLE()},
      {"c", velox::INTEGER()},
      {"d", velox::BIGINT()},
      {"e", velox::REAL()},
      {"f", velox::BIGINT()},
      {"g", velox::INTEGER()},
      {"h", velox::DOUBLE()},
  });

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  velox::VectorFuzzer fuzzer(
      {.vectorSize = 1000, .nullRatio = 0.1}, leafPool_.get(), seed);

  auto writerOptions = parallelWriterOptions();

  std::string seqFile;
  std::string parFile;

  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&seqFile);
    nimble::Writer writer(type, std::move(writeFile), *rootPool_, {});
    for (int i = 0; i < 10; ++i) {
      writer.write(fuzzer.fuzzInputRow(type));
    }
    writer.close();
  }

  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&parFile);
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, writerOptions);
    fuzzer.reSeed(seed);
    for (int i = 0; i < 10; ++i) {
      writer.write(fuzzer.fuzzInputRow(type));
    }
    writer.close();
  }

  auto seqRead = std::make_shared<velox::InMemoryReadFile>(seqFile);
  auto parRead = std::make_shared<velox::InMemoryReadFile>(parFile);
  nimble::BatchReader seqReader(seqRead.get(), *leafPool_);
  nimble::BatchReader parReader(parRead.get(), *leafPool_);

  velox::VectorPtr seqResult;
  velox::VectorPtr parResult;
  while (seqReader.next(1000, seqResult)) {
    ASSERT_TRUE(parReader.next(1000, parResult));
    ASSERT_EQ(seqResult->size(), parResult->size());
    for (velox::vector_size_t row = 0; row < seqResult->size(); ++row) {
      ASSERT_TRUE(seqResult->equalValueAt(parResult.get(), row, row))
          << "Mismatch at row " << row;
    }
  }
}

TEST_P(ParallelEncodeWriterTest, flatMapParallelEncode) {
  auto type = velox::ROW({
      {"flatmap", velox::MAP(velox::INTEGER(), velox::BIGINT())},
      {"col", velox::BIGINT()},
  });

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  velox::VectorFuzzer fuzzer(
      {.vectorSize = 1000, .nullRatio = 0.1, .containerLength = 20},
      leafPool_.get(),
      seed);

  // The fuzzer emits random INTEGER keys, so the number of distinct flat-map
  // keys varies per seed and can exceed the default limit. This test compares
  // parallel vs sequential encoding, so lift the limit (0 = unlimited) to keep
  // the run independent of the seed's key cardinality.
  auto writerOptions = parallelWriterOptions();
  writerOptions.flatMapColumns = {{"flatmap", {}}};
  writerOptions.maxFlatMapKeys = 0;

  nimble::WriterOptions seqOptions;
  seqOptions.flatMapColumns = {{"flatmap", {}}};
  seqOptions.maxFlatMapKeys = 0;

  std::string seqFile;
  std::string parFile;

  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&seqFile);
    nimble::Writer writer(type, std::move(writeFile), *rootPool_, seqOptions);
    for (int i = 0; i < 10; ++i) {
      writer.write(fuzzer.fuzzInputRow(type));
    }
    writer.close();
  }

  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&parFile);
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, writerOptions);
    fuzzer.reSeed(seed);
    for (int i = 0; i < 10; ++i) {
      writer.write(fuzzer.fuzzInputRow(type));
    }
    writer.close();
  }

  auto seqRead = std::make_shared<velox::InMemoryReadFile>(seqFile);
  auto parRead = std::make_shared<velox::InMemoryReadFile>(parFile);
  nimble::BatchReader seqReader(seqRead.get(), *leafPool_);
  nimble::BatchReader parReader(parRead.get(), *leafPool_);

  velox::VectorPtr seqResult;
  velox::VectorPtr parResult;
  while (seqReader.next(1000, seqResult)) {
    ASSERT_TRUE(parReader.next(1000, parResult));
    ASSERT_EQ(seqResult->size(), parResult->size());
    for (velox::vector_size_t row = 0; row < seqResult->size(); ++row) {
      ASSERT_TRUE(seqResult->equalValueAt(parResult.get(), row, row))
          << "Mismatch at row " << row;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ParallelEncodeWriterTestSuite,
    ParallelEncodeWriterTest,
    ::testing::Values(
        ParallelEncodeParam{2, 1},
        ParallelEncodeParam{4, 1},
        ParallelEncodeParam{8, 1},
        ParallelEncodeParam{4, 4},
        ParallelEncodeParam{100, 1}),
    [](const ::testing::TestParamInfo<ParallelEncodeParam>& info) {
      return info.param.debugString();
    });

DEBUG_ONLY_TEST_F(WriterTest, bufferingRemainsSequential) {
  velox::common::testutil::TestValue::enable();

  auto type = velox::ROW({
      {"a", velox::BIGINT()},
      {"b", velox::DOUBLE()},
      {"c", velox::INTEGER()},
      {"d", velox::BIGINT()},
      {"e", velox::REAL()},
      {"f", velox::BIGINT()},
      {"g", velox::INTEGER()},
      {"h", velox::DOUBLE()},
  });

  velox::VectorFuzzer fuzzer(
      {.vectorSize = 100, .nullRatio = 0}, leafPool_.get());

  struct TestCase {
    uint32_t maxEncodeParallelism;
    uint32_t minStreamsPerEncodingTask;
  };

  const std::vector<TestCase> testCases = {
      {4, 0},
      {2, 1},
      {4, 1},
      {8, 4},
  };

  folly::CPUThreadPoolExecutor executor(4);

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "maxParallel={}, minStreams={}",
            testCase.maxEncodeParallelism,
            testCase.minStreamsPerEncodingTask));

    nimble::WriterOptions writerOptions;
    writerOptions.encodingExecutor = folly::getKeepAliveToken(executor);
    writerOptions.maxEncodeParallelism = testCase.maxEncodeParallelism;
    writerOptions.minStreamsPerEncodingTask =
        testCase.minStreamsPerEncodingTask;

    uint32_t parallelWriteCount{0};
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::RowFieldWriter::co_write",
        std::function<void(const uint32_t*)>(
            [&](const uint32_t*) { ++parallelWriteCount; }));

    std::string file;
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    nimble::Writer writer(
        type, std::move(writeFile), *rootPool_, writerOptions);

    const size_t numBatches = 3;
    for (size_t i = 0; i < numBatches; ++i) {
      writer.write(fuzzer.fuzzInputRow(type));
    }
    writer.close();

    EXPECT_EQ(parallelWriteCount, 0);

    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    nimble::BatchReader reader(readFile.get(), *leafPool_);
    velox::VectorPtr result;
    ASSERT_TRUE(reader.next(numBatches * 100, result));
    EXPECT_EQ(result->size(), numBatches * 100);
  }
}

DEBUG_ONLY_TEST_F(WriterTest, parallelEncodingTaskCount) {
  velox::common::testutil::TestValue::enable();

  auto type = velox::ROW({
      {"a", velox::BIGINT()},
      {"b", velox::BIGINT()},
      {"c", velox::BIGINT()},
      {"d", velox::BIGINT()},
      {"e", velox::BIGINT()},
      {"f", velox::BIGINT()},
      {"g", velox::BIGINT()},
      {"h", velox::BIGINT()},
  });
  velox::VectorFuzzer fuzzer(
      {.vectorSize = 100, .nullRatio = 0}, leafPool_.get());
  folly::CPUThreadPoolExecutor executor{8};

  struct TestCase {
    uint32_t maxEncodeParallelism;
    uint32_t minStreamsPerEncodingTask;
    uint32_t expectedTaskCount;
  };
  const std::vector<TestCase> testCases = {
      {4, 0, 4},
      {4, 1, 4},
      {4, 3, 3},
      {2, 3, 2},
      {8, 32, 0},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "maxParallel={}, minStreams={}, expected={}",
            testCase.maxEncodeParallelism,
            testCase.minStreamsPerEncodingTask,
            testCase.expectedTaskCount));

    std::atomic<uint32_t> taskCount{0};
    SCOPED_TESTVALUE_SET(
        "facebook::nimble::Writer::parallelEncodeTask",
        std::function<void(const uint32_t*)>([&](const uint32_t*) {
          taskCount.fetch_add(1, std::memory_order_relaxed);
        }));

    nimble::WriterOptions writerOptions;
    writerOptions.enableChunking = false;
    writerOptions.encodingExecutor = folly::getKeepAliveToken(executor);
    writerOptions.maxEncodeParallelism = testCase.maxEncodeParallelism;
    writerOptions.minStreamsPerEncodingTask =
        testCase.minStreamsPerEncodingTask;

    std::string file;
    nimble::Writer writer(
        type,
        std::make_unique<velox::InMemoryWriteFile>(&file),
        *rootPool_,
        writerOptions);
    writer.write(fuzzer.fuzzInputRow(type));
    writer.close();

    EXPECT_EQ(
        taskCount.load(std::memory_order_relaxed), testCase.expectedTaskCount);
  }
}

DEBUG_ONLY_TEST_F(WriterTest, flatMapBufferingRemainsSequential) {
  velox::common::testutil::TestValue::enable();

  auto type = velox::ROW({
      {"flatmap", velox::MAP(velox::INTEGER(), velox::BIGINT())},
      {"col", velox::BIGINT()},
  });

  velox::VectorFuzzer fuzzer(
      {.vectorSize = 100, .nullRatio = 0, .containerLength = 20},
      leafPool_.get());

  folly::CPUThreadPoolExecutor executor(4);

  nimble::WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"flatmap", {}}};
  writerOptions.encodingExecutor = folly::getKeepAliveToken(executor);
  writerOptions.maxEncodeParallelism = 4;
  writerOptions.minStreamsPerEncodingTask = 1;

  uint32_t flatMapParallelCount{0};
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::FlatMapFieldWriter::co_writeMapValues",
      std::function<void(const uint32_t*)>(
          [&](const uint32_t*) { ++flatMapParallelCount; }));

  std::string file;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
  nimble::Writer writer(type, std::move(writeFile), *rootPool_, writerOptions);

  const size_t numBatches = 5;
  for (size_t i = 0; i < numBatches; ++i) {
    writer.write(fuzzer.fuzzInputRow(type));
  }
  writer.close();

  EXPECT_EQ(flatMapParallelCount, 0);

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  nimble::BatchReader reader(readFile.get(), *leafPool_);
  velox::VectorPtr result;
  ASSERT_TRUE(reader.next(numBatches * 100, result));
  EXPECT_EQ(result->size(), numBatches * 100);
}

// Fuzzer data written with the random encoding selection policy round-trips for
// every seed: the policy only ever picks encodings compatible with the data
// (via EncodingSizeEstimation), so no stream needs the writer's one-shot
// IncompatibleEncoding fallback. Regression seeds are pinned first; drive
// breadth with --stress-runs.
TEST_F(WriterTest, randomEncodingSelectionRoundTrip) {
  const auto rowType = randomEncodingSelectionTestType();
  std::vector<uint32_t> seeds = {1u, 7u, 42u};
  seeds.push_back(
      FLAGS_writer_tests_seed > 0 ? FLAGS_writer_tests_seed
                                  : folly::Random::rand32());

  for (const uint32_t seed : seeds) {
    LOG(INFO) << "randomEncodingSelectionRoundTrip seed: " << seed;
    velox::VectorFuzzer::Options fuzzerOptions;
    fuzzerOptions.vectorSize = 500;
    fuzzerOptions.nullRatio = 0.2;
    fuzzerOptions.stringLength = 16;
    fuzzerOptions.containerLength = 6;
    fuzzerOptions.containerVariableLength = true;
    velox::VectorFuzzer fuzzer(fuzzerOptions, leafPool_.get(), seed);
    auto vector = fuzzer.fuzzInputFlatRow(rowType);

    const auto file = writeWithEncodingSelectionCreator(
        *rootPool_, vector, makeRandomEncodingSelectionPolicyCreator(seed));

    auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
    nimble::BatchReader reader(readFile.get(), *leafPool_);
    velox::VectorPtr result;
    ASSERT_TRUE(reader.next(vector->size(), result));
    ASSERT_EQ(result->size(), vector->size());
    for (velox::vector_size_t i = 0; i < vector->size(); ++i) {
      ASSERT_TRUE(vector->equalValueAt(result.get(), i, i))
          << "mismatch at row " << i << " (seed " << seed << ")";
    }
    ASSERT_FALSE(reader.next(1, result));
  }
}

TEST_F(WriterTest, randomEncodingSelectionVariesByChunkContents) {
  constexpr int kNumChunks = 16;
  constexpr int kNumRowsPerChunk = 1'000;
  constexpr uint64_t kSeed = 42;
  const auto rowType =
      velox::ROW({{"c0", velox::BIGINT()}, {"c1", velox::BIGINT()}});
  velox::test::VectorMaker vectorMaker{leafPool_.get()};
  std::vector<velox::RowVectorPtr> batches;
  batches.reserve(kNumChunks);
  for (int chunkIndex = 0; chunkIndex < kNumChunks; ++chunkIndex) {
    std::vector<int64_t> values(kNumRowsPerChunk);
    std::vector<int64_t> otherValues(kNumRowsPerChunk);
    for (int row = 0; row < kNumRowsPerChunk; ++row) {
      values[row] = (row + chunkIndex) % (17 + chunkIndex);
      otherValues[row] = (row * 3 + chunkIndex) % (23 + chunkIndex);
    }
    batches.push_back(vectorMaker.rowVector(
        {"c0", "c1"},
        {vectorMaker.flatVector<int64_t>(values),
         vectorMaker.flatVector<int64_t>(otherValues)}));
  }

  auto makeOptions = [] {
    nimble::WriterOptions options;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = [] {
      return std::make_unique<nimble::LambdaFlushPolicy>(
          /*flushLambda=*/[](auto&) { return false; },
          /*chunkLambda=*/[](auto&) { return true; });
    };
    options.encodingSelectionPolicyCreator =
        makeRandomEncodingSelectionPolicyCreator(
            kSeed,
            {nimble::EncodingType::Trivial, nimble::EncodingType::Dictionary});
    return options;
  };

  const auto options = makeOptions();
  const auto first = writeAndCaptureChunkLayouts(
      rowType, batches, options, /*expectedStripeCount=*/1);
  const auto second = writeAndCaptureChunkLayouts(
      rowType, batches, options, /*expectedStripeCount=*/1);
  ASSERT_THAT(first, ::testing::SizeIs(kNumChunks));
  ASSERT_THAT(second, ::testing::SizeIs(first.size()));

  folly::CPUThreadPoolExecutor executor{2};
  auto parallelOptions = makeOptions();
  parallelOptions.encodingExecutor = folly::getKeepAliveToken(executor);
  parallelOptions.maxEncodeParallelism = 2;
  parallelOptions.minStreamsPerEncodingTask = 1;
  const auto parallel = writeAndCaptureChunkLayouts(
      rowType, batches, std::move(parallelOptions), /*expectedStripeCount=*/1);
  ASSERT_THAT(parallel, ::testing::SizeIs(first.size()));

  std::vector<nimble::EncodingType> firstTypes;
  std::vector<nimble::EncodingType> secondTypes;
  std::vector<nimble::EncodingType> parallelTypes;
  for (size_t chunk = 0; chunk < first.size(); ++chunk) {
    firstTypes.push_back(first[chunk].encodingType());
    secondTypes.push_back(second[chunk].encodingType());
    parallelTypes.push_back(parallel[chunk].encodingType());
  }

  EXPECT_THAT(secondTypes, ::testing::ElementsAreArray(firstTypes));
  EXPECT_THAT(parallelTypes, ::testing::ElementsAreArray(firstTypes));
  EXPECT_THAT(firstTypes, ::testing::Contains(nimble::EncodingType::Trivial));
  EXPECT_THAT(
      firstTypes, ::testing::Contains(nimble::EncodingType::Dictionary));
}

// The random layout is reproducible from its seed: identical input + seed
// yields byte-identical files independently of encode thread order. A different
// seed selects a different layout.
TEST_F(WriterTest, randomEncodingSelectionDeterministic) {
  const uint32_t seed = FLAGS_writer_tests_seed > 0 ? FLAGS_writer_tests_seed
                                                    : folly::Random::rand32();
  LOG(INFO) << "randomEncodingSelectionDeterministic seed: " << seed;

  const auto rowType = randomEncodingSelectionTestType();
  velox::VectorFuzzer::Options fuzzerOptions;
  fuzzerOptions.vectorSize = 500;
  fuzzerOptions.nullRatio = 0.2;
  fuzzerOptions.stringLength = 16;
  fuzzerOptions.containerLength = 6;
  fuzzerOptions.containerVariableLength = true;
  velox::VectorFuzzer fuzzer(fuzzerOptions, leafPool_.get(), seed);
  auto vector = fuzzer.fuzzInputFlatRow(rowType);

  const auto file = writeWithEncodingSelectionCreator(
      *rootPool_, vector, makeRandomEncodingSelectionPolicyCreator(seed));
  // Same seed reproduces the exact file.
  EXPECT_EQ(
      file,
      writeWithEncodingSelectionCreator(
          *rootPool_, vector, makeRandomEncodingSelectionPolicyCreator(seed)));

  folly::CPUThreadPoolExecutor executor{4};
  nimble::WriterOptions parallelOptions;
  parallelOptions.encodingSelectionPolicyCreator =
      makeRandomEncodingSelectionPolicyCreator(seed);
  parallelOptions.encodingExecutor = folly::getKeepAliveToken(executor);
  parallelOptions.maxEncodeParallelism = 4;
  parallelOptions.minStreamsPerEncodingTask = 1;
  EXPECT_EQ(
      file,
      writeWithWriterOptions(*rootPool_, vector, std::move(parallelOptions)));
  // A different seed should be able to select a different layout. For a given
  // fuzzed input some nodes may have a singleton compatible-encoding set, so an
  // individual alternate seed can legitimately reproduce the same file; only
  // require that at least one of several distinct alternate seeds diverges,
  // which proves the seed drives the layout without flaking on such inputs. XOR
  // a nonzero delta so every alternate seed stays distinct with no wraparound.
  bool anyDifferent = false;
  for (uint32_t delta = 1; delta <= 8 && !anyDifferent; ++delta) {
    anyDifferent =
        writeWithEncodingSelectionCreator(
            *rootPool_,
            vector,
            makeRandomEncodingSelectionPolicyCreator(seed ^ delta)) != file;
  }
  EXPECT_TRUE(anyDifferent)
      << "no alternate seed produced a different layout for seed " << seed;
}
} // namespace facebook
