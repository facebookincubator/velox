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

#include <faiss/IndexFlat.h>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/file/File.h"
#include "velox/common/io/Options.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/VectorIndex.h"
#include "velox/dwio/nimble/index/VectorIndexWriter.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/VectorIndexGenerated.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble::index::test {
namespace {

// Recall metrics for comparing ANN results against brute-force ground truth.
struct RecallMetrics {
  float recallAt1{0};
  float recallAt5{0};
  float recallAt10{0};

  std::string debugString() const {
    return fmt::format(
        "recall@1={:.1f}%, recall@5={:.1f}%, recall@10={:.1f}%",
        recallAt1 * 100,
        recallAt5 * 100,
        recallAt10 * 100);
  }
};

// Captures the two physical layers emitted by VectorIndexWriter.
struct WrittenIndexes {
  // Stores all FAISS data sections at their recorded offsets.
  std::string indexData;

  // Stores the optional VectorIndexDirectory section.
  std::string directoryData;
};

class VectorIndexTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

 protected:
  static constexpr uint32_t kDimensions = 32;
  static constexpr int kSeed = 42;

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("VectorIndexTest");
    leafPool_ = pool_->addLeafChild("leaf");
  }

  velox::memory::MemoryPool* pool() {
    return leafPool_.get();
  }

  velox::RowTypePtr createType() {
    return velox::ROW(
        {{"id", velox::INTEGER()}, {"embedding", velox::ARRAY(velox::REAL())}});
  }

  // Generates uniform random vectors in [0, 1].
  std::vector<float> generateRandomVectors(
      size_t numVectors,
      uint32_t dimensions,
      int seed = kSeed) {
    std::mt19937 randomGenerator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> vectors(numVectors * dimensions);
    for (auto& value : vectors) {
      value = distribution(randomGenerator);
    }
    return vectors;
  }

  // Creates a RowVector with id + embedding columns from a flat float array.
  velox::RowVectorPtr makeInputFromVectors(
      const std::vector<float>& vectors,
      uint32_t dimensions,
      int32_t startId = 0) {
    const auto numRows =
        static_cast<velox::vector_size_t>(vectors.size() / dimensions);

    auto idsVector =
        velox::BaseVector::create(velox::INTEGER(), numRows, pool());
    auto* rawIds = idsVector->asFlatVector<int32_t>()->mutableRawValues();
    for (int32_t i = 0; i < numRows; ++i) {
      rawIds[i] = startId + i;
    }

    const auto totalElements =
        numRows * static_cast<velox::vector_size_t>(dimensions);
    auto elementsVector =
        velox::BaseVector::create(velox::REAL(), totalElements, pool());
    auto* rawElements =
        elementsVector->asFlatVector<float>()->mutableRawValues();
    std::memcpy(rawElements, vectors.data(), vectors.size() * sizeof(float));

    auto arrayVector = std::make_shared<velox::ArrayVector>(
        pool(),
        velox::ARRAY(velox::REAL()),
        nullptr,
        numRows,
        velox::allocateOffsets(numRows, pool()),
        velox::allocateSizes(numRows, pool()),
        elementsVector);

    auto* rawOffsets =
        arrayVector->mutableOffsets(numRows)->asMutable<velox::vector_size_t>();
    auto* rawSizes =
        arrayVector->mutableSizes(numRows)->asMutable<velox::vector_size_t>();
    for (int32_t i = 0; i < numRows; ++i) {
      rawOffsets[i] = i * static_cast<velox::vector_size_t>(dimensions);
      rawSizes[i] = static_cast<velox::vector_size_t>(dimensions);
    }

    return std::make_shared<velox::RowVector>(
        pool(),
        createType(),
        nullptr,
        numRows,
        std::vector<velox::VectorPtr>{idsVector, arrayVector});
  }

  // Creates a single-index writer for focused unit tests.
  std::unique_ptr<VectorIndexWriter> createWriter(
      const VectorIndexConfig& config,
      const velox::RowTypePtr& type) {
    const std::array configs{config};
    return VectorIndexWriter::create(configs, type, pool());
  }

  // Closes a writer and captures its directory and data sections.
  WrittenIndexes closeWriter(VectorIndexWriter& writer) {
    WrittenIndexes written;
    const CreateMetadataSectionFn createMetadataFn =
        [&written](std::string_view content) {
          const auto offset = written.indexData.size();
          written.indexData.append(content);
          return MetadataSection{
              offset,
              static_cast<uint32_t>(content.size()),
              CompressionType::Uncompressed,
              static_cast<uint32_t>(content.size()),
          };
        };
    const WriteOptionalSectionFn writeMetadataFn =
        [&written](const std::string& name, std::string_view content) {
          EXPECT_EQ(name, kVectorIndexSection);
          written.directoryData = std::string(content);
        };
    writer.close(createMetadataFn, writeMetadataFn);
    return written;
  }

  // Writes vectors and captures the directory and referenced data sections.
  WrittenIndexes writeIndex(
      const VectorIndexConfig& config,
      const std::vector<velox::RowVectorPtr>& batches) {
    auto type = createType();
    auto writer = createWriter(config, type);
    EXPECT_NE(writer, nullptr);

    for (const auto& batch : batches) {
      writer->write(batch);
    }

    return closeWriter(*writer);
  }

  // Parses the directory from the captured optional section.
  VectorIndexDirectory readDirectory(const WrittenIndexes& written) {
    auto directoryBuffer = MetadataBuffer::decompress(
        written.directoryData, CompressionType::Uncompressed, pool());
    velox::io::ReaderOptions ioOptions(pool());
    ioOptions.setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
    IndexLookup::Options indexOptions{
        .file = std::make_shared<velox::InMemoryReadFile>(written.indexData),
        .ioOptions = &ioOptions,
    };
    return VectorIndexDirectory::create(
        Section{MetadataBuffer{std::move(directoryBuffer)}}, indexOptions);
  }

  // Materializes the single index expected by most focused tests.
  std::shared_ptr<const VectorIndex> readIndex(const WrittenIndexes& written) {
    const auto directory = readDirectory(written);
    NIMBLE_CHECK_EQ(directory.numIndexes(), 1);
    return directory.load("embedding");
  }

  // Returns one descriptor from the captured directory FlatBuffer.
  const serialization::VectorIndex* serializedIndex(
      const WrittenIndexes& written,
      uint32_t index = 0) {
    const auto* directory =
        flatbuffers::GetRoot<serialization::VectorIndexDirectory>(
            written.directoryData.data());
    NIMBLE_CHECK_NOT_NULL(directory);
    NIMBLE_CHECK_NOT_NULL(directory->indices());
    NIMBLE_CHECK_LT(index, directory->indices()->size());
    const auto* descriptor = directory->indices()->Get(index);
    NIMBLE_CHECK_NOT_NULL(descriptor);
    return descriptor;
  }

  // Computes recall by comparing ANN results against brute-force ground truth.
  // Uses FAISS IndexFlat as the exact search baseline.
  RecallMetrics computeRecall(
      const WrittenIndexes& written,
      const std::vector<float>& dataVectors,
      const std::vector<float>& queryVectors,
      uint32_t dimensions,
      uint32_t numNeighbors,
      uint32_t numProbes,
      VectorDistanceMetric metric) {
    const auto numData = dataVectors.size() / dimensions;
    const auto numQueries = queryVectors.size() / dimensions;

    // Build brute-force ground truth.
    const auto faissMetric = (metric == VectorDistanceMetric::kL2)
        ? faiss::METRIC_L2
        : faiss::METRIC_INNER_PRODUCT;
    faiss::IndexFlat groundTruth(dimensions, faissMetric);

    // For cosine, normalize data vectors before adding to ground truth.
    std::vector<float> normalizedData;
    if (metric == VectorDistanceMetric::kCosine) {
      normalizedData = dataVectors;
      for (size_t i = 0; i < numData; ++i) {
        float norm = 0;
        for (uint32_t j = 0; j < dimensions; ++j) {
          norm += normalizedData[i * dimensions + j] *
              normalizedData[i * dimensions + j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
          for (uint32_t j = 0; j < dimensions; ++j) {
            normalizedData[i * dimensions + j] /= norm;
          }
        }
      }
      groundTruth.add(
          static_cast<faiss::idx_t>(numData), normalizedData.data());
    } else {
      groundTruth.add(static_cast<faiss::idx_t>(numData), dataVectors.data());
    }

    // Search ground truth.
    std::vector<float> expectedDistances(numQueries * numNeighbors);
    std::vector<faiss::idx_t> expectedLabels(numQueries * numNeighbors);

    // For cosine, normalize query vectors too.
    std::vector<float> normalizedQueries;
    const float* queryData = queryVectors.data();
    if (metric == VectorDistanceMetric::kCosine) {
      normalizedQueries = queryVectors;
      for (size_t i = 0; i < numQueries; ++i) {
        float norm = 0;
        for (uint32_t j = 0; j < dimensions; ++j) {
          norm += normalizedQueries[i * dimensions + j] *
              normalizedQueries[i * dimensions + j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
          for (uint32_t j = 0; j < dimensions; ++j) {
            normalizedQueries[i * dimensions + j] /= norm;
          }
        }
      }
      queryData = normalizedQueries.data();
    }

    groundTruth.search(
        static_cast<faiss::idx_t>(numQueries),
        queryData,
        static_cast<faiss::idx_t>(numNeighbors),
        expectedDistances.data(),
        expectedLabels.data());

    // Search via VectorIndex.
    auto reader = readIndex(written);

    uint32_t recallAt1Matches{0};
    uint32_t recallAt5Matches{0};
    uint32_t recallAt10Matches{0};

    for (size_t queryIndex = 0; queryIndex < numQueries; ++queryIndex) {
      const auto queryOffset = static_cast<std::ptrdiff_t>(
          queryIndex * static_cast<size_t>(dimensions));
      const auto nextQueryOffset = static_cast<std::ptrdiff_t>(
          (queryIndex + 1) * static_cast<size_t>(dimensions));
      std::vector<float> queryVector(
          queryVectors.begin() + queryOffset,
          queryVectors.begin() + nextQueryOffset);

      VectorIndex::SearchConfig searchConfig{
          .queryVector = queryVector,
          .numNeighbors = numNeighbors,
          .numProbes = numProbes,
      };
      const auto results = reader->search(searchConfig);

      const auto countMatches = [&](uint32_t requestedNeighbors) {
        const auto resultWidth = std::min(requestedNeighbors, numNeighbors);
        const auto numComparedResults =
            std::min(resultWidth, static_cast<uint32_t>(results.size()));
        uint32_t numMatches{0};
        for (uint32_t resultIndex = 0; resultIndex < numComparedResults;
             ++resultIndex) {
          for (uint32_t expectedIndex = 0; expectedIndex < resultWidth;
               ++expectedIndex) {
            if (results[resultIndex].rowId ==
                expectedLabels[queryIndex * numNeighbors + expectedIndex]) {
              ++numMatches;
              break;
            }
          }
        }
        return numMatches;
      };

      recallAt1Matches += countMatches(1);
      recallAt5Matches += countMatches(5);
      recallAt10Matches += countMatches(10);
    }

    RecallMetrics metrics;
    metrics.recallAt1 =
        static_cast<float>(recallAt1Matches) / static_cast<float>(numQueries);
    metrics.recallAt5 = static_cast<float>(recallAt5Matches) /
        static_cast<float>(numQueries * std::min(5U, numNeighbors));
    metrics.recallAt10 = static_cast<float>(recallAt10Matches) /
        static_cast<float>(numQueries * std::min(10U, numNeighbors));
    return metrics;
  }

  VectorIndexConfig makeConfig(
      VectorIndexType indexType,
      VectorDistanceMetric metric = VectorDistanceMetric::kL2,
      uint32_t dimensions = kDimensions,
      uint32_t numPartitions = 8) {
    return VectorIndexConfig{
        .columnName = "embedding",
        .dimensions = dimensions,
        .metric = metric,
        .indexType = indexType,
        .numPartitions = numPartitions,
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(VectorIndexTest, recallNumProbesEffect) {
  const uint32_t numVectors = 1'000;
  const uint32_t numQueries = 20;
  const uint32_t numNeighbors = 10;

  auto data = generateRandomVectors(numVectors, kDimensions);
  auto queries = generateRandomVectors(numQueries, kDimensions, /*seed=*/99);

  auto config = makeConfig(VectorIndexType::kIvfFlat);
  auto batch = makeInputFromVectors(data, kDimensions);
  auto written = writeIndex(config, {batch});
  ASSERT_FALSE(written.directoryData.empty());

  // Low numProbes: lower recall.
  auto metricsLow = computeRecall(
      written,
      data,
      queries,
      kDimensions,
      numNeighbors,
      1,
      VectorDistanceMetric::kL2);
  // High numProbes (all partitions): should approach 100% recall.
  auto metricsHigh = computeRecall(
      written,
      data,
      queries,
      kDimensions,
      numNeighbors,
      8,
      VectorDistanceMetric::kL2);

  LOG(INFO) << "numProbes=1: " << metricsLow.debugString();
  LOG(INFO) << "numProbes=8: " << metricsHigh.debugString();

  // Higher numProbes should yield equal or better recall.
  EXPECT_GE(metricsHigh.recallAt1, metricsLow.recallAt1);
  EXPECT_GE(metricsHigh.recallAt10, metricsLow.recallAt10);
}

TEST_F(VectorIndexTest, concurrentSearchUsesPerRequestNumProbes) {
  const uint32_t numVectors = 1'000;
  const uint32_t numNeighbors = 50;
  const auto data = generateRandomVectors(numVectors, kDimensions);
  const auto batch = makeInputFromVectors(data, kDimensions);
  const auto written =
      writeIndex(makeConfig(VectorIndexType::kIvfFlat), {batch});
  const auto query = generateRandomVectors(1, kDimensions, /*seed=*/99);

  const auto searchRows = [&query](
                              const VectorIndex& index, uint32_t numProbes) {
    const auto results = index.search(
        {.queryVector = query,
         .numNeighbors = numNeighbors,
         .numProbes = numProbes});
    std::vector<int64_t> rows;
    rows.reserve(results.size());
    for (const auto& result : results) {
      rows.emplace_back(result.rowId);
    }
    return rows;
  };

  const auto expectedReader = readIndex(written);
  const auto expectedNarrow = searchRows(*expectedReader, 1);
  const auto expectedWide = searchRows(*expectedReader, 8);
  ASSERT_NE(expectedNarrow, expectedWide);

  constexpr uint32_t kNumConcurrentSearches = 16;
  std::barrier start{kNumConcurrentSearches};
  const auto concurrentReader = readIndex(written);
  std::vector<std::future<std::vector<int64_t>>> searches;
  searches.reserve(kNumConcurrentSearches);
  for (uint32_t i = 0; i < kNumConcurrentSearches; ++i) {
    const uint32_t numProbes = i % 2 == 0 ? 1 : 8;
    searches.emplace_back(std::async(std::launch::async, [&, numProbes] {
      start.arrive_and_wait();
      return searchRows(*concurrentReader, numProbes);
    }));
  }

  for (uint32_t i = 0; i < searches.size(); ++i) {
    SCOPED_TRACE(fmt::format("search={}", i));
    EXPECT_EQ(searches[i].get(), i % 2 == 0 ? expectedNarrow : expectedWide);
  }
}

// ---------------------------------------------------------------------------
// Exact match tests: deterministic vectors where the answer is known.
// ---------------------------------------------------------------------------

TEST_F(VectorIndexTest, exactMatchIvfFlat) {
  const uint32_t dimensions = 8;
  const int32_t numRows = 200;

  // Deterministic vectors use i * 0.1 + j * 0.01 for element j in row i.
  std::vector<float> vectors(numRows * dimensions);
  for (int32_t i = 0; i < numRows; ++i) {
    for (uint32_t j = 0; j < dimensions; ++j) {
      vectors[i * dimensions + j] =
          static_cast<float>(i) * 0.1f + static_cast<float>(j) * 0.01f;
    }
  }

  auto config = makeConfig(
      VectorIndexType::kIvfFlat, VectorDistanceMetric::kL2, dimensions, 4);
  auto batch = makeInputFromVectors(vectors, dimensions);
  auto written = writeIndex(config, {batch});
  ASSERT_FALSE(written.directoryData.empty());

  auto reader = readIndex(written);
  ASSERT_NE(reader, nullptr);

  // Query with exact vector for row 42.
  std::vector<float> queryVector(dimensions);
  for (uint32_t j = 0; j < dimensions; ++j) {
    queryVector[j] = 42 * 0.1f + static_cast<float>(j) * 0.01f;
  }

  VectorIndex::SearchConfig searchConfig{
      .queryVector = queryVector,
      .numNeighbors = 5,
      .numProbes = 4,
  };
  auto results = reader->search(searchConfig);

  ASSERT_FALSE(results.empty());
  EXPECT_EQ(results[0].rowId, 42);
  EXPECT_NEAR(results[0].score, 0.0f, 1e-4f);

  // Neighbors should be row 41 and 43.
  ASSERT_GE(results.size(), 3);
  std::set<int64_t> neighborIds;
  for (size_t i = 1; i < results.size(); ++i) {
    neighborIds.insert(results[i].rowId);
  }
  EXPECT_TRUE(neighborIds.count(41) > 0 || neighborIds.count(43) > 0);
}

TEST_F(VectorIndexTest, l2ScoreOrdering) {
  const uint32_t dimensions = 8;
  const int32_t numRows = 200;

  std::vector<float> vectors(numRows * dimensions);
  for (int32_t i = 0; i < numRows; ++i) {
    for (uint32_t j = 0; j < dimensions; ++j) {
      vectors[i * dimensions + j] =
          static_cast<float>(i) * 0.1f + static_cast<float>(j) * 0.01f;
    }
  }

  auto config = makeConfig(
      VectorIndexType::kIvfFlat, VectorDistanceMetric::kL2, dimensions, 4);
  auto batch = makeInputFromVectors(vectors, dimensions);
  auto written = writeIndex(config, {batch});

  auto reader = readIndex(written);

  std::vector<float> queryVector(dimensions, 0.5f);
  VectorIndex::SearchConfig searchConfig{
      .queryVector = queryVector,
      .numNeighbors = 10,
      .numProbes = 4,
  };
  auto results = reader->search(searchConfig);

  // Squared L2 scores are monotonically non-decreasing.
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_GE(results[i].score, results[i - 1].score)
        << "Score not sorted at position " << i;
  }
}

// ---------------------------------------------------------------------------
// Serialization round-trip tests.
// ---------------------------------------------------------------------------

TEST_F(VectorIndexTest, serializationRoundTrip) {
  const uint32_t numVectors = 500;
  auto data = generateRandomVectors(numVectors, kDimensions);

  auto config = makeConfig(VectorIndexType::kIvfSq8);
  auto batch = makeInputFromVectors(data, kDimensions);
  auto written = writeIndex(config, {batch});
  ASSERT_FALSE(written.directoryData.empty());

  // Deserialize and verify metadata.
  auto reader = readIndex(written);
  ASSERT_NE(reader, nullptr);
  EXPECT_EQ(reader->columnName(), "embedding");
  EXPECT_EQ(reader->dimensions(), kDimensions);
  EXPECT_EQ(reader->metric(), VectorDistanceMetric::kL2);
  EXPECT_EQ(reader->indexType(), VectorIndexType::kIvfSq8);
  EXPECT_EQ(reader->numVectors(), numVectors);

  // Search should return valid results.
  auto queries = generateRandomVectors(5, kDimensions, /*seed=*/77);
  for (size_t queryIndex = 0; queryIndex < 5; ++queryIndex) {
    const auto queryOffset = static_cast<std::ptrdiff_t>(
        queryIndex * static_cast<size_t>(kDimensions));
    const auto nextQueryOffset = static_cast<std::ptrdiff_t>(
        (queryIndex + 1) * static_cast<size_t>(kDimensions));
    std::vector<float> queryVector(
        queries.begin() + queryOffset, queries.begin() + nextQueryOffset);
    VectorIndex::SearchConfig searchConfig{
        .queryVector = queryVector,
        .numNeighbors = 5,
        .numProbes = 8,
    };
    auto results = reader->search(searchConfig);
    ASSERT_FALSE(results.empty());
    for (const auto& result : results) {
      EXPECT_GE(result.rowId, 0);
      EXPECT_LT(result.rowId, static_cast<int64_t>(numVectors));
      EXPECT_GE(result.score, 0.0f);
    }
  }
}

TEST_F(VectorIndexTest, truncatedFaissIndexRejected) {
  constexpr uint32_t kNumVectors{100};
  auto written = writeIndex(
      makeConfig(VectorIndexType::kIvfFlat),
      {makeInputFromVectors(
          generateRandomVectors(kNumVectors, kDimensions), kDimensions)});
  ASSERT_GT(written.indexData.size(), 1);
  written.indexData.resize(written.indexData.size() / 2);

  EXPECT_THROW(
      VectorIndex::create(
          VectorIndex::Metadata{
              .columnName = "embedding",
              .dimensions = kDimensions,
              .metric = VectorDistanceMetric::kL2,
              .indexType = VectorIndexType::kIvfFlat,
              .numVectors = kNumVectors,
          },
          written.indexData),
      NimbleUserError);
}

TEST_F(VectorIndexTest, mismatchedFaissMetadataRejected) {
  constexpr uint32_t kNumVectors{200};
  const auto written = writeIndex(
      makeConfig(VectorIndexType::kIvfFlat),
      {makeInputFromVectors(
          generateRandomVectors(kNumVectors, kDimensions), kDimensions)});
  const auto* original = serializedIndex(written);
  ASSERT_NE(original->config(), nullptr);
  ASSERT_NE(original->index_data(), nullptr);

  struct TestCase {
    serialization::VectorDistanceMetric metric;
    serialization::VectorIndexType indexType;
  };
  const std::array testCases{
      TestCase{
          serialization::VectorDistanceMetric_Cosine,
          serialization::VectorIndexType_IVF_FLAT,
      },
      TestCase{
          serialization::VectorDistanceMetric_L2,
          serialization::VectorIndexType_IVF_SQ8,
      },
  };
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "metric={}, indexType={}",
            static_cast<int>(testCase.metric),
            static_cast<int>(testCase.indexType)));
    flatbuffers::FlatBufferBuilder builder;
    const auto config = serialization::CreateVectorIndexMeta(
        builder,
        builder.CreateString(original->config()->column_name()->string_view()),
        original->config()->dimensions(),
        testCase.metric,
        testCase.indexType,
        original->config()->num_partitions(),
        original->config()->num_vectors());
    const auto originalSection = original->index_data();
    const auto section = serialization::CreateMetadataSection(
        builder,
        originalSection->offset(),
        originalSection->size(),
        originalSection->compression_type(),
        originalSection->uncompressed_size());
    const auto descriptor =
        serialization::CreateVectorIndex(builder, config, section);
    builder.Finish(
        serialization::CreateVectorIndexDirectory(
            builder, builder.CreateVector(&descriptor, 1)));

    auto mismatched = written;
    mismatched.directoryData.assign(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize());
    NIMBLE_ASSERT_THROW(readIndex(mismatched), "FAISS index");
  }
}

TEST_F(VectorIndexTest, zeroVectorMetadataRejected) {
  constexpr uint32_t kNumVectors{100};
  const auto written = writeIndex(
      makeConfig(VectorIndexType::kIvfFlat),
      {makeInputFromVectors(
          generateRandomVectors(kNumVectors, kDimensions), kDimensions)});

  NIMBLE_ASSERT_THROW(
      VectorIndex::create(
          VectorIndex::Metadata{
              .columnName = "embedding",
              .dimensions = kDimensions,
              .metric = VectorDistanceMetric::kL2,
              .indexType = VectorIndexType::kIvfFlat,
              .numVectors = 0,
          },
          written.indexData),
      "VectorIndex vector count must be positive");
}

TEST_F(VectorIndexTest, writerRoundTripWithMultipleIndexes) {
  constexpr uint32_t kNumVectors{200};
  constexpr uint32_t kQueryRow{42};
  const auto firstData = generateRandomVectors(kNumVectors, kDimensions);
  const auto secondData =
      generateRandomVectors(kNumVectors, kDimensions, /*seed=*/99);
  const auto firstInput = makeInputFromVectors(firstData, kDimensions);
  const auto secondInput = makeInputFromVectors(secondData, kDimensions);
  const auto type = velox::ROW({
      {"id", velox::INTEGER()},
      {"first_embedding", velox::ARRAY(velox::REAL())},
      {"second_embedding", velox::ARRAY(velox::REAL())},
  });
  const auto input = std::make_shared<velox::RowVector>(
      pool(),
      type,
      nullptr,
      kNumVectors,
      std::vector<velox::VectorPtr>{
          firstInput->childAt(0),
          firstInput->childAt(1),
          secondInput->childAt(1),
      });

  auto firstConfig = makeConfig(VectorIndexType::kIvfFlat);
  firstConfig.columnName = "first_embedding";
  auto secondConfig = makeConfig(VectorIndexType::kIvfSq8);
  secondConfig.columnName = "second_embedding";
  WriterOptions writerOptions;
  writerOptions.vectorIndexConfigs = {firstConfig, secondConfig};
  writerOptions.vectorIndexWriterFactory = VectorIndexWriter::create;

  std::string fileData;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&fileData);
  Writer writer(type, std::move(writeFile), *pool_, std::move(writerOptions));
  writer.write(input);
  writer.close();

  auto readFile =
      std::make_shared<velox::InMemoryReadFile>(std::string_view{fileData});
  TabletReader::Options readerOptions;
  readerOptions.ioOptions.emplace(pool())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
      .setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
  const auto tablet = TabletReader::create(readFile, pool(), readerOptions);
  auto directory =
      tablet->loadOptionalSection(std::string{kVectorIndexSection});
  ASSERT_TRUE(directory.has_value());

  velox::io::ReaderOptions indexIoOptions(pool());
  indexIoOptions.setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
  IndexLookup::Options indexOptions{
      .file = readFile,
      .ioOptions = &indexIoOptions,
  };
  const auto vectorIndexDirectory =
      VectorIndexDirectory::create(std::move(directory.value()), indexOptions);
  ASSERT_EQ(vectorIndexDirectory.numIndexes(), 2);

  const std::array expectedColumns{"first_embedding", "second_embedding"};
  const std::array<const std::vector<float>*, 2> data{
      &firstData,
      &secondData,
  };
  for (size_t i = 0; i < expectedColumns.size(); ++i) {
    SCOPED_TRACE(fmt::format("column={}", expectedColumns[i]));
    EXPECT_TRUE(vectorIndexDirectory.contains(expectedColumns[i]));
    const auto vectorIndex = vectorIndexDirectory.load(expectedColumns[i]);
    const auto queryBegin = data[i]->begin() + kQueryRow * kDimensions;
    const std::vector<float> query(queryBegin, queryBegin + kDimensions);
    const auto results = vectorIndex->search({
        .queryVector = query,
        .numNeighbors = 1,
        .numProbes = 8,
    });
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results.front().rowId, kQueryRow);
  }

  EXPECT_FALSE(vectorIndexDirectory.contains("missing"));
  NIMBLE_ASSERT_THROW(
      vectorIndexDirectory.load("missing"),
      "Vector index column does not exist");
}

TEST_F(VectorIndexTest, partitionMetadataMatchesBuiltIndex) {
  constexpr uint32_t kNumVectors = 100;
  auto config = makeConfig(VectorIndexType::kIvfFlat);
  config.numPartitions = 8;
  const auto data = generateRandomVectors(kNumVectors, kDimensions);
  const auto written =
      writeIndex(config, {makeInputFromVectors(data, kDimensions)});

  const auto* index = serializedIndex(written);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(index->config(), nullptr);
  EXPECT_EQ(index->config()->num_partitions(), 2);
}

TEST_F(VectorIndexTest, hnswMetadataHasNoIvfPartitions) {
  constexpr uint32_t kNumVectors = 100;
  auto config = makeConfig(VectorIndexType::kHnswSq8);
  config.numPartitions = 8;
  const auto data = generateRandomVectors(kNumVectors, kDimensions);
  const auto written =
      writeIndex(config, {makeInputFromVectors(data, kDimensions)});

  const auto* index = serializedIndex(written);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(index->config(), nullptr);
  EXPECT_EQ(index->config()->num_partitions(), 0);
}

TEST_F(VectorIndexTest, hnswSearchDepth) {
  constexpr uint32_t kNumVectors{200};
  const auto data = generateRandomVectors(kNumVectors, kDimensions);
  const auto written = writeIndex(
      makeConfig(VectorIndexType::kHnswSq8),
      {makeInputFromVectors(data, kDimensions)});
  const auto reader = readIndex(written);

  auto searchConfig = VectorIndex::SearchConfig{
      .queryVector =
          std::vector<float>(data.begin(), data.begin() + kDimensions),
      .numNeighbors = 1,
      .hnswSearchDepth = 16,
  };
  const auto results = reader->search(searchConfig);
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results.front().rowId, 0);

  searchConfig.hnswSearchDepth = 0;
  NIMBLE_ASSERT_THROW(
      reader->search(searchConfig), "HNSW search depth must be positive");
}

TEST_F(VectorIndexTest, metadataAllIndexTypes) {
  struct TestParam {
    VectorIndexType indexType;
    VectorDistanceMetric metric;
    std::string debugString() const {
      return fmt::format(
          "indexType={}, metric={}",
          static_cast<int>(indexType),
          static_cast<int>(metric));
    }
  };

  std::vector<TestParam> testParams = {
      {VectorIndexType::kIvfFlat, VectorDistanceMetric::kL2},
      {VectorIndexType::kIvfSq8, VectorDistanceMetric::kCosine},
      {VectorIndexType::kIvfPq, VectorDistanceMetric::kDotProduct},
      {VectorIndexType::kIvfRaBitQ, VectorDistanceMetric::kL2},
      {VectorIndexType::kHnswSq8, VectorDistanceMetric::kL2},
  };

  for (const auto& param : testParams) {
    SCOPED_TRACE(param.debugString());

    auto config = makeConfig(param.indexType, param.metric);
    if (param.indexType == VectorIndexType::kIvfPq) {
      config.pqSubQuantizers = 8;
    }
    auto data = generateRandomVectors(500, kDimensions);
    auto batch = makeInputFromVectors(data, kDimensions);
    auto written = writeIndex(config, {batch});
    ASSERT_FALSE(written.directoryData.empty());

    auto reader = readIndex(written);
    ASSERT_NE(reader, nullptr);
    EXPECT_EQ(reader->indexType(), param.indexType);
    EXPECT_EQ(reader->metric(), param.metric);
    EXPECT_EQ(reader->dimensions(), kDimensions);
    EXPECT_EQ(reader->numVectors(), 500);

    const auto results = reader->search({
        .queryVector =
            std::vector<float>(data.begin(), data.begin() + kDimensions),
        .numNeighbors = 10,
        .numProbes = 8,
        .hnswSearchDepth = 32,
    });
    ASSERT_FALSE(results.empty());
    EXPECT_LE(results.size(), 10);
    for (const auto& result : results) {
      EXPECT_GE(result.rowId, 0);
      EXPECT_LT(result.rowId, 500);
      EXPECT_TRUE(std::isfinite(result.score));
    }
  }
}

// ---------------------------------------------------------------------------
// Multiple batch / incremental write tests.
// ---------------------------------------------------------------------------

TEST_F(VectorIndexTest, multipleBatchesAccumulate) {
  const uint32_t numVectorsPerBatch = 200;
  const uint32_t numBatches = 5;
  const uint32_t totalVectors = numVectorsPerBatch * numBatches;

  auto config = makeConfig(VectorIndexType::kIvfFlat);

  std::vector<velox::RowVectorPtr> batches;
  for (uint32_t batchIndex = 0; batchIndex < numBatches; ++batchIndex) {
    auto data = generateRandomVectors(
        numVectorsPerBatch, kDimensions, kSeed + static_cast<int>(batchIndex));
    batches.push_back(makeInputFromVectors(
        data,
        kDimensions,
        static_cast<int32_t>(batchIndex * numVectorsPerBatch)));
  }

  auto written = writeIndex(config, batches);
  ASSERT_FALSE(written.directoryData.empty());

  auto reader = readIndex(written);
  ASSERT_NE(reader, nullptr);
  EXPECT_EQ(reader->numVectors(), totalVectors);

  // Search should find vectors from any batch.
  auto queries = generateRandomVectors(10, kDimensions, /*seed=*/77);
  for (size_t queryIndex = 0; queryIndex < 10; ++queryIndex) {
    const auto queryOffset = static_cast<std::ptrdiff_t>(
        queryIndex * static_cast<size_t>(kDimensions));
    const auto nextQueryOffset = static_cast<std::ptrdiff_t>(
        (queryIndex + 1) * static_cast<size_t>(kDimensions));
    std::vector<float> queryVector(
        queries.begin() + queryOffset, queries.begin() + nextQueryOffset);
    VectorIndex::SearchConfig searchConfig{
        .queryVector = queryVector,
        .numNeighbors = 5,
        .numProbes = 8,
    };
    auto results = reader->search(searchConfig);
    ASSERT_FALSE(results.empty());
    for (const auto& result : results) {
      EXPECT_GE(result.rowId, 0);
      EXPECT_LT(result.rowId, static_cast<int64_t>(totalVectors));
    }
  }
}

TEST_F(VectorIndexTest, invalidBatchDoesNotMutateOtherIndexes) {
  constexpr velox::vector_size_t kNumVectors{100};
  const auto firstInput = makeInputFromVectors(
      generateRandomVectors(kNumVectors, kDimensions), kDimensions);
  const auto secondInput = makeInputFromVectors(
      generateRandomVectors(kNumVectors, kDimensions, /*seed=*/99),
      kDimensions);
  const auto type = velox::ROW({
      {"id", velox::INTEGER()},
      {"first_embedding", velox::ARRAY(velox::REAL())},
      {"second_embedding", velox::ARRAY(velox::REAL())},
  });
  const auto input = std::make_shared<velox::RowVector>(
      pool(),
      type,
      nullptr,
      kNumVectors,
      std::vector<velox::VectorPtr>{
          firstInput->childAt(0),
          firstInput->childAt(1),
          secondInput->childAt(1),
      });

  auto firstConfig = makeConfig(VectorIndexType::kIvfFlat);
  firstConfig.columnName = "first_embedding";
  auto secondConfig = makeConfig(VectorIndexType::kIvfSq8);
  secondConfig.columnName = "second_embedding";
  const std::array configs{firstConfig, secondConfig};
  auto writer = VectorIndexWriter::create(configs, type, pool());

  input->childAt(2)->setNull(0, true);
  NIMBLE_ASSERT_THROW(
      writer->write(input), "Vector index column contains a null row");
  input->childAt(2)->setNull(0, false);
  writer->write(input);

  const auto written = closeWriter(*writer);
  const auto* directory =
      flatbuffers::GetRoot<serialization::VectorIndexDirectory>(
          written.directoryData.data());
  ASSERT_NE(directory, nullptr);
  ASSERT_NE(directory->indices(), nullptr);
  ASSERT_EQ(directory->indices()->size(), configs.size());
  for (const auto* descriptor : *directory->indices()) {
    ASSERT_NE(descriptor, nullptr);
    ASSERT_NE(descriptor->config(), nullptr);
    EXPECT_EQ(descriptor->config()->num_vectors(), kNumVectors);
  }
}

// ---------------------------------------------------------------------------
// Edge cases and error handling.
// ---------------------------------------------------------------------------

TEST_F(VectorIndexTest, emptyInputOmitsSections) {
  auto config = makeConfig(VectorIndexType::kIvfFlat);
  auto type = createType();
  auto writer = createWriter(config, type);

  bool dataSectionWritten = false;
  bool sectionWritten = false;
  const CreateMetadataSectionFn createMetadataFn =
      [&dataSectionWritten](std::string_view) {
        dataSectionWritten = true;
        return MetadataSection{};
      };
  auto writeMetadataFn = [&sectionWritten](
                             const std::string&, std::string_view) {
    sectionWritten = true;
  };
  writer->close(createMetadataFn, writeMetadataFn);
  EXPECT_FALSE(dataSectionWritten);
  EXPECT_FALSE(sectionWritten);
}

TEST_F(VectorIndexTest, configuredWithoutWriterFactoryFails) {
  const auto type = createType();
  WriterOptions writerOptions;
  writerOptions.vectorIndexConfigs = {makeConfig(VectorIndexType::kIvfFlat)};
  // vectorIndexWriterFactory deliberately left unset.

  std::string fileData;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&fileData);
  NIMBLE_ASSERT_USER_THROW(
      Writer(type, std::move(writeFile), *pool(), std::move(writerOptions)),
      "WriterOptions::vectorIndexWriterFactory must be set");
}

TEST_F(VectorIndexTest, emptyFileOmitsVectorIndex) {
  const auto type = createType();
  WriterOptions writerOptions;
  writerOptions.vectorIndexConfigs = {makeConfig(VectorIndexType::kIvfFlat)};
  writerOptions.vectorIndexWriterFactory = VectorIndexWriter::create;

  std::string fileData;
  auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&fileData);
  Writer writer(type, std::move(writeFile), *pool(), std::move(writerOptions));
  writer.close();

  auto readFile = std::make_shared<velox::InMemoryReadFile>(fileData);
  TabletReader::Options readerOptions;
  readerOptions.ioOptions.emplace(pool())
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
      .setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
  const auto tablet = TabletReader::create(readFile, pool(), readerOptions);
  EXPECT_FALSE(tablet->hasOptionalSection(std::string{kVectorIndexSection}));
}

TEST_F(VectorIndexTest, noConfig) {
  auto type = createType();
  NIMBLE_ASSERT_THROW(
      VectorIndexWriter::create(
          std::span<const VectorIndexConfig>{}, type, pool()),
      "Vector index configs must not be empty");
}

TEST_F(VectorIndexTest, nullVectorRejected) {
  auto writer =
      createWriter(makeConfig(VectorIndexType::kIvfFlat), createType());
  const auto data = generateRandomVectors(10, kDimensions);
  const auto batch = makeInputFromVectors(data, kDimensions);
  batch->childAt(1)->setNull(3, true);

  NIMBLE_ASSERT_THROW(
      writer->write(batch), "Vector index column contains a null row");
}

TEST_F(VectorIndexTest, topLevelNullRejected) {
  auto writer =
      createWriter(makeConfig(VectorIndexType::kIvfFlat), createType());
  const auto data = generateRandomVectors(10, kDimensions);
  const auto batch = makeInputFromVectors(data, kDimensions);
  batch->setNull(3, true);

  NIMBLE_ASSERT_THROW(
      writer->write(batch), "Input contains a null top-level row");
}

TEST_F(VectorIndexTest, usesTopLevelRowCount) {
  constexpr velox::vector_size_t kNumRows{5};
  const auto data = generateRandomVectors(10, kDimensions);
  const auto batch = makeInputFromVectors(data, kDimensions);
  const auto shorterBatch = std::make_shared<velox::RowVector>(
      pool(), createType(), nullptr, kNumRows, batch->children());

  const auto written =
      writeIndex(makeConfig(VectorIndexType::kIvfFlat), {shorterBatch});
  const auto* index = serializedIndex(written);
  ASSERT_NE(index, nullptr);
  ASSERT_NE(index->config(), nullptr);
  EXPECT_EQ(index->config()->num_vectors(), kNumRows);
}

TEST_F(VectorIndexTest, dictionaryEncodedVectors) {
  constexpr velox::vector_size_t kNumVectors{100};
  const auto data = generateRandomVectors(kNumVectors, kDimensions);
  const auto input = makeInputFromVectors(data, kDimensions);
  const auto* arrayVector = input->childAt(1)->as<velox::ArrayVector>();
  ASSERT_NE(arrayVector, nullptr);

  const auto numElements = kNumVectors * kDimensions;
  auto elementIndices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(numElements, pool());
  auto* rawElementIndices = elementIndices->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numElements; ++i) {
    rawElementIndices[i] = i;
  }
  const auto encodedElements = velox::BaseVector::wrapInDictionary(
      nullptr, elementIndices, numElements, arrayVector->elements());
  const auto encodedArray = std::make_shared<velox::ArrayVector>(
      pool(),
      velox::ARRAY(velox::REAL()),
      nullptr,
      kNumVectors,
      arrayVector->offsets(),
      arrayVector->sizes(),
      encodedElements);

  auto rowIndices =
      velox::AlignedBuffer::allocate<velox::vector_size_t>(kNumVectors, pool());
  auto* rawRowIndices = rowIndices->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t row = 0; row < kNumVectors; ++row) {
    rawRowIndices[row] = kNumVectors - row - 1;
  }
  const auto encodedEmbedding = velox::BaseVector::wrapInDictionary(
      nullptr, rowIndices, kNumVectors, encodedArray);
  const auto encodedInput = std::make_shared<velox::RowVector>(
      pool(),
      createType(),
      nullptr,
      kNumVectors,
      std::vector<velox::VectorPtr>{input->childAt(0), encodedEmbedding});

  auto config = makeConfig(VectorIndexType::kIvfFlat);
  config.numPartitions = 2;
  const auto written = writeIndex(config, {encodedInput});
  const auto reader = readIndex(written);
  const auto queryBegin = data.end() - kDimensions;
  const auto results = reader->search({
      .queryVector = std::vector<float>(queryBegin, data.end()),
      .numNeighbors = 1,
      .numProbes = 2,
  });

  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results.front().rowId, 0);
}

TEST_F(VectorIndexTest, nullVectorElementRejected) {
  auto writer =
      createWriter(makeConfig(VectorIndexType::kIvfFlat), createType());
  const auto data = generateRandomVectors(10, kDimensions);
  const auto batch = makeInputFromVectors(data, kDimensions);
  batch->childAt(1)->as<velox::ArrayVector>()->elements()->setNull(
      kDimensions + 1, true);

  NIMBLE_ASSERT_THROW(
      writer->write(batch), "Vector index column contains a null element");
}

TEST_F(VectorIndexTest, invalidPqConfigurationRejected) {
  auto config = makeConfig(VectorIndexType::kIvfPq);

  for (const uint32_t numSubQuantizers : {0, 3}) {
    SCOPED_TRACE(fmt::format("numSubQuantizers={}", numSubQuantizers));
    config.pqSubQuantizers = numSubQuantizers;
    NIMBLE_ASSERT_THROW(createWriter(config, createType()), "PQ");
  }

  config.pqSubQuantizers = 8;
  for (const uint8_t bitsPerCode : {0, 25}) {
    SCOPED_TRACE(fmt::format("bitsPerCode={}", bitsPerCode));
    config.pqBits = bitsPerCode;
    NIMBLE_ASSERT_THROW(createWriter(config, createType()), "PQ bits per code");
  }
}

TEST_F(VectorIndexTest, invalidIndexSizeLimitRejected) {
  for (const uint64_t maxIndexSizeBytes :
       std::array<uint64_t, 2>{0, kMaxVectorIndexSizeBytes + 1}) {
    SCOPED_TRACE(fmt::format("maxIndexSizeBytes={}", maxIndexSizeBytes));
    auto config = makeConfig(VectorIndexType::kIvfFlat);
    config.maxIndexSizeBytes = maxIndexSizeBytes;
    NIMBLE_ASSERT_THROW(
        createWriter(config, createType()), "Vector index size limit");
  }
}

TEST_F(VectorIndexTest, bufferedVectorSizeLimitExceeded) {
  auto config = makeConfig(VectorIndexType::kIvfFlat);
  config.maxBufferedVectorSizeBytes = 9 * kDimensions * sizeof(float);
  auto writer = createWriter(config, createType());
  const auto input =
      makeInputFromVectors(generateRandomVectors(10, kDimensions), kDimensions);

  NIMBLE_ASSERT_THROW(writer->write(input), "Buffered vector data exceeds");
}

TEST_F(VectorIndexTest, dimensionMismatchOnSearch) {
  auto data = generateRandomVectors(200, kDimensions);
  auto config = makeConfig(VectorIndexType::kIvfFlat);
  auto batch = makeInputFromVectors(data, kDimensions);
  auto written = writeIndex(config, {batch});

  auto reader = readIndex(written);
  ASSERT_NE(reader, nullptr);

  VectorIndex::SearchConfig searchConfig{
      .queryVector = std::vector<float>(kDimensions + 1, 0.0f),
      .numNeighbors = 5,
  };
  NIMBLE_ASSERT_THROW(
      reader->search(searchConfig),
      "Query vector dimensions do not match the index");
}

TEST_F(VectorIndexTest, emptyDirectoryRejected) {
  NIMBLE_ASSERT_THROW(
      readDirectory({}), "Vector index directory must not be empty");
}

TEST_F(VectorIndexTest, missingIoStatsDeferredUntilLoad) {
  constexpr uint32_t kNumVectors{100};
  const auto written = writeIndex(
      makeConfig(VectorIndexType::kIvfFlat),
      {makeInputFromVectors(
          generateRandomVectors(kNumVectors, kDimensions), kDimensions)});
  auto directoryBuffer = MetadataBuffer::decompress(
      written.directoryData, CompressionType::Uncompressed, pool());
  velox::io::ReaderOptions ioOptions(pool());
  IndexLookup::Options indexOptions{
      .file = std::make_shared<velox::InMemoryReadFile>(written.indexData),
      .ioOptions = &ioOptions,
  };

  const auto directory = VectorIndexDirectory::create(
      Section{MetadataBuffer{std::move(directoryBuffer)}}, indexOptions);
  EXPECT_TRUE(directory.contains("embedding"));
  NIMBLE_ASSERT_THROW(directory.load("embedding"), "indexIoStats must be set");
}

TEST_F(VectorIndexTest, invalidDirectoryLimitsRejected) {
  constexpr uint32_t kNumVectors{100};
  const auto written = writeIndex(
      makeConfig(VectorIndexType::kIvfFlat),
      {makeInputFromVectors(
          generateRandomVectors(kNumVectors, kDimensions), kDimensions)});
  const auto* original = serializedIndex(written);
  ASSERT_NE(original->config(), nullptr);
  ASSERT_NE(original->index_data(), nullptr);

  struct TestCase {
    uint32_t dimensions;
    uint64_t numVectors;
    uint32_t indexSize;
    uint32_t uncompressedSize;
  };
  const auto* originalSection = original->index_data();
  const auto originalUncompressedSize = originalSection->uncompressed_size();
  const std::array testCases{
      TestCase{
          0,
          kNumVectors,
          originalSection->size(),
          originalUncompressedSize,
      },
      TestCase{
          static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1,
          kNumVectors,
          originalSection->size(),
          originalUncompressedSize,
      },
      TestCase{
          kDimensions,
          0,
          originalSection->size(),
          originalUncompressedSize,
      },
      TestCase{
          kDimensions,
          static_cast<uint64_t>(std::numeric_limits<faiss::idx_t>::max()) + 1,
          originalSection->size(),
          originalUncompressedSize,
      },
      TestCase{
          kDimensions,
          kNumVectors,
          0,
          originalUncompressedSize,
      },
      TestCase{
          kDimensions,
          kNumVectors,
          static_cast<uint32_t>(kMaxVectorIndexSizeBytes + 1),
          originalUncompressedSize,
      },
      TestCase{
          kDimensions,
          kNumVectors,
          originalSection->size(),
          static_cast<uint32_t>(kMaxVectorIndexSizeBytes + 1),
      },
      TestCase{kDimensions, kNumVectors, originalSection->size(), 0},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "dimensions={}, numVectors={}, indexSize={}, uncompressedSize={}",
            testCase.dimensions,
            testCase.numVectors,
            testCase.indexSize,
            testCase.uncompressedSize));
    flatbuffers::FlatBufferBuilder builder;
    const auto config = serialization::CreateVectorIndexMeta(
        builder,
        builder.CreateString(original->config()->column_name()->string_view()),
        testCase.dimensions,
        original->config()->metric(),
        original->config()->index_type(),
        original->config()->num_partitions(),
        testCase.numVectors);
    const auto section = serialization::CreateMetadataSection(
        builder,
        originalSection->offset(),
        testCase.indexSize,
        originalSection->compression_type(),
        testCase.uncompressedSize);
    const auto descriptor =
        serialization::CreateVectorIndex(builder, config, section);
    builder.Finish(
        serialization::CreateVectorIndexDirectory(
            builder, builder.CreateVector(&descriptor, 1)));

    auto invalid = written;
    invalid.directoryData.assign(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize());
    NIMBLE_ASSERT_THROW(readDirectory(invalid), "VectorIndex");
  }
}

TEST_F(VectorIndexTest, invalidDirectoryEnumsRejectedAsCorruptFile) {
  constexpr uint32_t kNumVectors{100};
  const auto written = writeIndex(
      makeConfig(VectorIndexType::kIvfFlat),
      {makeInputFromVectors(
          generateRandomVectors(kNumVectors, kDimensions), kDimensions)});
  const auto* original = serializedIndex(written);
  ASSERT_NE(original->config(), nullptr);
  ASSERT_NE(original->index_data(), nullptr);

  enum class Field {
    kMetric,
    kIndexType,
    kCompressionType,
  };
  const std::array fields{
      Field::kMetric,
      Field::kIndexType,
      Field::kCompressionType,
  };

  for (const auto field : fields) {
    SCOPED_TRACE(fmt::format("field={}", static_cast<int>(field)));
    flatbuffers::FlatBufferBuilder builder;
    builder.ForceDefaults(true);
    const auto config = serialization::CreateVectorIndexMeta(
        builder,
        builder.CreateString(original->config()->column_name()->string_view()),
        original->config()->dimensions(),
        serialization::VectorDistanceMetric_Cosine,
        serialization::VectorIndexType_IVF_SQ8,
        original->config()->num_partitions(),
        original->config()->num_vectors());
    const auto section = serialization::CreateMetadataSection(
        builder,
        original->index_data()->offset(),
        original->index_data()->size(),
        serialization::CompressionType_Zstd,
        original->index_data()->uncompressed_size());
    const auto descriptorOffset =
        serialization::CreateVectorIndex(builder, config, section);
    builder.Finish(
        serialization::CreateVectorIndexDirectory(
            builder, builder.CreateVector(&descriptorOffset, 1)));

    WrittenIndexes invalid{
        .indexData = written.indexData,
        .directoryData = std::string(
            reinterpret_cast<const char*>(builder.GetBufferPointer()),
            builder.GetSize()),
    };
    const auto* directory =
        flatbuffers::GetRoot<serialization::VectorIndexDirectory>(
            invalid.directoryData.data());
    auto* descriptor =
        const_cast<serialization::VectorIndex*>(directory->indices()->Get(0));
    if (field == Field::kCompressionType) {
      auto* sectionTable = reinterpret_cast<flatbuffers::Table*>(
          const_cast<serialization::MetadataSection*>(
              descriptor->index_data()));
      ASSERT_TRUE(sectionTable->SetField<uint8_t>(
          serialization::MetadataSection::VT_COMPRESSION_TYPE, 255));
    } else {
      auto* configTable = reinterpret_cast<flatbuffers::Table*>(
          const_cast<serialization::VectorIndexMeta*>(descriptor->config()));
      const auto fieldOffset = field == Field::kMetric
          ? serialization::VectorIndexMeta::VT_METRIC
          : serialization::VectorIndexMeta::VT_INDEX_TYPE;
      ASSERT_TRUE(configTable->SetField<int8_t>(fieldOffset, 127));
    }
    EXPECT_THROW(readDirectory(invalid), NimbleUserError);
  }
}

TEST_F(VectorIndexTest, numNeighborsLargerThanDataset) {
  auto data = generateRandomVectors(50, kDimensions);
  auto config = makeConfig(VectorIndexType::kIvfFlat);
  config.numPartitions = 2;
  auto batch = makeInputFromVectors(data, kDimensions);
  auto written = writeIndex(config, {batch});

  auto reader = readIndex(written);
  ASSERT_NE(reader, nullptr);

  std::vector<float> queryVector(kDimensions, 0.5f);
  VectorIndex::SearchConfig searchConfig{
      .queryVector = queryVector,
      .numNeighbors = 100,
      .numProbes = 2,
  };
  auto results = reader->search(searchConfig);

  // Should return at most the number of vectors in the dataset.
  EXPECT_LE(results.size(), 50);
  EXPECT_FALSE(results.empty());
}

TEST_F(VectorIndexTest, singleVector) {
  std::vector<float> data(kDimensions, 1.0f);
  auto config = makeConfig(VectorIndexType::kIvfFlat);
  config.numPartitions = 1;
  auto batch = makeInputFromVectors(data, kDimensions);
  auto written = writeIndex(config, {batch});
  ASSERT_FALSE(written.directoryData.empty());

  auto reader = readIndex(written);
  ASSERT_NE(reader, nullptr);
  EXPECT_EQ(reader->numVectors(), 1);

  VectorIndex::SearchConfig searchConfig{
      .queryVector = std::vector<float>(kDimensions, 1.0f),
      .numNeighbors = 1,
      .numProbes = 1,
  };
  auto results = reader->search(searchConfig);
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].rowId, 0);
  EXPECT_NEAR(results[0].score, 0.0f, 1e-6f);
}

TEST_F(VectorIndexTest, maxIndexSizeExceeded) {
  auto data = generateRandomVectors(500, kDimensions);
  auto config = makeConfig(VectorIndexType::kIvfFlat);
  config.maxIndexSizeBytes = 1;

  auto type = createType();
  auto writer = createWriter(config, type);
  auto batch = makeInputFromVectors(data, kDimensions);
  writer->write(batch);

  bool dataSectionWritten = false;
  bool sectionWritten = false;
  const CreateMetadataSectionFn createMetadataFn =
      [&dataSectionWritten](std::string_view) {
        dataSectionWritten = true;
        return MetadataSection{};
      };
  auto writeMetadataFn = [&sectionWritten](
                             const std::string&, std::string_view) {
    sectionWritten = true;
  };
  NIMBLE_ASSERT_THROW(
      writer->close(createMetadataFn, writeMetadataFn),
      "Serialized vector index exceeds the configured limit");
  EXPECT_FALSE(dataSectionWritten);
  EXPECT_FALSE(sectionWritten);
}

TEST_F(VectorIndexTest, missingUncompressedIndexSizeRejected) {
  constexpr uint32_t kNumVectors{100};
  auto writer =
      createWriter(makeConfig(VectorIndexType::kIvfFlat), createType());
  writer->write(makeInputFromVectors(
      generateRandomVectors(kNumVectors, kDimensions), kDimensions));

  bool sectionWritten{false};
  const CreateMetadataSectionFn createMetadataFn =
      [](std::string_view content) {
        return MetadataSection{
            0,
            static_cast<uint32_t>(content.size()),
            CompressionType::Uncompressed,
        };
      };
  const WriteOptionalSectionFn writeMetadataFn =
      [&sectionWritten](const std::string&, std::string_view) {
        sectionWritten = true;
      };

  NIMBLE_ASSERT_THROW(
      writer->close(createMetadataFn, writeMetadataFn),
      "Persisted vector index must record its uncompressed size");
  EXPECT_FALSE(sectionWritten);
}

} // namespace
} // namespace facebook::nimble::index::test
