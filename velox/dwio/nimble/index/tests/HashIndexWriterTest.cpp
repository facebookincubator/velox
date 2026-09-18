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

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/HashIndex.h"
#include "velox/dwio/nimble/index/HashIndexConfig.h"
#include "velox/dwio/nimble/index/HashIndexUtils.h"
#include "velox/dwio/nimble/index/HashIndexWriter.h"
#include "velox/dwio/nimble/index/tests/HashIndexTestUtils.h"
#include "velox/dwio/nimble/tablet/HashIndexGenerated.h"

#include "velox/common/memory/Memory.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble::index::test {
namespace {

class HashIndexWriterTestBase : public velox::test::VectorTestBase {
 protected:
  static constexpr std::string_view kCol0{"col0"};
  static constexpr std::string_view kCol1{"col1"};
  static constexpr std::string_view kCol2{"col2"};

  void setupType() {
    type_ = velox::ROW(
        {{std::string(kCol0), velox::VARCHAR()},
         {std::string(kCol1), velox::INTEGER()},
         {std::string(kCol2), velox::VARCHAR()}});
  }

  static std::shared_ptr<const IndexConfig> config(
      std::vector<std::string> columns) {
    return HashIndexConfigBuilder{}.withKeyColumns(std::move(columns)).build();
  }

  // Creates a row vector with the given key column values.
  // col0 (VARCHAR) is derived from keyValues for uniqueness.
  velox::RowVectorPtr makeInput(const std::vector<int32_t>& keyValues) {
    col0Strings_.clear();
    col0Strings_.reserve(keyValues.size());
    std::vector<velox::StringView> col0Vals;
    std::vector<velox::StringView> col2Vals;
    col0Vals.reserve(keyValues.size());
    col2Vals.reserve(keyValues.size());
    for (const auto val : keyValues) {
      col0Strings_.emplace_back(fmt::format("s_{}", val));
      col0Vals.emplace_back(col0Strings_.back());
      col2Vals.emplace_back("b");
    }
    return makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeFlatVector<velox::StringView>(col0Vals),
         makeFlatVector<int32_t>(keyValues),
         makeFlatVector<velox::StringView>(col2Vals)});
  }

  // Builds a mock createMetadataSection function that stores sections
  // in memory and returns MetadataSection with sequential offsets.
  struct MockSectionStore {
    std::vector<std::string> sections;
    std::vector<MetadataSection> metadataSections;
    uint64_t nextOffset{0};

    CreateMetadataSectionFn createFn() {
      return [this](std::string_view content) -> MetadataSection {
        const auto offset = nextOffset;
        const auto size = content.size();
        nextOffset += size;
        auto section = MetadataSection{
            offset, static_cast<uint32_t>(size), CompressionType::Uncompressed};
        sections.emplace_back(content);
        metadataSections.emplace_back(section);
        return section;
      };
    }

    const std::string& sectionContent(const MetadataSection& section) const {
      for (size_t i = 0; i < metadataSections.size(); ++i) {
        if (metadataSections[i].offset() == section.offset()) {
          return sections[i];
        }
      }
      NIMBLE_UNREACHABLE("Missing metadata section");
    }
  };

  // No-op data write function for tests.
  static WriteDataFn noopWriteDataFn() {
    return [](const std::vector<std::string_view>&)
               -> std::pair<uint64_t, uint32_t> { return {0, 0}; };
  }

  static std::string closeAndReadDirectory(
      HashIndexWriter& writer,
      MockSectionStore& store) {
    auto descriptor = writer.close(noopWriteDataFn(), store.createFn());
    NIMBLE_CHECK(descriptor.has_value(), "Expected hash index descriptor");
    return store.sectionContent(descriptor->root);
  }

  // Validates the HashIndexDirectory has the expected indices with the
  // expected column names per index.
  static void validateDirectory(
      const std::string& directory,
      const std::vector<std::vector<std::string>>& expectedColumns) {
    const auto* root = flatbuffers::GetRoot<serialization::HashIndexDirectory>(
        directory.data());
    ASSERT_NE(root, nullptr);
    const auto* indices = root->indices();
    ASSERT_NE(indices, nullptr);
    ASSERT_EQ(indices->size(), expectedColumns.size());

    for (size_t idx = 0; idx < expectedColumns.size(); ++idx) {
      SCOPED_TRACE(fmt::format("index {}", idx));
      const auto* section = indices->Get(idx);
      ASSERT_NE(section, nullptr);
      const auto* columns = section->index_columns();
      ASSERT_NE(columns, nullptr);
      ASSERT_EQ(columns->size(), expectedColumns[idx].size());
      for (size_t i = 0; i < expectedColumns[idx].size(); ++i) {
        EXPECT_EQ(columns->Get(i)->string_view(), expectedColumns[idx][i]);
      }
    }
  }

  static const std::string& indexSectionContent(
      const std::string& directory,
      size_t index,
      const MockSectionStore& store) {
    const auto* root = flatbuffers::GetRoot<serialization::HashIndexDirectory>(
        directory.data());
    NIMBLE_CHECK_NOT_NULL(root);
    const auto* indices = root->indices();
    NIMBLE_CHECK_NOT_NULL(indices);
    NIMBLE_CHECK_LT(index, indices->size());
    const auto* indexDescriptor = indices->Get(index);
    NIMBLE_CHECK_NOT_NULL(indexDescriptor);
    const auto* section = indexDescriptor->section();
    NIMBLE_CHECK_NOT_NULL(section);
    return store.sectionContent(
        MetadataSection{
            section->offset(),
            section->size(),
            static_cast<CompressionType>(section->compression_type())});
  }

  // Validates the HashIndex FlatBuffer has the expected row/key counts
  // and valid partition structure. If expectedNumPartitions is 0, only
  // validates at least one partition exists.
  static void validateIndex(
      const std::string& indexSection,
      uint64_t expectedNumKeys,
      uint32_t expectedNumPartitions = 0) {
    const auto* hashIndex =
        flatbuffers::GetRoot<serialization::HashIndex>(indexSection.data());
    ASSERT_NE(hashIndex, nullptr);
    EXPECT_EQ(hashIndex->row_count(), expectedNumKeys);
    EXPECT_EQ(hashIndex->num_keys(), expectedNumKeys);
    EXPECT_GT(hashIndex->num_buckets(), 0u);

    const auto* partitionSections = hashIndex->partition_sections();
    ASSERT_NE(partitionSections, nullptr);
    if (expectedNumPartitions > 0) {
      EXPECT_EQ(partitionSections->size(), expectedNumPartitions);
    } else {
      EXPECT_GE(partitionSections->size(), 1u);
    }

    const auto* partitionStartBuckets = hashIndex->partition_start_buckets();
    ASSERT_NE(partitionStartBuckets, nullptr);
    EXPECT_EQ(partitionStartBuckets->size(), partitionSections->size());
  }

  std::unique_ptr<HashIndexWriter> createWriter(
      const std::shared_ptr<const IndexConfig>& config,
      const velox::TypePtr& type,
      velox::memory::MemoryPool* pool) const {
    const IndexConfig* configs[] = {config.get()};
    return HashIndexWriter::create(configs, type, pool);
  }

  velox::RowTypePtr type_;
  std::vector<std::string> col0Strings_;
};

// Non-parameterized fixture for tests with specific column configs.
class HashIndexWriterTest : public HashIndexWriterTestBase,
                            public testing::Test {
 public:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

 protected:
  void SetUp() override {
    setupType();
  }
};

// Parameterized fixture: runs with single-column and composite-key configs.
struct HashIndexWriterTestParam {
  std::vector<std::string> columns;

  std::string debugString() const {
    return fmt::format("columns: [{}]", fmt::join(columns, ", "));
  }
};

class HashIndexWriterParamTest
    : public HashIndexWriterTestBase,
      public testing::TestWithParam<HashIndexWriterTestParam> {
 public:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

 protected:
  void SetUp() override {
    setupType();
  }

  std::shared_ptr<const IndexConfig> config() const {
    return HashIndexConfigBuilder{}.withKeyColumns(GetParam().columns).build();
  }
};

INSTANTIATE_TEST_SUITE_P(
    SingleAndCompositeKey,
    HashIndexWriterParamTest,
    testing::Values(
        HashIndexWriterTestParam{{"col1"}},
        HashIndexWriterTestParam{{"col0", "col1"}}),
    [](const testing::TestParamInfo<HashIndexWriterTestParam>& info) {
      return fmt::format("columns_{}", fmt::join(info.param.columns, "_"));
    });

TEST_F(HashIndexWriterTest, createWithValidConfig) {
  auto writer = createWriter(config({"col1"}), type_, pool_.get());
  EXPECT_NE(writer, nullptr);
}

TEST_F(HashIndexWriterTest, createWithEmptyConfigs) {
  EXPECT_EQ(HashIndexWriter::create({}, type_, pool_.get()), nullptr);
}

TEST_F(HashIndexWriterTest, multipleIndices) {
  const std::vector<std::vector<std::vector<std::string>>> testCases{
      {{"col0"}, {"col1"}},
      {{"col0"}, {"col1"}, {"col0", "col1"}},
  };

  for (const auto& columnSets : testCases) {
    SCOPED_TRACE(fmt::format("indexCount={}", columnSets.size()));
    std::vector<std::shared_ptr<const IndexConfig>> configs;
    configs.reserve(columnSets.size());
    for (const auto& columns : columnSets) {
      configs.emplace_back(
          HashIndexConfigBuilder{}.withKeyColumns(columns).build());
    }
    std::vector<const IndexConfig*> configPtrs;
    configPtrs.reserve(configs.size());
    for (const auto& config : configs) {
      configPtrs.emplace_back(config.get());
    }

    auto writer = HashIndexWriter::create(configPtrs, type_, pool_.get());
    writer->write(makeInput({1, 2, 3, 4, 5}));

    HashIndexWriterTestHelper helper(writer.get());
    for (size_t i = 0; i < columnSets.size(); ++i) {
      EXPECT_EQ(helper.numEntries(i), 5);
    }

    MockSectionStore store;
    const auto directory = closeAndReadDirectory(*writer, store);
    validateDirectory(directory, columnSets);
  }
}

TEST_F(HashIndexWriterTest, rejectsDuplicateIndexColumns) {
  const auto first = config({"col1"});
  const auto second = config({"col1"});
  const IndexConfig* configs[] = {first.get(), second.get()};
  NIMBLE_ASSERT_THROW(
      HashIndexWriter::create(configs, type_, pool_.get()),
      "Duplicate hash index columns: [col1]");
}

TEST_F(HashIndexWriterTest, rejectsNonHashIndexName) {
  const auto config = std::make_shared<const HashIndexConfig>(
      std::string{kDenseSortedIndexName},
      std::vector<std::string>{"col1"},
      0.7f,
      std::nullopt,
      0);
  const IndexConfig* configs[] = {config.get()};
  NIMBLE_ASSERT_THROW(
      HashIndexWriter::create(configs, type_, pool_.get()),
      "Hash index writer must use the built-in hash index name");
}

TEST_F(HashIndexWriterTest, createWithNullPool) {
  NIMBLE_ASSERT_THROW(
      createWriter(config({"col1"}), type_, nullptr),
      "memory pool must not be null");
}

TEST_F(HashIndexWriterTest, createWithInvalidConfig) {
  {
    SCOPED_TRACE("empty columns");
    const auto invalidConfig = HashIndexConfigBuilder{}.build();
    NIMBLE_ASSERT_THROW(
        createWriter(invalidConfig, type_, pool_.get()),
        "Hash index must have at least one column");
  }

  for (const auto loadFactor : {0.0f, -0.5f, 1.5f}) {
    SCOPED_TRACE(loadFactor);
    const auto invalidConfig = HashIndexConfigBuilder{}
                                   .withKeyColumns({"col1"})
                                   .withLoadFactor(loadFactor)
                                   .build();
    NIMBLE_ASSERT_THROW(
        createWriter(invalidConfig, type_, pool_.get()),
        "Hash index load factor must be finite and in (0, 1]");
  }

  {
    SCOPED_TRACE("non-existent column");
    const auto invalidConfig = config({"non_existent"});
    VELOX_ASSERT_USER_THROW(
        createWriter(invalidConfig, type_, pool_.get()), "Field not found");
  }
}

TEST_P(HashIndexWriterParamTest, writeEmptyVector) {
  auto writer = createWriter(config(), type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  auto emptyBatch = makeInput({});
  writer->write(emptyBatch);

  HashIndexWriterTestHelper helper(writer.get());
  EXPECT_EQ(helper.numRows(), 0);
  EXPECT_EQ(helper.numEntries(), 0);

  MockSectionStore store;
  EXPECT_FALSE(writer->close(noopWriteDataFn(), store.createFn()).has_value());
  EXPECT_TRUE(store.sections.empty());
}

TEST_P(HashIndexWriterParamTest, writeSingleBatch) {
  auto writer = createWriter(config(), type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  writer->write(makeInput({1, 2, 3, 4, 5}));

  HashIndexWriterTestHelper helper(writer.get());
  EXPECT_EQ(helper.numRows(), 5);
  EXPECT_EQ(helper.numEntries(), 5);

  MockSectionStore store;
  auto directory = closeAndReadDirectory(*writer, store);

  const auto& columns = GetParam().columns;
  validateDirectory(directory, {columns});
  validateIndex(
      indexSectionContent(directory, /*index=*/0, store),
      5,
      /*expectedNumPartitions=*/1);
}

TEST_P(HashIndexWriterParamTest, writeMultipleBatches) {
  auto writer = createWriter(config(), type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  writer->write(makeInput({1, 2, 3}));
  writer->write(makeInput({4, 5}));
  writer->write(makeInput({6, 7, 8, 9}));

  HashIndexWriterTestHelper helper(writer.get());
  EXPECT_EQ(helper.numRows(), 9);
  EXPECT_EQ(helper.numEntries(), 9);

  MockSectionStore store;
  auto directory = closeAndReadDirectory(*writer, store);

  const auto& columns = GetParam().columns;
  validateDirectory(directory, {columns});
  validateIndex(
      indexSectionContent(directory, /*index=*/0, store),
      9,
      /*expectedNumPartitions=*/1);
}

TEST_P(HashIndexWriterParamTest, writeWithEmptyBatchesMixed) {
  auto writer = createWriter(config(), type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  writer->write(makeInput({}));
  writer->write(makeInput({1, 2}));
  writer->write(makeInput({}));
  writer->write(makeInput({3}));
  writer->write(makeInput({}));

  HashIndexWriterTestHelper helper(writer.get());
  EXPECT_EQ(helper.numRows(), 3);
  EXPECT_EQ(helper.numEntries(), 3);

  MockSectionStore store;
  auto directory = closeAndReadDirectory(*writer, store);

  const auto& columns = GetParam().columns;
  validateDirectory(directory, {columns});
  validateIndex(
      indexSectionContent(directory, /*index=*/0, store),
      3,
      /*expectedNumPartitions=*/1);
}

TEST_F(HashIndexWriterTest, rejectNullKeys) {
  struct TestCase {
    std::string name;
    std::vector<std::string> columns;
    std::vector<std::optional<velox::StringView>> col0;
    std::vector<std::optional<int32_t>> col1;
    bool expectThrow;

    std::string debugString() const {
      return name;
    }
  };

  const std::vector<TestCase> testCases = {
      {
          "single null in integer key",
          {"col1"},
          {{"a"}, {"b"}, {"c"}},
          {1, std::nullopt, 3},
          true,
      },
      {
          "all nulls in integer key",
          {"col1"},
          {{"a"}, {"b"}, {"c"}},
          {std::nullopt, std::nullopt, std::nullopt},
          true,
      },
      {
          "null in varchar key of composite index",
          {"col0", "col1"},
          {std::nullopt, {"b"}, {"c"}},
          {1, 2, 3},
          true,
      },
      {
          "no nulls passes",
          {"col1"},
          {{"a"}, {"b"}, {"c"}},
          {1, 2, 3},
          false,
      },
      {
          "no nulls composite passes",
          {"col0", "col1"},
          {{"a"}, {"b"}, {"c"}},
          {1, 2, 3},
          false,
      },
  };

  for (const auto& tc : testCases) {
    SCOPED_TRACE(tc.debugString());
    auto writer = createWriter(config(tc.columns), type_, pool_.get());
    auto batch = makeRowVector(
        {std::string(kCol0), std::string(kCol1), std::string(kCol2)},
        {makeNullableFlatVector<velox::StringView>(tc.col0),
         makeNullableFlatVector<int32_t>(tc.col1),
         makeFlatVector<velox::StringView>({"d", "e", "f"})});

    if (tc.expectThrow) {
      NIMBLE_ASSERT_USER_THROW(
          writer->write(batch), "Null value not allowed in index key column");
    } else {
      EXPECT_NO_THROW(writer->write(batch));
    }
  }
}

TEST_P(HashIndexWriterParamTest, writeAfterCloseThrows) {
  auto writer = createWriter(config(), type_, pool_.get());
  writer->write(makeInput({1}));

  MockSectionStore store;
  EXPECT_TRUE(writer->close(noopWriteDataFn(), store.createFn()).has_value());

  NIMBLE_ASSERT_THROW(
      writer->write(makeInput({2})), "IndexWriter has been closed");
}

TEST_P(HashIndexWriterParamTest, doubleCloseThrows) {
  auto writer = createWriter(config(), type_, pool_.get());
  writer->write(makeInput({1}));

  MockSectionStore store;
  EXPECT_TRUE(writer->close(noopWriteDataFn(), store.createFn()).has_value());

  NIMBLE_ASSERT_THROW(
      writer->close(noopWriteDataFn(), store.createFn()),
      "close() already called");
}

TEST_P(HashIndexWriterParamTest, rowCountOverflow) {
  auto writer = createWriter(config(), type_, pool_.get());
  ASSERT_NE(writer, nullptr);

  HashIndexWriterTestHelper helper(writer.get());
  helper.setNumRows(std::numeric_limits<uint32_t>::max() - 5);

  NIMBLE_ASSERT_USER_THROW(
      writer->write(makeInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})),
      "Hash index row count exceeds uint32 limit");
}

TEST_F(HashIndexWriterTest, withBloomFilter) {
  struct TestCase {
    std::string name;
    std::optional<float> bloomFilterBitsPerKey;
    bool expectBloomFilter;

    std::string debugString() const {
      return name;
    }
  };

  const std::vector<TestCase> testCases = {
      {"enabled", 10.0f, true},
      {"disabled", std::nullopt, false},
  };

  for (const auto& tc : testCases) {
    SCOPED_TRACE(tc.debugString());
    auto builder = HashIndexConfigBuilder{}.withKeyColumns({"col1"});
    if (tc.bloomFilterBitsPerKey.has_value()) {
      builder.withBloomFilter(tc.bloomFilterBitsPerKey.value());
    }
    auto writer = createWriter(builder.build(), type_, pool_.get());
    writer->write(makeInput({1, 2, 3, 4, 5}));

    MockSectionStore store;
    auto directory = closeAndReadDirectory(*writer, store);

    const auto* hashIndex = flatbuffers::GetRoot<serialization::HashIndex>(
        indexSectionContent(directory, /*index=*/0, store).data());
    ASSERT_NE(hashIndex, nullptr);
    if (tc.expectBloomFilter) {
      EXPECT_NE(hashIndex->bloom_filter(), nullptr);
    } else {
      EXPECT_EQ(hashIndex->bloom_filter(), nullptr);
    }
  }
}

TEST_F(HashIndexWriterTest, rejectsInvalidBloomFilter) {
  const auto invalidConfig = HashIndexConfigBuilder{}
                                 .withKeyColumns({"col1"})
                                 .withBloomFilter(0)
                                 .build();
  NIMBLE_ASSERT_THROW(
      createWriter(invalidConfig, type_, pool_.get()),
      "Bloom filter bits per key must be finite and positive");
}

TEST_F(HashIndexWriterTest, withPartitioning) {
  const auto indexConfig = HashIndexConfigBuilder{}
                               .withKeyColumns({"col1"})
                               .withMaxPartitionSizeBytes(32)
                               .build();
  auto writer = createWriter(indexConfig, type_, pool_.get());

  // Write enough rows to exceed partition size.
  std::vector<int32_t> values;
  values.reserve(100);
  for (int i = 0; i < 100; ++i) {
    values.push_back(i);
  }
  writer->write(makeInput(values));

  MockSectionStore store;
  auto directory = closeAndReadDirectory(*writer, store);

  validateDirectory(directory, {{"col1"}});
  // With 100 keys and 32-byte partition limit, expect multiple partitions.
  const auto* hashIndex = flatbuffers::GetRoot<serialization::HashIndex>(
      indexSectionContent(directory, /*index=*/0, store).data());
  ASSERT_NE(hashIndex, nullptr);
  ASSERT_NE(hashIndex->partition_sections(), nullptr);
  EXPECT_GT(hashIndex->partition_sections()->size(), 1u);
  validateIndex(indexSectionContent(directory, /*index=*/0, store), 100);
}

TEST_F(HashIndexWriterTest, loadFactorVariations) {
  struct TestParam {
    float loadFactor;

    std::string debugString() const {
      return fmt::format("loadFactor: {}", loadFactor);
    }
  };

  std::vector<TestParam> testSettings = {{0.1f}, {0.5f}, {0.7f}, {1.0f}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    const auto indexConfig = HashIndexConfigBuilder{}
                                 .withKeyColumns({"col1"})
                                 .withLoadFactor(testData.loadFactor)
                                 .build();
    auto writer = createWriter(indexConfig, type_, pool_.get());

    std::vector<int32_t> values;
    values.reserve(50);
    for (int i = 0; i < 50; ++i) {
      values.push_back(i);
    }
    writer->write(makeInput(values));

    MockSectionStore store;
    auto directory = closeAndReadDirectory(*writer, store);

    validateDirectory(directory, {{"col1"}});
    validateIndex(
        indexSectionContent(directory, /*index=*/0, store),
        50,
        /*expectedNumPartitions=*/1);
  }
}

TEST_P(HashIndexWriterParamTest, duplicateKeysAllowed) {
  auto writer = createWriter(config(), type_, pool_.get());

  writer->write(makeInput({1, 1, 2, 2, 3}));

  HashIndexWriterTestHelper helper(writer.get());
  EXPECT_EQ(helper.numRows(), 5);
  EXPECT_EQ(helper.numEntries(), 5);

  MockSectionStore store;
  auto directory = closeAndReadDirectory(*writer, store);

  validateDirectory(directory, {GetParam().columns});
  validateIndex(
      indexSectionContent(directory, /*index=*/0, store),
      5,
      /*expectedNumPartitions=*/1);
}

} // namespace
} // namespace facebook::nimble::index::test
