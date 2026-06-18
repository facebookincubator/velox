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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/VectorHasher.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace facebook::velox::exec::test {

class HashTableSerializationTest : public testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    tempDir_ = exec::test::TempDirectoryPath::create();
  }

  void TearDown() override {
    pool_.reset();
  }

  std::unique_ptr<HashTable<true>> createTestHashTable(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<TypePtr>& dependentTypes,
      bool allowDuplicates = false) {
    std::vector<std::unique_ptr<VectorHasher>> hashers;
    for (int i = 0; i < keyTypes.size(); ++i) {
      hashers.push_back(std::make_unique<VectorHasher>(keyTypes[i], i));
    }

    return std::make_unique<HashTable<true>>(
        std::move(hashers),
        std::vector<Accumulator>{},
        dependentTypes,
        allowDuplicates,
        true, // isJoinBuild
        false, // hasProbedFlag
        false, // hasCountFlag
        0, // minTableSizeForParallelJoinBuild
        pool_.get());
  }

  void insertData(
      HashTable<true>* table,
      const RowVectorPtr& data,
      const std::vector<column_index_t>& /*keyChannels*/) {
    SelectivityVector allRows(data->size());

    std::vector<char*> inserted(data->size());
    const auto nextOffset = table->rows()->nextOffset();
    for (int i = 0; i < data->size(); ++i) {
      inserted[i] = table->rows()->newRow();
      if (nextOffset > 0) {
        *reinterpret_cast<char**>(inserted[i] + nextOffset) = nullptr;
      }
    }

    for (int col = 0; col < data->childrenSize(); ++col) {
      DecodedVector decoded(*data->childAt(col), allRows);
      for (int row = 0; row < data->size(); ++row) {
        table->rows()->store(decoded, row, inserted[row], col);
      }
    }
  }

  void verifyHashTablesEqual(
      HashTable<true>* original,
      HashTable<true>* restored) {
    EXPECT_EQ(original->numDistinct(), restored->numDistinct());
    EXPECT_EQ(original->hashMode(), restored->hashMode());

    auto* origRows = original->rows();
    auto* restRows = restored->rows();

    ASSERT_EQ(origRows->numRows(), restRows->numRows());
    ASSERT_EQ(origRows->columnTypes().size(), restRows->columnTypes().size());

    std::vector<char*> origRowPtrs;
    std::vector<char*> restRowPtrs;

    RowContainerIterator origIter;
    RowContainerIterator restIter;

    std::vector<char*> buffer(1000);

    while (true) {
      auto numRows = origRows->listRows(
          &origIter, buffer.size(), RowContainer::kUnlimited, buffer.data());
      if (numRows == 0)
        break;
      origRowPtrs.insert(
          origRowPtrs.end(), buffer.begin(), buffer.begin() + numRows);
    }

    while (true) {
      auto numRows = restRows->listRows(
          &restIter, buffer.size(), RowContainer::kUnlimited, buffer.data());
      if (numRows == 0)
        break;
      restRowPtrs.insert(
          restRowPtrs.end(), buffer.begin(), buffer.begin() + numRows);
    }

    ASSERT_EQ(origRowPtrs.size(), restRowPtrs.size());

    auto encodeRows = [](RowContainer* rows,
                         const std::vector<char*>& rowPtrs) {
      std::vector<std::string> encodedRows;
      encodedRows.reserve(rowPtrs.size());
      for (auto* rowPtr : rowPtrs) {
        std::string encoded;
        for (int col = 0; col < rows->columnTypes().size(); ++col) {
          const auto column = rows->columnAt(col);
          const bool isNull = RowContainer::isNullAt(rowPtr, column);
          encoded += isNull ? "NULL:" : "VAL:";
          if (!isNull) {
            const auto& type = rows->columnTypes()[col];
            switch (type->kind()) {
              case TypeKind::BIGINT:
                encoded += std::to_string(*reinterpret_cast<const int64_t*>(
                    rowPtr + column.offset()));
                break;
              case TypeKind::INTEGER:
                encoded += std::to_string(*reinterpret_cast<const int32_t*>(
                    rowPtr + column.offset()));
                break;
              case TypeKind::DOUBLE:
                encoded += std::to_string(
                    *reinterpret_cast<const double*>(rowPtr + column.offset()));
                break;
              case TypeKind::BOOLEAN:
                encoded +=
                    *reinterpret_cast<const bool*>(rowPtr + column.offset())
                    ? "true"
                    : "false";
                break;
              case TypeKind::VARCHAR: {
                const auto* str = reinterpret_cast<const StringView*>(
                    rowPtr + column.offset());
                encoded.append(str->data(), str->size());
                break;
              }
              default:
                VELOX_FAIL(
                    "Unsupported type in HashTableSerializationTest comparison: {}",
                    type->toString());
            }
          }
          encoded += "|";
        }
        encodedRows.push_back(std::move(encoded));
      }
      std::sort(encodedRows.begin(), encodedRows.end());
      return encodedRows;
    };

    EXPECT_EQ(
        encodeRows(origRows, origRowPtrs), encodeRows(restRows, restRowPtrs));
  }

  void verifyJoinProbe(
      HashTable<true>* table,
      const RowVectorPtr& probe,
      int32_t expectedHits) {
    HashLookup lookup(table->hashers(), pool_.get());
    SelectivityVector rows(probe->size());
    rows.setAll();

    table->prepareForJoinProbe(lookup, probe, rows, true);
    table->joinProbe(lookup);

    int32_t hitCount = 0;
    for (int32_t row = 0; row < probe->size(); ++row) {
      if (lookup.hits[row] != nullptr) {
        ++hitCount;
      }
    }
    EXPECT_EQ(hitCount, expectedHits);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<TempDirectoryPath> tempDir_;
};

TEST_F(HashTableSerializationTest, BasicSerializationDefault) {
  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<std::string>({"a", "b", "c", "d", "e"})});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  std::stringstream ss;
  table->serialize(ss);

  auto restored = HashTable<true>::deserialize(ss, pool_.get());

  verifyHashTablesEqual(table.get(), restored.get());
}

TEST_F(HashTableSerializationTest, BasicSerializationJoinProbe) {
  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<std::string>({"a", "b", "c", "d", "e"})});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  std::stringstream ss;
  table->serialize(ss);

  auto restored = HashTable<true>::deserialize(ss, pool_.get());

  verifyHashTablesEqual(table.get(), restored.get());
  verifyJoinProbe(restored.get(), data, data->size());
}

TEST_F(HashTableSerializationTest, MultipleDataTypes) {
  auto table = createTestHashTable(
      {BIGINT(), INTEGER(), VARCHAR()}, {DOUBLE(), BOOLEAN()}, false);

  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int32_t>({10, 20, 30}),
       makeFlatVector<std::string>({"key1", "key2", "key3"}),
       makeFlatVector<double>({1.1, 2.2, 3.3}),
       makeFlatVector<bool>({true, false, true})});

  insertData(table.get(), data, {0, 1, 2});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  std::stringstream ss;
  table->serialize(ss);

  auto restored = HashTable<true>::deserialize(ss, pool_.get());

  verifyHashTablesEqual(table.get(), restored.get());
}

TEST_F(HashTableSerializationTest, NullValues) {
  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  auto data = makeRowVector(
      {makeNullableFlatVector<int64_t>({1, std::nullopt, 3, std::nullopt, 5}),
       makeNullableFlatVector<std::string>(
           {"a", std::nullopt, "c", "d", std::nullopt})});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  std::stringstream ss;
  table->serialize(ss);

  auto restored = HashTable<true>::deserialize(ss, pool_.get());

  verifyHashTablesEqual(table.get(), restored.get());
}

TEST_F(HashTableSerializationTest, LargeDataSet) {
  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  std::vector<int64_t> keys;
  std::vector<std::string> values;
  for (int i = 0; i < 10000; ++i) {
    keys.push_back(i);
    values.push_back("value_" + std::to_string(i));
  }

  auto data = makeRowVector({makeFlatVector(keys), makeFlatVector(values)});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  std::stringstream ss;
  table->serialize(ss);

  auto restored = HashTable<true>::deserialize(ss, pool_.get());

  verifyHashTablesEqual(table.get(), restored.get());
}

TEST_F(HashTableSerializationTest, LongStrings) {
  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  std::vector<std::string> values = {
      "short",
      "this is a very long string that exceeds 12 bytes",
      "medium",
      std::string(1000, 'x'),
      ""};

  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}), makeFlatVector(values)});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  std::stringstream ss;
  table->serialize(ss);

  auto restored = HashTable<true>::deserialize(ss, pool_.get());

  verifyHashTablesEqual(table.get(), restored.get());
}

TEST_F(HashTableSerializationTest, AllowDuplicates) {
  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, true);

  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 1, 2, 2, 3}),
       makeFlatVector<std::string>({"a1", "a2", "b1", "b2", "c"})});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  std::stringstream ss;
  table->serialize(ss);

  auto restored = HashTable<true>::deserialize(ss, pool_.get());

  verifyHashTablesEqual(table.get(), restored.get());
}

TEST_F(HashTableSerializationTest, CrossProcessSerialization) {
  std::string tempFile = tempDir_->getPath() + "/hashtable_cross_process.bin";

  pid_t pid = fork();

  if (pid == 0) {
    auto childPool = memory::memoryManager()->addLeafPool();
    auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

    auto data = makeRowVector(
        {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
         makeFlatVector<std::string>(
             {"child1", "child2", "child3", "child4", "child5"})});

    insertData(table.get(), data, {0});
    table->prepareJoinTable(
        {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

    std::ofstream out(tempFile, std::ios::binary);
    table->serialize(out);
    out.close();

    exit(0);
  } else {
    int status;
    waitpid(pid, &status, 0);
    ASSERT_EQ(WEXITSTATUS(status), 0) << "Child process failed";

    std::ifstream in(tempFile, std::ios::binary);
    ASSERT_TRUE(in.good()) << "Failed to open serialized file";

    auto restored = HashTable<true>::deserialize(in, pool_.get());
    in.close();

    EXPECT_EQ(restored->numDistinct(), 5);
    EXPECT_EQ(restored->rows()->numRows(), 5);

    std::remove(tempFile.c_str());
  }
}

TEST_F(HashTableSerializationTest, MultiProcessConcurrentSerialization) {
  const int numProcesses = 4;
  std::vector<std::string> tempFiles;

  for (int i = 0; i < numProcesses; ++i) {
    std::string tempFile = tempDir_->getPath() + "/hashtable_process_" +
        std::to_string(i) + ".bin";
    tempFiles.push_back(tempFile);

    pid_t pid = fork();

    if (pid == 0) {
      auto childPool = memory::memoryManager()->addLeafPool();
      auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

      int start = i * 1000;
      int end = start + 1000;

      std::vector<int64_t> keys;
      std::vector<std::string> values;
      for (int j = start; j < end; ++j) {
        keys.push_back(j);
        values.push_back(
            "process_" + std::to_string(i) + "_value_" + std::to_string(j));
      }

      auto data = makeRowVector({makeFlatVector(keys), makeFlatVector(values)});

      insertData(table.get(), data, {0});
      table->prepareJoinTable(
          {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

      std::ofstream out(tempFile, std::ios::binary);
      table->serialize(out);
      out.close();

      exit(0);
    }
  }

  for (int i = 0; i < numProcesses; ++i) {
    int status;
    wait(&status);
    ASSERT_EQ(WEXITSTATUS(status), 0) << "Child process " << i << " failed";
  }

  for (int i = 0; i < numProcesses; ++i) {
    std::ifstream in(tempFiles[i], std::ios::binary);
    ASSERT_TRUE(in.good()) << "Failed to open file from process " << i;

    auto restored = HashTable<true>::deserialize(in, pool_.get());
    in.close();

    EXPECT_EQ(restored->numDistinct(), 1000)
        << "Process " << i << " data mismatch";

    std::remove(tempFiles[i].c_str());
  }
}

TEST_F(HashTableSerializationTest, CrossProcessDataMerge) {
  const int numPartitions = 3;
  std::vector<std::string> partitionFiles;

  for (int i = 0; i < numPartitions; ++i) {
    std::string tempFile =
        tempDir_->getPath() + "/partition_" + std::to_string(i) + ".bin";
    partitionFiles.push_back(tempFile);

    pid_t pid = fork();

    if (pid == 0) {
      auto childPool = memory::memoryManager()->addLeafPool();
      auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

      std::vector<int64_t> keys;
      std::vector<std::string> values;

      for (int64_t key = 0; key < 1000; ++key) {
        if (key % numPartitions == i) {
          keys.push_back(key);
          values.push_back(
              "partition_" + std::to_string(i) + "_key_" + std::to_string(key));
        }
      }

      auto data = makeRowVector({makeFlatVector(keys), makeFlatVector(values)});

      insertData(table.get(), data, {0});
      table->prepareJoinTable(
          {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

      std::ofstream out(tempFile, std::ios::binary);
      table->serialize(out);
      out.close();

      exit(0);
    }
  }

  for (int i = 0; i < numPartitions; ++i) {
    int status;
    wait(&status);
    ASSERT_EQ(WEXITSTATUS(status), 0);
  }

  auto mergedTable = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  int totalRows = 0;
  for (int i = 0; i < numPartitions; ++i) {
    std::ifstream in(partitionFiles[i], std::ios::binary);
    ASSERT_TRUE(in.good());

    auto partition = HashTable<true>::deserialize(in, pool_.get());
    in.close();

    totalRows += partition->numDistinct();

    std::remove(partitionFiles[i].c_str());
  }

  EXPECT_EQ(totalRows, 1000) << "Merged data count mismatch";
}

TEST_F(HashTableSerializationTest, PerformanceBenchmark) {
  const int numRows = 100000;

  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  std::vector<int64_t> keys;
  std::vector<std::string> values;
  for (int i = 0; i < numRows; ++i) {
    keys.push_back(i);
    values.push_back("benchmark_value_" + std::to_string(i));
  }

  auto data = makeRowVector({makeFlatVector(keys), makeFlatVector(values)});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  {
    auto start = std::chrono::high_resolution_clock::now();

    std::stringstream ss;
    table->serialize(ss);

    auto serEnd = std::chrono::high_resolution_clock::now();

    auto restored = HashTable<true>::deserialize(ss, pool_.get());

    auto deserEnd = std::chrono::high_resolution_clock::now();

    auto serTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(serEnd - start)
            .count();
    auto deserTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(deserEnd - serEnd)
            .count();

    LOG(INFO) << "Default Serialization Performance (" << numRows << " rows):";
    LOG(INFO) << "  Serialization: " << serTime << " ms";
    LOG(INFO) << "  Deserialization: " << deserTime << " ms";
    LOG(INFO) << "  Total: " << (serTime + deserTime) << " ms";
  }
}

TEST_F(HashTableSerializationTest, InvalidMagicNumber) {
  std::stringstream ss;
  uint32_t invalidMagic = 0x12345678;
  ss.write(reinterpret_cast<const char*>(&invalidMagic), sizeof(invalidMagic));

  EXPECT_THROW(HashTable<true>::deserialize(ss, pool_.get()), VeloxException);
}

TEST_F(HashTableSerializationTest, UnsupportedVersion) {
  std::stringstream ss;
  uint32_t magic = 0x48415348;
  uint32_t version = 999;
  ss.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  ss.write(reinterpret_cast<const char*>(&version), sizeof(version));

  EXPECT_THROW(HashTable<true>::deserialize(ss, pool_.get()), VeloxException);
}

TEST_F(HashTableSerializationTest, CorruptedData) {
  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<std::string>({"a", "b", "c"})});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  std::stringstream ss;
  table->serialize(ss);

  std::string serialized = ss.str();
  ASSERT_GT(serialized.size(), 8);
  serialized.resize(serialized.size() - 8);

  std::stringstream corruptedSs(serialized);

  EXPECT_THROW(
      HashTable<true>::deserialize(corruptedSs, pool_.get()), VeloxException);
}

} // namespace facebook::velox::exec::test
