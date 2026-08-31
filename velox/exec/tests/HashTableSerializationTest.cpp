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
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace facebook::velox::exec::test {

namespace {
std::string readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) {
    return "";
  }
  in.seekg(0, std::ios::end);
  const auto end = in.tellg();
  if (end <= 0) {
    return "";
  }
  const auto size = static_cast<size_t>(end);
  in.seekg(0, std::ios::beg);
  std::string data;
  data.resize(size);
  in.read(data.data(), data.size());
  return data;
}

void writeFile(const std::string& path, const std::string& data) {
  std::ofstream out(path, std::ios::binary);
  out.write(data.data(), data.size());
}
} // namespace

class HashTableSerializationTest : public testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::string serialize(const HashTable<true>& table) const {
    std::string data;
    data.resize(table.serializedSize());
    table.serializeTo(data.data(), data.size());
    return data;
  }

  std::unique_ptr<HashTable<true>> deserialize(
      const std::string& data,
      memory::MemoryPool* pool) const {
    return HashTable<true>::deserializeFrom(data.data(), data.size(), pool);
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

  // Inserts 'data' and also feeds the key columns to the VectorHashers the way
  // a real join build does, so that prepareJoinTable() can settle on a value id
  // based hash mode instead of falling back to kHash.
  void insertDataWithValueIds(
      HashTable<true>* table,
      const RowVectorPtr& data) {
    insertData(table, data, {});

    const SelectivityVector allRows(data->size());
    raw_vector<uint64_t> valueIds(data->size());
    for (const auto& hasher : table->hashers()) {
      hasher->decode(*data->childAt(hasher->channel()), allRows);
      // A false return only means the ids do not fit the hasher's current
      // range; prepareJoinTable() settles the mode from the observed values.
      hasher->computeValueIds(allRows, valueIds);
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

    // Extract through RowContainer so that strings spread over several
    // HashStringAllocator blocks are read correctly, then compare the rows as
    // sets since the two containers enumerate them in different orders.
    auto encodeRows =
        [this](RowContainer* rows, const std::vector<char*>& rowPtrs) {
          std::vector<VectorPtr> columns;
          for (auto col = 0; col < rows->columnTypes().size(); ++col) {
            auto column = BaseVector::create(
                rows->columnTypes()[col], rowPtrs.size(), pool_.get());
            RowContainer::extractColumn(
                rowPtrs.data(),
                rowPtrs.size(),
                rows->columnAt(col),
                /*columnHasNulls=*/true,
                column);
            columns.push_back(std::move(column));
          }

          auto rowVector = makeRowVector(columns);
          std::vector<std::string> encodedRows;
          encodedRows.reserve(rowPtrs.size());
          for (vector_size_t i = 0; i < rowVector->size(); ++i) {
            encodedRows.push_back(rowVector->toString(i));
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

  const auto serialized = serialize(*table);
  auto restored = deserialize(serialized, pool_.get());

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

  const auto serialized = serialize(*table);
  auto restored = deserialize(serialized, pool_.get());

  verifyHashTablesEqual(table.get(), restored.get());
  verifyJoinProbe(restored.get(), data, data->size());
}

TEST_F(HashTableSerializationTest, arrayHashMode) {
  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<std::string>({"a", "b", "c", "d", "e"})});

  insertDataWithValueIds(table.get(), data);
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);
  ASSERT_EQ(table->hashMode(), BaseHashTable::HashMode::kArray);

  const auto serialized = serialize(*table);
  auto restored = deserialize(serialized, pool_.get());

  ASSERT_EQ(restored->hashMode(), BaseHashTable::HashMode::kArray);
  verifyHashTablesEqual(table.get(), restored.get());
  verifyJoinProbe(restored.get(), data, data->size());
}

TEST_F(HashTableSerializationTest, normalizedKeyHashMode) {
  auto table = createTestHashTable({BIGINT(), BIGINT()}, {VARCHAR()}, false);

  // Two wide key ranges do not fit an array, but their concatenation fits in 64
  // bits, which is what kNormalizedKey requires.
  const vector_size_t numRows = 1'000;
  auto data = makeRowVector(
      {makeFlatVector<int64_t>(
           numRows, [](auto row) { return row * 1'000'000; }),
       makeFlatVector<int64_t>(
           numRows, [](auto row) { return row * 2'000'000; }),
       makeFlatVector<std::string>(
           numRows, [](auto row) { return fmt::format("value_{}", row); })});

  insertDataWithValueIds(table.get(), data);
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);
  ASSERT_EQ(table->hashMode(), BaseHashTable::HashMode::kNormalizedKey);

  const auto serialized = serialize(*table);
  auto restored = deserialize(serialized, pool_.get());

  ASSERT_EQ(restored->hashMode(), BaseHashTable::HashMode::kNormalizedKey);
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

  const auto serialized = serialize(*table);
  auto restored = deserialize(serialized, pool_.get());

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

  const auto serialized = serialize(*table);
  auto restored = deserialize(serialized, pool_.get());

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

  const auto serialized = serialize(*table);
  auto restored = deserialize(serialized, pool_.get());

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

  const auto serialized = serialize(*table);
  auto restored = deserialize(serialized, pool_.get());

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

  const auto serialized = serialize(*table);
  auto restored = deserialize(serialized, pool_.get());

  verifyHashTablesEqual(table.get(), restored.get());
}

TEST_F(HashTableSerializationTest, CrossProcessSerialization) {
  std::string tempFile = tempDir_->getPath() + "/hashtable_cross_process.bin";

  pid_t pid = fork();

  if (pid == 0) {
    auto childPool = memory::memoryManager()->addLeafPool();
    pool_ = childPool;
    auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

    auto data = makeRowVector(
        {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
         makeFlatVector<std::string>(
             {"child1", "child2", "child3", "child4", "child5"})});

    insertData(table.get(), data, {0});
    table->prepareJoinTable(
        {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

    const auto serialized = serialize(*table);
    writeFile(tempFile, serialized);

    exit(0);
  } else {
    int status;
    waitpid(pid, &status, 0);
    ASSERT_EQ(WEXITSTATUS(status), 0) << "Child process failed";

    const auto serialized = readFile(tempFile);
    ASSERT_GT(serialized.size(), 0) << "Failed to read serialized file";
    auto restored = deserialize(serialized, pool_.get());

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
      pool_ = childPool;
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

      const auto serialized = serialize(*table);
      writeFile(tempFile, serialized);

      exit(0);
    }
  }

  for (int i = 0; i < numProcesses; ++i) {
    int status;
    wait(&status);
    ASSERT_EQ(WEXITSTATUS(status), 0) << "Child process " << i << " failed";
  }

  for (int i = 0; i < numProcesses; ++i) {
    const auto serialized = readFile(tempFiles[i]);
    ASSERT_GT(serialized.size(), 0) << "Failed to read file from process " << i;
    auto restored = deserialize(serialized, pool_.get());

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
      pool_ = childPool;
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

      const auto serialized = serialize(*table);
      writeFile(tempFile, serialized);

      exit(0);
    }
  }

  for (int i = 0; i < numPartitions; ++i) {
    int status;
    wait(&status);
    ASSERT_EQ(WEXITSTATUS(status), 0);
  }

  int totalRows = 0;
  for (int i = 0; i < numPartitions; ++i) {
    const auto serialized = readFile(partitionFiles[i]);
    ASSERT_GT(serialized.size(), 0) << "Failed to read partition file " << i;
    auto partition = deserialize(serialized, pool_.get());

    totalRows += partition->numDistinct();

    std::remove(partitionFiles[i].c_str());
  }

  EXPECT_EQ(totalRows, 1000) << "Merged data count mismatch";
}

TEST_F(HashTableSerializationTest, PerformanceBenchmark) {
  const int32_t numRows = 100'000;

  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  auto data = makeRowVector(
      {makeFlatVector<int64_t>(numRows, [](auto row) { return row; }),
       makeFlatVector<std::string>(numRows, [](auto row) {
         return fmt::format("benchmark_value_{}", row);
       })});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  using Clock = std::chrono::steady_clock;
  auto micros = [](Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
        .count();
  };

  const auto beforeSize = Clock::now();
  std::string serialized;
  serialized.resize(table->serializedSize());
  const auto beforeSerialize = Clock::now();
  table->serializeTo(serialized.data(), serialized.size());
  const auto beforeDeserialize = Clock::now();
  auto restored = deserialize(serialized, pool_.get());
  const auto end = Clock::now();

  LOG(INFO) << "Serialization performance (" << numRows << " rows, "
            << serialized.size() << " bytes):";
  LOG(INFO) << "  serializedSize: " << micros(beforeSize, beforeSerialize)
            << " us";
  LOG(INFO) << "  serializeTo: " << micros(beforeSerialize, beforeDeserialize)
            << " us";
  LOG(INFO) << "  deserializeFrom: " << micros(beforeDeserialize, end) << " us";

  verifyHashTablesEqual(table.get(), restored.get());
}

TEST_F(HashTableSerializationTest, InvalidMagicNumber) {
  uint32_t invalidMagic = 0x12345678;
  std::string data(sizeof(invalidMagic), '\0');
  std::memcpy(data.data(), &invalidMagic, sizeof(invalidMagic));
  EXPECT_THROW(deserialize(data, pool_.get()), VeloxException);
}

TEST_F(HashTableSerializationTest, UnsupportedVersion) {
  uint32_t magic = 0x48415348;
  uint32_t version = 999;
  std::string data(sizeof(magic) + sizeof(version), '\0');
  std::memcpy(data.data(), &magic, sizeof(magic));
  std::memcpy(data.data() + sizeof(magic), &version, sizeof(version));
  EXPECT_THROW(deserialize(data, pool_.get()), VeloxException);
}

TEST_F(HashTableSerializationTest, CorruptedData) {
  auto table = createTestHashTable({BIGINT()}, {VARCHAR()}, false);

  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<std::string>({"a", "b", "c"})});

  insertData(table.get(), data, {0});
  table->prepareJoinTable(
      {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);

  auto serialized = serialize(*table);
  ASSERT_GT(serialized.size(), 8);
  serialized.resize(serialized.size() - 8);
  EXPECT_THROW(deserialize(serialized, pool_.get()), VeloxException);
}

} // namespace facebook::velox::exec::test
