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

#include "velox/exec/OptHashTable.h"
#include "folly/experimental/EventCount.h"
#include "velox/common/base/SelectivityInfo.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/VectorHasher.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <memory>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace facebook::velox::exec::test {

template <bool ignoreNullKeys>
class OptHashTableTestHelper {
 public:
  static OptHashTableTestHelper create(HashTable<ignoreNullKeys>* table) {
    return HashTableTestHelper(table);
  }

  int64_t nextBucketOffset(int64_t offset) const {
    return table_->nextBucketOffset(offset);
  }

  uint64_t bucketSize() const {
    return HashTable<ignoreNullKeys>::kBucketSize;
  }

  void allocateTables(uint64_t size) {
    table_->allocateTables(size);
  }

  size_t tableSlotSize() const {
    return table_->tableSlotSize();
  }

  void insertForJoin(
      RowContainer* rows,
      char** groups,
      uint64_t* hashes,
      int32_t numGroups,
      TableInsertPartitionInfo* partitionInfo) {
    table_->insertForJoin(rows, groups, hashes, numGroups, partitionInfo);
  }

  void setHashMode(BaseHashTable::HashMode mode, int32_t numNew) {
    table_->setHashMode(mode, numNew);
  }

 private:
  explicit OptHashTableTestHelper(HashTable<ignoreNullKeys>* table)
      : table_(table) {
    VELOX_CHECK_NOT_NULL(table_);
  }

  HashTable<ignoreNullKeys>* const table_;
};

// Test framework for join hash tables. Generates probe keys, of which
// some percent are inserted in a hashTable. The placement of the
// payload is shuffled so as not to correlate with the probe
// order. Tests the presence/correctness of the hit for each key and
// measures the time for computing hashes/value ids vs the time spent
// probing the table. Covers kArray, kNormalizedKey and kHash hash
// modes.
class OptHashTableTest : public testing::TestWithParam<bool>,
                      public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    common::testutil::TestValue::enable();
    if (GetParam()) {
      executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(16);
    }
    aggregate::prestosql::registerAllAggregateFunctions();
  }

  void testCycle(
      BaseHashTable::HashMode mode,
      int32_t size,
      int32_t numWays,
      TypePtr buildType,
      int32_t numKeys) {
    std::vector<TypePtr> dependentTypes;
    int32_t sequence = 0;
    isInTable_.resize(
        bits::nwords(numWays * size),
        static_cast<const std::vector<
            unsigned long,
            std::allocator<unsigned long>>::value_type>(-1));
    if (insertPct_ != 100) {
      // If we probe with all keys but only mean to insert part, we deselect.
      folly::Random::DefaultGenerator rng;
      rng.seed(1);
      for (auto i = 0; i < size * numWays; ++i) {
        if (folly::Random::rand32(rng) % 100 > insertPct_) {
          bits::clearBit(isInTable_.data(), i);
        }
      }
    }
    int32_t startOffset = 0;
    std::vector<std::unique_ptr<BaseHashTable>> otherTables;
    uint64_t numRows{0};
    for (auto way = 0; way < numWays; ++way) {
      std::vector<RowVectorPtr> batches;
      std::vector<std::unique_ptr<VectorHasher>> keyHashers;
      for (auto channel = 0; channel < numKeys; ++channel) {
        keyHashers.emplace_back(std::make_unique<VectorHasher>(
            buildType->childAt(channel), channel));
      }
      auto table = OptHashTable<true>::createForJoin(
          std::move(keyHashers), dependentTypes, true, false, 1'000, pool());

      makeRows(size, 1, sequence, buildType, batches);
      copyVectorsToTable(batches, startOffset, table.get());
      sequence += size;
      if (topTable_ == nullptr) {
        topTable_ = std::move(table);
        numRows += topTable_->rows()->numRows();
      } else {
        numRows += table->rows()->numRows();
        otherTables.push_back(std::move(table));
      }
      batches_.insert(batches_.end(), batches.begin(), batches.end());
      startOffset += size;
    }

    const uint64_t estimatedTableSize =
        topTable_->estimateHashTableSize(numRows);
    const uint64_t usedMemoryBytes = topTable_->rows()->pool()->usedBytes();
    topTable_->prepareJoinTable(std::move(otherTables), executor_.get());
    ASSERT_GE(
        estimatedTableSize,
        topTable_->rows()->pool()->usedBytes() - usedMemoryBytes);
    ASSERT_EQ(topTable_->hashMode(), mode);
    ASSERT_EQ(topTable_->allRows().size(), numWays);
    uint64_t rowCount{0};
    for (auto* rowContainer : topTable_->allRows()) {
      rowCount += rowContainer->numRows();
    }
    ASSERT_EQ(rowCount, numRows);

    LOG(INFO) << "Made table " << describeTable();
    testProbe();
    /*testEraseEveryN(3);
    testProbe();
    testEraseEveryN(4);
    testProbe();*/
    const auto memoryUsage = pool()->usedBytes();
    topTable_->clear(true);
    for (const auto* rowContainer : topTable_->allRows()) {
      ASSERT_EQ(rowContainer->numRows(), 0);
    }
    ASSERT_EQ(topTable_->numDistinct(), 0);
    ASSERT_LT(pool()->usedBytes(), memoryUsage);
  }

  void insertGroups(
      const RowVector& input,
      HashLookup& lookup,
      HashTable<false>& table) {
    const SelectivityVector rows(input.size());
    insertGroups(input, rows, lookup, table);
  }

  void insertGroups(
      const RowVector& input,
      const SelectivityVector& rows,
      HashLookup& lookup,
      HashTable<false>& table) {
    lookup.reset(rows.end());
    lookup.rows.clear();
    rows.applyToSelected([&](auto row) { lookup.rows.push_back(row); });

    auto& hashers = table.hashers();
    auto mode = table.hashMode();
    bool rehash = false;
    for (int32_t i = 0; i < hashers.size(); ++i) {
      auto key = input.childAt(hashers[i]->channel());
      hashers[i]->decode(*key, rows);
      if (mode != BaseHashTable::HashMode::kHash) {
        if (!hashers[i]->computeValueIds(rows, lookup.hashes)) {
          rehash = true;
        }
      } else {
        hashers[i]->hash(rows, i > 0, lookup.hashes);
      }
    }

    if (rehash) {
      if (table.hashMode() != BaseHashTable::HashMode::kHash) {
        table.decideHashMode(input.size());
      }
      insertGroups(input, rows, lookup, table);
      return;
    }
    table.groupProbe(lookup);
  }

  std::string describeTable() {
    std::stringstream out;
    auto mode = topTable_->hashMode();
    if (mode == BaseHashTable::HashMode::kHash) {
      out << "Multipart key ";
    } else {
      out
          << (mode == BaseHashTable::HashMode::kArray ? "Array "
                                                      : "Normalized key ");
      out << "(";
      for (auto& hasher : topTable_->hashers()) {
        out << (hasher->isRange() ? "range " : "valueIds ");
      }
      out << ") ";
    }
    out << topTable_->numDistinct() << " entries";
    return out.str();
  }

  void copyVectorsToTable(
      const std::vector<RowVectorPtr>& batches,
      int32_t tableOffset,
      BaseHashTable* table) {
    const int32_t batchSize = batches[0]->size();
    raw_vector<uint64_t> dummy(batchSize);
    int32_t batchOffset = 0;
    rowOfKey_.resize(tableOffset + batchSize * batches.size());
    auto rowContainer = table->rows();
    auto& hashers = table->hashers();
    auto numKeys = hashers.size();
    // We init a DecodedVector for each member of the RowVectors in 'batches'.
    std::vector<std::vector<DecodedVector>> decoded;
    SelectivityVector rows(batchSize);
    SelectivityVector insertedRows(batchSize);
    for (auto& batch : batches) {
      // If we are only inserting a fraction of the rows, we set insertedRows to
      // that fraction so that the VectorHashers only see keys that will
      // actually be inserted.
      if (insertPct_ < 100) {
        bits::copyBits(
            isInTable_.data(),
            tableOffset + batchOffset,
            insertedRows.asMutableRange().bits(),
            0,
            batchSize);
        insertedRows.updateBounds();
      }
      decoded.emplace_back(batch->childrenSize());
      VELOX_CHECK_EQ(batch->size(), batchSize);
      auto& decoders = decoded.back();
      for (auto i = 0; i < batch->childrenSize(); ++i) {
        decoders[i].decode(*batch->childAt(i), rows);
        if (i < numKeys) {
          auto hasher = table->hashers()[i].get();
          hasher->decode(*batch->childAt(i), insertedRows);
          if (table->hashMode() != BaseHashTable::HashMode::kHash &&
              hasher->mayUseValueIds()) {
            hasher->computeValueIds(insertedRows, dummy);
          }
        }
      }
      batchOffset += batchSize;
    }

    const auto size = batchSize * batches.size();
    const auto powerOfTwo = bits::nextPowerOfTwo(size);
    const int32_t mask = powerOfTwo - 1;
    int32_t position = 0;
    int32_t delta = 1;
    const auto nextOffset = rowContainer->nextOffset();

    // We insert values in a geometric skip order. 1, 2, 4, 7,
    // 11,... where the skip increments by one. We wrap around at the
    // power of two boundary. This sequence hits every place in the
    // power of two range once. Like this, when we probe the data for
    // consecutive keys the hits will have no cache locality.
    for (auto count = 0; count < powerOfTwo; ++count) {
      if (position < size &&
          (insertPct_ == 100 ||
           bits::isBitSet(isInTable_.data(), tableOffset + position))) {
        char* newRow = rowContainer->newRow();
        rowOfKey_[tableOffset + position] = newRow;
        const auto batchIndex = position / batchSize;
        const auto rowIndex = position % batchSize;
        if (nextOffset > 0) {
          *reinterpret_cast<char**>(newRow + nextOffset) = nullptr;
        }
        for (auto i = 0; i < batches[batchIndex]->type()->size(); ++i) {
          rowContainer->store(decoded[batchIndex][i], rowIndex, newRow, i);
        }
      }
      position = (position + delta) & mask;
      ++delta;
    }
  }

  // Makes a vector of 'type' with 'size' unique elements, initialized
  // based on 'sequence'. If 'sequence' is incremented by 'size'
  // between the next call will not overlap with the results of the
  // previous one.
  VectorPtr makeVector(TypePtr type, int32_t size, int32_t sequence) {
    switch (type->kind()) {
      case TypeKind::BIGINT:
        return makeFlatVector<int64_t>(
            size,
            [&](vector_size_t row) { return keySpacing_ * (sequence + row); },
            nullptr);

      case TypeKind::VARCHAR: {
        auto strings =
            BaseVector::create<FlatVector<StringView>>(VARCHAR(), size, pool());
        for (auto row = 0; row < size; ++row) {
          auto string = fmt::format("{}", keySpacing_ * (sequence + row));
          // Make strings that overflow the inline limit for 1/10 of
          // the values after 10K,000. Datasets with only
          // range-encodable small strings can be made within the
          // first 10K values.
          if (row > 10000 && row % 10 == 0) {
            string += "----" + string + "----" + string;
          }
          strings->set(row, StringView(string));
        }
        return strings;
      }

      case TypeKind::ROW: {
        std::vector<VectorPtr> children;
        for (auto i = 0; i < type->size(); ++i) {
          children.push_back(makeVector(type->childAt(i), size, sequence));
        }
        return makeRowVector(children);
      }
      default:
        VELOX_FAIL("Unsupported kind for makeVector {}", type->kind());
    }
  }

  void makeRows(
      int32_t batchSize,
      int32_t numBatches,
      int32_t sequence,
      TypePtr buildType,
      std::vector<RowVectorPtr>& batches) {
    for (auto i = 0; i < numBatches; ++i) {
      batches.push_back(std::static_pointer_cast<RowVector>(
          makeVector(buildType, batchSize, sequence)));
      sequence += batchSize;
    }
  }

  void store(RowContainer& rowContainer, const RowVectorPtr& data) {
    std::vector<DecodedVector> decodedVectors;
    for (auto& vector : data->children()) {
      decodedVectors.emplace_back(*vector);
    }

    std::vector<char*> rows;
    for (auto i = 0; i < data->size(); ++i) {
      auto* row = rowContainer.newRow();

      for (auto j = 0; j < decodedVectors.size(); ++j) {
        rowContainer.store(decodedVectors[j], i, row, j);
      }
    }
  }

  void testProbe() {
    auto lookup = std::make_unique<HashLookup>(topTable_->hashers());
    const auto batchSize = batches_[0]->size();
    SelectivityVector rows(batchSize);
    const auto mode = topTable_->hashMode();
    SelectivityInfo hashTime;
    SelectivityInfo probeTime;
    auto& hashers = topTable_->hashers();
    VectorHasher::ScratchMemory scratchMemory;
    for (auto batchIndex = 0; batchIndex < batches_.size(); ++batchIndex) {
      const auto& batch = batches_[batchIndex];
      lookup->reset(batch->size());
      rows.setAll();
      {
        SelectivityTimer timer(hashTime, 0);
        for (auto i = 0; i < hashers.size(); ++i) {
          auto& key = batch->childAt(i);
          if (mode != BaseHashTable::HashMode::kHash) {
            hashers[i]->lookupValueIds(
                *key, rows, scratchMemory, lookup->hashes);
          } else {
            hashers[i]->decode(*key, rows);
            hashers[i]->hash(rows, i > 0, lookup->hashes);
          }
        }
      }

      lookup->rows.clear();
      if (rows.isAllSelected()) {
        lookup->rows.resize(rows.size());
        std::iota(lookup->rows.begin(), lookup->rows.end(), 0);
      } else {
        constexpr int32_t kPadding = simd::kPadding / sizeof(int32_t);
        lookup->rows.resize(bits::roundUp(rows.size() + kPadding, kPadding));
        const auto numRows = simd::indicesOfSetBits(
            rows.asRange().bits(), 0, batch->size(), lookup->rows.data());
        lookup->rows.resize(numRows);
      }

      const auto startOffset = batchIndex * batchSize;
      if (lookup->rows.empty()) {
        // the keys disqualify all entries. The table is not consulted.
        for (auto i = startOffset; i < startOffset + batch->size(); ++i) {
          ASSERT_EQ(nullptr, rowOfKey_[i]);
        }
      } else {
        {
          SelectivityTimer timer(probeTime, 0);
          topTable_->joinProbe(*lookup);
        }
        for (auto i = 0; i < lookup->rows.size(); ++i) {
          const auto key = lookup->rows[i];
          ASSERT_EQ(rowOfKey_[startOffset + key], lookup->hits[key]);
        }
      }
    }
  }

  // Erases every strideth non-erased item in the hash table.
  void testEraseEveryN(int32_t stride) {
    std::vector<char*> toErase;
    int32_t counter = 0;
    for (auto i = 0; i < rowOfKey_.size(); ++i) {
      if (rowOfKey_[i] && ++counter % stride == 0) {
        toErase.push_back(rowOfKey_[i]);
        rowOfKey_[i] = nullptr;
      }
    }
    topTable_->erase(folly::Range<char**>(toErase.data(), toErase.size()));
  }

  void testListNullKeyRows(
      const VectorPtr& keys,
      BaseHashTable::HashMode mode) {
    folly::F14FastSet<int> nullValues;
    for (int i = 0; i < keys->size(); ++i) {
      if (i % 97 == 0) {
        keys->setNull(i, true);
        nullValues.insert(i);
      }
    }
    auto batch = makeRowVector(
        {keys, makeFlatVector<int64_t>(keys->size(), folly::identity)});
    std::vector<std::unique_ptr<VectorHasher>> hashers;
    hashers.push_back(std::make_unique<VectorHasher>(keys->type(), 0));
    auto table = OptHashTable<false>::createForJoin(
        std::move(hashers), {BIGINT()}, true, false, 1'000, pool());
    copyVectorsToTable({batch}, 0, table.get());
    table->prepareJoinTable({}, executor_.get());
    ASSERT_EQ(table->hashMode(), mode);
    std::vector<char*> rows(nullValues.size());
    BaseHashTable::NullKeyRowsIterator iter;
    auto numRows = table->listNullKeyRows(&iter, rows.size(), rows.data());
    ASSERT_EQ(numRows, nullValues.size());
    auto actual =
        BaseVector::create<FlatVector<int64_t>>(BIGINT(), numRows, pool());
    table->rows()->extractColumn(rows.data(), numRows, 1, actual);
    for (int i = 0; i < actual->size(); ++i) {
      auto it = nullValues.find(actual->valueAt(i));
      ASSERT_TRUE(it != nullValues.end());
      nullValues.erase(it);
    }
    ASSERT_TRUE(nullValues.empty());
    ASSERT_EQ(0, table->listNullKeyRows(&iter, rows.size(), rows.data()));
  }

  // Bitmap of positions in batches_ that end up in the table.
  std::vector<uint64_t> isInTable_;
  // Test payload, keys first.
  std::vector<RowVectorPtr> batches_;

  // Corresponds 1:1 to data in 'batches_'. nullptr if the key is not
  // inserted, otherwise pointer into the RowContainer.
  std::vector<char*> rowOfKey_;
  std::unique_ptr<OptHashTable<true>> topTable_;
  // Percentage of keys inserted into the table. This is for measuring
  // joins that miss the table part of the time. Used in initializing
  // 'isInTable_'.
  int32_t insertPct_ = 100;
  // Spacing between consecutive generated keys. Affects whether
  // Vectorhashers make ranges or ids of distinct values.
  int64_t keySpacing_ = 1;
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
};

VELOX_INSTANTIATE_TEST_SUITE_P(
    OptHashTableTes, OptHashTableTest,
        testing::Values(true, false));

TEST_P(OptHashTableTest, int2SparseNormalized2) {
  auto type = ROW({"k1", "k2"}, {BIGINT(), BIGINT()});
  keySpacing_ = 1000;
  testCycle(BaseHashTable::HashMode::kNormalizedKey, 10000, 2, type, 2);
}

TEST_P(OptHashTableTest, int2SparseNormalizedMostMiss2) {
  auto type = ROW({"k1", "k2"}, {BIGINT(), BIGINT()});
  keySpacing_ = 1000;
  insertPct_ = 10;
  testCycle(BaseHashTable::HashMode::kNormalizedKey, 100000, 2, type, 2);
}


TEST_P(OptHashTableTest, string2Normalized) {
  auto type = ROW({"k1", "k2"}, {VARCHAR(), VARCHAR()});
  testCycle(BaseHashTable::HashMode::kNormalizedKey, 5000, 19, type, 2);
}


TEST_P(OptHashTableTest, int2SparseNormalized) {
  auto type = ROW({"k1", "k2"}, {BIGINT(), BIGINT()});
  keySpacing_ = 1000;
  testCycle(BaseHashTable::HashMode::kNormalizedKey, 10000, 2, type, 2);
}

TEST_P(OptHashTableTest, int2SparseNormalizedMostMiss) {
  auto type = ROW({"k1", "k2"}, {BIGINT(), BIGINT()});
  keySpacing_ = 1000;
  insertPct_ = 10;
  testCycle(BaseHashTable::HashMode::kNormalizedKey, 100000, 2, type, 2);
}



TEST_P(OptHashTableTest, regularHashingTableSize) {
  keySpacing_ = 1000;
  auto checkTableSize = [&](BaseHashTable::HashMode mode,
                            const RowTypePtr& type) {
    std::vector<std::unique_ptr<VectorHasher>> keyHashers;
    for (auto channel = 0; channel < type->size(); ++channel) {
      keyHashers.emplace_back(
          std::make_unique<VectorHasher>(type->childAt(channel), channel));
    }
    auto table = HashTable<true>::createForJoin(
        std::move(keyHashers), {}, true, false, 1'000, pool());
    std::vector<RowVectorPtr> batches;
    makeRows(1 << 12, 1, 0, type, batches);
    copyVectorsToTable(batches, 0, table.get());
    table->prepareJoinTable({}, executor_.get());
    ASSERT_EQ(table->hashMode(), mode);
    EXPECT_GE(table->rehashSize(), table->numDistinct());
  };
  {
    auto type = ROW({"k1", "k2"}, {BIGINT(), BIGINT()});
    checkTableSize(BaseHashTable::HashMode::kNormalizedKey, type);
  }
}

TEST(OptHashTableTest, modeString) {
  ASSERT_EQ("HASH", BaseHashTable::modeString(BaseHashTable::HashMode::kHash));
  ASSERT_EQ(
      "NORMALIZED_KEY",
      BaseHashTable::modeString(BaseHashTable::HashMode::kNormalizedKey));
  ASSERT_EQ(
      "ARRAY", BaseHashTable::modeString(BaseHashTable::HashMode::kArray));
  ASSERT_EQ(
      "Unknown HashTable mode:100",
      BaseHashTable::modeString(static_cast<BaseHashTable::HashMode>(100)));
}


VELOX_INSTANTIATE_TEST_SUITE_P(
    HashTableTests,
    OptHashTableTest,
    testing::Values(true, false));


TEST_P(OptHashTableTest, toStringSingleKey) {
  std::vector<std::unique_ptr<VectorHasher>> hashers;
  hashers.push_back(std::make_unique<VectorHasher>(BIGINT(), 0));

  auto table = HashTable<false>::createForJoin(
      std::move(hashers),
      {}, /*dependentTypes*/
      true /*allowDuplicates*/,
      false /*hasProbedFlag*/,
      1 /*minTableSizeForParallelJoinBuild*/,
      pool());

  auto data = makeRowVector({
      makeFlatVector<int64_t>(1'000, [](auto row) { return row / 2; }),
  });

  store(*table->rows(), data);

  table->prepareJoinTable({});

  ASSERT_NO_THROW(table->toString());
  ASSERT_NO_THROW(table->toString(0));
  ASSERT_NO_THROW(table->toString(10));
  ASSERT_NO_THROW(table->toString(1000));
  ASSERT_NO_THROW(table->toString(31, 5));
}

TEST_P(OptHashTableTest, toStringMultipleKeys) {
  std::vector<std::unique_ptr<VectorHasher>> hashers;
  hashers.push_back(std::make_unique<VectorHasher>(BIGINT(), 0));
  hashers.push_back(std::make_unique<VectorHasher>(VARCHAR(), 1));

  auto table = HashTable<false>::createForJoin(
      std::move(hashers),
      {}, /*dependentTypes*/
      true /*allowDuplicates*/,
      false /*hasProbedFlag*/,
      1 /*minTableSizeForParallelJoinBuild*/,
      pool());

  vector_size_t size = 1'000;
  auto data = makeRowVector({
      makeFlatVector<int64_t>(size, [](auto row) { return row / 2; }),
      makeFlatVector<std::string>(
          size, [](auto row) { return std::string(row, 'x'); }),
  });

  store(*table->rows(), data);

  table->prepareJoinTable({});

  ASSERT_NO_THROW(table->toString());
}

TEST(OptHashTableTest, tableInsertPartitionInfo) {
  std::vector<char*> overflows;
  const auto testFn = [&](PartitionBoundIndexType start,
                          PartitionBoundIndexType end) {
    TableInsertPartitionInfo info{start, end, overflows};
  };
  struct {
    PartitionBoundIndexType start;
    PartitionBoundIndexType end;

    std::string debugString() const {
      return fmt::format("start {}, end {}", start, end);
    }
  } badSettings[] = {
      {0, 0}, {-2, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {32, 1}, {32, 32}};
  for (const auto& badData : badSettings) {
    SCOPED_TRACE(badData.debugString());
    VELOX_ASSERT_THROW(testFn(badData.start, badData.end), "");
  }
  ASSERT_TRUE(overflows.empty());

  TableInsertPartitionInfo info{1, 1000, overflows};
  ASSERT_TRUE(info.inRange(1));
  ASSERT_FALSE(info.inRange(0));
  ASSERT_FALSE(info.inRange(-1));
  ASSERT_TRUE(info.inRange(999));
  ASSERT_FALSE(info.inRange(1'000));
  ASSERT_FALSE(info.inRange(12'000));
  ASSERT_TRUE(overflows.empty());

  const std::vector<uint64_t> insertBuffers{100, 200, 300, 500};
  for (const auto insertBuffer : insertBuffers) {
    info.addOverflow(reinterpret_cast<char*>(insertBuffer));
  }
  ASSERT_EQ(overflows.size(), insertBuffers.size());
  for (int i = 0; i < insertBuffers.size(); ++i) {
    ASSERT_EQ(insertBuffers[i], reinterpret_cast<uint64_t>(info.overflows[i]));
  }
  for (int i = 0; i < overflows.size(); ++i) {
    ASSERT_EQ(overflows[i], info.overflows[i]);
  }
}

} // namespace facebook::velox::exec::test
