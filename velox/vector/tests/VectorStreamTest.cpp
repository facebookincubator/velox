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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorStream.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::test {

class MockVectorSerde : public VectorSerde {
 public:
  MockVectorSerde() : VectorSerde(VectorSerde::Kind::kPresto) {}

  void estimateSerializedSize(
      const BaseVector* /*vector*/,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes) override {}

  std::unique_ptr<IterativeVectorSerializer> createIterativeSerializer(
      RowTypePtr type,
      int32_t numRows,
      StreamArena* streamArena,
      const Options* options = nullptr) override {
    return nullptr;
  };

  void deserialize(
      ByteInputStream* source,
      velox::memory::MemoryPool* pool,
      RowTypePtr type,
      RowVectorPtr* result,
      const Options* options = nullptr) override {}
};

TEST(VectorStreamTest, serdeRegistration) {
  deregisterVectorSerde();

  // Nothing registered yet.
  EXPECT_FALSE(isRegisteredVectorSerde());
  EXPECT_THROW(getVectorSerde(), VeloxRuntimeError);

  // Register a mock serde.
  registerVectorSerde(std::make_unique<MockVectorSerde>());

  EXPECT_TRUE(isRegisteredVectorSerde());
  auto serde = getVectorSerde();
  EXPECT_NE(serde, nullptr);
  EXPECT_NE(dynamic_cast<MockVectorSerde*>(serde), nullptr);

  // Can't double register.
  EXPECT_THROW(
      registerVectorSerde(std::make_unique<MockVectorSerde>()),
      VeloxRuntimeError);

  deregisterVectorSerde();
  EXPECT_FALSE(isRegisteredVectorSerde());
}

TEST(VectorStreamTest, namedSerdeRegistration) {
  const VectorSerde::Kind kind = VectorSerde::Kind::kPresto;

  // Nothing registered yet.
  deregisterNamedVectorSerde(kind);
  EXPECT_FALSE(isRegisteredNamedVectorSerde(kind));
  VELOX_ASSERT_THROW(
      getNamedVectorSerde(kind),
      "Named vector serde 'Presto' is not registered.");

  // Register a mock serde.
  registerNamedVectorSerde(kind, std::make_unique<MockVectorSerde>());

  auto serde = getNamedVectorSerde(kind);
  EXPECT_NE(serde, nullptr);
  EXPECT_NE(dynamic_cast<MockVectorSerde*>(serde), nullptr);

  const VectorSerde::Kind otherKind = VectorSerde::Kind::kUnsafeRow;
  EXPECT_FALSE(isRegisteredNamedVectorSerde(otherKind));
  VELOX_ASSERT_THROW(
      getNamedVectorSerde(otherKind),
      "Named vector serde 'UnsafeRow' is not registered.");

  // Can't double register.
  VELOX_ASSERT_THROW(
      registerNamedVectorSerde(kind, std::make_unique<MockVectorSerde>()),
      "Vector serde 'Presto' is already registered.");

  // Register another one.
  EXPECT_FALSE(isRegisteredNamedVectorSerde(otherKind));
  EXPECT_THROW(getNamedVectorSerde(otherKind), VeloxRuntimeError);
  registerNamedVectorSerde(otherKind, std::make_unique<MockVectorSerde>());
  EXPECT_TRUE(isRegisteredNamedVectorSerde(otherKind));

  deregisterNamedVectorSerde(otherKind);
  EXPECT_FALSE(isRegisteredNamedVectorSerde(otherKind));
}

class VectorStreamNullChildTest : public testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
    serde_ = std::make_unique<serializer::presto::PrestoVectorSerde>();
  }

  std::unique_ptr<serializer::presto::PrestoVectorSerde> serde_;
};

// Test that rowVectorToIOBuf handles RowVector with nullptr children.
// This reproduces an issue introduced in D87873415 where the switch from
// VectorStreamGroup to BatchVectorSerializer causes a crash when serializing
// a RowVector with nullptr children.
//
// RowVector children can be nullptr in legitimate cases (e.g., when columns
// are not projected/populated). The fix materializes null children as null
// constant vectors before serialization.
TEST_F(VectorStreamNullChildTest, rowVectorWithNullChild) {
  // Create a RowVector with a nullptr child
  auto rowType = ROW({"a", "b", "c"}, {BIGINT(), VARCHAR(), INTEGER()});

  // Create some valid children
  auto childA = makeFlatVector<int64_t>({1, 2, 3});
  auto childC = makeFlatVector<int32_t>({10, 20, 30});

  // Create RowVector with nullptr for child "b"
  auto rowVector = std::make_shared<RowVector>(
      pool_.get(),
      rowType,
      BufferPtr(nullptr),
      3, // numRows
      std::vector<VectorPtr>{childA, nullptr, childC});

  // Verify the setup: child at index 1 should be nullptr
  ASSERT_EQ(rowVector->childAt(0).get(), childA.get());
  ASSERT_EQ(rowVector->childAt(1), nullptr);
  ASSERT_EQ(rowVector->childAt(2).get(), childC.get());

  // Should now succeed by materializing null children.
  auto result = rowVectorToIOBuf(rowVector, *pool_, serde_.get());
  EXPECT_GT(result.computeChainDataLength(), 0);

  // Verify round-trip: deserialize and check that the structure is preserved.
  auto deserialized = IOBufToRowVector(result, rowType, *pool_, serde_.get());

  ASSERT_EQ(deserialized->size(), 3);
  ASSERT_EQ(deserialized->childrenSize(), 3);

  // Child "a" should be preserved
  auto deserializedA = deserialized->childAt(0)->asFlatVector<int64_t>();
  EXPECT_EQ(deserializedA->valueAt(0), 1);
  EXPECT_EQ(deserializedA->valueAt(1), 2);
  EXPECT_EQ(deserializedA->valueAt(2), 3);

  // Child "b" (index 1) should now exist as a null constant vector
  ASSERT_NE(deserialized->childAt(1), nullptr);
  EXPECT_EQ(
      deserialized->childAt(1)->encoding(), VectorEncoding::Simple::CONSTANT);
  EXPECT_TRUE(deserialized->childAt(1)->isNullAt(0));

  // Child "c" should be preserved
  auto deserializedC = deserialized->childAt(2)->asFlatVector<int32_t>();
  EXPECT_EQ(deserializedC->valueAt(0), 10);
  EXPECT_EQ(deserializedC->valueAt(1), 20);
  EXPECT_EQ(deserializedC->valueAt(2), 30);
}

// Test that rowVectorToIOBuf works correctly with all valid (non-null) children
TEST_F(VectorStreamNullChildTest, rowVectorWithAllValidChildren) {
  auto rowType = ROW({"a", "b", "c"}, {BIGINT(), VARCHAR(), INTEGER()});

  auto childA = makeFlatVector<int64_t>({1, 2, 3});
  auto childB = makeFlatVector<StringView>({"x", "y", "z"});
  auto childC = makeFlatVector<int32_t>({10, 20, 30});

  auto rowVector = std::make_shared<RowVector>(
      pool_.get(),
      rowType,
      BufferPtr(nullptr),
      3,
      std::vector<VectorPtr>{childA, childB, childC});

  // This should succeed without issues
  auto ioBuf = rowVectorToIOBuf(rowVector, *pool_, serde_.get());
  EXPECT_GT(ioBuf.computeChainDataLength(), 0);

  // Verify round-trip deserialization
  auto deserialized = IOBufToRowVector(ioBuf, rowType, *pool_, serde_.get());
  EXPECT_EQ(deserialized->size(), 3);

  for (int i = 0; i < 3; i++) {
    EXPECT_TRUE(rowVector->childAt(i)->equalValueAt(
        deserialized->childAt(i).get(), i, i));
  }
}

// Test that rowVectorToIOBuf handles NESTED RowVector with nullptr children.
// This is a realistic scenario similar to the Logarithm production crash where
// the schema includes nested ROW types like:
//   - tupperware: ROW({job_cluster, job_user, job_name, task_id})
//   - tensorboard: ROW({run_name, ttl_ms, acl, tag, data_type, error_message})
//
// The fix recursively materializes null children at all nesting levels.
TEST_F(VectorStreamNullChildTest, nestedRowVectorWithNullChild) {
  // Create a schema with nested ROW, similar to Logarithm's tupperware struct
  auto nestedRowType = ROW(
      {{"job_cluster", VARCHAR()},
       {"job_user", VARCHAR()},
       {"job_name", VARCHAR()},
       {"task_id", INTEGER()}});

  auto outerRowType = ROW({{"id", BIGINT()}, {"tupperware", nestedRowType}});

  // Create valid outer children
  auto idVector = makeFlatVector<int64_t>({1, 2, 3});

  // Create a nested RowVector with a nullptr child (job_user is nullptr)
  auto jobCluster =
      makeFlatVector<StringView>({"cluster1", "cluster2", "cluster3"});
  auto jobName = makeFlatVector<StringView>({"name1", "name2", "name3"});
  auto taskId = makeFlatVector<int32_t>({100, 200, 300});

  // Nested RowVector with nullptr for job_user (index 1)
  auto nestedRowVector = std::make_shared<RowVector>(
      pool_.get(),
      nestedRowType,
      BufferPtr(nullptr),
      3,
      std::vector<VectorPtr>{jobCluster, nullptr, jobName, taskId});

  // Outer RowVector containing the nested RowVector with nullptr child
  auto outerRowVector = std::make_shared<RowVector>(
      pool_.get(),
      outerRowType,
      BufferPtr(nullptr),
      3,
      std::vector<VectorPtr>{idVector, nestedRowVector});

  // Verify setup
  ASSERT_EQ(outerRowVector->childAt(1)->as<RowVector>()->childAt(1), nullptr);

  // Should now succeed by recursively materializing null children.
  auto result = rowVectorToIOBuf(outerRowVector, *pool_, serde_.get());
  EXPECT_GT(result.computeChainDataLength(), 0);

  // Verify round-trip: deserialize and check that the structure is preserved.
  auto deserialized =
      IOBufToRowVector(result, outerRowType, *pool_, serde_.get());

  ASSERT_EQ(deserialized->size(), 3);
  ASSERT_EQ(deserialized->childrenSize(), 2);

  // The id column should be preserved.
  auto deserializedId = deserialized->childAt(0)->asFlatVector<int64_t>();
  EXPECT_EQ(deserializedId->valueAt(0), 1);
  EXPECT_EQ(deserializedId->valueAt(1), 2);
  EXPECT_EQ(deserializedId->valueAt(2), 3);

  // The nested row should have all its children materialized (including the
  // previously null job_user field which should now be a null constant).
  auto deserializedNested = deserialized->childAt(1)->as<RowVector>();
  ASSERT_EQ(deserializedNested->childrenSize(), 4);

  // job_user (index 1) should now exist as a null constant vector.
  ASSERT_NE(deserializedNested->childAt(1), nullptr);
  EXPECT_EQ(
      deserializedNested->childAt(1)->encoding(),
      VectorEncoding::Simple::CONSTANT);
  EXPECT_TRUE(deserializedNested->childAt(1)->isNullAt(0));
}

} // namespace facebook::velox::test
