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

#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <thread>

namespace facebook::velox::common {
namespace {

using testing::Each;
using testing::ElementsAre;
using testing::Pointer;
using testing::SizeIs;

class ScanSpecTest : public testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(ScanSpecTest, applyFilter) {
  auto rowVector = makeRowVector({
      makeFlatVector<int64_t>(64, folly::identity),
      makeFlatVector<int64_t>(128, folly::identity),
  });
  ASSERT_EQ(rowVector->size(), 64);
  ScanSpec scanSpec("<root>");
  scanSpec.addAllChildFields(*rowVector->type());
  scanSpec.childByName("c1")->setFilter(createBigintValues({63, 64}, false));
  uint64_t result = -1ll;
  scanSpec.applyFilter(*rowVector, rowVector->size(), &result);
  ASSERT_EQ(result, 1ull << 63);
  result = -1ll;
  scanSpec.childByName("c1")->applyFilter(
      *rowVector->childAt("c1"), rowVector->size(), &result);
  ASSERT_EQ(result, 1ull << 63);
  rowVector = makeRowVector({
      makeFlatVector<int64_t>(128, folly::identity),
      makeFlatVector<int64_t>(64, folly::identity),
  });
  ASSERT_THROW(
      scanSpec.applyFilter(*rowVector, rowVector->size(), &result),
      VeloxRuntimeError);
}

TEST_F(ScanSpecTest, setFilterResetsHasFilter) {
  auto rowVector = makeRowVector({
      makeFlatVector<int64_t>(64, folly::identity),
      makeFlatVector<int64_t>(64, folly::identity),
  });

  ScanSpec scanSpec("<root>");
  scanSpec.addAllChildFields(*rowVector->type());

  // Initially no filter, hasFilter should be false.
  ASSERT_FALSE(scanSpec.hasFilter());
  ASSERT_FALSE(scanSpec.childByName("c0")->hasFilter());
  ASSERT_FALSE(scanSpec.childByName("c1")->hasFilter());

  // Set a filter on c0, hasFilter should be true for c0 and root.
  scanSpec.childByName("c0")->setFilter(createBigintValues({1, 2, 3}, false));
  ASSERT_FALSE(scanSpec.childByName("c0")->hasFilter());
  ASSERT_FALSE(scanSpec.hasFilter());
  // Root's hasFilter_ was cached as false, but setFilter should have reset it.
  // After setting filter on child, root should report hasFilter as true.
  scanSpec.resetCachedValues(false);
  ASSERT_TRUE(scanSpec.hasFilter());
  ASSERT_TRUE(scanSpec.childByName("c0")->hasFilter());

  // Set filter to nullptr, hasFilter should become false.
  scanSpec.childByName("c0")->setFilter(nullptr);
  ASSERT_TRUE(scanSpec.childByName("c0")->hasFilter());
  ASSERT_TRUE(scanSpec.hasFilter());
  scanSpec.resetCachedValues(false);
  ASSERT_FALSE(scanSpec.childByName("c0")->hasFilter());
  ASSERT_FALSE(scanSpec.hasFilter());

  // Set a new filter on c1, verify hasFilter updates correctly.
  scanSpec.childByName("c1")->setFilter(
      std::make_shared<BigintRange>(10, 50, false));
  ASSERT_FALSE(scanSpec.childByName("c1")->hasFilter());
  ASSERT_FALSE(scanSpec.childByName("c0")->hasFilter());
  scanSpec.resetCachedValues(false);
  ASSERT_FALSE(scanSpec.childByName("c0")->hasFilter());
  ASSERT_TRUE(scanSpec.childByName("c1")->hasFilter());
  ASSERT_TRUE(scanSpec.hasFilter());

  // Replace filter on c1 with a different filter.
  scanSpec.childByName("c1")->setFilter(
      std::make_shared<BigintRange>(20, 30, false));
  // hasFilter should still be true after replacing with another filter.
  ASSERT_TRUE(scanSpec.childByName("c1")->hasFilter());
  ASSERT_FALSE(scanSpec.childByName("c0")->hasFilter());
  ASSERT_TRUE(scanSpec.hasFilter());
}

TEST_F(ScanSpecTest, testFilterOnConstant) {
  auto test = [&](auto&& setup, bool expected) {
    ScanSpec scanSpec("<root>");
    auto* child = scanSpec.addField("c0", 0);
    setup(scanSpec, *child);
    ASSERT_EQ(
        dwio::common::SelectiveStructColumnReaderBase::testFilterOnConstant(
            *child),
        expected);
  };

  // Non-null constants are accepted regardless of filter kind.
  test(
      [&](ScanSpec&, ScanSpec& child) {
        child.setConstantValue(
            BaseVector::createConstant(BIGINT(), 1LL, 1, pool()));
        child.setFilter(std::make_shared<IsNull>());
      },
      true);
  test(
      [&](ScanSpec&, ScanSpec& child) {
        child.setConstantValue(
            BaseVector::createConstant(BIGINT(), 1LL, 1, pool()));
        child.setFilter(std::make_shared<IsNotNull>());
      },
      true);

  // Null constants are accepted only when the filter can match nulls.
  test(
      [&](ScanSpec&, ScanSpec& child) {
        child.setConstantValue(
            BaseVector::createNullConstant(BIGINT(), 1, pool()));
        child.setFilter(std::make_shared<IsNull>());
      },
      true);
  test(
      [&](ScanSpec& scanSpec, ScanSpec& child) {
        child.setConstantValue(
            BaseVector::createNullConstant(BIGINT(), 1, pool()));
        child.setFilter(std::make_shared<IsNotNull>());
      },
      false);

  // For non-constant specs, there is no filter or the filter accepts nulls.
  test([](ScanSpec&, ScanSpec&) {}, true);
  test(
      [](ScanSpec&, ScanSpec& child) {
        child.setFilter(std::make_shared<IsNull>());
      },
      true);
  test(
      [](ScanSpec&, ScanSpec& child) {
        child.setFilter(std::make_shared<IsNotNull>());
      },
      false);
}

// A child added after the first reader tree was built has to show up in
// stableChildren(), at the end.
TEST_F(ScanSpecTest, stableChildrenAfterAddingChild) {
  ScanSpec scanSpec("<root>");
  scanSpec.addField("c0", 0);
  scanSpec.addField("c1", 1);

  auto* first = scanSpec.childByName("c0");
  auto* second = scanSpec.childByName("c1");
  const auto beforeAdd = scanSpec.stableChildren();
  EXPECT_THAT(*beforeAdd, ElementsAre(Pointer(first), Pointer(second)));

  auto* third = scanSpec.addField("c2", 2);
  EXPECT_THAT(
      *scanSpec.stableChildren(),
      ElementsAre(Pointer(first), Pointer(second), Pointer(third)));

  // The snapshot a reader tree is walking is never mutated.
  EXPECT_THAT(*beforeAdd, ElementsAre(Pointer(first), Pointer(second)));

  // 'c2' is the only child with a filter, so it sorts to the front. The
  // stable order must not follow.
  scanSpec.childByName("c2")->setFilter(
      std::make_shared<BigintRange>(10, 20, false));
  scanSpec.resetCachedValues(true);
  ASSERT_EQ(scanSpec.children().front().get(), third);
  EXPECT_THAT(
      *scanSpec.stableChildren(),
      ElementsAre(Pointer(first), Pointer(second), Pointer(third)));
}

// An add drops the published snapshot. The next call republishes the whole
// order, whether or not anyone held the last one.
TEST_F(ScanSpecTest, stableChildrenRepublishedAfterAdd) {
  ScanSpec scanSpec("<root>");
  auto* first = scanSpec.addField("c0", 0);
  // Published and dropped, so nothing holds it when 'c1' is added.
  EXPECT_THAT(*scanSpec.stableChildren(), ElementsAre(Pointer(first)));

  auto* second = scanSpec.addField("c1", 1);
  const auto held = scanSpec.stableChildren();
  EXPECT_THAT(*held, ElementsAre(Pointer(first), Pointer(second)));

  // Held this time, so adding 'c2' must leave 'held' alone.
  auto* third = scanSpec.addField("c2", 2);
  EXPECT_THAT(*held, ElementsAre(Pointer(first), Pointer(second)));
  EXPECT_THAT(
      *scanSpec.stableChildren(),
      ElementsAre(Pointer(first), Pointer(second), Pointer(third)));
}

// Two threads calling getOrCreateChild() for one name must get one child. A
// lookup outside the insert's lock lets both miss and both create.
TEST_F(ScanSpecTest, getOrCreateChildConcurrently) {
  constexpr int32_t kNumThreads = 4;
  constexpr int32_t kNumIterations = 10;
  for (int32_t iteration = 0; iteration < kNumIterations; ++iteration) {
    ScanSpec scanSpec("<root>");
    // Releases every thread at once, widening the lookup-to-insert window.
    std::atomic_bool start{false};
    std::vector<ScanSpec*> children(kNumThreads);
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int32_t i = 0; i < kNumThreads; ++i) {
      threads.emplace_back([&, i] {
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        children[i] = scanSpec.getOrCreateChild("c0");
      });
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) {
      thread.join();
    }
    ASSERT_THAT(scanSpec.children(), SizeIs(1));
    auto* child = scanSpec.childByName("c0");
    ASSERT_EQ(child, scanSpec.children()[0].get());
    EXPECT_THAT(children, Each(child));
  }
}

// Stands in for a real updater: only the presence of the pointer decides
// whether a column counts as delta updated.
class NoopDeltaColumnUpdater : public dwio::common::DeltaColumnUpdater {
 public:
  void update(const RowSet& /*baseRows*/, VectorPtr& /*result*/) override {
    VELOX_UNREACHABLE();
  }
};

// setDeltaUpdate() takes filtering away from the reader and
// resetDeltaUpdates() gives it back. hasFilter() is memoized up the tree, so
// both have to invalidate it.
TEST_F(ScanSpecTest, deltaUpdateDisablesFilter) {
  ScanSpec scanSpec("<root>");
  auto* child = scanSpec.addField("c0", 0);
  child->setFilter(std::make_shared<BigintRange>(10, 20, false));
  scanSpec.resetCachedValues(false);
  ASSERT_TRUE(scanSpec.hasFilter());
  ASSERT_TRUE(child->hasFilter());

  NoopDeltaColumnUpdater updater;
  child->setDeltaUpdate(&updater);
  // Memoized as true just above, so this only holds if the root was reset too.
  EXPECT_FALSE(scanSpec.hasFilter());
  EXPECT_FALSE(child->hasFilter());
  // The filter stays on the spec for whoever applies it once values are final.
  EXPECT_TRUE(child->hasFilterApplicableToConstant());

  scanSpec.resetDeltaUpdates();
  EXPECT_EQ(child->deltaUpdate(), nullptr);
  EXPECT_TRUE(scanSpec.hasFilter());
  EXPECT_TRUE(child->hasFilter());
}

// resetDeltaUpdates() clears the columns that still carry an updater and
// leaves the rest alone.
TEST_F(ScanSpecTest, resetDeltaUpdatesOnlyClearsUpdatedColumns) {
  ScanSpec scanSpec("<root>");
  auto* first = scanSpec.addField("c0", 0);
  auto* second = scanSpec.addField("c1", 1);
  first->setFilter(std::make_shared<BigintRange>(10, 20, false));
  second->setFilter(std::make_shared<BigintRange>(30, 40, false));
  scanSpec.resetCachedValues(false);
  ASSERT_TRUE(scanSpec.hasFilter());

  NoopDeltaColumnUpdater updater;
  first->setDeltaUpdate(&updater);
  second->setDeltaUpdate(&updater);
  ASSERT_FALSE(scanSpec.hasFilter());

  first->setDeltaUpdate(nullptr);
  ASSERT_TRUE(first->hasFilter());
  ASSERT_FALSE(second->hasFilter());
  // The root filters again as soon as one column does.
  ASSERT_TRUE(scanSpec.hasFilter());

  scanSpec.resetDeltaUpdates();
  EXPECT_EQ(second->deltaUpdate(), nullptr);
  EXPECT_TRUE(first->hasFilter());
  EXPECT_TRUE(second->hasFilter());

  // Nothing carries an updater now, so this changes nothing.
  scanSpec.resetDeltaUpdates();
  EXPECT_TRUE(first->hasFilter());
  EXPECT_TRUE(second->hasFilter());
  EXPECT_TRUE(scanSpec.hasFilter());
}

std::shared_ptr<ScanSpec> makeAdaptationSpec() {
  auto spec = std::make_shared<ScanSpec>("<root>");
  spec->addField("c0", 0);
  spec->addField("c1", 1);
  return spec;
}

// moveAdaptationFrom() skips a child that is constant on either side, because
// a filter on a constant was evaluated at split start. Not so for a delta
// updated column: nothing evaluated its filter.
TEST_F(ScanSpecTest, moveAdaptationFromDeferredFilter) {
  auto from = makeAdaptationSpec();
  from->childByName("c0")->setFilter(
      std::make_shared<BigintRange>(10, 20, false));
  from->childByName("c1")->setFilter(
      std::make_shared<BigintRange>(30, 40, false));

  // Null constants, the shape a column missing from the file takes. Only 'c0'
  // is delta updated.
  auto to = makeAdaptationSpec();
  for (const auto& name : {"c0", "c1"}) {
    to->childByName(name)->setConstantValue(
        BaseVector::createNullConstant(BIGINT(), 1, pool()));
  }
  NoopDeltaColumnUpdater updater;
  to->childByName("c0")->setDeltaUpdate(&updater);

  to->moveAdaptationFrom(*from);

  EXPECT_TRUE(to->childByName("c0")->hasFilterApplicableToConstant());
  EXPECT_FALSE(to->childByName("c1")->hasFilterApplicableToConstant());
  // Moved out of 'from', not copied, and only for the child that received it.
  EXPECT_EQ(from->childByName("c0")->filter(), nullptr);
  EXPECT_NE(from->childByName("c1")->filter(), nullptr);
}

// A source holding no filter has nothing to carry. Taking its null filter
// anyway would drop the one the destination holds.
TEST_F(ScanSpecTest, moveAdaptationFromWithoutFilter) {
  auto from = makeAdaptationSpec();

  auto to = makeAdaptationSpec();
  auto* deferred = to->childByName("c0");
  deferred->setFilter(std::make_shared<BigintRange>(10, 20, false));
  deferred->setConstantValue(
      BaseVector::createNullConstant(BIGINT(), 1, pool()));
  NoopDeltaColumnUpdater updater;
  deferred->setDeltaUpdate(&updater);
  // Neither side is constant, the case that predates deferred filtering.
  auto* plain = to->childByName("c1");
  plain->setFilter(std::make_shared<BigintRange>(30, 40, false));

  to->moveAdaptationFrom(*from);

  EXPECT_TRUE(deferred->hasFilterApplicableToConstant());
  EXPECT_NE(plain->filter(), nullptr);
}

class TypedScanSpecTest : public testing::TestWithParam<TypePtr>,
                          public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  VectorPtr makeConstNullVector(TypePtr type, vector_size_t size) {
    return BaseVector::createNullConstant(type, size, pool());
  }

  void addIsNullFilterRecursive(ScanSpec& scanSpec) {
    scanSpec.setFilter(std::make_shared<velox::common::IsNull>());
    for (auto& child : scanSpec.children()) {
      addIsNullFilterRecursive(*child);
    }
  }

  void addIsNotNullFilterRecursive(ScanSpec& scanSpec) {
    scanSpec.setFilter(std::make_shared<velox::common::IsNotNull>());
    for (auto& child : scanSpec.children()) {
      addIsNullFilterRecursive(*child);
    }
  }

  void addIsNullFilterToLeaf(ScanSpec& scanSpec) {
    if (scanSpec.children().empty()) {
      scanSpec.setFilter(std::make_shared<velox::common::IsNull>());
    } else {
      for (auto& child : scanSpec.children()) {
        addIsNullFilterToLeaf(*child);
      }
    }
  }

  void addIsNotNullFilterToLeaf(ScanSpec& scanSpec) {
    if (scanSpec.children().empty()) {
      scanSpec.setFilter(std::make_shared<velox::common::IsNotNull>());
    } else {
      for (auto& child : scanSpec.children()) {
        addIsNotNullFilterToLeaf(*child);
      }
    }
  }
};

// Due to how subfield filters of maps and arrays are pruning
// and can't affect the row selectivity, the current test skips
// cases when maps and arrays are the lone child of (nested) structs.
INSTANTIATE_TEST_SUITE_P(
    TypedScanSpecTestSuite,
    TypedScanSpecTest,
    testing::Values(
        TINYINT(),
        SMALLINT(),
        INTEGER(),
        BIGINT(),
        REAL(),
        DOUBLE(),
        VARCHAR(),
        VARBINARY(),
        ROW({"int", "real"}, {INTEGER(), REAL()}),
        // TODO: the test cases fail when not specifying names for
        // the struct fields. This indicates bug in internal topology
        // when finding children of nested scan specs.
        ROW({"int", "map"}, {INTEGER(), MAP(INTEGER(), REAL())}),
        ROW({"int", "array"}, {INTEGER(), ARRAY(INTEGER())}),
        ROW({"int0", "array0", "row0"},
            {INTEGER(),
             ARRAY(INTEGER()),
             ROW({"int1", "real1", "row1"},
                 {INTEGER(),
                  REAL(),
                  ROW({"int2", "real2"}, {INTEGER(), REAL()})})})));

TEST_P(TypedScanSpecTest, applyFilterSchemaEvolution) {
  auto rowVector = makeRowVector({
      makeFlatVector<int64_t>(64, folly::identity),
      makeConstNullVector(GetParam(), 64),
  });
  ASSERT_EQ(rowVector->size(), 64);
  LOG(INFO) << "Testing with type: " << rowVector->type()->toString();

  {
    ScanSpec scanSpec("<root>");
    scanSpec.addAllChildFields(*rowVector->type());

    ASSERT_TRUE(scanSpec.childByName("c0"));
    scanSpec.childByName("c0")->setFilter(
        std::make_shared<BigintRange>(32, 64, false));

    ASSERT_TRUE(scanSpec.childByName("c1"));
    addIsNullFilterRecursive(*scanSpec.childByName("c1"));

    uint64_t result = -1ll;
    scanSpec.applyFilter(*rowVector, rowVector->size(), &result);
    ASSERT_EQ(result, -1ll << 32);

    // Now add a non-null filter on the missing column.
    ASSERT_TRUE(scanSpec.childByName("c1"));
    addIsNotNullFilterRecursive(*scanSpec.childByName("c1"));
    result = -1ll;
    scanSpec.applyFilter(*rowVector, rowVector->size(), &result);
    ASSERT_EQ(result, 0);
  }

  {
    ScanSpec scanSpec("<root>");
    scanSpec.addAllChildFields(*rowVector->type());

    ASSERT_TRUE(scanSpec.childByName("c0"));
    scanSpec.childByName("c0")->setFilter(
        std::make_shared<BigintRange>(32, 64, false));

    // Now add a null filter only on the innermost node of the missing column.
    // Should have the same result as recursive filters.
    ASSERT_TRUE(scanSpec.childByName("c1"));
    addIsNullFilterToLeaf(*scanSpec.childByName("c1"));
    uint64_t result = -1ll;
    scanSpec.applyFilter(*rowVector, rowVector->size(), &result);
    ASSERT_EQ(result, -1ll << 32);

    // Now add is not null filter only on the innermost node of the missing
    // column. Should have the same result as recursive filters.
    ASSERT_TRUE(scanSpec.childByName("c1"));
    addIsNotNullFilterToLeaf(*scanSpec.childByName("c1"));
    result = -1ll;
    scanSpec.applyFilter(*rowVector, rowVector->size(), &result);
    ASSERT_EQ(result, 0);
  }
}

} // namespace
} // namespace facebook::velox::common
