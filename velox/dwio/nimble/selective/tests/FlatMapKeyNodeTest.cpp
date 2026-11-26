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

#include "velox/dwio/nimble/selective/FlatMapKeyNode.h"

#include <gtest/gtest.h>

#include "velox/common/file/File.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/selective/ColumnReader.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

// A SeekableArrayInputStream that sets '*destroyed' when it is destroyed.
class DestroyTrackingInputStream
    : public dwio::common::SeekableArrayInputStream {
 public:
  DestroyTrackingInputStream(std::string_view data, bool* destroyed)
      : SeekableArrayInputStream(data.data(), data.size()),
        destroyed_{destroyed} {}

  ~DestroyTrackingInputStream() override {
    *destroyed_ = true;
  }

 private:
  bool* const destroyed_;
};

// Tests FlatMapKeyNodePool on key nodes whose reader buildColumnReader() builds
// for a column of a one-stripe file: an integer column, whose reader can
// rebind, and a string column, whose reader cannot.
class FlatMapKeyNodeTest : public ::testing::Test,
                           public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    file_ = test::createNimbleFile(
        *rootPool_,
        makeRowVector(
            {"values", "strings"},
            {makeFlatVector<int64_t>({1, 2, 3}),
             makeFlatVector<std::string>({"a", "b", "c"})}));
    readerBase_ = ReaderBase::create(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(file_),
            FileIoContext{.pool = pool()}),
        dwio::common::ReaderOptions(pool()));
    streams_ = std::make_unique<StripeStreams>(readerBase_);
    streams_->setStripe(0);
    scanSpec_ = std::make_shared<common::ScanSpec>("root");
    scanSpec_->addAllChildFields(*readerBase_->fileSchema());
  }

  // Returns a key node with an in-map decoder that sets '*inMapDestroyed' when
  // its input is destroyed, and the reader of column 'name'.
  FlatMapKeyNodeBase makeKeyNode(
      const std::string& name,
      bool* inMapDestroyed) {
    FlatMapKeyNodeBase node;
    node.inMap = std::make_unique<ChunkedDecoder>(
        std::make_unique<DestroyTrackingInputStream>(
            inMapData_, inMapDestroyed),
        /*streamIndex=*/nullptr,
        /*decodeValuesWithNulls=*/false,
        &encodingFactory_,
        pool());
    const auto column = readerBase_->fileSchema()->getChildIdx(name);
    NimbleParams params(
        *pool(),
        splitStats_,
        readerBase_->nimbleSchema()->asRow().childAt(column),
        *streams_,
        /*rowSizeTracker=*/nullptr,
        encodingFactory_);
    node.reader = buildColumnReader(
        readerBase_->fileSchema()->childAt(column),
        readerBase_->fileSchemaWithId()->childAt(column),
        params,
        *scanSpec_->childByName(name),
        /*isRoot=*/false);
    return node;
  }

  const EncodingFactory encodingFactory_;
  // Never decoded: the tests only check which decoder holds it.
  const std::string inMapData_{"in-map"};
  std::string file_;
  std::shared_ptr<ReaderBase> readerBase_;
  std::unique_ptr<StripeStreams> streams_;
  std::shared_ptr<common::ScanSpec> scanSpec_;
  dwio::common::SplitStats splitStats_{dwio::common::FileFormat::NIMBLE};
};

TEST_F(FlatMapKeyNodeTest, takeWithoutKeptNode) {
  FlatMapKeyNodePool pool;
  FlatMapKeyNodeBase node;
  EXPECT_FALSE(pool.take(/*inMapStream=*/7, node));
  EXPECT_EQ(node.inMap, nullptr);
  EXPECT_EQ(node.reader, nullptr);
}

// Keeping a node unbinds its decoders, which destroys the input of the in-map
// decoder, and taking it back for the same in-map stream returns the same
// readers once.
TEST_F(FlatMapKeyNodeTest, putAndTake) {
  bool inMapDestroyed{false};
  auto node = makeKeyNode("values", &inMapDestroyed);
  ASSERT_TRUE(node.reader->rebindable());
  const auto* inMap = node.inMap.get();
  const auto* reader = node.reader.get();

  FlatMapKeyNodePool pool;
  pool.put(/*inMapStream=*/7, node);
  EXPECT_EQ(node.inMap, nullptr);
  EXPECT_EQ(node.reader, nullptr);
  EXPECT_TRUE(inMapDestroyed);

  FlatMapKeyNodeBase taken;
  EXPECT_FALSE(pool.take(/*inMapStream=*/8, taken));
  ASSERT_TRUE(pool.take(/*inMapStream=*/7, taken));
  EXPECT_EQ(taken.inMap.get(), inMap);
  EXPECT_EQ(taken.reader.get(), reader);

  FlatMapKeyNodeBase takenAgain;
  EXPECT_FALSE(pool.take(/*inMapStream=*/7, takenAgain));
}

// The pool keeps one node per in-map stream. Keeping a second node of the same
// key leaves it to its caller, unbound.
TEST_F(FlatMapKeyNodeTest, putKeepsFirstNodeOfKey) {
  bool firstDestroyed{false};
  bool secondDestroyed{false};
  auto first = makeKeyNode("values", &firstDestroyed);
  auto second = makeKeyNode("values", &secondDestroyed);
  const auto* firstReader = first.reader.get();

  FlatMapKeyNodePool pool;
  pool.put(/*inMapStream=*/7, first);
  pool.put(/*inMapStream=*/7, second);
  EXPECT_TRUE(firstDestroyed);
  EXPECT_FALSE(secondDestroyed);
  EXPECT_NE(second.inMap, nullptr);
  EXPECT_NE(second.reader, nullptr);

  FlatMapKeyNodeBase taken;
  ASSERT_TRUE(pool.take(/*inMapStream=*/7, taken));
  EXPECT_EQ(taken.reader.get(), firstReader);
}

TEST_F(FlatMapKeyNodeTest, putNodeThatCannotRebind) {
  bool inMapDestroyed{false};
  auto node = makeKeyNode("strings", &inMapDestroyed);
  ASSERT_FALSE(node.reader->rebindable());
  FlatMapKeyNodePool pool;
  NIMBLE_ASSERT_THROW(
      pool.put(/*inMapStream=*/7, node),
      "Keeping a key node that cannot rebind.");
  EXPECT_FALSE(inMapDestroyed);
}

TEST_F(FlatMapKeyNodeTest, numReusedKeyNodes) {
  FlatMapKeyNodePool pool;
  EXPECT_EQ(pool.numReusedKeyNodes(), 0);
  pool.addReusedKeyNodes();
  pool.addReusedKeyNodes();
  EXPECT_EQ(pool.numReusedKeyNodes(), 2);
}

} // namespace
} // namespace facebook::nimble
