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

#include <folly/Random.h>
#include <folly/init/Init.h>

#include "velox/row/UnsafeRowDeserializers.h"
#include "velox/row/UnsafeRowSerializers.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::row {
namespace {

using namespace facebook::velox::test;

class UnsafeRowFuzzTests : public ::testing::Test {
 public:
  UnsafeRowFuzzTests() {
    clearBuffer();
  }

  void clearBuffer() {
    std::memset(buffer_, 0, BUFFER_SIZE);
  }

  std::shared_ptr<memory::MemoryPool> pool_ = memory::getDefaultMemoryPool();
  BufferPtr bufferPtr_ =
      AlignedBuffer::allocate<char>(BUFFER_SIZE, pool_.get(), true);
  char* buffer_ = bufferPtr_->asMutable<char>();
  static constexpr uint64_t BUFFER_SIZE = 20 << 10; // 20k
};

TEST_F(UnsafeRowFuzzTests, simpleTypeRoundTripTest) {
  auto rowType = ROW(
      {BOOLEAN(),
       TINYINT(),
       SMALLINT(),
       INTEGER(),
       BIGINT(),
       REAL(),
       DOUBLE(),
       VARCHAR(),
       TIMESTAMP(),
       ROW({VARCHAR(), INTEGER()}),
       ARRAY(INTEGER()),
       ARRAY(INTEGER()),
       MAP(VARCHAR(), ARRAY(INTEGER()))});

  VectorFuzzer::Options opts;
  opts.vectorSize = 1;
  opts.nullRatio = 0.1;
  opts.containerHasNulls = false;
  opts.dictionaryHasNulls = false;
  opts.stringVariableLength = true;
  opts.stringLength = 20;
  opts.containerVariableLength = false;
  opts.complexElementsMaxSize = 1000000;

  // Spark uses microseconds to store timestamp
  opts.timestampPrecision =
      VectorFuzzer::Options::TimestampPrecision::kMicroSeconds,
  opts.containerLength = 10;

  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  SCOPED_TRACE(fmt::format("seed: {}", seed));
  VectorFuzzer fuzzer(opts, pool_.get(), seed);

  const auto iterations = 1000;
  for (size_t i = 0; i < iterations; ++i) {
    clearBuffer();
    const auto& inputVector = fuzzer.fuzzRow(rowType);
    // Serialize rowVector into bytes.
    UnsafeRowDynamicSerializer::preloadVector(inputVector);
    auto rowSize = UnsafeRowDynamicSerializer::serialize(
        rowType, inputVector, buffer_, /*idx=*/0);

    auto rowSizeMeasured =
        UnsafeRowDynamicSerializer::getSizeRow(rowType, inputVector.get(), 0);
    EXPECT_EQ(rowSize.value_or(0), rowSizeMeasured);

    // Deserialize previous bytes back to row vector
    VectorPtr outputVector =
        UnsafeRowDynamicVectorBatchDeserializer::deserializeComplex(
            std::string_view(buffer_, rowSize.value()), rowType, pool_.get());

    assertEqualVectors(inputVector, outputVector);
  }
}

} // namespace
} // namespace facebook::velox::row
