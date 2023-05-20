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

#include <optional>

#include "velox/row/UnsafeRowDeserializers.h"

#include "velox/row/UnsafeRowFast.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;

namespace facebook::velox::row {
class UnsafeRowSerdeTest : public testing::Test, public test::VectorTestBase {
 public:
  RowVectorPtr createInputRowWithUnknownType(int32_t batchSize) {
    auto flatVector = makeAllNullFlatVector<UnknownValue>(batchSize);
    auto arrayVector = makeAllNullArrayVector(batchSize, UNKNOWN());
    auto mapVector = makeAllNullMapVector(batchSize, UNKNOWN(), UNKNOWN());
    auto rowVector = makeRowVector({flatVector, arrayVector, mapVector});
    return makeRowVector({arrayVector, mapVector, flatVector, rowVector});
  }

  void testVectorSerde(const RowVectorPtr& inputVector) {
    std::vector<std::optional<std::string_view>> serializedRows;
    serializedRows.reserve(inputVector->size());
    buffers_.reserve(inputVector->size());
    UnsafeRowFast fast(inputVector);

    for (size_t i = 0; i < inputVector->size(); ++i) {
      const auto expectedRowSize = fast.rowSize(i);
      buffers_.push_back(
          AlignedBuffer::allocate<char>(expectedRowSize, pool_.get()));

      // Serialize rowVector into bytes.
      const auto rowSize = fast.serialize(i, buffers_[i]->asMutable<char>());

      EXPECT_EQ(expectedRowSize, rowSize);

      serializedRows.push_back(
          std::string_view(buffers_[i]->asMutable<char>(), rowSize));
    }
    const auto outputVector = UnsafeRowDeserializer::deserialize(
        serializedRows, inputVector->type(), pool());
    test::assertEqualVectors(inputVector, outputVector);
  }

 private:
  std::vector<BufferPtr> buffers_{};
};

TEST_F(UnsafeRowSerdeTest, unknownRows) {
  for (int32_t batchSize : {1, 5, 10}) {
    const auto& inputVector = createInputRowWithUnknownType(batchSize);
    testVectorSerde(inputVector);
  }
}

} // namespace facebook::velox::row
