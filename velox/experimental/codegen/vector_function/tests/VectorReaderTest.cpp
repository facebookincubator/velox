/*
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

#include <folly/Random.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <optional>
#include <vector>
#include "velox/experimental/codegen/vector_function/GeneratedVectorFunction-inl.h" // NOLINT (CLANGTIDY  )
#include "velox/experimental/codegen/vector_function/tests/VectorReaderTestBase.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/VectorMaker.h"

namespace facebook::velox::codegen {

TEST_F(VectorReaderTestBase, ReadDoublesVectors) {
  const size_t vectorSize = 1000;
  auto inRowType = ROW({"columnA", "columnB"}, {DOUBLE(), DOUBLE()});
  auto inRowVector = BaseVector::create(inRowType, vectorSize, pool_.get());

  VectorPtr& in1 = inRowVector->as<RowVector>()->childAt(0);

  SelectivityVector selectivityVector(vectorSize);
  selectivityVector.setAll();
  in1->resize(vectorSize);
  in1->addNulls(nullptr, selectivityVector);
  VectorReader<DoubleType, OutputReaderConfig<false, false>> writer(in1);
  VectorReader<DoubleType, InputReaderConfig<false, false>> reader(in1);

  for (size_t row = 0; row < vectorSize; row++) {
    writer[row] = (double)row;
  }

  for (size_t row = 0; row < vectorSize; row++) {
    ASSERT_DOUBLE_EQ((double)row, *reader[row]);
  }

  for (size_t row = 0; row < vectorSize; row++) {
    ASSERT_DOUBLE_EQ(*reader[row], in1->asFlatVector<double>()->valueAt(row));
  }
}

TEST_F(VectorReaderTestBase, ReadBoolVectors) {
  const size_t vectorSize = 1000;
  auto inRowType = ROW({"columnA", "columnB"}, {BOOLEAN(), BOOLEAN()});

  auto inRowVector = BaseVector::create(inRowType, vectorSize, pool_.get());

  VectorPtr& inputVector = inRowVector->as<RowVector>()->childAt(0);
  inputVector->resize(vectorSize);
  VectorReader<BooleanType, InputReaderConfig<false, false>> reader(
      inputVector);
  VectorReader<BooleanType, OutputReaderConfig<false, false>> writer(
      inputVector);

  for (size_t row = 0; row < vectorSize; row++) {
    writer[row] = row % 2 == 0;
  }

  // Check that writing of values to the reader was success
  for (size_t row = 0; row < vectorSize; row++) {
    ASSERT_DOUBLE_EQ((row % 2 == 0), *reader[row]);
    ASSERT_DOUBLE_EQ(
        (row % 2 == 0), inputVector->asFlatVector<bool>()->valueAt(row));
  }

  // Write a null at even indices
  for (size_t row = 0; row < vectorSize; row++) {
    if (row % 2) {
      writer[row] = std::nullopt;
    }
  }

  for (size_t row = 0; row < vectorSize; row++) {
    ASSERT_EQ(inputVector->asFlatVector<bool>()->isNullAt(row), row % 2);
  }
}

TEST_F(VectorReaderTestBase, ReadStringVectors) {
  const size_t vectorSize = 4;
  auto inRowType = ROW({"columnA"}, {VARCHAR()});

  auto inRowVector = BaseVector::create(inRowType, vectorSize, pool_.get());

  VectorPtr& inputVector = inRowVector->as<RowVector>()->childAt(0);
  inputVector->resize(vectorSize);

  VectorReader<VarcharType, OutputReaderConfig<false, false>> writer(
      inputVector);
  VectorReader<VarcharType, InputReaderConfig<false, false>> reader(
      inputVector);

  auto helloWorldRef = facebook::velox::StringView(u8"Hello, World!", 13);
  InputReferenceStringNullable helloWorld{InputReferenceString(helloWorldRef)};
  auto emptyStringRef = StringView(u8"", 0);
  InputReferenceStringNullable emptyString{
      InputReferenceString(emptyStringRef)};
  auto inlineRef = StringView(u8"INLINE", 6);
  InputReferenceStringNullable inlineString{InputReferenceString(inlineRef)};

  writer[0] = helloWorld;
  writer[1] = emptyString;
  writer[2] = std::nullopt;
  writer[3] = inlineString;

  ASSERT_TRUE(reader[0].has_value());
  ASSERT_EQ(reader[0].value().size(), 13);
  ASSERT_TRUE(gtestMemcmp(
      (*reader[0]).data(), (void*)"Hello, World!", (*reader[0]).size()));

  ASSERT_TRUE(reader[1].has_value());
  ASSERT_EQ(reader[1].value().size(), 0);
  ASSERT_TRUE(gtestMemcmp((*reader[1]).data(), (void*)"", (*reader[1]).size()));

  ASSERT_FALSE(reader[2].has_value());

  ASSERT_TRUE(reader[3].has_value());
  ASSERT_EQ(reader[3].value().size(), 6);
  ASSERT_TRUE(
      gtestMemcmp((*reader[3]).data(), (void*)"INLINE", (*reader[3]).size()));
}

} // namespace facebook::velox::codegen
