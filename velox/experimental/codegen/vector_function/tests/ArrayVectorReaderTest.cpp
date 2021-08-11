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

namespace facebook::velox::codegen {

TEST_F(ComplexVectorReaderTest, ReadArraySmallintVectors) {
  /// ArrayVector<FlatVector<int16_t>>:
  /// [ null, [0x0333, null, 0x1444], [0x1666, 0x0777, null, 0x0999] ]
  /// size: 3
  /// offsets: [0, 0, 3]
  /// lengths: [0, 3, 4]
  /// nulls: 0b001
  /// elements:
  ///  FlatVector<int16_t>:
  ///  size: 7
  ///  [0x0333, null, 0x1444, 0x1666, 0x0777, null, 0x0999]
  ///  nulls: 0b0100000
  size_t flatVectorSize = 7;
  auto flatVector =
      makeFlatVectorPtr<int16_t>(flatVectorSize, SMALLINT(), pool_.get());

  size_t arrayVectorSize = 3;
  auto arrayVector = makeArrayVectorPtr(
      arrayVectorSize, pool_.get(), ARRAY(SMALLINT()), flatVector);

  ComplexVectorReader<Array<SmallintType>, OutputReaderConfig<false, false>>
      writer(arrayVector);
  ComplexVectorReader<Array<SmallintType>, InputReaderConfig<false, false>>
      reader(arrayVector);

  writer[0].setNullAndSize();
  ASSERT_FALSE(reader[0].has_value());

  writer[1].setNotNullAndSize(3);
  writer[1][0] = 0x0333;
  writer[1][1] = std::nullopt;
  writer[1][2] = 0x1444;
  ASSERT_EQ(reader[1][0].value(), 0x0333);
  ASSERT_FALSE(reader[1][1].has_value());
  ASSERT_EQ(reader[1][2].value(), 0x1444);
  ASSERT_EQ(reader[1].size(), 3);

  using InputType = ComplexVectorReader<
      Array<SmallintType>,
      OutputReaderConfig<false, false>>::InputType;
  using ElementInputType = ComplexVectorReader<
      Array<SmallintType>,
      OutputReaderConfig<false, false>>::ElementInputType;

  InputType smallint2 = std::make_optional(
      std::vector<ElementInputType>{0x1666, 0x0777, std::nullopt, 0x0999});
  writer[2] = smallint2;
  ASSERT_EQ(reader[2][0].value(), 0x1666);
  ASSERT_EQ(reader[2][1].value(), 0x0777);
  ASSERT_FALSE(reader[2][2].has_value());
  ASSERT_EQ(reader[2][3].value(), 0x0999);
}

TEST_F(ComplexVectorReaderTest, ReadArrayBoolVectors) {
  /// ArrayVector<FlatVector<bool>>:
  /// [ [true, false, null], null, [false, false, null, false], null]
  /// size: 4
  /// offsets: [0, 3, 3, 7]
  /// lengths: [3, 0, 4, 0]
  /// nulls: 0b1010
  /// elements:
  ///  FlatVector<bool>:
  ///  size: 7
  ///  [true, false, null, false, false, null, false]
  ///  nulls: 0b0100100
  size_t flatVectorSize = 7;
  auto flatVector =
      makeFlatVectorPtr<bool>(flatVectorSize, BOOLEAN(), pool_.get());

  size_t arrayVectorSize = 4;
  auto arrayVector = makeArrayVectorPtr(
      arrayVectorSize, pool_.get(), ARRAY(BOOLEAN()), flatVector);

  ComplexVectorReader<Array<BooleanType>, OutputReaderConfig<false, false>>
      writer(arrayVector);
  ComplexVectorReader<Array<BooleanType>, InputReaderConfig<false, false>>
      reader(arrayVector);

  // if we set elements one-by-one, we need to explicitly call
  // setNotNullAndSize()
  writer[0].setNotNullAndSize(3);
  writer[0][0] = true;
  writer[0][1] = false;
  writer[0][2] = std::nullopt;
  ASSERT_EQ(reader[0][0].value(), true);
  ASSERT_EQ(reader[0][1].value(), false);
  ASSERT_FALSE(reader[0][2].has_value());
  ASSERT_EQ(reader[0].size(), 3);

  writer[1].setNullAndSize();
  ASSERT_FALSE(reader[1].has_value());
  EXPECT_THROW(reader[1].size(), std::logic_error);

  using InputType = ComplexVectorReader<
      Array<BooleanType>,
      OutputReaderConfig<false, false>>::InputType;
  using ElementInputType = ComplexVectorReader<
      Array<BooleanType>,
      OutputReaderConfig<false, false>>::ElementInputType;

  InputType bool2 = std::make_optional(
      std::vector<ElementInputType>{false, false, std::nullopt, false});
  writer[2] = bool2;
  ASSERT_EQ(reader[2][0].value(), false);
  ASSERT_EQ(reader[2][1].value(), false);
  ASSERT_FALSE(reader[2][2].has_value());
  ASSERT_EQ(reader[2][3].value(), false);
  ASSERT_EQ(reader[2].size(), 4);

  InputType bool3 = std::nullopt;
  writer[3] = bool3;
  ASSERT_FALSE(reader[3].has_value());
  EXPECT_THROW(reader[3].size(), std::logic_error);
}

TEST_F(ComplexVectorReaderTest, ReadArrayStringVectors) {
  /// ArrayVector<FlatVector<StringView>>:
  /// [ hello, longString, emptyString, null ], [null, world], null, null]
  /// size: 4
  /// offsets: [0, 4, 6, 6]
  /// lengths: [4, 2, 0, 0]
  /// nulls: 0b1100
  /// elements:
  ///  FlatVector<StringView>:
  ///  size: 6
  ///  [ hello, longString, emptyString, null, null, world]
  ///  nulls: 0b011000

  auto helloRef = facebook::velox::StringView(u8"Hello", 5);
  InputReferenceStringNullable hello{InputReferenceString(helloRef)};
  auto longStringRef =
      StringView(u8"This is a rather long string.  Quite long indeed.", 49);
  InputReferenceStringNullable longString{InputReferenceString(longStringRef)};
  auto emptyStringRef = StringView(u8"", 0);
  InputReferenceStringNullable emptyString{
      InputReferenceString(emptyStringRef)};
  auto worldRef = StringView(u8"World", 5);
  InputReferenceStringNullable world{InputReferenceString(worldRef)};

  size_t flatVectorSize = 6;
  auto flatVector =
      makeFlatVectorPtr<StringView>(flatVectorSize, VARCHAR(), pool_.get());

  size_t arrayVectorSize = 4;
  auto arrayVector = makeArrayVectorPtr(
      arrayVectorSize, pool_.get(), ARRAY(VARCHAR()), flatVector);

  ComplexVectorReader<Array<VarcharType>, OutputReaderConfig<false, false>>
      writer(arrayVector);
  ComplexVectorReader<Array<VarcharType>, InputReaderConfig<false, false>>
      reader(arrayVector);

  writer[0].setNotNullAndSize(4);
  writer[0][0] = hello;
  writer[0][1] = longString;
  writer[0][2] = emptyString;
  writer[0][3] = std::nullopt;

  ASSERT_TRUE(reader[0].has_value());

  ASSERT_TRUE(reader[0][0].has_value());
  ASSERT_EQ(reader[0][0].value().size(), 5);
  ASSERT_TRUE(gtestMemcmp(
      (*reader[0][0]).data(), (void*)helloRef.data(), (*reader[0][0]).size()));

  ASSERT_TRUE(reader[0][1].has_value());
  ASSERT_EQ(reader[0][1].value().size(), 49);
  ASSERT_TRUE(gtestMemcmp(
      (*reader[0][1]).data(),
      (void*)longStringRef.data(),
      (*reader[0][1]).size()));

  ASSERT_TRUE(reader[0][2].has_value());
  ASSERT_EQ(reader[0][2].value().size(), 0);
  ASSERT_TRUE(gtestMemcmp(
      (*reader[0][2]).data(), (void*)helloRef.data(), (*reader[0][2]).size()));

  ASSERT_FALSE(reader[0][3].has_value());

  using InputType = ComplexVectorReader<
      Array<VarcharType>,
      OutputReaderConfig<false, false>>::InputType;
  using ElementInputType = ComplexVectorReader<
      Array<VarcharType>,
      OutputReaderConfig<false, false>>::ElementInputType;
  InputType string1 = std::make_optional(
      std::vector<ElementInputType>{InputReferenceStringNullable{}, world});
  writer[1] = string1;

  ASSERT_TRUE(reader[1].has_value());

  ASSERT_FALSE(reader[1][0].has_value());

  ASSERT_TRUE(reader[1][1].has_value());
  ASSERT_EQ(reader[1][1].value().size(), 5);
  ASSERT_TRUE(gtestMemcmp(
      (*reader[1][1]).data(), (void*)worldRef.data(), (*reader[1][1]).size()));

  writer[2].setNullAndSize();
  ASSERT_FALSE(reader[2].has_value());
  EXPECT_THROW(reader[2].size(), std::logic_error);

  InputType val = std::nullopt;
  writer[3] = val;
  ASSERT_FALSE(reader[3].has_value());
  EXPECT_THROW(reader[3].size(), std::logic_error);
}

} // namespace facebook::velox::codegen
