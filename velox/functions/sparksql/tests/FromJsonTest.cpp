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
#include "velox/functions/sparksql/tests/JsonTestUtil.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {
class FromJsonTest : public SparkFunctionBaseTest {
 protected:
  void testFromJson(const VectorPtr& input, const VectorPtr& expected) {
    auto expr = createFromJson(expected->type());
    testEncodings(expr, {input}, expected);
  }

  VectorPtr evaluateFromJson(
      const VectorPtr& input,
      const TypePtr& outputType) {
    return evaluate(createFromJson(outputType), makeRowVector({input}));
  }
};

TEST_F(FromJsonTest, basicStruct) {
  auto expected = makeFlatVector<int64_t>({1, 2, 3});
  auto input = makeFlatVector<std::string>(
      {R"({"Id": 1})", R"({"Id": 2})", R"({"Id": 3})"});
  testFromJson(input, makeRowVector({"Id"}, {expected}));
}

TEST_F(FromJsonTest, structFieldMatchingAndDuplicates) {
  auto expected =
      makeNullableFlatVector<int64_t>({std::nullopt, 2, 3, std::nullopt, 4, 5});
  auto input = makeFlatVector<std::string>(
      {R"({"Id": 2})",
       R"({"id": 1, "id": 2})",
       R"({"id": null, "id": 3})",
       R"({"id": 4, "id": null})",
       R"({"id": 4, "id": "x"})",
       R"({"id": "x", "id": 5})"});
  testFromJson(input, makeRowVector({"id"}, {expected}));
}

TEST_F(FromJsonTest, rootStructFailures) {
  auto nullBigint = makeNullableFlatVector<int64_t>(
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
  auto nullString = makeNullableFlatVector<StringView>(
      {std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt});
  auto expected = makeRowVector(
      {"a", "b"}, {nullBigint, nullString}, [](vector_size_t row) {
        return row < 2;
      });
  auto input =
      makeFlatVector<std::string>({"", " ", "null", "[]", "1", R"({"a":)"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, nestedStructFailure) {
  auto nested = makeRowVector(
      {"a", "b"},
      {makeNullableFlatVector<int64_t>(
           {std::nullopt, 2, std::nullopt, std::nullopt, std::nullopt}),
       makeNullableFlatVector<int64_t>(
           {std::nullopt, 3, std::nullopt, std::nullopt, std::nullopt})},
      [](vector_size_t row) { return row != 1; });
  auto expected = makeRowVector({"s"}, {nested});
  auto input = makeFlatVector<std::string>(
      {R"({"s":{"a":1,"b":"x"}})",
       R"({"s":{"a":2,"b":3}})",
       R"({"s":null})",
       R"({"s":{"a":"x","a":4,"b":5}})",
       R"({"s":{"a":"x","a":null,"b":6}})"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, duplicateComplexStructFields) {
  auto stringInput = makeFlatVector<std::string>(
      {R"({"a":"x","a":"y"})", R"({"a":"x","a":"y"})", R"({"a":"x","a":"y"})"});
  testFromJson(
      stringInput,
      makeRowVector({"a"}, {makeFlatVector<StringView>({"y", "y", "y"})}));

  auto arrayInput = makeFlatVector<std::string>(
      {R"({"a":[1],"a":[2,3]})",
       R"({"a":[1],"a":[2,3]})",
       R"({"a":[1],"a":[2,3]})"});
  testFromJson(
      arrayInput,
      makeRowVector(
          {"a"}, {makeArrayVector<int64_t>({{2, 3}, {2, 3}, {2, 3}})}));

  auto mapInput = makeFlatVector<std::string>(
      {R"({"a":{"x":1},"a":{"y":2}})",
       R"({"a":{"x":1},"a":{"y":2}})",
       R"({"a":{"x":1},"a":{"y":2}})"});
  auto maps =
      makeMapVector<std::string, int64_t>({{{"y", 2}}, {{"y", 2}}, {{"y", 2}}});
  testFromJson(mapInput, makeRowVector({"a"}, {maps}));
}

TEST_F(FromJsonTest, partialStructsInCollections) {
  auto arrayInput = makeFlatVector<std::string>(
      {R"([{"a":"x","b":1},{"a":2,"b":3}])",
       R"([{"a":"x","b":1},{"a":2,"b":3}])",
       R"([{"a":"x","b":1},{"a":2,"b":3}])"});
  auto arrayRows = makeRowVector(
      {"a", "b"},
      {makeNullableFlatVector<int64_t>(
           {std::nullopt, 2, std::nullopt, 2, std::nullopt, 2}),
       makeFlatVector<int64_t>({1, 3, 1, 3, 1, 3})});
  testFromJson(arrayInput, makeArrayVector({0, 2, 4}, arrayRows));

  auto mapInput = makeFlatVector<std::string>(
      {R"({"k":{"a":"x","b":1},"j":{"a":2,"b":3}})",
       R"({"k":{"a":"x","b":1},"j":{"a":2,"b":3}})",
       R"({"k":{"a":"x","b":1},"j":{"a":2,"b":3}})"});
  auto mapKeys = makeFlatVector<StringView>({"k", "j", "k", "j", "k", "j"});
  auto mapRows = makeRowVector(
      {"a", "b"},
      {makeNullableFlatVector<int64_t>(
           {std::nullopt, 2, std::nullopt, 2, std::nullopt, 2}),
       makeFlatVector<int64_t>({1, 3, 1, 3, 1, 3})});
  testFromJson(mapInput, makeMapVector({0, 2, 4}, mapKeys, mapRows));
}

TEST_F(FromJsonTest, nonObjectStructElementsInvalidateCollections) {
  const auto rowType = ROW("a", INTEGER());
  {
    auto input = makeFlatVector<std::string>(
        {R"([5,{"a":2}])", R"([5,{"a":2}])", R"([5,{"a":2}])"});
    auto expected =
        BaseVector::createNullConstant(ARRAY(rowType), input->size(), pool());
    testFromJson(input, expected);
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"k":5,"j":{"a":1}})",
         R"({"k":5,"j":{"a":1}})",
         R"({"k":5,"j":{"a":1}})"});
    auto expected = BaseVector::createNullConstant(
        MAP(VARCHAR(), rowType), input->size(), pool());
    testFromJson(input, expected);
  }
}

TEST_F(FromJsonTest, partialStructCollectionsInsideStructAreNull) {
  {
    const auto itemType = ROW({{"id", BIGINT()}, {"name", VARCHAR()}});
    auto input = makeFlatVector<std::string>(
        {R"({"items":[{"id":"abc","name":"x"}],"total":5})",
         R"({"items":[{"id":"abc","name":"x"}],"total":5})",
         R"({"items":[{"id":"abc","name":"x"}],"total":5})"});
    auto items =
        BaseVector::createNullConstant(ARRAY(itemType), input->size(), pool());
    testFromJson(
        input,
        makeRowVector(
            {"items", "total"}, {items, makeFlatVector<int64_t>({5, 5, 5})}));
  }
  {
    const auto itemType = ROW("id", BIGINT());
    auto input = makeFlatVector<std::string>(
        {R"({"items":{"k":{"id":"abc"}},"total":5})",
         R"({"items":{"k":{"id":"abc"}},"total":5})",
         R"({"items":{"k":{"id":"abc"}},"total":5})"});
    auto items = BaseVector::createNullConstant(
        MAP(VARCHAR(), itemType), input->size(), pool());
    testFromJson(
        input,
        makeRowVector(
            {"items", "total"}, {items, makeFlatVector<int64_t>({5, 5, 5})}));
  }
}

TEST_F(FromJsonTest, malformedFieldsInvalidateStruct) {
  auto input = makeFlatVector<std::string>(
      {R"({"z":1,"x":tru})", R"({"z":1,"x":[1,,2]})", R"({"z":1,"a":"\q"})"});
  auto nullBigint = makeNullableFlatVector<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  auto nullString = makeNullableFlatVector<StringView>(
      {std::nullopt, std::nullopt, std::nullopt});
  testFromJson(input, makeRowVector({"z", "a"}, {nullBigint, nullString}));
}

TEST_F(FromJsonTest, malformedBooleanAtomsInvalidateStruct) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"b":tru,"t":2})", R"({"b":fals,"t":2})", R"({"b":truex,"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"b", "t"},
            {makeNullableFlatVector<bool>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":tru,"d":1.0,"s":"ok","t":2})",
         R"({"a":1,"d":fals,"s":"ok","t":2})",
         R"({"a":1,"d":1.0,"s":truex,"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "d", "s", "t"},
            {makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeNullableFlatVector<double>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeNullableFlatVector<StringView>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    const auto nestedType = ROW("b", BOOLEAN());
    auto input = makeFlatVector<std::string>(
        {R"({"s":{"b":tru},"t":2})",
         R"({"s":{"b":fals},"t":2})",
         R"({"s":{"b":truex},"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t"},
            {BaseVector::createNullConstant(nestedType, input->size(), pool()),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    const auto nestedType = ROW("z", INTEGER());
    auto input = makeFlatVector<std::string>(
        {R"({"s":{"z":1,"x":tru},"t":2})",
         R"({"s":{"z":1,"x":fals},"t":2})",
         R"({"s":{"z":1,"x":truex},"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t"},
            {BaseVector::createNullConstant(nestedType, input->size(), pool()),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
}

TEST_F(FromJsonTest, booleanStringValues) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":true})", R"({"a":false})", R"({"a":true})"});
    testFromJson(
        input,
        makeRowVector(
            {"a"}, {makeFlatVector<StringView>({"true", "false", "true"})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"(["x",true])", R"([false,"y"])", R"([true,false])"});
    testFromJson(
        input,
        makeArrayVector<std::string>(
            {{"x", "true"}, {"false", "y"}, {"true", "false"}}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"k":true,"j":"x"})",
         R"({"k":false,"j":"y"})",
         R"({"k":true,"j":"z"})"});
    testFromJson(
        input,
        makeMapVector<std::string, std::string>(
            {{{"k", "true"}, {"j", "x"}},
             {{"k", "false"}, {"j", "y"}},
             {{"k", "true"}, {"j", "z"}}}));
  }
}

TEST_F(FromJsonTest, malformedBooleanRecovery) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":[1,tru],"t":2})",
         R"({"a":[tru,1],"t":2})",
         R"({"a":[fals],"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "t"},
            {BaseVector::createNullConstant(
                 ARRAY(INTEGER()), input->size(), pool()),
             makeFlatVector<int32_t>({2, 2, 2})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":[tru],"t":2})",
         R"({"a":[fals],"t":2})",
         R"({"a":[truex],"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "t"},
            {makeNullableFlatVector<StringView>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeFlatVector<int32_t>({2, 2, 2})}));
  }
  {
    const auto nestedType = ROW("b", BOOLEAN());
    auto input = makeFlatVector<std::string>(
        {R"({"s":{"b":tru},"t":2})",
         R"({"t":2,"s":{"b":tru}})",
         R"({"s":{"b":fals},"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t"},
            {BaseVector::createNullConstant(nestedType, input->size(), pool()),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, 2, std::nullopt})}));
  }
  {
    const auto nestedType = ROW({{"a", ARRAY(INTEGER())}, {"b", INTEGER()}});
    auto input = makeFlatVector<std::string>(
        {R"({"s":{"a":[1,tru],"b":2},"t":3})",
         R"({"s":{"a":[tru,1],"b":2},"t":3})",
         R"({"s":{"a":[fals],"b":2},"t":3})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t"},
            {BaseVector::createNullConstant(nestedType, input->size(), pool()),
             makeFlatVector<int32_t>({3, 3, 3})}));
  }
}

TEST_F(FromJsonTest, malformedAtomRecoveryAndObjectTruncation) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":abc,"t":2})", R"({"a":+1,"t":2})", R"({"a":nul,"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "t"},
            {makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    const auto nestedType = ROW("a", INTEGER());
    auto input = makeFlatVector<std::string>(
        {R"({"t":2,"s":{"a":nul}})",
         R"({"s":{"a":nan},"t":2})",
         R"({"t":2,"s":{"a":+1}})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t"},
            {BaseVector::createNullConstant(nestedType, input->size(), pool()),
             makeNullableFlatVector<int32_t>({2, std::nullopt, 2})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"m":{"k":tru},"t":2})",
         R"({"m":{"k":fals},"t":2})",
         R"({"m":{"k":nul},"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"m", "t"},
            {BaseVector::createNullConstant(
                 MAP(VARCHAR(), BOOLEAN()), input->size(), pool()),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    const auto elementType = ROW("b", BOOLEAN());
    auto input = makeFlatVector<std::string>(
        {R"({"a":[{"b":tru}],"t":2})",
         R"({"a":[{"b":fals}],"t":2})",
         R"({"a":[{"b":nul}],"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "t"},
            {BaseVector::createNullConstant(
                 ARRAY(elementType), input->size(), pool()),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":{"x":tru},"t":2})",
         R"({"a":{"x":fals},"t":2})",
         R"({"a":{"x":nul},"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "t"},
            {makeNullableFlatVector<StringView>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
}

TEST_F(FromJsonTest, failedDuplicateStructRetainsPreviousValue) {
  const auto nested =
      makeRowVector({"a"}, {makeFlatVector<int32_t>({1, 1, 1})});
  auto input = makeFlatVector<std::string>(
      {R"({"t":2,"s":{"a":1},"s":{"a":tru},"u":3})",
       R"({"t":2,"s":{"a":1},"s":{"a":fals},"u":3})",
       R"({"t":2,"s":{"a":1},"s":{"a":nul},"u":3})"});
  testFromJson(
      input,
      makeRowVector(
          {"t", "s", "u"},
          {makeFlatVector<int32_t>({2, 2, 2}),
           nested,
           makeNullableFlatVector<int32_t>(
               {std::nullopt, std::nullopt, std::nullopt})}));
}

TEST_F(FromJsonTest, duplicateFieldsUseLastSuccessfulValue) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":[1,2],"a":[3],"m":{"k":1},"m":{"j":2},"i":1,"i":2})",
         R"({"a":[1,2],"a":[3],"m":{"k":1},"m":{"j":2},"i":1,"i":2})",
         R"({"a":[1,2],"a":[3],"m":{"k":1},"m":{"j":2},"i":1,"i":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "m", "i"},
            {makeArrayVector<int32_t>({{3}, {3}, {3}}),
             makeMapVector<std::string, int32_t>(
                 {{{"j", 2}}, {{"j", 2}}, {{"j", 2}}}),
             makeFlatVector<int32_t>({2, 2, 2})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":[1,2],"a":[3,"x"],"m":{"k":1},"m":{"j":"x"},"i":1,"i":"x"})",
         R"({"a":[1,2],"a":[3,"x"],"m":{"k":1},"m":{"j":"x"},"i":1,"i":"x"})",
         R"({"a":[1,2],"a":[3,"x"],"m":{"k":1},"m":{"j":"x"},"i":1,"i":"x"})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "m", "i"},
            {makeArrayVector<int32_t>({{1, 2}, {1, 2}, {1, 2}}),
             makeMapVector<std::string, int32_t>(
                 {{{"k", 1}}, {{"k", 1}}, {{"k", 1}}}),
             makeFlatVector<int32_t>({1, 1, 1})}));
  }
}

TEST_F(FromJsonTest, scalarDuplicateStructRetainsPreviousValue) {
  const auto nested =
      makeRowVector({"a"}, {makeFlatVector<int32_t>({1, 1, 1})});
  auto input = makeFlatVector<std::string>(
      {R"({"s":{"a":1},"s":5,"t":1})",
       R"({"s":{"a":1},"s":"x","t":1})",
       R"({"s":{"a":1},"s":true,"t":1})"});
  testFromJson(
      input,
      makeRowVector({"s", "t"}, {nested, makeFlatVector<int32_t>({1, 1, 1})}));
}

TEST_F(FromJsonTest, failedDuplicateStructWithCollectionsRetainsPreviousValue) {
  {
    auto nested = makeRowVector(
        {"a"}, {makeArrayVector<int32_t>({{1, 2, 3}, {1, 2, 3}, {1, 2, 3}})});
    auto input = makeFlatVector<std::string>(
        {R"({"s":{"a":[1,2,3]},"s":{"a":[4,"x"]},"t":1})",
         R"({"s":{"a":[1,2,3]},"s":{"a":[4,"x"]},"t":1})",
         R"({"s":{"a":[1,2,3]},"s":{"a":[4,"x"]},"t":1})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t"}, {nested, makeFlatVector<int32_t>({1, 1, 1})}));
  }
  {
    auto nested = makeRowVector(
        {"m"},
        {makeMapVector<std::string, int32_t>(
            {{{"k", 1}, {"j", 2}},
             {{"k", 1}, {"j", 2}},
             {{"k", 1}, {"j", 2}}})});
    auto input = makeFlatVector<std::string>(
        {R"({"s":{"m":{"k":1,"j":2}},"s":{"m":{"k":"x"}},"t":1})",
         R"({"s":{"m":{"k":1,"j":2}},"s":{"m":{"k":"x"}},"t":1})",
         R"({"s":{"m":{"k":1,"j":2}},"s":{"m":{"k":"x"}},"t":1})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t"}, {nested, makeFlatVector<int32_t>({1, 1, 1})}));
  }
  {
    auto nested = makeRowVector(
        {"a", "b"},
        {makeArrayVector<int32_t>({{1, 2}, {1, 2}, {1, 2}}),
         makeFlatVector<int32_t>({1, 1, 1})});
    auto input = makeFlatVector<std::string>(
        {R"({"s":{"a":[1,2],"b":1},"s":{"a":[3],"b":tru},"t":1})",
         R"({"s":{"a":[1,2],"b":1},"s":{"a":[3],"b":fals},"t":1})",
         R"({"s":{"a":[1,2],"b":1},"s":{"a":[3],"b":nul},"t":1})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t"},
            {nested,
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
}

TEST_F(FromJsonTest, malformedStringAndContainerRecovery) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"c":false,"b":"\q"})",
         R"({"c":true,"b":"\q"})",
         R"({"c":false,"b":"\q"})"});
    testFromJson(
        input,
        makeRowVector(
            {"c", "b"},
            {makeNullableFlatVector<bool>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeNullableFlatVector<bool>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    const auto nestedType = ROW("a", VARCHAR());
    auto input = makeFlatVector<std::string>(
        {R"({"t":2,"s":{"a":"\q"},"u":3})",
         R"({"t":2,"s":{"a":"\q"},"u":3})",
         R"({"t":2,"s":{"a":"\q"},"u":3})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t", "u"},
            {BaseVector::createNullConstant(nestedType, input->size(), pool()),
             makeFlatVector<int32_t>({2, 2, 2}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    const auto nestedType = ROW("z", INTEGER());
    auto input = makeFlatVector<std::string>(
        {R"({"t":2,"s":{"x":[1,tru],"z":5},"u":3})",
         R"({"t":2,"s":{"x":[1,fals],"z":5},"u":3})",
         R"({"t":2,"s":{"x":[1,nul],"z":5},"u":3})"});
    testFromJson(
        input,
        makeRowVector(
            {"s", "t", "u"},
            {BaseVector::createNullConstant(nestedType, input->size(), pool()),
             makeFlatVector<int32_t>({2, 2, 2}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":{"k":1e2,"a":-128,"A":True},"b":"hello"})",
         R"({"a":{"k":1e2,"a":-128,"A":False},"b":"hello"})",
         R"({"a":{"k":1e2,"a":-128,"A":nul},"b":"hello"})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "b"},
            {BaseVector::createNullConstant(
                 MAP(VARCHAR(), SMALLINT()), input->size(), pool()),
             makeNullableFlatVector<StringView>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":[1.5,True],"b":"hello"})",
         R"({"a":[1.5,False],"b":"hello"})",
         R"({"a":[1.5,nul],"b":"hello"})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "b"},
            {BaseVector::createNullConstant(
                 ARRAY(INTEGER()), input->size(), pool()),
             makeNullableFlatVector<StringView>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
}

TEST_F(FromJsonTest, specialNumbersAreConversionErrors) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":-INF,"b":1})",
         R"({"a":-Infinity,"b":1})",
         R"({"a":-INF,"b":1})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "b"},
            {makeNullableFlatVector<int64_t>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeFlatVector<int32_t>({1, 1, 1})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":-INF,"b":1})",
         R"({"a":-Infinity,"b":1})",
         R"({"a":-INF,"b":1})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "b"},
            {makeNullableFlatVector<int64_t>(
                 {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(7, 2)),
             makeFlatVector<int32_t>({1, 1, 1})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":[-INF],"b":1})",
         R"({"a":[-Infinity],"b":1})",
         R"({"a":[-INF],"b":1})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "b"},
            {BaseVector::createNullConstant(
                 ARRAY(INTEGER()), input->size(), pool()),
             makeFlatVector<int32_t>({1, 1, 1})}));
  }
}

TEST_F(FromJsonTest, duplicateStructBuffersStayBounded) {
  std::vector<std::string> json(
      40, R"({"s":{"a":[1],"m":{"x":1}},"s":{"a":[2,3],"m":{"y":2,"z":3}}})");
  auto input = makeFlatVector<std::string>(json);
  const auto outputType = ROW(
      "s", ROW({{"a", ARRAY(INTEGER())}, {"m", MAP(VARCHAR(), INTEGER())}}));
  auto expectedNested = makeRowVector(
      {"a", "m"},
      {makeArrayVector<int32_t>(std::vector<std::vector<int32_t>>(40, {2, 3})),
       makeMapVector<std::string, int32_t>(
           std::vector<
               std::vector<std::pair<std::string, std::optional<int32_t>>>>(
               40, {{"y", 2}, {"z", 3}}))});
  testFromJson(input, makeRowVector({"s"}, {expectedNested}));

  auto result = evaluateFromJson(input, outputType);
  result->validate();

  const auto* rows = result->as<RowVector>();
  const auto* nestedRows = rows->childAt(0)->as<RowVector>();
  const auto* arrays = nestedRows->childAt(0)->as<ArrayVector>();
  const auto* maps = nestedRows->childAt(1)->as<MapVector>();
  EXPECT_LE(arrays->elements()->size(), 256);
  EXPECT_LE(maps->mapValues()->size(), 256);
}

TEST_F(FromJsonTest, deeplyNestedDuplicateStructCopy) {
  TypePtr outputType = ROW("a", INTEGER());
  std::string json = R"({"a":1})";
  for (int32_t i = 0; i < 20; ++i) {
    outputType = ROW("s", outputType);
    json.insert(0, R"({"s":{},"s":)");
    json.push_back('}');
  }

  auto input = makeFlatVector<std::string>({json, json, json});
  auto result = evaluateFromJson(input, outputType);
  result->validate();
  EXPECT_FALSE(result->isNullAt(0));
}

TEST_F(FromJsonTest, duplicateStructCopyUsesNormalWriterCasts) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"s":{"a":null},"s":{"a":[1,2]}})",
         R"({"s":{"a":[3]}})",
         R"({"s":{"a":[]},"s":{"a":[4]}})"});
    auto nested =
        makeRowVector({"a"}, {makeArrayVector<int32_t>({{1, 2}, {3}, {4}})});
    testFromJson(input, makeRowVector({"s"}, {nested}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"s":{},"s":{"m":{"k":1}}})",
         R"({"s":{"m":{"j":2}}})",
         R"({"s":{"m":null},"s":{"m":{"z":3}}})"});
    auto nested = makeRowVector(
        {"m"},
        {makeMapVector<std::string, int32_t>(
            {{{"k", 1}}, {{"j", 2}}, {{"z", 3}}})});
    testFromJson(input, makeRowVector({"s"}, {nested}));
  }
}

TEST_F(FromJsonTest, failedNestedDuplicateStructNullsEnclosingStruct) {
  {
    const auto outerType = ROW("s", ROW("a", INTEGER()));
    auto input = makeFlatVector<std::string>(
        {R"({"o":{"s":{"a":1},"s":{"a":"x"}},"t":2})",
         R"({"o":{"s":{"a":1},"s":{"a":"x"}},"t":2})",
         R"({"o":{"s":{"a":1},"s":{"a":"x"}},"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"o", "t"},
            {BaseVector::createNullConstant(outerType, input->size(), pool()),
             makeFlatVector<int32_t>({2, 2, 2})}));
  }
  {
    const auto outerType = ROW("s", ROW("a", ARRAY(INTEGER())));
    auto input = makeFlatVector<std::string>(
        {R"({"o":{"s":{"a":[1]},"s":{"a":[2,"x"]}},"t":2})",
         R"({"o":{"s":{"a":[1]},"s":{"a":[2,"x"]}},"t":2})",
         R"({"o":{"s":{"a":[1]},"s":{"a":[2,"x"]}},"t":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"o", "t"},
            {BaseVector::createNullConstant(outerType, input->size(), pool()),
             makeFlatVector<int32_t>({2, 2, 2})}));
  }
}

TEST_F(FromJsonTest, arrayPostConversionMalformedTokensInvalidateRoot) {
  auto input = makeFlatVector<std::string>(
      {R"({"t":2,"a":[1,"x",tru],"u":3})",
       R"({"t":2,"a":[1,"x","\q"],"u":3})",
       R"({"t":2,"a":[1,"x",nul],"u":3})"});
  testFromJson(
      input,
      makeRowVector(
          {"t", "a", "u"},
          {makeNullableFlatVector<int32_t>(
               {std::nullopt, std::nullopt, std::nullopt}),
           BaseVector::createNullConstant(
               ARRAY(INTEGER()), input->size(), pool()),
           makeNullableFlatVector<int32_t>(
               {std::nullopt, std::nullopt, std::nullopt})}));
}

TEST_F(FromJsonTest, nestedArrayMalformedTokensInvalidateRoot) {
  auto input = makeFlatVector<std::string>(
      {R"({"t":2,"a":[[1,"x",tru]],"u":3})",
       R"({"t":2,"a":[[1,"x",fals]],"u":3})",
       R"({"t":2,"a":[[1,"x",nul]],"u":3})"});
  testFromJson(
      input,
      makeRowVector(
          {"t", "a", "u"},
          {makeNullableFlatVector<int32_t>(
               {std::nullopt, std::nullopt, std::nullopt}),
           BaseVector::createNullConstant(
               ARRAY(ARRAY(INTEGER())), input->size(), pool()),
           makeNullableFlatVector<int32_t>(
               {std::nullopt, std::nullopt, std::nullopt})}));
}

TEST_F(FromJsonTest, malformedArrayStringInvalidatesEnclosingObject) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"t":2,"b":["x","\q","y"],"u":3})",
         R"({"t":2,"b":["x","\q","y"],"u":3})",
         R"({"t":2,"b":["x","\q","y"],"u":3})"});
    testFromJson(
        input,
        makeRowVector(
            {"t", "b", "u"},
            {makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             BaseVector::createNullConstant(
                 ARRAY(VARCHAR()), input->size(), pool()),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
  {
    const auto nestedType = ROW("b", ARRAY(INTEGER()));
    auto input = makeFlatVector<std::string>(
        {R"({"t":2,"s":{"b":[1,"\q"]},"u":3})",
         R"({"t":2,"s":{"b":[1,"\q"]},"u":3})",
         R"({"t":2,"s":{"b":[1,"\q"]},"u":3})"});
    testFromJson(
        input,
        makeRowVector(
            {"t", "s", "u"},
            {makeFlatVector<int32_t>({2, 2, 2}),
             BaseVector::createNullConstant(nestedType, input->size(), pool()),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt, std::nullopt, std::nullopt})}));
  }
}

TEST_F(FromJsonTest, malformedScalarInContainerArrayIsRecoverable) {
  const auto elementType = ROW("c", BOOLEAN());
  auto input = makeFlatVector<std::string>(
      {R"({"a":1.5,"c":[{},+1,{"c":false}]})",
       R"({"a":1.5,"c":[{},truex,{"c":false}]})",
       R"({"a":1.5,"c":[{},tru,{"c":false}]})"});
  testFromJson(
      input,
      makeRowVector(
          {"a", "c"},
          {makeFlatVector<double>({1.5, 1.5, 1.5}),
           BaseVector::createNullConstant(
               ARRAY(elementType), input->size(), pool())}));
}

TEST_F(FromJsonTest, malformedStringInMapTruncatesCurrentField) {
  auto input = makeFlatVector<std::string>(
      {R"({"d":"","b":{"k":"\q","j":1e10}})",
       R"({"d":"","b":{"k":"\q","j":1e10}})",
       R"({"d":"","b":{"k":"\q","j":1e10}})"});
  testFromJson(
      input,
      makeRowVector(
          {"d", "b"},
          {makeFlatVector<std::string>({"", "", ""}),
           BaseVector::createNullConstant(
               MAP(VARCHAR(), DOUBLE()), input->size(), pool())}));
}

TEST_F(FromJsonTest, rootMapDirectMalformedStructValueIsNull) {
  auto input = makeFlatVector<std::string>(
      {R"({"A":{"zz":tru,"d":-128,"e":255}})",
       R"({"A":{"d":1,"zz":tru}})",
       R"({"A":{"d":1,"zz":[1,tru]}})"});
  const auto outputType =
      MAP(VARCHAR(), ROW({{"d", INTEGER()}, {"e", INTEGER()}}));
  testFromJson(
      input, BaseVector::createNullConstant(outputType, input->size(), pool()));
}

TEST_F(FromJsonTest, failedComplexStructFieldDoesNotLeak) {
  auto elements = makeFlatVector<int32_t>({2, 3});
  auto arrays = makeArrayVector({0, 0, 1}, elements, {0});
  auto expected = makeRowVector({"a"}, {arrays});
  auto input = makeFlatVector<std::string>(
      {R"({"a":[1,"x"]})", R"({"a":[2]})", R"({"a":[3]})"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, basicArray) {
  auto expected = makeArrayVector<int64_t>({{1}, {2}, {}});
  auto input = makeFlatVector<std::string>({R"([1])", R"([2])", R"([])"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, arrayOfRowWithScalarInput) {
  auto input = makeFlatVector<std::string>(
      {R"(746314092223)", R"(true)", R"("invalid")"});
  const auto outputType = ARRAY(ROW({"text", "type"}, {VARCHAR(), VARCHAR()}));
  auto expected =
      BaseVector::createNullConstant(outputType, input->size(), pool());
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, arrayOfRowWithPartialObjectInput) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":"x"})", R"({"a":"x"})", R"({"a":"x"})"});
  const auto outputType = ARRAY(ROW("a", INTEGER()));
  auto expected =
      BaseVector::createNullConstant(outputType, input->size(), pool());
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, basicMap) {
  auto expected = makeMapVector<std::string, int64_t>(
      {{{"a", 1}}, {{"b", 2}}, {{"c", 3}}, {{"3", 3}}});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})", R"({"b": 2})", R"({"c": 3})", R"({"3": 3})"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, singleQuotedJson) {
  auto expected = makeMapVector<std::string, std::string>(
      {{{"a", "1"}},
       {{"a", "two"}},
       {{"a", "it's"}},
       {{"a", "it's"}},
       {{"a", "he said \"hi\""}}});
  auto input = makeFlatVector<std::string>(
      {R"({'a': '1'})",
       R"({"a": 'two'})",
       R"({'a': 'it\'s'})",
       R"({"a": "it\'s"})",
       R"({'a': 'he said "hi"'})"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, invalidUtf8) {
  const auto invalidJson = std::string(R"({"before)") + "\x80" + R"(after":1})";
  const auto repairedKey = std::string("before") + "\xef\xbf\xbd" + "after";
  auto expected = makeMapVector<std::string, int64_t>(
      {{{repairedKey, 1}}, {{repairedKey, 1}}, {{"valid", 2}}});
  auto input =
      makeFlatVector<std::string>({invalidJson, invalidJson, R"({"valid":2})"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, documentFraming) {
  auto expected = makeRowVector({"a"}, {makeFlatVector<int64_t>({1, 2, 4, 1})});
  auto input = makeFlatVector<std::string>(
      {R"({"a":1} trailing)",
       R"({"a":2}{"a":3})",
       R"({"a":4} "unterminated)",
       R"({'a':1} ')"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, byteOrderMark) {
  const auto byteOrderMark = std::string("\xef\xbb\xbf");
  auto input = makeFlatVector<std::string>(
      {byteOrderMark + R"([1])",
       byteOrderMark + R"([2])",
       byteOrderMark + R"([])",
       " \t" + byteOrderMark + R"([3])"});
  auto expected =
      BaseVector::createNullConstant(ARRAY(INTEGER()), input->size(), pool());
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, numericTrailingWhitespace) {
  auto expected = makeMapVector<std::string, double>(
      {{{"a", kInfDouble}}, {{"a", -kInfDouble}}, {{"a", kInfDouble}}});
  auto input = makeFlatVector<std::string>(
      {R"({"a":1e400 })", "{\"a\":-1e400\n}", "{\"a\":+INF\t}"});
  testFromJson(input, expected);
}

TEST_F(FromJsonTest, basicBool) {
  auto expected = makeNullableFlatVector<bool>(
      {true, false, std::nullopt, std::nullopt, std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": true})",
       R"({"a": false})",
       R"({"a": 1})",
       R"({"a": 0.0})",
       R"({"a": "true"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicTinyInt) {
  auto expected = makeNullableFlatVector<int8_t>(
      {1,
       std::nullopt,
       -128,
       -56,
       -1,
       std::nullopt,
       std::nullopt,
       std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": -129})",
       R"({"a": 128})",
       R"({"a": 200})",
       R"({"a": 255})",
       R"({"a": 256})",
       R"({"a": 1.0})",
       R"({"a": "1"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicSmallInt) {
  auto expected = makeNullableFlatVector<int16_t>(
      {1, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": -32769})",
       R"({"a": 32768})",
       R"({"a": 1.0})",
       R"({"a": "1"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicInt) {
  auto expected = makeNullableFlatVector<int32_t>(
      {1, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": -2147483649})",
       R"({"a": 2147483648})",
       R"({"a": 2.0})",
       R"({"a": "3"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicBigInt) {
  auto expected =
      makeNullableFlatVector<int32_t>({1, std::nullopt, std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})", R"({"a": 2.0})", R"({"a": "3"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicFloat) {
  auto expected = makeNullableFlatVector<float>(
      {1.0,          2.0,          -3.4028235E38, 3.4028235E38, -kInfFloat,
       kInfFloat,    0.0,          0.0,           std::nullopt, std::nullopt,
       std::nullopt, std::nullopt, std::nullopt,  std::nullopt, std::nullopt,
       std::nullopt, kNaNFloat,    kNaNFloat,     -kInfFloat,   -kInfFloat,
       -kInfFloat,   -kInfFloat,   kInfFloat,     kInfFloat,    kInfFloat,
       kInfFloat,    kInfFloat,    kInfFloat});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 2.0})",
       R"({"a": -3.4028235E38})", // Min float value
       R"({"a": 3.4028235E38})", // Max float value
       R"({"a": -3.4028235E39})",
       R"({"a": 3.4028235E39})",
       R"({"a": 0})",
       R"({"a": 1.0e-200})",
       R"({"a": "3"})",
       R"({"a": 1.1.0})", // Multiple decimal points.
       R"({"a": 1.})", // Missing fraction digits after a decimal point.
       R"({"a": 01})", // Leading zero.
       R"({"a": 1e})", // Missing exponent digits after ‘e’ or ‘E’.
       R"({"a": 1e+})", // Missing exponent digits after ‘e’ or ‘E’.
       R"({"a": .e10})", // Missing digits.
       R"({"a": -.})", // Missing digits entirely.
       R"({"a": "NaN"})",
       R"({"a": NaN})",
       R"({"a": "-Infinity"})",
       R"({"a": -Infinity})",
       R"({"a": "-INF"})",
       R"({"a": -INF})",
       R"({"a": "+Infinity"})",
       R"({"a": +Infinity})",
       R"({"a": "Infinity"})",
       R"({"a": Infinity})",
       R"({"a": "+INF"})",
       R"({"a": +INF})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, floatDirectRounding) {
  auto expected = makeFlatVector<float>({1.0000001, 1.0000001, 1.0000001});
  auto input = makeFlatVector<std::string>(
      {R"({"a":1.000000059604644775390625000001})",
       R"({"a":1.000000059604644775390625000001})",
       R"({"a":1.000000059604644775390625000001})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicDouble) {
  auto expected = makeNullableFlatVector<double>(
      {1.0,
       2.0,
       -1.7976931348623158e+308,
       1.7976931348623158e+308,
       -kInfDouble,
       kInfDouble,
       0.0,
       0.0,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       kNaNDouble,
       kNaNDouble,
       -kInfDouble,
       -kInfDouble,
       -kInfDouble,
       -kInfDouble,
       kInfDouble,
       kInfDouble,
       kInfDouble,
       kInfDouble,
       kInfDouble,
       kInfDouble});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 2.0})",
       R"({"a": -1.7976931348623158e+308})", // Min double value
       R"({"a": 1.7976931348623158e+308})", // Max double value
       R"({"a": -1.7976931348623158e+309})",
       R"({"a": 1.7976931348623158e+309})",
       R"({"a": 0})",
       R"({"a": 1.0e-2000})",
       R"({"a": "3"})",
       R"({"a": 1.1.0})", // Multiple decimal points.
       R"({"a": 1.})", // Missing fraction digits after a decimal point.
       R"({"a": 01})", // Leading zero.
       R"({"a": 1e})", // Missing exponent digits after ‘e’ or ‘E’.
       R"({"a": 1e+})", // Missing exponent digits after ‘e’ or ‘E’.
       R"({"a": .e10})", // Missing digits.
       R"({"a": -.})", // Missing digits entirely.
       R"({"a": "NaN"})",
       R"({"a": NaN})",
       R"({"a": "-Infinity"})",
       R"({"a": -Infinity})",
       R"({"a": "-INF"})",
       R"({"a": -INF})",
       R"({"a": "+Infinity"})",
       R"({"a": +Infinity})",
       R"({"a": "Infinity"})",
       R"({"a": Infinity})",
       R"({"a": "+INF"})",
       R"({"a": +INF})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, malformedFloatingPointNumbers) {
  auto expected = makeNullableFlatVector<double>(
      {std::nullopt, std::nullopt, std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": .5})", R"({"a": +1.5})", R"({"a": -01})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, conversionErrorsPreserveSiblingFields) {
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":"1.5","b":2})", R"({"a":"abc","b":2})", R"({"a":"","b":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "b"},
            {makeNullableFlatVector<double>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeFlatVector<int32_t>({2, 2, 2})}));
  }
  {
    auto input = makeFlatVector<std::string>(
        {R"({"a":123456789012345678901234567890,"b":2})",
         R"({"a":123456789012345678901234567890,"b":2})",
         R"({"a":123456789012345678901234567890,"b":2})"});
    testFromJson(
        input,
        makeRowVector(
            {"a", "b"},
            {makeNullableFlatVector<int64_t>(
                 {std::nullopt, std::nullopt, std::nullopt}),
             makeFlatVector<int32_t>({2, 2, 2})}));
  }
}

TEST_F(FromJsonTest, malformedDecimalNumbers) {
  auto expected = makeNullableFlatVector<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(7, 2));
  auto input = makeFlatVector<std::string>(
      {R"({"a": .5})", R"({"a": +1.5})", R"({"a": -01})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicDate) {
  auto expected = makeNullableFlatVector<int32_t>(
      {18809,
       18809,
       18809,
       0,
       18809,
       18809,
       -713975,
       15,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt},
      DATE());
  auto input = makeFlatVector<std::string>(
      {R"({"a": "2021-07-01T"})",
       R"({"a": "2021-07-01"})",
       R"({"a": "2021-7"})",
       R"({"a": "1970"})",
       R"({"a": "2021-07-01T00:00GMT+01:00"})",
       R"({"a": "2021-7-GMTGMT1"})",
       R"({"a": "0015-03-16T123123"})",
       R"({"a": "015"})",
       R"({"a": "15.0"})",
       R"({"a": "AA"})",
       R"({"a": "1999-08 01"})",
       R"({"a": "2020/12/1"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, extendedDateYearsAndWhitespace) {
  auto expected = makeNullableFlatVector<int32_t>(
      {3'789'391,
       -365'962'028,
       18'809,
       -354'285,
       999,
       -999,
       20'210'701,
       -12'345'437,
       -10'033'152,
       std::nullopt,
       std::nullopt,
       12,
       18'809,
       2'021,
       std::nullopt},
      DATE());
  auto input = makeFlatVector<std::string>(
      {R"({"a":"12345"})",
       R"({"a":"-1000000"})",
       R"({"a":"2021-07-01 "})",
       R"({"a":"1000"})",
       R"({"a":"999"})",
       R"({"a":"-999"})",
       R"({"a":"20210701"})",
       R"({"a":"-12345678"})",
       R"({"a":"-10033346"})",
       R"({"a":" 12"})",
       R"({"a":"12 "})",
       R"({"a":"١٢"})",
       "{\"a\":\"2021-07-01\\t\"}",
       R"({"a":"٢٠٢١"})",
       R"({"a":"٢٠٢١-٠٧-٠١"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicShortDecimal) {
  auto expected = makeNullableFlatVector<int64_t>(
      {53'210,
       100'050,
       12'300,
       12'300,
       std::nullopt,
       150,
       -100,
       std::nullopt,
       std::nullopt},
      DECIMAL(7, 2));
  auto input = makeFlatVector<std::string>(
      {R"({"a": "5.321E2"})",
       R"({"a": "1,000.50"})",
       R"({"a": ",,,123"})",
       R"({"a": "1,2,3"})",
       R"({"a": ","})",
       R"({"a": 1.5 })",
       R"({"a": -1})",
       R"({"a": 55555555555555555555.5555})",
       R"({"a": "+1BD"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, unicodeDecimalDigits) {
  auto expected = makeNullableFlatVector<int64_t>(
      {12'345, 12'345, 50, std::nullopt, 101, std::nullopt}, DECIMAL(7, 2));
  auto input = makeFlatVector<std::string>(
      {R"({"a":"١٢٣.٤٥"})",
       R"({"a":"１２３.４５"})",
       R"({"a":".5"})",
       R"({"a":" 1.5"})",
       R"({"a":1.005})",
       R"({"a":"𝟏𝟐"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicLongDecimal) {
  auto expected = makeNullableFlatVector<int128_t>(
      {53210000,
       -100000,
       HugeInt::build(0xffff, 0xffffffffffffffff),
       std::nullopt},
      DECIMAL(38, 5));
  auto input = makeFlatVector<std::string>(
      {R"({"a": "5.321E2"})",
       R"({"a": -1})",
       R"({"a": 12089258196146291747.06175})",
       R"({"a": "+1BD"})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, basicString) {
  auto expected = makeNullableFlatVector<StringView>(
      {"1",
       "2.0",
       "true",
       R"({"b":"test"})",
       R"([1,2.0,"a",{"x":true}])",
       "0",
       "100.0",
       R"("Infinity")",
       R"("NaN")",
       std::nullopt});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})",
       R"({"a": 2.00})",
       R"({"a": "true"})",
       R"({"a": {"b": "test"}})",
       R"({"a": [1, 2.00, "\u0061", {"x" : true}]})",
       R"({"a": -0})",
       R"({"a": 1e2})",
       R"({"a": 1e400})",
       R"({"a": NaN})",
       R"({"a": null})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, stringNumericRendering) {
  auto expected = makeFlatVector<StringView>(
      {"1.0E-7",
       "1.0E7",
       "4.9E-324",
       "2.2250738585072014E-308",
       "2.82879384806159008E17",
       "2.13777331479296358E18",
       "5.3680830904980979E17",
       R"({"b":"\u001F"})"});
  auto input = makeFlatVector<std::string>(
      {R"({"a":1e-7})",
       R"({"a":1e7})",
       R"({"a":4.9e-324})",
       R"({"a":2.2250738585072014e-308})",
       R"({"a":2.82879384806159E17})",
       R"({"a":2.1377733147929636e18})",
       R"({"a":5.368083090498098e17})",
       R"({"a":{"b":"\u001f"}})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, supplementaryCharactersInNestedStrings) {
  auto expected = makeFlatVector<StringView>(
      {R"({"b":"\uD83D\uDE00"})",
       R"({"b":"\uD83D\uDE00"})",
       R"({"b":"\uD83D\uDE00"})"});
  auto input = makeFlatVector<std::string>(
      {R"({"a":{"b":"😀"}})",
       R"({"a":{"b":"\uD83D\uDE00"}})",
       R"({"a":{"b":"😀"}})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, javaDoubleRenderingEdgeCases) {
  // Java's Double.toString is not shortest-round-trip, and these are the four
  // families where it prints something else: an integer whose exact digits run
  // past the precision of the double, an exact power of two, a decimal exactly
  // halfway between two doubles, and the binade where the JDK's own 64-bit
  // arithmetic overflows.
  auto expected = makeFlatVector<StringView>({
      "2.82879384806159008E17",
      "4.6116860184273879E18",
      "7.1202363472230444E-307",
      "4.9999999999999996E22",
      "9.999999999999999E22",
      "4.729999999999999E21",
      "4.9E-323",
      "3.2329848831414594E25",
  });
  auto input = makeFlatVector<std::string>({
      R"({"a":2.82879384806159E17})", // Integer, 2^57 <= value < 2^58.
      R"({"a":4.611686018427388E18})", // Integer, exactly 2^62.
      R"({"a":7.1202363472230444E-307})", // Exactly 2^-1018.
      R"({"a":5e22})", // Halfway between two doubles.
      R"({"a":1e23})", // Halfway between two doubles.
      R"({"a":473e19})", // Halfway between two doubles.
      R"({"a":5e-323})", // Subnormal, halfway between two doubles.
      R"({"a":3.2329848831414595E25})", // In [2^84, 2^85).
  });
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, malformedNullTokenInStringValue) {
  auto input = makeFlatVector<std::string>(
      {R"({"a":{"b":nul}})", R"({"a":{"b":nullx}})", R"({"a":{"b":n}})"});
  auto expected = makeNullableFlatVector<StringView>(
      {std::nullopt, std::nullopt, std::nullopt});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, loneSurrogateStrings) {
  auto stringInput = makeFlatVector<std::string>(
      {R"({"a":"\uD800"})", R"({"a":{"x":"\uD800"}})", R"({"a":["\uD800"]})"});
  auto expectedString =
      makeFlatVector<StringView>({"?", R"({"x":"\uD800"})", R"(["\uD800"])"});
  testFromJson(stringInput, makeRowVector({"a"}, {expectedString}));

  auto mapInput = makeFlatVector<std::string>(
      {R"({"\uD800":1})", R"({"\uD800":1})", R"({"\uD800":1})"});
  auto expectedMap =
      makeMapVector<std::string, int64_t>({{{"?", 1}}, {{"?", 1}}, {{"?", 1}}});
  testFromJson(mapInput, expectedMap);

  auto rowInput = makeFlatVector<std::string>(
      {R"({"\uD800":1})", R"({"\uD800":1})", R"({"\uD800":1})"});
  auto expectedRow = makeRowVector(
      {"?"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt})});
  testFromJson(rowInput, expectedRow);
}

TEST_F(FromJsonTest, nestedComplexType) {
  // ARRAY(ROW(BIGINT))
  std::vector<vector_size_t> offsets;
  offsets.push_back(0);
  offsets.push_back(1);
  offsets.push_back(2);
  auto arrayVector = makeArrayVector(
      offsets, makeRowVector({"a"}, {makeFlatVector<int64_t>({1, 2, 2})}));
  auto input = makeFlatVector<std::string>(
      {R"({"a": 1})", R"([{"a": 2}])", R"([{"a": 2}])"});
  testFromJson(input, arrayVector);

  // MAP(ARRAY(ROW(BIGINT, INTEGER)))
  auto keyVector = makeFlatVector<StringView>({"a", "b", "c"});
  auto valueVector = makeArrayVector(
      offsets,
      makeRowVector(
          {"d", "e"},
          {makeFlatVector<int64_t>({1, 2, 3}),
           makeNullableFlatVector<int32_t>({3, 4, std::nullopt})}));
  auto mapVector = makeMapVector(offsets, keyVector, valueVector);
  auto mapInput = makeFlatVector<std::string>(
      {R"({"a": [{"d": 1, "e": 3}]})",
       R"({"b": [{"d": 2, "e": 4}]})",
       R"({"c": [{"d": 3}]})"});
  testFromJson(mapInput, mapVector);

  // ROW(ROW(ROW(BIGINT, INTEGER)))
  auto rowVector = makeRowVector(
      {"a"},
      {makeRowVector(
          {"b"},
          {makeRowVector(
              {"d1", "e1"},
              {makeNullableFlatVector<int64_t>({1, std::nullopt, 3}),
               makeNullableFlatVector<int32_t>({4, 4, std::nullopt})})})});
  auto rowInput = makeFlatVector<std::string>(
      {R"({"a": {"b": {"d1": 1, "e1": 3, "e1": 4}}})", // Last duplicate wins.
       R"({"a": {"b": {"D1": 2, "e1": 4}}})", // Field names match exactly.
       R"({"a": {"b": {"d1": 3, "f3": 3}}})"}); // Key not in schema.
  testFromJson(rowInput, rowVector);

  // ROW(ARRAY[BIGINT], BIGINT)
  std::vector<vector_size_t> offsets1;
  offsets1.push_back(0);
  offsets1.push_back(1);
  offsets1.push_back(3);
  auto arrayVector1 =
      makeArrayVector(offsets1, makeFlatVector<int64_t>({1, 2, 2, 3, 3, 3}));
  auto rowVector1 = makeRowVector(
      {"a", "b"}, {arrayVector1, makeFlatVector<int64_t>({1, 2, 3})});
  auto rowInput1 = makeFlatVector<std::string>(
      {R"({"a": [1], "b": 1})",
       R"({"a": [2, 2], "b": 2})",
       R"({"a": [3, 3, 3], "b": 3})"});
  testFromJson(rowInput1, rowVector1);

  // ROW(ROW(BIGINT, ARRAY[BIGINT]), ARRAY[BIGINT])
  auto rowVector2 = makeRowVector(
      {"a", "b"},
      {makeRowVector(
           {"c", "d"}, {makeFlatVector<int64_t>({1, 3, 4}), arrayVector1}),
       arrayVector1});
  auto rowInput2 = makeFlatVector<std::string>(
      {R"({"a": {"c": 1, "d": [1]}, "b": [1]})",
       R"({"a": {"c": 3, "d": [2, 2]}, "b": [2, 2]})",
       R"({"a": {"c": 4, "d": [3, 3, 3]}, "b": [3, 3, 3]})"});
  testFromJson(rowInput2, rowVector2);

  // ROW(ROW(ROW(BIGINT)), ROW(ROW(BIGINT, BIGINT)))
  auto rowVector3 = makeRowVector(
      {"a", "d"},
      {makeRowVector(
           {"b"}, {makeRowVector({"c"}, {makeFlatVector<int64_t>({1, 2, 3})})}),
       makeRowVector(
           {"e"},
           {makeRowVector(
               {"f", "g"},
               {makeFlatVector<int64_t>({1, 2, 3}),
                makeFlatVector<int64_t>({4, 5, 6})})})});
  auto rowInput3 = makeFlatVector<std::string>(
      {R"({"a": {"b": {"c": 1}}, "d": {"e": {"f": 1, "g": 4}}})",
       R"({"a": {"b": {"c": 2}}, "d": {"e": {"f": 2, "g": 5}}})",
       R"({"a": {"b": {"c": 3}}, "d": {"e": {"f": 3, "g": 6}}})"});
  testFromJson(rowInput3, rowVector3);

  // ROW(ARRAY[ROW(BIGINT)], ARRAY[ROW(BIGINT, BIGINT)])
  std::vector<vector_size_t> offsets2;
  offsets2.push_back(0);
  offsets2.push_back(1);
  offsets2.push_back(2);
  auto arrayVector3 = makeArrayVector(
      offsets2,
      makeRowVector({"c"}, {makeFlatVector<int64_t>({1, 2, 3, 4, 5})}));
  std::vector<vector_size_t> offsets3;
  offsets3.push_back(0);
  offsets3.push_back(1);
  offsets3.push_back(3);
  auto arrayVector4 = makeArrayVector(
      offsets3,
      makeRowVector(
          {"d", "e"},
          {makeFlatVector<int64_t>({3, 2, 2, 1}),
           makeFlatVector<int64_t>({7, 4, 8, 9})}));
  auto rowVector4 = makeRowVector({"a", "b"}, {arrayVector3, arrayVector4});
  auto rowInput4 = makeFlatVector<std::string>(
      {R"({"a": [{"c": 1}], "b": [{"d": 3, "e": 7}]})",
       R"({"a": [{"c": 2}], "b": [{"d": 2, "e": 4}, {"d": 2, "e": 8}]})",
       R"({"a": [{"c": 3}, {"c": 4}, {"c": 5}], "b": [{"d": 1, "e": 9}]})"});
  testFromJson(rowInput4, rowVector4);
}

TEST_F(FromJsonTest, structEmptyArray) {
  auto expected = makeNullableFlatVector<int64_t>({std::nullopt, 2, 3});
  auto input =
      makeFlatVector<std::string>({R"([])", R"({"a": 2})", R"({"a": 3})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, structEmptyStruct) {
  auto expected = makeNullableFlatVector<int64_t>({std::nullopt, 2, 3});
  auto input =
      makeFlatVector<std::string>({R"({ })", R"({"a": 2})", R"({"a": 3})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, structWrongSchema) {
  auto expected = makeNullableFlatVector<int64_t>({std::nullopt, 2, 3});
  auto input = makeFlatVector<std::string>(
      {R"({"b": 2})", R"({"a": 2})", R"({"a": 3})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, structWrongData) {
  auto expected = makeNullableFlatVector<int64_t>({std::nullopt, 2, 3});
  auto input = makeFlatVector<std::string>(
      {R"({"a": 2.1})", R"({"a": 2})", R"({"a": 3})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

TEST_F(FromJsonTest, invalidType) {
  auto primitiveTypeOutput = makeFlatVector<int64_t>({2, 2, 3});
  auto mapOutput =
      makeMapVector<int64_t, int64_t>({{{1, 1}}, {{2, 2}}, {{3, 3}}});
  auto input = makeFlatVector<std::string>({R"(2)", R"({2)", R"({3)"});
  VELOX_ASSERT_USER_THROW(
      testFromJson(input, primitiveTypeOutput), "Unsupported type BIGINT.");
  VELOX_ASSERT_USER_THROW(
      testFromJson(input, mapOutput), "Unsupported type MAP<BIGINT,BIGINT>.");
}

TEST_F(FromJsonTest, invalidNullLikeTokens) {
  auto mapInput = makeFlatVector<std::string>(
      {R"({"a":nan})",
       R"({"a":nil})",
       R"({"a":none})",
       R"({"a":nope})",
       R"({"a":nullx})"});
  auto expectedMap = BaseVector::createNullConstant(
      MAP(VARCHAR(), BIGINT()), mapInput->size(), pool());
  testFromJson(mapInput, expectedMap);

  auto arrayInput =
      makeFlatVector<std::string>({R"([1,nil])", R"([nan])", R"([nullx])"});
  auto expectedArray = BaseVector::createNullConstant(
      ARRAY(INTEGER()), arrayInput->size(), pool());
  testFromJson(arrayInput, expectedArray);
}

TEST_F(FromJsonTest, invalidJson) {
  auto expected = makeNullableFlatVector<int32_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  auto input =
      makeFlatVector<std::string>({R"("a": 1})", R"({a: 1})", R"({"a" 1})"});
  testFromJson(input, makeRowVector({"a"}, {expected}));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
