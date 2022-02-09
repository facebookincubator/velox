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

#include <cstdint>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "velox/expression/Expr.h"
#include "velox/functions/Udf.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"

namespace {

using namespace facebook::velox;

class SimpleFunctionTest : public functions::test::FunctionBaseTest {
 protected:
  VectorPtr arraySum(const std::vector<std::vector<int64_t>>& data) {
    return makeFlatVector<int64_t>(data.size(), [&](auto row) {
      return std::accumulate(data[row].begin(), data[row].end(), 0);
    });
  }
};

template <typename T>
struct UnnamedFunction {
  bool call(bool&, const int64_t&) {
    return true;
  }
};

template <typename T>
struct NamedFunction {
  static constexpr auto name{"named_function"};

  bool call(bool&, const int64_t&) {
    return true;
  }
};

// Functions that provide a "name" member don't need aliases; functions that do
// not have a "name" member do.
TEST_F(SimpleFunctionTest, nameOrAliasRegistration) {
  // This one needs alias; will throw.
  auto registerThrow = [&]() {
    registerFunction<UnnamedFunction, bool, int64_t>();
  };
  EXPECT_THROW(registerThrow(), std::runtime_error);

  // These are good.
  auto registerNoThrow = [&]() {
    registerFunction<UnnamedFunction, bool, int64_t>({"my_alias"});
  };
  EXPECT_NO_THROW(registerNoThrow());

  auto registerNoThrow2 = [&]() {
    registerFunction<NamedFunction, bool, int64_t>();
  };
  EXPECT_NO_THROW(registerNoThrow2());
}

// Some input data.
static std::vector<std::vector<int64_t>> arrayData = {
    {0, 1, 2, 4},
    {99, 98},
    {101, 42},
    {10001, 12345676},
};

// Function that returns an array of bigints.
template <typename T>
struct ArrayWriterFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<int64_t>>& out,
      const arg_type<int64_t>& input) {
    const size_t size = arrayData[input].size();
    out.reserve(size);
    for (const auto i : arrayData[input]) {
      out.append(i);
    }
    return true;
  }
};

TEST_F(SimpleFunctionTest, arrayWriter) {
  registerFunction<ArrayWriterFunction, Array<int64_t>, int64_t>(
      {"array_writer_func"}, ARRAY(BIGINT()));

  const size_t rows = arrayData.size();
  auto flatVector = makeFlatVector<int64_t>(rows, [](auto row) { return row; });
  auto result = evaluate<ArrayVector>(
      "array_writer_func(c0)", makeRowVector({flatVector}));

  auto expected = vectorMaker_.arrayVector(arrayData);
  assertEqualVectors(expected, result);
}

static std::vector<std::vector<std::string>> stringArrayData = {
    {"a", "b", "c"},
    {"A long-ish sentence about apples.",
     "Another one about oranges.",
     "Just plum."},
    {"MA", "RI", "NY", "CA", "MI"},
};

// Function that returns an array of strings.
template <typename T>
struct ArrayOfStringsWriterFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<Varchar>>& out,
      const arg_type<int64_t>& input) {
    const size_t size = stringArrayData[input].size();
    out.reserve(size);
    for (const auto value : stringArrayData[input]) {
      out.append(out_type<Varchar>(StringView(value)));
    }
    return true;
  }
};

TEST_F(SimpleFunctionTest, arrayOfStringsWriter) {
  registerFunction<ArrayOfStringsWriterFunction, Array<Varchar>, int64_t>(
      {"array_of_strings_writer_func"}, ARRAY(VARCHAR()));

  const size_t rows = stringArrayData.size();
  auto flatVector = makeFlatVector<int64_t>(rows, [](auto row) { return row; });
  auto result = evaluate<ArrayVector>(
      "array_of_strings_writer_func(c0)", makeRowVector({flatVector}));

  std::vector<std::vector<StringView>> stringViews;
  for (auto i = 0; i < rows; i++) {
    stringViews.push_back({});
    for (auto j = 0; j < stringArrayData[i].size(); j++) {
      stringViews[i].push_back(StringView(stringArrayData[i][j]));
    }
  }

  auto expected = vectorMaker_.arrayVector(stringViews);
  assertEqualVectors(expected, result);
}

// Function that takes an array as input.
template <typename T>
struct ArrayReaderFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& out,
      const arg_type<Array<int64_t>>& input) {
    out = 0;
    for (const auto& v : input) {
      if (v) {
        out += v.value();
      }
    }
    return true;
  }
};

TEST_F(SimpleFunctionTest, arrayReader) {
  registerFunction<ArrayReaderFunction, int64_t, Array<int64_t>>(
      {"array_reader_func"});

  auto arrayVector = makeArrayVector(arrayData);
  auto result = evaluate<FlatVector<int64_t>>(
      "array_reader_func(c0)", makeRowVector({arrayVector}));

  assertEqualVectors(arraySum(arrayData), result);
}

// Function that takes an array of arrays as input.
template <typename T>
struct ArrayArrayReaderFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& out,
      const arg_type<Array<Array<int64_t>>>& input) {
    out = 0;
    for (const auto& inner : input) {
      if (inner) {
        for (const auto& v : inner.value()) {
          if (v) {
            out += v.value();
          }
        }
      }
    }
    return true;
  }
};

TEST_F(SimpleFunctionTest, arrayArrayReader) {
  registerFunction<ArrayArrayReaderFunction, int64_t, Array<Array<int64_t>>>(
      {"array_array_reader_func"});

  const size_t rows = arrayData.size();
  auto arrayVector = makeArrayVector(arrayData);
  auto result = evaluate<FlatVector<int64_t>>(
      "array_array_reader_func(array_constructor(c0, c0))",
      makeRowVector({arrayVector}));

  auto expected = makeFlatVector<int64_t>(rows, [&](auto row) {
    return 2 * std::accumulate(arrayData[row].begin(), arrayData[row].end(), 0);
  });
  assertEqualVectors(expected, result);
}

// Some input data for the rowVector.
static std::vector<int64_t> rowVectorCol1 = {0, 22, 44, 55, 99, 101, 9, 0};
static std::vector<double> rowVectorCol2 =
    {9.1, 22.4, 44.55, 99.9, 1.01, 9.8, 10001.1, 0.1};

// Function that returns a tuple.
template <typename T>
struct RowWriterFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Row<int64_t, double>>& out,
      const arg_type<int64_t>& input) {
    out = std::make_tuple(rowVectorCol1[input], rowVectorCol2[input]);
    return true;
  }
};

TEST_F(SimpleFunctionTest, rowWriter) {
  registerFunction<RowWriterFunction, Row<int64_t, double>, int64_t>(
      {"row_writer_func"}, ROW({BIGINT(), DOUBLE()}));

  const size_t rows = rowVectorCol1.size();
  auto flatVector = makeFlatVector<int64_t>(rows, [](auto row) { return row; });
  auto result =
      evaluate<RowVector>("row_writer_func(c0)", makeRowVector({flatVector}));

  auto vector1 = vectorMaker_.flatVector(rowVectorCol1);
  auto vector2 = vectorMaker_.flatVector(rowVectorCol2);
  auto expected = makeRowVector({vector1, vector2});
  assertEqualVectors(expected, result);
}

// Function that takes a tuple as a parameter.
template <typename T>
struct RowReaderFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& out,
      const arg_type<Row<int64_t, double>>& input) {
    out = *input.template at<0>();
    return true;
  }
};

TEST_F(SimpleFunctionTest, rowReader) {
  registerFunction<RowReaderFunction, int64_t, Row<int64_t, double>>(
      {"row_reader_func"});

  auto vector1 = vectorMaker_.flatVector(rowVectorCol1);
  auto vector2 = vectorMaker_.flatVector(rowVectorCol2);
  auto internalRowVector = makeRowVector({vector1, vector2});
  auto result = evaluate<FlatVector<int64_t>>(
      "row_reader_func(c0)", makeRowVector({internalRowVector}));

  auto expected = vectorMaker_.flatVector(rowVectorCol1);
  assertEqualVectors(expected, result);
}

// Function that takes a tuple of an array and a double.
template <typename T>
struct RowArrayReaderFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& out,
      const arg_type<Row<Array<int64_t>, double>>& input) {
    out = 0;
    const auto& arrayInput = *input.template at<0>();
    for (const auto& v : arrayInput) {
      if (v) {
        out += v.value();
      }
    }
    out += *input.template at<1>();
    return true;
  }
};

TEST_F(SimpleFunctionTest, rowArrayReader) {
  registerFunction<
      RowArrayReaderFunction,
      int64_t,
      Row<Array<int64_t>, double>>({"row_array_reader_func"});

  auto rows = arrayData.size();
  auto vector1 = makeArrayVector(arrayData);
  auto vector2 =
      makeFlatVector<double>(rows, [](auto row) { return row + 0.1; });
  auto internalRowVector = makeRowVector({vector1, vector2});
  auto result = evaluate<FlatVector<int64_t>>(
      "row_array_reader_func(c0)", makeRowVector({internalRowVector}));

  auto expected = makeFlatVector<int64_t>(rows, [&](auto row) {
    return row + 0.1 +
        std::accumulate(arrayData[row].begin(), arrayData[row].end(), 0);
  });
  assertEqualVectors(expected, result);
}

// Function that returns an array of rows.
template <typename T>
struct ArrayRowWriterFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<Row<int64_t, double>>>& out,
      const arg_type<int32_t>& input) {
    // Appends each row three times.
    auto tuple = std::make_tuple(rowVectorCol1[input], rowVectorCol2[input]);
    out.append(std::optional(tuple));
    out.append(std::optional(tuple));
    out.append(std::optional(tuple));
    return true;
  }
};

TEST_F(SimpleFunctionTest, arrayRowWriter) {
  registerFunction<
      ArrayRowWriterFunction,
      Array<Row<int64_t, double>>,
      int32_t>({"array_row_writer_func"}, ARRAY(ROW({BIGINT(), DOUBLE()})));

  const size_t rows = rowVectorCol1.size();
  auto flatVector = makeFlatVector<int32_t>(rows, [](auto row) { return row; });
  auto result = evaluate<ArrayVector>(
      "array_row_writer_func(c0)", makeRowVector({flatVector}));

  std::vector<std::vector<variant>> data;
  for (int64_t i = 0; i < rows; ++i) {
    data.push_back({
        variant::row({rowVectorCol1[i], rowVectorCol2[i]}),
        variant::row({rowVectorCol1[i], rowVectorCol2[i]}),
        variant::row({rowVectorCol1[i], rowVectorCol2[i]}),
    });
  }
  auto expected =
      vectorMaker_.arrayOfRowVector(ROW({BIGINT(), DOUBLE()}), data);
  assertEqualVectors(expected, result);
}

// Function that takes an array of rows as an argument.
template <typename T>
struct ArrayRowReaderFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& out,
      const arg_type<Array<Row<int64_t, double>>>& input) {
    out = 0;
    for (size_t i = 0; i < input.size(); i++) {
      auto&& row = *input.at(i);
      out += *row.template at<0>();
    }
    return true;
  }
};

TEST_F(SimpleFunctionTest, arrayRowReader) {
  registerFunction<
      ArrayRowReaderFunction,
      int64_t,
      Array<Row<int64_t, double>>>({"array_row_reader_func"});

  const size_t rows = rowVectorCol1.size();
  std::vector<std::vector<variant>> data;

  for (int64_t i = 0; i < rows; ++i) {
    data.push_back({
        variant::row({rowVectorCol1[i], rowVectorCol2[i]}),
        variant::row({rowVectorCol1[i], rowVectorCol2[i]}),
        variant::row({rowVectorCol1[i], rowVectorCol2[i]}),
    });
  }
  auto arrayVector =
      vectorMaker_.arrayOfRowVector(ROW({BIGINT(), DOUBLE()}), data);
  auto result = evaluate<FlatVector<int64_t>>(
      "array_row_reader_func(c0)", makeRowVector({arrayVector}));

  auto localData = rowVectorCol1;
  auto expected = makeFlatVector<int64_t>(
      rows, [&localData](auto row) { return localData[row] * 3; });
  assertEqualVectors(expected, result);
}

using MyType = std::pair<int64_t, double>;

// Function that returns a tuple containing an opaque type
template <typename T>
struct RowOpaqueWriterFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Row<std::shared_ptr<MyType>, int64_t>>& out,
      const arg_type<int64_t>& input) {
    out = std::make_tuple(
        std::make_shared<MyType>(rowVectorCol1[input], rowVectorCol2[input]),
        input + 10);
    return true;
  }
};

TEST_F(SimpleFunctionTest, rowOpaqueWriter) {
  registerFunction<
      RowOpaqueWriterFunction,
      Row<std::shared_ptr<MyType>, int64_t>,
      int64_t>({"row_opaque_writer_func"});

  const size_t rows = rowVectorCol1.size();
  auto flatVector = makeFlatVector<int64_t>(rows, [](auto row) { return row; });
  auto result = evaluate<RowVector>(
      "row_opaque_writer_func(c0)", makeRowVector({flatVector}));
  auto opaqueOutput =
      std::dynamic_pointer_cast<FlatVector<std::shared_ptr<void>>>(
          result->childAt(0));
  auto bigintOutput =
      std::dynamic_pointer_cast<FlatVector<int64_t>>(result->childAt(1));

  // Opaque flat vector are not comparable with equalValueAt(), so we check it
  // manually.
  for (size_t i = 0; i < rows; i++) {
    auto val = std::static_pointer_cast<MyType>(opaqueOutput->valueAt(i));
    ASSERT_EQ(rowVectorCol1[i], val->first);
    ASSERT_EQ(rowVectorCol2[i], val->second);

    ASSERT_EQ(i + 10, bigintOutput->valueAt(i));
  }
}

// Function that takes a tuple containing an opaque type as a parameter.
template <typename T>
struct RowOpaqueReaderFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& out,
      const arg_type<Row<std::shared_ptr<MyType>, int64_t>>& input) {
    const auto& myType = *input.template at<0>();
    out = myType->first;
    return true;
  }
};

TEST_F(SimpleFunctionTest, rowOpaqueReader) {
  registerFunction<
      RowOpaqueReaderFunction,
      int64_t,
      Row<std::shared_ptr<MyType>, int64_t>>({"row_opaque_reader_func"});

  const size_t rows = rowVectorCol1.size();
  auto vector1 = makeFlatVector<std::shared_ptr<void>>(rows, [&](auto row) {
    return std::make_shared<MyType>(rowVectorCol1[row], rowVectorCol2[row]);
  });
  auto vector2 = vectorMaker_.flatVector(rowVectorCol1);
  auto internalRowVector = makeRowVector({vector1, vector2});
  auto result = evaluate<FlatVector<int64_t>>(
      "row_opaque_reader_func(c0)", makeRowVector({internalRowVector}));

  auto expected = vectorMaker_.flatVector(rowVectorCol1);
  assertEqualVectors(expected, result);
}

// Nullability tests:

// Test that function with default null behavior won't get called when inputs
// are all null.
template <typename T>
struct DefaultNullBehaviorFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(out_type<bool>&, int64_t) {
    throw std::runtime_error(
        "Function not supposed to be called on null inputs.");
    return true;
  }
};

TEST_F(SimpleFunctionTest, defaultNullBehavior) {
  registerFunction<DefaultNullBehaviorFunction, bool, int64_t>(
      {"default_null_behavior"});

  // Make a vector filled with nulls.
  auto flatVector = makeFlatVector<int64_t>(
      10, [](auto row) { return row; }, [](auto) { return true; });

  // Check that default null behavior functions don't get called on a null
  // input.
  EXPECT_NO_THROW(evaluate<SimpleVector<bool>>(
      "default_null_behavior(c0)", makeRowVector({flatVector})));
}

// Test that function with non-default null behavior receives parameters as
// nulls. Returns whether the received parameter was null.
template <typename T>
struct NonDefaultNullBehaviorFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool callNullable(
      out_type<bool>& out,
      const int64_t* input) {
    out = (input == nullptr);
    return true;
  }
};

TEST_F(SimpleFunctionTest, nonDefaultNullBehavior) {
  registerFunction<NonDefaultNullBehaviorFunction, bool, int64_t>(
      {"non_default_null_behavior"});

  // Make a vector filled with nulls.
  const size_t rows = 10;
  auto flatVector = makeFlatVector<int64_t>(
      rows, [](auto row) { return row; }, [](auto) { return true; });

  // Check that nullable function is returning the right results.
  auto result = evaluate<FlatVector<bool>>(
      "non_default_null_behavior(c0)", makeRowVector({flatVector}));
  auto expected = makeFlatVector<bool>(rows, [](auto) { return true; });
  assertEqualVectors(expected, result);
}

// Ensures that the call method can be templated.
template <typename T>
struct IsInputVarcharFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  template <typename TType>
  FOLLY_ALWAYS_INLINE bool call(out_type<bool>& out, const TType&) {
    if constexpr (std::is_same_v<TType, StringView>) {
      out = true;
    } else {
      out = false;
    }
    return true;
  }
};

TEST_F(SimpleFunctionTest, templatedCall) {
  registerFunction<IsInputVarcharFunction, bool, int64_t>({"is_input_varchar"});
  registerFunction<IsInputVarcharFunction, bool, Varchar>({"is_input_varchar"});

  const size_t rows = 10;
  auto flatVector = makeFlatVector<int64_t>(rows, [](auto row) { return row; });

  // Ensure that functions passing varchars and non-varchars return the expected
  // boolean values.
  auto result = evaluate<FlatVector<bool>>(
      "is_input_varchar(c0)", makeRowVector({flatVector}));
  auto expected = makeFlatVector<bool>(rows, [](auto) { return false; });
  assertEqualVectors(expected, result);

  auto flatVectorStr =
      makeFlatVector<StringView>(rows, [](auto) { return StringView("asdf"); });
  result = evaluate<FlatVector<bool>>(
      "is_input_varchar(c0)", makeRowVector({flatVectorStr}));
  expected = makeFlatVector<bool>(rows, [](auto) { return true; });
  assertEqualVectors(expected, result);
}

// Function that takes a map as input.
template <typename T>
struct MapReaderFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& out,
      const arg_type<Map<int64_t, double>>& input) {
    out = 0;
    for (const auto& entry : input) {
      out += entry.first;
      if (entry.second) {
        out += entry.second.value();
      }
    }
    return true;
  }
};

TEST_F(SimpleFunctionTest, mapReader) {
  registerFunction<MapReaderFunction, int64_t, Map<int64_t, double>>(
      {"map_reader_func"});

  const vector_size_t size = 10;
  auto mapVector = vectorMaker_.mapVector<int64_t, double>(
      size,
      [](auto row) { return row % 5; },
      [](auto /*row*/, auto index) { return index; },
      [](auto /*row*/, auto index) { return 1.2 * index; });
  auto result = evaluate<FlatVector<int64_t>>(
      "map_reader_func(c0)", makeRowVector({mapVector}));

  auto expected = makeFlatVector<int64_t>(size, [](auto row) {
    int64_t sum = 0;
    for (auto index = 0; index < row % 5; index++) {
      sum += index + 1.2 * index;
    }
    return sum;
  });
  assertEqualVectors(expected, result);
}

// Function that takes a map from integer to array of doubles as input.
template <typename T>
struct MapArrayReaderFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      double& out,
      const arg_type<Map<int64_t, Array<double>>>& input) {
    out = 0;
    for (const auto& entry : input) {
      out += entry.first;
      if (entry.second) {
        for (const auto& v : entry.second.value()) {
          if (v) {
            out += v.value();
          }
        }
      }
    }
    return true;
  }
};

TEST_F(SimpleFunctionTest, mapArrayReader) {
  registerFunction<MapArrayReaderFunction, double, Map<int64_t, Array<double>>>(
      {"map_array_reader_func"});

  const vector_size_t size = 10;
  auto keys = makeArrayVector<int64_t>(
      size,
      [](auto /*row*/) { return 2; },
      [](auto /*row*/, auto index) { return index; });
  auto values = makeArrayVector<double>(
      size,
      [](auto row) { return row % 5; },
      [](auto /*row*/, auto index) { return 1.2 * index; });
  auto moreValues = makeArrayVector<double>(
      size,
      [](auto row) { return row % 3; },
      [](auto /*row*/, auto index) { return 0.1 * index; });
  auto result = evaluate<FlatVector<double>>(
      "map_array_reader_func(map(c0, array_constructor(c1, c2)))",
      makeRowVector({keys, values, moreValues}));

  auto expected = makeFlatVector<double>(size, [](auto row) {
    double sum = 1; // Sum of keys: 0 and 1.
    for (auto index = 0; index < row % 5; index++) {
      sum += 1.2 * index;
    }
    for (auto index = 0; index < row % 3; index++) {
      sum += 0.1 * index;
    }
    return sum;
  });

  ASSERT_EQ(size, result->size());
  for (auto i = 0; i < size; i++) {
    EXPECT_NEAR(expected->valueAt(i), result->valueAt(i), 0.0000001);
  }
}

template <typename T>
struct MyArrayStringReuseFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  static constexpr int32_t reuse_strings_from_arg = 0;

  bool call(out_type<Array<Varchar>>& out, const arg_type<Varchar>& input) {
    auto start = input.begin();
    auto cur = start;

    do {
      cur = std::find(start, input.end(), ' ');
      out.append(std::optional{StringView(start, cur - start)});
      start = cur + 1;
    } while (cur < input.end());
    return true;
  }
};

TEST_F(SimpleFunctionTest, arrayStringReuse) {
  registerFunction<MyArrayStringReuseFunction, Array<Varchar>, Varchar>(
      {"my_array_string_reuse_func"});

  std::vector<StringView> inputData = {
      "my input data that will be tokenized"_sv, "some more tokens"_sv};
  std::vector<std::vector<StringView>> outputData = {
      {"my"_sv,
       "input"_sv,
       "data"_sv,
       "that"_sv,
       "will"_sv,
       "be"_sv,
       "tokenized"_sv},
      {"some"_sv, "more"_sv, "tokens"_sv},
  };

  auto flatVector = vectorMaker_.flatVector(inputData);
  auto result = evaluate<ArrayVector>(
      "my_array_string_reuse_func(c0)", makeRowVector({flatVector}));

  auto expected = vectorMaker_.arrayVector(outputData);
  assertEqualVectors(expected, result);
}

template <typename T>
struct MapStringOut {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  bool call(out_type<Map<Varchar, Varchar>>& out, int64_t n) {
    auto string = std::to_string(n);
    out.emplace(StringView(string), std::optional{StringView(string)});
    return true;
  }
};

// Output map with string.
TEST_F(SimpleFunctionTest, mapStringOut) {
  registerFunction<MapStringOut, Map<Varchar, Varchar>, int64_t>(
      {"func_map_string_out"});

  auto input = vectorMaker_.flatVector<int64_t>({1, 2, 3, 4});
  auto result = evaluate<facebook::velox::MapVector>(
      "func_map_string_out(c0)", makeRowVector({input}));

  DecodedVector decoded;
  SelectivityVector rows(4);
  decoded.decode(*result, rows);
  exec::VectorReader<Map<Varchar, Varchar>> reader(&decoded);
  for (auto i = 0; i < 4; i++) {
    auto mapView = reader[i];
    for (const auto& [key, value] : mapView) {
      ASSERT_EQ(std::string(key.data(), key.size()), std::to_string(i + 1));
      ASSERT_EQ(
          std::string(value.value().data(), value.value().size()),
          std::to_string(i + 1));
    }
  }
}

} // namespace
