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
#include <fmt/core.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <optional>

#include "velox/expression/VectorUdfTypeSystem.h"
#include "velox/functions/Udf.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/type/StringView.h"
namespace facebook::velox {
namespace {

// Function that creates a map with elements i->i*2+1 if i is odd.
// and i->null if i is even. For every i = [0..n).
template <typename T>
struct Func {
  template <typename TOut>
  bool call(TOut& out, const int64_t& n) {
    for (int i = 0; i < n; i++) {
      switch (i % 2) {
        case 0: {
          auto [keyWriter, valueWriter] = out.add_item();
          keyWriter = i;
          valueWriter = i * 2 + 1;
          break;
        }
        case 1:
          out.add_null() = i;
          break;
      }
    }
    return true;
  }
};

class MapWriterTest : public functions::test::FunctionBaseTest {
 public:
  template <typename K, typename V>
  using map_pairs_t = std::vector<std::pair<K, std::optional<V>>>;

  VectorPtr prepareResult(const TypePtr& mapType, vector_size_t size = 1) {
    VectorPtr result;
    BaseVector::ensureWritable(
        SelectivityVector(size), mapType, this->execCtx_.pool(), &result);
    return result;
  }

  template <typename T>
  void testE2E(const std::string& testFunctionName) {
    registerFunction<Func, MapWriterT<T, T>, int64_t>({testFunctionName});

    auto result = evaluate(
        fmt::format("{}(c0)", testFunctionName),
        makeRowVector(
            {makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})}));

    std::vector<map_pairs_t<T, T>> expected;
    for (auto i = 1; i <= 10; i++) {
      expected.push_back({});
      auto& currentExpected = expected[expected.size() - 1];
      for (auto j = 0; j < i; j++) {
        switch (j % 2) {
          case 0:
            currentExpected.push_back({j, {j * 2 + 1}});
            break;
          case 1:
            currentExpected.push_back({j, std::nullopt});
            break;
        }
      }
    }
    assertEqualVectors(result, makeMapVector<T, T>(expected));
  }

  struct TestWriter {
    VectorPtr result;
    std::unique_ptr<exec::VectorWriter<MapWriterT<int64_t, int64_t>>> writer =
        std::make_unique<exec::VectorWriter<MapWriterT<int64_t, int64_t>>>();
  };

  TestWriter makeTestWriter() {
    TestWriter writer;

    writer.result =
        prepareResult(std::make_shared<MapType>(MapType(BIGINT(), BIGINT())));
    writer.writer->init(*writer.result->as<MapVector>());
    writer.writer->setOffset(0);
    return writer;
  }
};

TEST_F(MapWriterTest, addNull) {
  auto [result, vectorWriter] = makeTestWriter();

  auto& mapWriter = vectorWriter->current();
  mapWriter.add_null() = 1;
  mapWriter.add_null() = 2;
  vectorWriter->commit();

  vectorWriter->finish();
  map_pairs_t<int64_t, int64_t> expected = {
      {1, std::nullopt}, {2, std::nullopt}};
  assertEqualVectors(result, makeMapVector<int64_t, int64_t>({expected}));
}

TEST_F(MapWriterTest, writeThenCommitNull) {
  auto [result, vectorWriter] = makeTestWriter();
  vectorWriter->ensureSize(2);

  {
    vectorWriter->setOffset(0);
    auto& mapWriter = vectorWriter->current();
    mapWriter.add_null() = 1;
    mapWriter.add_null() = 2;
    mapWriter.add_item() = std::make_tuple(100, 100);
    vectorWriter->commitNull();
  }

  {
    vectorWriter->setOffset(1);
    auto& mapWriter = vectorWriter->current();
    mapWriter.add_item() = std::make_tuple(200, 200);
    mapWriter.add_null() = 1;
    vectorWriter->commit();
  }
  vectorWriter->finish();

  map_pairs_t<int64_t, int64_t> expected = {{200, 200}, {1, std::nullopt}};
  auto expexctedVector = makeMapVector<int64_t, int64_t>({{}, expected});
  expexctedVector->setNull(0, true);
  assertEqualVectors(result, expexctedVector);
}

TEST_F(MapWriterTest, addItem) {
  auto [result, vectorWriter] = makeTestWriter();

  auto& mapWriter = vectorWriter->current();
  // Item 1.
  auto [key, value] = mapWriter.add_item();
  key = 1;
  value = 1;

  // Item 2.
  mapWriter.add_item() = std::make_tuple(1, 3);

  // Item 3.
  auto writers = mapWriter.add_item();
  std::get<0>(writers) = 11;
  std::get<1>(writers) = 12;

  vectorWriter->commit();
  vectorWriter->finish();

  map_pairs_t<int64_t, int64_t> expected = {{1, 1}, {1, 3}, {11, 12}};
  assertEqualVectors(result, makeMapVector<int64_t, int64_t>({expected}));
}

TEST_F(MapWriterTest, e2ePrimitives) {
  testE2E<int8_t>("f_int6");
  testE2E<int16_t>("f_int16");
  testE2E<int32_t>("f_int32");
  testE2E<int64_t>("f_int64");
  testE2E<float>("f_float");
  testE2E<double>("f_double");
  testE2E<bool>("f_bool");
}

TEST_F(MapWriterTest, testTimeStamp) {
  auto result =
      prepareResult(std::make_shared<MapType>(MapType(BIGINT(), TIMESTAMP())));

  exec::VectorWriter<MapWriterT<int64_t, Timestamp>> writer;
  writer.init(*result->as<MapVector>());
  writer.setOffset(0);
  auto& mapWriter = writer.current();
  // General interface.
  mapWriter.add_item() = std::make_tuple(1, Timestamp::fromMillis(1));
  mapWriter.add_null() = 2;
  writer.commit();
  writer.finish();
  map_pairs_t<int64_t, Timestamp> expected = {
      {1, Timestamp::fromMillis(1)}, {2, std::nullopt}};
  assertEqualVectors(result, makeMapVector<int64_t, Timestamp>({expected}));
}

TEST_F(MapWriterTest, testVarChar) {
  auto result =
      prepareResult(std::make_shared<MapType>(MapType(VARCHAR(), VARCHAR())));

  exec::VectorWriter<MapWriterT<Varchar, Varchar>> writer;
  writer.init(*result->as<MapVector>());
  writer.setOffset(0);
  auto& mapWriter = writer.current();
  {
    auto [keyWriter, valueWriter] = mapWriter.add_item();
    keyWriter.resize(2);
    keyWriter.data()[0] = 'h';
    keyWriter.data()[1] = 'i';

    valueWriter += "welcome"_sv;
  }

  {
    auto& keyWriter = mapWriter.add_null();
    keyWriter += "null"_sv;
  }

  writer.commit();
  writer.finish();

  map_pairs_t<StringView, StringView> expected = {
      {"hi"_sv, "welcome"_sv}, {"null"_sv, std::nullopt}};
  assertEqualVectors(result, makeMapVector<StringView, StringView>({expected}));
}

TEST_F(MapWriterTest, testVarBinary) {
  auto result = prepareResult(
      std::make_shared<MapType>(MapType(VARBINARY(), VARBINARY())));

  exec::VectorWriter<MapWriterT<Varbinary, Varbinary>> writer;
  writer.init(*result->as<MapVector>());
  writer.setOffset(0);

  auto& mapWriter = writer.current();
  {
    auto [keyWriter, valueWriter] = mapWriter.add_item();
    keyWriter.resize(2);
    keyWriter.data()[0] = 'h';
    keyWriter.data()[1] = 'i';

    valueWriter += "welcome"_sv;
  }

  {
    auto& keyWriter = mapWriter.add_null();
    keyWriter += "null"_sv;
  }

  writer.commit();
  writer.finish();

  map_pairs_t<StringView, StringView> expected = {
      {"hi"_sv, "welcome"_sv}, {"null"_sv, std::nullopt}};

  // Test results.
  DecodedVector decoded;
  SelectivityVector rows(result->size());
  decoded.decode(*result, rows);
  exec::VectorReader<Map<Varbinary, Varbinary>> reader(&decoded);
  auto mapView = reader[0];

  ASSERT_EQ(mapView.size(), expected.size());

  auto i = 0;
  for (auto [key, value] : mapView) {
    ASSERT_EQ(key, expected[i].first);
    ASSERT_EQ(value, expected[i].second);
    i++;
  }
}

// A function that returns map<array<int>, map<int, int>> as output.
template <typename T>
struct MakeComplexMapFunction {
  template <typename TOut>
  bool call(TOut& out, const int64_t& n) {
    auto [arrayWriter, mapWriter] = out.add_item();
    arrayWriter.resize(n);
    for (auto i = 0; i < n; i++) {
      arrayWriter[i] = i;
    }
    for (auto i = 0; i < n; i++) {
      mapWriter.add_item() = std::make_tuple(i, i + 1);
    }
    return true;
  }
};

// Test a function that writes out map<array, map<>>.
TEST_F(MapWriterTest, nestedMap) {
  using out_t = MapWriterT<ArrayProxyT<int64_t>, MapWriterT<int64_t, int64_t>>;
  registerFunction<MakeComplexMapFunction, out_t, int64_t>({"complex_map"});

  auto result = evaluate(
      "complex_map(c0)",
      makeRowVector({makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})}));

  // Test results.
  DecodedVector decoded;
  SelectivityVector rows(result->size());
  decoded.decode(*result, rows);
  exec::VectorReader<Map<Array<int64_t>, Map<int64_t, int64_t>>> reader(
      &decoded);
  for (auto i = 0; i < rows.size(); i++) {
    auto outerMap = reader[i];
    for (const auto& [array, innerMap] : outerMap) {
      ASSERT_EQ(array.size(), i);
      // Check key.
      for (auto j = 0; j < i; j++) {
        ASSERT_EQ(array[j].value(), j);
      }

      // Check value.
      auto j = 0;
      for (const auto& [key, val] : innerMap.value()) {
        ASSERT_EQ(key, j);
        ASSERT_EQ(*val, j + 1);
        j++;
      }
    }
  }
}

} // namespace
} // namespace facebook::velox
