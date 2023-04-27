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
#include "velox/vector/VectorSaver.h"
#include <fstream>
#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/tests/utils/TempFilePath.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/functions/prestosql/types/JsonType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::test {

class VectorSaverTest : public testing::Test, public VectorTestBase {
 protected:
  VectorSaverTest() {
    registerJsonType();
    registerHyperLogLogType();
    registerTimestampWithTimeZoneType();
  }

  void SetUp() override {
    LOG(ERROR) << "Seed: " << seed_;
  }

  void assertEqualEncodings(
      const VectorPtr& expected,
      const VectorPtr& actual) {
    ASSERT_EQ(expected->encoding(), actual->encoding());
    ASSERT_EQ(*expected->type(), *actual->type());
    assertEqualVectors(expected, actual);

    // Recursively compare inner vectors to ensure that all layers of wrappings
    // are the same.
    switch (expected->encoding()) {
      case VectorEncoding::Simple::CONSTANT:
      case VectorEncoding::Simple::DICTIONARY:
        if (expected->valueVector()) {
          ASSERT_TRUE(actual->valueVector() != nullptr);
          assertEqualEncodings(expected->valueVector(), actual->valueVector());
        } else {
          ASSERT_TRUE(actual->valueVector() == nullptr);
        }
        break;
      case VectorEncoding::Simple::ARRAY:
        assertEqualEncodings(
            expected->as<ArrayVector>()->elements(),
            actual->as<ArrayVector>()->elements());
        break;
      case VectorEncoding::Simple::MAP:
        assertEqualEncodings(
            expected->as<MapVector>()->mapKeys(),
            actual->as<MapVector>()->mapKeys());
        assertEqualEncodings(
            expected->as<MapVector>()->mapValues(),
            actual->as<MapVector>()->mapValues());
        break;
      case VectorEncoding::Simple::ROW: {
        auto expectedRow = expected->as<RowVector>();
        auto actualRow = actual->as<RowVector>();
        for (auto i = 0; i < expectedRow->childrenSize(); ++i) {
          ASSERT_EQ(
              expectedRow->childAt(i) != nullptr,
              actualRow->childAt(i) != nullptr);
          if (expectedRow->childAt(i)) {
            assertEqualEncodings(
                expectedRow->childAt(i), actualRow->childAt(i));
          }
        }
        break;
      }
      default:
          // Do nothing.
          ;
    }
  }

  VectorFuzzer::Options fuzzerOptions() {
    VectorFuzzer::Options opts;
    opts.vectorSize = 128;
    opts.stringVariableLength = true;
    opts.stringLength = 64;
    return opts;
  }

  void testRoundTrip(VectorFuzzer::Options options, const TypePtr& type) {
    SCOPED_TRACE(fmt::format("seed: {}", seed_));
    VectorFuzzer fuzzer(options, pool(), seed_);
    testRoundTrip(fuzzer.fuzzFlat(type));
  }

  // Writes the passed vector to file and reads it back, returns the read
  // vector.
  VectorPtr takeRoundTrip(const VectorPtr& vector) {
    auto path = exec::test::TempFilePath::create();

    std::ofstream outputFile(path->path, std::ofstream::binary);
    saveVector(*vector, outputFile);
    outputFile.close();

    std::ifstream inputFile(path->path, std::ifstream::binary);
    auto copy = restoreVector(inputFile, pool());
    inputFile.close();
    return copy;
  }

  void testRoundTrip(const VectorPtr& vector) {
    auto copy = takeRoundTrip(vector);
    // Verify encodings and data recursively.
    assertEqualEncodings(vector, copy);
  }

  void testTypeRoundTrip(const TypePtr& type) {
    auto path = exec::test::TempFilePath::create();

    std::ofstream outputFile(path->path, std::ofstream::binary);
    saveType(type, outputFile);
    outputFile.close();

    std::ifstream inputFile(path->path, std::ifstream::binary);
    auto copy = restoreType(inputFile);
    inputFile.close();

    ASSERT_EQ(*type, *copy)
        << "Expected: " << type->toString() << ". Got: " << copy->toString();
  }

  void testSelectivityVectorRoundTrip(const SelectivityVector& rows) {
    std::ostringstream out;
    saveSelectivityVector(rows, out);

    std::istringstream in(out.str());
    auto copy = restoreSelectivityVector(in);
    ASSERT_EQ(rows, copy);
  }

  template <typename T>
  void testFlatIntegers() {
    // No nulls.
    testRoundTrip(makeFlatVector<T>({1, 2, 3, 4, 5}));

    // Some nulls.
    testRoundTrip(
        makeNullableFlatVector<T>({1, std::nullopt, 3, std::nullopt, 5}));

    // Empty vector.
    testRoundTrip(BaseVector::create(CppToType<T>::create(), 0, pool()));

    // Long vector.
    testRoundTrip(makeFlatVector<T>(10'000, [](auto row) { return row; }));

    // Long vector with nulls.
    testRoundTrip(makeFlatVector<T>(
        10'000, [](auto row) { return row; }, nullEvery(17)));
  }

  // Verifies that the lazy vectors have the right internal state and loaded
  // vector (if already loaded).
  void assertEqualLazyVectors(
      const VectorPtr& expected,
      const VectorPtr& actual,
      const SelectivityVector& rows) {
    // Verify encoding of Lazy.
    ASSERT_EQ(expected->encoding(), actual->encoding());
    ASSERT_EQ(*expected->type(), *actual->type());
    ASSERT_EQ(expected->size(), actual->size());

    // Verify whether the loaded vector is present.
    ASSERT_EQ(isLazyNotLoaded(*expected), isLazyNotLoaded(*actual));
    if (!isLazyNotLoaded(*actual)) {
      // Check only the rows loaded.
      assertEqualVectors(expected, actual, rows);
    }
  }

  // Test round trip of lazy vector both with and without loading. The vector
  // gets loaded with a randomly generated selectivity vector.
  void testRoundTripOfLazy(VectorPtr vector, VectorFuzzer& fuzzer) {
    SelectivityVector rows(vector->size());
    auto copy = takeRoundTrip(vector);
    assertEqualLazyVectors(vector, copy, rows);

    // Verify loaded lazy vector makes the round trip by loading it for a
    // random set of rows.
    for (int i = 0; i < vector->size(); ++i) {
      if (fuzzer.coinToss(0.3)) {
        rows.setValid(i, false);
      }
    }
    rows.updateBounds();
    LazyVector::ensureLoadedRows(vector, rows);
    copy = takeRoundTrip(vector);
    LazyVector::ensureLoadedRows(copy, rows);
    assertEqualLazyVectors(vector, copy, rows);
  }

  const uint32_t seed_{folly::Random::rand32()};
};

TEST_F(VectorSaverTest, types) {
  testTypeRoundTrip(BOOLEAN());

  testTypeRoundTrip(TINYINT());
  testTypeRoundTrip(SMALLINT());
  testTypeRoundTrip(INTEGER());
  testTypeRoundTrip(BIGINT());

  testTypeRoundTrip(REAL());
  testTypeRoundTrip(DOUBLE());

  testTypeRoundTrip(VARCHAR());
  testTypeRoundTrip(VARBINARY());

  testTypeRoundTrip(TIMESTAMP());
  testTypeRoundTrip(DATE());
  testTypeRoundTrip(INTERVAL_DAY_TIME());

  testTypeRoundTrip(ARRAY(BIGINT()));
  testTypeRoundTrip(ARRAY(ARRAY(VARCHAR())));

  testTypeRoundTrip(MAP(INTEGER(), REAL()));
  testTypeRoundTrip(MAP(VARCHAR(), ARRAY(BIGINT())));
  testTypeRoundTrip(MAP(BIGINT(), MAP(INTEGER(), DOUBLE())));

  testTypeRoundTrip(ROW({}, {}));
  testTypeRoundTrip(ROW({"a", "b", "c"}, {INTEGER(), BOOLEAN(), VARCHAR()}));
  testTypeRoundTrip(
      ROW({"a", "b", "c"},
          {INTEGER(), ARRAY(BOOLEAN()), MAP(VARCHAR(), VARCHAR())}));
  testTypeRoundTrip(
      ROW({"a", "b"}, {ROW({"c", "d"}, {VARCHAR(), DATE()}), INTEGER()}));

  testTypeRoundTrip(UNKNOWN());

  VELOX_ASSERT_THROW(
      testTypeRoundTrip(OPAQUE<std::string>()),
      "No serialization persistent name registered for OPAQUE");

  testTypeRoundTrip(JSON());
  testTypeRoundTrip(HYPERLOGLOG());
  testTypeRoundTrip(TIMESTAMP_WITH_TIME_ZONE());
}

TEST_F(VectorSaverTest, selectivityVector) {
  // Short vector. All selected.
  {
    SelectivityVector rows(10);
    testSelectivityVectorRoundTrip(rows);
  }

  // Longer vector. All selected.
  {
    SelectivityVector rows(1024);
    testSelectivityVectorRoundTrip(rows);
  }

  // Longer vector. No rows selected.
  {
    SelectivityVector rows(1024, false);
    testSelectivityVectorRoundTrip(rows);
  }

  // Even rows selected.
  {
    SelectivityVector rows(1024);
    for (auto i = 0; i < rows.size(); ++i) {
      rows.setValid(i, i % 2 == 0);
    }
    rows.updateBounds();
    testSelectivityVectorRoundTrip(rows);
  }

  // Odd rows selected.
  {
    SelectivityVector rows(1024);
    for (auto i = 0; i < rows.size(); ++i) {
      rows.setValid(i, i % 2 != 0);
    }
    rows.updateBounds();
    testSelectivityVectorRoundTrip(rows);
  }

  // Rows in the middle selected.
  {
    SelectivityVector rows(1024, false);
    rows.setValidRange(123, 955, true);
    rows.updateBounds();
    testSelectivityVectorRoundTrip(rows);
  }

  // Rows at ends selected.
  {
    SelectivityVector rows(1024, false);
    rows.setValidRange(0, 123, true);
    rows.setValidRange(887, rows.size(), true);
    rows.updateBounds();
    testSelectivityVectorRoundTrip(rows);
  }
}

TEST_F(VectorSaverTest, flatBigint) {
  testFlatIntegers<int64_t>();
}

TEST_F(VectorSaverTest, flatInteger) {
  testFlatIntegers<int32_t>();
}

TEST_F(VectorSaverTest, flatSmallint) {
  testFlatIntegers<int16_t>();
}

TEST_F(VectorSaverTest, flatTinyint) {
  testFlatIntegers<int8_t>();
}

TEST_F(VectorSaverTest, flatBoolean) {
  // No nulls.
  testRoundTrip(makeFlatVector<bool>({true, false, true, true, false}));

  // Some nulls.
  testRoundTrip(makeNullableFlatVector<bool>(
      {true, std::nullopt, true, std::nullopt, false}));

  // Empty vector.
  testRoundTrip(BaseVector::create(BOOLEAN(), 0, pool()));

  // Long vector.
  testRoundTrip(
      makeFlatVector<bool>(10'000, [](auto row) { return row % 7 == 2; }));

  // Long vector with nulls.
  testRoundTrip(makeFlatVector<bool>(
      10'000, [](auto row) { return row % 2 == 1; }, nullEvery(17)));
}

TEST_F(VectorSaverTest, flatVarchar) {
  VectorFuzzer::Options opts = fuzzerOptions();

  testRoundTrip(opts, VARCHAR());

  // Add some nulls.
  opts.nullRatio = 0.1;
  testRoundTrip(opts, VARCHAR());

  // Make short strings only.
  opts.stringLength = 6;
  opts.vectorSize = 1024;
  testRoundTrip(opts, VARCHAR());
}

TEST_F(VectorSaverTest, flatIntervalDayTime) {
  VectorFuzzer::Options opts = fuzzerOptions();
  testRoundTrip(opts, INTERVAL_DAY_TIME());
}

TEST_F(VectorSaverTest, row) {
  auto opts = fuzzerOptions();

  auto flatRowType =
      ROW({"a_i8", "b_i16", "c_i32", "d_i64", "e_bool", "f_varchar"},
          {TINYINT(), SMALLINT(), INTEGER(), BIGINT(), BOOLEAN(), VARCHAR()});
  auto nestedRowType =
      ROW({"a", "b"},
          {INTEGER(), ROW({"b1", "b2", "b3"}, {BIGINT(), REAL(), DOUBLE()})});

  testRoundTrip(opts, flatRowType);
  testRoundTrip(opts, nestedRowType);

  // Add some nulls.
  opts.nullRatio = 0.1;
  testRoundTrip(opts, flatRowType);
  testRoundTrip(opts, nestedRowType);
}

TEST_F(VectorSaverTest, array) {
  auto opts = fuzzerOptions();

  testRoundTrip(opts, ARRAY(BIGINT()));
  testRoundTrip(opts, ARRAY(VARCHAR()));

  // Add some nulls.
  opts.nullRatio = 0.1;
  testRoundTrip(opts, ARRAY(BIGINT()));
  testRoundTrip(opts, ARRAY(VARCHAR()));
}

TEST_F(VectorSaverTest, map) {
  auto opts = fuzzerOptions();

  testRoundTrip(opts, MAP(INTEGER(), REAL()));
  testRoundTrip(opts, MAP(BIGINT(), VARCHAR()));

  // Add some nulls.
  opts.nullRatio = 0.1;
  testRoundTrip(opts, MAP(INTEGER(), REAL()));
  testRoundTrip(opts, MAP(BIGINT(), VARCHAR()));
}

TEST_F(VectorSaverTest, constantInteger) {
  testRoundTrip(makeConstant<int64_t>(-1234987634, 100));
  testRoundTrip(makeConstant<int32_t>(12389, 100));
  testRoundTrip(makeConstant<int16_t>(1032, 100));
  testRoundTrip(makeConstant<int8_t>(78, 100));

  // Nulls.
  testRoundTrip(makeConstant<int64_t>(std::nullopt, 100));
  testRoundTrip(makeConstant<int32_t>(std::nullopt, 100));
  testRoundTrip(makeConstant<int16_t>(std::nullopt, 100));
  testRoundTrip(makeConstant<int8_t>(std::nullopt, 100));
}

TEST_F(VectorSaverTest, constantBoolean) {
  testRoundTrip(makeConstant<bool>(true, 100));
  testRoundTrip(makeConstant<bool>(false, 100));

  // Null.
  testRoundTrip(makeConstant<bool>(std::nullopt, 100));
}

TEST_F(VectorSaverTest, constantString) {
  testRoundTrip(makeConstant<std::string>("", 100));
  testRoundTrip(makeConstant<std::string>("Short.", 100));
  testRoundTrip(
      makeConstant<std::string>("This is somewhat longer string.", 100));

  // Null.
  testRoundTrip(makeConstant<std::string>(std::nullopt, 100));
}

TEST_F(VectorSaverTest, constantUnknown) {
  testRoundTrip(BaseVector::createNullConstant(UNKNOWN(), 100, pool()));
}

TEST_F(VectorSaverTest, constantMap) {
  auto opts = fuzzerOptions();
  SCOPED_TRACE(fmt::format("seed: {}", seed_));

  VectorFuzzer fuzzer(opts, pool(), seed_);
  auto mapType = MAP(INTEGER(), VARCHAR());
  auto flatVector = fuzzer.fuzzFlat(mapType);
  testRoundTrip(BaseVector::wrapInConstant(100, 17, flatVector));

  // Null.
  testRoundTrip(BaseVector::createNullConstant(mapType, 100, pool()));
}

TEST_F(VectorSaverTest, constantArray) {
  auto opts = fuzzerOptions();
  SCOPED_TRACE(fmt::format("seed: {}", seed_));

  VectorFuzzer fuzzer(opts, pool(), seed_);
  auto arrayType = ARRAY(BIGINT());
  auto flatVector = fuzzer.fuzzFlat(arrayType);
  testRoundTrip(BaseVector::wrapInConstant(100, 23, flatVector));

  // Null.
  testRoundTrip(BaseVector::createNullConstant(arrayType, 100, pool()));
}

TEST_F(VectorSaverTest, constantRow) {
  auto opts = fuzzerOptions();
  SCOPED_TRACE(fmt::format("seed: {}", seed_));

  VectorFuzzer fuzzer(opts, pool(), seed_);
  auto rowType = ROW({"a", "b"}, {INTEGER(), REAL()});
  auto flatVector = fuzzer.fuzzRow(rowType);
  testRoundTrip(BaseVector::wrapInConstant(100, 51, flatVector));

  // Null.
  testRoundTrip(BaseVector::createNullConstant(rowType, 100, pool()));
}

TEST_F(VectorSaverTest, dictionaryBigint) {
  // No nulls. One level.
  auto data = wrapInDictionary(
      makeIndicesInReverse(5), makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5}));
  testRoundTrip(data);

  // No nulls. 2 levels.
  data = wrapInDictionary(
      makeIndicesInReverse(5),
      wrapInDictionary(
          makeIndicesInReverse(5),
          makeFlatVector<int64_t>({0, 1, 2, 3, 4, 5})));
  testRoundTrip(data);

  // Nulls.
  data = BaseVector::wrapInDictionary(
      makeNulls(100, nullEvery(7)),
      makeIndicesInReverse(100),
      100,
      makeFlatVector<int64_t>(100, [](auto row) { return row; }));
  testRoundTrip(data);

  data = wrapInDictionary(
      makeOddIndices(40),
      BaseVector::wrapInDictionary(
          makeNulls(100, nullEvery(7)),
          makeIndicesInReverse(100),
          100,
          makeFlatVector<int64_t>(100, [](auto row) { return row; })));
  testRoundTrip(data);

  // All-nulls dictionary vector over empty base vector.
  data = std::make_shared<DictionaryVector<int64_t>>(
      pool(),
      makeNulls(100, nullEvery(1)),
      100,
      BaseVector::create(BIGINT(), 0, pool()),
      makeIndices(100, [](auto /* row */) { return 0; }));
  testRoundTrip(data);
}

TEST_F(VectorSaverTest, dictionaryArray) {
  auto opts = fuzzerOptions();
  SCOPED_TRACE(fmt::format("seed: {}", seed_));

  VectorFuzzer fuzzer(opts, pool(), seed_);
  auto flatVector = fuzzer.fuzzFlat(ARRAY(INTEGER()));

  testRoundTrip(fuzzer.fuzzDictionary(flatVector));
  testRoundTrip(fuzzer.fuzzDictionary(flatVector, 64));
  testRoundTrip(fuzzer.fuzzDictionary(flatVector, 512));

  // 2 levels of dictionary.
  testRoundTrip(fuzzer.fuzzDictionary(fuzzer.fuzzDictionary(flatVector)));

  // Array vector with dictionary-encoded elements vector.
  auto elementsVector = fuzzer.fuzzDictionary(fuzzer.fuzzFlat(INTEGER()));
  // 0, 2, 4,...
  auto offsets = makeEvenIndices(64);
  auto sizes = makeIndices(64, [](auto /* row */) { return 2; });

  testRoundTrip(std::make_shared<ArrayVector>(
      pool(),
      ARRAY(INTEGER()),
      makeNulls(64, nullEvery(7)),
      64,
      offsets,
      sizes,
      elementsVector));
}

TEST_F(VectorSaverTest, dictionaryMap) {
  auto opts = fuzzerOptions();
  SCOPED_TRACE(fmt::format("seed: {}", seed_));

  VectorFuzzer fuzzer(opts, pool(), seed_);
  auto flatVector = fuzzer.fuzzFlat(MAP(INTEGER(), VARCHAR()));

  testRoundTrip(fuzzer.fuzzDictionary(flatVector));
  testRoundTrip(fuzzer.fuzzDictionary(flatVector, 64));
  testRoundTrip(fuzzer.fuzzDictionary(flatVector, 512));

  // 2 levels of dictionary.
  testRoundTrip(fuzzer.fuzzDictionary(fuzzer.fuzzDictionary(flatVector)));
}

TEST_F(VectorSaverTest, dictionaryRow) {
  auto opts = fuzzerOptions();
  SCOPED_TRACE(fmt::format("seed: {}", seed_));

  VectorFuzzer fuzzer(opts, pool(), seed_);
  auto flatVector =
      fuzzer.fuzzFlat(ROW({"a", "b", "c"}, {INTEGER(), REAL(), BOOLEAN()}));

  testRoundTrip(fuzzer.fuzzDictionary(flatVector));
  testRoundTrip(fuzzer.fuzzDictionary(flatVector, 64));
  testRoundTrip(fuzzer.fuzzDictionary(flatVector, 512));

  // 2 levels of dictionary.
  testRoundTrip(fuzzer.fuzzDictionary(fuzzer.fuzzDictionary(flatVector)));
}

TEST_F(VectorSaverTest, LazyVector) {
  auto opts = fuzzerOptions();
  opts.nullRatio = 0.5;
  SCOPED_TRACE(fmt::format("seed: {}", seed_));

  VectorFuzzer fuzzer(opts, pool(), seed_);
  auto flatVector =
      fuzzer.fuzzFlat(ROW({"a", "b", "c"}, {INTEGER(), REAL(), BOOLEAN()}));

  testRoundTripOfLazy(
      fuzzer.wrapInLazyVector(fuzzer.fuzzDictionary(flatVector)), fuzzer);
  testRoundTripOfLazy(
      fuzzer.wrapInLazyVector(fuzzer.fuzzDictionary(flatVector, 64)), fuzzer);
  testRoundTripOfLazy(
      fuzzer.wrapInLazyVector(fuzzer.fuzzDictionary(flatVector, 512)), fuzzer);

  // 2 levels of dictionary.
  testRoundTripOfLazy(
      fuzzer.wrapInLazyVector(
          fuzzer.fuzzDictionary(fuzzer.fuzzDictionary(flatVector))),
      fuzzer);
}

TEST_F(VectorSaverTest, stdVector) {
  std::vector<column_index_t> intVector = {1, 2, 3, 4, 5};
  auto path = exec::test::TempFilePath::create();
  saveStdVectorToFile<column_index_t>(intVector, path->path.c_str());
  auto copy = restoreStdVectorFromFile<column_index_t>(path->path.c_str());
  ASSERT_EQ(intVector, copy);
}

namespace {
struct VectorSaverInfo {
  // Path to directory where to store the vector.
  const char* path;

  // Vector to store.
  BaseVector* vector;
};
} // namespace

/// A demonstration of using VectorSaver to save 'current' vector being
/// processed to disk in case of an exception.
TEST_F(VectorSaverTest, exceptionContext) {
  auto tempDirectory = exec::test::TempDirectoryPath::create();

  auto messageFunction = [](VeloxException::Type /*exceptionType*/,
                            auto* arg) -> std::string {
    auto* info = static_cast<VectorSaverInfo*>(arg);
    auto filePath = common::generateTempFilePath(info->path, "vector");
    if (!filePath.has_value()) {
      return "Cannot generate file path to store the vector.";
    }

    std::ofstream outputFile(filePath.value(), std::ofstream::binary);
    saveVector(*info->vector, outputFile);
    outputFile.close();

    return filePath.value();
  };

  VectorPtr data = makeFlatVector<int64_t>({1, 2, 3, 4, 5});
  VectorSaverInfo info{tempDirectory.get()->path.c_str(), data.get()};
  {
    ExceptionContextSetter context({messageFunction, &info});
    try {
      VELOX_FAIL("Test failure.");
    } catch (VeloxRuntimeError& e) {
      auto path = e.context();
      std::ifstream inputFile(path, std::ifstream::binary);
      ASSERT_FALSE(inputFile.fail()) << "Cannot open file: " << path;
      auto copy = restoreVector(inputFile, pool());
      inputFile.close();

      // Verify encodings and data recursively.
      assertEqualEncodings(data, copy);
    }
  }
}

TEST_F(VectorSaverTest, multipleVectors) {
  // Save and restore multiple vectors to/from a single binary.

  std::vector<VectorPtr> vectors = {
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      makeFlatVector<int64_t>({10, 11, 12, 13, 15, 16, 17}),
      makeConstant<double>(1.5, 100),
  };

  std::vector<SelectivityVector> rows;
  for (const auto& vector : vectors) {
    rows.emplace_back(vector->size());
  }

  std::ostringstream out;

  for (auto i = 0; i < vectors.size(); ++i) {
    saveVector(*vectors[i], out);
    saveSelectivityVector(rows[i], out);
  }

  std::istringstream in(out.str());
  for (auto i = 0; i < vectors.size(); ++i) {
    assertEqualVectors(vectors[i], restoreVector(in, pool()));
    ASSERT_EQ(rows[i], restoreSelectivityVector(in));
  }

  ASSERT_EQ(out.str().size(), in.tellg());
}
} // namespace facebook::velox::test
