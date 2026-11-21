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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

template <typename TExecParams>
struct SimpleSize {
  VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

  template <typename TInput>
  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const TInput* /*input*/,
      const bool* legacySizeOfNull) {
    VELOX_CHECK_NOT_NULL(legacySizeOfNull);
    legacySizeOfNull_ = *legacySizeOfNull;
  }

  template <typename TInput>
  FOLLY_ALWAYS_INLINE bool callNullable(
      int32_t& out,
      const TInput* input,
      const bool* /*legacySizeOfNull*/) {
    if (input == nullptr) {
      if (legacySizeOfNull_) {
        out = -1;
        return true;
      }
      return false;
    }
    out = input->size();
    return true;
  }

 private:
  bool legacySizeOfNull_;
};

class SizeBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  SizeBenchmark() {
    parse::registerTypeResolver();
    functions::sparksql::registerFunctions("");
    registerFunction<SimpleSize, int32_t, Array<Any>, bool>({"size_simple"});
    registerFunction<SimpleSize, int32_t, Map<Any, Any>, bool>({"size_simple"});

    const auto sizeAt = [](vector_size_t row) { return 1 + row % 7; };
    const auto valueAt = [](vector_size_t row) { return row; };
    const auto keyAt = [](vector_size_t entry) { return entry % 16; };
    auto arrays = vectorMaker_.arrayVector<int32_t>(kNumRows, sizeAt, valueAt);
    flatArrayData_ = vectorMaker_.rowVector({arrays});
    nullableArrayData_ =
        vectorMaker_.rowVector({vectorMaker_.arrayVector<int32_t>(
            kNumRows, sizeAt, valueAt, [](vector_size_t row) {
              return row % 17 == 0;
            })});

    auto indices = allocateIndices(kNumRows, pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    for (vector_size_t row = 0; row < kNumRows; ++row) {
      rawIndices[row] = kNumRows - row - 1;
    }
    auto nulls = allocateNulls(kNumRows, pool());
    auto* rawNulls = nulls->asMutable<uint64_t>();
    for (vector_size_t row = 0; row < kNumRows; row += 17) {
      bits::setNull(rawNulls, row);
    }
    dictionaryArrayData_ = vectorMaker_.rowVector(
        {BaseVector::wrapInDictionary(nulls, indices, kNumRows, arrays)});
    constantArrayData_ = vectorMaker_.rowVector(
        {BaseVector::wrapInConstant(kNumRows, 1, arrays)});

    mapData_ = vectorMaker_.rowVector({vectorMaker_.mapVector<int32_t, int32_t>(
        kNumRows, sizeAt, keyAt, valueAt)});
    auto nullableMaps = vectorMaker_.mapVector<int32_t, int32_t>(
        kNumRows, sizeAt, keyAt, valueAt, [](vector_size_t row) {
          return row % 17 == 0;
        });
    nullableMapData_ = vectorMaker_.rowVector({nullableMaps});
    dictionaryMapData_ = vectorMaker_.rowVector({BaseVector::wrapInDictionary(
        nulls, indices, kNumRows, mapData_->childAt(0))});
    constantMapData_ = vectorMaker_.rowVector(
        {BaseVector::wrapInConstant(kNumRows, 1, mapData_->childAt(0))});
    flatMapData_ =
        vectorMaker_.rowVector({FlatMapVector::fromMapVector(*nullableMaps)});

    for (const auto legacy : {false, true}) {
      for (const auto& data :
           {flatArrayData_,
            nullableArrayData_,
            dictionaryArrayData_,
            constantArrayData_,
            mapData_,
            nullableMapData_,
            dictionaryMapData_,
            constantMapData_}) {
        verify(data, data, legacy);
      }
      // The simple MAP reader requires MapVector, not FlatMapVector.
      verify(flatMapData_, nullableMapData_, legacy);
    }
  }

  size_t run(
      size_t iterations,
      const std::string& function,
      const RowVectorPtr& data,
      bool legacySizeOfNull,
      bool sparse = false) {
    folly::BenchmarkSuspender suspender;
    auto expression = compileExpression(
        function + (legacySizeOfNull ? "(c0, true)" : "(c0, false)"),
        asRowType(data->type()));
    SelectivityVector rows(data->size(), !sparse);
    if (sparse) {
      for (vector_size_t row = 1; row < data->size(); row += 19) {
        rows.setValid(row, true);
      }
      rows.updateBounds();
      vector_size_t numNulls = 0;
      rows.applyToSelected([&](vector_size_t row) {
        numNulls += data->childAt(0)->isNullAt(row);
      });
      VELOX_CHECK_GT(
          numNulls, 0, "Sparse size benchmark must select null inputs");
      VELOX_CHECK_LT(
          numNulls,
          rows.countSelected(),
          "Sparse size benchmark must select non-null inputs");
    }
    const auto numSelected = rows.countSelected();
    suspender.dismiss();

    size_t numRows = 0;
    for (size_t i = 0; i < iterations; ++i) {
      auto result = evaluate(expression, data, rows);
      folly::doNotOptimizeAway(result);
      numRows += numSelected;
    }
    return numRows;
  }

  const RowVectorPtr& flatArrayData() const {
    return flatArrayData_;
  }

  const RowVectorPtr& nullableArrayData() const {
    return nullableArrayData_;
  }

  const RowVectorPtr& dictionaryArrayData() const {
    return dictionaryArrayData_;
  }

  const RowVectorPtr& constantArrayData() const {
    return constantArrayData_;
  }

  const RowVectorPtr& mapData() const {
    return mapData_;
  }

  const RowVectorPtr& nullableMapData() const {
    return nullableMapData_;
  }

  const RowVectorPtr& dictionaryMapData() const {
    return dictionaryMapData_;
  }

  const RowVectorPtr& constantMapData() const {
    return constantMapData_;
  }

  const RowVectorPtr& flatMapData() const {
    return flatMapData_;
  }

 private:
  void verify(
      const RowVectorPtr& data,
      const RowVectorPtr& referenceData,
      bool legacySizeOfNull) {
    const auto arguments = legacySizeOfNull ? "(c0, true)" : "(c0, false)";
    auto baseline =
        evaluate("size_simple" + std::string(arguments), referenceData);
    auto candidate = evaluate("size" + std::string(arguments), data);
    facebook::velox::test::assertEqualVectors(baseline, candidate);
    VELOX_CHECK(
        !testing::Test::HasFailure(),
        "Size benchmark reference comparison failed");
  }

  static constexpr vector_size_t kNumRows = 10'024;
  RowVectorPtr flatArrayData_;
  RowVectorPtr nullableArrayData_;
  RowVectorPtr dictionaryArrayData_;
  RowVectorPtr constantArrayData_;
  RowVectorPtr mapData_;
  RowVectorPtr nullableMapData_;
  RowVectorPtr dictionaryMapData_;
  RowVectorPtr constantMapData_;
  RowVectorPtr flatMapData_;
};

std::unique_ptr<SizeBenchmark> benchmark;

BENCHMARK(simpleFlat, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->flatArrayData(), false));
}

BENCHMARK_RELATIVE(vectorFlat, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->flatArrayData(), false));
}

BENCHMARK(simpleNullableArray, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->nullableArrayData(), false));
}

BENCHMARK_RELATIVE(vectorNullableArray, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size", benchmark->nullableArrayData(), false));
}

BENCHMARK(simpleNullableArrayLegacy, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->nullableArrayData(), true));
}

BENCHMARK_RELATIVE(vectorNullableArrayLegacy, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->nullableArrayData(), true));
}

BENCHMARK(simpleDictionary, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->dictionaryArrayData(), false));
}

BENCHMARK_RELATIVE(vectorDictionary, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size", benchmark->dictionaryArrayData(), false));
}

BENCHMARK(simpleDictionaryLegacy, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->dictionaryArrayData(), true));
}

BENCHMARK_RELATIVE(vectorDictionaryLegacy, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size", benchmark->dictionaryArrayData(), true));
}

BENCHMARK(simpleConstantArray, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->constantArrayData(), false));
}

BENCHMARK_RELATIVE(vectorConstantArray, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size", benchmark->constantArrayData(), false));
}

BENCHMARK(simpleSparseArray, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->nullableArrayData(), false, true));
}

BENCHMARK_RELATIVE(vectorSparseArray, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size", benchmark->nullableArrayData(), false, true));
}

BENCHMARK(simpleMap, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size_simple", benchmark->mapData(), false));
}

BENCHMARK_RELATIVE(vectorMap, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->mapData(), false));
}

BENCHMARK(simpleNullableMap, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->nullableMapData(), false));
}

BENCHMARK_RELATIVE(vectorNullableMap, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->nullableMapData(), false));
}

BENCHMARK(simpleNullableMapLegacy, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->nullableMapData(), true));
}

BENCHMARK_RELATIVE(vectorNullableMapLegacy, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->nullableMapData(), true));
}

BENCHMARK(simpleDictionaryMap, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->dictionaryMapData(), false));
}

BENCHMARK_RELATIVE(vectorDictionaryMap, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size", benchmark->dictionaryMapData(), false));
}

BENCHMARK(simpleDictionaryMapLegacy, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->dictionaryMapData(), true));
}

BENCHMARK_RELATIVE(vectorDictionaryMapLegacy, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->dictionaryMapData(), true));
}

BENCHMARK(simpleConstantMap, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->constantMapData(), false));
}

BENCHMARK_RELATIVE(vectorConstantMap, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->constantMapData(), false));
}

BENCHMARK(simpleSparseMapLegacy, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->nullableMapData(), true, true));
}

BENCHMARK_RELATIVE(vectorSparseMapLegacy, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size", benchmark->nullableMapData(), true, true));
}

// There is no direct FlatMap baseline: the simple MAP reader cannot consume it.
BENCHMARK(vectorFlatMap, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->flatMapData(), false));
}

BENCHMARK(vectorFlatMapLegacy, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->flatMapData(), true));
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  benchmark = std::make_unique<SizeBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
