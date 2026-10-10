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
    dictionaryArrayData_ = vectorMaker_.rowVector({BaseVector::wrapInDictionary(
        std::move(nulls), std::move(indices), kNumRows, arrays)});

    mapData_ = vectorMaker_.rowVector({vectorMaker_.mapVector<int32_t, int32_t>(
        kNumRows, sizeAt, valueAt, valueAt)});
    auto nullableMaps = vectorMaker_.mapVector<int32_t, int32_t>(
        kNumRows, sizeAt, valueAt, valueAt, [](vector_size_t row) {
          return row % 17 == 0;
        });
    nullableMapData_ = vectorMaker_.rowVector({nullableMaps});
    flatMapData_ = vectorMaker_.rowVector(
        {vectorMaker_.flatMapVector<int32_t>(nullableMaps)});

    verify(flatArrayData_, false);
    verify(nullableArrayData_, false);
    verify(nullableArrayData_, true);
    verify(dictionaryArrayData_, false);
    verify(dictionaryArrayData_, true);
    verify(mapData_, false);
    verify(nullableMapData_, false);
    verify(nullableMapData_, true);
    verify(flatMapData_, false);
    verify(flatMapData_, true);
  }

  size_t run(
      size_t iterations,
      const std::string& function,
      const RowVectorPtr& data,
      bool legacySizeOfNull) {
    folly::BenchmarkSuspender suspender;
    auto expression = compileExpression(
        function + (legacySizeOfNull ? "(c0, true)" : "(c0, false)"),
        asRowType(data->type()));
    suspender.dismiss();

    size_t rows = 0;
    for (size_t i = 0; i < iterations; ++i) {
      rows += evaluate(expression, data)->size();
    }
    return rows;
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

  const RowVectorPtr& mapData() const {
    return mapData_;
  }

  const RowVectorPtr& nullableMapData() const {
    return nullableMapData_;
  }

  const RowVectorPtr& flatMapData() const {
    return flatMapData_;
  }

 private:
  void verify(const RowVectorPtr& data, bool legacySizeOfNull) {
    const auto arguments = legacySizeOfNull ? "(c0, true)" : "(c0, false)";
    auto baseline = evaluate("size_simple" + std::string(arguments), data);
    auto candidate = evaluate("size" + std::string(arguments), data);
    facebook::velox::test::assertEqualVectors(baseline, candidate);
  }

  static constexpr vector_size_t kNumRows = 10'024;
  RowVectorPtr flatArrayData_;
  RowVectorPtr nullableArrayData_;
  RowVectorPtr dictionaryArrayData_;
  RowVectorPtr mapData_;
  RowVectorPtr nullableMapData_;
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

BENCHMARK(simpleFlatMap, iterations) {
  folly::doNotOptimizeAway(benchmark->run(
      iterations, "size_simple", benchmark->flatMapData(), false));
}

BENCHMARK_RELATIVE(vectorFlatMap, iterations) {
  folly::doNotOptimizeAway(
      benchmark->run(iterations, "size", benchmark->flatMapData(), false));
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
