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
#include <gflags/gflags.h>

#include <string_view>
#include <unordered_map>

#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int64(fuzzer_seed, 99887766, "Seed for random input dataset generator");

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {

class DecodedVectorBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  explicit DecodedVectorBenchmark(size_t vectorSize)
      : FunctionBenchmarkBase(), vectorSize_(vectorSize), rows_(vectorSize) {
    makeVectors();
    makeNullableVectors();
    makeRowSelections();
    makeSmallVectors();
  }

  // Runs a fast path over a flat vector (no decoding).
  size_t runFlat() {
    const int64_t* flatBuffer = flatVector_->values()->as<int64_t>();
    size_t sum = 0;

    for (auto i = 0; i < vectorSize_; i++) {
      sum += flatBuffer[i];
    }
    folly::doNotOptimizeAway(sum);
    return vectorSize_;
  }

  // Runs over a decoded flat vector.
  size_t decodedRunFlat() {
    folly::BenchmarkSuspender suspender;
    DecodedVector decodedVector(*flatVector_, rows_);
    suspender.dismiss();
    decodedRun(decodedVector);
    return vectorSize_;
  }

  // Runs over a decoded constant vector.
  size_t decodedRunConstant() {
    folly::BenchmarkSuspender suspender;
    DecodedVector decodedVector(*constantVector_, rows_);
    suspender.dismiss();
    decodedRun(decodedVector);
    return vectorSize_;
  }

  // Runs over a decoded dictionary vector.
  size_t decodedRunDict() {
    folly::BenchmarkSuspender suspender;
    DecodedVector decodedVector(*dictionaryVector_, rows_);
    suspender.dismiss();
    decodedRun(decodedVector);
    return vectorSize_;
  }

  // Runs over a decoded nested dictionary vector, with 5 layers of indirection.
  size_t decodedRunDict5Nested() {
    folly::BenchmarkSuspender suspender;
    DecodedVector decodedVector(*dictionaryNestedVector_, rows_);
    suspender.dismiss();
    decodedRun(decodedVector);
    return vectorSize_;
  }

  // Measure time to decode a flat vector.
  void decodeFlat() {
    DecodedVector decodedVector(*flatVector_, rows_);
  }

  // Measure time to decode a constant vector.
  void decodeConstant() {
    DecodedVector decodedVector(*constantVector_, rows_);
  }

  // Measure time to decode a dictionary vector.
  void decodeDictionary() {
    DecodedVector decodedVector(*dictionaryVector_, rows_);
  }

  // Measure time to decode a 5-way nested dictionary vector.
  void decodeDictionary5Nested() {
    DecodedVector decodedVector(*dictionaryNestedVector_, rows_);
  }

  // Measure time to decode each nullable vector, and to decode it and resolve
  // its nulls into row order. Each pair differs by what nulls() costs.
  void decodeNullableFlat() {
    decode(nullableFlatVector_, nullptr);
  }

  void nullsNullableFlat() {
    decodeAndNulls(nullableFlatVector_, nullptr);
  }

  void decodeDictionaryWrapperNulls() {
    decode(dictionaryWrapperNullsVector_, nullptr);
  }

  void nullsDictionaryWrapperNulls() {
    decodeAndNulls(dictionaryWrapperNullsVector_, nullptr);
  }

  void decodeDictionaryBaseNulls() {
    decode(dictionaryBaseNullsVector_, nullptr);
  }

  void nullsDictionaryBaseNulls() {
    decodeAndNulls(dictionaryBaseNullsVector_, nullptr);
  }

  void decodeDictionaryBothNulls() {
    decode(dictionaryBothNullsVector_, nullptr);
  }

  void nullsDictionaryBothNulls() {
    decodeAndNulls(dictionaryBothNullsVector_, nullptr);
  }

  void decodeDictionary5NestedBaseNulls() {
    decode(dictionaryNestedBaseNullsVector_, nullptr);
  }

  void nullsDictionary5NestedBaseNulls() {
    decodeAndNulls(dictionaryNestedBaseNullsVector_, nullptr);
  }

  void decodeDictionaryBaseNullsAllSelected() {
    decode(dictionaryBaseNullsVector_, &rows_);
  }

  void nullsDictionaryBaseNullsAllSelected() {
    decodeAndNulls(dictionaryBaseNullsVector_, &rows_);
  }

  void decodeDictionaryBaseNullsHalfSelected() {
    decode(dictionaryBaseNullsVector_, &halfRows_);
  }

  void nullsDictionaryBaseNullsHalfSelected() {
    decodeAndNulls(dictionaryBaseNullsVector_, &halfRows_);
  }

  // Decodes each of the kNumSmallVectors small vectors of `shape` in turn with
  // one reused DecodedVector, after evicting the caches if `cold`. Returns the
  // number of vectors, so folly reports time per vector.
  unsigned decodeSmall(std::string_view shape, bool cold) {
    const auto& vectors = smallVectors_.at(std::string(shape));
    if (cold) {
      folly::BenchmarkSuspender suspender;
      evictCaches();
    }
    DecodedVector decodedVector;
    for (const auto& vector : vectors) {
      decodedVector.decode(*vector);
      folly::doNotOptimizeAway(decodedVector.data<int64_t>());
    }
    return vectors.size();
  }

 private:
  // Decodes 'vector', over the rows in 'rows', or over every row when 'rows' is
  // null.
  void decode(const VectorPtr& vector, const SelectivityVector* rows) {
    if (rows) {
      DecodedVector decodedVector(*vector, *rows);
      folly::doNotOptimizeAway(decodedVector);
    } else {
      DecodedVector decodedVector(*vector);
      folly::doNotOptimizeAway(decodedVector);
    }
  }

  // Same, then resolves the nulls. Decoded fresh each time: nulls() caches its
  // result.
  void decodeAndNulls(const VectorPtr& vector, const SelectivityVector* rows) {
    if (rows) {
      DecodedVector decodedVector(*vector, *rows);
      folly::doNotOptimizeAway(decodedVector.nulls(rows));
    } else {
      DecodedVector decodedVector(*vector);
      folly::doNotOptimizeAway(decodedVector.nulls());
    }
  }

  static constexpr int kNumSmallVectors = 10'000;
  static constexpr vector_size_t kSmallVectorSize = 64;

  // The vectors the scan and decode benchmarks read: no nulls anywhere.
  void makeVectors() {
    auto fuzzer = makeFuzzer(/*nullRatio=*/0);

    flatVector_ = fuzzer.fuzzFlat(BIGINT());
    constantVector_ = fuzzer.fuzzConstant(BIGINT());
    dictionaryVector_ = fuzzer.fuzzDictionary(fuzzer.fuzzFlat(BIGINT()));

    // Generate nested dictionary vector.
    dictionaryNestedVector_ = fuzzer.fuzzFlat(BIGINT());
    for (size_t i = 0; i < 5; ++i) {
      dictionaryNestedVector_ = fuzzer.fuzzDictionary(dictionaryNestedVector_);
    }
  }

  // The vectors the null benchmarks read, with nulls in about 1 of 10 rows on
  // each level that has them. Dictionaries are wrapped with
  // wrapInDictionary() rather than fuzzDictionary(), which decides on its own
  // whether to put nulls on the wrapper.
  void makeNullableVectors() {
    auto fuzzer = makeFuzzer(/*nullRatio=*/0.1);
    const auto wrap = [&](const BufferPtr& nulls, const VectorPtr& base) {
      return BaseVector::wrapInDictionary(
          nulls, fuzzer.fuzzIndices(numRows(), numRows()), numRows(), base);
    };

    nullableFlatVector_ = fuzzer.fuzzFlat(BIGINT());

    const auto wrapperNulls = fuzzer.fuzzNulls(numRows());
    dictionaryWrapperNullsVector_ = wrap(wrapperNulls, flatVector_);

    // Nulls on both levels: decoding merges the base's nulls into the
    // wrapper's, which the shape above skips.
    dictionaryBothNullsVector_ = wrap(wrapperNulls, nullableFlatVector_);

    dictionaryBaseNullsVector_ = wrap(nullptr, nullableFlatVector_);

    dictionaryNestedBaseNullsVector_ = nullableFlatVector_;
    for (size_t i = 0; i < 5; ++i) {
      dictionaryNestedBaseNullsVector_ =
          wrap(nullptr, dictionaryNestedBaseNullsVector_);
    }
  }

  // kNumSmallVectors distinct vectors of each shape the small-vector
  // benchmarks decode, kSmallVectorSize rows each: flat without nulls and
  // with nulls in about 1 of 10 rows, entirely null with no values buffer, a
  // dictionary over a flat vector, and a constant.
  void makeSmallVectors() {
    auto fuzzer = makeFuzzer(/*nullRatio=*/0, kSmallVectorSize);
    auto nullableFuzzer = makeFuzzer(/*nullRatio=*/0.1, kSmallVectorSize);
    for (int i = 0; i < kNumSmallVectors; ++i) {
      smallVectors_["flat"].push_back(fuzzer.fuzzFlat(BIGINT()));
      smallVectors_["flatWithNulls"].push_back(
          nullableFuzzer.fuzzFlat(BIGINT()));
      smallVectors_["flatAllNull"].push_back(
          std::make_shared<FlatVector<int64_t>>(
              pool(),
              BIGINT(),
              allocateNulls(kSmallVectorSize, pool(), bits::kNull),
              kSmallVectorSize,
              /*values=*/nullptr,
              std::vector<BufferPtr>{}));
      smallVectors_["dictionary"].push_back(
          fuzzer.fuzzDictionary(fuzzer.fuzzFlat(BIGINT())));
      smallVectors_["constant"].push_back(fuzzer.fuzzConstant(BIGINT()));
    }
  }

  // Writes a buffer larger than the last-level cache, so the next decodes read
  // every vector's metadata from memory.
  static void evictCaches() {
    static std::vector<char> buffer(256 << 20);
    for (size_t i = 0; i < buffer.size(); i += 64) {
      ++buffer[i];
    }
  }

  // A random half of the rows, so most words are partly selected.
  void makeRowSelections() {
    auto fuzzer = makeFuzzer(/*nullRatio=*/0.5);
    const auto selected = fuzzer.fuzzNulls(numRows());
    VELOX_CHECK_NOT_NULL(selected);
    halfRows_.setFromBits(selected->as<uint64_t>(), numRows());
  }

  // 'vectorSize_' as the row count Velox APIs take.
  vector_size_t numRows() const {
    return static_cast<vector_size_t>(vectorSize_);
  }

  VectorFuzzer makeFuzzer(double nullRatio) {
    return makeFuzzer(nullRatio, vectorSize_);
  }

  VectorFuzzer makeFuzzer(double nullRatio, size_t vectorSize) {
    VectorFuzzer::Options opts;
    opts.vectorSize = vectorSize;
    opts.nullRatio = nullRatio;
    return VectorFuzzer(opts, pool(), FLAGS_fuzzer_seed);
  }

  void decodedRun(const DecodedVector& decodedVector) {
    size_t sum = 0;
    for (auto i = 0; i < vectorSize_; i++) {
      sum += decodedVector.valueAt<int64_t>(i);
    }
    folly::doNotOptimizeAway(sum);
  }

  const size_t vectorSize_;

  VectorPtr flatVector_;
  VectorPtr constantVector_;
  VectorPtr dictionaryVector_;
  VectorPtr dictionaryNestedVector_;

  VectorPtr nullableFlatVector_;
  VectorPtr dictionaryWrapperNullsVector_;
  VectorPtr dictionaryBothNullsVector_;
  VectorPtr dictionaryBaseNullsVector_;
  VectorPtr dictionaryNestedBaseNullsVector_;

  SelectivityVector rows_;
  SelectivityVector halfRows_;

  std::unordered_map<std::string, std::vector<VectorPtr>> smallVectors_;
};

std::unique_ptr<DecodedVectorBenchmark> benchmark;

template <typename Func>
void run(Func&& func, size_t iterations = 100) {
  for (auto i = 0; i < iterations; i++) {
    func();
  }
}

BENCHMARK(scanFlat) {
  run([&] { benchmark->runFlat(); });
}

BENCHMARK(scanDecodedFlat) {
  run([&] { benchmark->decodedRunFlat(); });
}

BENCHMARK(scanDecodedConstant) {
  run([&] { benchmark->decodedRunConstant(); });
}

BENCHMARK(scanDecodedDict) {
  run([&] { benchmark->decodedRunDict(); });
}

BENCHMARK(scanDecodedDict5Nested) {
  run([&] { benchmark->decodedRunDict5Nested(); });
}

BENCHMARK_DRAW_LINE();

// For those we alwast report total runtime.
BENCHMARK(decodeFlat) {
  run([&] { benchmark->decodeFlat(); });
}

BENCHMARK(decodeConstant) {
  run([&] { benchmark->decodeConstant(); });
}

BENCHMARK(decodeDictionary) {
  run([&] { benchmark->decodeDictionary(); });
}

BENCHMARK(decodeDictionary5Nested) {
  run([&] { benchmark->decodeDictionary5Nested(); });
}

BENCHMARK_DRAW_LINE();

// nulls() on each shape, next to decoding the same vector alone: the gap is
// what nulls() costs. A flat vector and a dictionary with nulls of its own
// already hold them in row order, so those two only hand back a bitmap. A
// dictionary over a nullable base resolves each row through its indices, and
// one with nulls on both levels merges the two at decode time.
BENCHMARK(decodeNullableFlat) {
  run([&] { benchmark->decodeNullableFlat(); });
}

BENCHMARK(nullsNullableFlat) {
  run([&] { benchmark->nullsNullableFlat(); });
}

BENCHMARK(decodeDictionaryWrapperNulls) {
  run([&] { benchmark->decodeDictionaryWrapperNulls(); });
}

BENCHMARK(nullsDictionaryWrapperNulls) {
  run([&] { benchmark->nullsDictionaryWrapperNulls(); });
}

BENCHMARK(decodeDictionaryBaseNulls) {
  run([&] { benchmark->decodeDictionaryBaseNulls(); });
}

BENCHMARK(nullsDictionaryBaseNulls) {
  run([&] { benchmark->nullsDictionaryBaseNulls(); });
}

BENCHMARK(decodeDictionaryBothNulls) {
  run([&] { benchmark->decodeDictionaryBothNulls(); });
}

BENCHMARK(nullsDictionaryBothNulls) {
  run([&] { benchmark->nullsDictionaryBothNulls(); });
}

BENCHMARK(decodeDictionary5NestedBaseNulls) {
  run([&] { benchmark->decodeDictionary5NestedBaseNulls(); });
}

BENCHMARK(nullsDictionary5NestedBaseNulls) {
  run([&] { benchmark->nullsDictionary5NestedBaseNulls(); });
}

BENCHMARK(decodeDictionaryBaseNullsAllSelected) {
  run([&] { benchmark->decodeDictionaryBaseNullsAllSelected(); });
}

BENCHMARK(nullsDictionaryBaseNullsAllSelected) {
  run([&] { benchmark->nullsDictionaryBaseNullsAllSelected(); });
}

BENCHMARK(decodeDictionaryBaseNullsHalfSelected) {
  run([&] { benchmark->decodeDictionaryBaseNullsHalfSelected(); });
}

BENCHMARK(nullsDictionaryBaseNullsHalfSelected) {
  run([&] { benchmark->nullsDictionaryBaseNullsHalfSelected(); });
}

BENCHMARK_DRAW_LINE();

// Decoding many distinct small vectors, one after another, as when reading the
// children of a wide ROW a few rows at a time. The fixed cost of decode()
// dominates there, and the benchmarks above, which decode one large vector
// over and over, hide it. Reported per vector. The "Cold" variants evict the
// caches first, so every vector's metadata comes from memory, as it does when
// other work runs between decodes.
#define DECODE_SMALL_BENCHMARKS(name, shape)     \
  BENCHMARK_MULTI(decodeSmall##name) {           \
    return benchmark->decodeSmall(shape, false); \
  }                                              \
  BENCHMARK_MULTI(decodeSmall##name##Cold) {     \
    return benchmark->decodeSmall(shape, true);  \
  }

DECODE_SMALL_BENCHMARKS(Flat, "flat")
DECODE_SMALL_BENCHMARKS(FlatWithNulls, "flatWithNulls")
DECODE_SMALL_BENCHMARKS(FlatAllNull, "flatAllNull")
DECODE_SMALL_BENCHMARKS(Dictionary, "dictionary")
DECODE_SMALL_BENCHMARKS(Constant, "constant")

} // namespace

int main(int argc, char* argv[]) {
  folly::Init init{&argc, &argv};
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  benchmark = std::make_unique<DecodedVectorBenchmark>(10'000);
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
