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
#include <gtest/gtest.h>

#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox {
namespace {

class ContainsNull {
 public:
  explicit ContainsNull(const VectorPtr& vector)
      : typeKind_{vector->typeKind()}, decoded_{*vector} {
    auto base = decoded_.base();
    switch (typeKind_) {
      case TypeKind::ARRAY: {
        auto arrayBase = base->as<ArrayVector>();
        children_.push_back(ContainsNull(arrayBase->elements()));
        rawOffsets_ = arrayBase->rawOffsets();
        rawSizes_ = arrayBase->rawSizes();
        break;
      }
      case TypeKind::MAP: {
        auto mapBase = base->as<MapVector>();
        children_.push_back(ContainsNull(mapBase->mapKeys()));
        children_.push_back(ContainsNull(mapBase->mapValues()));
        rawOffsets_ = mapBase->rawOffsets();
        rawSizes_ = mapBase->rawSizes();
        break;
      }
      case TypeKind::ROW: {
        auto rowBase = base->as<RowVector>();
        for (const auto& child : rowBase->children()) {
          children_.push_back(ContainsNull(child));
        }
        break;
      }
      default:;
    }
  }

  bool isNullAt(vector_size_t row) {
    return decoded_.isNullAt(row);
  }

  bool containsNull(vector_size_t row) {
    VELOX_CHECK(!decoded_.isNullAt(row));
    return containsNull(row, true);
  }

 private:
  bool containsNullInternal(vector_size_t row) {
    switch (typeKind_) {
      case TypeKind::ARRAY:
        [[fallthrough]];
      case TypeKind::MAP: {
        if (decoded_.isNullAt(row)) {
          return true;
        }

        auto baseRow = decoded_.index(row);
        auto offset = rawOffsets_[baseRow];
        auto size = rawSizes_[baseRow];
        for (auto& child : children_) {
          if (child.containsNull(offset, size)) {
            return true;
          }
        }

        return false;
      }
      case TypeKind::ROW: {
        if (decoded_.isNullAt(row)) {
          return true;
        }

        auto baseRow = decoded_.index(row);
        for (auto& child : children_) {
          if (child.containsNullInternal(baseRow)) {
            return true;
          }
        }

        return false;
      }
      default:
        return decoded_.isNullAt(row);
    }
  }

  bool containsNull(vector_size_t offset, vector_size_t size) {
    for (auto row = offset; row < offset + size; ++row) {
      if (containsNullInternal(row)) {
        return true;
      }
    }

    return false;
  }

  const TypeKind typeKind_;
  DecodedVector decoded_;
  std::vector<ContainsNull> children_;
  const vector_size_t* rawOffsets_;
  const vector_size_t* rawSizes_;
};

class ContainsNullBenchmark : public test::VectorTestBase {
 public:
  ContainsNullBenchmark() {
    VectorFuzzer::Options opts;
    opts.vectorSize = 10'000;
    opts.containerLength = 5;
    opts.nullRatio = 0.1;
    opts.containerHasNulls = true;
    opts.dictionaryHasNulls = false;
    opts.containerVariableLength = true;
    opts.complexElementsMaxSize = 1'000;

    VectorFuzzer fuzzer(opts, pool());

    data_ =
        fuzzer.fuzz(ARRAY(MAP(INTEGER(), ROW({DOUBLE(), ARRAY(VARCHAR())}))));
  }

  void run1() {
    int32_t cnt = 0;
    for (auto i = 0; i < data_->size(); ++i) {
      if (!data_->isNullAt(i) && data_->containsNullAt(i)) {
        ++cnt;
      }
    }

    folly::doNotOptimizeAway(cnt);
  }

  void run2() {
    int32_t cnt = 0;

    ContainsNull containsNull(data_);
    for (auto i = 0; i < data_->size(); ++i) {
      if (!containsNull.isNullAt(i) && containsNull.containsNull(i)) {
        ++cnt;
      }
    }

    folly::doNotOptimizeAway(cnt);
  }

 private:
  VectorPtr data_;
};

std::unique_ptr<ContainsNullBenchmark> benchmark;

BENCHMARK(x) {
  benchmark->run1();
}

BENCHMARK_RELATIVE(y) {
  benchmark->run2();
}

} // namespace
} // namespace facebook::velox

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  facebook::velox::benchmark =
      std::make_unique<facebook::velox::ContainsNullBenchmark>();
  folly::runBenchmarks();
  facebook::velox::benchmark.reset();
  return 0;
}
