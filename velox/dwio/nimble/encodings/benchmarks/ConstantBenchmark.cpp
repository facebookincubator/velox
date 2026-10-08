/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "folly/Benchmark.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

// ConstantEncoding requires all values to be identical, so only the Constant
// pattern produces valid encoded output. We benchmark encode on Constant data
// and decode on the same.

BENCHMARK(Constant_Encode, iters) {
  Vector<uint32_t> data{benchmarkPool().get()};
  BENCHMARK_SUSPEND {
    data = makeConstant<uint32_t>(42);
  }
  encodeBenchmark<ConstantEncoding<uint32_t>>(
      EncodingType::Constant, data, iters);
}

BENCHMARK(Constant_Decode, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    auto data = makeConstant<uint32_t>(42);
    encoded =
        encodeData<ConstantEncoding<uint32_t>>(EncodingType::Constant, data);
  }
  decodeBenchmark<uint32_t>(encoded, kNumElements, iters);
}

int main() {
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
