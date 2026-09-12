/*
 * Copyright (a) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/SkewResilientRadixPartitioner.h"
#include <gtest/gtest.h>
#include <random>

namespace facebook::velox::exec {
namespace {

TEST(SkewResilientRadixPartitionerTest, MemoryAlignmentCheck) {
  SkewResilientRadixPartitioner partitioner;
  for (size_t i = 0; i < kDefaultNumBuckets; ++i) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(partitioner.heads[i]->data);
    EXPECT_EQ(addr % kCacheLineAlignment, 0)
        << "Bucket " << i << " memory is not 64-byte aligned!";
  }
}

TEST(SkewResilientRadixPartitionerTest, ZipfianSkewBenchmark) {
  constexpr size_t N = 5000000;
  std::vector<RadixTuple> uniform_data(N);
  std::vector<RadixTuple> skewed_data(N);

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<uint64_t> dist(0, 1ULL << 32);

  for (size_t i = 0; i < N; ++i) {
    uniform_data[i] = {dist(rng), i};
  }

  for (size_t i = 0; i < N; ++i) {
    if (i < N * 0.70) {
      skewed_data[i] = {256, i}; // 70% Zipfian Skew
    } else {
      skewed_data[i] = {dist(rng), i};
    }
  }

  SkewResilientRadixPartitioner partitioner_uniform;
  auto t0 = std::chrono::high_resolution_clock::now();
  partitioner_uniform.partition(uniform_data);
  auto t1 = std::chrono::high_resolution_clock::now();
  double uniform_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  SkewResilientRadixPartitioner partitioner_skewed;
  auto t2 = std::chrono::high_resolution_clock::now();
  partitioner_skewed.partition(skewed_data);
  auto t3 = std::chrono::high_resolution_clock::now();
  double skewed_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::cout << "[Velox Benchmark] Uniform 5M Latency: " << uniform_ms << " ms\n";
  std::cout << "[Velox Benchmark] Skewed 5M Latency:  " << skewed_ms << " ms\n";

  EXPECT_GT(uniform_ms, skewed_ms);
}

} // namespace
} // namespace facebook::velox::exec
