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

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

constexpr int kDecodeIters = 100;

struct Result {
  std::string encoding;
  std::string pattern;
  uint32_t rawBytes;
  uint32_t encodedBytes;
  double decodeNsPerElement;
};

template <typename EncodingT, typename T>
Result measure(
    const std::string& encodingName,
    const std::string& patternName,
    EncodingType encodingType,
    const Vector<T>& data) {
  auto& pool = benchmarkPool();
  uint32_t rawBytes = data.size() * sizeof(T);

  std::string encoded;
  try {
    encoded = encodeData<EncodingT>(encodingType, data);
  } catch (...) {
    return {encodingName, patternName, rawBytes, 0, -1.0};
  }

  uint32_t encodedBytes = static_cast<uint32_t>(encoded.size());

  std::vector<T> output(data.size());
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kDecodeIters; ++i) {
    auto enc = EncodingFactory{}.create(*pool, encoded, nullFactory());
    enc->materialize(data.size(), output.data());
  }
  auto end = std::chrono::high_resolution_clock::now();
  double totalNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double nsPerElement = totalNs / (kDecodeIters * data.size());

  return {encodingName, patternName, rawBytes, encodedBytes, nsPerElement};
}

void printHeader() {
  std::cout << std::left << std::setw(22) << "Encoding" << std::setw(18)
            << "Pattern" << std::right << std::setw(10) << "Raw(KB)"
            << std::setw(12) << "Encoded(KB)" << std::setw(10) << "Ratio"
            << std::setw(14) << "Decode(ns/el)" << "\n";
  std::cout << std::string(86, '-') << "\n";
}

void printResult(const Result& r) {
  std::cout << std::left << std::setw(22) << r.encoding << std::setw(18)
            << r.pattern << std::right << std::setw(10) << std::fixed
            << std::setprecision(1) << (r.rawBytes / 1024.0);
  if (r.decodeNsPerElement < 0) {
    std::cout << std::setw(12) << "SKIP" << std::setw(10) << "N/A"
              << std::setw(14) << "N/A" << "\n";
  } else {
    double ratio = static_cast<double>(r.encodedBytes) / r.rawBytes;
    std::cout << std::setw(12) << std::fixed << std::setprecision(1)
              << (r.encodedBytes / 1024.0) << std::setw(9) << std::fixed
              << std::setprecision(3) << ratio << "x" << std::setw(14)
              << std::fixed << std::setprecision(1) << r.decodeNsPerElement
              << "\n";
  }
}

void printSeparator() {
  std::cout << std::string(86, '-') << "\n";
}

template <typename T>
struct PatternDef {
  std::string name;
  std::function<Vector<T>()> generator;
};

template <typename T>
std::vector<PatternDef<T>> makePatterns() {
  return {
      {"Random", [] { return makeRandom<T>(); }},
      {"Narrow8bit", [] { return makeNarrow<T>(8); }},
      {"Constant", [] { return makeConstant<T>(T{42}); }},
      {"MainlyConst", [] { return makeMainlyConstant<T>(T{42}); }},
      {"RunLength", [] { return makeRunLength<T>(); }},
      {"Increasing", [] { return makeIncreasing<T>(); }},
      {"LowCard64", [] { return makeLowCardinality<T>(64); }},
  };
}

template <typename EncodingT, typename T>
void runEncoding(
    const std::string& name,
    EncodingType type,
    const std::vector<PatternDef<T>>& patterns) {
  for (const auto& p : patterns) {
    auto data = p.generator();
    auto result = measure<EncodingT, T>(name, p.name, type, data);
    printResult(result);
  }
  printSeparator();
}

} // namespace

int main() {
  facebook::velox::memory::MemoryManager::initialize({});

  std::cout << "\n=== Nimble Encoding Comparison (uint32_t, " << kNumElements
            << " elements, decode averaged over " << kDecodeIters
            << " iterations) ===\n\n";

  auto patterns = makePatterns<uint32_t>();
  printHeader();

  runEncoding<TrivialEncoding<uint32_t>>(
      "Trivial", EncodingType::Trivial, patterns);
  runEncoding<FixedBitWidthEncoding<uint32_t>>(
      "FixedBitWidth", EncodingType::FixedBitWidth, patterns);
  runEncoding<RLEEncoding<uint32_t>>("RLE", EncodingType::RLE, patterns);
  runEncoding<DictionaryEncoding<uint32_t>>(
      "Dictionary", EncodingType::Dictionary, patterns);
  runEncoding<DeltaEncoding<uint32_t>>("Delta", EncodingType::Delta, patterns);
  runEncoding<ConstantEncoding<uint32_t>>(
      "Constant", EncodingType::Constant, patterns);
  runEncoding<MainlyConstantEncoding<uint32_t>>(
      "MainlyConstant", EncodingType::MainlyConstant, patterns);
  runEncoding<VarintEncoding<uint32_t>>(
      "Varint", EncodingType::Varint, patterns);

  std::cout << "\n=== Nimble Encoding Comparison (uint64_t, " << kNumElements
            << " elements, decode averaged over " << kDecodeIters
            << " iterations) ===\n\n";

  auto patterns64 = makePatterns<uint64_t>();
  printHeader();

  runEncoding<TrivialEncoding<uint64_t>>(
      "Trivial", EncodingType::Trivial, patterns64);
  runEncoding<FixedBitWidthEncoding<uint64_t>>(
      "FixedBitWidth", EncodingType::FixedBitWidth, patterns64);
  runEncoding<RLEEncoding<uint64_t>>("RLE", EncodingType::RLE, patterns64);
  runEncoding<DictionaryEncoding<uint64_t>>(
      "Dictionary", EncodingType::Dictionary, patterns64);
  runEncoding<DeltaEncoding<uint64_t>>(
      "Delta", EncodingType::Delta, patterns64);
  runEncoding<VarintEncoding<uint64_t>>(
      "Varint", EncodingType::Varint, patterns64);

  return 0;
}
