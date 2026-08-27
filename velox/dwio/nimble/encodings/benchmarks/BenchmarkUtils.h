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
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "folly/Random.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

namespace facebook::nimble::benchmarks {

inline constexpr uint32_t kNumElements = 100000;

inline std::shared_ptr<velox::memory::MemoryPool>& benchmarkPool() {
  static auto pool = velox::memory::memoryManager()->addLeafPool("benchmark");
  return pool;
}

inline auto nullFactory() {
  return [](uint32_t) -> void* { return nullptr; };
}

// Random uniform data — worst case for most specialized encodings.
template <typename T>
Vector<T> makeRandom(uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    if constexpr (sizeof(T) <= 4) {
      data[i] = static_cast<T>(folly::Random::secureRand32());
    } else {
      data[i] = static_cast<T>(folly::Random::secureRand64());
    }
  }
  return data;
}

// Narrow bit-width data — values fit in bitWidth bits.
template <typename T>
Vector<T> makeNarrow(int bitWidth, uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  using U = std::make_unsigned_t<T>;
  U mask = (bitWidth >= static_cast<int>(sizeof(T) * 8))
      ? static_cast<U>(~U{0})
      : (static_cast<U>(1) << bitWidth) - 1;
  for (uint32_t i = 0; i < n; ++i) {
    if constexpr (sizeof(T) <= 4) {
      data[i] = static_cast<T>(folly::Random::secureRand32() & mask);
    } else {
      data[i] = static_cast<T>(folly::Random::secureRand64() & mask);
    }
  }
  return data;
}

// All values identical.
template <typename T>
Vector<T> makeConstant(T value, uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = value;
  }
  return data;
}

// 95% one dominant value, 5% random outliers.
template <typename T>
Vector<T> makeMainlyConstant(T dominant, uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    if constexpr (sizeof(T) <= 4) {
      data[i] = (folly::Random::secureRand32() % 100 < 95)
          ? dominant
          : static_cast<T>(folly::Random::secureRand32());
    } else {
      data[i] = (folly::Random::secureRand32() % 100 < 95)
          ? dominant
          : static_cast<T>(folly::Random::secureRand64());
    }
  }
  return data;
}

// Runs of 10-60 identical values — ideal for RLE.
template <typename T>
Vector<T> makeRunLength(uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  T val = static_cast<T>(folly::Random::secureRand32() % 1000);
  uint32_t i = 0;
  while (i < n) {
    uint32_t runLen =
        std::min<uint32_t>(10 + folly::Random::secureRand32() % 50, n - i);
    for (uint32_t j = 0; j < runLen; ++j) {
      data[i + j] = val;
    }
    i += runLen;
    val = static_cast<T>(folly::Random::secureRand32() % 1000);
  }
  return data;
}

// Monotonically increasing with small random deltas — ideal for Delta/FOR.
template <typename T>
Vector<T> makeIncreasing(uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  T val = 0;
  for (uint32_t i = 0; i < n; ++i) {
    val += static_cast<T>(folly::Random::secureRand32() % 10);
    data[i] = val;
  }
  return data;
}

// Low cardinality — ideal for Dictionary.
template <typename T>
Vector<T> makeLowCardinality(uint32_t cardinality, uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = static_cast<T>(folly::Random::secureRand32() % cardinality);
  }
  return data;
}

// Sparse bool — 5% true, 95% false.
inline Vector<bool> makeSparseBool(uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<bool> data{pool.get()};
  data.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = (folly::Random::secureRand32() % 100 < 5);
  }
  return data;
}

// Dense bool — 50% true, 50% false.
inline Vector<bool> makeDenseBool(uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<bool> data{pool.get()};
  data.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = (folly::Random::secureRand32() % 2 == 0);
  }
  return data;
}

inline std::unique_ptr<EncodingSelectionPolicyBase> makeDefaultPolicy(
    DataType dataType) {
  static ManualEncodingSelectionPolicyFactory factory{
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
      std::nullopt};
  return factory.createPolicy(dataType);
}

// Encode helper: encode data with a specific encoding type.
template <typename EncodingT, typename T>
std::string encodeData(
    EncodingType encodingType,
    const Vector<T>& data,
    const Encoding::Options& options = {}) {
  using P = typename TypeTraits<T>::physicalType;
  auto& pool = benchmarkPool();
  Buffer buffer{*pool};
  auto values =
      std::span<const P>(reinterpret_cast<const P*>(data.data()), data.size());
  EncodingSelectionResult result{.encodingType = encodingType};
  auto selection = EncodingSelection<P>{
      std::move(result),
      Statistics<P>::create(values),
      makeDefaultPolicy(TypeTraits<T>::dataType)};
  auto enc = EncodingT::encode(selection, values, buffer, options);
  return std::string{enc.data(), enc.size()};
}

// Decode benchmark loop.
template <typename T>
void decodeBenchmark(const std::string& encoded, uint32_t n, uint32_t iters) {
  auto& pool = benchmarkPool();
  std::vector<T> output(n);
  while (iters--) {
    auto encoding = EncodingFactory{}.create(*pool, encoded, nullFactory());
    encoding->materialize(n, output.data());
  }
}

// Encode benchmark loop.
template <typename EncodingT, typename T>
void encodeBenchmark(
    EncodingType encodingType,
    const Vector<T>& data,
    uint32_t iters,
    const Encoding::Options& options = {}) {
  using P = typename TypeTraits<T>::physicalType;
  auto& pool = benchmarkPool();
  auto values =
      std::span<const P>(reinterpret_cast<const P*>(data.data()), data.size());
  while (iters--) {
    Buffer buffer{*pool};
    EncodingSelectionResult result{.encodingType = encodingType};
    auto selection = EncodingSelection<P>{
        std::move(result),
        Statistics<P>::create(values),
        makeDefaultPolicy(TypeTraits<T>::dataType)};
    EncodingT::encode(selection, values, buffer, options);
  }
}

} // namespace facebook::nimble::benchmarks
