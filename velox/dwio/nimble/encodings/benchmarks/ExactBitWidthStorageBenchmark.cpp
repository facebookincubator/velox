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

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

namespace facebook::nimble {
namespace {

constexpr uint32_t kRows = 130;
constexpr uint32_t kBaseline = 1000;
constexpr std::array<uint8_t, 18>
    kValueBits{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 20, 21, 27};
constexpr std::array<uint8_t, 8> kIndexBits{0, 1, 2, 3, 4, 5, 6, 7};

uint32_t maxResidual(uint8_t bitWidth) {
  if (bitWidth == 0) {
    return 0;
  }
  return bitWidth == 32 ? ~uint32_t{0} : ((uint32_t{1} << bitWidth) - 1);
}

uint64_t savedPercent(uint64_t nonExactBytes, uint64_t exactBytes) {
  if (nonExactBytes == 0 || nonExactBytes <= exactBytes) {
    return 0;
  }
  return ((nonExactBytes - exactBytes) * 100 + nonExactBytes / 2) /
      nonExactBytes;
}

Encoding::Options exactBitWidthOptions() {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = true;
  return options;
}

Encoding::Options nonExactBitWidthOptions() {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = false;
  return options;
}

void printHeader(std::string_view title) {
  std::cout << "\n" << title << "\n";
  std::cout << std::left << std::setw(24) << "encoding" << std::right
            << std::setw(10) << "bits" << std::setw(12) << "non-exact B"
            << std::setw(12) << "exact B" << std::setw(12) << "saved"
            << std::setw(10) << "saved %" << "\n";
}

void printRow(
    std::string_view encoding,
    uint32_t bits,
    uint64_t nonExactBytes,
    uint64_t exactBytes) {
  const int64_t saved =
      static_cast<int64_t>(nonExactBytes) - static_cast<int64_t>(exactBytes);
  std::cout << std::left << std::setw(24) << encoding << std::right
            << std::setw(10) << bits << std::setw(12) << nonExactBytes
            << std::setw(12) << exactBytes << std::setw(12) << saved
            << std::setw(10) << savedPercent(nonExactBytes, exactBytes) << "\n";
}

template <typename T>
Statistics<T> statsFor(const Vector<T>& values) {
  return Statistics<T>::create(
      std::span<const T>{values.data(), values.size()});
}

Vector<uint32_t> makeRangeData(uint8_t bitWidth) {
  Vector<uint32_t> values{benchmarks::benchmarkPool().get()};
  values.resize(kRows);
  const auto residualMax = maxResidual(bitWidth);
  for (uint32_t i = 0; i < kRows; ++i) {
    values[i] = kBaseline +
        (residualMax == 0 ? 0 : i % static_cast<uint32_t>(residualMax + 1));
  }
  values[0] = kBaseline;
  values[kRows - 1] = kBaseline + residualMax;
  return values;
}

Vector<uint32_t> makeDictionaryData(uint8_t indexBits) {
  Vector<uint32_t> values{benchmarks::benchmarkPool().get()};
  values.resize(kRows);
  const uint32_t cardinality = indexBits == 0 ? 1 : uint32_t{1} << indexBits;
  for (uint32_t i = 0; i < kRows; ++i) {
    values[i] = kBaseline + (i % cardinality);
  }
  return values;
}

Vector<uint32_t> makeRunData(uint8_t valueBits) {
  Vector<uint32_t> values{benchmarks::benchmarkPool().get()};
  values.resize(kRows);
  const auto residualMax = maxResidual(valueBits);
  uint32_t row = 0;
  uint32_t value = 0;
  while (row < kRows) {
    const uint32_t runLength = std::min<uint32_t>(2 + (value % 5), kRows - row);
    for (uint32_t i = 0; i < runLength; ++i) {
      values[row + i] = kBaseline +
          (residualMax == 0 ? 0
                            : value % static_cast<uint32_t>(residualMax + 1));
    }
    row += runLength;
    ++value;
  }
  values[kRows - 1] = kBaseline + residualMax;
  return values;
}

Vector<uint32_t> makeMainlyConstantData(uint8_t valueBits) {
  Vector<uint32_t> values{benchmarks::benchmarkPool().get()};
  values.resize(kRows);
  const auto residualMax = maxResidual(valueBits);
  for (uint32_t i = 0; i < kRows; ++i) {
    values[i] = kBaseline;
  }
  for (uint32_t i = 19; i + 1 < kRows; i += 20) {
    values[i] = kBaseline +
        (residualMax == 0 ? 0 : i % static_cast<uint32_t>(residualMax + 1));
  }
  values[kRows - 1] = kBaseline + residualMax;
  return values;
}

Vector<uint32_t> makePforData(uint8_t baseBits) {
  Vector<uint32_t> values{benchmarks::benchmarkPool().get()};
  values.resize(kRows);
  const uint32_t coveredRows = (kRows * 9) / 10;
  const auto baseMax = maxResidual(baseBits);
  for (uint32_t i = 0; i < coveredRows; ++i) {
    values[i] = kBaseline + (baseMax == 0 ? 0 : i % baseMax);
  }
  if (coveredRows > 0) {
    values[coveredRows - 1] = kBaseline + baseMax;
  }
  const uint8_t nonExactBucketBits =
      static_cast<uint8_t>(std::min<uint32_t>(((baseBits + 6) / 7) * 7, 32));
  const uint32_t outlier = nonExactBucketBits == 32
      ? ~uint32_t{0}
      : uint32_t{1} << nonExactBucketBits;
  for (uint32_t i = coveredRows; i < kRows; ++i) {
    values[i] = kBaseline + outlier + (i - coveredRows);
  }
  return values;
}

template <typename EncodingT>
void printEstimateRow(
    std::string_view label,
    uint8_t bits,
    Vector<uint32_t> data) {
  const auto statistics = statsFor(data);
  const auto nonExactBytes = EncodingT::estimateSize(
      data.size(), statistics, nonExactBitWidthOptions());
  const auto exactBytes =
      EncodingT::estimateSize(data.size(), statistics, exactBitWidthOptions());
  printRow(label, bits, nonExactBytes, exactBytes);
}

void printFixedBitWidthActual() {
  printHeader("FixedBitWidth actual serialized size");
  for (const auto bitWidth : kValueBits) {
    auto data = makeRangeData(bitWidth);
    const auto nonExactEncoded =
        benchmarks::encodeData<FixedBitWidthEncoding<uint32_t>>(
            EncodingType::FixedBitWidth, data, nonExactBitWidthOptions());
    const auto exactEncoded =
        benchmarks::encodeData<FixedBitWidthEncoding<uint32_t>>(
            EncodingType::FixedBitWidth, data, exactBitWidthOptions());
    printRow(
        "FixedBitWidth", bitWidth, nonExactEncoded.size(), exactEncoded.size());
  }
}

void printEstimatorSweeps() {
  printHeader("Estimator size: value bit-width children");
  for (const auto bitWidth : kValueBits) {
    printEstimateRow<FixedBitWidthEncoding<uint32_t>>(
        "FixedBitWidth", bitWidth, makeRangeData(bitWidth));
    printEstimateRow<RLEEncoding<uint32_t>>(
        "RLE", bitWidth, makeRunData(bitWidth));
    printEstimateRow<MainlyConstantEncoding<uint32_t>>(
        "MainlyConstant", bitWidth, makeMainlyConstantData(bitWidth));
  }

  printHeader("Estimator size: dictionary index bit width");
  for (const auto indexBits : kIndexBits) {
    printEstimateRow<DictionaryEncoding<uint32_t>>(
        "Dictionary", indexBits, makeDictionaryData(indexBits));
  }
}

void printPforSweep() {
  printHeader("PFOR actual size with non-exact bucket-width payload");
  for (const auto baseBits : kValueBits) {
    if (baseBits == 0) {
      continue;
    }
    auto data = makePforData(baseBits);
    const auto nonExactEncoded = benchmarks::encodeData<PFOREncoding<uint32_t>>(
        EncodingType::PFOR, data, nonExactBitWidthOptions());
    const auto exactEncoded = benchmarks::encodeData<PFOREncoding<uint32_t>>(
        EncodingType::PFOR, data, exactBitWidthOptions());
    printRow("PFOR", baseBits, nonExactEncoded.size(), exactEncoded.size());
  }
}

} // namespace
} // namespace facebook::nimble

int main() {
  facebook::velox::memory::MemoryManager::initialize({});

  std::cout << "Exact bit-width storage benchmark; rows="
            << facebook::nimble::kRows << "\n";
  facebook::nimble::printFixedBitWidthActual();
  facebook::nimble::printEstimatorSweeps();
  facebook::nimble::printPforSweep();
  return 0;
}
