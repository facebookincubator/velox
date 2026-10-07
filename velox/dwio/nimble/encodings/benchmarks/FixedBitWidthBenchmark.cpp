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

// Micro-benchmarks for FixedBitWidthEncoding decode paths.
// Measures materialize() and bulkGet64WithBaseline() throughput.

#include <cstring>
#include <memory>
#include <vector>

#include "folly/Benchmark.h"
#include "folly/Random.h"
#include "folly/init/Init.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

using namespace facebook::nimble;

namespace {

constexpr int kNumElements = 100000;
constexpr uint64_t kBaseline = 12345;

std::shared_ptr<facebook::velox::memory::MemoryPool> pool;

/// Creates a Vector of random values fitting in the given bit width.
template <typename T>
Vector<T> makeData(int bitWidth, int n = kNumElements) {
  Vector<T> data{pool.get()};
  data.resize(n);
  using U = typename std::make_unsigned<T>::type;
  U maxVal = bitWidth >= static_cast<int>(sizeof(T) * 8)
      ? static_cast<U>(~U{0})
      : (static_cast<U>(1) << bitWidth) - 1;
  for (int i = 0; i < n; ++i) {
    if constexpr (sizeof(T) <= 4) {
      data[i] = static_cast<T>(folly::Random::secureRand32() % (maxVal + 1));
    } else {
      data[i] = static_cast<T>(folly::Random::secureRand64() % (maxVal + 1));
    }
  }
  return data;
}

Vector<uint64_t> makeDataWithBaseline(int bitWidth, int n = kNumElements) {
  Vector<uint64_t> data{pool.get()};
  data.resize(n);
  const uint64_t mask = bitWidth == 64 ? ~0ULL : ((1ULL << bitWidth) - 1);
  const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;
  for (int i = 0; i < n; ++i) {
    data[i] = ((static_cast<uint64_t>(i) * 1000003ULL) & mask) + baseline;
  }
  return data;
}

Vector<uint64_t> makePforMaskedResiduals(
    int baseBitWidth,
    int n = kNumElements) {
  Vector<uint64_t> data{pool.get()};
  data.resize(n);
  const uint64_t mask =
      baseBitWidth == 64 ? ~0ULL : ((1ULL << baseBitWidth) - 1);
  for (int i = 0; i < n; ++i) {
    // PFOR zeroes exception slots before bitpacking the base residual stream.
    data[i] =
        i % 10 == 0 ? 0 : ((static_cast<uint64_t>(i) * 1000003ULL) & mask);
  }
  return data;
}

void observeWrittenBuffer(
    const std::vector<char>& buffer,
    uint32_t elementCount,
    int bitWidth) {
  const uint64_t writtenBytes =
      ((elementCount * static_cast<uint64_t>(bitWidth)) + 7) >> 3;
  folly::doNotOptimizeAway(buffer[writtenBytes - 1]);
}

/// Encodes data and returns (encoded string, encoding).
template <typename T>
std::string encodeFixedBitWidth(
    Buffer& buffer,
    const Vector<T>& data,
    const Encoding::Options& options = {}) {
  using physicalType = typename TypeTraits<T>::physicalType;

  auto physicalValues = std::span<const physicalType>(
      reinterpret_cast<const physicalType*>(data.data()), data.size());
  EncodingSelectionResult selectionResult{
      .encodingType = EncodingType::FixedBitWidth};
  auto selection = EncodingSelection<physicalType>{
      std::move(selectionResult),
      Statistics<physicalType>::create(physicalValues),
      nullptr};

  auto encoded = FixedBitWidthEncoding<T>::encode(
      selection, physicalValues, buffer, options);
  return std::string{encoded.data(), encoded.size()};
}

Encoding::Options exactBitWidthOptions() {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = true;
  return options;
}

Encoding::Options byteRoundedBitWidthOptions() {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = false;
  return options;
}

std::string encodeByteRoundedUint32(
    Buffer& buffer,
    const Vector<uint32_t>& data) {
  return encodeFixedBitWidth(buffer, data, byteRoundedBitWidthOptions());
}

} // namespace

// ============================================================================
// FixedBitWidthEncoding encode packing — 64-bit
// Mirrors the packing portion of FixedBitWidthEncoding::encode(). The scalar
// benchmark is the old path; the relative benchmark is the bulk path used by
// D107301473.
// ============================================================================

#define FBW_PACK64_BENCHMARKS(bitWidth)                                 \
  BENCHMARK(FBW_PackUint64_Scalar_##bitWidth##bit, iters) {             \
    Vector<uint64_t> data{pool.get()};                                  \
    std::vector<char> buffer;                                           \
    BENCHMARK_SUSPEND {                                                 \
      data = makeDataWithBaseline(bitWidth);                            \
      buffer.resize(FixedBitArray::bufferSize(kNumElements, bitWidth)); \
    }                                                                   \
    const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;           \
    while (iters--) {                                                   \
      BENCHMARK_SUSPEND {                                               \
        std::memset(buffer.data(), 0, buffer.size());                   \
      }                                                                 \
      FixedBitArray fba(buffer.data(), bitWidth);                       \
      for (uint32_t i = 0; i < data.size(); ++i) {                      \
        fba.set(i, data[i] - baseline);                                 \
      }                                                                 \
      observeWrittenBuffer(buffer, kNumElements, bitWidth);             \
    }                                                                   \
  }                                                                     \
  BENCHMARK_RELATIVE(FBW_PackUint64_Bulk_##bitWidth##bit, iters) {      \
    Vector<uint64_t> data{pool.get()};                                  \
    std::vector<char> buffer;                                           \
    BENCHMARK_SUSPEND {                                                 \
      data = makeDataWithBaseline(bitWidth);                            \
      buffer.resize(FixedBitArray::bufferSize(kNumElements, bitWidth)); \
    }                                                                   \
    const uint64_t baseline = bitWidth == 64 ? 0 : kBaseline;           \
    while (iters--) {                                                   \
      BENCHMARK_SUSPEND {                                               \
        std::memset(buffer.data(), 0, buffer.size());                   \
      }                                                                 \
      FixedBitArray fba(buffer.data(), bitWidth);                       \
      fba.bulkSetWithBaseline(0, kNumElements, data.data(), baseline);  \
      observeWrittenBuffer(buffer, kNumElements, bitWidth);             \
    }                                                                   \
  }

FBW_PACK64_BENCHMARKS(32)
FBW_PACK64_BENCHMARKS(40)
FBW_PACK64_BENCHMARKS(48)
FBW_PACK64_BENCHMARKS(56)
FBW_PACK64_BENCHMARKS(64)

BENCHMARK_DRAW_LINE();

// ============================================================================
// FixedBitWidthEncoding encode packing — narrow (1-/2-byte) types
// Scalar = old per-element set() path; Bulk = bulkSetWithBaseline, which widens
// in fixed-size stack chunks onto the bulkSet32 path. Confirms the unified path
// is not a regression for narrow types.
// ============================================================================

#define FBW_PACK_NARROW_BENCHMARKS(Type, bitWidth)                      \
  BENCHMARK(FBW_Pack_##Type##_Scalar_##bitWidth##bit, iters) {          \
    Vector<Type> data{pool.get()};                                      \
    std::vector<char> buffer;                                           \
    BENCHMARK_SUSPEND {                                                 \
      data = makeData<Type>(bitWidth);                                  \
      buffer.resize(FixedBitArray::bufferSize(kNumElements, bitWidth)); \
    }                                                                   \
    while (iters--) {                                                   \
      BENCHMARK_SUSPEND {                                               \
        std::memset(buffer.data(), 0, buffer.size());                   \
      }                                                                 \
      FixedBitArray fba(buffer.data(), bitWidth);                       \
      for (uint32_t i = 0; i < data.size(); ++i) {                      \
        fba.set(i, data[i]);                                            \
      }                                                                 \
      observeWrittenBuffer(buffer, kNumElements, bitWidth);             \
    }                                                                   \
  }                                                                     \
  BENCHMARK_RELATIVE(FBW_Pack_##Type##_Bulk_##bitWidth##bit, iters) {   \
    Vector<Type> data{pool.get()};                                      \
    std::vector<char> buffer;                                           \
    BENCHMARK_SUSPEND {                                                 \
      data = makeData<Type>(bitWidth);                                  \
      buffer.resize(FixedBitArray::bufferSize(kNumElements, bitWidth)); \
    }                                                                   \
    while (iters--) {                                                   \
      BENCHMARK_SUSPEND {                                               \
        std::memset(buffer.data(), 0, buffer.size());                   \
      }                                                                 \
      FixedBitArray fba(buffer.data(), bitWidth);                       \
      fba.bulkSetWithBaseline(                                          \
          0, kNumElements, data.data(), static_cast<Type>(0));          \
      observeWrittenBuffer(buffer, kNumElements, bitWidth);             \
    }                                                                   \
  }

FBW_PACK_NARROW_BENCHMARKS(uint8_t, 5)
FBW_PACK_NARROW_BENCHMARKS(uint16_t, 12)

BENCHMARK_DRAW_LINE();

// ============================================================================
// PFOREncoding base-residual packing — 64-bit
// Mirrors the final bitpacking step in PFOREncoding::encode(). PFOR has already
// materialized `maskedResiduals` with exception slots zeroed before this point.
// ============================================================================

#define PFOR_PACK64_BENCHMARKS(bitWidth)                                \
  BENCHMARK(PFOR_PackUint64_Scalar_##bitWidth##bit, iters) {            \
    Vector<uint64_t> maskedResiduals{pool.get()};                       \
    std::vector<char> buffer;                                           \
    BENCHMARK_SUSPEND {                                                 \
      maskedResiduals = makePforMaskedResiduals(bitWidth);              \
      buffer.resize(FixedBitArray::bufferSize(kNumElements, bitWidth)); \
    }                                                                   \
    while (iters--) {                                                   \
      BENCHMARK_SUSPEND {                                               \
        std::memset(buffer.data(), 0, buffer.size());                   \
      }                                                                 \
      FixedBitArray fba(buffer.data(), bitWidth);                       \
      for (uint32_t i = 0; i < maskedResiduals.size(); ++i) {           \
        fba.set(i, maskedResiduals[i]);                                 \
      }                                                                 \
      observeWrittenBuffer(buffer, kNumElements, bitWidth);             \
    }                                                                   \
  }                                                                     \
  BENCHMARK_RELATIVE(PFOR_PackUint64_Bulk_##bitWidth##bit, iters) {     \
    Vector<uint64_t> maskedResiduals{pool.get()};                       \
    std::vector<char> buffer;                                           \
    BENCHMARK_SUSPEND {                                                 \
      maskedResiduals = makePforMaskedResiduals(bitWidth);              \
      buffer.resize(FixedBitArray::bufferSize(kNumElements, bitWidth)); \
    }                                                                   \
    while (iters--) {                                                   \
      BENCHMARK_SUSPEND {                                               \
        std::memset(buffer.data(), 0, buffer.size());                   \
      }                                                                 \
      FixedBitArray fba(buffer.data(), bitWidth);                       \
      fba.bulkSetWithBaseline(                                          \
          0, kNumElements, maskedResiduals.data(), /*baseline=*/0);     \
      observeWrittenBuffer(buffer, kNumElements, bitWidth);             \
    }                                                                   \
  }

PFOR_PACK64_BENCHMARKS(32)
PFOR_PACK64_BENCHMARKS(40)
PFOR_PACK64_BENCHMARKS(48)
PFOR_PACK64_BENCHMARKS(56)
PFOR_PACK64_BENCHMARKS(64)

BENCHMARK_DRAW_LINE();

// ============================================================================
// FixedBitWidthEncoding materialize() — exact bit width vs byte-rounded
// payloads. This isolates the storage-size change: exact streams read fewer
// bytes, while byte-rounded streams may use byte-aligned bit widths.
// ============================================================================

#define FBW_MATERIALIZE_EXACT_VS_BYTE_ROUNDED(exactBitWidth, roundedBitWidth) \
  BENCHMARK(FBW_Materialize_uint32_exact_##exactBitWidth##bit, iters) {       \
    std::string encoded;                                                      \
    std::vector<uint32_t> output(kNumElements);                               \
    BENCHMARK_SUSPEND {                                                       \
      Buffer buffer{*pool};                                                   \
      auto data = makeData<uint32_t>(exactBitWidth);                          \
      encoded = encodeFixedBitWidth(buffer, data, exactBitWidthOptions());    \
    }                                                                         \
    while (iters--) {                                                         \
      auto encoding = EncodingFactory{}.create(                               \
          *pool, encoded, [](uint32_t) -> void* { return nullptr; });         \
      encoding->materialize(kNumElements, output.data());                     \
    }                                                                         \
  }                                                                           \
  BENCHMARK_RELATIVE(                                                         \
      FBW_Materialize_uint32_byteRounded_##roundedBitWidth##bit, iters) {     \
    std::string encoded;                                                      \
    std::vector<uint32_t> output(kNumElements);                               \
    BENCHMARK_SUSPEND {                                                       \
      Buffer buffer{*pool};                                                   \
      auto data = makeData<uint32_t>(exactBitWidth);                          \
      encoded = encodeByteRoundedUint32(buffer, data);                        \
    }                                                                         \
    while (iters--) {                                                         \
      auto encoding = EncodingFactory{}.create(                               \
          *pool, encoded, [](uint32_t) -> void* { return nullptr; });         \
      encoding->materialize(kNumElements, output.data());                     \
    }                                                                         \
  }

FBW_MATERIALIZE_EXACT_VS_BYTE_ROUNDED(6, 8)
FBW_MATERIALIZE_EXACT_VS_BYTE_ROUNDED(9, 16)

BENCHMARK_DRAW_LINE();

// ============================================================================
// FixedBitWidthEncoding materialize() — 32-bit
// ============================================================================

BENCHMARK(FBW_Materialize_uint32_8bit, iters) {
  std::string encoded;
  std::vector<uint32_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    Buffer buffer{*pool};
    auto data = makeData<uint32_t>(8);
    encoded = encodeFixedBitWidth(buffer, data);
  }
  while (iters--) {
    auto encoding = EncodingFactory{}.create(
        *pool, encoded, [](uint32_t) -> void* { return nullptr; });
    encoding->materialize(kNumElements, output.data());
  }
}

BENCHMARK(FBW_Materialize_uint32_16bit, iters) {
  std::string encoded;
  std::vector<uint32_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    Buffer buffer{*pool};
    auto data = makeData<uint32_t>(16);
    encoded = encodeFixedBitWidth(buffer, data);
  }
  while (iters--) {
    auto encoding = EncodingFactory{}.create(
        *pool, encoded, [](uint32_t) -> void* { return nullptr; });
    encoding->materialize(kNumElements, output.data());
  }
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// FixedBitWidthEncoding materialize() — 64-bit
// These benefit from the new bulkGet64WithBaseline optimization.
// ============================================================================

BENCHMARK(FBW_Materialize_uint64_8bit, iters) {
  std::string encoded;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    Buffer buffer{*pool};
    auto data = makeData<uint64_t>(8);
    encoded = encodeFixedBitWidth(buffer, data);
  }
  while (iters--) {
    auto encoding = EncodingFactory{}.create(
        *pool, encoded, [](uint32_t) -> void* { return nullptr; });
    encoding->materialize(kNumElements, output.data());
  }
}

BENCHMARK(FBW_Materialize_uint64_16bit, iters) {
  std::string encoded;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    Buffer buffer{*pool};
    auto data = makeData<uint64_t>(16);
    encoded = encodeFixedBitWidth(buffer, data);
  }
  while (iters--) {
    auto encoding = EncodingFactory{}.create(
        *pool, encoded, [](uint32_t) -> void* { return nullptr; });
    encoding->materialize(kNumElements, output.data());
  }
}

BENCHMARK(FBW_Materialize_uint64_32bit, iters) {
  std::string encoded;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    Buffer buffer{*pool};
    auto data = makeData<uint64_t>(32);
    encoded = encodeFixedBitWidth(buffer, data);
  }
  while (iters--) {
    auto encoding = EncodingFactory{}.create(
        *pool, encoded, [](uint32_t) -> void* { return nullptr; });
    encoding->materialize(kNumElements, output.data());
  }
}

BENCHMARK(FBW_Materialize_uint64_48bit, iters) {
  std::string encoded;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    Buffer buffer{*pool};
    auto data = makeData<uint64_t>(48);
    encoded = encodeFixedBitWidth(buffer, data);
  }
  while (iters--) {
    auto encoding = EncodingFactory{}.create(
        *pool, encoded, [](uint32_t) -> void* { return nullptr; });
    encoding->materialize(kNumElements, output.data());
  }
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// FixedBitArray: bulkGet64WithBaseline (NEW) vs per-element get() (OLD)
// The OLD materialize() code path used get() in a loop for 64-bit types
// with bitWidth > 32. The NEW path uses bulkGet64WithBaseline with
// branchless byte-aligned loads.
// Each pair shows baseline (per-element) then relative (bulk) speedup.
// ============================================================================
BENCHMARK(PerElementGet64_16bit, iters) {
  FixedBitArray fba;
  std::vector<char> buffer;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    int bw = 16;
    buffer.resize(FixedBitArray::bufferSize(kNumElements, bw), 0);
    fba = FixedBitArray{buffer.data(), bw};
    for (int i = 0; i < kNumElements; ++i) {
      fba.set(i, folly::Random::secureRand64() % (1ULL << bw));
    }
  }
  while (iters--) {
    for (int i = 0; i < kNumElements; ++i) {
      output[i] = fba.get(i);
    }
    folly::doNotOptimizeAway(output.back());
  }
}

BENCHMARK_RELATIVE(BulkGet64_vs_PerElem_16bit, iters) {
  FixedBitArray fba;
  std::vector<char> buffer;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    int bw = 16;
    buffer.resize(FixedBitArray::bufferSize(kNumElements, bw), 0);
    fba = FixedBitArray{buffer.data(), bw};
    for (int i = 0; i < kNumElements; ++i) {
      fba.set(i, folly::Random::secureRand64() % (1ULL << bw));
    }
  }
  while (iters--) {
    fba.bulkGetWithBaseline(0, kNumElements, output.data(), 0);
    folly::doNotOptimizeAway(output.back());
  }
}

BENCHMARK_DRAW_LINE();

BENCHMARK(PerElementGet64_40bit, iters) {
  FixedBitArray fba;
  std::vector<char> buffer;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    int bw = 40;
    buffer.resize(FixedBitArray::bufferSize(kNumElements, bw), 0);
    fba = FixedBitArray{buffer.data(), bw};
    for (int i = 0; i < kNumElements; ++i) {
      fba.set(i, folly::Random::secureRand64() % (1ULL << bw));
    }
  }
  while (iters--) {
    for (int i = 0; i < kNumElements; ++i) {
      output[i] = fba.get(i);
    }
    folly::doNotOptimizeAway(output.back());
  }
}

BENCHMARK_RELATIVE(BulkGet64_vs_PerElem_40bit, iters) {
  FixedBitArray fba;
  std::vector<char> buffer;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    int bw = 40;
    buffer.resize(FixedBitArray::bufferSize(kNumElements, bw), 0);
    fba = FixedBitArray{buffer.data(), bw};
    for (int i = 0; i < kNumElements; ++i) {
      fba.set(i, folly::Random::secureRand64() % (1ULL << bw));
    }
  }
  while (iters--) {
    fba.bulkGetWithBaseline(0, kNumElements, output.data(), 0);
    folly::doNotOptimizeAway(output.back());
  }
}

BENCHMARK(PerElementGet64_56bit, iters) {
  FixedBitArray fba;
  std::vector<char> buffer;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    int bw = 56;
    buffer.resize(FixedBitArray::bufferSize(kNumElements, bw), 0);
    fba = FixedBitArray{buffer.data(), bw};
    for (int i = 0; i < kNumElements; ++i) {
      fba.set(i, folly::Random::secureRand64() % (1ULL << bw));
    }
  }
  while (iters--) {
    for (int i = 0; i < kNumElements; ++i) {
      output[i] = fba.get(i);
    }
    folly::doNotOptimizeAway(output.back());
  }
}

BENCHMARK_RELATIVE(BulkGet64_vs_PerElem_56bit, iters) {
  FixedBitArray fba;
  std::vector<char> buffer;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    int bw = 56;
    buffer.resize(FixedBitArray::bufferSize(kNumElements, bw), 0);
    fba = FixedBitArray{buffer.data(), bw};
    for (int i = 0; i < kNumElements; ++i) {
      fba.set(i, folly::Random::secureRand64() % (1ULL << bw));
    }
  }
  while (iters--) {
    fba.bulkGetWithBaseline(0, kNumElements, output.data(), 0);
    folly::doNotOptimizeAway(output.back());
  }
}

BENCHMARK(PerElementGet64_57bit, iters) {
  FixedBitArray fba;
  std::vector<char> buffer;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    int bw = 57;
    buffer.resize(FixedBitArray::bufferSize(kNumElements, bw), 0);
    fba = FixedBitArray{buffer.data(), bw};
    for (int i = 0; i < kNumElements; ++i) {
      fba.set(i, folly::Random::secureRand64() % (1ULL << bw));
    }
  }
  while (iters--) {
    for (int i = 0; i < kNumElements; ++i) {
      output[i] = fba.get(i);
    }
    folly::doNotOptimizeAway(output.back());
  }
}

BENCHMARK_RELATIVE(BulkGet64_vs_PerElem_57bit, iters) {
  FixedBitArray fba;
  std::vector<char> buffer;
  std::vector<uint64_t> output(kNumElements);
  BENCHMARK_SUSPEND {
    int bw = 57;
    buffer.resize(FixedBitArray::bufferSize(kNumElements, bw), 0);
    fba = FixedBitArray{buffer.data(), bw};
    for (int i = 0; i < kNumElements; ++i) {
      fba.set(i, folly::Random::secureRand64() % (1ULL << bw));
    }
  }
  while (iters--) {
    fba.bulkGetWithBaseline(0, kNumElements, output.data(), 0);
    folly::doNotOptimizeAway(output.back());
  }
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});
  pool = facebook::velox::memory::memoryManager()->addLeafPool("benchmark");
  folly::runBenchmarks();
}
