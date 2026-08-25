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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"

namespace facebook::nimble::mlidc {
namespace {

// Round-trips every encoder against every dataset for one element type.
// Returns the number of failures so the caller can total them across types.
template <typename Elem>
int runSmoke(uint32_t n, uint64_t seed) {
  auto encoders = buildDefaultEncoders<Elem>();
  auto datasets = defaultDatasets<Elem>();
  int failures = 0;

  std::cout << "== dtype: " << elemTypeName<Elem>() << " ==\n";

  for (const auto& ds : datasets) {
    auto data = ds.generate(n, seed);
    for (const auto& enc : encoders) {
      // Skip FPE/fpe_noindex validation (reorders without restoration index)
      if (enc.variant == "fpe_noindex") {
        continue;
      }

      facebook::nimble::Encoding::Options opts;
      try {
        auto target = enc.factory(data, opts);
        std::vector<Elem> out(n);
        target->materializeAll(out.data(), n);
        bool ok = true;
        for (uint32_t i = 0; i < n; ++i) {
          // Exact equality is the right test even for float and double: the
          // encodings preserve the bit pattern, and the generators produce no
          // NaN, which would compare unequal to itself.
          if (out[i] != data[i]) {
            ok = false;
            break;
          }
        }
        if (!ok) {
          std::cerr << "FAIL: " << elemTypeName<Elem>() << " / " << enc.name
                    << " / " << ds.name << "\n";
          ++failures;
        } else {
          std::cout << "  OK: " << enc.name << " / " << ds.name << "\n";
        }
      } catch (const std::exception& ex) {
        std::cerr << "EXCEPTION: " << elemTypeName<Elem>() << " / " << enc.name
                  << " / " << ds.name << ": " << ex.what() << "\n";
        ++failures;
      }
    }
  }
  return failures;
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});
  using namespace facebook::nimble::mlidc;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  // Unlike the other drivers this one ignores --mlidc_dtype and sweeps every
  // supported type: it is the gate proving the suite instantiates and round
  // trips for all of them, so running one type at a time would defeat it.
  using facebook::nimble::DataType;
  const std::array<DataType, 6> dtypes = {
      DataType::Int32,
      DataType::Uint32,
      DataType::Int64,
      DataType::Uint64,
      DataType::Float,
      DataType::Double};

  int failures = 0;
  for (const auto dtype : dtypes) {
    failures += dispatchElemType(
        dtype, [&]<typename T>() { return runSmoke<T>(n, seed); });
  }

  std::cout << (failures == 0 ? "All passed.\n" : "FAILURES detected.\n");
  return failures > 0 ? 1 : 0;
}

#else
#include <iostream>
int main() {
  std::cerr << "bench_smoke requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}
#endif
