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

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/ML_ID_Compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"

namespace facebook::nimble::mlidc {
namespace {

using Elem = int64_t;

std::vector<EncoderEntry<Elem>> buildEncoders() {
  std::vector<EncoderEntry<Elem>> encoders;

  encoders.push_back(
      makeEncoderEntry<TrivialEncoding<Elem>>(
          "Trivial", "Baseline", "trivial", true, true, false));
  encoders.push_back(
      makeEncoderEntry<FixedBitWidthEncoding<Elem>>(
          "FixedBitWidth", "Baseline", "fbw", true, true, false));
  encoders.push_back(
      makeEncoderEntry<DictionaryEncoding<Elem>>(
          "Dictionary", "Baseline", "dict", true, false, false));
  encoders.push_back(
      makeEncoderEntry<RLEEncoding<Elem>>(
          "RLE", "Baseline", "rle", true, true, false));

  const std::array<std::string, 4> fpeNames = {
      "fpe_noindex", "fpe_pertier", "fpe_tagtag", "fpe_elias"};
  const std::array<bool, 4> fpeRA = {false, true, true, true};
  const std::array<bool, 4> fpeSkip = {false, true, true, true};

  for (int idx = 0; idx < 4; ++idx) {
    EncoderEntry<Elem> entry;
    entry.name = "FPE/" + fpeNames[idx];
    entry.family = "FrequencyPartition";
    entry.variant = fpeNames[idx];
    entry.isSequential = true;
    entry.fastSkip = fpeSkip[idx];
    entry.randomAccess = fpeRA[idx];
    entry.factory = [idx](const Vector<Elem>& data,
                          const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleBenchTargetImpl<
              FrequencyPartitionEncoding<Elem>>>();
      Encoding::Options o = opts;
      o.frequencyPartitionIndex = static_cast<uint8_t>(idx);
      impl->target.encode(data, o);
      return std::unique_ptr<NimbleBenchTargetBase<Elem>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    EncoderEntry<Elem> entry;
    entry.name = "SIS/realNested";
    entry.family = "SubIntSplit";
    entry.variant = "real_nested";
    entry.isSequential = false;
    entry.fastSkip = false;
    entry.randomAccess = false;
    entry.factory = [](const Vector<Elem>& data,
                       const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleBenchTargetImpl<
              SubIntSplitEncoding<Elem>>>();
      impl->target.encode(data, opts, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<Elem>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  return encoders;
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});
  using namespace facebook::nimble::mlidc;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);
  auto encoders = buildEncoders();
  auto datasets = defaultInt64Datasets<Elem>();
  int failures = 0;

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
          if (out[i] != data[i]) {
            ok = false;
            break;
          }
        }
        if (!ok) {
          std::cerr << "FAIL: " << enc.name << " / " << ds.name << "\n";
          ++failures;
        } else {
          std::cout << "  OK: " << enc.name << " / " << ds.name << "\n";
        }
      } catch (const std::exception& ex) {
        std::cerr << "EXCEPTION: " << enc.name << " / " << ds.name << ": "
                  << ex.what() << "\n";
        ++failures;
      }
    }
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
